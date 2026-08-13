"""
Extração do score de efeito de mutação via ESM-1v, usando a abordagem de masked marginal.

Observação importante do plano: será usado apenas UM dos 5 modelos do ensemble ESM-1v
(definido em config.ESM1V_MODEL_NAME), não o ensemble completo.
"""
import logging
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch

from config import ESM1V_MODEL_NAME, ESM_DEVICE, ESM_MAX_TOKENS, ESM_SCORES_CACHE

logger = logging.getLogger(__name__)

MUTANT_PATTERN = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


def parse_mutant(mutant: str) -> Tuple[str, int, str]:
    """Converte uma string de mutação, ex.: 'A329V', em (aa_original, posição_1_indexada, aa_mutante)."""
    match = MUTANT_PATTERN.match(str(mutant).strip())
    if not match:
        raise ValueError(f"Formato de mutação inesperado: {mutant!r} (esperado algo como 'A329V')")
    wt_aa, pos, mt_aa = match.groups()
    return wt_aa.upper(), int(pos), mt_aa.upper()


def load_esm1v_model(model_name: str = ESM1V_MODEL_NAME, device: str = ESM_DEVICE):
    """Carrega um único modelo do ensemble ESM-1v (nunca os 5 ao mesmo tempo)."""
    import esm  # import local: só é exigido quando de fato formos calcular scores

    logger.info("Carregando modelo ESM-1v: %s", model_name)
    load_fn = getattr(esm.pretrained, model_name)
    model, alphabet = load_fn()
    model.eval()

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA solicitado mas indisponível neste ambiente — usando CPU.")
        device = "cpu"
    model = model.to(device)

    return model, alphabet, device


def _get_window(sequence: str, position0: int, max_len: int) -> Tuple[str, int]:
    """
    Extrai uma janela da sequência centralizada na posição mutada (0-indexada), para lidar
    com proteínas cujo comprimento excede o limite de contexto do ESM-1v (>1024 tokens).
    Retorna (janela, nova_posição_na_janela).
    """
    seq_len = len(sequence)
    if seq_len <= max_len:
        return sequence, position0

    half = max_len // 2
    start = max(0, position0 - half)
    end = start + max_len
    if end > seq_len:
        end = seq_len
        start = end - max_len

    return sequence[start:end], position0 - start


class MaskedMarginalScorer:
    """
    Calcula scores de masked marginal do ESM-1v: para cada variante, mascara a posição
    mutada na sequência selvagem e compara a log-probabilidade do aminoácido mutante
    com a do aminoácido original nessa posição.

    score = log P(aa_mutante | contexto mascarado) - log P(aa_original | contexto mascarado)

    Mantém um cache por (janela_de_sequência, posição) para evitar forward passes repetidos
    quando várias mutações caem na mesma posição da mesma proteína.
    """

    def __init__(self, model, alphabet, device: str, max_tokens: int = ESM_MAX_TOKENS):
        self.model = model
        self.alphabet = alphabet
        self.device = device
        self.batch_converter = alphabet.get_batch_converter()
        self.max_len = max_tokens - 2  # reserva espaço para os tokens BOS/EOS
        self._logprob_cache: Dict[Tuple[str, int], torch.Tensor] = {}

    def _log_probs_at_position(self, sequence: str, position0: int) -> torch.Tensor:
        """Retorna o vetor de log-probabilidades (todo o vocabulário) na posição mascarada."""
        window_seq, window_pos0 = _get_window(sequence, position0, self.max_len)
        cache_key = (window_seq, window_pos0)
        if cache_key in self._logprob_cache:
            return self._logprob_cache[cache_key]

        _, _, tokens = self.batch_converter([("variant", window_seq)])
        tokens = tokens.to(self.device)

        token_pos = window_pos0 + 1  # +1 pelo token BOS adicionado no início
        masked_tokens = tokens.clone()
        masked_tokens[0, token_pos] = self.alphabet.mask_idx

        with torch.no_grad():
            logits = self.model(masked_tokens)["logits"]
        log_probs = torch.log_softmax(logits[0, token_pos], dim=-1).cpu()

        self._logprob_cache[cache_key] = log_probs
        return log_probs

    def score_variant(self, sequence: str, mutant: str) -> float:
        """Calcula o score ESM-1v (masked marginal) de uma única substituição."""
        wt_aa, pos1, mt_aa = parse_mutant(mutant)
        position0 = pos1 - 1

        if position0 < 0 or position0 >= len(sequence):
            raise ValueError(f"Posição {pos1} fora dos limites da sequência (tamanho {len(sequence)}).")
        if sequence[position0] != wt_aa:
            logger.warning(
                "Divergência entre aa esperado pela mutação (%s) e aa na sequência (%s) na posição %d.",
                wt_aa, sequence[position0], pos1,
            )

        log_probs = self._log_probs_at_position(sequence, position0)
        wt_idx = self.alphabet.get_idx(wt_aa)
        mt_idx = self.alphabet.get_idx(mt_aa)

        return (log_probs[mt_idx] - log_probs[wt_idx]).item()


def compute_esm_scores(
    data: pd.DataFrame,
    cache_path: Path = ESM_SCORES_CACHE,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Calcula (ou recupera do cache) o score ESM-1v de cada variante em `data`.
    Espera as colunas 'protein', 'protein_sequence', 'mutant' e 'DMS_id'.
    """
    cache_path = Path(cache_path)
    data = data.copy()

    if cache_path.exists() and not force_recompute:
        logger.info("Carregando scores ESM-1v do cache: %s", cache_path)
        cached = pd.read_csv(cache_path)
        data = data.merge(cached, on=["protein", "mutant", "DMS_id"], how="left")
        missing_mask = data["esm1v_score"].isna()
        if not missing_mask.any():
            return data
        logger.info("%d variantes sem score no cache — calculando o restante.", missing_mask.sum())
        to_compute = data.loc[missing_mask]
    else:
        data["esm1v_score"] = float("nan")
        to_compute = data

    if len(to_compute) == 0:
        return data

    model, alphabet, device = load_esm1v_model()
    scorer = MaskedMarginalScorer(model, alphabet, device)

    scores = []
    for count, (idx, row) in enumerate(to_compute.iterrows(), start=1):
        try:
            score = scorer.score_variant(row["protein_sequence"], row["mutant"])
        except Exception as exc:
            logger.warning("Falha ao pontuar %s/%s: %s", row["protein"], row["mutant"], exc)
            score = float("nan")
        scores.append(score)
        if count % 200 == 0:
            logger.info("Processadas %d/%d variantes.", count, len(to_compute))

    data.loc[to_compute.index, "esm1v_score"] = scores

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing_cache = pd.read_csv(cache_path) if cache_path.exists() else None
    new_cache_rows = data[["protein", "mutant", "DMS_id", "esm1v_score"]].dropna(subset=["esm1v_score"])
    if existing_cache is not None:
        combined = pd.concat([existing_cache, new_cache_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["protein", "mutant", "DMS_id"], keep="last")
    else:
        combined = new_cache_rows
    combined.to_csv(cache_path, index=False)

    return data
