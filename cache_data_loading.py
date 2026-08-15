"""
Carregamento e alinhamento dos arquivos de cache já existentes:

    cache/metadata.csv        -> row_id, protein, mutant, position, original_aa, mutant_aa,
                                  DMS_bin_score, source_file
    cache/embeddings.npz      -> wt_embeddings (N, 1280), mutant_embeddings, delta_embeddings
    cache/esm1vScores.csv     -> protein, mutant, DMS_id, esm1v_score

Ponto crítico: a correspondência entre metadata.csv e wt_embeddings é posicional
(metadata.iloc[i] <-> wt_embeddings[i]), conforme informado. Este módulo NUNCA assume
que essa correspondência posicional sobrevive a merges — em vez disso, usa `row_id`
explicitamente para reindexar o array de embeddings sempre que o DataFrame é
reordenado ou filtrado, prevenindo desalinhamento silencioso.
"""
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from config import LABEL_MAP

logger = logging.getLogger(__name__)

EXPECTED_EMBEDDING_DIM = 1280

METADATA_REQUIRED_COLUMNS = [
    "row_id", "protein", "mutant", "position", "original_aa", "mutant_aa",
    "DMS_bin_score", "source_file",
]
SCORES_REQUIRED_COLUMNS = ["protein", "mutant", "DMS_id", "esm1v_score"]


def load_metadata(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"metadata.csv não encontrado em {path}")

    metadata = pd.read_csv(path)
    missing = [c for c in METADATA_REQUIRED_COLUMNS if c not in metadata.columns]
    if missing:
        raise ValueError(f"metadata.csv sem colunas obrigatórias: {missing}")

    if not (metadata["row_id"].values == np.arange(len(metadata))).all():
        logger.warning(
            "row_id de metadata.csv não é uma sequência 0..N-1 estrita — "
            "o alinhamento com embeddings.npz será feito via row_id explicitamente, "
            "então isso continua seguro, mas vale confirmar que row_id é de fato o "
            "índice original usado na extração dos embeddings."
        )

    # DMS_id derivado do arquivo de origem, para casar com esm1vScores.csv de forma inequívoca
    metadata["DMS_id"] = metadata["source_file"].apply(lambda f: Path(str(f)).stem)

    before = len(metadata)
    metadata = metadata[metadata["DMS_bin_score"].isin(LABEL_MAP.keys())].copy()
    if len(metadata) < before:
        logger.info(
            "Descartadas %d linhas de metadata.csv com DMS_bin_score fora de {Benign, Pathogenic}.",
            before - len(metadata),
        )
    metadata["label"] = metadata["DMS_bin_score"].map(LABEL_MAP)

    logger.info(
        "metadata.csv carregado: %d variantes (%d proteínas)",
        len(metadata), metadata["protein"].nunique(),
    )
    return metadata


def load_wt_embeddings(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"embeddings.npz não encontrado em {path}")

    npz = np.load(path)
    if "wt_embeddings" not in npz:
        raise ValueError(f"embeddings.npz não contém o array 'wt_embeddings' (arrays presentes: {npz.files})")

    wt_embeddings = npz["wt_embeddings"]
    logger.info("wt_embeddings carregado: shape=%s, dtype=%s", wt_embeddings.shape, wt_embeddings.dtype)

    # Validação #2: dimensionalidade original
    if wt_embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"Dimensionalidade inesperada dos embeddings WT: {wt_embeddings.shape[1]} "
            f"(esperado {EXPECTED_EMBEDDING_DIM})"
        )
    return wt_embeddings


def load_esm1v_scores(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"esm1vScores.csv não encontrado em {path}")

    scores = pd.read_csv(path)
    missing = [c for c in SCORES_REQUIRED_COLUMNS if c not in scores.columns]
    if missing:
        raise ValueError(f"esm1vScores.csv sem colunas obrigatórias: {missing}")

    dup = scores.duplicated(subset=["protein", "mutant", "DMS_id"]).sum()
    if dup:
        logger.warning(
            "%d linhas duplicadas em esm1vScores.csv (mesma protein+mutant+DMS_id) — "
            "mantendo a primeira ocorrência.", dup,
        )
        scores = scores.drop_duplicates(subset=["protein", "mutant", "DMS_id"], keep="first")

    logger.info("esm1vScores.csv carregado: %d scores.", len(scores))
    return scores


def load_and_align(
    metadata_path: Path,
    embeddings_path: Path,
    scores_path: Path,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Carrega os três arquivos de cache, junta metadata + esm1v_score (via protein+mutant+DMS_id)
    e devolve (dataframe_alinhado, matriz_de_embeddings_wt_na_mesma_ordem_do_dataframe).

    A matriz de embeddings devolvida está SEMPRE na mesma ordem de linhas do DataFrame
    devolvido, indexada explicitamente por row_id — nunca por posição pós-merge.
    """
    metadata = load_metadata(metadata_path)
    wt_embeddings = load_wt_embeddings(embeddings_path)
    scores = load_esm1v_scores(scores_path)

    # Validação #1 / #8: contagem de embeddings vs metadata original (antes de qualquer filtro)
    raw_metadata = pd.read_csv(metadata_path)
    if len(raw_metadata) != wt_embeddings.shape[0]:
        raise ValueError(
            f"Número de linhas de metadata.csv ({len(raw_metadata)}) difere do número de "
            f"embeddings WT ({wt_embeddings.shape[0]}) — o alinhamento por row_id não é confiável "
            f"se as duas fontes não têm a mesma contagem original."
        )
    logger.info(
        "Checagem de alinhamento OK: %d linhas em metadata.csv == %d embeddings WT.",
        len(raw_metadata), wt_embeddings.shape[0],
    )

    # mapa row_id -> posição no array de embeddings (posição original, antes de qualquer filtro)
    row_id_to_pos = {rid: pos for pos, rid in enumerate(raw_metadata["row_id"].values)}

    # junta o score do ESM-1v via protein+mutant+DMS_id (chave inequívoca)
    data = metadata.merge(
        scores[["protein", "mutant", "DMS_id", "esm1v_score"]],
        on=["protein", "mutant", "DMS_id"],
        how="left",
    )

    if len(data) != len(metadata):
        raise RuntimeError(
            "O merge com esm1vScores.csv alterou o número de linhas "
            f"({len(metadata)} -> {len(data)}) — há duplicatas de protein+mutant+DMS_id "
            "gerando produto cartesiano. Verifique esm1vScores.csv."
        )

    n_missing_score = data["esm1v_score"].isna().sum()
    if n_missing_score:
        logger.warning(
            "%d variantes de metadata.csv não têm esm1v_score correspondente em "
            "esm1vScores.csv — serão descartadas.", n_missing_score,
        )
        data = data.dropna(subset=["esm1v_score"]).reset_index(drop=True)

    # Validação #9: reconstrói a mutação a partir de original_aa/position/mutant_aa e
    # confere consistência interna com a coluna 'mutant'
    reconstructed = data["original_aa"].astype(str) + data["position"].astype(int).astype(str) + data["mutant_aa"].astype(str)
    mismatch = (reconstructed != data["mutant"].astype(str)).sum()
    if mismatch:
        logger.warning(
            "%d linhas com inconsistência entre 'mutant' e original_aa/position/mutant_aa "
            "em metadata.csv — dados possivelmente corrompidos nessas linhas.", mismatch,
        )

    # reindexação explícita do array de embeddings pela row_id (NUNCA por posição pós-merge)
    positions = data["row_id"].map(row_id_to_pos)
    if positions.isna().any():
        raise RuntimeError("Algum row_id de metadata.csv não foi encontrado no mapa original — alinhamento quebrado.")
    embedding_matrix = wt_embeddings[positions.astype(int).values]

    # Validação #7: sem NaN/inf nos embeddings finais
    if not np.isfinite(embedding_matrix).all():
        n_bad = (~np.isfinite(embedding_matrix)).any(axis=1).sum()
        raise ValueError(f"{n_bad} embeddings WT contêm NaN/Inf após alinhamento.")

    # Validação #10: garante que o rótulo não está entre as features (checagem estrutural,
    # reforçada de novo no ponto onde X é montado em pca_pipeline.py)
    assert "DMS_bin_score" not in ("esm1v_score", "embedding_matrix")

    logger.info(
        "Alinhamento final: %d variantes, embeddings shape=%s, %d proteínas.",
        len(data), embedding_matrix.shape, data["protein"].nunique(),
    )
    return data, embedding_matrix
