"""
Classificação em lote de variantes novas, a partir de um arquivo de mutações.

O modelo ESM-1v + o modelo final (Regressão Logística) são carregados UMA ÚNICA VEZ,
independentemente de quantas variantes existirem no arquivo de entrada.

--------------------------------------------------------------------------------------
Estrutura esperada do arquivo de entrada (CSV):

    protein,protein_sequence,mutant
    NP_000013.2,MKTLLILAVVAAALA...,A329V
    NP_000013.2,MKTLLILAVVAAALA...,K45R
    Q9Y6K9,GATTACAGATTACA...,G12D

Colunas obrigatórias:
    protein            identificador da proteína (livre, usado só para o relatório de saída)
    protein_sequence   sequência selvagem (wild-type) completa
    mutant             mutação no formato <aa_original><posição><aa_mutante>, ex.: A329V

Uma coluna opcional 'id' pode ser incluída para rastrear cada linha; se ausente, o
índice da linha é usado.
--------------------------------------------------------------------------------------

Uso:
    python batch_predict.py --input mutacoes.csv --output resultados_predicao.csv
    python batch_predict.py --input mutacoes.csv                      # gera <input>_predicoes.csv
    python batch_predict.py --input mutacoes.csv --model-path outro_modelo.joblib
"""
import argparse
import logging
import time
from pathlib import Path

import joblib
import pandas as pd

import numpy as np

import config
from esm_scoring import MaskedMarginalScorer, extract_wt_embedding, load_esm1v_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("batch_predict")

REQUIRED_COLUMNS = ["protein", "protein_sequence", "mutant"]


def load_artifact(path: Path = None) -> dict:
    path = Path(path) if path else config.OUTPUT_DIR / "modelo_final.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {path}. Rode `python pipeline.py` primeiro para treiná-lo."
        )
    artifact = joblib.load(path)
    uses_embeddings = "pca" in artifact
    logger.info(
        "Modelo carregado: C=%.4g | threshold=%.2f | class_weight=%s | esm1v_model=%s | features=%s",
        artifact["C"], artifact["threshold"], artifact.get("class_weight"),
        artifact.get("esm1v_model", config.ESM1V_MODEL_NAME),
        f"esm1v_score + {artifact.get('n_pca_components')} PCA" if uses_embeddings else "esm1v_score (única)",
    )
    return artifact


def load_input_file(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {path}")

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colunas ausentes no arquivo de entrada: {missing}. "
            f"Esperado ao menos: {REQUIRED_COLUMNS}"
        )
    if "id" not in df.columns:
        df["id"] = df.index

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    if len(df) < before:
        logger.warning("%d linhas descartadas por valores ausentes em colunas obrigatórias.", before - len(df))

    return df


def score_and_predict(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """
    Calcula as features e a predição final para cada linha, com o modelo ESM-1v
    carregado UMA ÚNICA VEZ. Detecta automaticamente o formato do artefato:
      - artefato "simples" (pipeline.py):            1 feature  -> esm1v_score
      - artefato "com embeddings" (pipeline_with_embeddings.py): 201 features -> esm1v_score + PCA(wt_embedding)

    No modo com embeddings, o embedding WT também é calculado uma única vez POR PROTEÍNA
    (cacheado por protein_sequence), não uma vez por variante — todas as variantes da
    mesma proteína reaproveitam o mesmo embedding.
    """
    uses_embeddings = "pca" in artifact
    model_name = artifact.get("esm1v_model", config.ESM1V_MODEL_NAME)

    logger.info("Carregando modelo ESM-1v (%s) — isso acontece uma única vez...", model_name)
    model, alphabet, device = load_esm1v_model(model_name)
    scorer = MaskedMarginalScorer(model, alphabet, device)

    # cache de embedding WT por sequência (reaproveitado entre todas as variantes da mesma proteína)
    embedding_cache: dict = {}

    def get_wt_embedding(sequence: str) -> np.ndarray:
        if sequence not in embedding_cache:
            emb = extract_wt_embedding(model, alphabet, device, sequence)
            embedding_cache[sequence] = emb.numpy()
        return embedding_cache[sequence]

    df = df.copy()
    scores, errors = [], []
    embeddings = [] if uses_embeddings else None

    total = len(df)
    start = time.time()
    for count, (_, row) in enumerate(df.iterrows(), start=1):
        error_msg = None
        score = float("nan")
        embedding = None
        try:
            score = scorer.score_variant(row["protein_sequence"], row["mutant"])
        except Exception as exc:
            error_msg = str(exc)
            logger.warning("Falha ao pontuar (esm1v_score) linha id=%s (%s): %s", row["id"], row["mutant"], exc)

        if uses_embeddings:
            try:
                embedding = get_wt_embedding(row["protein_sequence"])
            except Exception as exc:
                msg = f"falha no embedding WT: {exc}"
                error_msg = f"{error_msg}; {msg}" if error_msg else msg
                logger.warning("Falha ao extrair embedding WT, linha id=%s (%s): %s", row["id"], row["protein"], exc)

        scores.append(score)
        errors.append(error_msg)
        if uses_embeddings:
            embeddings.append(embedding)

        if count % 100 == 0 or count == total:
            elapsed = time.time() - start
            n_unique_proteins = len(embedding_cache) if uses_embeddings else "-"
            logger.info(
                "Processadas %d/%d variantes (%.1fs decorridos, %s proteínas únicas já embedadas)",
                count, total, elapsed, n_unique_proteins,
            )

    df["esm1v_score"] = scores
    df["error"] = errors

    if uses_embeddings:
        has_embedding = pd.Series([e is not None for e in embeddings], index=df.index)
        valid = df["esm1v_score"].notna() & has_embedding
    else:
        valid = df["esm1v_score"].notna()

    if valid.any():
        esm_scores_valid = df.loc[valid, "esm1v_score"].values.reshape(-1, 1)

        if uses_embeddings:
            emb_matrix = np.vstack([embeddings[i] for i in np.where(valid.values)[0]])
            pca_components = artifact["pca"].transform(emb_matrix)  # só transform, nunca fit aqui
            X = np.hstack([esm_scores_valid, pca_components])
        else:
            X = esm_scores_valid

        X = artifact["scaler"].transform(X)
        prob = artifact["model"].predict_proba(X)[:, 1]
        df.loc[valid, "prob_pathogenic"] = prob
        df.loc[valid, "prediction"] = [
            "Pathogenic" if p >= artifact["threshold"] else "Benign" for p in prob
        ]

    n_failed = (~valid).sum()
    if n_failed:
        logger.warning("%d/%d variantes não puderam ser pontuadas (ver coluna 'error').", n_failed, total)
    if uses_embeddings:
        logger.info("Total de proteínas únicas embedadas: %d (de %d variantes).", len(embedding_cache), total)

    return df


def main():
    parser = argparse.ArgumentParser(description="Classifica em lote variantes de um arquivo de mutações.")
    parser.add_argument("--input", required=True, help="CSV de entrada (protein, protein_sequence, mutant).")
    parser.add_argument("--output", default=None, help="CSV de saída. Padrão: <input>_predicoes.csv")
    parser.add_argument("--model-path", default=None, help="Caminho para modelo_final.joblib.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_predicoes.csv")

    artifact = load_artifact(args.model_path)
    data = load_input_file(input_path)
    logger.info("%d variantes carregadas de %s", len(data), input_path)

    result = score_and_predict(data, artifact)

    result.to_csv(output_path, index=False)
    logger.info("Resultado salvo em %s", output_path)

    n_ok = result["prob_pathogenic"].notna().sum() if "prob_pathogenic" in result.columns else 0
    n_pathogenic = (result.get("prediction") == "Pathogenic").sum()
    logger.info(
        "Resumo: %d/%d classificadas com sucesso | %d previstas como Pathogenic | %d como Benign",
        n_ok, len(result), n_pathogenic, n_ok - n_pathogenic,
    )


if __name__ == "__main__":
    main()