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
    protein            identificador da proteína
    protein_sequence   sequência selvagem (wild-type) completa
    mutant             mutação no formato <aa_original><posição><aa_mutante>

Uma coluna opcional 'id' pode ser incluída para rastrear cada linha.

A validação verifica:
    - formato da mutação
    - aminoácidos válidos
    - posição existente na sequência
    - aminoácido original realmente presente naquela posição
    - aminoácido mutante diferente do original

--------------------------------------------------------------------------------------
"""

import argparse
import logging
import re
import time
from pathlib import Path

import joblib
import pandas as pd

import config
from esm_scoring import MaskedMarginalScorer, load_esm1v_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger("batch_predict")

REQUIRED_COLUMNS = [
    "protein",
    "protein_sequence",
    "mutant"
]

# Aminoácidos padrão
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Formato esperado:
# A329V
# K45R
# G12D
MUTATION_PATTERN = re.compile(
    r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$"
)


def validate_mutation(sequence: str, mutation: str):
    """
    Valida uma mutação contra a sequência wild-type.

    Retorna:
        (True, None)
    se a mutação for válida.

    Retorna:
        (False, motivo)
    se a mutação for inválida.
    """

    # ------------------------------------------------------------------
    # 1. Verificar se a sequência existe
    # ------------------------------------------------------------------
    if not isinstance(sequence, str) or not sequence:
        return False, "Sequência da proteína vazia ou inválida."

    sequence = sequence.strip().upper()

    # ------------------------------------------------------------------
    # 2. Verificar caracteres da sequência
    # ------------------------------------------------------------------
    invalid_residues = set(sequence) - VALID_AMINO_ACIDS

    if invalid_residues:
        return False, (
            f"Sequência contém aminoácidos/caracteres inválidos: "
            f"{sorted(invalid_residues)}"
        )

    # ------------------------------------------------------------------
    # 3. Verificar formato da mutação
    # ------------------------------------------------------------------
    if not isinstance(mutation, str):
        return False, "Mutação não é uma string."

    mutation = mutation.strip().upper()

    match = MUTATION_PATTERN.fullmatch(mutation)

    if not match:
        return False, (
            f"Formato de mutação inválido: '{mutation}'. "
            "Esperado: <AA_original><posição><AA_mutante>, "
            "por exemplo A329V."
        )

    original_aa, position_str, mutant_aa = match.groups()

    position = int(position_str)

    # ------------------------------------------------------------------
    # 4. Verificar se a posição existe
    # ------------------------------------------------------------------
    if position < 1 or position > len(sequence):
        return False, (
            f"Posição {position} fora dos limites da sequência "
            f"(tamanho={len(sequence)})."
        )

    # ------------------------------------------------------------------
    # 5. Verificar o aminoácido original
    #
    # Sequências biológicas usam indexação começando em 1.
    # Python usa indexação começando em 0.
    #
    # Portanto:
    # posição biológica 1 -> sequence[0]
    # posição biológica 329 -> sequence[328]
    # ------------------------------------------------------------------
    actual_aa = sequence[position - 1]

    if actual_aa != original_aa:
        return False, (
            f"Aminoácido original incorreto na posição {position}: "
            f"mutação informa '{original_aa}', "
            f"mas a sequência possui '{actual_aa}'."
        )

    # ------------------------------------------------------------------
    # 6. Verificar se realmente existe uma substituição
    # ------------------------------------------------------------------
    if original_aa == mutant_aa:
        return False, (
            f"A mutação '{mutation}' não altera o aminoácido "
            f"(original e mutante são '{original_aa}')."
        )

    # ------------------------------------------------------------------
    # Tudo certo
    # ------------------------------------------------------------------
    return True, None


def load_artifact(path: Path = None) -> dict:
    path = Path(path) if path else config.OUTPUT_DIR / "modelo_final.joblib"

    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {path}. "
            "Rode `python pipeline.py` primeiro para treiná-lo."
        )

    artifact = joblib.load(path)

    logger.info(
        "Modelo carregado: C=%.4g | threshold=%.2f | class_weight=%s | esm1v_model=%s",
        artifact["C"],
        artifact["threshold"],
        artifact.get("class_weight"),
        artifact.get("esm1v_model", config.ESM1V_MODEL_NAME),
    )

    return artifact


def load_input_file(path: Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado: {path}"
        )

    df = pd.read_csv(path)

    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colunas ausentes no arquivo de entrada: {missing}. "
            f"Esperado ao menos: {REQUIRED_COLUMNS}"
        )

    if "id" not in df.columns:
        df["id"] = df.index

    # ---------------------------------------------------------------
    # Remover apenas valores ausentes
    # ---------------------------------------------------------------
    before = len(df)

    df = df.dropna(
        subset=REQUIRED_COLUMNS
    ).reset_index(drop=True)

    if len(df) < before:
        logger.warning(
            "%d linhas descartadas por valores ausentes "
            "em colunas obrigatórias.",
            before - len(df)
        )

    # ---------------------------------------------------------------
    # VALIDAR CADA MUTAÇÃO
    # ---------------------------------------------------------------
    validation_results = []

    for _, row in df.iterrows():

        is_valid, error = validate_mutation(
            row["protein_sequence"],
            row["mutant"]
        )

        validation_results.append(
            (is_valid, error)
        )

    df["mutation_valid"] = [
        result[0]
        for result in validation_results
    ]

    df["validation_error"] = [
        result[1]
        for result in validation_results
    ]

    # ---------------------------------------------------------------
    # Estatísticas da validação
    # ---------------------------------------------------------------
    n_valid = df["mutation_valid"].sum()
    n_invalid = len(df) - n_valid

    logger.info(
        "Validação das mutações: %d válidas | %d inválidas",
        n_valid,
        n_invalid
    )

    # Mostrar alguns erros no log
    if n_invalid:
        invalid_rows = df[
            ~df["mutation_valid"]
        ]

        for _, row in invalid_rows.head(10).iterrows():
            logger.warning(
                "Mutação inválida | id=%s | protein=%s | mutant=%s | motivo=%s",
                row["id"],
                row["protein"],
                row["mutant"],
                row["validation_error"]
            )

        if n_invalid > 10:
            logger.warning(
                "... e mais %d mutações inválidas.",
                n_invalid - 10
            )

    return df


def score_and_predict(
    df: pd.DataFrame,
    artifact: dict
) -> pd.DataFrame:
    """
    Calcula o score ESM-1v e a predição final para cada
    mutação válida.

    Mutações inválidas são mantidas no resultado, mas não
    são enviadas para o ESM-1v.
    """

    model_name = artifact.get(
        "esm1v_model",
        config.ESM1V_MODEL_NAME
    )

    logger.info(
        "Carregando modelo ESM-1v (%s) — isso acontece uma única vez...",
        model_name
    )

    model, alphabet, device = load_esm1v_model(
        model_name
    )

    scorer = MaskedMarginalScorer(
        model,
        alphabet,
        device
    )

    df = df.copy()

    scores = []
    errors = []

    total = len(df)

    start = time.time()

    for count, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        # -----------------------------------------------------------
        # Não enviar mutações inválidas para o ESM-1v
        # -----------------------------------------------------------
        if not row["mutation_valid"]:

            scores.append(float("nan"))
            errors.append(
                row["validation_error"]
            )

            continue

        try:

            score = scorer.score_variant(
                row["protein_sequence"],
                row["mutant"]
            )

            error_msg = None

        except Exception as exc:

            score = float("nan")
            error_msg = str(exc)

            logger.warning(
                "Falha na linha id=%s (%s): %s",
                row["id"],
                row["mutant"],
                exc
            )

        scores.append(score)
        errors.append(error_msg)

        if count % 100 == 0 or count == total:

            elapsed = time.time() - start

            logger.info(
                "Processadas %d/%d variantes (%.1fs decorridos)",
                count,
                total,
                elapsed
            )

    df["esm1v_score"] = scores
    df["error"] = errors

    # ---------------------------------------------------------------
    # Modelo supervisionado
    # ---------------------------------------------------------------

    valid = df["esm1v_score"].notna()

    if valid.any():

        X = artifact["scaler"].transform(
            df.loc[
                valid,
                ["esm1v_score"]
            ].values
        )

        prob = artifact["model"].predict_proba(X)[:, 1]

        df.loc[
            valid,
            "prob_pathogenic"
        ] = prob

        df.loc[
            valid,
            "prediction"
        ] = [
            "Pathogenic"
            if p >= artifact["threshold"]
            else "Benign"
            for p in prob
        ]

    n_failed = (~valid).sum()

    if n_failed:

        logger.warning(
            "%d/%d variantes não puderam ser pontuadas "
            "(ver colunas 'error'/'validation_error').",
            n_failed,
            total
        )

    return df


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Classifica em lote variantes de um "
            "arquivo de mutações."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "CSV de entrada "
            "(protein, protein_sequence, mutant)."
        )
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "CSV de saída. "
            "Padrão: <input>_predicoes.csv"
        )
    )

    parser.add_argument(
        "--model-path",
        default=None,
        help="Caminho para modelo_final.joblib."
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(
            input_path.stem + "_predicoes.csv"
        )
    )

    # ---------------------------------------------------------------
    # Carregar modelo
    # ---------------------------------------------------------------

    artifact = load_artifact(
        args.model_path
    )

    # ---------------------------------------------------------------
    # Carregar + validar mutações
    # ---------------------------------------------------------------

    data = load_input_file(
        input_path
    )

    logger.info(
        "%d variantes carregadas de %s",
        len(data),
        input_path
    )

    # ---------------------------------------------------------------
    # ESM-1v + modelo supervisionado
    # ---------------------------------------------------------------

    result = score_and_predict(
        data,
        artifact
    )

    # ---------------------------------------------------------------
    # Salvar resultado
    # ---------------------------------------------------------------

    result.to_csv(
        output_path,
        index=False
    )

    logger.info(
        "Resultado salvo em %s",
        output_path
    )

    # ---------------------------------------------------------------
    # Resumo
    # ---------------------------------------------------------------

    n_ok = (
        result["prob_pathogenic"].notna().sum()
        if "prob_pathogenic" in result.columns
        else 0
    )

    n_pathogenic = (
        result.get("prediction") == "Pathogenic"
    ).sum()

    n_benign = (
        result.get("prediction") == "Benign"
    ).sum()

    n_invalid = (
        (~result["mutation_valid"]).sum()
    )

    logger.info(
        "Resumo: %d/%d classificadas com sucesso | "
        "%d Pathogenic | %d Benign | %d mutações inválidas",
        n_ok,
        len(result),
        n_pathogenic,
        n_benign,
        n_invalid
    )


if __name__ == "__main__":
    main()