"""
Carregamento e consolidação do conjunto de substituições clínicas do ProteinGym.
"""
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config import CLINICAL_SUBSTITUTIONS_DIR, REFERENCE_FILE, LABEL_MAP

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["protein", "protein_sequence", "mutant", "mutated_sequence", "DMS_bin_score"]


def _load_single_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Remove coluna de índice não nomeada, exportada pelo pandas (primeira coluna do CSV)
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"colunas ausentes {missing}")

    # DMS_id é derivado do nome do arquivo (sem extensão), seguindo a convenção do ProteinGym.
    # Isso é usado depois para juntar com o arquivo de referência (clinical_substitutions.csv).
    df["DMS_id"] = csv_path.stem
    return df


def load_clinical_substitutions(directory: Path = CLINICAL_SUBSTITUTIONS_DIR) -> pd.DataFrame:
    """Lê todos os CSVs da pasta de substituições clínicas e concatena em um único DataFrame."""
    directory = Path(directory)
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {directory}. Ajuste CLINICAL_SUBSTITUTIONS_DIR em config.py."
        )

    frames = []
    for csv_path in csv_files:
        try:
            frames.append(_load_single_csv(csv_path))
        except ValueError as exc:
            logger.warning("Ignorando %s: %s", csv_path.name, exc)

    if not frames:
        raise ValueError("Nenhum arquivo válido foi carregado — verifique o formato dos CSVs.")

    data = pd.concat(frames, ignore_index=True)

    # Mantém somente Benign / Pathogenic (descarta indels e eventuais rótulos ambíguos,
    # conforme especificado no plano: "sem incluir variantes do tipo indel")
    before = len(data)
    data = data[data["DMS_bin_score"].isin(LABEL_MAP.keys())].copy()
    if len(data) < before:
        logger.info("Descartadas %d linhas com DMS_bin_score fora de {Benign, Pathogenic}.", before - len(data))

    data["label"] = data["DMS_bin_score"].map(LABEL_MAP)

    # Remove duplicatas exatas (mesma proteína + mesma mutação + mesmo DMS_id)
    data = data.drop_duplicates(subset=["protein", "mutant", "DMS_id"]).reset_index(drop=True)

    logger.info(
        "Total de variantes carregadas: %d (%d proteínas, %d arquivos DMS)",
        len(data), data["protein"].nunique(), data["DMS_id"].nunique(),
    )
    return data


def load_reference_table(path: Path = REFERENCE_FILE) -> Optional[pd.DataFrame]:
    """Carrega o arquivo de referência (metadados por DMS_id). Retorna None se não existir."""
    path = Path(path)
    if not path.exists():
        logger.warning(
            "Arquivo de referência não encontrado em %s — seguindo sem metadados adicionais.", path
        )
        return None
    return pd.read_csv(path)


def attach_reference_metadata(data: pd.DataFrame, reference: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Junta metadados do arquivo de referência (file_length, MSA_len, target_seq etc.) ao
    DataFrame principal, usando DMS_id como chave (derivado de DMS_filename quando necessário).

    Como apenas o score do ESM-1v é usado como feature (o EVE_model_path e os arquivos de MSA
    não são utilizados neste projeto), essa junção serve principalmente para rastreabilidade
    e para checagens de tamanho de sequência, não é estritamente necessária para o treinamento.
    """
    if reference is None:
        return data

    ref = reference.copy()
    if "DMS_id" not in ref.columns:
        if "DMS_filename" in ref.columns:
            ref["DMS_id"] = ref["DMS_filename"].apply(lambda f: Path(str(f)).stem)
        else:
            logger.warning(
                "Referência sem coluna DMS_id/DMS_filename reconhecível — pulando junção de metadados."
            )
            return data

    merged = data.merge(ref, on="DMS_id", how="left", suffixes=("", "_ref"))
    missing_ref = merged["DMS_id"].isin(ref["DMS_id"]) == False  # noqa: E712
    if missing_ref.any():
        logger.warning(
            "%d variantes não encontraram entrada correspondente no arquivo de referência.",
            missing_ref.sum(),
        )
    return merged
