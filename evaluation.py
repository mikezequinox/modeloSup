"""
Métricas de avaliação: globais e por proteína (macro), além da seleção de threshold.
"""
import logging
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import MIN_VARIANTS_PER_PROTEIN

logger = logging.getLogger(__name__)


def compute_global_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    """Calcula métricas globais, considerando todas as variantes juntas."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    metrics: Dict[str, float] = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }

    if len(set(y_true)) > 1:
        metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
        metrics["auroc"] = roc_auc_score(y_true, y_prob)
        metrics["auprc"] = average_precision_score(y_true, y_prob)
    else:
        metrics["mcc"] = float("nan")
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
        logger.warning("Apenas uma classe presente em y_true — AUROC/AUPRC/MCC não são definidos.")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    return metrics


def compute_per_protein_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_prob_col: str,
    threshold: float,
    protein_col: str = "protein",
    min_variants: int = MIN_VARIANTS_PER_PROTEIN,
) -> pd.DataFrame:
    """
    Calcula métricas individualmente por proteína (uma linha por proteína).
    Proteínas com apenas uma classe presente recebem NaN nas métricas que exigem
    as duas classes (AUROC, AUPRC, MCC), mas continuam contribuindo com
    F1/precision/recall/accuracy (threshold-dependentes).
    """
    rows = []
    for protein, group in df.groupby(protein_col):
        if len(group) < min_variants:
            continue

        y_true = group[y_true_col].values
        y_prob = group[y_prob_col].values
        y_pred = (y_prob >= threshold).astype(int)
        n_classes = len(set(y_true))

        row = {
            "protein": protein,
            "n_variants": len(group),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
        }

        if n_classes > 1:
            row["auroc"] = roc_auc_score(y_true, y_prob)
            row["auprc"] = average_precision_score(y_true, y_prob)
            row["mcc"] = matthews_corrcoef(y_true, y_pred)
        else:
            row["auroc"] = float("nan")
            row["auprc"] = float("nan")
            row["mcc"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_macro_by_protein(per_protein_df: pd.DataFrame) -> Dict[str, float]:
    """Agrega as métricas por proteína em uma média macro (ignora NaN por proteína/métrica)."""
    metric_cols = ["f1", "precision", "recall", "accuracy", "auroc", "auprc", "mcc"]
    summary: Dict[str, float] = {}
    for col in metric_cols:
        if col in per_protein_df.columns:
            summary[f"macro_{col}"] = float(per_protein_df[col].mean(skipna=True))
    summary["n_proteins_evaluated"] = int(len(per_protein_df))
    return summary


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Iterable[float],
    metric: str = "f1",
) -> float:
    """Seleciona, entre `thresholds`, o valor que maximiza `metric` no conjunto de validação."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    best_threshold, best_score = 0.5, -np.inf
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "mcc":
            score = matthews_corrcoef(y_true, y_pred)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Métrica de seleção de threshold desconhecida: {metric!r}")

        if score > best_score:
            best_score, best_threshold = score, t

    logger.info("Threshold selecionado: %.2f (%s de validação = %.4f)", best_threshold, metric, best_score)
    return best_threshold
