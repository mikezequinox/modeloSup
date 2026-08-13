"""
Pipeline completo: classificação supervisionada de substituições clínicas
(Benign vs Pathogenic) usando score ESM-1v (masked marginal, um único modelo do
ensemble) + Regressão Logística com regularização L2, com divisão treino/validação/
teste agrupada por proteína para evitar data leakage.

Fluxo (conforme o plano):
  Clinical Substitutions -> pré-processamento -> separação por proteína ->
  treino/validação/teste -> ESM-1v -> score da mutação -> padronização ->
  regressão logística -> seleção de C (GroupKFold) -> seleção de threshold (validação) ->
  modelo final -> avaliação no teste (global + por proteína)

Uso:
    python pipeline.py
"""
import json
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

import config
from data_loading import attach_reference_metadata, load_clinical_substitutions, load_reference_table
from esm_scoring import compute_esm_scores
from evaluation import (
    compute_global_metrics,
    compute_per_protein_metrics,
    select_threshold,
    summarize_macro_by_protein,
)
from splitting import group_train_val_test_split, make_group_kfold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline")


def analyze_class_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """Reporta a distribuição de classes Benign/Pathogenic (etapa anterior ao treino, no plano)."""
    counts = data["DMS_bin_score"].value_counts()
    proportions = data["DMS_bin_score"].value_counts(normalize=True)
    summary = pd.DataFrame({"contagem": counts, "proporção": proportions})
    logger.info("Distribuição de classes:\n%s", summary)
    return summary


def decide_class_weight(train: pd.DataFrame) -> bool:
    """Decide se class_weight='balanced' deve ser usado, com base no desbalanceamento no treino."""
    balance = train["label"].value_counts(normalize=True)
    ratio = balance.max() / balance.min()
    use_balanced = ratio > config.IMBALANCE_RATIO_THRESHOLD
    logger.info(
        "Razão de desbalanceamento no treino: %.2f -> class_weight=%s",
        ratio, "'balanced'" if use_balanced else "None",
    )
    return use_balanced


def select_best_C(trainval: pd.DataFrame, class_weight) -> float:
    """
    Seleciona o hiperparâmetro C da Regressão Logística via validação cruzada agrupada
    por proteína (GroupKFold), usando treino+validação — o teste nunca é tocado aqui.
    """
    X = trainval[["esm1v_score"]].values
    y = trainval["label"].values
    groups = trainval["protein"].values

    cv = make_group_kfold(config.N_SPLITS_CV)

    best_C, best_score = config.C_GRID[0], -np.inf
    for C in config.C_GRID:
        fold_scores = []
        for train_idx, val_idx in cv.split(X, y, groups=groups):
            if len(set(y[val_idx])) < 2:
                continue  # AUPRC não é definida com uma única classe no fold

            scaler = StandardScaler().fit(X[train_idx])
            X_train_fold = scaler.transform(X[train_idx])
            X_val_fold = scaler.transform(X[val_idx])

            clf = LogisticRegression(penalty="l2", C=C, class_weight=class_weight, max_iter=1000)
            clf.fit(X_train_fold, y[train_idx])

            y_prob = clf.predict_proba(X_val_fold)[:, 1]
            fold_scores.append(average_precision_score(y[val_idx], y_prob))

        mean_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
        logger.info("C=%.4g -> AUPRC médio (GroupKFold, treino+validação): %.4f", C, mean_score)
        if mean_score > best_score:
            best_score, best_C = mean_score, C

    logger.info("Melhor C selecionado: %.4g (AUPRC médio = %.4f)", best_C, best_score)
    return best_C


def run_pipeline():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carregamento e pré-processamento
    logger.info("Carregando substituições clínicas de %s ...", config.CLINICAL_SUBSTITUTIONS_DIR)
    data = load_clinical_substitutions()
    reference = load_reference_table()
    data = attach_reference_metadata(data, reference)
    analyze_class_distribution(data)

    # 2. Extração de características via ESM-1v (masked marginal, um único modelo)
    logger.info("Calculando/recuperando scores ESM-1v (%s) ...", config.ESM1V_MODEL_NAME)
    data = compute_esm_scores(data)
    before = len(data)
    data = data.dropna(subset=["esm1v_score"]).reset_index(drop=True)
    if len(data) < before:
        logger.warning("%d variantes descartadas por falha no cálculo do score ESM-1v.", before - len(data))

    # 3. Divisão treino/validação/teste agrupada por proteína (sem sobreposição)
    train, val, test = group_train_val_test_split(data)

    class_weight = "balanced" if decide_class_weight(train) else None

    # 4. Seleção de C via GroupKFold, usando treino + validação (teste intocado)
    trainval_for_cv = pd.concat([train, val], ignore_index=True)
    best_C = select_best_C(trainval_for_cv, class_weight)

    # 5. Padronização — fit somente no treino
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[["esm1v_score"]].values)
    X_val = scaler.transform(val[["esm1v_score"]].values)
    X_test = scaler.transform(test[["esm1v_score"]].values)

    y_train = train["label"].values
    y_val = val["label"].values
    y_test = test["label"].values

    # 6. Treino do modelo final, com o C escolhido, usando apenas o conjunto de treino
    model = LogisticRegression(penalty="l2", C=best_C, class_weight=class_weight, max_iter=1000)
    model.fit(X_train, y_train)

    # 7. Seleção do threshold no conjunto de validação
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = select_threshold(
        y_val, val_prob, config.THRESHOLD_GRID, metric=config.THRESHOLD_SELECTION_METRIC
    )

    # 8. Avaliação final no conjunto de teste (usado somente aqui)
    test_prob = model.predict_proba(X_test)[:, 1]
    global_metrics = compute_global_metrics(y_test, test_prob, threshold)

    test_eval = test.copy()
    test_eval["y_prob"] = test_prob
    per_protein = compute_per_protein_metrics(
        test_eval, y_true_col="label", y_prob_col="y_prob", threshold=threshold
    )
    macro_summary = summarize_macro_by_protein(per_protein)

    logger.info("Métricas globais (teste): %s", json.dumps(global_metrics, indent=2, default=str))
    logger.info("Métricas macro por proteína (teste): %s", json.dumps(macro_summary, indent=2, default=str))

    # 9. Persistência dos resultados
    with open(config.OUTPUT_DIR / "metricas_globais.json", "w") as f:
        json.dump(global_metrics, f, indent=2, default=str)
    with open(config.OUTPUT_DIR / "metricas_macro_por_proteina.json", "w") as f:
        json.dump(macro_summary, f, indent=2, default=str)
    per_protein.to_csv(config.OUTPUT_DIR / "metricas_por_proteina.csv", index=False)

    try:
        import joblib
        joblib.dump(
            {"model": model, "scaler": scaler, "threshold": threshold, "C": best_C,
             "class_weight": class_weight, "esm1v_model": config.ESM1V_MODEL_NAME},
            config.OUTPUT_DIR / "modelo_final.joblib",
        )
    except ImportError:
        logger.warning("joblib não instalado — modelo final não foi salvo em disco.")

    logger.info("Pipeline concluído. Resultados salvos em %s", config.OUTPUT_DIR)

    return {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "C": best_C,
        "class_weight": class_weight,
        "global_metrics": global_metrics,
        "macro_summary": macro_summary,
        "per_protein": per_protein,
    }


if __name__ == "__main__":
    run_pipeline()
