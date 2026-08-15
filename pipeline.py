"""
Pipeline com features estendidas: esm1v_score + 200 componentes PCA do embedding WT
do ESM-1v (201 features no total), mantendo a mesma Regressão Logística L2, a mesma
divisão treino/validação/teste agrupada por proteína e as mesmas métricas do pipeline
original (pipeline.py) — que continua intocado e funcional como está.

ATENÇÃO: esta é uma extensão explícita além do plano original, que especifica uma
única feature de entrada (esm1v_score). Ver README.md para mais contexto.

Fontes de dados esperadas (pré-computadas por outro processo):
    cache/metadata.csv     -> row_id, protein, mutant, position, original_aa, mutant_aa,
                               DMS_bin_score, source_file
    cache/embeddings.npz   -> wt_embeddings (N, 1280)
    cache/esm1vScores.csv  -> protein, mutant, DMS_id, esm1v_score

Este script NÃO recalcula embeddings nem scores ESM-1v — apenas consome os arquivos
de cache já existentes. Por isso não depende de torch/fair-esm para treinar (só a
inferência em variantes novas, em batch_predict.py, precisa desses pacotes).

Uso:
    python pipeline_with_embeddings.py
"""
import json
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

import config
from cache_data_loading import load_and_align
from evaluation import (
    compute_global_metrics,
    compute_per_protein_metrics,
    select_threshold,
    summarize_macro_by_protein,
)
from pca_features import explained_variance_table, fit_pca, transform_embeddings
from splitting import group_train_val_test_split_with_arrays, make_group_kfold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pipeline_with_embeddings")


def decide_class_weight(train: pd.DataFrame) -> bool:
    balance = train["label"].value_counts(normalize=True)
    ratio = balance.max() / balance.min()
    use_balanced = ratio > config.IMBALANCE_RATIO_THRESHOLD
    logger.info(
        "Razão de desbalanceamento no treino: %.2f -> class_weight=%s",
        ratio, "'balanced'" if use_balanced else "None",
    )
    return use_balanced


def build_features(esm1v_score: np.ndarray, pca_embeddings: np.ndarray) -> np.ndarray:
    """Concatena esm1v_score (1 coluna) com os componentes PCA (N colunas) -> matriz final."""
    esm1v_score = np.asarray(esm1v_score).reshape(-1, 1)
    X = np.hstack([esm1v_score, pca_embeddings])
    if not np.isfinite(X).all():
        raise ValueError("Features finais contêm NaN/Inf — verifique esm1v_score e o PCA dos embeddings.")
    return X


def select_best_C(
    trainval_data: pd.DataFrame,
    trainval_embeddings: np.ndarray,
    class_weight,
    n_pca_components: int,
) -> float:
    """
    Seleciona C via GroupKFold. Para evitar QUALQUER vazamento (mesmo dentro da CV),
    o PCA e o StandardScaler são reajustados a cada fold, usando somente a porção de
    treino daquele fold — o mesmo padrão de rigor que o pipeline.py original já usa
    para o StandardScaler na seleção de C.
    """
    y = trainval_data["label"].values
    groups = trainval_data["protein"].values
    esm_scores = trainval_data["esm1v_score"].values

    cv = make_group_kfold(config.N_SPLITS_CV)

    best_C, best_score = config.C_GRID[0], -np.inf
    for C in config.C_GRID:
        fold_scores = []
        for train_idx, val_idx in cv.split(trainval_embeddings, y, groups=groups):
            if len(set(y[val_idx])) < 2:
                continue

            pca_fold = fit_pca(trainval_embeddings[train_idx], n_pca_components, config.RANDOM_SEED)
            pca_train = transform_embeddings(pca_fold, trainval_embeddings[train_idx])
            pca_val = transform_embeddings(pca_fold, trainval_embeddings[val_idx])

            X_train_fold = build_features(esm_scores[train_idx], pca_train)
            X_val_fold = build_features(esm_scores[val_idx], pca_val)

            scaler = StandardScaler().fit(X_train_fold)
            X_train_fold = scaler.transform(X_train_fold)
            X_val_fold = scaler.transform(X_val_fold)

            clf = LogisticRegression(penalty="l2", C=C, class_weight=class_weight, max_iter=2000)
            clf.fit(X_train_fold, y[train_idx])

            y_prob = clf.predict_proba(X_val_fold)[:, 1]
            fold_scores.append(average_precision_score(y[val_idx], y_prob))

        mean_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
        logger.info("C=%.4g -> AUPRC médio (GroupKFold, PCA refeito por fold): %.4f", C, mean_score)
        if mean_score > best_score:
            best_score, best_C = mean_score, C

    logger.info("Melhor C selecionado: %.4g (AUPRC médio = %.4f)", best_C, best_score)
    return best_C


def run_pipeline():
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carrega e alinha metadata + embeddings WT + scores ESM-1v (validações #1,#2,#7,#8,#9,#10
    #    já são feitas dentro de load_and_align)
    logger.info("Carregando e alinhando arquivos de cache...")
    data, embeddings = load_and_align(
        config.CACHE_METADATA_FILE, config.CACHE_EMBEDDINGS_FILE, config.CACHE_ESM1V_SCORES_FILE
    )

    # Validação #3 (n_components configurado) só faz sentido comparar depois do fit, mas o
    # limite superior já pode ser checado agora:
    if config.N_PCA_COMPONENTS >= embeddings.shape[1]:
        raise ValueError(
            f"N_PCA_COMPONENTS ({config.N_PCA_COMPONENTS}) deve ser menor que a dimensão "
            f"original dos embeddings ({embeddings.shape[1]})."
        )

    # 2. Divisão treino/validação/teste agrupada por proteína — o embedding de cada linha
    #    viaja junto, particionado com os MESMOS índices, nunca reindexado por posição.
    train, val, test, train_arr, val_arr, test_arr = group_train_val_test_split_with_arrays(
        data, {"wt_embedding": embeddings}
    )
    emb_train, emb_val, emb_test = train_arr["wt_embedding"], val_arr["wt_embedding"], test_arr["wt_embedding"]

    class_weight = "balanced" if decide_class_weight(train) else None

    # 3. Seleção de C via GroupKFold sobre treino+validação (teste nunca é tocado;
    #    PCA e scaler são reajustados a cada fold dentro de select_best_C)
    trainval_data = pd.concat([train, val], ignore_index=True)
    trainval_embeddings = np.vstack([emb_train, emb_val])
    best_C = select_best_C(trainval_data, trainval_embeddings, class_weight, config.N_PCA_COMPONENTS)

    # 4. PCA final — ajustado SOMENTE com o embedding do conjunto de treino definitivo
    logger.info("Ajustando PCA final (somente com o conjunto de treino, %d amostras)...", len(train))
    pca = fit_pca(emb_train, config.N_PCA_COMPONENTS, config.RANDOM_SEED)

    pca_train = transform_embeddings(pca, emb_train)   # fit já feito acima
    pca_val = transform_embeddings(pca, emb_val)        # só transform
    pca_test = transform_embeddings(pca, emb_test)      # só transform

    # Validação #4: número final de features
    n_features = 1 + pca_train.shape[1]
    assert n_features == 1 + config.N_PCA_COMPONENTS, f"Esperado {1+config.N_PCA_COMPONENTS} features, obtido {n_features}"
    logger.info("Features finais: %d (1 esm1v_score + %d componentes PCA)", n_features, pca_train.shape[1])

    X_train_raw = build_features(train["esm1v_score"].values, pca_train)
    X_val_raw = build_features(val["esm1v_score"].values, pca_val)
    X_test_raw = build_features(test["esm1v_score"].values, pca_test)

    # Validação #10 (reforço): confirma que DMS_bin_score/label não entraram nas colunas de X
    # (estrutural: build_features só recebe esm1v_score + pca, nunca o dataframe inteiro)

    # 5. Padronização final — fit somente no treino
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    y_train = train["label"].values
    y_val = val["label"].values
    y_test = test["label"].values

    # 6. Treino do modelo final
    model = LogisticRegression(penalty="l2", C=best_C, class_weight=class_weight, max_iter=2000)
    model.fit(X_train, y_train)

    # 7. Seleção do threshold no conjunto de validação
    val_prob = model.predict_proba(X_val)[:, 1]
    threshold = select_threshold(
        y_val, val_prob, config.THRESHOLD_GRID, metric=config.THRESHOLD_SELECTION_METRIC
    )

    # 8. Avaliação final no conjunto de teste
    test_prob = model.predict_proba(X_test)[:, 1]
    global_metrics = compute_global_metrics(y_test, test_prob, threshold)

    test_eval = test.copy()
    test_eval["y_prob"] = test_prob
    per_protein = compute_per_protein_metrics(
        test_eval, y_true_col="label", y_prob_col="y_prob", threshold=threshold
    )
    macro_summary = summarize_macro_by_protein(per_protein)

    logger.info("Métricas globais (teste, com embeddings): %s", json.dumps(global_metrics, indent=2, default=str))
    logger.info("Métricas macro por proteína (teste, com embeddings): %s", json.dumps(macro_summary, indent=2, default=str))

    # 9. Persistência — inclui o PCA, para reuso idêntico na inferência
    ratios, cumulative = explained_variance_table(pca)
    with open(config.OUTPUT_DIR / "pca_variancia_explicada.json", "w") as f:
        json.dump({
            "n_components": int(pca.n_components_),
            "variancia_individual": ratios.tolist(),
            "variancia_acumulada": cumulative.tolist(),
        }, f, indent=2)

    with open(config.OUTPUT_DIR / "metricas_globais_embeddings.json", "w") as f:
        json.dump(global_metrics, f, indent=2, default=str)
    with open(config.OUTPUT_DIR / "metricas_macro_por_proteina_embeddings.json", "w") as f:
        json.dump(macro_summary, f, indent=2, default=str)
    per_protein.to_csv(config.OUTPUT_DIR / "metricas_por_proteina_embeddings.csv", index=False)

    try:
        import joblib
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "pca": pca,
                "threshold": threshold,
                "C": best_C,
                "class_weight": class_weight,
                "n_pca_components": config.N_PCA_COMPONENTS,
                "wt_embedding_dim": config.WT_EMBEDDING_DIM,
                "esm1v_model": config.ESM1V_MODEL_NAME,
                "feature_order": ["esm1v_score"] + [f"pca_{i+1}" for i in range(config.N_PCA_COMPONENTS)],
            },
            config.EMBEDDINGS_MODEL_ARTIFACT,
        )
        logger.info("Modelo final (com embeddings) salvo em %s", config.EMBEDDINGS_MODEL_ARTIFACT)
    except ImportError:
        logger.warning("joblib não instalado — modelo final não foi salvo em disco.")

    logger.info("Pipeline (embeddings) concluído. Resultados em %s", config.OUTPUT_DIR)

    return {
        "model": model,
        "scaler": scaler,
        "pca": pca,
        "threshold": threshold,
        "C": best_C,
        "class_weight": class_weight,
        "global_metrics": global_metrics,
        "macro_summary": macro_summary,
        "per_protein": per_protein,
    }


if __name__ == "__main__":
    run_pipeline()