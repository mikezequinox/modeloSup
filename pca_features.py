"""
Redução de dimensionalidade dos embeddings WT via PCA.

Regra central (sem exceções): o PCA é ajustado (`fit`) exclusivamente com os
embeddings do conjunto de TREINO. Validação e teste passam apenas por `transform`.
"""
import logging

import numpy as np
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def fit_pca(embeddings_train: np.ndarray, n_components: int, random_state: int) -> PCA:
    """Ajusta o PCA usando SOMENTE os embeddings de treino."""
    # svd_solver='randomized' é bem mais rápido aqui: 200 componentes extraídos de 1280
    # dimensões e potencialmente dezenas de milhares de amostras.
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    pca.fit(embeddings_train)

    # Validação #5: confirma que o PCA só viu os dados de treino
    assert pca.n_samples_ == embeddings_train.shape[0], (
        "PCA.n_samples_ não bate com o tamanho do conjunto de treino — "
        "possível vazamento no fit do PCA."
    )

    log_explained_variance(pca)
    return pca


def transform_embeddings(pca: PCA, embeddings: np.ndarray) -> np.ndarray:
    """Aplica um PCA já ajustado (nunca refit) a um novo conjunto de embeddings."""
    return pca.transform(embeddings)


def log_explained_variance(pca: PCA) -> None:
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)

    logger.info(
        "PCA ajustado: %d componentes | variância explicada acumulada: %.2f%%",
        pca.n_components_, cumulative[-1] * 100,
    )
    # Log dos 10 primeiros componentes individualmente, para não poluir o log inteiro
    for i in range(min(10, len(ratios))):
        logger.info(
            "  componente %3d: variância individual = %.4f%% | acumulada = %.4f%%",
            i + 1, ratios[i] * 100, cumulative[i] * 100,
        )
    if len(ratios) > 10:
        logger.info("  ... (%d componentes restantes omitidos do log)", len(ratios) - 10)


def explained_variance_table(pca: PCA):
    """Retorna (variância_individual, variância_acumulada) como arrays, para salvar/plotar."""
    ratios = pca.explained_variance_ratio_
    return ratios, np.cumsum(ratios)
