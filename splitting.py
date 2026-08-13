"""
Divisão dos dados em treino/validação/teste agrupada por proteína, evitando data leakage
(nenhuma proteína pode aparecer em mais de um conjunto).
"""
import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from config import RANDOM_SEED, TEST_SIZE, VAL_SIZE

logger = logging.getLogger(__name__)


def group_train_val_test_split(
    data: pd.DataFrame,
    group_col: str = "protein",
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide `data` em treino/validação/teste garantindo que nenhuma proteína apareça
    em mais de um conjunto simultaneamente.
    """
    groups = data[group_col].values

    # 1ª divisão: separa o teste do restante (treino + validação)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss_test.split(data, groups=groups))

    trainval = data.iloc[trainval_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)

    # 2ª divisão: separa validação do treino, dentro do que sobrou
    relative_val_size = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(trainval, groups=trainval[group_col].values))

    train = trainval.iloc[train_idx].reset_index(drop=True)
    val = trainval.iloc[val_idx].reset_index(drop=True)

    _assert_no_group_overlap(train, val, test, group_col)

    logger.info(
        "Split -> treino: %d variantes / %d proteínas | validação: %d / %d | teste: %d / %d",
        len(train), train[group_col].nunique(),
        len(val), val[group_col].nunique(),
        len(test), test[group_col].nunique(),
    )
    return train, val, test


def _assert_no_group_overlap(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, group_col: str) -> None:
    train_groups = set(train[group_col])
    val_groups = set(val[group_col])
    test_groups = set(test[group_col])

    overlaps = {
        "treino/validação": train_groups & val_groups,
        "treino/teste": train_groups & test_groups,
        "validação/teste": val_groups & test_groups,
    }
    for name, overlap in overlaps.items():
        if overlap:
            raise RuntimeError(f"Vazamento de dados detectado entre {name}: {sorted(overlap)}")


def make_group_kfold(n_splits: int) -> GroupKFold:
    """GroupKFold para validação cruzada agrupada por proteína (usado na seleção de C)."""
    return GroupKFold(n_splits=n_splits)
