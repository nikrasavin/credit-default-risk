"""Загрузка данных и разделение выборок."""

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, RANDOM_STATE, TARGET, TEST_SIZE


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def split_data(df: pd.DataFrame, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
