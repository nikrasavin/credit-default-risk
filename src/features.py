"Конструирование признаков на основе находок из разведочного анализа."

from functools import partial

import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from .config import LATE_COLS, SENTINEL_MIN


def add_features(
    X: pd.DataFrame,
    mask_codes: bool = True,
    add_flags: bool = True,
    add_total: bool = True,
) -> pd.DataFrame:
    X = X.copy()  
    if add_flags:
        X["late_code"] = (X[LATE_COLS] >= SENTINEL_MIN).any(axis=1).astype(int)

    if mask_codes:
        X[LATE_COLS] = X[LATE_COLS].mask(X[LATE_COLS] >= SENTINEL_MIN)

    if add_flags:
        X["income_missing"] = X["MonthlyIncome"].isna().astype(int)
        X["dependents_missing"] = X["NumberOfDependents"].isna().astype(int)

    if add_total:
        X["late_total"] = X[LATE_COLS].sum(axis=1)

    return X


def make_featurizer(**kwargs) -> FunctionTransformer:
    return FunctionTransformer(partial(add_features, **kwargs))
