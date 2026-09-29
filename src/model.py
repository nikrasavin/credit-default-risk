import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import N_SPLITS, RANDOM_STATE
from .features import make_featurizer


def make_cv(n_splits=N_SPLITS, random_state=RANDOM_STATE) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def build_logreg(**feature_kwargs) -> Pipeline:
    "Логистическая регрессия — базовая модель для сравнения."
    return Pipeline([
        ("features", make_featurizer(**feature_kwargs)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])


def build_boosting(**feature_kwargs) -> Pipeline:
    "Градиентный бустинг — основная модель проекта."
    return Pipeline([
        ("features", make_featurizer(**feature_kwargs)),
        ("model", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])


def evaluate(model, X, y, cv=None, scoring="roc_auc") -> dict:
    "Оценка модели на кросс-валидации."
    cv = cv or make_cv()
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {"mean": float(scores.mean()), "std": float(scores.std()), "folds": scores}


def oof_proba(model, X, y, cv=None) -> np.ndarray:
    "Out-of-fold вероятности дефолта для всей обучающей выборки."
    cv = cv or make_cv()
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
