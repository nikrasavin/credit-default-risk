"""Кредитный скоринг на данных Give Me Some Credit.

Модули:
    config     — константы проекта
    data       — загрузка и разделение выборок
    features   — конструирование признаков
    model      — сборка моделей и оценка на кросс-валидации
    economics  — порог отказа и прибыль портфеля
"""

from .data import load_raw, split_data
from .features import add_features, make_featurizer
from .model import build_boosting, build_logreg, evaluate, make_cv, oof_proba
from .economics import (
    baseline_profit,
    optimal_threshold,
    portfolio_profit,
    profit_curve,
    theoretical_threshold,
)

__all__ = [
    "load_raw", "split_data",
    "add_features", "make_featurizer",
    "build_logreg", "build_boosting", "evaluate", "make_cv", "oof_proba",
    "portfolio_profit", "baseline_profit", "profit_curve",
    "optimal_threshold", "theoretical_threshold",
]
