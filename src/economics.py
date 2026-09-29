import numpy as np

from .config import GAIN, LOSS


def portfolio_profit(y_true, proba, threshold, gain=GAIN, loss=LOSS) -> float:
    "Прибыль портфеля при заданном пороге отказа."
    y_true = np.asarray(y_true)
    approved = np.asarray(proba) < threshold
    return float(
        (approved & (y_true == 0)).sum() * gain
        + (approved & (y_true == 1)).sum() * loss
    )


def baseline_profit(y_true, gain=GAIN, loss=LOSS) -> float:
    "Прибыль при политике одобряем всех"
    y_true = np.asarray(y_true)
    return float((y_true == 0).sum() * gain + (y_true == 1).sum() * loss)


def profit_curve(y_true, proba, thresholds=None, gain=GAIN, loss=LOSS):
    "Прибыль для набора порогов."
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.60, 120)
    profits = np.array([
        portfolio_profit(y_true, proba, t, gain, loss) for t in thresholds
    ])
    return thresholds, profits


def optimal_threshold(y_true, proba, thresholds=None, gain=GAIN, loss=LOSS) -> dict:
    thresholds, profits = profit_curve(y_true, proba, thresholds, gain, loss)
    i = int(profits.argmax())
    base = baseline_profit(y_true, gain, loss)
    return {
        "threshold": float(thresholds[i]),
        "profit": float(profits[i]),
        "baseline_profit": base,
        "uplift": float(profits[i] / base - 1),
        "approval_rate": float((np.asarray(proba) < thresholds[i]).mean()),
    }


def theoretical_threshold(gain=GAIN, loss=LOSS) -> float:
    return gain / (gain + abs(loss))
