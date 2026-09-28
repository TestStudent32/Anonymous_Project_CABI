"""Accuracy metrics. Predictions are clipped at 0 (trip counts cannot be negative) before scoring."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error over hours with non-zero demand."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else np.nan


def wmape(y_true, y_pred) -> float:
    """Weighted MAPE = sum|error| / sum(demand); robust for low-volume stations."""
    total = float(y_true.sum())
    return float(np.abs(y_true - y_pred).sum() / total * 100) if total else np.nan


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 0)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "WMAPE": wmape(y_true, y_pred),
    }
