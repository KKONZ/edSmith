from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score


def accuracy(y_true: list[float], y_pred: list[float]) -> float:
    arr_true = np.array(y_true)
    arr_pred = np.array(y_pred)
    return float(np.mean(arr_true == arr_pred))


def adjacent_accuracy(
    y_true: list[float],
    y_pred: list[float],
    bands: list[float] | None = None,
) -> float:
    """Fraction of predictions within one grade band of the true score.

    Tolerance is the smallest step between consecutive bands (defaults to 1.0).
    """
    if bands is not None:
        sorted_bands = sorted(set(bands))
        steps = [b - a for a, b in zip(sorted_bands, sorted_bands[1:])]
        tolerance = min(steps) if steps else 1.0
    else:
        tolerance = 1.0

    arr_true = np.array(y_true, dtype=float)
    arr_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(arr_true - arr_pred) <= tolerance))


def quadratic_weighted_kappa(y_true: list[float], y_pred: list[float]) -> float:
    # Convert half-band floats to integers (5.0→10, 5.5→11) so sklearn sees
    # multiclass, not continuous — avoids "mix of continuous and binary" error.
    y_true_int = [round(v * 2) for v in y_true]
    y_pred_int = [round(v * 2) for v in y_pred]
    try:
        return float(cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic"))
    except ValueError:
        return float("nan")


def standardized_mean_difference(y_true: list[float], y_pred: list[float]) -> float:
    """(mean_pred - mean_true) / std_true. Positive = over-prediction."""
    arr_true = np.array(y_true, dtype=float)
    arr_pred = np.array(y_pred, dtype=float)
    std = arr_true.std()
    if std == 0:
        return 0.0
    return float((arr_pred.mean() - arr_true.mean()) / std)


def compute_all(
    y_true: list[float],
    y_pred: list[float],
    bands: list[float] | None = None,
) -> dict[str, float]:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "adjacent_accuracy": adjacent_accuracy(y_true, y_pred, bands=bands),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "smd": standardized_mean_difference(y_true, y_pred),
    }
