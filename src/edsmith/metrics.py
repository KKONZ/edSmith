from __future__ import annotations

import math
from typing import Any

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


def quadratic_weighted_kappa(y_true: list[float], y_pred: list[float]) -> float | None:
    # Convert half-band floats to integers (5.0→10, 5.5→11) so sklearn sees
    # multiclass, not continuous — avoids "mix of continuous and binary" error.
    y_true_int = [round(v * 2) for v in y_true]
    y_pred_int = [round(v * 2) for v in y_pred]
    try:
        result = float(cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic"))
        return None if math.isnan(result) else result
    except ValueError:
        return None


def standardized_mean_difference(y_true: list[float], y_pred: list[float]) -> float:
    """(mean_pred - mean_true) / std_true. Positive = over-prediction."""
    arr_true = np.array(y_true, dtype=float)
    arr_pred = np.array(y_pred, dtype=float)
    std = arr_true.std()
    if std == 0:
        return 0.0
    return float((arr_pred.mean() - arr_true.mean()) / std)


def confusion_matrix(
    y_true: list[float], y_pred: list[float]
) -> dict[str, dict[str, int]]:
    """Confusion matrix as {true_band: {pred_band: count}}, sorted by band."""
    labels = sorted(set(y_true) | set(y_pred))
    label_keys = [f"{v:.1f}" for v in labels]
    matrix: dict[str, dict[str, int]] = {k: {p: 0 for p in label_keys} for k in label_keys}
    for t, p in zip(y_true, y_pred):
        matrix[f"{t:.1f}"][f"{p:.1f}"] += 1
    return matrix


def compute_all(
    y_true: list[float],
    y_pred: list[float],
    bands: list[float] | None = None,
) -> dict[str, Any]:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "adjacent_accuracy": adjacent_accuracy(y_true, y_pred, bands=bands),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "smd": standardized_mean_difference(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
