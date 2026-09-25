from __future__ import annotations

import numpy as np


def dice_binary(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * np.logical_and(pred, target).sum() + eps) / (denom + eps))


def dice_per_class(pred: np.ndarray, target: np.ndarray, classes=(1, 2, 3)) -> dict[int, float]:
    return {int(c): dice_binary(pred == c, target == c) for c in classes}


def macro_dice(pred: np.ndarray, target: np.ndarray, classes=(1, 2, 3)) -> float:
    scores = dice_per_class(pred, target, classes)
    return float(np.mean(list(scores.values())))
