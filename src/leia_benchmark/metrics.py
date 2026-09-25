from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinaryCounts:
    tp: int
    fp: int
    fn: int
    tn: int
    valid_pixels: int


def _as_bool_mask(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype == bool:
        return arr
    return arr != 0


def binary_counts(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> BinaryCounts:
    """Count binary outcomes on explicitly valid pixels only.

    `pred` and `target` are interpreted as binary foreground masks. Unknown
    semantic labels must be excluded by the caller through `valid_mask` or by
    using `label_map_counts` below.
    """
    pred_b = _as_bool_mask(pred)
    target_b = _as_bool_mask(target)
    if pred_b.shape != target_b.shape:
        raise ValueError("pred and target must have identical shape")

    if valid_mask is None:
        valid = np.ones(pred_b.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != pred_b.shape:
            raise ValueError("valid_mask must have the same shape as pred/target")

    p = pred_b[valid]
    t = target_b[valid]
    tp = int(np.logical_and(p, t).sum())
    fp = int(np.logical_and(p, ~t).sum())
    fn = int(np.logical_and(~p, t).sum())
    tn = int(np.logical_and(~p, ~t).sum())
    return BinaryCounts(tp=tp, fp=fp, fn=fn, tn=tn, valid_pixels=int(valid.sum()))


def label_map_counts(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    positive_label: int = 1,
    ignore_label: int | None = 255,
) -> BinaryCounts:
    """Count positive/negative outcomes while excluding UNKNOWN target pixels."""
    pred_arr = np.asarray(pred)
    target_arr = np.asarray(target)
    if pred_arr.shape != target_arr.shape:
        raise ValueError("pred and target must have identical shape")

    valid = (
        np.ones(target_arr.shape, dtype=bool)
        if ignore_label is None
        else target_arr != ignore_label
    )
    pred_fg = pred_arr == positive_label
    target_fg = target_arr == positive_label
    return binary_counts(pred_fg, target_fg, valid_mask=valid)


def dice_from_counts(counts: BinaryCounts, *, empty_value: float = 1.0) -> float:
    denom = 2 * counts.tp + counts.fp + counts.fn
    if denom == 0:
        return float(empty_value)
    return float((2 * counts.tp) / denom)


def iou_from_counts(counts: BinaryCounts, *, empty_value: float = 1.0) -> float:
    denom = counts.tp + counts.fp + counts.fn
    if denom == 0:
        return float(empty_value)
    return float(counts.tp / denom)


def precision_from_counts(counts: BinaryCounts, *, empty_value: float = 1.0) -> float:
    denom = counts.tp + counts.fp
    if denom == 0:
        return float(empty_value)
    return float(counts.tp / denom)


def recall_from_counts(counts: BinaryCounts, *, empty_value: float = 1.0) -> float:
    denom = counts.tp + counts.fn
    if denom == 0:
        return float(empty_value)
    return float(counts.tp / denom)


def dice_binary(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    empty_value: float = 1.0,
) -> float:
    """Binary Dice with optional validity mask.

    For semantic LIDC targets containing 255, prefer `dice_label_map` so UNKNOWN
    cannot accidentally become foreground through boolean casting.
    """
    return dice_from_counts(
        binary_counts(pred, target, valid_mask=valid_mask),
        empty_value=empty_value,
    )


def dice_label_map(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    positive_label: int = 1,
    ignore_label: int | None = 255,
    empty_value: float = 1.0,
) -> float:
    return dice_from_counts(
        label_map_counts(
            pred,
            target,
            positive_label=positive_label,
            ignore_label=ignore_label,
        ),
        empty_value=empty_value,
    )


def iou_label_map(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    positive_label: int = 1,
    ignore_label: int | None = 255,
    empty_value: float = 1.0,
) -> float:
    return iou_from_counts(
        label_map_counts(
            pred,
            target,
            positive_label=positive_label,
            ignore_label=ignore_label,
        ),
        empty_value=empty_value,
    )


def dice_per_class(
    pred: np.ndarray,
    target: np.ndarray,
    classes=(1, 2, 3),
    *,
    ignore_label: int | None = None,
) -> dict[int, float]:
    target_arr = np.asarray(target)
    valid = (
        np.ones(target_arr.shape, dtype=bool)
        if ignore_label is None
        else target_arr != ignore_label
    )
    return {
        int(c): dice_binary(pred == c, target_arr == c, valid_mask=valid)
        for c in classes
    }


def macro_dice(
    pred: np.ndarray,
    target: np.ndarray,
    classes=(1, 2, 3),
    *,
    ignore_label: int | None = None,
) -> float:
    scores = dice_per_class(pred, target, classes, ignore_label=ignore_label)
    if not scores:
        raise ValueError("classes must not be empty")
    return float(np.mean(list(scores.values())))
