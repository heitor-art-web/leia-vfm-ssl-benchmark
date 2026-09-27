from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np


@dataclass
class BinarySegmentationCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    valid_pixels: int = 0
    ignored_pixels: int = 0

    def __iadd__(self, other: "BinarySegmentationCounts") -> "BinarySegmentationCounts":
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn
        self.valid_pixels += other.valid_pixels
        self.ignored_pixels += other.ignored_pixels
        return self

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def counts_from_masks(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    positive_label: int = 1,
    ignore_label: int = 255,
) -> BinarySegmentationCounts:
    """Count binary segmentation outcomes while excluding the ignore label entirely."""
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    if target.shape != prediction.shape:
        raise ValueError(f"shape mismatch: target={target.shape}, prediction={prediction.shape}")

    valid = target != ignore_label
    target_positive = target == positive_label
    prediction_positive = prediction == positive_label

    return BinarySegmentationCounts(
        tp=int(np.count_nonzero(valid & target_positive & prediction_positive)),
        fp=int(np.count_nonzero(valid & ~target_positive & prediction_positive)),
        fn=int(np.count_nonzero(valid & target_positive & ~prediction_positive)),
        tn=int(np.count_nonzero(valid & ~target_positive & ~prediction_positive)),
        valid_pixels=int(np.count_nonzero(valid)),
        ignored_pixels=int(np.count_nonzero(~valid)),
    )


def metrics_from_counts(counts: BinarySegmentationCounts) -> dict[str, float | None]:
    """Return foreground Dice/IoU/precision/recall from accumulated pixel counts."""
    tp = float(counts.tp)
    fp = float(counts.fp)
    fn = float(counts.fn)

    dice_den = 2.0 * tp + fp + fn
    iou_den = tp + fp + fn
    precision_den = tp + fp
    recall_den = tp + fn

    return {
        "dice": None if dice_den == 0 else (2.0 * tp) / dice_den,
        "iou": None if iou_den == 0 else tp / iou_den,
        "precision": None if precision_den == 0 else tp / precision_den,
        "recall": None if recall_den == 0 else tp / recall_den,
    }


def macro_average_metrics(
    groups: list[BinarySegmentationCounts],
) -> dict[str, dict[str, float | int | None]]:
    """Macro-average defined group metrics without turning empty-negative groups into perfect scores."""
    metric_names = ("dice", "iou", "precision", "recall")
    values: dict[str, list[float]] = {name: [] for name in metric_names}
    for counts in groups:
        metrics = metrics_from_counts(counts)
        for name in metric_names:
            value = metrics[name]
            if value is not None and isfinite(value):
                values[name].append(float(value))

    output: dict[str, dict[str, float | int | None]] = {}
    for name in metric_names:
        series = values[name]
        output[name] = {
            "mean": None if not series else float(np.mean(series)),
            "std": None if len(series) < 2 else float(np.std(series, ddof=1)),
            "n_defined": len(series),
        }
    return output
