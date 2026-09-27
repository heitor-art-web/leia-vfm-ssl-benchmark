from __future__ import annotations

import numpy as np
import pytest

from leia_benchmark.evaluation import (
    BinarySegmentationCounts,
    counts_from_masks,
    macro_average_metrics,
    metrics_from_counts,
)


def test_ignore_pixels_never_enter_confusion_counts() -> None:
    target = np.array([[1, 0, 255], [1, 255, 0]], dtype=np.uint8)
    prediction = np.array([[1, 1, 1], [0, 0, 0]], dtype=np.uint8)

    counts = counts_from_masks(target, prediction)

    assert counts == BinarySegmentationCounts(
        tp=1,
        fp=1,
        fn=1,
        tn=1,
        valid_pixels=4,
        ignored_pixels=2,
    )


def test_foreground_metrics_are_computed_from_counts() -> None:
    metrics = metrics_from_counts(BinarySegmentationCounts(tp=3, fp=1, fn=2, tn=10))

    assert metrics["dice"] == pytest.approx(6 / 9)
    assert metrics["iou"] == pytest.approx(3 / 6)
    assert metrics["precision"] == pytest.approx(3 / 4)
    assert metrics["recall"] == pytest.approx(3 / 5)


def test_empty_negative_group_is_undefined_not_perfect() -> None:
    metrics = metrics_from_counts(BinarySegmentationCounts(tn=100, valid_pixels=100))

    assert metrics["dice"] is None
    assert metrics["iou"] is None
    assert metrics["precision"] is None
    assert metrics["recall"] is None


def test_macro_average_reports_number_of_defined_groups() -> None:
    groups = [
        BinarySegmentationCounts(tp=2, fp=0, fn=0, valid_pixels=10),
        BinarySegmentationCounts(tn=10, valid_pixels=10),
    ]

    macro = macro_average_metrics(groups)

    assert macro["dice"]["mean"] == pytest.approx(1.0)
    assert macro["dice"]["n_defined"] == 1
    assert macro["recall"]["n_defined"] == 1


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        counts_from_masks(np.zeros((2, 2)), np.zeros((2, 3)))
