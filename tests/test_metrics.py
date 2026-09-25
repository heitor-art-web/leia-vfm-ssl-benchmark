import numpy as np
import pytest

from leia_benchmark.metrics import (
    binary_counts,
    dice_binary,
    dice_label_map,
    iou_label_map,
    label_map_counts,
)


def test_unknown_semantic_pixels_are_not_cast_to_foreground():
    target = np.array([[1, 255], [0, 0]], dtype=np.uint8)
    pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)

    # Perfect on the three valid pixels; the prediction on UNKNOWN is ignored.
    assert dice_label_map(pred, target) == 1.0
    assert iou_label_map(pred, target) == 1.0
    counts = label_map_counts(pred, target)
    assert counts.valid_pixels == 3
    assert (counts.tp, counts.fp, counts.fn, counts.tn) == (1, 0, 0, 2)


def test_false_positive_on_valid_background_is_counted():
    target = np.array([[1, 255], [0, 0]], dtype=np.uint8)
    pred = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    counts = label_map_counts(pred, target)
    assert (counts.tp, counts.fp, counts.fn, counts.tn) == (1, 1, 0, 1)
    assert dice_label_map(pred, target) == pytest.approx(2 / 3)
    assert iou_label_map(pred, target) == pytest.approx(1 / 2)


def test_binary_dice_supports_explicit_validity_mask():
    target = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    pred = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    valid = np.array([[1, 0], [1, 1]], dtype=bool)
    assert dice_binary(pred, target, valid_mask=valid) == 1.0


def test_binary_counts_validates_shapes():
    with pytest.raises(ValueError):
        binary_counts(np.zeros((2, 2)), np.zeros((2, 3)))


def test_empty_valid_foreground_has_explicit_empty_convention():
    target = np.zeros((3, 3), dtype=np.uint8)
    pred = np.zeros_like(target)
    assert dice_label_map(pred, target, empty_value=1.0) == 1.0
    assert dice_label_map(pred, target, empty_value=0.0) == 0.0
