import numpy as np
import pytest

from leia_benchmark.foundation.medsam import (
    agreement_mask,
    bbox_from_binary,
    boxes_from_binary_components,
    scale_box_xyxy,
)


def test_scale_box_xyxy_matches_square_resize_geometry():
    box = np.array([10, 20, 110, 220], dtype=np.float32)
    scaled = scale_box_xyxy(box, source_hw=(512, 512), target_size=1024)
    np.testing.assert_allclose(scaled, [20, 40, 220, 440])


def test_scale_box_xyxy_uses_independent_x_y_scales():
    box = np.array([10, 20, 110, 220], dtype=np.float32)
    scaled = scale_box_xyxy(box, source_hw=(400, 800), target_size=1000)
    np.testing.assert_allclose(scaled, [12.5, 50.0, 137.5, 550.0])


def test_bbox_from_binary_is_exclusive_at_max_edge_and_clipped():
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[1:3, 2:5] = 1
    box = bbox_from_binary(mask, min_pixels=1, pad_pixels=2)
    np.testing.assert_array_equal(box, [0, 0, 7, 5])


def test_bbox_from_binary_returns_none_when_component_too_small():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5, 5] = 1
    assert bbox_from_binary(mask, min_pixels=2) is None


def test_component_boxes_keep_separate_nodules_and_drop_tiny_noise():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:5, 2:5] = 1      # 9 px: keep
    mask[12:16, 13:17] = 1  # 16 px: keep
    mask[9, 9] = 1          # 1 px: drop

    boxes = boxes_from_binary_components(mask, min_pixels=9, pad_pixels=0)
    assert len(boxes) == 2
    np.testing.assert_array_equal(boxes[0], [2, 2, 5, 5])
    np.testing.assert_array_equal(boxes[1], [13, 12, 17, 16])


def test_component_boxes_use_eight_connectivity():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1, 1] = 1
    mask[2, 2] = 1
    boxes = boxes_from_binary_components(mask, min_pixels=2, pad_pixels=0)
    assert len(boxes) == 1
    np.testing.assert_array_equal(boxes[0], [1, 1, 3, 3])


def test_agreement_mask_validates_shape():
    a = np.zeros((2, 2), dtype=np.uint8)
    b = np.zeros((2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        agreement_mask(a, b)


def test_agreement_mask_is_pixelwise():
    a = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    b = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    expected = np.array([[True, False], [True, False]])
    np.testing.assert_array_equal(agreement_mask(a, b), expected)
