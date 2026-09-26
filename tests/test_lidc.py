import numpy as np
import pytest

from leia_benchmark.data.lidc import (
    ConsensusPolicy,
    IGNORE_LABEL,
    dicom_images_to_hu,
    make_25d_slice,
    merge_semantic_target,
    normalize_hu,
    reader_vote_count,
    semantic_target_from_reader_masks,
)


def test_reader_vote_count_treats_missing_readers_as_zero():
    a = np.zeros((2, 2, 2), dtype=bool)
    b = np.zeros_like(a)
    a[0, 0, 0] = True
    b[0, 0, 0] = True
    b[1, 1, 1] = True

    votes = reader_vote_count([a, b], total_readers=4)

    assert votes[0, 0, 0] == 2
    assert votes[1, 1, 1] == 1


def test_semantic_target_marks_disagreement_as_unknown():
    masks = [np.zeros((2, 2, 1), dtype=bool) for _ in range(4)]
    for mask in masks[:3]:
        mask[0, 0, 0] = True
    masks[0][1, 0, 0] = True

    target = semantic_target_from_reader_masks(
        masks,
        ConsensusPolicy(total_readers=4, positive_readers=3),
    )

    assert target[0, 0, 0] == 1
    assert target[1, 0, 0] == IGNORE_LABEL
    assert target[1, 1, 0] == 0


def test_normalize_hu_clips_and_scales():
    image = np.array([[-1200.0, -1000.0, -300.0, 400.0, 900.0]])
    out = normalize_hu(image, low=-1000, high=400)

    assert out.dtype == np.uint8
    assert out[0, 0] == 0
    assert out[0, 1] == 0
    assert out[0, 3] == 255
    assert out[0, 4] == 255
    assert 0 < out[0, 2] < 255


def test_make_25d_slice_uses_adjacent_slices_and_repeats_edges():
    volume = np.zeros((2, 2, 3), dtype=np.float32)
    volume[:, :, 0] = -1000
    volume[:, :, 1] = -300
    volume[:, :, 2] = 400

    middle = make_25d_slice(volume, 1)
    first = make_25d_slice(volume, 0)

    assert middle.shape == (2, 2, 3)
    assert np.all(middle[:, :, 0] == 0)
    assert np.all(middle[:, :, 2] == 255)
    assert np.array_equal(first[:, :, 0], first[:, :, 1])


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError):
        ConsensusPolicy(total_readers=4, positive_readers=5)


class _FakeDicom:
    def __init__(self, pixels, slope=1.0, intercept=0.0):
        self.pixel_array = np.asarray(pixels)
        self.RescaleSlope = slope
        self.RescaleIntercept = intercept


def test_dicom_images_to_hu_applies_modality_rescale_per_slice():
    a = _FakeDicom([[0, 100], [200, 300]], slope=1, intercept=-1024)
    b = _FakeDicom([[10, 20], [30, 40]], slope=2, intercept=-1000)

    volume = dicom_images_to_hu([a, b])

    assert volume.shape == (2, 2, 2)
    assert volume.dtype == np.float32
    assert volume[0, 0, 0] == -1024
    assert volume[0, 0, 1] == -980


def test_merge_semantic_target_preserves_positive_over_unknown():
    full = np.zeros((3, 3, 2), dtype=np.uint8)
    local = np.array(
        [
            [[1], [255]],
            [[255], [0]],
        ],
        dtype=np.uint8,
    )
    bbox = (slice(0, 2), slice(0, 2), slice(0, 1))
    merge_semantic_target(full, local, bbox)

    assert full[0, 0, 0] == 1
    assert full[0, 1, 0] == 255

    second = np.zeros((2, 2, 1), dtype=np.uint8)
    second[0, 0, 0] = 255
    merge_semantic_target(full, second, bbox)

    assert full[0, 0, 0] == 1
