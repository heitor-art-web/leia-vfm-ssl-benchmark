import numpy as np
import pytest

from leia_benchmark.data.lidc import (
    BACKGROUND_LABEL,
    IGNORE_LABEL,
    NODULE_LABEL,
    ConsensusPolicy,
    build_scan_target,
    dicom_slices_to_hu,
    conservative_cluster_target,
    merge_cluster_target,
    patient_id_from_path,
    reader_vote_count,
    stack_25d,
    window_to_uint8,
)


def test_dicom_slices_to_hu_applies_modality_rescale():
    class FakeDicom:
        def __init__(self, arr, slope, intercept):
            self.pixel_array = np.asarray(arr, dtype=np.int16)
            self.RescaleSlope = slope
            self.RescaleIntercept = intercept

    images = [
        FakeDicom([[0, 10], [20, 30]], 2.0, -1000.0),
        FakeDicom([[1, 11], [21, 31]], 2.0, -1000.0),
    ]
    volume = dicom_slices_to_hu(images)
    assert volume.shape == (2, 2, 2)
    assert volume.dtype == np.float32
    assert volume[0, 0, 0] == -1000.0
    assert volume[0, 0, 1] == -998.0


def test_window_to_uint8_clips_and_scales():
    x = np.array([-2000.0, -1000.0, -300.0, 400.0, 1000.0])
    y = window_to_uint8(x, -1000.0, 400.0)
    assert y.dtype == np.uint8
    assert y[0] == 0
    assert y[1] == 0
    assert 120 <= y[2] <= 135
    assert y[3] == 255
    assert y[4] == 255


def test_stack_25d_replicates_edges():
    vol = np.zeros((2, 2, 3), dtype=np.int16)
    vol[:, :, 0] = 10
    vol[:, :, 1] = 20
    vol[:, :, 2] = 30

    first = stack_25d(vol, 0)
    middle = stack_25d(vol, 1)
    last = stack_25d(vol, 2)

    assert first[0, 0].tolist() == [10, 10, 20]
    assert middle[0, 0].tolist() == [10, 20, 30]
    assert last[0, 0].tolist() == [20, 30, 30]


def test_reader_vote_count_validates_shapes():
    a = np.zeros((2, 2, 1), dtype=bool)
    b = np.zeros((2, 3, 1), dtype=bool)
    with pytest.raises(ValueError):
        reader_vote_count([a, b])


def test_low_reader_cluster_becomes_unknown_not_negative():
    a = np.zeros((3, 3, 1), dtype=bool)
    b = np.zeros_like(a)
    a[1, 1, 0] = True
    b[1, 1, 0] = True

    target = conservative_cluster_target(
        [a, b], ConsensusPolicy(min_cluster_readers=3, positive_fraction=0.5)
    )
    assert target[1, 1, 0] == IGNORE_LABEL
    assert target[0, 0, 0] == BACKGROUND_LABEL


def test_eligible_cluster_uses_consensus_and_preserves_dispute():
    masks = []
    for _ in range(4):
        masks.append(np.zeros((3, 3, 1), dtype=bool))

    # Core: 3/4 readers.
    masks[0][1, 1, 0] = True
    masks[1][1, 1, 0] = True
    masks[2][1, 1, 0] = True

    # Edge disagreement: 1/4 reader.
    masks[0][1, 2, 0] = True

    target = conservative_cluster_target(
        masks, ConsensusPolicy(min_cluster_readers=3, positive_fraction=0.5)
    )
    assert target[1, 1, 0] == NODULE_LABEL
    assert target[1, 2, 0] == IGNORE_LABEL
    assert target[0, 0, 0] == BACKGROUND_LABEL


def test_positive_precedence_when_merging_overlapping_clusters():
    full = np.zeros((4, 4, 1), dtype=np.uint8)
    bbox = (slice(1, 3), slice(1, 3), slice(0, 1))

    ambiguous = np.full((2, 2, 1), IGNORE_LABEL, dtype=np.uint8)
    positive = np.zeros((2, 2, 1), dtype=np.uint8)
    positive[0, 0, 0] = NODULE_LABEL

    merge_cluster_target(full, ambiguous, bbox)
    merge_cluster_target(full, positive, bbox)

    assert full[1, 1, 0] == NODULE_LABEL
    assert full[2, 2, 0] == IGNORE_LABEL


def test_build_scan_target_keeps_unaffected_background():
    reader_masks = []
    for _ in range(3):
        m = np.zeros((2, 2, 1), dtype=bool)
        m[0, 0, 0] = True
        reader_masks.append(m)

    bbox = (slice(1, 3), slice(1, 3), slice(0, 1))
    target = build_scan_target(
        (4, 4, 1),
        [(reader_masks, bbox)],
        ConsensusPolicy(min_cluster_readers=3, positive_fraction=0.5),
    )
    assert target[1, 1, 0] == NODULE_LABEL
    assert target[0, 0, 0] == BACKGROUND_LABEL


def test_patient_id_from_path():
    assert patient_id_from_path("LIDC-IDRI-0007_z0012") == "LIDC-IDRI-0007"
