import numpy as np
import pytest

from leia_benchmark.data.lidc_targets import semantic_target_from_annotation_count


def test_annotation_count_target_uses_three_vote_threshold():
    count = np.array(
        [
            [[0, 1], [2, 3]],
            [[4, 5], [0, 2]],
        ],
        dtype=np.uint8,
    )

    target = semantic_target_from_annotation_count(count)

    assert target[0, 0, 0] == 0
    assert target[0, 0, 1] == 255
    assert target[0, 1, 0] == 255
    assert target[0, 1, 1] == 1
    assert target[1, 0, 0] == 1
    assert target[1, 0, 1] == 1
    assert target[1, 1, 0] == 0
    assert target[1, 1, 1] == 255


def test_annotation_count_target_does_not_depend_on_default_consensus_mask():
    count = np.zeros((2, 2, 2), dtype=np.uint8)
    count[0, 0, 0] = 2
    count[0, 1, 0] = 3

    target = semantic_target_from_annotation_count(count)

    assert target[0, 0, 0] == 255
    assert target[0, 1, 0] == 1


def test_annotation_count_target_rejects_invalid_counts():
    with pytest.raises(ValueError, match="3D"):
        semantic_target_from_annotation_count(np.zeros((2, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="negative"):
        semantic_target_from_annotation_count(np.array([[[-1]]], dtype=np.int16))
    with pytest.raises(ValueError, match="integer-valued"):
        semantic_target_from_annotation_count(np.array([[[1.5]]], dtype=np.float32))


def test_annotation_count_target_validates_thresholds():
    count = np.zeros((1, 1, 1), dtype=np.uint8)
    with pytest.raises(ValueError, match="positive_annotations"):
        semantic_target_from_annotation_count(count, positive_annotations=0)
    with pytest.raises(ValueError, match="ambiguous_min_annotations"):
        semantic_target_from_annotation_count(
            count,
            positive_annotations=3,
            ambiguous_min_annotations=4,
        )
