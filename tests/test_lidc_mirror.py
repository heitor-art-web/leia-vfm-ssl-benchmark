import numpy as np
import pytest

from leia_benchmark.data.lidc_mirror import (
    medotter_case_map,
    medotter_scan_summaries,
    selected_instance_target,
    semantic_target_from_default_and_annotation_count,
)
from leia_benchmark.data.lidc_validation import (
    choose_max_positive_slice,
    select_visual_validation_cohort,
)


def test_medotter_case_map_uses_patient_and_series_uid():
    rows = [
        {"case_id": "P1_abc", "patient_id": "P1", "series_uid": "uid-a"},
        {"case_id": "P1_def", "patient_id": "P1", "series_uid": "uid-b"},
    ]
    mapping = medotter_case_map(rows)
    assert mapping[("P1", "uid-a")] == "P1_abc"
    assert mapping[("P1", "uid-b")] == "P1_def"


def test_medotter_target_keeps_excluded_annotation_evidence_unknown():
    default = np.zeros((2, 2, 2), dtype=np.uint8)
    count = np.zeros_like(default)
    default[0, 0, 0] = 1
    count[0, 0, 0] = 3
    count[0, 1, 0] = 2
    count[1, 0, 0] = 1

    target = semantic_target_from_default_and_annotation_count(default, count)

    assert target[0, 0, 0] == 1
    assert target[0, 1, 0] == 255
    assert target[1, 0, 0] == 255
    assert target[1, 1, 0] == 0


def test_medotter_target_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shapes differ"):
        semantic_target_from_default_and_annotation_count(
            np.zeros((2, 2, 2), dtype=np.uint8),
            np.zeros((2, 2, 3), dtype=np.uint8),
        )


def test_selected_instance_target_isolates_requested_nodule_and_slice():
    instances = np.zeros((6, 6, 4), dtype=np.uint16)
    # A large unrelated nodule peaks on z=0.
    instances[0:5, 0:5, 0] = 1
    # Selected nodule id=2 peaks on z=3.
    instances[0, 0, 1] = 2
    instances[2:4, 2:5, 3] = 2

    selected = selected_instance_target(instances, 2)

    assert set(np.unique(selected)) == {0, 1}
    assert int(selected.sum()) == 7
    assert not np.any(selected[instances == 1])
    assert choose_max_positive_slice(selected) == 3


def test_selected_instance_target_rejects_missing_id():
    instances = np.zeros((2, 2, 2), dtype=np.uint16)
    instances[0, 0, 0] = 1
    with pytest.raises(ValueError, match="available ids: \[1\]"):
        selected_instance_target(instances, 2)


def test_selected_instance_target_rejects_zero_id():
    with pytest.raises(ValueError, match=">= 1"):
        selected_instance_target(np.zeros((2, 2, 2), dtype=np.uint16), 0)


def test_medotter_summaries_and_selection_with_live_field_names():
    scans = [
        {"case_id": "CTRL_case", "patient_id": "CTRL", "series_uid": "uid-ctrl", "n_nodules": "0"},
        {"case_id": "A_case", "patient_id": "A", "series_uid": "uid-a", "n_nodules": "1"},
        {"case_id": "B_case", "patient_id": "B", "series_uid": "uid-b", "n_nodules": "1"},
        {"case_id": "C_case", "patient_id": "C", "series_uid": "uid-c", "n_nodules": "1"},
        {"case_id": "D_case", "patient_id": "D", "series_uid": "uid-d", "n_nodules": "1"},
    ]
    nodules = [
        {
            "case_id": f"{patient}_case",
            "nodule_id": str(i),
            "n_annotations": "4",
            "in_default_gt": "True",
            "malignancy_score": "5",
            "diameter_mm": str(diameter),
            "volume_mm3": "100",
        }
        for i, (patient, diameter) in enumerate(
            [("A", 5), ("B", 10), ("C", 15), ("D", 20)], start=1
        )
    ]

    summaries = medotter_scan_summaries(scans, nodules)
    cohort = select_visual_validation_cohort(summaries, n_suspicious=4)

    assert cohort[0][0] == "control_no_volumetric_nodule"
    assert cohort[0][1].patient_id == "CTRL"
    assert {scan.patient_id for _, scan in cohort[1:]} == {"A", "B", "C", "D"}


def test_medotter_parser_prioritizes_default_multi_annotation_nodule():
    scans = [
        {"case_id": "P_case", "patient_id": "P", "series_uid": "uid-p", "n_nodules": "2"}
    ]
    nodules = [
        {
            "case_id": "P_case",
            "nodule_id": "1",
            "n_annotations": "1",
            "in_default_gt": "False",
            "malignancy_score": "5",
            "diameter_mm": "30",
        },
        {
            "case_id": "P_case",
            "nodule_id": "2",
            "n_annotations": "4",
            "in_default_gt": "True",
            "malignancy_score": "4",
            "diameter_mm": "12",
        },
    ]

    summary = medotter_scan_summaries(scans, nodules)[0]
    assert summary.best_cluster_index == 2
    assert summary.best_annotation_count == 4
    assert summary.best_malignancy_median == 4.0


def test_medotter_parser_fails_on_missing_nodule_rows():
    scans = [
        {"case_id": "P_case", "patient_id": "P", "series_uid": "uid-p", "n_nodules": "1"}
    ]
    with pytest.raises(ValueError, match="no matching rows"):
        medotter_scan_summaries(scans, [])
