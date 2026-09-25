import numpy as np
import pytest

from leia_benchmark.data.lidc_mirror import (
    medotter_case_map,
    medotter_scan_summaries,
    semantic_target_from_default_and_annotation_count,
)
from leia_benchmark.data.lidc_validation import select_visual_validation_cohort


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


def test_medotter_summaries_and_selection():
    scans = [
        {
            "case_id": "CTRL_case",
            "patient_id": "CTRL",
            "series_uid": "uid-ctrl",
            "n_nodules": "0",
        },
        {
            "case_id": "A_case",
            "patient_id": "A",
            "series_uid": "uid-a",
            "n_nodules": "1",
        },
        {
            "case_id": "B_case",
            "patient_id": "B",
            "series_uid": "uid-b",
            "n_nodules": "1",
        },
        {
            "case_id": "C_case",
            "patient_id": "C",
            "series_uid": "uid-c",
            "n_nodules": "1",
        },
        {
            "case_id": "D_case",
            "patient_id": "D",
            "series_uid": "uid-d",
            "n_nodules": "1",
        },
    ]
    nodules = [
        {
            "case_id": f"{patient}_case",
            "nodule_id": str(i),
            "n_annotations": "4",
            "in_default_gt": "true",
            "malignancy": "5",
            "diameter_mm": str(diameter),
            "volume_mm3": "100",
        }
        for i, (patient, diameter) in enumerate(
            [("A", 5), ("B", 10), ("C", 15), ("D", 20)]
        )
    ]

    summaries = medotter_scan_summaries(scans, nodules)
    cohort = select_visual_validation_cohort(summaries, n_suspicious=4)

    assert cohort[0][0] == "control_no_volumetric_nodule"
    assert cohort[0][1].patient_id == "CTRL"
    assert {scan.patient_id for _, scan in cohort[1:]} == {"A", "B", "C", "D"}


def test_medotter_parser_prioritizes_default_multi_annotation_nodule():
    scans = [
        {
            "case_id": "P_case",
            "patient_id": "P",
            "series_uid": "uid-p",
            "n_nodules": "2",
        }
    ]
    nodules = [
        {
            "case_id": "P_case",
            "nodule_id": "1",
            "n_annotations": "1",
            "in_default_gt": "false",
            "malignancy": "5",
            "diameter_mm": "30",
        },
        {
            "case_id": "P_case",
            "nodule_id": "2",
            "n_annotations": "4",
            "in_default_gt": "true",
            "malignancy": "4",
            "diameter_mm": "12",
        },
    ]

    summary = medotter_scan_summaries(scans, nodules)[0]
    assert summary.best_cluster_index == 2
    assert summary.best_reader_count == 4
    assert summary.best_malignancy_median == 4.0


def test_medotter_parser_fails_on_missing_nodule_rows():
    scans = [
        {
            "case_id": "P_case",
            "patient_id": "P",
            "series_uid": "uid-p",
            "n_nodules": "1",
        }
    ]
    with pytest.raises(ValueError, match="no matching rows"):
        medotter_scan_summaries(scans, [])
