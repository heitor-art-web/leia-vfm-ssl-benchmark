import numpy as np
import pytest

from leia_benchmark.data.lidc_validation import (
    LIDCScanSummary,
    choose_lung_like_slice,
    choose_max_positive_slice,
    select_visual_validation_cohort,
)


def _scan(
    patient_id: str,
    *,
    series_uid: str | None = None,
    n_clusters: int = 1,
    cluster_index: int | None = 0,
    annotation_ids: tuple[int, ...] = (1, 2, 3, 4),
    annotation_count: int = 4,
    malignancy: float | None = 4.0,
    diameter: float | None = 10.0,
) -> LIDCScanSummary:
    if n_clusters == 0:
        cluster_index = None
        annotation_ids = ()
        annotation_count = 0
        malignancy = None
        diameter = None
    return LIDCScanSummary(
        patient_id=patient_id,
        series_instance_uid=series_uid or f"uid-{patient_id}",
        n_clusters=n_clusters,
        best_cluster_index=cluster_index,
        best_annotation_ids=annotation_ids,
        best_annotation_count=annotation_count,
        best_malignancy_median=malignancy,
        best_malignancy_mean=malignancy,
        best_malignancy_min=None if malignancy is None else int(malignancy),
        best_malignancy_max=None if malignancy is None else int(malignancy),
        best_diameter_mm_median=diameter,
        best_volume_mm3_median=None if diameter is None else diameter**3,
    )


def test_validation_cohort_prefers_no_nodule_control_and_spreads_sizes():
    scans = [_scan("P000", n_clusters=0)]
    scans += [
        _scan(f"P{i:03d}", diameter=float(i), malignancy=4.0)
        for i in range(1, 9)
    ]

    cohort = select_visual_validation_cohort(scans, n_suspicious=4)

    assert len(cohort) == 5
    assert cohort[0][0] == "control_no_volumetric_nodule"
    assert cohort[0][1].patient_id == "P000"
    suspicious_diameters = [entry.best_diameter_mm_median for _, entry in cohort[1:]]
    assert suspicious_diameters == [1.0, 3.0, 6.0, 8.0]


def test_validation_cohort_uses_low_suspicion_fallback():
    scans = [
        _scan("LOW", malignancy=1.0, diameter=7.0),
        _scan("H1", malignancy=4.0, diameter=5.0),
        _scan("H2", malignancy=4.0, diameter=10.0),
        _scan("H3", malignancy=5.0, diameter=15.0),
        _scan("H4", malignancy=5.0, diameter=20.0),
    ]

    cohort = select_visual_validation_cohort(scans, n_suspicious=4)

    assert cohort[0][0] == "control_low_suspicion"
    assert cohort[0][1].patient_id == "LOW"


def test_validation_cohort_is_patient_unique_for_multi_scan_patients():
    scans = [
        _scan("CONTROL", n_clusters=0),
        _scan("DUAL", series_uid="uid-dual-empty", n_clusters=0),
        _scan("DUAL", series_uid="uid-dual-suspicious", malignancy=5.0, diameter=12.0),
        _scan("H2", malignancy=4.0, diameter=8.0),
        _scan("H3", malignancy=4.0, diameter=16.0),
        _scan("H4", malignancy=5.0, diameter=24.0),
    ]

    cohort = select_visual_validation_cohort(scans, n_suspicious=4)
    patient_ids = [scan.patient_id for _, scan in cohort]

    assert len(patient_ids) == len(set(patient_ids))
    dual = next(scan for _, scan in cohort if scan.patient_id == "DUAL")
    assert dual.series_instance_uid == "uid-dual-suspicious"


def test_validation_cohort_rejects_insufficient_high_suspicion_cases():
    scans = [
        _scan("CONTROL", n_clusters=0),
        _scan("H1", malignancy=4.0),
    ]
    with pytest.raises(ValueError, match="high-suspicion"):
        select_visual_validation_cohort(scans, n_suspicious=4)


def test_choose_lung_like_slice_stays_in_central_range():
    volume = np.zeros((4, 4, 10), dtype=np.float32)
    volume[:] = 100.0
    volume[:, :, 0] = -700.0
    volume[:, :, 7] = -700.0
    volume[:2, :2, 5] = -700.0

    # z=0 has more lung-like pixels but is outside the central 20-80% range.
    assert choose_lung_like_slice(volume) == 7


def test_choose_max_positive_slice_uses_largest_area():
    mask = np.zeros((5, 5, 4), dtype=np.uint8)
    mask[0, 0, 1] = 1
    mask[0:2, 0:3, 3] = 1
    assert choose_max_positive_slice(mask) == 3


def test_choose_max_positive_slice_rejects_empty_mask():
    with pytest.raises(ValueError, match="no positive"):
        choose_max_positive_slice(np.zeros((2, 2, 2), dtype=np.uint8))
