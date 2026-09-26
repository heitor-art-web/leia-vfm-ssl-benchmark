import random

import pytest

from leia_benchmark.data.lidc_splits import (
    labelled_subset_size,
    nested_labelled_subsets,
    patient_summaries,
    stratified_patient_split,
)


def _rows(n_positive: int = 80, n_negative: int = 20):
    rows = []
    for i in range(n_positive + n_negative):
        positive = i < n_positive
        rows.append(
            {
                "patient_id": f"P{i:04d}",
                "series_uid": f"UID-{i}",
                "n_slices": "100",
                "n_nodules_default": "1" if positive else "0",
                "default_gt_voxels": "500" if positive else "0",
            }
        )
    return rows


def test_patient_summaries_aggregate_multiple_scans_per_patient():
    rows = [
        {
            "patient_id": "P1",
            "series_uid": "A",
            "n_slices": "10",
            "n_nodules_default": "1",
            "default_gt_voxels": "100",
        },
        {
            "patient_id": "P1",
            "series_uid": "B",
            "n_slices": "20",
            "n_nodules_default": "2",
            "default_gt_voxels": "200",
        },
    ]
    patient = patient_summaries(rows)[0]
    assert patient.patient_id == "P1"
    assert patient.n_scans == 2
    assert patient.n_slices == 30
    assert patient.n_default_nodules == 3
    assert patient.default_gt_voxels == 300
    assert patient.has_default_nodule


def test_patient_summaries_reject_duplicate_series_rows():
    rows = _rows(1, 0)
    with pytest.raises(ValueError, match="duplicate patient/series"):
        patient_summaries(rows + rows)


def test_heldout_split_is_disjoint_complete_and_deterministic():
    patients = patient_summaries(_rows())
    first = stratified_patient_split(patients, seed=42)

    shuffled = list(patients)
    random.Random(123).shuffle(shuffled)
    second = stratified_patient_split(shuffled, seed=42)

    assert first == second
    assert len(first["train"]) == 70
    assert len(first["val"]) == 15
    assert len(first["test"]) == 15
    assert not (set(first["train"]) & set(first["val"]))
    assert not (set(first["train"]) & set(first["test"]))
    assert not (set(first["val"]) & set(first["test"]))
    assert len(set().union(*(set(values) for values in first.values()))) == 100


def test_heldout_split_preserves_positive_prevalence_by_stratum():
    patients = patient_summaries(_rows(n_positive=80, n_negative=20))
    split = stratified_patient_split(patients, seed=7)
    positive = {patient.patient_id for patient in patients if patient.has_default_nodule}

    assert len(set(split["train"]) & positive) == 56
    assert len(set(split["val"]) & positive) == 12
    assert len(set(split["test"]) & positive) == 12


def test_labelled_subset_sizes_use_explicit_half_up_rounding():
    assert labelled_subset_size(707, 0.01) == 7
    assert labelled_subset_size(707, 0.05) == 35
    assert labelled_subset_size(707, 0.10) == 71
    assert labelled_subset_size(707, 0.25) == 177


def test_labelled_subsets_are_nested_and_target_blind():
    train_ids = [f"P{i:04d}" for i in range(100)]
    subsets = nested_labelled_subsets(train_ids, seeds=(1, 2))

    for seed in (1, 2):
        one = set(subsets[(seed, 0.01)])
        five = set(subsets[(seed, 0.05)])
        ten = set(subsets[(seed, 0.10)])
        twenty_five = set(subsets[(seed, 0.25)])
        assert len(one) == 1
        assert len(five) == 5
        assert len(ten) == 10
        assert len(twenty_five) == 25
        assert one <= five <= ten <= twenty_five

    assert subsets[(1, 0.25)] != subsets[(2, 0.25)]


def test_labelled_subsets_reject_duplicate_train_ids():
    with pytest.raises(ValueError, match="duplicates"):
        nested_labelled_subsets(["P1", "P1"])
