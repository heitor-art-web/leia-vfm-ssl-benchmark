import pytest

from leia_benchmark.splits import make_label_split, make_stratified_patient_partition


def test_split_is_disjoint_and_deterministic():
    ids = [f"p{i:03d}" for i in range(70)]
    a = make_label_split(ids, 0.05, 1337)
    b = make_label_split(ids, 0.05, 1337)
    assert a == b
    assert set(a.labelled).isdisjoint(a.unlabelled)
    assert set(a.labelled) | set(a.unlabelled) == set(ids)
    assert len(a.labelled) == 4


def test_one_percent_rounds_up_to_one():
    ids = [f"p{i:03d}" for i in range(70)]
    split = make_label_split(ids, 0.01, 1)
    assert len(split.labelled) == 1


def test_stratified_patient_partition_is_deterministic_and_disjoint():
    mapping = {f"p{i:03d}": int(i % 2 == 0) for i in range(100)}
    a = make_stratified_patient_partition(mapping, seed=1337)
    b = make_stratified_patient_partition(mapping, seed=1337)
    assert a == b
    a.validate()
    assert len(a.train) == 80
    assert len(a.val) == 10
    assert len(a.test) == 10
    assert set(a.train) | set(a.val) | set(a.test) == set(mapping)


def test_stratified_partition_preserves_binary_strata_counts():
    mapping = {f"pos{i:03d}": 1 for i in range(50)}
    mapping.update({f"neg{i:03d}": 0 for i in range(50)})
    split = make_stratified_patient_partition(mapping, seed=7)

    def positives(ids):
        return sum(int(mapping[patient]) for patient in ids)

    assert positives(split.train) == 40
    assert positives(split.val) == 5
    assert positives(split.test) == 5


def test_stratified_partition_rejects_bad_fractions():
    with pytest.raises(ValueError):
        make_stratified_patient_partition(
            {"p001": 1, "p002": 0},
            train_fraction=0.8,
            val_fraction=0.2,
            test_fraction=0.2,
        )
