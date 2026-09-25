import pytest

from leia_benchmark.splits import (
    make_label_split,
    make_stratified_label_split,
    make_stratified_patient_order,
    make_stratified_patient_partition,
)


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


def test_stratified_patient_order_is_deterministic_and_complete():
    mapping = {f"p{i:03d}": int(i < 30) for i in range(100)}
    a = make_stratified_patient_order(mapping, seed=11)
    b = make_stratified_patient_order(mapping, seed=11)
    assert a == b
    assert len(a) == 100
    assert set(a) == set(mapping)


def test_stratified_prefix_tracks_population_mix():
    mapping = {f"pos{i:03d}": 1 for i in range(30)}
    mapping.update({f"neg{i:03d}": 0 for i in range(70)})
    order = make_stratified_patient_order(mapping, seed=123)

    # Every practical prefix should remain close to the 30% positive target.
    for n in (10, 20, 50, 100):
        positives = sum(mapping[patient] == 1 for patient in order[:n])
        assert abs(positives - 0.30 * n) <= 1


def test_stratified_label_budgets_are_nested():
    mapping = {f"pos{i:03d}": 1 for i in range(40)}
    mapping.update({f"neg{i:03d}": 0 for i in range(60)})
    one = make_stratified_label_split(mapping, 0.01, seed=9)
    five = make_stratified_label_split(mapping, 0.05, seed=9)
    ten = make_stratified_label_split(mapping, 0.10, seed=9)
    twenty_five = make_stratified_label_split(mapping, 0.25, seed=9)

    assert set(one.labelled) <= set(five.labelled)
    assert set(five.labelled) <= set(ten.labelled)
    assert set(ten.labelled) <= set(twenty_five.labelled)
    assert len(one.labelled) == 1
    assert len(five.labelled) == 5
    assert len(ten.labelled) == 10
    assert len(twenty_five.labelled) == 25


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
