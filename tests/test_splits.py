from leia_benchmark.splits import make_label_split


def test_split_is_disjoint_and_deterministic():
    ids = [f'p{i:03d}' for i in range(70)]
    a = make_label_split(ids, 0.05, 1337)
    b = make_label_split(ids, 0.05, 1337)
    assert a == b
    assert set(a.labelled).isdisjoint(a.unlabelled)
    assert set(a.labelled) | set(a.unlabelled) == set(ids)
    assert len(a.labelled) == 4


def test_one_percent_rounds_up_to_one():
    ids = [f'p{i:03d}' for i in range(70)]
    split = make_label_split(ids, 0.01, 1)
    assert len(split.labelled) == 1
