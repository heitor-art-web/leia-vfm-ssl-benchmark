from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class LabelSplit:
    seed: int
    fraction: float
    labelled: list[str]
    unlabelled: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PatientPartition:
    seed: int
    train: list[str]
    val: list[str]
    test: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        train = set(self.train)
        val = set(self.val)
        test = set(self.test)
        if len(train) != len(self.train) or len(val) != len(self.val) or len(test) != len(self.test):
            raise ValueError("duplicate patient id inside a partition")
        if train & val or train & test or val & test:
            raise ValueError("patient leakage across train/val/test")


def make_label_split(patient_ids: Iterable[str], fraction: float, seed: int) -> LabelSplit:
    """Split a frozen train-patient pool into labelled and unlabelled subsets.

    The unit is always a patient, never a slice. At least one patient is labelled.
    """
    ids = sorted(set(patient_ids))
    if not ids:
        raise ValueError("patient_ids is empty")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")

    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_labelled = max(1, math.ceil(len(ids) * fraction))
    labelled = sorted(shuffled[:n_labelled])
    unlabelled = sorted(shuffled[n_labelled:])
    return LabelSplit(seed=seed, fraction=fraction, labelled=labelled, unlabelled=unlabelled)


def _allocate_counts(n: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    """Hamilton/largest-remainder allocation that always sums exactly to n."""
    raw = [n * fraction for fraction in fractions]
    base = [math.floor(value) for value in raw]
    remainder = n - sum(base)
    order = sorted(
        range(3),
        key=lambda index: (raw[index] - base[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        base[index] += 1
    return int(base[0]), int(base[1]), int(base[2])


def make_stratified_patient_partition(
    patient_to_stratum: Mapping[str, str | int],
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 1337,
) -> PatientPartition:
    """Create a deterministic patient split while preserving a coarse stratum.

    The function is deliberately generic. For LIDC v1, the planned stratum is
    whether a patient has at least one nodule cluster supported by >=3 readers.
    More granular stratification should only be added after the annotation audit
    shows that each stratum is large enough.
    """
    if not patient_to_stratum:
        raise ValueError("patient_to_stratum is empty")

    fractions = (train_fraction, val_fraction, test_fraction)
    if any(fraction < 0 for fraction in fractions):
        raise ValueError("split fractions must be non-negative")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train/val/test fractions must sum to 1")

    strata: dict[str, list[str]] = {}
    for patient_id, value in patient_to_stratum.items():
        patient = str(patient_id).strip()
        if not patient:
            raise ValueError("empty patient id")
        strata.setdefault(str(value), []).append(patient)

    rng = random.Random(seed)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []

    for stratum in sorted(strata):
        ids = sorted(set(strata[stratum]))
        rng.shuffle(ids)
        n_train, n_val, n_test = _allocate_counts(len(ids), fractions)
        train.extend(ids[:n_train])
        val.extend(ids[n_train : n_train + n_val])
        test.extend(ids[n_train + n_val : n_train + n_val + n_test])

    partition = PatientPartition(
        seed=seed,
        train=sorted(train),
        val=sorted(val),
        test=sorted(test),
    )
    partition.validate()

    expected = set(patient_to_stratum)
    observed = set(partition.train) | set(partition.val) | set(partition.test)
    if observed != expected:
        raise ValueError("partition does not cover the input patient set exactly")
    return partition
