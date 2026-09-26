from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class LIDCPatientSummary:
    patient_id: str
    n_scans: int
    n_slices: int
    n_default_nodules: int
    default_gt_voxels: int

    @property
    def has_default_nodule(self) -> bool:
        return self.n_default_nodules > 0 or self.default_gt_voxels > 0


def _as_int(value: object | None, default: int = 0) -> int:
    if value is None or str(value).strip() == "":
        return default
    return int(float(str(value)))


def patient_summaries(scan_rows: Iterable[Mapping[str, object]]) -> list[LIDCPatientSummary]:
    """Aggregate MedOtter ``scans.csv`` rows to patient-level benchmark units."""
    aggregate: dict[str, dict[str, int]] = {}
    seen_series: set[tuple[str, str]] = set()

    for row in scan_rows:
        patient_id = str(row.get("patient_id", "")).strip()
        if not patient_id:
            raise ValueError("scan row is missing patient_id")
        series_uid = str(row.get("series_uid", row.get("series_instance_uid", ""))).strip()
        if not series_uid:
            raise ValueError(f"{patient_id} scan row is missing series UID")
        key = (patient_id, series_uid)
        if key in seen_series:
            raise ValueError(f"duplicate patient/series row: {patient_id} / {series_uid}")
        seen_series.add(key)

        current = aggregate.setdefault(
            patient_id,
            {
                "n_scans": 0,
                "n_slices": 0,
                "n_default_nodules": 0,
                "default_gt_voxels": 0,
            },
        )
        current["n_scans"] += 1
        current["n_slices"] += _as_int(row.get("n_slices"))
        current["n_default_nodules"] += _as_int(row.get("n_nodules_default"))
        current["default_gt_voxels"] += _as_int(row.get("default_gt_voxels"))

    if not aggregate:
        raise ValueError("no scan rows supplied")

    return [
        LIDCPatientSummary(patient_id=patient_id, **values)
        for patient_id, values in sorted(aggregate.items())
    ]


def _stable_key(patient_id: str, *, seed: int, namespace: str) -> bytes:
    return hashlib.sha256(f"{namespace}|{seed}|{patient_id}".encode("utf-8")).digest()


def _allocate_counts(n: int, fractions: Sequence[float]) -> tuple[int, ...]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if not fractions or any(value < 0 for value in fractions):
        raise ValueError("fractions must be non-negative")
    total = sum(fractions)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    raw = [n * value for value in fractions]
    counts = [math.floor(value) for value in raw]
    remainder = n - sum(counts)
    order = sorted(
        range(len(raw)),
        key=lambda idx: (-(raw[idx] - counts[idx]), idx),
    )
    for idx in order[:remainder]:
        counts[idx] += 1
    return tuple(counts)


def stratified_patient_split(
    patients: Sequence[LIDCPatientSummary],
    *,
    seed: int = 20260925,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> dict[str, tuple[str, ...]]:
    """Create deterministic patient-level train/val/test sets.

    Only the held-out split is stratified, using whether a patient has any
    foreground in the mirror's documented default reference. Label-budget
    subsets inside the training pool are sampled independently of this field.
    """
    if not patients:
        raise ValueError("patients cannot be empty")
    ids = [patient.patient_id for patient in patients]
    if len(ids) != len(set(ids)):
        raise ValueError("patient summaries contain duplicate patient IDs")

    fractions = (train_fraction, val_fraction, test_fraction)
    groups: dict[bool, list[str]] = {False: [], True: []}
    for patient in patients:
        groups[patient.has_default_nodule].append(patient.patient_id)

    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    names = ("train", "val", "test")
    for has_default_nodule, patient_ids in groups.items():
        ranked = sorted(
            patient_ids,
            key=lambda patient_id: _stable_key(
                patient_id,
                seed=seed,
                namespace=f"heldout:{int(has_default_nodule)}",
            ),
        )
        counts = _allocate_counts(len(ranked), fractions)
        offset = 0
        for name, count in zip(names, counts):
            result[name].extend(ranked[offset : offset + count])
            offset += count

    frozen = {name: tuple(sorted(values)) for name, values in result.items()}
    sets = {name: set(values) for name, values in frozen.items()}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise RuntimeError("patient leakage detected while constructing split")
    if set().union(*sets.values()) != set(ids):
        raise RuntimeError("split does not cover every patient exactly once")
    return frozen


def labelled_subset_size(n_train: int, fraction: float) -> int:
    if n_train < 1:
        raise ValueError("n_train must be >= 1")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")
    # Conventional half-up rounding avoids Python's bankers rounding and is
    # explicit for benchmark manifests.
    return min(n_train, max(1, int(math.floor(n_train * fraction + 0.5))))


def nested_labelled_subsets(
    train_patient_ids: Sequence[str],
    *,
    fractions: Sequence[float] = (0.01, 0.05, 0.10, 0.25),
    seeds: Sequence[int] = (1337, 2026, 31415),
) -> dict[tuple[int, float], tuple[str, ...]]:
    """Sample nested labelled subsets without looking at patient target status."""
    patient_ids = sorted(set(train_patient_ids))
    if len(patient_ids) != len(train_patient_ids):
        raise ValueError("train_patient_ids contains duplicates")
    if not patient_ids:
        raise ValueError("train_patient_ids cannot be empty")

    ordered_fractions = tuple(sorted(set(float(value) for value in fractions)))
    if len(ordered_fractions) != len(fractions):
        raise ValueError("fractions must be unique")
    for fraction in ordered_fractions:
        labelled_subset_size(len(patient_ids), fraction)

    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")

    subsets: dict[tuple[int, float], tuple[str, ...]] = {}
    for seed in seeds:
        ranked = sorted(
            patient_ids,
            key=lambda patient_id: _stable_key(
                patient_id, seed=int(seed), namespace="label-budget"
            ),
        )
        previous: set[str] = set()
        for fraction in ordered_fractions:
            size = labelled_subset_size(len(patient_ids), fraction)
            selected = set(ranked[:size])
            if not previous.issubset(selected):
                raise RuntimeError("labelled subsets are unexpectedly non-nested")
            subsets[(int(seed), fraction)] = tuple(sorted(selected))
            previous = selected
    return subsets
