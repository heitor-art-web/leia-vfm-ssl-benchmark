from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class LIDCScanSummary:
    """Compact scan-level metadata used to select a visual validation cohort.

    `best_*` refers to the most suspicious volumetrically annotated nodule
    cluster in the scan according to radiologist-provided malignancy scores.
    These scores are subjective likelihood ratings, not pathology-confirmed
    cancer labels.
    """

    patient_id: str
    series_instance_uid: str
    n_clusters: int
    best_cluster_index: int | None
    best_reader_count: int
    best_malignancy_median: float | None
    best_malignancy_mean: float | None
    best_malignancy_min: int | None
    best_malignancy_max: int | None
    best_diameter_mm_median: float | None
    best_volume_mm3_median: float | None

    @property
    def has_volumetric_nodule(self) -> bool:
        return self.n_clusters > 0 and self.best_cluster_index is not None

    @property
    def high_suspicion(self) -> bool:
        return (
            self.has_volumetric_nodule
            and self.best_reader_count >= 3
            and self.best_malignancy_median is not None
            and self.best_malignancy_median >= 4.0
        )

    @property
    def low_suspicion(self) -> bool:
        return (
            self.has_volumetric_nodule
            and self.best_reader_count >= 3
            and self.best_malignancy_median is not None
            and self.best_malignancy_median <= 2.0
        )


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("count must be >= 1")
    if length < count:
        raise ValueError("not enough candidates")
    if count == 1:
        return [length // 2]

    # Round positions on an inclusive linspace, then repair any duplicate
    # indices deterministically. This gives broad size coverage without
    # cherry-picking individual cases.
    raw = np.rint(np.linspace(0, length - 1, count)).astype(int).tolist()
    used: set[int] = set()
    out: list[int] = []
    for idx in raw:
        if idx not in used:
            used.add(idx)
            out.append(idx)
            continue
        for candidate in range(length):
            if candidate not in used:
                used.add(candidate)
                out.append(candidate)
                break
    return sorted(out)


def select_visual_validation_cohort(
    scans: Sequence[LIDCScanSummary] | Iterable[LIDCScanSummary],
    *,
    n_suspicious: int = 4,
) -> list[tuple[str, LIDCScanSummary]]:
    """Select one control-like case and several high-suspicion cases.

    Selection policy:
    1. Prefer a scan with no volumetrically annotated >=3 mm nodule as the
       control. If none is available, use a low-suspicion scan (median score
       <=2, >=3 readers). The role name records which fallback was used.
    2. High-suspicion cases require median malignancy >=4 and >=3 readers.
    3. Suspicious cases are spread across the observed nodule-diameter range
       rather than hand-picked.

    The LIDC malignancy score is a radiologist likelihood rating and must not
    be relabelled as pathology-confirmed cancer.
    """
    items = sorted(list(scans), key=lambda x: x.patient_id)
    if not items:
        raise ValueError("no scan summaries supplied")
    if n_suspicious < 1:
        raise ValueError("n_suspicious must be >= 1")

    no_nodule = [scan for scan in items if not scan.has_volumetric_nodule]
    if no_nodule:
        control = no_nodule[0]
        control_role = "control_no_volumetric_nodule"
    else:
        low = [scan for scan in items if scan.low_suspicion]
        if not low:
            raise ValueError(
                "no control candidate: need a no-nodule scan or a low-suspicion scan"
            )
        control = sorted(
            low,
            key=lambda x: (
                x.best_malignancy_median if x.best_malignancy_median is not None else 9,
                x.patient_id,
            ),
        )[0]
        control_role = "control_low_suspicion"

    suspicious = [
        scan for scan in items if scan.high_suspicion and scan.patient_id != control.patient_id
    ]
    if len(suspicious) < n_suspicious:
        raise ValueError(
            f"need {n_suspicious} high-suspicion scans, found {len(suspicious)}"
        )

    suspicious.sort(
        key=lambda x: (
            x.best_diameter_mm_median
            if x.best_diameter_mm_median is not None
            else float("inf"),
            x.patient_id,
        )
    )
    selected = [suspicious[i] for i in _evenly_spaced_indices(len(suspicious), n_suspicious)]

    return [(control_role, control)] + [("high_suspicion", scan) for scan in selected]


def choose_lung_like_slice(volume_hu: np.ndarray) -> int:
    """Choose a representative axial slice for a scan without a target nodule.

    The score is the number of lung-like voxels (-950..-400 HU) in the central
    60% of axial slices. This is only for visual QC; it is never used for
    training labels or evaluation.
    """
    volume = np.asarray(volume_hu)
    if volume.ndim != 3 or volume.shape[2] < 1:
        raise ValueError("volume_hu must have shape (H, W, Z) with Z >= 1")

    z_count = volume.shape[2]
    lo = int(np.floor(0.2 * z_count))
    hi = int(np.ceil(0.8 * z_count))
    hi = max(lo + 1, min(z_count, hi))

    candidates = range(lo, hi)
    scores = [
        int(np.count_nonzero((volume[:, :, z] >= -950.0) & (volume[:, :, z] <= -400.0)))
        for z in candidates
    ]
    return lo + int(np.argmax(scores))


def choose_max_positive_slice(mask: np.ndarray, *, positive_label: int = 1) -> int:
    """Return the axial slice containing the most trusted positive pixels."""
    arr = np.asarray(mask)
    if arr.ndim != 3 or arr.shape[2] < 1:
        raise ValueError("mask must have shape (H, W, Z) with Z >= 1")
    areas = np.count_nonzero(arr == positive_label, axis=(0, 1))
    if int(areas.max()) == 0:
        raise ValueError("mask contains no positive pixels")
    return int(np.argmax(areas))
