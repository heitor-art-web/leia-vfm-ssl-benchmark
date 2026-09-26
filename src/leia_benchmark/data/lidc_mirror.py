from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from leia_benchmark.data.lidc_validation import LIDCScanSummary


@dataclass(frozen=True)
class LIDCAmbiguousCase:
    """One scan selected to exercise the semantic UNKNOWN/ignore path."""

    role: str
    case_id: str
    patient_id: str
    series_instance_uid: str
    nodule_index: int
    annotation_count: int
    malignancy_score: float | None
    diameter_mm: float | None


def _first_value(
    row: Mapping[str, str], names: Iterable[str], *, required: bool = True
) -> str | None:
    lowered = {key.lower(): key for key in row}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            value = str(row[key]).strip()
            if value != "":
                return value
    if required:
        raise KeyError(f"None of the expected columns are present/populated: {list(names)}")
    return None


def _as_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def _as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def semantic_target_from_default_and_annotation_count(
    default_mask: np.ndarray,
    annotation_count: np.ndarray,
) -> np.ndarray:
    """Build a 0/1/255 semantic target from MedOtter masks.

    1   = trusted default-reference foreground
    255 = some contour evidence exists but the voxel is outside the trusted
          default reference
    0   = no volumetric contour evidence at the voxel

    This target is appropriate for the full semantic segmentation task. It is
    deliberately conservative: excluded contour evidence is UNKNOWN, not
    forced to background.
    """
    mask = np.asarray(default_mask)
    count = np.asarray(annotation_count)
    if mask.shape != count.shape:
        raise ValueError(
            f"default_mask and annotation_count shapes differ: {mask.shape} vs {count.shape}"
        )
    if mask.ndim != 3:
        raise ValueError("default_mask and annotation_count must be 3D")
    if np.any(count < 0):
        raise ValueError("annotation_count cannot contain negative values")

    trusted = mask > 0
    evidence = count > 0
    target = np.zeros(mask.shape, dtype=np.uint8)
    target[evidence & ~trusted] = 255
    target[trusted] = 1
    return target


def selected_instance_target(
    instance_mask: np.ndarray,
    nodule_index: int,
) -> np.ndarray:
    """Return a binary target containing exactly one LIDC nodule instance.

    The MedOtter mirror documents ``masks_instance`` as a per-scan nodule-id
    volume. Visual QC of a malignancy-selected nodule must therefore isolate
    that id instead of displaying the whole scan-level semantic mask, which may
    contain larger unrelated nodules.
    """
    instances = np.asarray(instance_mask)
    if instances.ndim != 3:
        raise ValueError("instance_mask must be 3D")
    if nodule_index < 1:
        raise ValueError("nodule_index must be >= 1")
    if not np.all(np.isfinite(instances)):
        raise ValueError("instance_mask contains non-finite values")

    selected = instances == nodule_index
    if not np.any(selected):
        available = sorted(int(v) for v in np.unique(instances) if int(v) > 0)
        raise ValueError(
            f"nodule_index {nodule_index} is absent from instance_mask; "
            f"available ids: {available}"
        )
    return selected.astype(np.uint8)


def medotter_case_map(scan_rows: Iterable[Mapping[str, str]]) -> dict[tuple[str, str], str]:
    """Return (patient_id, series_uid) -> case_id for the MedOtter mirror."""
    mapping: dict[tuple[str, str], str] = {}
    for row in scan_rows:
        patient_id = _first_value(row, ["patient_id"])
        series_uid = _first_value(row, ["series_uid", "series_instance_uid"])
        case_id = _first_value(row, ["case_id"])
        assert patient_id is not None and series_uid is not None and case_id is not None
        key = (patient_id, series_uid)
        if key in mapping and mapping[key] != case_id:
            raise ValueError(f"Conflicting case_id values for {key}")
        mapping[key] = case_id
    return mapping


def select_ambiguous_validation_cases(
    scan_rows: Iterable[Mapping[str, str]],
    nodule_rows: Iterable[Mapping[str, str]],
    *,
    exclude_patient_ids: Iterable[str] = (),
) -> list[LIDCAmbiguousCase]:
    """Select clean one-reader and two-reader UNKNOWN examples for visual QC.

    Each selected scan must contain exactly one volumetrically annotated nodule,
    no nodule in the documented default ground truth, and exactly one metadata
    row for that nodule. These constraints make the scan-wide annotation-count
    mask interpretable for QC: its contour evidence cannot belong to a second
    volumetric nodule in the same scan.

    Candidate selection is deterministic and deliberately uses the middle of
    the observed diameter ordering instead of manually choosing visually easy
    examples. Existing QC patients can be excluded to keep the cohort unique.
    """
    scans = list(scan_rows)
    nodules = list(nodule_rows)
    excluded = {str(value) for value in exclude_patient_ids}

    nodules_by_case: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in nodules:
        case_id = _first_value(row, ["case_id"])
        assert case_id is not None
        nodules_by_case[case_id].append(row)

    by_reader_count: dict[int, list[LIDCAmbiguousCase]] = {1: [], 2: []}
    for scan in scans:
        case_id = _first_value(scan, ["case_id"])
        patient_id = _first_value(scan, ["patient_id"])
        series_uid = _first_value(scan, ["series_uid", "series_instance_uid"])
        assert case_id is not None and patient_id is not None and series_uid is not None
        if patient_id in excluded:
            continue

        n_nodules = _as_int(
            _first_value(scan, ["n_nodules", "n_clusters"], required=False),
            default=0,
        )
        n_default = _as_int(
            _first_value(scan, ["n_nodules_default"], required=False),
            default=0,
        )
        if n_nodules != 1 or n_default != 0:
            continue

        case_nodules = nodules_by_case.get(case_id, [])
        if len(case_nodules) != 1:
            continue
        row = case_nodules[0]
        annotation_count = _as_int(
            _first_value(row, ["n_annotations", "annotation_count"], required=False),
            default=0,
        )
        if annotation_count not in by_reader_count:
            continue
        if _as_bool(
            _first_value(row, ["in_default_gt", "default_gt"], required=False),
            default=False,
        ):
            continue

        nodule_index = _as_int(
            _first_value(
                row,
                ["nodule_id", "nodule_index", "cluster_id", "nodule_number"],
                required=False,
            ),
            default=1,
        )
        if nodule_index < 1:
            continue
        diameter = _as_float(
            _first_value(
                row,
                ["diameter_mm", "diameter_mm_median", "diameter", "median_diameter"],
                required=False,
            )
        )
        malignancy = _as_float(
            _first_value(
                row,
                ["malignancy_score", "malignancy", "malignancy_median"],
                required=False,
            )
        )
        by_reader_count[annotation_count].append(
            LIDCAmbiguousCase(
                role=f"ambiguous_{annotation_count}_reader",
                case_id=case_id,
                patient_id=patient_id,
                series_instance_uid=series_uid,
                nodule_index=nodule_index,
                annotation_count=annotation_count,
                malignancy_score=malignancy,
                diameter_mm=diameter,
            )
        )

    selected: list[LIDCAmbiguousCase] = []
    used_patients = set(excluded)
    for reader_count in (1, 2):
        candidates = [
            item
            for item in by_reader_count[reader_count]
            if item.patient_id not in used_patients
        ]
        if not candidates:
            raise ValueError(
                f"no clean {reader_count}-reader ambiguous QC candidate found"
            )
        candidates.sort(
            key=lambda item: (
                item.diameter_mm if item.diameter_mm is not None else float("inf"),
                item.patient_id,
                item.case_id,
            )
        )
        choice = candidates[(len(candidates) - 1) // 2]
        selected.append(choice)
        used_patients.add(choice.patient_id)

    return selected


def medotter_scan_summaries(
    scan_rows: Iterable[Mapping[str, str]],
    nodule_rows: Iterable[Mapping[str, str]],
) -> list[LIDCScanSummary]:
    """Convert MedOtter scans.csv/nodules.csv into benchmark scan summaries.

    The current mirror exposes the median SR characteristics with names such as
    ``malignancy_score``, ``diameter_mm`` and ``volume_mm3``. A few aliases are
    accepted for backwards compatibility. The function fails loudly when a
    scan reports nodules but no matching nodule metadata are available.
    """
    scans = list(scan_rows)
    nodules = list(nodule_rows)

    nodules_by_case: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in nodules:
        case_id = _first_value(row, ["case_id"])
        assert case_id is not None
        nodules_by_case[case_id].append(row)

    summaries: list[LIDCScanSummary] = []
    for scan_row in scans:
        case_id = _first_value(scan_row, ["case_id"])
        patient_id = _first_value(scan_row, ["patient_id"])
        series_uid = _first_value(scan_row, ["series_uid", "series_instance_uid"])
        n_clusters = _as_int(
            _first_value(scan_row, ["n_nodules", "n_clusters"], required=False),
            default=0,
        )
        assert case_id is not None and patient_id is not None and series_uid is not None

        case_nodules = nodules_by_case.get(case_id, [])
        if n_clusters > 0 and not case_nodules:
            raise ValueError(
                f"{case_id} reports {n_clusters} nodules but nodules.csv has no matching rows"
            )

        if not case_nodules:
            summaries.append(
                LIDCScanSummary(
                    patient_id=patient_id,
                    series_instance_uid=series_uid,
                    n_clusters=0,
                    best_cluster_index=None,
                    best_annotation_ids=(),
                    best_annotation_count=0,
                    best_malignancy_median=None,
                    best_malignancy_mean=None,
                    best_malignancy_min=None,
                    best_malignancy_max=None,
                    best_diameter_mm_median=None,
                    best_volume_mm3_median=None,
                )
            )
            continue

        parsed: list[dict[str, object]] = []
        for fallback_index, row in enumerate(case_nodules):
            in_default = _as_bool(
                _first_value(row, ["in_default_gt", "default_gt"], required=False),
                default=True,
            )
            annotation_count = _as_int(
                _first_value(row, ["n_annotations", "annotation_count"], required=False),
                default=0,
            )
            malignancy = _as_float(
                _first_value(
                    row,
                    [
                        "malignancy_score",
                        "malignancy",
                        "malignancy_median",
                        "median_malignancy",
                    ],
                    required=False,
                )
            )
            diameter = _as_float(
                _first_value(
                    row,
                    ["diameter_mm", "diameter_mm_median", "diameter", "median_diameter"],
                    required=False,
                )
            )
            volume = _as_float(
                _first_value(
                    row,
                    ["volume_mm3", "volume_mm3_median", "volume", "median_volume"],
                    required=False,
                )
            )
            nodule_index = _as_int(
                _first_value(
                    row,
                    ["nodule_id", "nodule_index", "cluster_id", "nodule_number"],
                    required=False,
                ),
                default=fallback_index,
            )
            parsed.append(
                {
                    "index": nodule_index,
                    "in_default": in_default,
                    "annotation_count": annotation_count,
                    "malignancy": malignancy,
                    "diameter": diameter,
                    "volume": volume,
                }
            )

        def key(item: dict[str, object]) -> tuple:
            malignancy = item["malignancy"]
            diameter = item["diameter"]
            return (
                bool(item["in_default"]),
                int(item["annotation_count"]) >= 3,
                float(malignancy) if malignancy is not None else -1.0,
                int(item["annotation_count"]),
                float(diameter) if diameter is not None else -1.0,
                -int(item["index"]),
            )

        best = max(parsed, key=key)
        malignancy = best["malignancy"]
        summaries.append(
            LIDCScanSummary(
                patient_id=patient_id,
                series_instance_uid=series_uid,
                n_clusters=max(n_clusters, len(case_nodules)),
                best_cluster_index=int(best["index"]),
                best_annotation_ids=(),
                best_annotation_count=int(best["annotation_count"]),
                best_malignancy_median=None if malignancy is None else float(malignancy),
                best_malignancy_mean=None if malignancy is None else float(malignancy),
                best_malignancy_min=None if malignancy is None else int(round(float(malignancy))),
                best_malignancy_max=None if malignancy is None else int(round(float(malignancy))),
                best_diameter_mm_median=None
                if best["diameter"] is None
                else float(best["diameter"]),
                best_volume_mm3_median=None
                if best["volume"] is None
                else float(best["volume"]),
            )
        )

    return summaries
