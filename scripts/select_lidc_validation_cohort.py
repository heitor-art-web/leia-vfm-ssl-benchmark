from __future__ import annotations

import argparse
import csv
from pathlib import Path

from leia_benchmark.data.lidc_validation import (
    LIDCScanSummary,
    select_visual_validation_cohort,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a deterministic 5-case LIDC visual QC cohort: one control-like "
            "scan and four high-suspicion scans spread across nodule sizes."
        )
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-suspicious", type=int, default=4)
    return parser.parse_args()


def _optional_float(value: str) -> float | None:
    value = value.strip()
    return None if value == "" else float(value)


def _optional_int(value: str) -> int | None:
    value = value.strip()
    return None if value == "" else int(value)


def _annotation_ids(value: str) -> tuple[int, ...]:
    value = value.strip()
    if not value:
        return ()
    return tuple(sorted(int(part) for part in value.split(";") if part))


def _load(path: Path) -> list[LIDCScanSummary]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")

    scans: list[LIDCScanSummary] = []
    for row in rows:
        scans.append(
            LIDCScanSummary(
                patient_id=row["patient_id"],
                series_instance_uid=row["series_instance_uid"],
                n_clusters=int(row["n_clusters"]),
                best_cluster_index=_optional_int(row["best_cluster_index"]),
                best_annotation_ids=_annotation_ids(row.get("best_annotation_ids", "")),
                best_reader_count=int(row["best_reader_count"]),
                best_malignancy_median=_optional_float(row["best_malignancy_median"]),
                best_malignancy_mean=_optional_float(row["best_malignancy_mean"]),
                best_malignancy_min=_optional_int(row["best_malignancy_min"]),
                best_malignancy_max=_optional_int(row["best_malignancy_max"]),
                best_diameter_mm_median=_optional_float(row["best_diameter_mm_median"]),
                best_volume_mm3_median=_optional_float(row["best_volume_mm3_median"]),
            )
        )
    return scans


def main() -> None:
    args = parse_args()
    cohort = select_visual_validation_cohort(
        _load(args.metadata), n_suspicious=args.n_suspicious
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "role",
        "patient_id",
        "series_instance_uid",
        "cluster_index",
        "annotation_ids",
        "reader_count",
        "malignancy_median",
        "malignancy_mean",
        "malignancy_min",
        "malignancy_max",
        "diameter_mm_median",
        "volume_mm3_median",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for role, scan in cohort:
            writer.writerow(
                {
                    "role": role,
                    "patient_id": scan.patient_id,
                    "series_instance_uid": scan.series_instance_uid,
                    "cluster_index": ""
                    if scan.best_cluster_index is None
                    else scan.best_cluster_index,
                    "annotation_ids": ";".join(str(value) for value in scan.best_annotation_ids),
                    "reader_count": scan.best_reader_count,
                    "malignancy_median": ""
                    if scan.best_malignancy_median is None
                    else scan.best_malignancy_median,
                    "malignancy_mean": ""
                    if scan.best_malignancy_mean is None
                    else scan.best_malignancy_mean,
                    "malignancy_min": ""
                    if scan.best_malignancy_min is None
                    else scan.best_malignancy_min,
                    "malignancy_max": ""
                    if scan.best_malignancy_max is None
                    else scan.best_malignancy_max,
                    "diameter_mm_median": ""
                    if scan.best_diameter_mm_median is None
                    else scan.best_diameter_mm_median,
                    "volume_mm3_median": ""
                    if scan.best_volume_mm3_median is None
                    else scan.best_volume_mm3_median,
                }
            )

    print(f"Wrote {len(cohort)} validation cases to {args.out.resolve()}")
    print(
        "Roles describe radiologist annotation/risk strata only. "
        "high_suspicion is not a pathology-confirmed cancer label."
    )


if __name__ == "__main__":
    main()
