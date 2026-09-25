#!/usr/bin/env python3
"""Audit LIDC-IDRI annotation density before freezing train/val/test splits.

The output contains scan-level and patient-level metadata only. It does not
export medical images and it does not use malignancy ratings. The purpose is to
understand reader agreement, nodule burden, and multi-scan patients before a
split policy is frozen.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from leia_benchmark.data.lidc import stable_scan_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="Scan-level CSV path.")
    parser.add_argument(
        "--patient-output",
        type=Path,
        default=None,
        help="Patient-level CSV path. Defaults to <output_stem>_patients.csv.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patient_output = args.patient_output or args.output.with_name(
        f"{args.output.stem}_patients{args.output.suffix or '.csv'}"
    )

    for path in (args.output, patient_output):
        if path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing audit file: {path}")

    try:
        import pylidc as pl
    except ImportError as exc:
        raise SystemExit(
            "Missing pylidc. Install with: pip install -r requirements-lidc.txt"
        ) from exc

    scan_rows: list[dict[str, object]] = []
    scans = sorted(
        pl.query(pl.Scan).all(),
        key=lambda scan: (scan.patient_id, scan.series_instance_uid),
    )

    for scan in scans:
        key = stable_scan_key(scan.patient_id, scan.series_instance_uid)
        try:
            clusters = scan.cluster_annotations(verbose=False)
        except Exception as exc:
            raise RuntimeError(f"{key}: annotation clustering failed") from exc
        reader_counts = [len(cluster) for cluster in clusters]

        scan_rows.append(
            {
                "patient_id": scan.patient_id,
                "scan_key": key,
                "study_instance_uid": scan.study_instance_uid,
                "series_instance_uid": scan.series_instance_uid,
                "n_clusters": len(clusters),
                "clusters_r1": sum(count == 1 for count in reader_counts),
                "clusters_r2": sum(count == 2 for count in reader_counts),
                "clusters_r3": sum(count == 3 for count in reader_counts),
                "clusters_r4": sum(count == 4 for count in reader_counts),
                "clusters_r3plus": sum(count >= 3 for count in reader_counts),
                "annotations_total": sum(reader_counts),
                "slice_thickness_mm": float(scan.slice_thickness),
                "slice_spacing_mm": float(scan.slice_spacing),
                "pixel_spacing_mm": float(scan.pixel_spacing),
            }
        )

    scan_fields = [
        "patient_id",
        "scan_key",
        "study_instance_uid",
        "series_instance_uid",
        "n_clusters",
        "clusters_r1",
        "clusters_r2",
        "clusters_r3",
        "clusters_r4",
        "clusters_r3plus",
        "annotations_total",
        "slice_thickness_mm",
        "slice_spacing_mm",
        "pixel_spacing_mm",
    ]

    patient_acc: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "n_scans": 0,
            "n_clusters": 0,
            "clusters_r1": 0,
            "clusters_r2": 0,
            "clusters_r3": 0,
            "clusters_r4": 0,
            "clusters_r3plus": 0,
            "annotations_total": 0,
        }
    )
    for row in scan_rows:
        acc = patient_acc[str(row["patient_id"])]
        acc["n_scans"] += 1
        for field in (
            "n_clusters",
            "clusters_r1",
            "clusters_r2",
            "clusters_r3",
            "clusters_r4",
            "clusters_r3plus",
            "annotations_total",
        ):
            acc[field] += int(row[field])

    patient_rows = []
    for patient_id in sorted(patient_acc):
        values = patient_acc[patient_id]
        patient_rows.append(
            {
                "patient_id": patient_id,
                **values,
                "has_r3plus": int(values["clusters_r3plus"] > 0),
                "has_any_cluster": int(values["n_clusters"] > 0),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    patient_output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scan_fields)
        writer.writeheader()
        writer.writerows(scan_rows)

    patient_fields = [
        "patient_id",
        "n_scans",
        "n_clusters",
        "clusters_r1",
        "clusters_r2",
        "clusters_r3",
        "clusters_r4",
        "clusters_r3plus",
        "annotations_total",
        "has_r3plus",
        "has_any_cluster",
    ]
    with patient_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=patient_fields)
        writer.writeheader()
        writer.writerows(patient_rows)

    multi_scan = sum(int(row["n_scans"]) > 1 for row in patient_rows)
    print(f"Wrote {len(scan_rows)} scan rows to {args.output}")
    print(f"Wrote {len(patient_rows)} patient rows to {patient_output}")
    print(f"Patients with >1 CT scan: {multi_scan}")


if __name__ == "__main__":
    main()
