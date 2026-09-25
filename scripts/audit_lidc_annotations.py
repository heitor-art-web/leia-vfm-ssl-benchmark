#!/usr/bin/env python3
"""Audit LIDC-IDRI annotation density before freezing train/val/test splits.

The output is patient/scan-level metadata only. It does not export medical
images and it does not use malignancy ratings. The purpose is to understand
reader agreement and nodule burden before choosing a stratification policy.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import pylidc as pl
    except ImportError as exc:
        raise SystemExit(
            "Missing pylidc. Install optional preparation dependencies first."
        ) from exc

    rows: list[dict[str, object]] = []
    scans = sorted(pl.query(pl.Scan).all(), key=lambda scan: scan.patient_id)

    for scan in scans:
        clusters = scan.cluster_annotations(verbose=False)
        reader_counts = [len(cluster) for cluster in clusters]

        row = {
            "patient_id": scan.patient_id,
            "scan_id": int(scan.id),
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
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "patient_id",
        "scan_id",
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
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} scan rows to {args.output}")


if __name__ == "__main__":
    main()
