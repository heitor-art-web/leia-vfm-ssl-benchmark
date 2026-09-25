from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize LIDC-IDRI scans and radiologist nodule ratings from the "
            "local pylidc database."
        )
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _cluster_summary(cluster) -> dict[str, object]:
    malignancies = [int(ann.malignancy) for ann in cluster]
    diameters = [float(ann.diameter) for ann in cluster]
    volumes = [float(ann.volume) for ann in cluster]
    return {
        "reader_count": len(cluster),
        "malignancy_median": float(median(malignancies)),
        "malignancy_mean": float(mean(malignancies)),
        "malignancy_min": min(malignancies),
        "malignancy_max": max(malignancies),
        "diameter_mm_median": float(median(diameters)),
        "volume_mm3_median": float(median(volumes)),
    }


def summarize_scan(scan) -> dict[str, object]:
    clusters = scan.cluster_annotations()
    if not clusters:
        return {
            "patient_id": scan.patient_id,
            "series_instance_uid": scan.series_instance_uid,
            "n_clusters": 0,
            "best_cluster_index": "",
            "best_reader_count": 0,
            "best_malignancy_median": "",
            "best_malignancy_mean": "",
            "best_malignancy_min": "",
            "best_malignancy_max": "",
            "best_diameter_mm_median": "",
            "best_volume_mm3_median": "",
        }

    summaries = [_cluster_summary(cluster) for cluster in clusters]
    best_index = max(
        range(len(summaries)),
        key=lambda idx: (
            summaries[idx]["malignancy_median"],
            summaries[idx]["reader_count"],
            summaries[idx]["malignancy_mean"],
            summaries[idx]["diameter_mm_median"],
            -idx,
        ),
    )
    best = summaries[best_index]
    return {
        "patient_id": scan.patient_id,
        "series_instance_uid": scan.series_instance_uid,
        "n_clusters": len(clusters),
        "best_cluster_index": best_index,
        "best_reader_count": best["reader_count"],
        "best_malignancy_median": best["malignancy_median"],
        "best_malignancy_mean": best["malignancy_mean"],
        "best_malignancy_min": best["malignancy_min"],
        "best_malignancy_max": best["malignancy_max"],
        "best_diameter_mm_median": best["diameter_mm_median"],
        "best_volume_mm3_median": best["volume_mm3_median"],
    }


def main() -> None:
    args = parse_args()
    import pylidc as pl

    scans = pl.query(pl.Scan).order_by(pl.Scan.patient_id).all()
    if not scans:
        raise SystemExit("No scans found in the local pylidc database")

    rows = [summarize_scan(scan) for scan in scans]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} scan summaries to {args.out.resolve()}")
    print(
        "Reminder: LIDC malignancy 1-5 is a subjective radiologist rating, "
        "not a pathology-confirmed cancer label."
    )


if __name__ == "__main__":
    main()
