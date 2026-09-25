#!/usr/bin/env python3
"""Prepare LIDC-IDRI scans as 2.5D PNG + semantic-mask samples for YOLO26-sem.

This script intentionally uses pylidc instead of maintaining a custom XML parser.
TCIA explicitly recommends pylidc / standardized representations before custom
XML tooling.

Prerequisites
-------------
1. Download the LIDC-IDRI DICOM + annotations from TCIA.
2. Configure ~/.pylidcrc so pylidc can find the DICOM root.
3. Install optional dependencies:
       pip install pylidc Pillow

Example
-------
python scripts/prepare_lidc_yolo26.py \
    --output data/lidc_yolo26 \
    --patient-id LIDC-IDRI-0078 \
    --min-cluster-readers 3 \
    --positive-fraction 0.5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from leia_benchmark.data.lidc import (
    ConsensusPolicy,
    build_scan_target,
    dicom_slices_to_hu,
    slice_has_signal,
    stack_25d,
    window_to_uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patient-id", action="append", default=[])
    parser.add_argument("--min-cluster-readers", type=int, default=3)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--window-low", type=float, default=-1000.0)
    parser.add_argument("--window-high", type=float, default=400.0)
    parser.add_argument(
        "--signal-only",
        action="store_true",
        help="Development/smoke-test mode: export only slices containing positive/UNKNOWN mask signal. Benchmark exports should normally keep all slices.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="train",
        help="Output split name only; patient assignment must be decided upstream.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import pylidc as pl
        from pylidc.utils import consensus
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Missing optional LIDC dependencies. Install with: pip install pylidc Pillow"
        ) from exc

    policy = ConsensusPolicy(
        min_cluster_readers=args.min_cluster_readers,
        positive_fraction=args.positive_fraction,
    )

    image_dir = args.output / "images" / args.split
    mask_dir = args.output / "masks" / args.split
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.patient_id)
    scans = pl.query(pl.Scan).all()
    if selected:
        scans = [scan for scan in scans if scan.patient_id in selected]
        missing = selected - {scan.patient_id for scan in scans}
        if missing:
            raise SystemExit(f"Unknown/unavailable patient ids: {sorted(missing)}")

    rows: list[dict[str, object]] = []

    for scan in scans:
        images = scan.load_all_dicom_images(verbose=False)
        volume = dicom_slices_to_hu(images)
        if volume.ndim != 3:
            raise RuntimeError(f"{scan.patient_id}: unexpected volume shape {volume.shape}")

        clusters_for_target = []
        clusters = scan.cluster_annotations(verbose=False)
        for cluster in clusters:
            if not cluster:
                continue

            # consensus(..., ret_masks=True) gives masks for all annotations in a
            # common bounding box. We use those aligned masks, but apply our own
            # conservative target policy instead of treating disagreement as 0.
            _, bbox, reader_masks = consensus(
                cluster,
                clevel=policy.positive_fraction,
                ret_masks=True,
                verbose=False,
            )
            clusters_for_target.append((reader_masks, bbox))

        target = build_scan_target(volume.shape, clusters_for_target, policy=policy)

        for k in range(volume.shape[2]):
            mask = target[:, :, k]
            if args.signal_only and not slice_has_signal(mask):
                continue

            sample = stack_25d(volume, k)
            sample = window_to_uint8(
                sample,
                lower_hu=args.window_low,
                upper_hu=args.window_high,
            )

            stem = f"{scan.patient_id}_z{k:04d}"
            Image.fromarray(sample).save(image_dir / f"{stem}.png")
            Image.fromarray(mask).save(mask_dir / f"{stem}.png")

            rows.append(
                {
                    "patient_id": scan.patient_id,
                    "slice_index": k,
                    "image": str((image_dir / f"{stem}.png").relative_to(args.output)),
                    "mask": str((mask_dir / f"{stem}.png").relative_to(args.output)),
                    "nodule_pixels": int(np.sum(mask == 1)),
                    "unknown_pixels": int(np.sum(mask == 255)),
                    "cluster_count": len(clusters),
                }
            )

    manifest_path = args.output / f"manifest_{args.split}.csv"
    fieldnames = [
        "patient_id",
        "slice_index",
        "image",
        "mask",
        "nodule_pixels",
        "unknown_pixels",
        "cluster_count",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} slices to {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
