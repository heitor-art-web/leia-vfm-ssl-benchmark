#!/usr/bin/env python3
"""Prepare LIDC-IDRI scans as 2.5D PNG + semantic-mask samples for YOLO26-sem.

This script intentionally uses pylidc instead of maintaining a custom XML parser.
TCIA recommends reviewing pylidc / standardized DICOM representations before
building custom XML tooling.

Safety properties
-----------------
- patient selection is explicit; an unscoped full-dataset export is refused;
- split name is explicit;
- files are not overwritten unless `--overwrite` is provided;
- filenames include a stable SeriesInstanceUID-derived scan key because LIDC
  contains more CT series than patients.
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
    stable_scan_key,
    stack_25d,
    window_to_uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patient-id", action="append", default=[])
    parser.add_argument(
        "--patient-list",
        type=Path,
        help="Text file containing one LIDC-IDRI patient ID per line.",
    )
    parser.add_argument(
        "--all-patients",
        action="store_true",
        help="Explicitly process every pylidc scan. Do not use this before the patient split is frozen.",
    )
    parser.add_argument("--min-cluster-readers", type=int, default=3)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--window-low", type=float, default=-1000.0)
    parser.add_argument("--window-high", type=float, default=400.0)
    parser.add_argument(
        "--signal-only",
        action="store_true",
        help="Development/smoke mode: export only slices with positive/UNKNOWN target signal. Benchmark exports should keep all slices.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        required=True,
        help="Output split. Patient assignment must already be decided upstream.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing split manifest / sample files.",
    )
    return parser.parse_args()


def _read_patient_list(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    selected = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        selected.add(value)
    return selected


def main() -> None:
    args = parse_args()

    selected = set(args.patient_id)
    if args.patient_list is not None:
        selected |= _read_patient_list(args.patient_list)

    if args.all_patients and selected:
        raise SystemExit("Use either explicit patient selection or --all-patients, not both.")
    if not args.all_patients and not selected:
        raise SystemExit(
            "Refusing unscoped export. Provide --patient-id/--patient-list, or explicitly use --all-patients after the split is frozen."
        )

    try:
        import pylidc as pl
        from pylidc.utils import consensus
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Missing optional LIDC dependencies. Install with: pip install -r requirements-lidc.txt"
        ) from exc

    policy = ConsensusPolicy(
        min_cluster_readers=args.min_cluster_readers,
        positive_fraction=args.positive_fraction,
    )

    image_dir = args.output / "images" / args.split
    mask_dir = args.output / "masks" / args.split
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output / f"manifest_{args.split}.csv"
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Manifest already exists: {manifest_path}. Use --overwrite only if replacement is intentional."
        )

    scans = pl.query(pl.Scan).all()
    if not args.all_patients:
        scans = [scan for scan in scans if scan.patient_id in selected]
        found = {scan.patient_id for scan in scans}
        missing = selected - found
        if missing:
            raise SystemExit(f"Unknown/unavailable patient ids: {sorted(missing)}")

    scans = sorted(scans, key=lambda scan: (scan.patient_id, scan.series_instance_uid))
    rows: list[dict[str, object]] = []

    for scan in scans:
        scan_key = stable_scan_key(scan.patient_id, scan.series_instance_uid)
        images = scan.load_all_dicom_images(verbose=False)
        volume = dicom_slices_to_hu(images)
        if volume.ndim != 3:
            raise RuntimeError(f"{scan_key}: unexpected volume shape {volume.shape}")

        clusters_for_target = []
        try:
            clusters = scan.cluster_annotations(verbose=False)
        except Exception as exc:
            raise RuntimeError(f"{scan_key}: annotation clustering failed") from exc

        for cluster in clusters:
            if not cluster:
                continue

            # consensus(..., ret_masks=True) provides each reader mask in a
            # common bounding box. The returned consensus mask itself is not
            # used as ground truth; our conservative 0/1/255 policy is applied.
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

            stem = f"{scan_key}_z{k:04d}"
            image_path = image_dir / f"{stem}.png"
            mask_path = mask_dir / f"{stem}.png"
            if not args.overwrite and (image_path.exists() or mask_path.exists()):
                raise RuntimeError(
                    f"Refusing to overwrite existing sample {stem}; use --overwrite only if intentional."
                )

            Image.fromarray(sample).save(image_path)
            Image.fromarray(mask).save(mask_path)

            rows.append(
                {
                    "patient_id": scan.patient_id,
                    "scan_key": scan_key,
                    "study_instance_uid": scan.study_instance_uid,
                    "series_instance_uid": scan.series_instance_uid,
                    "slice_index": k,
                    "image": str(image_path.relative_to(args.output)),
                    "mask": str(mask_path.relative_to(args.output)),
                    "nodule_pixels": int(np.sum(mask == 1)),
                    "unknown_pixels": int(np.sum(mask == 255)),
                    "cluster_count": len(clusters),
                }
            )

    fieldnames = [
        "patient_id",
        "scan_key",
        "study_instance_uid",
        "series_instance_uid",
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

    print(f"Exported {len(rows)} slices from {len(scans)} scans to {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
