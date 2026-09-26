from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from leia_benchmark.data.lidc import (
    ConsensusPolicy,
    dicom_images_to_hu,
    make_25d_slice,
    merge_semantic_target,
    semantic_target_from_reader_masks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export LIDC-IDRI volumetric nodule annotations to a 2.5D "
            "Ultralytics semantic-segmentation dataset."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--patient-list",
        type=Path,
        default=None,
        help="Optional text file with one LIDC-IDRI patient ID per line.",
    )
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--positive-readers", type=int, default=3)
    parser.add_argument("--window-low", type=float, default=-1000.0)
    parser.add_argument("--window-high", type=float, default=400.0)
    parser.add_argument(
        "--drop-empty-slices",
        action="store_true",
        help="Debug option: omit slices with neither positive nor ambiguous nodule pixels.",
    )
    return parser.parse_args()


def _read_patient_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ids = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not ids:
        raise ValueError(f"No patient IDs found in {path}")
    return ids


def _build_scan_target(scan, shape: tuple[int, int, int], policy: ConsensusPolicy):
    from pylidc.utils import consensus

    target = np.zeros(shape, dtype=np.uint8)
    clusters = scan.cluster_annotations()

    for annotations in clusters:
        # consensus() is used here to align the individual reader masks in one
        # common local reference frame. The returned consensus mask itself is
        # intentionally ignored; our 0/1/255 policy is applied below.
        _, bbox, masks = consensus(
            annotations,
            clevel=0.5,
            ret_masks=True,
            verbose=False,
        )
        local = semantic_target_from_reader_masks(masks, policy)
        merge_semantic_target(target, local, bbox)

    return target, len(clusters)


def export_scan(
    scan,
    *,
    output: Path,
    split: str,
    policy: ConsensusPolicy,
    window_low: float,
    window_high: float,
    drop_empty_slices: bool,
) -> list[dict[str, object]]:
    images = scan.load_all_dicom_images(verbose=False)
    volume_hu = dicom_images_to_hu(images)
    target, n_clusters = _build_scan_target(scan, volume_hu.shape, policy)

    image_dir = output / "images" / split
    mask_dir = output / "masks" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for z in range(volume_hu.shape[2]):
        mask = target[:, :, z]
        positive_pixels = int(np.count_nonzero(mask == 1))
        unknown_pixels = int(np.count_nonzero(mask == 255))

        if drop_empty_slices and positive_pixels == 0 and unknown_pixels == 0:
            continue

        stem = f"{scan.patient_id}_z{z:04d}"
        rgb = make_25d_slice(
            volume_hu,
            z,
            low=window_low,
            high=window_high,
        )

        Image.fromarray(rgb, mode="RGB").save(image_dir / f"{stem}.png")
        Image.fromarray(mask, mode="L").save(mask_dir / f"{stem}.png")

        rows.append(
            {
                "patient_id": scan.patient_id,
                "series_instance_uid": scan.series_instance_uid,
                "slice_index": z,
                "image": str(Path("images") / split / f"{stem}.png"),
                "mask": str(Path("masks") / split / f"{stem}.png"),
                "positive_pixels": positive_pixels,
                "unknown_pixels": unknown_pixels,
                "nodule_clusters_in_scan": n_clusters,
            }
        )

    return rows


def main() -> None:
    args = parse_args()

    if args.window_high <= args.window_low:
        raise SystemExit("--window-high must be greater than --window-low")

    import pylidc as pl

    selected = _read_patient_ids(args.patient_list)
    policy = ConsensusPolicy(
        total_readers=4,
        positive_readers=args.positive_readers,
        ambiguous_min_readers=1,
    )

    scans = pl.query(pl.Scan).order_by(pl.Scan.patient_id).all()
    if selected is not None:
        scans = [scan for scan in scans if scan.patient_id in selected]
        found = {scan.patient_id for scan in scans}
        missing = sorted(selected - found)
        if missing:
            raise SystemExit(
                "Patient IDs not found in pylidc database: " + ", ".join(missing)
            )

    all_rows: list[dict[str, object]] = []
    for scan in scans:
        all_rows.extend(
            export_scan(
                scan,
                output=args.output,
                split=args.split,
                policy=policy,
                window_low=args.window_low,
                window_high=args.window_high,
                drop_empty_slices=args.drop_empty_slices,
            )
        )

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / f"manifest_{args.split}.csv"
    fieldnames = [
        "patient_id",
        "series_instance_uid",
        "slice_index",
        "image",
        "mask",
        "positive_pixels",
        "unknown_pixels",
        "nodule_clusters_in_scan",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(
        f"Exported {len(all_rows)} slices from {len(scans)} scans "
        f"to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
