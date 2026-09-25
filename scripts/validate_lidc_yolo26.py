#!/usr/bin/env python3
"""Validate an exported Ultralytics semantic-segmentation dataset.

Checks:
- every image has a matching mask and vice versa;
- image/mask dimensions match;
- masks contain only expected values {0, 1, 255};
- patient ids inferred from filenames do not overlap across train/val/test;
- per-split counts and pixel totals are reported.

This validator is intentionally independent of pylidc.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from leia_benchmark.data.lidc import patient_id_from_path


EXPECTED_MASK_VALUES = {0, 1, 255}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Split directories to validate.",
    )
    return parser.parse_args()


def _png_stems(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        return {}
    return {p.stem: p for p in directory.glob("*.png")}


def validate_split(root: Path, split: str) -> tuple[set[str], Counter]:
    images = _png_stems(root / "images" / split)
    masks = _png_stems(root / "masks" / split)

    if not images and not masks:
        return set(), Counter()

    image_only = sorted(set(images) - set(masks))
    mask_only = sorted(set(masks) - set(images))
    if image_only or mask_only:
        raise ValueError(
            f"{split}: unmatched stems; image_only={image_only[:5]}, mask_only={mask_only[:5]}"
        )

    patient_ids: set[str] = set()
    counts = Counter()

    for stem in sorted(images):
        image_path = images[stem]
        mask_path = masks[stem]

        image = np.asarray(Image.open(image_path))
        mask = np.asarray(Image.open(mask_path))

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{image_path}: expected HxWx3 RGB image, got {image.shape}")
        if mask.ndim != 2:
            raise ValueError(f"{mask_path}: expected single-channel mask, got {mask.shape}")
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"{stem}: image/mask size mismatch {image.shape[:2]} vs {mask.shape}"
            )

        values, value_counts = np.unique(mask, return_counts=True)
        unexpected = set(int(v) for v in values) - EXPECTED_MASK_VALUES
        if unexpected:
            raise ValueError(f"{mask_path}: unexpected mask values {sorted(unexpected)}")

        for value, count in zip(values, value_counts):
            counts[f"pixels_{int(value)}"] += int(count)

        pid = patient_id_from_path(stem)
        patient_ids.add(pid)
        counts["samples"] += 1

    counts["patients"] = len(patient_ids)
    return patient_ids, counts


def main() -> None:
    args = parse_args()
    root = args.dataset_root

    split_patients: dict[str, set[str]] = {}
    for split in args.splits:
        pids, counts = validate_split(root, split)
        split_patients[split] = pids
        if counts:
            print(
                f"{split}: samples={counts['samples']} patients={counts['patients']} "
                f"background={counts['pixels_0']} nodule={counts['pixels_1']} "
                f"unknown={counts['pixels_255']}"
            )
        else:
            print(f"{split}: empty / absent")

    splits = list(split_patients)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            overlap = split_patients[left] & split_patients[right]
            if overlap:
                raise ValueError(
                    f"patient leakage between {left} and {right}: {sorted(overlap)[:10]}"
                )

    print("Dataset validation passed.")


if __name__ == "__main__":
    main()
