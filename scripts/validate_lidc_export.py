from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

ALLOWED_MASK_VALUES = {0, 1, 255}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an exported LIDC semantic dataset.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=("train", "val", "test"),
    )
    return parser.parse_args()


def _read_manifest(dataset: Path, split: str) -> list[dict[str, str]]:
    path = dataset / f"manifest_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_split(dataset: Path, split: str) -> set[str]:
    rows = _read_manifest(dataset, split)
    if not rows:
        raise ValueError(f"{split}: manifest is empty")

    patient_ids: set[str] = set()
    seen_images: set[str] = set()

    for row in rows:
        patient_id = row["patient_id"]
        image_rel = row["image"]
        mask_rel = row["mask"]
        image_path = dataset / image_rel
        mask_path = dataset / mask_rel

        if image_rel in seen_images:
            raise ValueError(f"{split}: duplicate image in manifest: {image_rel}")
        seen_images.add(image_rel)
        patient_ids.add(patient_id)

        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        if image_path.stem != mask_path.stem:
            raise ValueError(f"{split}: image/mask stems differ: {image_rel}, {mask_rel}")

        with Image.open(image_path) as image:
            image_size = image.size
            if image.mode != "RGB":
                raise ValueError(f"{image_rel}: expected RGB image, got {image.mode}")

        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image)
            mask_size = mask_image.size

        if image_size != mask_size:
            raise ValueError(
                f"{split}: image/mask size mismatch for {image_path.stem}: "
                f"{image_size} vs {mask_size}"
            )

        values = set(np.unique(mask).tolist())
        invalid = values - ALLOWED_MASK_VALUES
        if invalid:
            raise ValueError(f"{mask_rel}: invalid mask values {sorted(invalid)}")

    return patient_ids


def validate_dataset(dataset: Path, splits: list[str]) -> None:
    split_patients = {split: validate_split(dataset, split) for split in splits}

    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            overlap = split_patients[left] & split_patients[right]
            if overlap:
                sample = ", ".join(sorted(overlap)[:10])
                raise ValueError(
                    f"Patient leakage between {left} and {right}: {sample}"
                )


def main() -> None:
    args = parse_args()
    validate_dataset(args.dataset, args.splits)
    print("LIDC export validation passed.")


if __name__ == "__main__":
    main()
