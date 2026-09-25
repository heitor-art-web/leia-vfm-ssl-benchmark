from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
import yaml

from leia_benchmark.data.lidc import make_25d_slice
from leia_benchmark.data.lidc_mirror import (
    semantic_target_from_default_and_annotation_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a deliberately tiny MedOtter LIDC cohort to the Ultralytics "
            "semantic PNG-mask format for an interface smoke test. This is NOT the "
            "benchmark ground-truth export: the scientific benchmark must use the "
            "canonical individual-reader contour path."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--val-patient",
        action="append",
        default=[],
        help="Patient ID assigned to smoke validation. Repeat for multiple patients.",
    )
    parser.add_argument("--max-evidence-slices", type=int, default=12)
    parser.add_argument("--negative-slices-per-case", type=int, default=2)
    parser.add_argument("--window-low", type=float, default=-1000.0)
    parser.add_argument("--window-high", type=float, default=400.0)
    return parser.parse_args()


def _load(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.asarray(nib.load(str(path)).dataobj)


def _subsample(indices: np.ndarray, limit: int) -> list[int]:
    values = [int(v) for v in indices]
    if limit <= 0 or len(values) <= limit:
        return values
    positions = np.linspace(0, len(values) - 1, num=limit)
    return sorted({values[int(round(pos))] for pos in positions})


def _negative_indices(target: np.ndarray, count: int) -> list[int]:
    if count <= 0:
        return []
    empty = np.where(np.count_nonzero(target != 0, axis=(0, 1)) == 0)[0]
    if empty.size == 0:
        return []
    if empty.size <= count:
        return [int(v) for v in empty]
    positions = np.linspace(0, empty.size - 1, num=count + 2)[1:-1]
    return sorted({int(empty[int(round(pos))]) for pos in positions})


def main() -> None:
    args = parse_args()
    if args.window_high <= args.window_low:
        raise SystemExit("--window-high must be greater than --window-low")
    if args.max_evidence_slices < 1:
        raise SystemExit("--max-evidence-slices must be >= 1")
    if args.negative_slices_per_case < 0:
        raise SystemExit("--negative-slices-per-case must be >= 0")

    cohort_path = args.root / "cohort.csv"
    with cohort_path.open(newline="", encoding="utf-8") as handle:
        cohort = list(csv.DictReader(handle))
    if len(cohort) < 3:
        raise RuntimeError("smoke cohort must contain at least three cases")

    val_patients = set(args.val_patient)
    if not val_patients:
        # Deterministic fallback only for local convenience. CI passes explicit IDs.
        val_patients = {row["patient_id"] for row in sorted(cohort, key=lambda r: r["patient_id"])[-2:]}

    known_patients = {row["patient_id"] for row in cohort}
    unknown_val = val_patients - known_patients
    if unknown_val:
        raise RuntimeError(f"unknown --val-patient values: {sorted(unknown_val)}")
    if val_patients == known_patients:
        raise RuntimeError("smoke split would leave no training patients")

    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "masks" / split).mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, object]] = []
    for row in cohort:
        case_id = row["case_id"]
        patient_id = row["patient_id"]
        split = "val" if patient_id in val_patients else "train"

        volume_hu = _load(args.root / "images" / f"{case_id}.nii.gz").astype(np.float32)
        default_mask = _load(args.root / "masks" / f"{case_id}.nii.gz")
        annotation_count = _load(
            args.root / "masks_annotation_count" / f"{case_id}.nii.gz"
        )
        if volume_hu.shape != default_mask.shape or volume_hu.shape != annotation_count.shape:
            raise RuntimeError(f"geometry mismatch for {case_id}")
        if volume_hu.ndim != 3:
            raise RuntimeError(f"expected HxWxZ volume for {case_id}, got {volume_hu.shape}")

        target = semantic_target_from_default_and_annotation_count(
            default_mask, annotation_count
        )
        evidence = np.where(np.count_nonzero(target != 0, axis=(0, 1)) > 0)[0]
        selected = set(_subsample(evidence, args.max_evidence_slices))
        selected.update(_negative_indices(target, args.negative_slices_per_case))
        if not selected:
            raise RuntimeError(f"no smoke slices selected for {case_id}")

        for z in sorted(selected):
            stem = f"{patient_id}__z{z:04d}"
            image = make_25d_slice(
                volume_hu,
                z,
                low=args.window_low,
                high=args.window_high,
            )
            mask = target[:, :, z].astype(np.uint8, copy=False)
            unique = set(int(v) for v in np.unique(mask))
            if not unique.issubset({0, 1, 255}):
                raise RuntimeError(f"unexpected mask values for {case_id} z={z}: {sorted(unique)}")

            image_path = args.output / "images" / split / f"{stem}.png"
            mask_path = args.output / "masks" / split / f"{stem}.png"
            Image.fromarray(image, mode="RGB").save(image_path)
            Image.fromarray(mask, mode="L").save(mask_path)
            manifest.append(
                {
                    "split": split,
                    "patient_id": patient_id,
                    "case_id": case_id,
                    "slice_index": z,
                    "trusted_pixels": int(np.count_nonzero(mask == 1)),
                    "unknown_pixels": int(np.count_nonzero(mask == 255)),
                    "image": str(image_path.relative_to(args.output)),
                    "mask": str(mask_path.relative_to(args.output)),
                }
            )

    if not any(row["split"] == "train" for row in manifest):
        raise RuntimeError("smoke export contains no training images")
    if not any(row["split"] == "val" for row in manifest):
        raise RuntimeError("smoke export contains no validation images")
    if not any(int(row["trusted_pixels"]) > 0 for row in manifest if row["split"] == "train"):
        raise RuntimeError("smoke training split contains no trusted foreground")
    if not any(int(row["trusted_pixels"]) > 0 for row in manifest if row["split"] == "val"):
        raise RuntimeError("smoke validation split contains no trusted foreground")
    if not any(int(row["unknown_pixels"]) > 0 for row in manifest):
        raise RuntimeError("smoke export does not exercise ignore label 255")

    manifest_path = args.output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    dataset = {
        "path": str(args.output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "masks_dir": "masks",
        "names": {0: "background", 1: "pulmonary_nodule_ge_3mm"},
    }
    dataset_path = args.output / "dataset.yaml"
    dataset_path.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")

    train_n = sum(row["split"] == "train" for row in manifest)
    val_n = sum(row["split"] == "val" for row in manifest)
    print(f"Smoke export complete: train={train_n} slices, val={val_n} slices")
    print(f"Dataset YAML: {dataset_path.resolve()}")
    print("NOTE: mirror-derived smoke targets are interface validation only, not benchmark GT.")


if __name__ == "__main__":
    main()
