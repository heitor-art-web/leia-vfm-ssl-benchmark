#!/usr/bin/env python3
"""Materialize a run-specific LIDC semantic-dataset view without copying images.

The exported LIDC dataset contains masks for every frozen split so evaluation is
possible, but a low-label supervised run must only *see* masks from its declared
labelled-patient subset. This script creates image-list files and a dedicated
Ultralytics YAML for that subset.

No images or masks are copied. Paths are absolute in the generated list files,
which avoids ambiguity in Ultralytics path resolution.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("labelled_patients", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    ids: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        if value in ids:
            raise ValueError(f"duplicate patient id in {path}: {value}")
        ids.add(value)
    if not ids:
        raise ValueError(f"no patient ids in {path}")
    return ids


def _read_manifest(dataset_root: Path, split: str) -> list[dict[str, str]]:
    path = dataset_root / f"manifest_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"patient_id", "image", "mask"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if not rows:
        raise ValueError(f"{path}: empty manifest")
    return rows


def _image_paths(
    dataset_root: Path,
    rows: list[dict[str, str]],
    *,
    patient_ids: set[str] | None = None,
) -> list[Path]:
    paths = []
    for row in rows:
        if patient_ids is not None and row["patient_id"] not in patient_ids:
            continue
        image = (dataset_root / row["image"]).resolve()
        mask = (dataset_root / row["mask"]).resolve()
        if not image.exists():
            raise FileNotFoundError(image)
        if not mask.exists():
            raise FileNotFoundError(mask)
        paths.append(image)
    return sorted(paths)


def _write_list(path: Path, items: list[Path]) -> None:
    path.write_text("".join(f"{item}\n" for item in items), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    labelled = _read_ids(args.labelled_patients)

    manifests = {
        split: _read_manifest(dataset_root, split)
        for split in ("train", "val", "test")
    }
    train_patients = {row["patient_id"] for row in manifests["train"]}
    missing_labelled = labelled - train_patients
    if missing_labelled:
        raise ValueError(
            "labelled-patient list contains patients absent from train manifest: "
            f"{sorted(missing_labelled)}"
        )

    unlabelled = train_patients - labelled
    labelled_paths = _image_paths(dataset_root, manifests["train"], patient_ids=labelled)
    unlabelled_paths = _image_paths(dataset_root, manifests["train"], patient_ids=unlabelled)
    val_paths = _image_paths(dataset_root, manifests["val"])
    test_paths = _image_paths(dataset_root, manifests["test"])

    if not labelled_paths:
        raise ValueError("labelled subset contains no exported training images")
    if not val_paths:
        raise ValueError("validation split contains no exported images")
    if not test_paths:
        raise ValueError("test split contains no exported images")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        "labelled": args.output_dir / "train_labelled.txt",
        "unlabelled": args.output_dir / "train_unlabelled.txt",
        "val": args.output_dir / "val.txt",
        "test": args.output_dir / "test.txt",
        "yaml": args.output_dir / "supervised.yaml",
        "manifest": args.output_dir / "view.json",
    }
    existing = [path for path in output_files.values() if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "Refusing to overwrite dataset-view files: "
            + ", ".join(str(path) for path in existing)
        )

    _write_list(output_files["labelled"], labelled_paths)
    _write_list(output_files["unlabelled"], unlabelled_paths)
    _write_list(output_files["val"], val_paths)
    _write_list(output_files["test"], test_paths)

    data_yaml = {
        "path": str(dataset_root),
        "train": str(output_files["labelled"].resolve()),
        "val": str(output_files["val"].resolve()),
        "test": str(output_files["test"].resolve()),
        "masks_dir": "masks",
        "names": {0: "background", 1: "pulmonary_nodule"},
    }
    output_files["yaml"].write_text(
        yaml.safe_dump(data_yaml, sort_keys=False),
        encoding="utf-8",
    )

    payload = {
        "dataset_root": str(dataset_root),
        "source_labelled_patients": str(args.labelled_patients.resolve()),
        "labelled_patients": sorted(labelled),
        "unlabelled_patients": sorted(unlabelled),
        "counts": {
            "train_patients_total": len(train_patients),
            "labelled_patients": len(labelled),
            "unlabelled_patients": len(unlabelled),
            "labelled_slices": len(labelled_paths),
            "unlabelled_slices": len(unlabelled_paths),
            "val_slices": len(val_paths),
            "test_slices": len(test_paths),
        },
        "supervised_yaml": str(output_files["yaml"].resolve()),
        "unlabelled_image_list": str(output_files["unlabelled"].resolve()),
    }
    output_files["manifest"].write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(payload["counts"], indent=2, sort_keys=True))
    print(f"Supervised dataset YAML: {output_files['yaml']}")
    print(f"Unlabelled image list for MT: {output_files['unlabelled']}")


if __name__ == "__main__":
    main()
