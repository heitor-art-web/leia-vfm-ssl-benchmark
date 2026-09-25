#!/usr/bin/env python3
"""Plan a frozen patient-level LIDC train/val/test partition from the audit CSV.

The v1 stratification variable is deliberately coarse: whether the patient has
at least one nodule cluster supported by >=3 readers. This avoids creating many
small strata before the real audit distribution has been inspected.

This script plans the split; it does not export any images.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from leia_benchmark.splits import make_stratified_patient_partition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("patient_audit_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_mapping(path: Path) -> dict[str, int]:
    if not path.exists():
        raise FileNotFoundError(path)
    mapping: dict[str, int] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"patient_id", "has_r3plus"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"patient audit missing required columns: {sorted(missing)}")
        for row in reader:
            patient_id = str(row["patient_id"]).strip()
            if not patient_id:
                raise ValueError("empty patient_id in patient audit")
            if patient_id in mapping:
                raise ValueError(f"duplicate patient_id in patient audit: {patient_id}")
            value = int(row["has_r3plus"])
            if value not in (0, 1):
                raise ValueError(f"has_r3plus must be 0/1 for {patient_id}")
            mapping[patient_id] = value
    if not mapping:
        raise ValueError("patient audit is empty")
    return mapping


def _write_ids(path: Path, ids: list[str]) -> None:
    path.write_text("".join(f"{patient}\n" for patient in ids), encoding="utf-8")


def main() -> None:
    args = parse_args()
    mapping = _read_mapping(args.patient_audit_csv)
    partition = make_stratified_patient_partition(
        mapping,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": args.output_dir / "train_patients.txt",
        "val": args.output_dir / "val_patients.txt",
        "test": args.output_dir / "test_patients.txt",
        "manifest": args.output_dir / "patient_split.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "Refusing to overwrite split files: " + ", ".join(str(path) for path in existing)
        )

    _write_ids(paths["train"], partition.train)
    _write_ids(paths["val"], partition.val)
    _write_ids(paths["test"], partition.test)

    payload = {
        **partition.to_dict(),
        "stratification": "has_r3plus",
        "fractions": {
            "train": args.train_fraction,
            "val": args.val_fraction,
            "test": args.test_fraction,
        },
        "counts": {
            "train": len(partition.train),
            "val": len(partition.val),
            "test": len(partition.test),
            "total": len(mapping),
        },
        "r3plus_counts": {
            "train": sum(mapping[p] for p in partition.train),
            "val": sum(mapping[p] for p in partition.val),
            "test": sum(mapping[p] for p in partition.test),
        },
        "source_patient_audit": str(args.patient_audit_csv),
    }
    paths["manifest"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(payload["counts"], indent=2, sort_keys=True))
    print(json.dumps(payload["r3plus_counts"], indent=2, sort_keys=True))
    print(f"Split manifest: {paths['manifest']}")


if __name__ == "__main__":
    main()
