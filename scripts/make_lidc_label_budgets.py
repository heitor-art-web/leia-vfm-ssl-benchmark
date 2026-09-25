#!/usr/bin/env python3
"""Create deterministic, nested, stratified LIDC annotation budgets.

Budgets are drawn only from the frozen training-patient pool. The default
stratification variable is `has_r3plus` from the patient-level annotation audit,
which records whether a patient contains at least one nodule cluster supported
by three or more readers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from leia_benchmark.splits import make_stratified_label_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_patients", type=Path)
    parser.add_argument("patient_audit_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stratum-column", default="has_r3plus")
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.10, 0.25],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[1337, 2026, 31415],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    ids = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        ids.append(value)
    if not ids:
        raise ValueError("train patient list is empty")
    if len(ids) != len(set(ids)):
        raise ValueError("train patient list contains duplicates")
    return sorted(ids)


def _read_strata(path: Path, column: str) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    mapping: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"patient_id", column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"patient audit missing required columns: {sorted(missing)}")
        for row in reader:
            patient = str(row["patient_id"]).strip()
            value = str(row[column]).strip()
            if not patient or not value:
                raise ValueError("empty patient/stratum in audit")
            if patient in mapping:
                raise ValueError(f"duplicate patient in audit: {patient}")
            mapping[patient] = value
    return mapping


def _fraction_tag(fraction: float) -> str:
    percent = fraction * 100.0
    text = f"{percent:g}".replace(".", "p")
    return f"{text}pct"


def _write(path: Path, ids: list[str]) -> None:
    path.write_text("".join(f"{patient}\n" for patient in ids), encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_ids = _read_ids(args.train_patients)
    audit_strata = _read_strata(args.patient_audit_csv, args.stratum_column)

    missing = set(train_ids) - set(audit_strata)
    if missing:
        raise ValueError(f"train patients missing from audit: {sorted(missing)}")
    train_strata = {patient: audit_strata[patient] for patient in train_ids}

    fractions = sorted(set(args.fractions))
    if any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("all fractions must be in (0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "source_train_patients": str(args.train_patients),
        "source_patient_audit": str(args.patient_audit_csv),
        "stratum_column": args.stratum_column,
        "train_patient_count": len(train_ids),
        "fractions": fractions,
        "seeds": args.seeds,
        "budgets": {},
    }

    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        previous_labelled: set[str] = set()
        seed_manifest = {}

        for fraction in fractions:
            split = make_stratified_label_split(
                train_strata,
                fraction=fraction,
                seed=seed,
            )
            labelled = set(split.labelled)
            if not previous_labelled.issubset(labelled):
                raise RuntimeError("label budgets are unexpectedly non-nested")
            previous_labelled = labelled

            tag = _fraction_tag(fraction)
            labelled_path = seed_dir / f"{tag}_labelled.txt"
            unlabelled_path = seed_dir / f"{tag}_unlabelled.txt"
            existing = [path for path in (labelled_path, unlabelled_path) if path.exists()]
            if existing and not args.overwrite:
                raise SystemExit(
                    "Refusing to overwrite budget files: "
                    + ", ".join(str(path) for path in existing)
                )

            _write(labelled_path, split.labelled)
            _write(unlabelled_path, split.unlabelled)

            labelled_stratum_counts: dict[str, int] = {}
            for patient in split.labelled:
                stratum = train_strata[patient]
                labelled_stratum_counts[stratum] = labelled_stratum_counts.get(stratum, 0) + 1

            seed_manifest[tag] = {
                "fraction": fraction,
                "labelled_count": len(split.labelled),
                "unlabelled_count": len(split.unlabelled),
                "labelled_stratum_counts": labelled_stratum_counts,
                "labelled_file": str(labelled_path),
                "unlabelled_file": str(unlabelled_path),
            }

        manifest["budgets"][str(seed)] = seed_manifest

    manifest_path = args.output_dir / "label_budgets.json"
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing manifest: {manifest_path}")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Budget manifest: {manifest_path}")


if __name__ == "__main__":
    main()
