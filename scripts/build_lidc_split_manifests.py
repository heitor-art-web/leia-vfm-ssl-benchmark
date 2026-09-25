from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from leia_benchmark.data.lidc_splits import (
    labelled_subset_size,
    nested_labelled_subsets,
    patient_summaries,
    stratified_patient_split,
)


FRACTIONS = (0.01, 0.05, 0.10, 0.25)
SEEDS = (1337, 2026, 31415)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic patient-level LIDC-IDRI train/val/test and "
            "nested labelled-budget manifests from MedOtter scans.csv metadata."
        )
    )
    parser.add_argument("--scans", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, default=None)
    parser.add_argument("--split-seed", type=int, default=20260925)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _fraction_tag(fraction: float) -> str:
    return f"{int(round(100 * fraction)):03d}pct"


def _write_patient_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty manifest: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    scans = _read_csv(args.scans)
    patients = patient_summaries(scans)
    by_id = {patient.patient_id: patient for patient in patients}

    split = stratified_patient_split(
        patients,
        seed=args.split_seed,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    labelled = nested_labelled_subsets(split["train"], fractions=FRACTIONS, seeds=SEEDS)

    args.out.mkdir(parents=True, exist_ok=True)
    patient_to_split = {
        patient_id: split_name
        for split_name, patient_ids in split.items()
        for patient_id in patient_ids
    }

    all_rows: list[dict[str, object]] = []
    for patient in patients:
        all_rows.append(
            {
                "patient_id": patient.patient_id,
                "split": patient_to_split[patient.patient_id],
                "n_scans": patient.n_scans,
                "n_slices": patient.n_slices,
                "has_default_nodule": int(patient.has_default_nodule),
                "n_default_nodules": patient.n_default_nodules,
                "default_gt_voxels": patient.default_gt_voxels,
            }
        )
    _write_patient_rows(args.out / "patients.csv", all_rows)

    for split_name, patient_ids in split.items():
        _write_patient_rows(
            args.out / f"{split_name}.csv",
            [
                {
                    "patient_id": patient_id,
                    "has_default_nodule": int(by_id[patient_id].has_default_nodule),
                    "n_scans": by_id[patient_id].n_scans,
                    "n_slices": by_id[patient_id].n_slices,
                }
                for patient_id in patient_ids
            ],
        )

    budget_summary: dict[str, dict[str, object]] = {}
    train_set = set(split["train"])
    for seed in SEEDS:
        for fraction in FRACTIONS:
            selected = labelled[(seed, fraction)]
            selected_set = set(selected)
            unlabelled = tuple(sorted(train_set - selected_set))
            tag = _fraction_tag(fraction)
            prefix = f"seed{seed}_{tag}"
            labelled_rows = [
                {
                    "patient_id": patient_id,
                    "has_default_nodule": int(by_id[patient_id].has_default_nodule),
                }
                for patient_id in selected
            ]
            unlabelled_rows = [
                {
                    "patient_id": patient_id,
                    "has_default_nodule": int(by_id[patient_id].has_default_nodule),
                }
                for patient_id in unlabelled
            ]
            _write_patient_rows(args.out / "labelled" / f"{prefix}.csv", labelled_rows)
            _write_patient_rows(args.out / "unlabelled" / f"{prefix}.csv", unlabelled_rows)
            budget_summary[prefix] = {
                "fraction": fraction,
                "labelled_patients": len(selected),
                "expected_labelled_patients": labelled_subset_size(len(train_set), fraction),
                "labelled_with_default_nodule": sum(
                    by_id[patient_id].has_default_nodule for patient_id in selected
                ),
                "unlabelled_patients": len(unlabelled),
            }

    source: dict[str, object] = {}
    if args.source_json is not None:
        source = json.loads(args.source_json.read_text(encoding="utf-8"))

    summary = {
        "source": {
            "repo_id": source.get("repo_id"),
            "resolved_revision": source.get("resolved_revision"),
        },
        "split_policy": {
            "unit": "patient",
            "seed": args.split_seed,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
            "heldout_stratification": "has_default_nodule",
            "label_budget_sampling": "target-blind deterministic hash ordering; nested within seed",
            "label_fractions": list(FRACTIONS),
            "label_seeds": list(SEEDS),
        },
        "counts": {
            "patients_total": len(patients),
            "patients_with_default_nodule": sum(patient.has_default_nodule for patient in patients),
            **{
                split_name: {
                    "patients": len(patient_ids),
                    "with_default_nodule": sum(
                        by_id[patient_id].has_default_nodule for patient_id in patient_ids
                    ),
                }
                for split_name, patient_ids in split.items()
            },
        },
        "label_budgets": budget_summary,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
