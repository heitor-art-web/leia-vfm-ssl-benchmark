from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from leia_benchmark.data.lidc_targets import semantic_target_from_annotation_count

DEFAULT_REPO = "MedOtter/LIDC-IDRI"
DEFAULT_REVISION = "6ddf7cab582ee2f9da278f996f3bc9c1940f57b2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile only the frozen labelled TRAIN target distribution for one LIDC budget. "
            "No validation/test masks are opened and no CT images are downloaded."
        )
    )
    parser.add_argument("--split-manifest", type=Path, default=Path("splits/lidc/v1/manifest.json"))
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, choices=(1337, 2026, 31415), default=1337)
    parser.add_argument(
        "--budget",
        choices=("001pct", "005pct", "010pct", "025pct"),
        default="001pct",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _split_patients(manifest: dict, split: str) -> list[str]:
    if isinstance(manifest.get("splits"), dict) and split in manifest["splits"]:
        return list(manifest["splits"][split])
    if split in manifest and isinstance(manifest[split], list):
        return list(manifest[split])
    raise KeyError(f"could not find patient list for split={split!r}")


def _labelled_patients(manifest: dict, seed: int, budget: str) -> list[str]:
    labelled = manifest.get("labelled")
    if not isinstance(labelled, dict):
        raise KeyError("split manifest lacks labelled budgets")
    seed_key = f"seed{seed}"
    if seed_key not in labelled or budget not in labelled[seed_key]:
        raise KeyError(f"split manifest lacks labelled/{seed_key}/{budget}")
    return list(labelled[seed_key][budget])


def _safe_fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    labelled = sorted(_labelled_patients(manifest, args.seed, args.budget))
    frozen_train = set(_split_patients(manifest, "train"))
    if not set(labelled).issubset(frozen_train):
        raise RuntimeError("labelled budget contains patients outside frozen train split")

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    info = api.dataset_info(args.repo_id, revision=args.revision)
    if info.sha != args.revision:
        raise RuntimeError(
            f"requested pinned revision {args.revision} resolved to {info.sha}; refusing drift"
        )

    local_dir = args.cache_dir or (args.output.parent / "lidc_target_profile_cache")
    local_dir.mkdir(parents=True, exist_ok=True)
    scans_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename="scans.csv",
            repo_type="dataset",
            revision=args.revision,
            local_dir=local_dir,
        )
    )
    scans = _read_csv(scans_path)
    scans_by_patient: dict[str, list[dict[str, str]]] = {}
    for row in scans:
        scans_by_patient.setdefault(row["patient_id"], []).append(row)

    totals = {
        "patients": len(labelled),
        "scans": 0,
        "slices": 0,
        "trusted_slices": 0,
        "unknown_only_slices": 0,
        "background_only_slices": 0,
        "all_ignore_slices": 0,
        "voxels": 0,
        "background_voxels": 0,
        "trusted_voxels": 0,
        "unknown_voxels": 0,
    }
    scan_rows: list[dict[str, object]] = []

    for patient_id in labelled:
        patient_scans = scans_by_patient.get(patient_id, [])
        if not patient_scans:
            raise RuntimeError(f"labelled patient {patient_id} is absent from scans.csv")
        for row in patient_scans:
            case_id = row["case_id"]
            count_path = Path(
                hf_hub_download(
                    repo_id=args.repo_id,
                    filename=f"masks_annotation_count/{case_id}.nii.gz",
                    repo_type="dataset",
                    revision=args.revision,
                    local_dir=local_dir,
                )
            )
            count_nii = nib.load(str(count_path))
            annotation_count = np.asarray(count_nii.dataobj)
            target = semantic_target_from_annotation_count(annotation_count)

            if target.ndim != 3:
                raise RuntimeError(f"expected 3D target for {case_id}, got shape={target.shape}")

            per_slice_trusted = np.count_nonzero(target == 1, axis=(0, 1))
            per_slice_unknown = np.count_nonzero(target == 255, axis=(0, 1))
            slice_pixels = target.shape[0] * target.shape[1]
            trusted_slices = int(np.count_nonzero(per_slice_trusted > 0))
            unknown_only_slices = int(np.count_nonzero((per_slice_trusted == 0) & (per_slice_unknown > 0)))
            all_ignore_slices = int(np.count_nonzero(per_slice_unknown == slice_pixels))
            background_only_slices = int(
                np.count_nonzero((per_slice_trusted == 0) & (per_slice_unknown == 0))
            )

            trusted_voxels = int(np.count_nonzero(target == 1))
            unknown_voxels = int(np.count_nonzero(target == 255))
            voxels = int(target.size)
            background_voxels = voxels - trusted_voxels - unknown_voxels

            scan_rows.append(
                {
                    "patient_id": patient_id,
                    "case_id": case_id,
                    "shape": list(target.shape),
                    "zooms_mm": [float(value) for value in count_nii.header.get_zooms()[:3]],
                    "slices": int(target.shape[2]),
                    "trusted_slices": trusted_slices,
                    "unknown_only_slices": unknown_only_slices,
                    "background_only_slices": background_only_slices,
                    "all_ignore_slices": all_ignore_slices,
                    "trusted_voxels": trusted_voxels,
                    "unknown_voxels": unknown_voxels,
                    "background_voxels": background_voxels,
                }
            )

            totals["scans"] += 1
            totals["slices"] += int(target.shape[2])
            totals["trusted_slices"] += trusted_slices
            totals["unknown_only_slices"] += unknown_only_slices
            totals["background_only_slices"] += background_only_slices
            totals["all_ignore_slices"] += all_ignore_slices
            totals["voxels"] += voxels
            totals["background_voxels"] += background_voxels
            totals["trusted_voxels"] += trusted_voxels
            totals["unknown_voxels"] += unknown_voxels

    if totals["scans"] == 0:
        raise RuntimeError("no scans were profiled")

    payload = {
        "schema_version": 1,
        "scope": "labelled training subset only; validation/test labels are not inspected",
        "repo_id": args.repo_id,
        "revision": args.revision,
        "seed": args.seed,
        "budget": args.budget,
        "labelled_patients": labelled,
        "target_policy": {
            "background": 0,
            "trusted": 1,
            "unknown_ignore": 255,
            "trusted_rule": ">=3 contour-annotation votes",
            "unknown_rule": "1-2 contour-annotation votes",
        },
        "totals": totals,
        "fractions": {
            "trusted_slice_fraction": _safe_fraction(totals["trusted_slices"], totals["slices"]),
            "unknown_only_slice_fraction": _safe_fraction(
                totals["unknown_only_slices"], totals["slices"]
            ),
            "background_only_slice_fraction": _safe_fraction(
                totals["background_only_slices"], totals["slices"]
            ),
            "trusted_voxel_fraction": _safe_fraction(totals["trusted_voxels"], totals["voxels"]),
            "unknown_voxel_fraction": _safe_fraction(totals["unknown_voxels"], totals["voxels"]),
        },
        "scans": scan_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Target profile written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
