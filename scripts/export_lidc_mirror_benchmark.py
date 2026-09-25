from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
import yaml

from leia_benchmark.data.lidc import make_25d_slice
from leia_benchmark.data.lidc_targets import semantic_target_from_annotation_count

DEFAULT_REPO = "MedOtter/LIDC-IDRI"
DEFAULT_REVISION = "6ddf7cab582ee2f9da278f996f3bc9c1940f57b2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a frozen supervised LIDC-IDRI budget from the pinned NIfTI mirror "
            "to Ultralytics YOLO26 semantic PNG masks. The benchmark target is derived "
            "directly from voxelwise contour-annotation counts: >=3 trusted, 1-2 UNKNOWN."
        )
    )
    parser.add_argument("--split-manifest", type=Path, default=Path("splits/lidc/v1/manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--budget",
        choices=("001pct", "005pct", "010pct", "025pct"),
        default="001pct",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val"],
        help="Export labelled train patients for the selected budget plus complete requested held-out splits.",
    )
    parser.add_argument("--window-low", type=float, default=-1000.0)
    parser.add_argument("--window-high", type=float, default=400.0)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write the exact case plan and provenance without downloading NIfTI volumes.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _split_patients(manifest: dict, split: str) -> list[str]:
    if isinstance(manifest.get("splits"), dict) and split in manifest["splits"]:
        return list(manifest["splits"][split])
    if split in manifest and isinstance(manifest[split], list):
        return list(manifest[split])
    raise KeyError(f"could not find patient list for split={split!r} in split manifest")


def _labelled_patients(manifest: dict, seed: int, budget: str) -> list[str]:
    labelled = manifest.get("labelled")
    if not isinstance(labelled, dict):
        raise KeyError("split manifest lacks labelled budgets")
    seed_key = f"seed{seed}"
    if seed_key not in labelled or budget not in labelled[seed_key]:
        raise KeyError(f"split manifest lacks labelled/{seed_key}/{budget}")
    return list(labelled[seed_key][budget])


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.window_high <= args.window_low:
        raise SystemExit("--window-high must be greater than --window-low")

    split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    labelled = set(_labelled_patients(split_manifest, args.seed, args.budget))
    frozen_train = set(_split_patients(split_manifest, "train"))
    if not labelled.issubset(frozen_train):
        raise RuntimeError("labelled budget contains patients outside the frozen train split")

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install the LIDC extras: pip install -e '.[lidc]'") from exc

    args.output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.dataset_info(args.repo_id, revision=args.revision)
    resolved_revision = info.sha
    if resolved_revision != args.revision:
        raise RuntimeError(
            f"requested pinned revision {args.revision} resolved to {resolved_revision}; refusing drift"
        )

    scans_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename="scans.csv",
            repo_type="dataset",
            revision=resolved_revision,
            local_dir=args.output / "source",
        )
    )
    scans = _read_csv(scans_path)
    scans_by_patient: dict[str, list[dict[str, str]]] = {}
    for row in scans:
        scans_by_patient.setdefault(row["patient_id"], []).append(row)

    requested_patients: dict[str, list[str]] = {}
    for split in args.splits:
        requested_patients[split] = (
            sorted(labelled) if split == "train" else sorted(_split_patients(split_manifest, split))
        )

    plan: list[dict[str, object]] = []
    for split in args.splits:
        for patient_id in requested_patients[split]:
            patient_scans = scans_by_patient.get(patient_id, [])
            if not patient_scans:
                raise RuntimeError(f"patient {patient_id} from frozen split is absent from scans.csv")
            for row in patient_scans:
                case_id = row["case_id"]
                plan.append(
                    {
                        "split": split,
                        "patient_id": patient_id,
                        "case_id": case_id,
                        "series_uid": row["series_uid"],
                        "n_slices": int(row["n_slices"]),
                        "image_source": f"images/{case_id}.nii.gz",
                        "annotation_count_source": f"masks_annotation_count/{case_id}.nii.gz",
                    }
                )

    _write_rows(args.output / "export_plan.csv", plan)
    provenance = {
        "repo_id": args.repo_id,
        "revision": resolved_revision,
        "split_manifest": str(args.split_manifest),
        "seed": args.seed,
        "budget": args.budget,
        "splits": args.splits,
        "target_policy": {
            "0": "no volumetric contour annotation covers voxel",
            "1": ">=3 contour annotations cover voxel",
            "255": "1-2 contour annotations cover voxel; ignore/UNKNOWN",
        },
        "input": "2.5D z-1/z/z+1 with fixed HU window",
        "window_hu": [args.window_low, args.window_high],
        "note": (
            "Reader identity is not exposed; public text must refer to annotation votes. "
            "The mirror is pinned for reproducibility and traces to TCIA/IDC + QIICR annotations."
        ),
    }
    (args.output / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    if args.metadata_only:
        counts: dict[str, int] = {}
        patients: dict[str, set[str]] = {}
        for row in plan:
            split = str(row["split"])
            counts[split] = counts.get(split, 0) + 1
            patients.setdefault(split, set()).add(str(row["patient_id"]))
        summary = ", ".join(
            f"{split}: {len(patients[split])} patients / {counts[split]} scans"
            for split in args.splits
        )
        print(f"Pinned export plan written: {summary}")
        return

    manifests: dict[str, list[dict[str, object]]] = {split: [] for split in args.splits}
    for item in plan:
        split = str(item["split"])
        patient_id = str(item["patient_id"])
        case_id = str(item["case_id"])
        image_rel = str(item["image_source"])
        count_rel = str(item["annotation_count_source"])

        image_path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=image_rel,
                repo_type="dataset",
                revision=resolved_revision,
                local_dir=args.output / "source",
            )
        )
        count_path = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=count_rel,
                repo_type="dataset",
                revision=resolved_revision,
                local_dir=args.output / "source",
            )
        )

        image_nii = nib.load(str(image_path))
        count_nii = nib.load(str(count_path))
        if image_nii.shape != count_nii.shape:
            raise RuntimeError(f"shape mismatch for {case_id}: {image_nii.shape} vs {count_nii.shape}")
        if not np.allclose(image_nii.affine, count_nii.affine, atol=1e-4, rtol=0):
            raise RuntimeError(f"affine mismatch for {case_id}")

        volume_hu = np.asarray(image_nii.dataobj, dtype=np.float32)
        annotation_count = np.asarray(count_nii.dataobj)
        target = semantic_target_from_annotation_count(annotation_count)

        image_dir = args.output / "images" / split
        mask_dir = args.output / "masks" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for z in range(volume_hu.shape[2]):
            stem = f"{case_id}__z{z:04d}"
            image = make_25d_slice(
                volume_hu,
                z,
                low=args.window_low,
                high=args.window_high,
            )
            mask = target[:, :, z].astype(np.uint8, copy=False)
            image_out = image_dir / f"{stem}.png"
            mask_out = mask_dir / f"{stem}.png"
            Image.fromarray(image, mode="RGB").save(image_out)
            Image.fromarray(mask, mode="L").save(mask_out)
            manifests[split].append(
                {
                    "patient_id": patient_id,
                    "case_id": case_id,
                    "series_uid": item["series_uid"],
                    "slice_index": z,
                    "image": str(image_out.relative_to(args.output)),
                    "mask": str(mask_out.relative_to(args.output)),
                    "trusted_pixels": int(np.count_nonzero(mask == 1)),
                    "unknown_pixels": int(np.count_nonzero(mask == 255)),
                }
            )

    for split, rows in manifests.items():
        _write_rows(args.output / f"manifest_{split}.csv", rows)

    dataset = {
        "path": str(args.output.resolve()),
        "train": "images/train" if "train" in args.splits else None,
        "val": "images/val" if "val" in args.splits else None,
        "test": "images/test" if "test" in args.splits else None,
        "masks_dir": "masks",
        "names": {0: "background", 1: "pulmonary_nodule_ge_3mm"},
    }
    dataset = {key: value for key, value in dataset.items() if value is not None}
    (args.output / "dataset.yaml").write_text(
        yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8"
    )
    print(f"Export complete: {args.output.resolve()}")


if __name__ == "__main__":
    main()
