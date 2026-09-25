from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from leia_benchmark.data.lidc_mirror import medotter_case_map, medotter_scan_summaries
from leia_benchmark.data.lidc_validation import select_visual_validation_cohort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select and download a five-case LIDC-IDRI visual-QC cohort from the "
            "public MedOtter mirror: one no-contour control and four high-suspicion "
            "radiologist-rated nodules."
        )
    )
    parser.add_argument("--output", type=Path, default=Path("data/lidc_bootstrap"))
    parser.add_argument("--repo-id", default="MedOtter/LIDC-IDRI")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--n-suspicious", type=int, default=4)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Select the cohort and write its manifest without downloading NIfTI volumes.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    if args.n_suspicious < 1:
        raise SystemExit("--n-suspicious must be >= 1")

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install -e '.[lidc]'"
        ) from exc

    args.output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    info = api.dataset_info(args.repo_id, revision=args.revision)
    resolved_revision = info.sha
    if not resolved_revision:
        raise RuntimeError(f"Could not resolve a commit SHA for {args.repo_id}@{args.revision}")

    metadata_paths: dict[str, Path] = {}
    for filename in ("scans.csv", "nodules.csv"):
        downloaded = hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            repo_type="dataset",
            revision=resolved_revision,
            local_dir=args.output,
        )
        metadata_paths[filename] = Path(downloaded)

    scan_rows = _read_csv(metadata_paths["scans.csv"])
    nodule_rows = _read_csv(metadata_paths["nodules.csv"])
    summaries = medotter_scan_summaries(scan_rows, nodule_rows)
    cohort = select_visual_validation_cohort(
        summaries, n_suspicious=args.n_suspicious
    )
    case_map = medotter_case_map(scan_rows)

    cohort_rows: list[dict[str, object]] = []
    for role, scan in cohort:
        case_id = case_map[(scan.patient_id, scan.series_instance_uid)]
        cohort_rows.append(
            {
                "role": role,
                "case_id": case_id,
                "patient_id": scan.patient_id,
                "series_uid": scan.series_instance_uid,
                "n_nodules": scan.n_clusters,
                "selected_nodule_index": ""
                if scan.best_cluster_index is None
                else scan.best_cluster_index,
                "n_annotations": scan.best_reader_count,
                "malignancy_median": ""
                if scan.best_malignancy_median is None
                else scan.best_malignancy_median,
                "diameter_mm_median": ""
                if scan.best_diameter_mm_median is None
                else scan.best_diameter_mm_median,
            }
        )

    cohort_path = args.output / "cohort.csv"
    with cohort_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cohort_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cohort_rows)

    source = {
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "cohort_manifest": cohort_path.name,
        "note": (
            "LIDC malignancy is a subjective radiologist likelihood rating, not a "
            "pathology-confirmed cancer label. This mirror is a development convenience; "
            "the canonical dataset source remains TCIA/IDC."
        ),
    }
    (args.output / "SOURCE.json").write_text(
        json.dumps(source, indent=2), encoding="utf-8"
    )

    if args.metadata_only:
        print(f"Selected {len(cohort_rows)} cases; metadata written to {cohort_path.resolve()}")
        return

    downloaded_files = 0
    for row in cohort_rows:
        case_id = str(row["case_id"])
        for filename in (
            f"images/{case_id}.nii.gz",
            f"masks/{case_id}.nii.gz",
            f"masks_annotation_count/{case_id}.nii.gz",
        ):
            hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                repo_type="dataset",
                revision=resolved_revision,
                local_dir=args.output,
            )
            downloaded_files += 1

    print(
        f"Selected {len(cohort_rows)} cases and downloaded {downloaded_files} NIfTI files "
        f"to {args.output.resolve()}"
    )
    print("Next: python scripts/render_lidc_mirror_previews.py --root " + str(args.output))


if __name__ == "__main__":
    main()
