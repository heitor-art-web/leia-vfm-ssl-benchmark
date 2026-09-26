from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from leia_benchmark.data.lidc_mirror import (
    medotter_case_map,
    medotter_scan_summaries,
    select_ambiguous_validation_cases,
)
from leia_benchmark.data.lidc_validation import select_visual_validation_cohort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select and download a seven-case LIDC-IDRI visual-QC cohort from the "
            "public MedOtter mirror: one no-contour control, four high-suspicion "
            "trusted nodules, and clean one-reader/two-reader ambiguous examples."
        )
    )
    parser.add_argument("--output", type=Path, default=Path("data/lidc_bootstrap"))
    parser.add_argument("--repo-id", default="MedOtter/LIDC-IDRI")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--n-suspicious", type=int, default=4)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help=(
            "Select the cohort and verify all required NIfTI paths exist, but do not "
            "download the large image/mask volumes."
        ),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _required_case_files(case_id: str) -> tuple[str, str, str, str]:
    return (
        f"images/{case_id}.nii.gz",
        f"masks/{case_id}.nii.gz",
        f"masks_instance/{case_id}.nii.gz",
        f"masks_annotation_count/{case_id}.nii.gz",
    )


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
    trusted_cohort = select_visual_validation_cohort(
        summaries, n_suspicious=args.n_suspicious
    )
    case_map = medotter_case_map(scan_rows)

    cohort_rows: list[dict[str, object]] = []
    for role, scan in trusted_cohort:
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
                "n_annotations": scan.best_annotation_count,
                "malignancy_median": ""
                if scan.best_malignancy_median is None
                else scan.best_malignancy_median,
                "diameter_mm_median": ""
                if scan.best_diameter_mm_median is None
                else scan.best_diameter_mm_median,
            }
        )

    ambiguous = select_ambiguous_validation_cases(
        scan_rows,
        nodule_rows,
        exclude_patient_ids={str(row["patient_id"]) for row in cohort_rows},
    )
    for case in ambiguous:
        cohort_rows.append(
            {
                "role": case.role,
                "case_id": case.case_id,
                "patient_id": case.patient_id,
                "series_uid": case.series_instance_uid,
                "n_nodules": 1,
                "selected_nodule_index": case.nodule_index,
                "n_annotations": case.annotation_count,
                "malignancy_median": ""
                if case.malignancy_score is None
                else case.malignancy_score,
                "diameter_mm_median": ""
                if case.diameter_mm is None
                else case.diameter_mm,
            }
        )

    if len({str(row["patient_id"]) for row in cohort_rows}) != len(cohort_rows):
        raise RuntimeError("visual-QC cohort unexpectedly contains duplicate patients")

    expected_size = 1 + args.n_suspicious + 2
    if len(cohort_rows) != expected_size:
        raise RuntimeError(
            f"expected {expected_size} QC cases, selected {len(cohort_rows)}"
        )

    missing_files: list[str] = []
    required_files: list[str] = []
    for row in cohort_rows:
        case_id = str(row["case_id"])
        for filename in _required_case_files(case_id):
            required_files.append(filename)
            if not api.file_exists(
                repo_id=args.repo_id,
                filename=filename,
                repo_type="dataset",
                revision=resolved_revision,
            ):
                missing_files.append(filename)
    if missing_files:
        formatted = "\n  - ".join(missing_files)
        raise RuntimeError(
            "Selected cohort references missing mirror files at the pinned revision:\n  - "
            + formatted
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
        "required_files_verified": required_files,
        "selection": {
            "control": "one patient with no volumetrically annotated >=3 mm nodule when available",
            "high_suspicion": (
                "four patient-unique scans with >=3 nodule annotations and median "
                "radiologist malignancy score >=4, spread across the observed diameter range"
            ),
            "ambiguous": (
                "one one-reader and one two-reader case; each scan has exactly one "
                "volumetric nodule, zero default-reference nodules, and is chosen "
                "deterministically near the middle of its diameter distribution"
            ),
            "visual_qc": (
                "trusted high-suspicion nodules are isolated by masks_instance id; "
                "ambiguous cases exercise the scan-wide annotation-count -> UNKNOWN path"
            ),
        },
        "note": (
            "LIDC malignancy is a subjective radiologist likelihood rating, not a "
            "pathology-confirmed cancer label. Annotation count is not treated as exposed "
            "reader identity. This mirror is a development convenience; the canonical "
            "dataset source remains TCIA/IDC."
        ),
    }
    (args.output / "SOURCE.json").write_text(
        json.dumps(source, indent=2), encoding="utf-8"
    )

    if args.metadata_only:
        print(
            f"Selected {len(cohort_rows)} patient-unique cases and verified "
            f"{len(required_files)} required NIfTI paths at revision {resolved_revision}."
        )
        print(f"Metadata written to {cohort_path.resolve()}")
        return

    for filename in required_files:
        hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            repo_type="dataset",
            revision=resolved_revision,
            local_dir=args.output,
        )

    print(
        f"Selected {len(cohort_rows)} cases and downloaded {len(required_files)} NIfTI files "
        f"to {args.output.resolve()}"
    )
    print("Next: python scripts/render_lidc_mirror_previews.py --root " + str(args.output))


if __name__ == "__main__":
    main()
