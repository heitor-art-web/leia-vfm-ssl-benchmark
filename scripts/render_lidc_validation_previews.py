from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from leia_benchmark.data.lidc import (
    ConsensusPolicy,
    dicom_images_to_hu,
    merge_semantic_target,
    normalize_hu,
    semantic_target_from_reader_masks,
)
from leia_benchmark.data.lidc_validation import (
    choose_lung_like_slice,
    choose_max_positive_slice,
)

TRUSTED_RGB = np.array([30.0, 220.0, 70.0], dtype=np.float32)
UNKNOWN_RGB = np.array([255.0, 0.0, 255.0], dtype=np.float32)
OVERLAY_ALPHA = 0.58


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render CT, mask, overlay and crop previews for a selected LIDC visual-QC cohort."
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-low", type=float, default=-1000.0)
    parser.add_argument("--window-high", type=float, default=400.0)
    parser.add_argument("--positive-readers", type=int, default=3)
    parser.add_argument("--crop-margin", type=int, default=48)
    return parser.parse_args()


def _parse_annotation_ids(value: str) -> tuple[int, ...]:
    value = value.strip()
    if not value:
        return ()
    return tuple(sorted(int(part) for part in value.split(";") if part))


def _find_cluster(scan, annotation_ids: tuple[int, ...]):
    wanted = set(annotation_ids)
    for cluster in scan.cluster_annotations():
        current = {int(ann.id) for ann in cluster}
        if current == wanted:
            return cluster
    raise RuntimeError(
        f"Could not find cluster {sorted(wanted)} in {scan.patient_id}; "
        "the pylidc database does not match the cohort metadata."
    )


def _cluster_target(scan, cluster, shape: tuple[int, int, int], policy: ConsensusPolicy) -> np.ndarray:
    from pylidc.utils import consensus

    _, bbox, masks = consensus(cluster, clevel=0.5, ret_masks=True, verbose=False)
    local = semantic_target_from_reader_masks(masks, policy)
    target = np.zeros(shape, dtype=np.uint8)
    merge_semantic_target(target, local, bbox)
    return target


def _choose_evidence_slice(mask: np.ndarray) -> int:
    try:
        return choose_max_positive_slice(mask)
    except ValueError:
        evidence = np.count_nonzero(mask != 0, axis=(0, 1))
        if int(evidence.max()) == 0:
            raise RuntimeError("selected nodule cluster produced an empty mask")
        return int(np.argmax(evidence))


def _make_overlay(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    out = base.copy()
    positive = mask == 1
    unknown = mask == 255
    if np.any(positive):
        out[positive] = (1.0 - OVERLAY_ALPHA) * out[positive] + OVERLAY_ALPHA * TRUSTED_RGB
    if np.any(unknown):
        out[unknown] = (1.0 - OVERLAY_ALPHA) * out[unknown] + OVERLAY_ALPHA * UNKNOWN_RGB
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _mask_preview(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=np.uint8)
    out[mask == 255] = 127
    out[mask == 1] = 255
    return out


def _crop_bounds(mask: np.ndarray, margin: int) -> tuple[slice, slice]:
    evidence = np.argwhere(mask != 0)
    if evidence.size == 0:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    r0, c0 = evidence.min(axis=0)
    r1, c1 = evidence.max(axis=0)
    return (
        slice(max(0, int(r0) - margin), min(mask.shape[0], int(r1) + margin + 1)),
        slice(max(0, int(c0) - margin), min(mask.shape[1], int(c1) + margin + 1)),
    )


def _labelled_panel(image: Image.Image, title: str) -> Image.Image:
    image = image.convert("RGB")
    header = 28
    panel = Image.new("RGB", (image.width, image.height + header), "white")
    panel.paste(image, (0, header))
    ImageDraw.Draw(panel).text((8, 7), title, fill="black")
    return panel


def _contact_sheet(ct: np.ndarray, mask: np.ndarray, overlay: np.ndarray, crop: np.ndarray) -> Image.Image:
    panels = [
        _labelled_panel(Image.fromarray(ct), "CT lung window"),
        _labelled_panel(Image.fromarray(_mask_preview(mask)), "Mask: white=trusted, gray=UNKNOWN"),
        _labelled_panel(Image.fromarray(overlay), "Overlay: green=trusted, magenta=UNKNOWN"),
        _labelled_panel(Image.fromarray(crop), "Nodule/evidence crop"),
    ]
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (2 * width, 2 * height), "white")
    for idx, panel in enumerate(panels):
        sheet.paste(panel, ((idx % 2) * width, (idx // 2) * height))
    return sheet


def main() -> None:
    args = parse_args()
    if args.window_high <= args.window_low:
        raise SystemExit("--window-high must be greater than --window-low")
    if args.crop_margin < 0:
        raise SystemExit("--crop-margin must be >= 0")

    import pylidc as pl

    with args.cohort.open(newline="", encoding="utf-8") as handle:
        cohort = list(csv.DictReader(handle))
    if not cohort:
        raise SystemExit("Cohort CSV is empty")

    policy = ConsensusPolicy(
        total_readers=4,
        positive_readers=args.positive_readers,
        ambiguous_min_readers=1,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    report_rows: list[dict[str, object]] = []
    for row in cohort:
        patient_id = row["patient_id"]
        series_uid = row["series_instance_uid"]
        annotation_ids = _parse_annotation_ids(row.get("annotation_ids", ""))

        matches = (
            pl.query(pl.Scan)
            .filter(pl.Scan.patient_id == patient_id)
            .filter(pl.Scan.series_instance_uid == series_uid)
            .all()
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one pylidc scan for {patient_id}/{series_uid}, found {len(matches)}"
            )
        scan = matches[0]
        volume_hu = dicom_images_to_hu(scan.load_all_dicom_images(verbose=False))

        if annotation_ids:
            cluster = _find_cluster(scan, annotation_ids)
            target = _cluster_target(scan, cluster, volume_hu.shape, policy)
            z = _choose_evidence_slice(target)
        else:
            target = np.zeros(volume_hu.shape, dtype=np.uint8)
            z = choose_lung_like_slice(volume_hu)

        ct = normalize_hu(volume_hu[:, :, z], low=args.window_low, high=args.window_high)
        mask = target[:, :, z]
        overlay = _make_overlay(ct, mask)
        rs, cs = _crop_bounds(mask, args.crop_margin)
        crop = overlay[rs, cs]

        case_dir = args.output / patient_id
        case_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(ct).save(case_dir / "ct.png")
        Image.fromarray(_mask_preview(mask)).save(case_dir / "mask.png")
        Image.fromarray(overlay).save(case_dir / "overlay.png")
        Image.fromarray(crop).save(case_dir / "crop.png")
        _contact_sheet(ct, mask, overlay, crop).save(case_dir / "contact_sheet.png")

        report_rows.append(
            {
                "role": row["role"],
                "patient_id": patient_id,
                "series_instance_uid": series_uid,
                "annotation_ids": row.get("annotation_ids", ""),
                "slice_index": z,
                "trusted_positive_pixels": int(np.count_nonzero(mask == 1)),
                "unknown_pixels": int(np.count_nonzero(mask == 255)),
                "contact_sheet": str(Path(patient_id) / "contact_sheet.png"),
            }
        )

    report_path = args.output / "validation_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    print(f"Rendered {len(report_rows)} cases to {args.output.resolve()}")
    print("Open the per-case contact_sheet.png files for human validation.")


if __name__ == "__main__":
    main()
