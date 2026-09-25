from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

from leia_benchmark.data.lidc import normalize_hu
from leia_benchmark.data.lidc_mirror import (
    semantic_target_from_default_and_annotation_count,
)
from leia_benchmark.data.lidc_validation import (
    choose_lung_like_slice,
    choose_max_positive_slice,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render human-QC PNGs from the five-case MedOtter LIDC bootstrap cohort."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("data/lidc_bootstrap"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--window-low", type=float, default=-1000.0)
    parser.add_argument("--window-high", type=float, default=400.0)
    parser.add_argument("--crop-margin", type=int, default=48)
    return parser.parse_args()


def _load_aligned(path: Path) -> nib.Nifti1Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return nib.load(str(path))


def _verify_alignment(reference: nib.Nifti1Image, other: nib.Nifti1Image, name: str) -> None:
    if reference.shape != other.shape:
        raise RuntimeError(
            f"NIfTI shape mismatch for {name}: {reference.shape} vs {other.shape}"
        )
    if not np.allclose(reference.affine, other.affine, atol=1e-4, rtol=0):
        raise RuntimeError(f"NIfTI affine mismatch for {name}")


def _overlay(gray: np.ndarray, target: np.ndarray) -> np.ndarray:
    base = np.repeat(gray[:, :, None], 3, axis=2).astype(np.float32)
    out = base.copy()
    alpha = 0.45
    positive = target == 1
    unknown = target == 255
    if np.any(positive):
        color = np.array([30.0, 220.0, 70.0], dtype=np.float32)
        out[positive] = (1.0 - alpha) * out[positive] + alpha * color
    if np.any(unknown):
        color = np.array([255.0, 170.0, 20.0], dtype=np.float32)
        out[unknown] = (1.0 - alpha) * out[unknown] + alpha * color
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _mask_preview(target: np.ndarray) -> np.ndarray:
    out = np.zeros(target.shape, dtype=np.uint8)
    out[target == 255] = 127
    out[target == 1] = 255
    return out


def _crop_bounds(target: np.ndarray, margin: int) -> tuple[slice, slice]:
    evidence = np.argwhere(target != 0)
    if evidence.size == 0:
        return slice(0, target.shape[0]), slice(0, target.shape[1])
    r0, c0 = evidence.min(axis=0)
    r1, c1 = evidence.max(axis=0)
    r0 = max(0, int(r0) - margin)
    c0 = max(0, int(c0) - margin)
    r1 = min(target.shape[0], int(r1) + margin + 1)
    c1 = min(target.shape[1], int(c1) + margin + 1)
    return slice(r0, r1), slice(c0, c1)


def _panel(image: Image.Image, title: str) -> Image.Image:
    image = image.convert("RGB")
    header = 30
    out = Image.new("RGB", (image.width, image.height + header), "white")
    out.paste(image, (0, header))
    ImageDraw.Draw(out).text((8, 8), title, fill="black")
    return out


def _contact_sheet(ct: np.ndarray, target: np.ndarray, overlay: np.ndarray, crop: np.ndarray) -> Image.Image:
    panels = [
        _panel(Image.fromarray(ct, mode="L"), "CT: lung window"),
        _panel(Image.fromarray(_mask_preview(target), mode="L"), "Mask: white=trusted, gray=UNKNOWN"),
        _panel(Image.fromarray(overlay, mode="RGB"), "Overlay: green=trusted, amber=UNKNOWN"),
        _panel(Image.fromarray(crop, mode="RGB"), "Evidence crop"),
    ]
    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (2 * width, 2 * height), "white")
    for idx, panel in enumerate(panels):
        sheet.paste(panel, ((idx % 2) * width, (idx // 2) * height))
    return sheet


def _write_html(rows: list[dict[str, object]], output: Path) -> None:
    cards = []
    for row in rows:
        role = html.escape(str(row["role"]))
        case_id = html.escape(str(row["case_id"]))
        malignancy = html.escape(str(row.get("malignancy_median", "")))
        image_path = html.escape(str(row["contact_sheet"]))
        cards.append(
            f"<section><h2>{case_id}</h2><p>{role} | radiologist malignancy median: "
            f"{malignancy or 'n/a'}</p><img src='{image_path}' style='max-width:100%'></section>"
        )
    page = """<!doctype html>
<html><head><meta charset='utf-8'><title>LIDC visual validation</title>
<style>body{font-family:system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}section{margin:2rem 0;border-bottom:1px solid #ddd;padding-bottom:2rem}</style>
</head><body><h1>LIDC-IDRI five-case visual validation</h1>
<p>Green = trusted default-reference foreground. Amber = contour evidence excluded from the default reference and treated as UNKNOWN for QC. Malignancy is a radiologist likelihood score, not pathology confirmation.</p>
""" + "\n".join(cards) + "\n</body></html>"
    (output / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.window_high <= args.window_low:
        raise SystemExit("--window-high must be greater than --window-low")
    if args.crop_margin < 0:
        raise SystemExit("--crop-margin must be >= 0")

    cohort_path = args.root / "cohort.csv"
    if not cohort_path.exists():
        raise SystemExit(
            f"Missing {cohort_path}. Run scripts/bootstrap_lidc_hf.py first."
        )
    with cohort_path.open(newline="", encoding="utf-8") as handle:
        cohort = list(csv.DictReader(handle))
    if not cohort:
        raise SystemExit("cohort.csv is empty")

    output = args.output or (args.root / "previews")
    output.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, object]] = []

    for row in cohort:
        case_id = row["case_id"]
        image_nii = _load_aligned(args.root / "images" / f"{case_id}.nii.gz")
        mask_nii = _load_aligned(args.root / "masks" / f"{case_id}.nii.gz")
        count_nii = _load_aligned(
            args.root / "masks_annotation_count" / f"{case_id}.nii.gz"
        )
        _verify_alignment(image_nii, mask_nii, f"{case_id} default mask")
        _verify_alignment(image_nii, count_nii, f"{case_id} annotation-count mask")

        volume_hu = np.asarray(image_nii.dataobj, dtype=np.float32)
        default_mask = np.asarray(mask_nii.dataobj)
        annotation_count = np.asarray(count_nii.dataobj)
        if volume_hu.ndim != 3:
            raise RuntimeError(f"{case_id} CT is not 3D: shape={volume_hu.shape}")

        target = semantic_target_from_default_and_annotation_count(
            default_mask, annotation_count
        )
        role = row["role"]
        if role == "high_suspicion":
            if not np.any(target == 1):
                raise RuntimeError(
                    f"{case_id} is marked high_suspicion but its default mask is empty"
                )
            z = choose_max_positive_slice(target)
        else:
            if role == "control_no_volumetric_nodule" and np.any(annotation_count > 0):
                raise RuntimeError(
                    f"{case_id} was selected as no-contour control but annotation evidence exists"
                )
            z = choose_lung_like_slice(volume_hu)

        ct = normalize_hu(volume_hu[:, :, z], low=args.window_low, high=args.window_high)
        slice_target = target[:, :, z]
        overlay = _overlay(ct, slice_target)
        rs, cs = _crop_bounds(slice_target, args.crop_margin)
        crop = overlay[rs, cs]

        case_dir = output / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(ct, mode="L").save(case_dir / "ct.png")
        Image.fromarray(_mask_preview(slice_target), mode="L").save(case_dir / "mask.png")
        Image.fromarray(overlay, mode="RGB").save(case_dir / "overlay.png")
        Image.fromarray(crop, mode="RGB").save(case_dir / "crop.png")
        _contact_sheet(ct, slice_target, overlay, crop).save(case_dir / "contact_sheet.png")

        report.append(
            {
                **row,
                "slice_index": z,
                "trusted_positive_pixels": int(np.count_nonzero(slice_target == 1)),
                "unknown_pixels": int(np.count_nonzero(slice_target == 255)),
                "contact_sheet": str(Path(case_id) / "contact_sheet.png"),
            }
        )

    report_path = output / "validation_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0].keys()))
        writer.writeheader()
        writer.writerows(report)
    _write_html(report, output)

    print(f"Rendered {len(report)} cases to {output.resolve()}")
    print(f"Open {output / 'index.html'} to validate all five cases.")


if __name__ == "__main__":
    main()
