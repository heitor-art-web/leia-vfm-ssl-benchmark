from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from leia_benchmark.evaluation import (
    BinarySegmentationCounts,
    counts_from_masks,
    macro_average_metrics,
    metrics_from_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a YOLO26 semantic checkpoint on an exported LIDC split using "
            "foreground Dice/IoU/precision/recall with 255 excluded from every metric."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--hard-slices", type=int, default=20)
    return parser.parse_args()


def _read_manifest(dataset: Path, split: str) -> list[dict[str, str]]:
    path = dataset / f"manifest_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    return rows


def _hard_slice_record(row: dict[str, str], counts: BinarySegmentationCounts) -> dict[str, object]:
    return {
        "patient_id": row["patient_id"],
        "case_id": row["case_id"],
        "slice_index": int(row["slice_index"]),
        "image": row["image"],
        "mask": row["mask"],
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
    }


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if args.batch < 1:
        raise SystemExit("--batch must be >= 1")

    rows = _read_manifest(args.dataset, args.split)

    from ultralytics import YOLO

    model = YOLO(str(args.checkpoint))
    global_counts = BinarySegmentationCounts()
    patient_counts: dict[str, BinarySegmentationCounts] = defaultdict(BinarySegmentationCounts)
    case_counts: dict[str, BinarySegmentationCounts] = defaultdict(BinarySegmentationCounts)
    hard_rows: list[tuple[int, int, dict[str, object]]] = []

    started = time.time()
    prediction_seconds = 0.0

    for start in range(0, len(rows), args.batch):
        batch_rows = rows[start : start + args.batch]
        sources = [str(args.dataset / row["image"]) for row in batch_rows]
        predict_started = time.time()
        results = model.predict(
            source=sources,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
            stream=False,
        )
        prediction_seconds += time.time() - predict_started
        if len(results) != len(batch_rows):
            raise RuntimeError(
                f"prediction count mismatch: expected {len(batch_rows)}, got {len(results)}"
            )

        for row, result in zip(batch_rows, results):
            semantic_mask = getattr(result, "semantic_mask", None)
            if semantic_mask is None:
                raise RuntimeError("YOLO result does not expose semantic_mask")
            prediction = semantic_mask.data.cpu().numpy()

            with Image.open(args.dataset / row["mask"]) as mask_image:
                target = np.asarray(mask_image)

            if prediction.shape != target.shape:
                raise RuntimeError(
                    f"shape mismatch for {row['image']}: prediction={prediction.shape}, target={target.shape}"
                )
            predicted_values = set(np.unique(prediction).tolist())
            if not predicted_values.issubset({0, 1}):
                raise RuntimeError(
                    f"unexpected predicted class ids for {row['image']}: {sorted(predicted_values)}"
                )

            counts = counts_from_masks(target, prediction, positive_label=1, ignore_label=255)
            global_counts += counts
            patient_counts[row["patient_id"]] += counts
            case_counts[row["case_id"]] += counts
            hard_rows.append((counts.fn, counts.fp, _hard_slice_record(row, counts)))

    elapsed = time.time() - started
    total_pixels = global_counts.valid_pixels + global_counts.ignored_pixels

    fn_ranked = [record for _, _, record in sorted(hard_rows, key=lambda item: (item[0], item[1]), reverse=True)]
    fp_ranked = [record for _, _, record in sorted(hard_rows, key=lambda item: (item[1], item[0]), reverse=True)]

    payload = {
        "schema_version": 1,
        "task": "LIDC-IDRI pulmonary nodule semantic segmentation",
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "positive_class": 1,
        "ignore_label": 255,
        "global": {
            "counts": global_counts.to_dict(),
            "metrics": metrics_from_counts(global_counts),
            "unknown_fraction": None
            if total_pixels == 0
            else global_counts.ignored_pixels / total_pixels,
        },
        "patient_macro": macro_average_metrics(list(patient_counts.values())),
        "case_macro": macro_average_metrics(list(case_counts.values())),
        "population": {
            "patients": len(patient_counts),
            "cases": len(case_counts),
            "slices": len(rows),
        },
        "inference": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "prediction_seconds": round(prediction_seconds, 3),
            "wall_seconds": round(elapsed, 3),
            "seconds_per_slice": round(prediction_seconds / len(rows), 6),
        },
        "hard_slices": {
            "largest_false_negative_pixel_counts": fn_ranked[: args.hard_slices],
            "largest_false_positive_pixel_counts": fp_ranked[: args.hard_slices],
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Metrics written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
