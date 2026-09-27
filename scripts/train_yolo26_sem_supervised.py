from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a supervised YOLO26 semantic baseline on the exported LIDC-IDRI dataset."
    )
    parser.add_argument("--data", type=Path, default=Path("configs/lidc_yolo26_sem.yaml"))
    parser.add_argument("--model", default="yolo26n-sem.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--project", default="runs/lidc_yolo26_sem")
    parser.add_argument("--name", default="sup")
    parser.add_argument(
        "--cls-pw",
        type=float,
        default=1.0,
        help=(
            "Ultralytics semantic class-weight power. The frozen LIDC baseline uses 1.0 so the "
            "trainer applies its full ENet inverse-log pixel-frequency weighting on the labelled TRAIN masks."
        ),
    )
    parser.add_argument("--degrees", type=float, default=5.0)
    parser.add_argument("--translate", type=float, default=0.05)
    parser.add_argument("--scale", type=float, default=0.10)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument(
        "--val-during-training",
        action="store_true",
        help=(
            "Run held-out validation inside each training epoch. The frozen benchmark default is off: "
            "train for a fixed number of epochs, then evaluate the final checkpoint once."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.cls_pw <= 1.0:
        raise SystemExit("--cls-pw must be in [0, 1]")
    if not 0.0 <= args.fliplr <= 1.0:
        raise SystemExit("--fliplr must be in [0, 1]")

    from ultralytics import YOLO

    model = YOLO(args.model)
    kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": True,
        "project": args.project,
        "name": args.name,
        "val": args.val_during_training,
        "plots": False,
        # LIDC TRAIN masks are extremely foreground-sparse. Ultralytics semantic training
        # computes pixel-frequency class weights when cls_pw > 0; 1.0 freezes full
        # inverse-log weighting using TRAIN labels only.
        "cls_pw": args.cls_pw,
        # The 2.5D input channels are adjacent CT slices, not natural-image RGB. Color-space
        # and mosaic transforms would violate that representation, so they are disabled.
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "bgr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        # Keep only conservative spatial transforms that are applied jointly to all channels
        # and the semantic mask.
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "fliplr": args.fliplr,
        "flipud": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
    }
    if args.device is not None:
        kwargs["device"] = args.device

    model.train(**kwargs)


if __name__ == "__main__":
    main()
