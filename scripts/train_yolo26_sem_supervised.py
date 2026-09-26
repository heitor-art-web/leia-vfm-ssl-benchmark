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
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--project", default="runs/lidc_yolo26_sem")
    parser.add_argument("--name", default="sup")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "deterministic": True,
        "project": args.project,
        "name": args.name,
    }
    if args.device is not None:
        kwargs["device"] = args.device

    model.train(**kwargs)


if __name__ == "__main__":
    main()
