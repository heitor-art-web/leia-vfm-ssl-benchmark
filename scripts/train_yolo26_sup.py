#!/usr/bin/env python3
"""Run the supervised YOLO26-sem baseline from the frozen benchmark config.

The script has a dependency-free `--dry-run` path (apart from PyYAML) so the
resolved experiment can be checked before installing/running Ultralytics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/lidc_benchmark.yaml"),
    )
    parser.add_argument(
        "--model-scale",
        choices=("smoke", "benchmark"),
        default="smoke",
        help="Use n-sem for smoke tests or the pre-declared s-sem core checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/lidc_sup")
    parser.add_argument("--name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_training_config(config: dict, args: argparse.Namespace) -> dict:
    dataset = config["dataset"]
    model = config["model"]
    training = config["training"]

    checkpoint_key = (
        "smoke_checkpoint" if args.model_scale == "smoke" else "benchmark_checkpoint"
    )
    checkpoint = model[checkpoint_key]
    seed = int(args.seed if args.seed is not None else config["seed"])

    run_name = args.name or f"sup_{args.model_scale}_seed{seed}"
    resolved = {
        "checkpoint": checkpoint,
        "data": dataset["ultralytics_data_yaml"],
        "epochs": int(training["epochs"]),
        "imgsz": int(training["image_size"]),
        "batch": int(training["batch_size"]),
        "optimizer": str(training["optimizer"]),
        "patience": int(training["early_stopping_patience"]),
        "seed": seed,
        "deterministic": bool(training["deterministic"]),
        "pretrained": bool(training["pretrained"]),
        "project": args.project,
        "name": run_name,
    }
    if args.device is not None:
        resolved["device"] = args.device
    return resolved


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    resolved = resolve_training_config(config, args)
    print(json.dumps(resolved, indent=2, sort_keys=True))

    if args.dry_run:
        return

    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing YOLO26 runtime. Install with: pip install -r requirements-yolo26.txt"
        ) from exc

    print(f"ultralytics={ultralytics.__version__}")
    print(f"torch={torch.__version__}")

    model = YOLO(resolved.pop("checkpoint"))
    model.train(**resolved)


if __name__ == "__main__":
    main()
