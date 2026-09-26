from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one frozen supervised LIDC budget end to end: pinned export, integrity "
            "validation and YOLO26-sem training. Use --export-only to materialize and "
            "validate the dataset without starting model training."
        )
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(1337, 2026, 31415), default=1337)
    parser.add_argument(
        "--budget",
        choices=("001pct", "005pct", "010pct", "025pct"),
        default="001pct",
    )
    parser.add_argument("--model", default="yolo26n-sem.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--export-only", action="store_true")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    dataset_dir = args.workdir / f"lidc_seed{args.seed}_{args.budget}"
    run_dir = args.workdir / "runs"
    run_name = f"sup_seed{args.seed}_{args.budget}"

    started = time.time()
    metadata = {
        "arm": "SUP",
        "seed": args.seed,
        "budget": args.budget,
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "dataset_dir": str(dataset_dir.resolve()),
        "run_name": run_name,
        "status": "started",
    }
    metadata_path = args.workdir / f"{run_name}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    try:
        _run(
            [
                sys.executable,
                "scripts/export_lidc_mirror_benchmark.py",
                "--output",
                str(dataset_dir),
                "--seed",
                str(args.seed),
                "--budget",
                args.budget,
                "--splits",
                "train",
                "val",
            ]
        )
        _run(
            [
                sys.executable,
                "scripts/validate_lidc_export.py",
                "--dataset",
                str(dataset_dir),
                "--splits",
                "train",
                "val",
            ]
        )

        if not args.export_only:
            _run(
                [
                    sys.executable,
                    "scripts/train_yolo26_sem_supervised.py",
                    "--data",
                    str(dataset_dir / "dataset.yaml"),
                    "--model",
                    args.model,
                    "--epochs",
                    str(args.epochs),
                    "--imgsz",
                    str(args.imgsz),
                    "--batch",
                    str(args.batch),
                    "--device",
                    args.device,
                    "--seed",
                    str(args.seed),
                    "--project",
                    str(run_dir),
                    "--name",
                    run_name,
                ]
            )

        metadata["status"] = "exported" if args.export_only else "completed"
    except Exception:
        metadata["status"] = "failed"
        raise
    finally:
        metadata["elapsed_seconds"] = round(time.time() - started, 3)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Run metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
