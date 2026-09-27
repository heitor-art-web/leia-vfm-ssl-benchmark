from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one frozen supervised LIDC budget end to end: pinned export, integrity "
            "validation, fixed-schedule YOLO26-sem training and one held-out evaluation "
            "of the final checkpoint. Use --export-only to materialize and validate the "
            "dataset without starting model training."
        )
    )
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Persistent run-output directory. Defaults to <workdir>/results/<run-name>.",
    )
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
    parser.add_argument("--eval-batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--export-only", action="store_true")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    dataset_dir = args.workdir / "datasets" / f"lidc_seed{args.seed}_{args.budget}"
    run_name = f"sup_seed{args.seed}_{args.budget}"
    results_dir = args.results_dir or (args.workdir / "results" / run_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    training_project = results_dir / "training"

    started = time.time()
    metadata = {
        "schema_version": 1,
        "arm": "SUP",
        "seed": args.seed,
        "budget": args.budget,
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "eval_batch": args.eval_batch,
        "workers": args.workers,
        "device": args.device,
        "dataset_dir": str(dataset_dir.resolve()),
        "results_dir": str(results_dir.resolve()),
        "run_name": run_name,
        "checkpoint_rule": "fixed final epoch; no held-out validation during training",
        "git_revision": _git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "status": "started",
    }
    metadata_path = results_dir / "run.json"
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

        if args.export_only:
            metadata["status"] = "exported"
        else:
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
                    "--workers",
                    str(args.workers),
                    "--device",
                    args.device,
                    "--seed",
                    str(args.seed),
                    "--project",
                    str(training_project),
                    "--name",
                    run_name,
                ]
            )

            checkpoint = training_project / run_name / "weights" / "last.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(
                    f"expected fixed final-epoch checkpoint was not produced: {checkpoint}"
                )
            metrics_path = results_dir / "metrics.json"
            _run(
                [
                    sys.executable,
                    "scripts/evaluate_lidc_semantic.py",
                    "--dataset",
                    str(dataset_dir),
                    "--checkpoint",
                    str(checkpoint),
                    "--split",
                    "val",
                    "--output",
                    str(metrics_path),
                    "--imgsz",
                    str(args.imgsz),
                    "--batch",
                    str(args.eval_batch),
                    "--device",
                    args.device,
                ]
            )
            metadata["checkpoint"] = str(checkpoint.resolve())
            metadata["metrics"] = str(metrics_path.resolve())
            metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["elapsed_seconds"] = round(time.time() - started, 3)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Run metadata: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
