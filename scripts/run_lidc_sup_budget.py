from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

FROZEN_FIRST_RUN = {
    "seed": 1337,
    "budget": "001pct",
    "model": "yolo26n-sem.pt",
    "epochs": 100,
    "imgsz": 512,
    "batch": 8,
    "workers": 8,
}


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
    parser.add_argument(
        "--allow-protocol-override",
        action="store_true",
        help=(
            "Permit deviations from the preregistered seed1337/001pct supervised baseline. "
            "Without this flag, the first scientific baseline refuses schedule drift."
        ),
    )
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_metadata() -> dict[str, object]:
    payload: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("ultralytics", "torch", "torchvision", "numpy", "nibabel", "Pillow")
        },
    }
    try:
        import torch

        payload["torch_runtime"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cudnn_version": None if not torch.backends.cudnn.is_available() else torch.backends.cudnn.version(),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        payload["torch_runtime_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def _enforce_first_run_protocol(args: argparse.Namespace) -> None:
    if args.export_only or args.allow_protocol_override:
        return
    if args.seed != FROZEN_FIRST_RUN["seed"] or args.budget != FROZEN_FIRST_RUN["budget"]:
        return
    mismatches: list[str] = []
    for key in ("model", "epochs", "imgsz", "batch", "workers"):
        actual = getattr(args, key)
        expected = FROZEN_FIRST_RUN[key]
        if actual != expected:
            mismatches.append(f"{key}: expected {expected!r}, got {actual!r}")
    if mismatches:
        detail = "; ".join(mismatches)
        raise SystemExit(
            "Refusing to drift from the preregistered SUP/seed1337/1% baseline: "
            f"{detail}. Use --allow-protocol-override only for an explicitly documented non-baseline run."
        )


def main() -> None:
    args = parse_args()
    _enforce_first_run_protocol(args)

    args.workdir.mkdir(parents=True, exist_ok=True)
    dataset_dir = args.workdir / "datasets" / f"lidc_seed{args.seed}_{args.budget}"
    run_name = f"sup_seed{args.seed}_{args.budget}"
    results_dir = args.results_dir or (args.workdir / "results" / run_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    training_project = results_dir / "training"
    split_manifest = Path("splits/lidc/v1/manifest.json")

    started = time.time()
    metadata = {
        "schema_version": 2,
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
        "split_manifest": str(split_manifest.resolve()),
        "split_manifest_sha256": _sha256(split_manifest),
        "runtime": _runtime_metadata(),
        "protocol_override": bool(args.allow_protocol_override),
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
        provenance_path = dataset_dir / "PROVENANCE.json"
        if not provenance_path.exists():
            raise FileNotFoundError(provenance_path)
        metadata["dataset_provenance_sha256"] = _sha256(provenance_path)

        if args.export_only:
            metadata["status"] = "exported"
        else:
            _run(
                [
                    sys.executable,
                    "scripts/verify_ultralytics_semantic_contract.py",
                    "--expected-version",
                    "8.4.163",
                ]
            )
            metadata["ultralytics_semantic_contract"] = "verified"

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
            _run(
                [
                    sys.executable,
                    "scripts/verify_yolo_training_artifacts.py",
                    "--run-dir",
                    str(training_project / run_name),
                ]
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
            metadata["checkpoint_sha256"] = _sha256(checkpoint)
            metadata["metrics"] = str(metrics_path.resolve())
            metadata["metrics_sha256"] = _sha256(metrics_path)
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
