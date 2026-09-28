from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that a YOLO training preflight produced a checkpoint and finite logged losses."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_csv = args.run_dir / "results.csv"
    checkpoint = args.run_dir / "weights" / "last.pt"
    if not results_csv.exists():
        raise FileNotFoundError(results_csv)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if checkpoint.stat().st_size == 0:
        raise RuntimeError(f"empty checkpoint: {checkpoint}")

    with results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"training results contain no rows: {results_csv}")

    last = rows[-1]
    numeric: dict[str, float] = {}
    for key, value in last.items():
        if value is None or not value.strip():
            continue
        try:
            numeric[key.strip()] = float(value)
        except ValueError:
            continue

    loss_values = {key: value for key, value in numeric.items() if "loss" in key.lower()}
    if not loss_values:
        raise RuntimeError(f"no loss columns found in {results_csv}")
    bad = {key: value for key, value in loss_values.items() if not math.isfinite(value)}
    if bad:
        raise RuntimeError(f"non-finite training losses: {bad}")

    print(f"YOLO training artifact verification passed for {args.run_dir}")
    print(f"checkpoint_bytes={checkpoint.stat().st_size}")
    for key, value in sorted(loss_values.items()):
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
