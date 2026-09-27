from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Abort before held-out validation if the final checkpoint is degenerate even on the labelled TRAIN set. "
            "This gate uses no validation/test labels."
        )
    )
    parser.add_argument("--metrics", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    if payload.get("split") != "train":
        raise RuntimeError(f"sanity gate requires train metrics, got split={payload.get('split')!r}")

    counts = payload["global"]["counts"]
    tp = int(counts["tp"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    reference_positive = tp + fn
    predicted_positive = tp + fp

    if reference_positive <= 0:
        raise RuntimeError("labelled TRAIN sanity set contains no trusted foreground; protocol is invalid")
    if predicted_positive <= 0:
        raise RuntimeError(
            "degenerate final checkpoint: zero predicted foreground pixels on labelled TRAIN; "
            "held-out validation must not be inspected"
        )
    if tp <= 0:
        raise RuntimeError(
            "degenerate final checkpoint: predicted foreground never overlaps trusted TRAIN foreground; "
            "held-out validation must not be inspected"
        )

    recall = tp / reference_positive
    precision = tp / predicted_positive
    print("TRAIN sanity gate passed without touching held-out labels.")
    print(f"tp={tp} fp={fp} fn={fn} precision={precision:.6f} recall={recall:.6f}")


if __name__ == "__main__":
    main()
