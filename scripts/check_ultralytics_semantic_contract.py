from __future__ import annotations

import argparse
import inspect
from pathlib import Path

EXPECTED_VERSION = "8.4.163"
REQUIRED_FRAGMENTS = (
    "ignore_index=255",
    "masks.reshape(-1) != 255",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the pinned Ultralytics semantic ignore-label contract."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional path to upstream loss.py. If omitted, inspect the installed package.",
    )
    return parser.parse_args()


def _check(source: str, label: str) -> None:
    missing = [fragment for fragment in REQUIRED_FRAGMENTS if fragment not in source]
    if missing:
        raise SystemExit(
            f"{label} no longer exposes the expected ignore-label contract: " + ", ".join(missing)
        )
    print(f"{label}: target 255 is excluded from semantic loss.")


def main() -> None:
    args = parse_args()
    if args.source is not None:
        _check(args.source.read_text(encoding="utf-8"), f"Ultralytics v{EXPECTED_VERSION} upstream source")
        return

    import ultralytics
    from ultralytics.utils.loss import SemanticSegmentationLoss

    if ultralytics.__version__ != EXPECTED_VERSION:
        raise SystemExit(
            f"Expected ultralytics=={EXPECTED_VERSION}, found {ultralytics.__version__}. "
            "The benchmark semantic-loss contract is pinned to this version."
        )
    _check(inspect.getsource(SemanticSegmentationLoss), f"Installed Ultralytics {EXPECTED_VERSION}")


if __name__ == "__main__":
    main()
