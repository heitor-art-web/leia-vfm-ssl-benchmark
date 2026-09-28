from __future__ import annotations

import inspect


def main() -> None:
    import ultralytics
    from ultralytics.utils.loss import SemanticSegmentationLoss

    expected = "8.4.163"
    if ultralytics.__version__ != expected:
        raise SystemExit(
            f"Expected ultralytics=={expected}, found {ultralytics.__version__}. "
            "The benchmark semantic-loss contract is pinned to this version."
        )

    source = inspect.getsource(SemanticSegmentationLoss)
    required_fragments = (
        "ignore_index=255",
        "masks.reshape(-1) != 255",
    )
    missing = [fragment for fragment in required_fragments if fragment not in source]
    if missing:
        raise SystemExit(
            "Pinned Ultralytics semantic loss no longer exposes the expected ignore-label contract: "
            + ", ".join(missing)
        )

    print(f"Ultralytics {expected} semantic ignore contract verified: target 255 is excluded from loss.")


if __name__ == "__main__":
    main()
