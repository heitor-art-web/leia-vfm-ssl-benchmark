from __future__ import annotations

import argparse
import inspect
import math
from types import SimpleNamespace

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the pinned Ultralytics semantic-segmentation contract relied on by the LIDC benchmark: "
            "PNG-mask mode, two explicit classes, and target value 255 excluded from CE/Dice loss."
        )
    )
    parser.add_argument("--expected-version", default="8.4.163")
    return parser.parse_args()


class _FakeSemanticHead:
    nc = 2


class _FakeSemanticModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.model = [_FakeSemanticHead()]
        self.args = SimpleNamespace(data="lidc.yaml")
        self.class_weights = None


def main() -> None:
    args = parse_args()

    import ultralytics
    from ultralytics.utils.loss import SemanticSegmentationLoss

    if ultralytics.__version__ != args.expected_version:
        raise RuntimeError(
            f"Ultralytics version drift: expected {args.expected_version}, got {ultralytics.__version__}"
        )

    source = inspect.getsource(SemanticSegmentationLoss)
    required_source_markers = (
        "ignore_index=255",
        "valid = masks.reshape(-1) != 255",
        "Compute Dice loss excluding ignore pixels",
    )
    missing = [marker for marker in required_source_markers if marker not in source]
    if missing:
        raise RuntimeError(f"Pinned semantic loss no longer exposes expected ignore contract: {missing}")

    loss_fn = SemanticSegmentationLoss(_FakeSemanticModel())
    target = torch.tensor([[[1, 0], [255, 0]]], dtype=torch.long)

    logits_a = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    logits_b = logits_a.clone()
    # Change only the ignored pixel by a huge amount. Loss must remain unchanged.
    logits_b[0, 0, 1, 0] = -100.0
    logits_b[0, 1, 1, 0] = 100.0

    total_a, items_a = loss_fn(logits_a, {"semantic_mask": target})
    total_b, items_b = loss_fn(logits_b, {"semantic_mask": target})

    if not torch.allclose(total_a, total_b, atol=1e-7, rtol=0):
        raise RuntimeError(
            f"changing logits at target=255 changed total loss: {total_a.item()} vs {total_b.item()}"
        )
    for key in ("ce_loss", "dice_loss"):
        if key not in items_a or key not in items_b:
            raise RuntimeError(f"semantic loss output is missing {key}")
        if not torch.allclose(items_a[key], items_b[key], atol=1e-7, rtol=0):
            raise RuntimeError(f"changing logits at target=255 changed {key}")

    # A completely ignored target must not produce NaN/inf in the pinned implementation.
    all_ignore = torch.full((1, 2, 2), 255, dtype=torch.long)
    total_ignore, items_ignore = loss_fn(logits_a, {"semantic_mask": all_ignore})
    scalar_values = [float(total_ignore.detach().cpu())]
    scalar_values.extend(float(value.detach().cpu()) for value in items_ignore.values())
    if not all(math.isfinite(value) for value in scalar_values):
        raise RuntimeError(f"all-ignore semantic batch produced a non-finite loss: {scalar_values}")

    print(f"Ultralytics {ultralytics.__version__} semantic contract verified.")
    print("- benchmark uses nc=2: explicit background=0 and pulmonary_nodule=1")
    print("- target 255 is excluded from cross-entropy and Dice loss")
    print("- changing logits only at ignored pixels leaves loss unchanged")
    print("- all-ignore synthetic batch remains finite")


if __name__ == "__main__":
    main()
