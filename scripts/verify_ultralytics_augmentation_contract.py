from __future__ import annotations

import argparse
import inspect

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the pinned Ultralytics semantic spatial-augmentation contract relied on by the 2.5D LIDC baseline."
        )
    )
    parser.add_argument("--expected-version", default="8.4.163")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import ultralytics
    from ultralytics.data.augment import RandomFlip, RandomPerspective

    if ultralytics.__version__ != args.expected_version:
        raise RuntimeError(
            f"Ultralytics version drift: expected {args.expected_version}, got {ultralytics.__version__}"
        )

    perspective_source = inspect.getsource(RandomPerspective.apply_semantic)
    for marker in ("cv2.INTER_NEAREST", "borderValue=255"):
        if marker not in perspective_source:
            raise RuntimeError(f"semantic affine contract is missing {marker!r}")

    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 1
    mask[1, 1] = 255

    # Translate right by one pixel. New support introduced by the affine transform must be
    # UNKNOWN rather than false background, and class ids must remain discrete.
    matrix = np.eye(3, dtype=np.float32)
    matrix[0, 2] = 1.0
    labels = {"semantic_mask": mask.copy()}
    transform = RandomPerspective(degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0)
    transformed = transform.apply_semantic(labels, {"M": matrix, "size": (5, 5)})["semantic_mask"]

    values = set(np.unique(transformed).tolist())
    if not values.issubset({0, 1, 255}):
        raise RuntimeError(f"affine augmentation manufactured invalid class ids: {sorted(values)}")
    if not np.all(transformed[:, 0] == 255):
        raise RuntimeError("affine augmentation did not mark newly introduced border pixels as UNKNOWN=255")
    if transformed[2, 3] != 1:
        raise RuntimeError("trusted foreground was not translated with nearest-neighbor semantic geometry")
    if transformed[1, 2] != 255:
        raise RuntimeError("existing UNKNOWN pixel was not preserved by semantic affine geometry")

    # Horizontal flip must transform the semantic mask exactly, without class remapping.
    flip = RandomFlip(p=1.0, direction="horizontal")
    flipped = flip.apply_semantic(
        {"semantic_mask": mask.copy()},
        {"flip": True, "direction": "horizontal", "h": 5, "w": 5, "flip_idx": None},
    )["semantic_mask"]
    if not np.array_equal(flipped, np.fliplr(mask)):
        raise RuntimeError("horizontal semantic flip does not exactly match the image-space flip")

    print(f"Ultralytics {ultralytics.__version__} semantic augmentation contract verified.")
    print("- semantic affine interpolation is nearest-neighbor")
    print("- transformed border support is UNKNOWN=255, not background")
    print("- trusted and UNKNOWN class ids remain discrete")
    print("- horizontal flip transforms the semantic target exactly")


if __name__ == "__main__":
    main()
