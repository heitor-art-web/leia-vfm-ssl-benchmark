from __future__ import annotations

import numpy as np


def semantic_target_from_annotation_count(
    annotation_count: np.ndarray,
    *,
    positive_annotations: int = 3,
    ambiguous_min_annotations: int = 1,
) -> np.ndarray:
    """Build the benchmark 0/1/255 target from voxelwise LIDC annotation counts.

    The MedOtter mirror documents ``masks_annotation_count`` as the number of
    individual LIDC contour annotations covering each voxel. This lets the
    benchmark implement its policy directly without using the mirror's default
    50%-consensus mask:

    * 0   = no volumetric contour annotation covers the voxel;
    * 1   = at least ``positive_annotations`` contours cover the voxel;
    * 255 = some contour evidence exists but is below the trusted threshold.

    Reader identity is not exposed by LIDC, so the public benchmark should call
    these *annotation votes*, not claim that the mask encodes identified readers.
    """
    count = np.asarray(annotation_count)
    if count.ndim != 3:
        raise ValueError("annotation_count must be a 3D HxWxZ volume")
    if positive_annotations < 1:
        raise ValueError("positive_annotations must be >= 1")
    if not (1 <= ambiguous_min_annotations <= positive_annotations):
        raise ValueError(
            "ambiguous_min_annotations must be between 1 and positive_annotations"
        )
    if not np.all(np.isfinite(count)):
        raise ValueError("annotation_count contains non-finite values")
    if np.any(count < 0):
        raise ValueError("annotation_count cannot contain negative values")
    if not np.allclose(count, np.rint(count), atol=0, rtol=0):
        raise ValueError("annotation_count must contain integer-valued counts")

    target = np.zeros(count.shape, dtype=np.uint8)
    trusted = count >= positive_annotations
    unknown = (count >= ambiguous_min_annotations) & ~trusted
    target[unknown] = 255
    target[trusted] = 1
    return target
