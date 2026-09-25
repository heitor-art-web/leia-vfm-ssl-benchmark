from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

IGNORE_LABEL = np.uint8(255)
BACKGROUND_LABEL = np.uint8(0)
NODULE_LABEL = np.uint8(1)


@dataclass(frozen=True)
class ConsensusPolicy:
    """Rules for turning per-reader binary nodule masks into a semantic target."""

    total_readers: int = 4
    positive_readers: int = 3
    ambiguous_min_readers: int = 1

    def __post_init__(self) -> None:
        if self.total_readers < 1:
            raise ValueError("total_readers must be >= 1")
        if not (1 <= self.positive_readers <= self.total_readers):
            raise ValueError("positive_readers must be between 1 and total_readers")
        if not (1 <= self.ambiguous_min_readers <= self.positive_readers):
            raise ValueError(
                "ambiguous_min_readers must be between 1 and positive_readers"
            )


def _as_bool_masks(masks: Iterable[np.ndarray]) -> list[np.ndarray]:
    items = [np.asarray(mask, dtype=bool) for mask in masks]
    if not items:
        raise ValueError("at least one reader mask is required")

    shape = items[0].shape
    if len(shape) != 3:
        raise ValueError("reader masks must be 3D")

    if any(mask.shape != shape for mask in items[1:]):
        raise ValueError("all reader masks must have the same shape")

    return items


def reader_vote_count(
    masks: Iterable[np.ndarray], *, total_readers: int = 4
) -> np.ndarray:
    """Count positive reader votes voxelwise.

    Missing reader masks are treated as zero votes. This matches the LIDC-IDRI
    use case where a nodule cluster can contain fewer than the four possible
    reader annotations.
    """
    items = _as_bool_masks(masks)
    if len(items) > total_readers:
        raise ValueError("number of masks cannot exceed total_readers")
    if total_readers < 1:
        raise ValueError("total_readers must be >= 1")

    return np.sum(np.stack(items, axis=0), axis=0, dtype=np.uint8)


def semantic_target_from_reader_masks(
    masks: Iterable[np.ndarray],
    policy: ConsensusPolicy = ConsensusPolicy(),
) -> np.ndarray:
    """Build a 0/1/255 semantic target from independent reader masks.

    0   = background (no reader marked the voxel)
    1   = trusted nodule (at least ``positive_readers`` marked the voxel)
    255 = ambiguous / incomplete annotation (some evidence, below threshold)
    """
    votes = reader_vote_count(masks, total_readers=policy.total_readers)

    target = np.full(votes.shape, BACKGROUND_LABEL, dtype=np.uint8)
    positive = votes >= policy.positive_readers
    ambiguous = (
        (votes >= policy.ambiguous_min_readers)
        & (votes < policy.positive_readers)
    )

    target[positive] = NODULE_LABEL
    target[ambiguous] = IGNORE_LABEL
    return target


def dicom_images_to_hu(images: Iterable[object]) -> np.ndarray:
    """Convert ordered CT DICOM slices to an HxWxZ float32 HU volume.

    ``pydicom.Dataset.pixel_array`` is not assumed to be in Hounsfield units.
    RescaleSlope and RescaleIntercept are applied per slice.
    """
    items = list(images)
    if not items:
        raise ValueError("at least one DICOM image is required")

    converted: list[np.ndarray] = []
    shape: tuple[int, int] | None = None
    for image in items:
        pixels = np.asarray(image.pixel_array, dtype=np.float32)
        if pixels.ndim != 2:
            raise ValueError("each DICOM slice must be 2D")
        if shape is None:
            shape = pixels.shape
        elif pixels.shape != shape:
            raise ValueError("all DICOM slices must have the same shape")

        slope = float(getattr(image, "RescaleSlope", 1.0))
        intercept = float(getattr(image, "RescaleIntercept", 0.0))
        converted.append(pixels * slope + intercept)

    return np.stack(converted, axis=-1).astype(np.float32, copy=False)


def normalize_hu(
    image: np.ndarray,
    *,
    low: float = -1000.0,
    high: float = 400.0,
) -> np.ndarray:
    """Clip a CT slice/window in HU and map it deterministically to uint8."""
    if not high > low:
        raise ValueError("high must be greater than low")

    arr = np.asarray(image, dtype=np.float32)
    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low)
    return np.rint(arr * 255.0).astype(np.uint8)


def make_25d_slice(
    volume_hu: np.ndarray,
    z_index: int,
    *,
    low: float = -1000.0,
    high: float = 400.0,
) -> np.ndarray:
    """Create an HxWx3 2.5D input from z-1, z, z+1 CT slices.

    Border slices use edge replication. The expected volume order is H, W, Z.
    """
    volume = np.asarray(volume_hu)
    if volume.ndim != 3:
        raise ValueError("volume_hu must have shape (H, W, Z)")
    if not (0 <= z_index < volume.shape[2]):
        raise IndexError("z_index out of range")

    indices = (
        max(0, z_index - 1),
        z_index,
        min(volume.shape[2] - 1, z_index + 1),
    )
    channels = [normalize_hu(volume[:, :, z], low=low, high=high) for z in indices]
    return np.stack(channels, axis=-1)


def merge_semantic_target(
    destination: np.ndarray,
    local_target: np.ndarray,
    bbox: tuple[slice, slice, slice],
) -> None:
    """Merge one local nodule target into a full-volume target in place.

    Trusted positive labels have highest precedence. Ambiguous labels (255)
    replace background but never overwrite a trusted positive.
    """
    if destination.ndim != 3 or local_target.ndim != 3:
        raise ValueError("destination and local_target must be 3D")

    region = destination[bbox]
    if region.shape != local_target.shape:
        raise ValueError("bbox region and local_target shapes do not match")

    positive = local_target == NODULE_LABEL
    ambiguous = local_target == IGNORE_LABEL

    region[positive] = NODULE_LABEL
    region[ambiguous & (region == BACKGROUND_LABEL)] = IGNORE_LABEL
