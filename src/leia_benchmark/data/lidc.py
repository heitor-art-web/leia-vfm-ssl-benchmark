from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Sequence
import re

import numpy as np


IGNORE_LABEL = np.uint8(255)
BACKGROUND_LABEL = np.uint8(0)
NODULE_LABEL = np.uint8(1)


@dataclass(frozen=True)
class ConsensusPolicy:
    """Rules for turning multiple reader masks into a conservative semantic target.

    `min_cluster_readers` controls whether a nodule cluster is considered strong
    enough to contribute positive pixels. Clusters below this threshold are
    represented as UNKNOWN rather than forced to background.

    Within an eligible cluster, pixels included by at least
    `positive_fraction` of the available annotation masks are positive.
    Remaining pixels in the union of reader masks are UNKNOWN.
    """

    min_cluster_readers: int = 3
    positive_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.min_cluster_readers < 1:
            raise ValueError("min_cluster_readers must be >= 1")
        if not 0 < self.positive_fraction <= 1:
            raise ValueError("positive_fraction must be in (0, 1]")


def dicom_slices_to_hu(images: Sequence[object]) -> np.ndarray:
    """Convert an ordered sequence of DICOM slices into an HxWxZ HU volume.

    `pydicom.Dataset.pixel_array` exposes stored pixel values; modality rescale
    slope/intercept are therefore applied explicitly here.
    """
    if not images:
        raise ValueError("images is empty")

    planes = []
    expected_shape = None
    for image in images:
        pixels = np.asarray(image.pixel_array, dtype=np.float32)
        if pixels.ndim != 2:
            raise ValueError("each DICOM slice must be 2D")
        if expected_shape is None:
            expected_shape = pixels.shape
        elif pixels.shape != expected_shape:
            raise ValueError("all DICOM slices must have identical in-plane shape")

        slope = float(getattr(image, "RescaleSlope", 1.0))
        intercept = float(getattr(image, "RescaleIntercept", 0.0))
        planes.append(pixels * slope + intercept)

    return np.stack(planes, axis=-1).astype(np.float32, copy=False)


def window_to_uint8(
    image: np.ndarray,
    lower_hu: float = -1000.0,
    upper_hu: float = 400.0,
) -> np.ndarray:
    """Clip CT values to a configurable window and map to uint8.

    The default is an engineering starting point, not a clinically fixed
    window. It must remain configurable and be frozen before benchmarking.
    """
    if lower_hu >= upper_hu:
        raise ValueError("lower_hu must be smaller than upper_hu")
    arr = np.asarray(image, dtype=np.float32)
    arr = np.clip(arr, lower_hu, upper_hu)
    arr = (arr - lower_hu) / (upper_hu - lower_hu)
    return np.rint(arr * 255.0).astype(np.uint8)


def stack_25d(volume: np.ndarray, k: int) -> np.ndarray:
    """Return [z-1, z, z+1] as HxWx3 with edge replication.

    Input volume is expected in HxWxZ order, matching pylidc.
    """
    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError("volume must have shape HxWxZ")
    if not 0 <= k < vol.shape[2]:
        raise IndexError("slice index out of range")
    idx = [max(0, k - 1), k, min(vol.shape[2] - 1, k + 1)]
    return np.stack([vol[:, :, i] for i in idx], axis=-1)


def reader_vote_count(reader_masks: Sequence[np.ndarray]) -> np.ndarray:
    """Count positive reader votes per voxel.

    All masks must be boolean-compatible and have identical shape.
    """
    if not reader_masks:
        raise ValueError("reader_masks is empty")
    shape = np.asarray(reader_masks[0]).shape
    masks = []
    for mask in reader_masks:
        arr = np.asarray(mask, dtype=bool)
        if arr.shape != shape:
            raise ValueError("all reader masks must have the same shape")
        masks.append(arr)
    return np.sum(np.stack(masks, axis=0), axis=0, dtype=np.uint8)


def conservative_cluster_target(
    reader_masks: Sequence[np.ndarray],
    policy: ConsensusPolicy = ConsensusPolicy(),
) -> np.ndarray:
    """Build a binary target for one clustered nodule.

    Returns uint8 values:
    - 0: background within the common bounding box
    - 1: trusted nodule
    - 255: disputed / insufficient evidence

    This function deliberately does not invent a negative label for regions
    that at least one reader considered part of the nodule.
    """
    votes = reader_vote_count(reader_masks)
    n_readers = len(reader_masks)
    target = np.full(votes.shape, BACKGROUND_LABEL, dtype=np.uint8)
    union = votes > 0

    if n_readers < policy.min_cluster_readers:
        target[union] = IGNORE_LABEL
        return target

    threshold = max(1, int(np.ceil(n_readers * policy.positive_fraction)))
    positive = votes >= threshold
    disputed = union & ~positive

    target[positive] = NODULE_LABEL
    target[disputed] = IGNORE_LABEL
    return target


def merge_cluster_target(
    scan_target: np.ndarray,
    cluster_target: np.ndarray,
    bbox: tuple[slice, slice, slice],
) -> None:
    """Merge one cluster target into a full-volume semantic target in-place.

    Precedence is positive > ignore > background. This ensures trusted positive
    nodule evidence is never erased by another overlapping ambiguous cluster.
    """
    dst = scan_target[bbox]
    if dst.shape != cluster_target.shape:
        raise ValueError("bbox shape does not match cluster_target shape")

    positive = cluster_target == NODULE_LABEL
    unknown = cluster_target == IGNORE_LABEL

    dst[(unknown) & (dst == BACKGROUND_LABEL)] = IGNORE_LABEL
    dst[positive] = NODULE_LABEL


def build_scan_target(
    shape: tuple[int, int, int],
    clusters: Iterable[tuple[Sequence[np.ndarray], tuple[slice, slice, slice]]],
    policy: ConsensusPolicy = ConsensusPolicy(),
) -> np.ndarray:
    """Build a full HxWxZ target from pre-aligned cluster masks + bounding boxes."""
    target = np.full(shape, BACKGROUND_LABEL, dtype=np.uint8)
    for reader_masks, bbox in clusters:
        cluster_target = conservative_cluster_target(reader_masks, policy=policy)
        merge_cluster_target(target, cluster_target, bbox)
    return target


def slice_has_signal(mask: np.ndarray) -> bool:
    """True when a 2D target contains nodule or UNKNOWN pixels."""
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("mask must be 2D")
    return bool(np.any((arr == NODULE_LABEL) | (arr == IGNORE_LABEL)))


def stable_scan_key(patient_id: str, series_instance_uid: str) -> str:
    """Return a filesystem-safe stable scan identifier.

    LIDC-IDRI has 1,010 patients but 1,018 CT series, so patient ID alone is not
    a unique scan key. Hashing the DICOM SeriesInstanceUID prevents filename
    collisions while keeping patient identity visible for leakage checks.
    """
    patient = patient_id_from_path(patient_id)
    uid = str(series_instance_uid).strip()
    if not uid:
        raise ValueError("series_instance_uid is empty")
    suffix = hashlib.sha1(uid.encode("utf-8")).hexdigest()[:12]
    return f"{patient}_{suffix}"


def patient_id_from_path(path: str | Path) -> str:
    """Extract a canonical `LIDC-IDRI-dddd` patient id from a path or filename."""
    match = re.search(r"LIDC-IDRI-\d{4}", str(path))
    if match:
        return match.group(0)
    raise ValueError(f"no LIDC patient id found in path: {path}")
