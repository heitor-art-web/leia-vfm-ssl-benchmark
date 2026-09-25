from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class ACDCPair:
    patient_id: str
    image_path: Path
    mask_path: Path


def discover_pairs(root: str | Path) -> list[ACDCPair]:
    """Discover image/ground-truth frame pairs in the standard ACDC patient folders."""
    root = Path(root)
    pairs: list[ACDCPair] = []
    for gt in sorted(root.glob("patient*/patient*_frame*_gt.nii.gz")):
        image = Path(str(gt).replace("_gt.nii.gz", ".nii.gz"))
        if not image.exists():
            continue
        patient_id = gt.parent.name
        pairs.append(ACDCPair(patient_id=patient_id, image_path=image, mask_path=gt))
    return pairs


def load_volume(pair: ACDCPair) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(nib.load(str(pair.image_path)).get_fdata(), dtype=np.float32)
    mask = np.asarray(nib.load(str(pair.mask_path)).get_fdata(), dtype=np.int16)
    return image, mask


def zscore_nonzero(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = image.astype(np.float32, copy=True)
    region = x[np.isfinite(x)]
    if region.size == 0:
        return np.zeros_like(x)
    mean = float(region.mean())
    std = float(region.std())
    return (x - mean) / max(std, eps)


def parse_patient_group(patient_dir: str | Path) -> str | None:
    info = Path(patient_dir) / "Info.cfg"
    if not info.exists():
        return None
    text = info.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^Group:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else None
