from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class MedSAMConfig:
    checkpoint: str
    model_type: str = "vit_b"
    device: str = "cuda"


class MedSAMRefiner:
    """Thin wrapper around the official Segment Anything predictor used by MedSAM.

    The dependency is deliberately optional so the benchmark can run SUP/MT without
    installing the foundation model stack.
    """

    def __init__(self, cfg: MedSAMConfig):
        try:
            from segment_anything import sam_model_registry, SamPredictor
        except ImportError as exc:
            raise RuntimeError(
                "Install the official MedSAM/segment-anything dependencies before enabling MT+MedSAM"
            ) from exc

        sam = sam_model_registry[cfg.model_type](checkpoint=cfg.checkpoint)
        sam.to(device=cfg.device)
        sam.eval()
        self.predictor = SamPredictor(sam)

    def predict_from_box(self, image_rgb_uint8: np.ndarray, box_xyxy: np.ndarray) -> np.ndarray:
        if image_rgb_uint8.ndim != 3 or image_rgb_uint8.shape[2] != 3:
            raise ValueError("MedSAM expects HxWx3 image input")
        self.predictor.set_image(image_rgb_uint8)
        box = np.asarray(box_xyxy, dtype=np.float32)
        transformed = self.predictor.transform.apply_boxes(box[None, :], image_rgb_uint8.shape[:2])
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=transformed,
            multimask_output=False,
        )
        return masks[0].astype(np.uint8)


def bbox_from_binary(mask: np.ndarray, min_pixels: int = 64) -> np.ndarray | None:
    ys, xs = np.where(mask.astype(bool))
    if len(xs) < min_pixels:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def agreement_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pixels where two hard pseudo-label maps agree; disagreement remains unknown."""
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    return a == b
