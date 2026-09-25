from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MedSAMConfig:
    checkpoint: str
    model_type: str = "vit_b"
    device: str = "cuda"
    input_size: int = 1024
    mask_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.input_size <= 0:
            raise ValueError("input_size must be positive")
        if not 0.0 < self.mask_threshold < 1.0:
            raise ValueError("mask_threshold must be in (0, 1)")


def _validate_rgb_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("MedSAM expects HxWx3 image input")
    if image.dtype != np.uint8:
        raise ValueError("MedSAM image input must be uint8")
    return image


def _validate_box_xyxy(box_xyxy: np.ndarray, height: int, width: int) -> np.ndarray:
    box = np.asarray(box_xyxy, dtype=np.float32).reshape(-1)
    if box.shape != (4,):
        raise ValueError("box must contain exactly four XYXY coordinates")
    if not np.all(np.isfinite(box)):
        raise ValueError("box coordinates must be finite")

    x0, y0, x1, y1 = box.tolist()
    x0 = float(np.clip(x0, 0, width))
    x1 = float(np.clip(x1, 0, width))
    y0 = float(np.clip(y0, 0, height))
    y1 = float(np.clip(y1, 0, height))
    if not (x1 > x0 and y1 > y0):
        raise ValueError("box must have positive width and height after clipping")
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def scale_box_xyxy(
    box_xyxy: np.ndarray,
    source_hw: tuple[int, int],
    target_size: int = 1024,
) -> np.ndarray:
    """Scale an XYXY box from an HxW image to MedSAM's square input frame."""
    height, width = source_hw
    if height <= 0 or width <= 0 or target_size <= 0:
        raise ValueError("source dimensions and target_size must be positive")
    box = _validate_box_xyxy(box_xyxy, height=height, width=width)
    scale = np.asarray(
        [target_size / width, target_size / height, target_size / width, target_size / height],
        dtype=np.float32,
    )
    return box * scale


class MedSAMRefiner:
    """Frozen MedSAM box-prompt refiner using the official inference preprocessing.

    This intentionally mirrors the public MedSAM inference path rather than
    routing through `SamPredictor`: MedSAM resizes the medical image to a square
    1024x1024 frame, min-max normalizes it to [0, 1], encodes the image directly,
    scales the box into the same frame, and decodes one binary mask.

    Keeping this wrapper small makes the VFM contribution auditable and avoids
    silently changing prompt/image transforms relative to MedSAM's reference
    inference implementation.
    """

    def __init__(self, cfg: MedSAMConfig):
        try:
            import torch
            from segment_anything import sam_model_registry
        except ImportError as exc:
            raise RuntimeError(
                "MedSAM dependencies are unavailable. Use the pinned MedSAM source revision documented in the repository before enabling MT+MedSAM."
            ) from exc

        if cfg.model_type not in sam_model_registry:
            raise ValueError(f"unknown MedSAM model_type: {cfg.model_type}")

        self.cfg = cfg
        self._torch = torch
        self.model = sam_model_registry[cfg.model_type](checkpoint=cfg.checkpoint)
        self.model = self.model.to(cfg.device)
        self.model.eval()

    def _image_tensor(self, image_rgb_uint8: np.ndarray):
        try:
            from skimage import transform
        except ImportError as exc:
            raise RuntimeError(
                "scikit-image is required for reference-compatible MedSAM preprocessing"
            ) from exc

        image = _validate_rgb_uint8(image_rgb_uint8)
        resized = transform.resize(
            image,
            (self.cfg.input_size, self.cfg.input_size),
            order=3,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.uint8)

        resized = resized.astype(np.float32)
        lo = float(resized.min())
        hi = float(resized.max())
        resized = (resized - lo) / max(hi - lo, 1e-8)

        return (
            self._torch.as_tensor(resized, dtype=self._torch.float32, device=self.cfg.device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

    def predict_from_box(
        self,
        image_rgb_uint8: np.ndarray,
        box_xyxy: np.ndarray,
    ) -> np.ndarray:
        image = _validate_rgb_uint8(image_rgb_uint8)
        height, width = image.shape[:2]
        box_1024 = scale_box_xyxy(
            box_xyxy,
            source_hw=(height, width),
            target_size=self.cfg.input_size,
        )

        torch = self._torch
        import torch.nn.functional as F

        image_tensor = self._image_tensor(image)
        box_tensor = torch.as_tensor(
            box_1024[None, :],
            dtype=torch.float32,
            device=self.cfg.device,
        )[:, None, :]

        with torch.no_grad():
            image_embedding = self.model.image_encoder(image_tensor)
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=box_tensor,
                masks=None,
            )
            low_res_logits, _ = self.model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            probability = torch.sigmoid(low_res_logits)
            probability = F.interpolate(
                probability,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        mask = probability.squeeze().detach().cpu().numpy()
        return (mask > self.cfg.mask_threshold).astype(np.uint8)

    def predict_from_boxes(
        self,
        image_rgb_uint8: np.ndarray,
        boxes_xyxy: list[np.ndarray],
    ) -> np.ndarray:
        """Refine multiple teacher components independently and union the masks."""
        image = _validate_rgb_uint8(image_rgb_uint8)
        union = np.zeros(image.shape[:2], dtype=np.uint8)
        for box in boxes_xyxy:
            union |= self.predict_from_box(image, box)
        return union


def bbox_from_binary(
    mask: np.ndarray,
    min_pixels: int = 1,
    pad_pixels: int = 0,
) -> np.ndarray | None:
    """Return one clipped XYXY box around a single binary foreground region."""
    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("mask must be 2D")
    if min_pixels < 1:
        raise ValueError("min_pixels must be >= 1")
    if pad_pixels < 0:
        raise ValueError("pad_pixels must be >= 0")

    ys, xs = np.where(arr)
    if len(xs) < min_pixels:
        return None

    height, width = arr.shape
    x0 = max(0, int(xs.min()) - pad_pixels)
    y0 = max(0, int(ys.min()) - pad_pixels)
    x1 = min(width, int(xs.max()) + 1 + pad_pixels)
    y1 = min(height, int(ys.max()) + 1 + pad_pixels)
    return np.asarray([x0, y0, x1, y1], dtype=np.float32)


def boxes_from_binary_components(
    mask: np.ndarray,
    min_pixels: int = 9,
    pad_pixels: int = 5,
) -> list[np.ndarray]:
    """Return one box per 8-connected teacher foreground component.

    A pulmonary CT slice can contain multiple spatially separate candidate
    nodules. Prompting MedSAM with one global box would conflate those objects,
    so components are kept independent.
    """
    try:
        from scipy import ndimage
    except ImportError as exc:
        raise RuntimeError("scipy is required for connected-component prompts") from exc

    arr = np.asarray(mask, dtype=bool)
    if arr.ndim != 2:
        raise ValueError("mask must be 2D")
    if min_pixels < 1:
        raise ValueError("min_pixels must be >= 1")
    if pad_pixels < 0:
        raise ValueError("pad_pixels must be >= 0")

    labels, count = ndimage.label(arr, structure=np.ones((3, 3), dtype=np.uint8))
    boxes: list[np.ndarray] = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        box = bbox_from_binary(
            component,
            min_pixels=min_pixels,
            pad_pixels=pad_pixels,
        )
        if box is not None:
            boxes.append(box)
    return boxes


def agreement_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pixels where two hard maps agree; disagreement can remain UNKNOWN."""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.shape != b_arr.shape:
        raise ValueError("shape mismatch")
    return a_arr == b_arr
