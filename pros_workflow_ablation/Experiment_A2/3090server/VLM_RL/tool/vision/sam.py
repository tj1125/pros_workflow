from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from tool.runtime.memory import release_cuda_memory


class BBoxLike(Protocol):
    x1: float
    y1: float
    x2: float
    y2: float


def clamp_bbox(bbox: BBoxLike, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Clamp bounding box coordinates to image bounds."""
    h, w = image_shape[:2]
    left = max(0, min(w, int(math.floor(bbox.x1))))
    top = max(0, min(h, int(math.floor(bbox.y1))))
    right = max(0, min(w, int(math.ceil(bbox.x2))))
    bottom = max(0, min(h, int(math.ceil(bbox.y2))))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid bbox after clamp: {(left, top, right, bottom)}")
    return left, top, right, bottom


def sam_segment_with_bbox(
    image_bgr: np.ndarray,
    bbox: BBoxLike,
    model_type: str,
    checkpoint: Path,
    device: str,
) -> np.ndarray:
    """Run SAM segmentation using the YOLO bbox as the box prompt; returns the best mask."""
    from segment_anything import SamPredictor, sam_model_registry  # type: ignore

    if model_type not in sam_model_registry:
        raise ValueError(f"Unsupported SAM model type: {model_type}")

    sam = None
    predictor = None
    try:
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam = sam.to(device=device)
        predictor = SamPredictor(sam)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        box = np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2], dtype=np.float32)

        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=True,
        )
        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM returned no masks for the YOLO bbox prompt.")

        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(bool)
    finally:
        if predictor is not None:
            del predictor
        if sam is not None:
            try:
                sam.cpu()
            except Exception:
                pass
            del sam
        release_cuda_memory()
