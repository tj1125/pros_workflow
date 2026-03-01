from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch

from pipeline.types import BoundingBox


def clamp_bbox(bbox: BoundingBox, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
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
    bbox: BoundingBox,
    model_type: str,
    checkpoint: Path,
    device: str,
) -> np.ndarray:
    """Run SAM segmentation using the YOLO bbox as the box prompt; returns the best mask."""
    from segment_anything import SamPredictor, sam_model_registry  # type: ignore

    if model_type not in sam_model_registry:
        raise ValueError(f"Unsupported SAM model type: {model_type}")

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


def build_sam3d_inputs(
    image_bgr: np.ndarray,
    seg_mask_bool: np.ndarray,
    bbox: BoundingBox,
    base_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop the image/mask to the bbox region and append a magenta base strip for SAM3D."""
    left, top, right, bottom = clamp_bbox(bbox, image_bgr.shape)
    crop_bgr = image_bgr[top:bottom, left:right]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_mask = seg_mask_bool[top:bottom, left:right]
    crop_rgb[~crop_mask] = 0

    base_h = max(1, int(math.ceil(crop_rgb.shape[0] * base_ratio)))
    base_rgb = np.full((base_h, crop_rgb.shape[1], 3), [255, 0, 255], dtype=np.uint8)
    base_mask = np.ones((base_h, crop_mask.shape[1]), dtype=bool)

    image_with_base = np.vstack([crop_rgb, base_rgb])
    mask_with_base = np.vstack([crop_mask, base_mask])
    return image_with_base, mask_with_base


def infer_depth_u8(
    image_rgb: np.ndarray,
    mask_bool: np.ndarray,
    depth_model_path: Path,
    device: str,
    input_size: int,
) -> np.ndarray:
    """Run DepthAnything inference and return a uint8 depth map with background masked to 0."""
    from src.seg_crop_depth import infer_depth_image, load_depth_model, normalize_to_uint8  # type: ignore

    model = load_depth_model(depth_model_path, torch.device(device))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    with torch.no_grad():
        depth = infer_depth_image(model, image_bgr, input_size=input_size)

    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()

    depth = depth.astype(np.float32)
    depth[~mask_bool] = 0.0
    return normalize_to_uint8(depth)
