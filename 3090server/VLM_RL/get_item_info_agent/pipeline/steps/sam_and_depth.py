from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch

from get_item_info_agent.pipeline.types import BoundingBox
from tool.vision.sam import clamp_bbox, sam_segment_with_bbox


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
