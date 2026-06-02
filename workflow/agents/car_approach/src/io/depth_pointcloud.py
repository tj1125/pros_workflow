"""Build point clouds from live or provided depth images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path

import numpy as np

from ...debug_log import debug_stage
from ..geometry import coordinate_transforms as coord
from .camera_capture import AmclPoseSnapshot, capture_rgbd_snapshot
from .depth_image import decode_depth_png_bytes
from .intrinsics import load_camera_intrinsics


@dataclass(frozen=True)
class DepthPointcloudConfig:
    camera_name: str
    intrinsics_path: Path
    min_depth_m: float = 0.19
    max_depth_m: float = 1.5
    pixel_stride: int = 1
    mirror_camera_x: bool = True
    capture_timeout_sec: float = 15.0
    amcl_topic: str = "/amcl_pose"
    pre_capture_amcl_timeout_sec: float = 2.0
    exclude_mask: Any | None = None
    exclude_bbox_xyxy: Any | None = None
    exclude_bbox_flip_y: bool = False


@dataclass(frozen=True)
class DepthPointcloudResult:
    points_xyz: np.ndarray
    source: str
    pointcloud_frame: str
    camera_name: str
    depth_shape: tuple[int, int]
    depth_format: str
    depth_metric_m: np.ndarray | None = None
    depth_camera_x_mirrored: bool = False
    amcl_pose: dict[str, object] | None = None
    excluded_mask_shape: tuple[int, int] | None = None
    excluded_mask_pixel_count: int = 0
    target_mask_shape: tuple[int, int] | None = None
    target_mask_pixel_count: int = 0
    excluded_bbox_xyxy: tuple[int, int, int, int] | None = None
    excluded_bbox_pixel_count: int = 0
    excluded_bbox_flip_y: bool = False


def capture_depth_pointcloud(config: DepthPointcloudConfig) -> DepthPointcloudResult:
    debug_stage("depth_pointcloud", "收深度圖：開始向 ROS camera capture 取 live depth", camera=config.camera_name)
    snapshot = capture_rgbd_snapshot(
        config.camera_name,
        timeout_sec=config.capture_timeout_sec,
        amcl_topic=config.amcl_topic,
        pre_capture_amcl_timeout_sec=config.pre_capture_amcl_timeout_sec,
    )
    debug_stage(
        "depth_pointcloud",
        "收深度圖：已收到 depth bytes，準備 decode 並轉點雲",
        depth_format=snapshot.depth_format,
        has_amcl=snapshot.amcl_pose is not None,
    )
    return depth_png_bytes_to_pointcloud(
        snapshot.depth_bytes,
        config,
        source="ros_depth",
        depth_format=snapshot.depth_format,
        amcl_pose=_amcl_snapshot_to_dict(snapshot.amcl_pose),
    )


def depth_png_bytes_to_pointcloud(
    depth_png_bytes: bytes,
    config: DepthPointcloudConfig,
    *,
    source: str = "depth_png_bytes",
    depth_format: str = "",
    amcl_pose: dict[str, object] | None = None,
) -> DepthPointcloudResult:
    debug_stage("depth_pointcloud", "深度圖轉點雲：讀 intrinsics 並 decode depth PNG", source=source)
    intrinsics = load_camera_intrinsics(config.intrinsics_path)
    depth_metric_m = decode_depth_png_bytes(depth_png_bytes)
    target_mask = _aligned_exclude_mask(config.exclude_mask, depth_metric_m.shape)
    bbox_mask, bbox_xyxy = _bbox_xyxy_to_mask(
        config.exclude_bbox_xyxy,
        depth_metric_m.shape,
        flip_y=bool(config.exclude_bbox_flip_y),
    )
    exclude_mask = _combine_exclude_masks(target_mask, bbox_mask)
    points = coord.backproject_depth_to_points(
        depth_metric_m,
        intrinsics.k,
        min_depth_m=config.min_depth_m,
        max_depth_m=config.max_depth_m,
        pixel_stride=config.pixel_stride,
        mirror_x=config.mirror_camera_x,
        exclude_mask=exclude_mask,
    )
    debug_stage(
        "depth_pointcloud",
        "深度圖轉點雲完成：已得到 camera frame pointcloud",
        shape=(int(depth_metric_m.shape[0]), int(depth_metric_m.shape[1])),
        points=len(points),
        pixel_stride=config.pixel_stride,
        mirror_camera_x=config.mirror_camera_x,
        target_mask_applied=target_mask is not None,
        target_bbox_applied=bbox_mask is not None,
        excluded_mask_pixels=0 if exclude_mask is None else int(np.count_nonzero(exclude_mask)),
        excluded_bbox_pixels=0 if bbox_mask is None else int(np.count_nonzero(bbox_mask)),
    )
    return DepthPointcloudResult(
        points_xyz=points.astype(np.float32),
        source=source,
        pointcloud_frame="camera",
        camera_name=config.camera_name or intrinsics.camera_name,
        depth_shape=(int(depth_metric_m.shape[0]), int(depth_metric_m.shape[1])),
        depth_format=depth_format,
        depth_metric_m=depth_metric_m.astype(np.float32, copy=False),
        depth_camera_x_mirrored=bool(config.mirror_camera_x),
        amcl_pose=amcl_pose,
        excluded_mask_shape=None if exclude_mask is None else (int(exclude_mask.shape[0]), int(exclude_mask.shape[1])),
        excluded_mask_pixel_count=0 if exclude_mask is None else int(np.count_nonzero(exclude_mask)),
        target_mask_shape=None if target_mask is None else (int(target_mask.shape[0]), int(target_mask.shape[1])),
        target_mask_pixel_count=0 if target_mask is None else int(np.count_nonzero(target_mask)),
        excluded_bbox_xyxy=bbox_xyxy,
        excluded_bbox_pixel_count=0 if bbox_mask is None else int(np.count_nonzero(bbox_mask)),
        excluded_bbox_flip_y=bool(config.exclude_bbox_flip_y),
    )


def _aligned_exclude_mask(mask: Any | None, depth_shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, (bytes, bytearray, memoryview)):
        mask_array = _mask_image_bytes_to_array(bytes(mask))
    else:
        mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    if mask_array.ndim != 2:
        raise ValueError(f"target mask must be 2D, got shape {mask_array.shape}.")
    bool_mask = mask_array.astype(bool, copy=False)
    target_shape = (int(depth_shape[0]), int(depth_shape[1]))
    if bool_mask.shape == target_shape:
        return bool_mask
    return _resize_mask_nearest(bool_mask, target_shape)


def _bbox_xyxy_to_mask(
    bbox_xyxy: Any | None,
    depth_shape: tuple[int, int],
    *,
    flip_y: bool = False,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    if bbox_xyxy is None:
        return None, None
    bbox = np.asarray(bbox_xyxy, dtype=np.float64).reshape(-1)
    if len(bbox) != 4 or not np.all(np.isfinite(bbox)):
        raise ValueError(f"exclude_bbox_xyxy must contain four finite values, got {bbox_xyxy!r}.")

    height, width = int(depth_shape[0]), int(depth_shape[1])
    if height <= 0 or width <= 0:
        return None, None

    x0, y0, x1, y1 = [float(v) for v in bbox]
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0:
        x0 *= width
        x1 *= width
        y0 *= height
        y1 *= height
    if flip_y:
        y0, y1 = float(height) - y1, float(height) - y0

    left = max(0, min(width, int(np.floor(min(x0, x1)))))
    right = max(0, min(width, int(np.ceil(max(x0, x1)))))
    top = max(0, min(height, int(np.floor(min(y0, y1)))))
    bottom = max(0, min(height, int(np.ceil(max(y0, y1)))))
    if right <= left or bottom <= top:
        return None, None

    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask, (left, top, right, bottom)


def _combine_exclude_masks(*masks: np.ndarray | None) -> np.ndarray | None:
    combined: np.ndarray | None = None
    for mask in masks:
        if mask is None:
            continue
        bool_mask = np.asarray(mask, dtype=bool)
        combined = bool_mask.copy() if combined is None else (combined | bool_mask)
    return combined


def _mask_image_bytes_to_array(mask_bytes: bytes) -> np.ndarray:
    try:
        from PIL import Image
        import io
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Missing Pillow dependency. Install Pillow before decoding target mask images.") from exc
    with Image.open(io.BytesIO(mask_bytes)) as image:
        return np.asarray(image)


def _resize_mask_nearest(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Missing Pillow dependency. Install Pillow before resizing target masks.") from exc
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    resized = image.resize((int(target_shape[1]), int(target_shape[0])), resample=Image.NEAREST)
    return np.asarray(resized).astype(bool)


def _amcl_snapshot_to_dict(snapshot: AmclPoseSnapshot | None) -> dict[str, object] | None:
    if snapshot is None:
        return None
    return {
        "stamp_sec": snapshot.stamp_sec,
        "position_xyz": list(snapshot.position_xyz),
        "orientation_xyzw": list(snapshot.orientation_xyzw),
        "covariance": list(snapshot.covariance) if snapshot.covariance is not None else None,
    }
