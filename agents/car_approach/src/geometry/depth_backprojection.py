from __future__ import annotations

import io

import numpy as np


def decode_depth_png_bytes(depth_png_bytes: bytes) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Missing Pillow dependency. Install Pillow in the project container before decoding depth PNGs."
        ) from exc

    with Image.open(io.BytesIO(depth_png_bytes)) as image:
        depth = np.asarray(image)

    if depth.ndim == 3:
        depth = depth[..., 0]

    if np.issubdtype(depth.dtype, np.integer):
        depth_int = depth.astype(np.int64, copy=False)
        if np.any(depth_int < 0):
            raise RuntimeError(f"Depth PNG contains negative integer values: dtype={depth.dtype}")
        return depth_int.astype(np.float32) * 0.001
    if np.issubdtype(depth.dtype, np.floating):
        return depth.astype(np.float32)

    raise RuntimeError(f"Unsupported depth PNG dtype: {depth.dtype}")


def backproject_depth_to_points(
    depth_metric_m: np.ndarray,
    intrinsic_k: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
    pixel_stride: int = 1,
) -> np.ndarray:
    if depth_metric_m.ndim != 2:
        raise ValueError("depth_metric_m must be a 2D array.")
    stride = max(int(pixel_stride), 1)
    h, w = depth_metric_m.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_metric_m[::stride, ::stride].astype(np.float32)

    valid_mask = np.isfinite(z) & (z >= float(min_depth_m)) & (z <= float(max_depth_m))
    if not np.any(valid_mask):
        return np.empty((0, 3), dtype=np.float32)

    fx = float(intrinsic_k[0, 0])
    fy = float(intrinsic_k[1, 1])
    cx = float(intrinsic_k[0, 2])
    cy = float(intrinsic_k[1, 2])

    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    return points[valid_mask].astype(np.float32)
