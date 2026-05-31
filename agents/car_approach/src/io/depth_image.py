"""Depth image decoding helpers."""

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
