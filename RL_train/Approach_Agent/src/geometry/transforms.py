from __future__ import annotations

import numpy as np


def quaternion_xyzw_to_rotation_matrix(quaternion_xyzw: tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in quaternion_xyzw]
    norm = (x * x) + (y * y) + (z * z) + (w * w)
    if norm <= 1e-12:
        raise ValueError("Quaternion norm must be non-zero.")
    x /= norm ** 0.5
    y /= norm ** 0.5
    z /= norm ** 0.5
    w /= norm ** 0.5
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def transform_points(
    points_xyz: np.ndarray,
    *,
    rotation_matrix: np.ndarray,
    translation_xyz: tuple[float, float, float],
) -> np.ndarray:
    if len(points_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)
    translation = np.asarray(translation_xyz, dtype=np.float32).reshape(1, 3)
    return (np.asarray(points_xyz, dtype=np.float32) @ rotation_matrix.T) + translation
