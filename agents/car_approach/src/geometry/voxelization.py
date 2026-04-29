from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WorkspaceBounds:
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]


def crop_points_to_workspace(points_xyz: np.ndarray, bounds: WorkspaceBounds) -> np.ndarray:
    if len(points_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)
    min_xyz = np.asarray(bounds.min_xyz, dtype=np.float32).reshape(1, 3)
    max_xyz = np.asarray(bounds.max_xyz, dtype=np.float32).reshape(1, 3)
    mask = np.all(points_xyz >= min_xyz, axis=1) & np.all(points_xyz <= max_xyz, axis=1)
    return np.asarray(points_xyz[mask], dtype=np.float32)


def voxelize_points(
    points_xyz: np.ndarray,
    *,
    voxel_size_m: float,
    max_voxels: int | None = None,
    selection_origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    if len(points_xyz) == 0:
        return np.empty((0, 3), dtype=np.float32)

    voxel_size = float(voxel_size_m)
    voxel_indices = np.floor(np.asarray(points_xyz, dtype=np.float32) / voxel_size).astype(np.int32)
    unique_voxel_indices = np.unique(voxel_indices, axis=0)
    voxel_centers = (unique_voxel_indices.astype(np.float32) + 0.5) * voxel_size

    if max_voxels is not None and len(voxel_centers) > int(max_voxels):
        origin = np.asarray(selection_origin_xyz, dtype=np.float32).reshape(1, 3)
        distances = np.linalg.norm(voxel_centers - origin, axis=1)
        keep_indices = np.argsort(distances, kind="stable")[: int(max_voxels)]
        voxel_centers = voxel_centers[keep_indices]

    sort_order = np.lexsort((voxel_centers[:, 2], voxel_centers[:, 1], voxel_centers[:, 0]))
    return voxel_centers[sort_order].astype(np.float32)
