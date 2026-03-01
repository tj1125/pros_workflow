from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def filter_by_approach_direction(
    grasps: np.ndarray,
    scores: np.ndarray,
    max_angle_to_y: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove grasps whose approach direction is too aligned with the Y axis (top/bottom grasps)."""
    target_pos = np.array([0.0, 1.0, 0.0], dtype=float)
    target_neg = np.array([0.0, -1.0, 0.0], dtype=float)
    approach = grasps[:, :3, 2]
    norms = np.linalg.norm(approach, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    approach = approach / norms
    cos_th = np.cos(np.deg2rad(max_angle_to_y))
    keep = (approach @ target_pos < cos_th) & (approach @ target_neg < cos_th)
    return grasps[keep], scores[keep]


def infer_grasps_from_mesh(
    mesh: Any,
    gripper_config: Path,
    grasp_threshold: float,
    num_grasps: int,
    topk_num_grasps: int,
    num_sample_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample points from the mesh and run GraspGen inference to produce scored grasp candidates."""
    import trimesh
    import trimesh.transformations as tra
    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

    points, _ = trimesh.sample.sample_surface(mesh, num_sample_points)
    points = np.asarray(points)
    if len(points) == 0:
        raise RuntimeError("No points sampled from aligned mesh.")

    t_subtract = tra.translation_matrix(-points.mean(axis=0))
    points_centered = tra.transform_points(points, t_subtract)

    cfg = load_grasp_cfg(str(gripper_config))
    sampler = GraspGenSampler(cfg)
    grasps_t, conf_t = GraspGenSampler.run_inference(
        points_centered,
        sampler,
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        topk_num_grasps=topk_num_grasps,
        remove_outliers=False,
    )
    if len(grasps_t) == 0:
        raise RuntimeError("No grasps returned by GraspGen inference.")

    grasps = grasps_t.cpu().numpy()
    conf = conf_t.cpu().numpy()
    grasps, conf = filter_by_approach_direction(grasps, conf)
    if len(grasps) == 0:
        raise RuntimeError("No grasps remain after approach filtering.")

    t_restore = tra.inverse_matrix(t_subtract)
    grasps = np.array([t_restore @ g for g in grasps])
    return grasps, conf
