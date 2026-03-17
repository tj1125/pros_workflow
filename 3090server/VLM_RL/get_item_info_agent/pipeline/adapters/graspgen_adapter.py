from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _mesh_reference_center(mesh: Any) -> np.ndarray:
    bounds = np.asarray(mesh.bounds, dtype=float)
    if bounds.shape != (2, 3):
        raise RuntimeError(f"Unexpected mesh bounds shape: {bounds.shape}")
    return bounds.mean(axis=0)


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
    """Run GraspGen in Unity-local coordinates and return Unity-local grasp poses."""
    import trimesh
    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

    points, _ = trimesh.sample.sample_surface(mesh, num_sample_points)
    points = np.asarray(points)
    if len(points) == 0:
        raise RuntimeError("No points sampled from aligned mesh.")

    reference_center = _mesh_reference_center(mesh)
    points_centered_unity_local = points - reference_center

    cfg = load_grasp_cfg(str(gripper_config))
    sampler = GraspGenSampler(cfg)
    grasps_t, conf_t = GraspGenSampler.run_inference(
        points_centered_unity_local,
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

    return grasps, conf


def infer_grasps_from_mesh_with_collision(
    mesh: Any,
    gripper_config: Path,
    grasp_threshold: float,
    num_grasps: int,
    topk_num_grasps: int,
    num_sample_points: int,
    scene_pc: np.ndarray | None,
    collision_threshold: float,
    max_scene_points: int = 8192,
    num_collision_samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, np.ndarray]]:
    """Run GraspGen/collision entirely in Unity-local coordinates."""
    import trimesh
    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
    from grasp_gen.robot import get_gripper_info
    from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps  # type: ignore

    points, _ = trimesh.sample.sample_surface(mesh, num_sample_points)
    points = np.asarray(points)
    if len(points) == 0:
        raise RuntimeError("No points sampled from aligned mesh.")

    reference_center = _mesh_reference_center(mesh)
    points_centered_unity_local = points - reference_center

    cfg = load_grasp_cfg(str(gripper_config))
    sampler = GraspGenSampler(cfg)
    grasps_t, conf_t = GraspGenSampler.run_inference(
        points_centered_unity_local,
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

    all_grasps_unity_local = np.array(grasps, copy=True)
    all_scores = np.array(conf, copy=True)
    total_after_approach = int(len(grasps))
    scene_points_used = 0
    collision_free_mask = np.ones(len(grasps), dtype=bool)
    scene_unity_local = np.zeros((0, 3), dtype=float)
    if scene_pc is not None and len(scene_pc) > 0:
        scene_unity_local = np.asarray(scene_pc, dtype=float)
        if len(scene_unity_local) > max_scene_points:
            idx = np.random.choice(len(scene_unity_local), max_scene_points, replace=False)
            scene_unity_local = scene_unity_local[idx]
        scene_points_used = int(len(scene_unity_local))

        gripper_info = get_gripper_info(cfg.data.gripper_name)
        collision_free_mask = filter_colliding_grasps(
            scene_pc=scene_unity_local,
            grasp_poses=grasps,
            gripper_collision_mesh=gripper_info.collision_mesh,
            collision_threshold=collision_threshold,
            num_collision_samples=num_collision_samples,
        )
        grasps = grasps[collision_free_mask]
        conf = conf[collision_free_mask]
        if len(grasps) == 0:
            raise RuntimeError("No collision-free grasps remain after obstacle filtering.")

    stats = {
        "num_grasps_after_approach_filter": total_after_approach,
        "num_grasps_after_collision_filter": int(len(grasps)),
        "num_scene_points_used_for_collision": scene_points_used,
    }
    debug_data = {
        "all_grasps": all_grasps_unity_local,
        "all_scores": all_scores,
        "collision_free_mask": collision_free_mask,
        "collision_free_grasps": all_grasps_unity_local[collision_free_mask],
        "collision_free_scores": all_scores[collision_free_mask],
        "pc_object": points_centered_unity_local,
        "pc_object_raw": points_centered_unity_local,
        "pc_scene": scene_unity_local,
        "object_reference_center_mesh_local": np.asarray(reference_center, dtype=float),
        "saved_coordinate_frame": np.array("unity_local"),
        "collision_inference_coordinate_frame": np.array("unity_local"),
    }
    return grasps, conf, stats, debug_data
