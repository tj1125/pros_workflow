from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


SERVER_ROOT = Path(__file__).resolve().parents[2]


def prepare_graspgen_runtime_imports(graspgen_root: Path) -> None:
    """Add shared GraspGen runtime roots to sys.path and set headless defaults."""
    graspgen_root = graspgen_root.expanduser().resolve()
    pointnet2_root = graspgen_root / "pointnet2_ops"

    if "CONDA_PREFIX" not in os.environ:
        os.environ["CONDA_PREFIX"] = str(Path(sys.executable).resolve().parents[1])
    os.environ.setdefault("GRASPGEN_NO_VIS", "1")
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR",
        str(SERVER_ROOT / ".cache" / "torch_extensions"),
    )
    Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)

    for path in (graspgen_root, pointnet2_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def filter_by_approach_direction(
    grasps: np.ndarray,
    scores: np.ndarray,
    max_angle_to_y: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove grasps whose approach direction is too aligned with the Y axis."""
    target_pos = np.array([0.0, 1.0, 0.0], dtype=float)
    target_neg = np.array([0.0, -1.0, 0.0], dtype=float)
    approach = grasps[:, :3, 2]
    norms = np.linalg.norm(approach, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    approach = approach / norms
    cos_th = np.cos(np.deg2rad(max_angle_to_y))
    keep = (approach @ target_pos < cos_th) & (approach @ target_neg < cos_th)
    return grasps[keep], scores[keep]


def infer_grasps_from_point_cloud_with_collision(
    object_pc_local: np.ndarray,
    gripper_config: Path,
    grasp_threshold: float,
    num_grasps: int,
    topk_num_grasps: int,
    scene_pc_local: np.ndarray | None,
    collision_threshold: float,
    max_scene_points: int = 8192,
    num_collision_samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, np.ndarray]]:
    """Run GraspGen on an object-local point cloud and filter grasps by scene collision."""
    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
    from grasp_gen.robot import get_gripper_info
    from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps  # type: ignore

    object_pc_local = np.asarray(object_pc_local, dtype=float)
    if object_pc_local.ndim != 2 or object_pc_local.shape[1] != 3 or len(object_pc_local) == 0:
        raise RuntimeError("Object point cloud must have shape (N, 3) with N > 0.")

    cfg = load_grasp_cfg(str(gripper_config))
    sampler = GraspGenSampler(cfg)
    grasps_t, conf_t = GraspGenSampler.run_inference(
        object_pc_local,
        sampler,
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        topk_num_grasps=topk_num_grasps,
        remove_outliers=True,
    )
    if len(grasps_t) == 0:
        raise RuntimeError("No grasps returned by GraspGen inference.")

    grasps = grasps_t.detach().cpu().numpy()
    confidences = conf_t.detach().cpu().numpy()
    grasps, confidences = filter_by_approach_direction(grasps, confidences)
    if len(grasps) == 0:
        raise RuntimeError("No grasps remain after approach filtering.")

    all_grasps_local = np.array(grasps, copy=True)
    all_scores = np.array(confidences, copy=True)
    total_after_approach = int(len(grasps))
    scene_points_used = 0
    collision_free_mask = np.ones(len(grasps), dtype=bool)
    scene_pc_used = np.zeros((0, 3), dtype=float)
    if scene_pc_local is not None and len(scene_pc_local) > 0:
        scene_pc_local = np.asarray(scene_pc_local, dtype=float)
        if len(scene_pc_local) > max_scene_points:
            idx = np.random.choice(len(scene_pc_local), max_scene_points, replace=False)
            scene_pc_local = scene_pc_local[idx]
        scene_pc_used = np.array(scene_pc_local, copy=True)
        scene_points_used = int(len(scene_pc_local))

        gripper_info = get_gripper_info(cfg.data.gripper_name)
        collision_free_mask = filter_colliding_grasps(
            scene_pc=scene_pc_local,
            grasp_poses=grasps,
            gripper_collision_mesh=gripper_info.collision_mesh,
            collision_threshold=collision_threshold,
            num_collision_samples=num_collision_samples,
        )
        grasps = grasps[collision_free_mask]
        confidences = confidences[collision_free_mask]
        if len(grasps) == 0:
            raise RuntimeError("No collision-free grasps remain after obstacle filtering.")

    stats = {
        "num_grasps_after_approach_filter": total_after_approach,
        "num_grasps_after_collision_filter": int(len(grasps)),
        "num_scene_points_used_for_collision": scene_points_used,
    }
    debug_data = {
        "all_grasps_local": all_grasps_local,
        "all_scores": all_scores,
        "collision_free_mask": collision_free_mask,
        "collision_free_grasps_local": all_grasps_local[collision_free_mask],
        "collision_free_scores": all_scores[collision_free_mask],
        "object_pc_local": np.asarray(object_pc_local, dtype=float),
        "scene_pc_local": scene_pc_used,
        "grasp_inference_coordinate_frame": np.array("object_local"),
    }
    return grasps, confidences, stats, debug_data
