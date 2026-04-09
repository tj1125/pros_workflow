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


def grasp_pitch_degrees(grasps: np.ndarray) -> np.ndarray:
    """Return grasp pitch angles in degrees for R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    grasps = np.asarray(grasps, dtype=float)
    if grasps.size == 0:
        return np.zeros((0,), dtype=float)
    if grasps.ndim == 2:
        grasps = grasps[None, ...]

    rotation = grasps[:, :3, :3]
    sy = np.sqrt(rotation[:, 0, 0] ** 2 + rotation[:, 1, 0] ** 2)
    return np.degrees(np.arctan2(-rotation[:, 2, 0], sy))


def filter_by_pitch(
    grasps: np.ndarray,
    scores: np.ndarray,
    max_abs_pitch_deg: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep only grasps whose pitch stays within +/-threshold."""
    if max_abs_pitch_deg < 0:
        raise ValueError("max_abs_pitch_deg must be non-negative.")

    pitch_deg = grasp_pitch_degrees(grasps)
    keep = np.abs(pitch_deg) <= float(max_abs_pitch_deg)
    return grasps[keep], scores[keep], keep, pitch_deg


def filter_to_object_front_side(
    grasps: np.ndarray,
    scores: np.ndarray,
    max_local_z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep only grasps whose origin stays on the object's front half in object-local Z.

    In grasp_agent, object_local is centered at the target point-cloud centroid and keeps
    camera-axis orientation. `local_z <= 0` therefore means the grasp origin is no farther
    away from the camera than the object center, which is the same front/back split used in
    the MeshCat debugging view.
    """
    local_z = np.asarray(grasps, dtype=float)[:, 2, 3]
    keep = local_z <= float(max_local_z)
    return grasps[keep], scores[keep], keep, local_z


def infer_grasps_from_point_cloud_with_collision(
    object_pc_local: np.ndarray,
    gripper_config: Path,
    grasp_threshold: float,
    num_grasps: int,
    topk_num_grasps: int,
    scene_pc_local: np.ndarray | None,
    collision_threshold: float,
    max_local_z: float = 0.0,
    max_pitch_deg: float = 30.0,
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
    total_after_approach = int(len(grasps))

    grasps, confidences, front_keep_mask, local_z_all = filter_to_object_front_side(
        grasps,
        confidences,
        max_local_z=max_local_z,
    )
    if len(grasps) == 0:
        raise RuntimeError("No grasps remain after front-side filtering.")
    total_after_front = int(len(grasps))

    grasps, confidences, pitch_keep_mask, pitch_deg_all = filter_by_pitch(
        grasps,
        confidences,
        max_abs_pitch_deg=max_pitch_deg,
    )
    if len(grasps) == 0:
        raise RuntimeError("No grasps remain after pitch filtering.")

    all_grasps_local = np.array(grasps, copy=True)
    all_scores = np.array(confidences, copy=True)
    all_local_z = np.array(local_z_all[front_keep_mask][pitch_keep_mask], copy=True)
    all_pitch_deg = np.array(pitch_deg_all[pitch_keep_mask], copy=True)
    total_after_pitch = int(len(grasps))
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
        "num_grasps_after_front_filter": total_after_front,
        "num_grasps_after_pitch_filter": total_after_pitch,
        "num_grasps_after_collision_filter": int(len(grasps)),
        "num_scene_points_used_for_collision": scene_points_used,
    }
    debug_data = {
        "all_grasps_local": all_grasps_local,
        "all_scores": all_scores,
        "all_local_z": all_local_z,
        "all_pitch_deg": all_pitch_deg,
        "collision_free_mask": collision_free_mask,
        "collision_free_grasps_local": all_grasps_local[collision_free_mask],
        "collision_free_scores": all_scores[collision_free_mask],
        "collision_free_local_z": all_local_z[collision_free_mask],
        "collision_free_pitch_deg": all_pitch_deg[collision_free_mask],
        "object_pc_local": np.asarray(object_pc_local, dtype=float),
        "scene_pc_local": scene_pc_used,
        "grasp_inference_coordinate_frame": np.array("object_local"),
        "front_grasp_split_axis": np.array("object_local_z"),
        "max_grasp_local_z": np.array(float(max_local_z)),
        "grasp_pitch_convention": np.array("Rz(yaw)*Ry(pitch)*Rx(roll)"),
        "max_grasp_pitch_deg": np.array(float(max_pitch_deg)),
    }
    return grasps, confidences, stats, debug_data
