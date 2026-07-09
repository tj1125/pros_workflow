from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tool.runtime.memory import release_cuda_memory


SERVER_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class RawGraspInference:
    cfg: Any
    object_points_local: np.ndarray
    object_colors_local: np.ndarray | None
    object_reference_center: np.ndarray
    object_to_local_transform: np.ndarray
    grasps_local: np.ndarray
    confidences: np.ndarray


@dataclass
class CollisionFilterResult:
    collision_free_mask: np.ndarray
    scene_points_local: np.ndarray
    scene_points_used: int


def bundled_graspgen_root_candidates() -> tuple[Path, ...]:
    """Return bundled runtime candidates ordered by preference."""
    return (
        SERVER_ROOT / "tool" / "graspgen_runtime",
        SERVER_ROOT / "shared_vendor" / "graspgen_runtime",
        SERVER_ROOT / "get_item_info_agent" / "vendor" / "graspgen_runtime",
    )


def resolve_graspgen_runtime_root(graspgen_root: Path | str | None) -> Path:
    """Resolve the effective GraspGen runtime root.

    Explicit existing paths win. Otherwise we fall back to the preferred bundled
    runtime so every agent consumes the same GraspGen source by default.
    """
    if graspgen_root is not None:
        requested_root = Path(graspgen_root).expanduser().resolve()
        if requested_root.exists():
            return requested_root
    else:
        requested_root = None

    for candidate in bundled_graspgen_root_candidates():
        if candidate.exists():
            return candidate.resolve()

    requested_text = str(requested_root) if requested_root is not None else "<unset>"
    raise FileNotFoundError(
        "No GraspGen runtime root found. "
        f"Requested: {requested_text}. "
        f"Tried bundled candidates: {[str(path) for path in bundled_graspgen_root_candidates()]}"
    )


def prepare_graspgen_runtime_imports(graspgen_root: Path) -> None:
    """Add GraspGen runtime roots to sys.path and set headless defaults."""
    graspgen_root = resolve_graspgen_runtime_root(graspgen_root)
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
    approach = np.asarray(grasps, dtype=float)[:, :3, 2]
    norms = np.linalg.norm(approach, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    approach = approach / norms
    cos_th = np.cos(np.deg2rad(max_angle_to_y))
    keep = (approach @ target_pos < cos_th) & (approach @ target_neg < cos_th)
    return np.asarray(grasps)[keep], np.asarray(scores)[keep]


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
    return np.asarray(grasps)[keep], np.asarray(scores)[keep], keep, pitch_deg


def filter_to_object_front_side(
    grasps: np.ndarray,
    scores: np.ndarray,
    max_local_z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep only grasps whose origin stays on the object's front half in object-local Z."""
    local_z = np.asarray(grasps, dtype=float)[:, 2, 3]
    keep = local_z <= float(max_local_z)
    return np.asarray(grasps)[keep], np.asarray(scores)[keep], keep, local_z


def _as_point_cloud(points: np.ndarray, *, name: str, allow_empty: bool = False) -> np.ndarray:
    point_cloud = np.asarray(points, dtype=np.float32)
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise RuntimeError(f"{name} must have shape (N, 3), got {point_cloud.shape}.")
    if not allow_empty and len(point_cloud) == 0:
        raise RuntimeError(f"{name} must contain at least one point.")
    return point_cloud


def _as_color_cloud(colors: np.ndarray | None, expected_len: int) -> np.ndarray | None:
    if colors is None:
        return None
    color_cloud = np.asarray(colors, dtype=np.uint8)
    if color_cloud.ndim != 2 or color_cloud.shape[1] != 3:
        raise RuntimeError(f"point_colors must have shape (N, 3), got {color_cloud.shape}.")
    if len(color_cloud) != expected_len:
        raise RuntimeError(
            f"point_colors length {len(color_cloud)} does not match point cloud length {expected_len}."
        )
    return color_cloud


def _translation_matrix(offset_xyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = np.asarray(offset_xyz, dtype=np.float32).reshape(3)
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = _as_point_cloud(points, name="points", allow_empty=True)
    transform = np.asarray(transform, dtype=np.float32)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return points @ rotation.T + translation.reshape(1, 3)


def sample_mesh_surface_local(mesh: Any, num_sample_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample a mesh surface and recenter it to a mesh-local frame."""
    import trimesh

    points, _ = trimesh.sample.sample_surface(mesh, num_sample_points)
    points = _as_point_cloud(points, name="mesh surface samples")
    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    if bounds.shape != (2, 3):
        raise RuntimeError(f"Unexpected mesh bounds shape: {bounds.shape}")
    reference_center = bounds.mean(axis=0)
    return points - reference_center.reshape(1, 3), reference_center


def _filter_object_point_cloud(
    object_points: np.ndarray,
    point_colors: np.ndarray | None,
    *,
    remove_outliers: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    object_points = _as_point_cloud(object_points, name="object_points")
    point_colors = _as_color_cloud(point_colors, len(object_points))

    if not remove_outliers:
        return object_points, point_colors

    import torch

    from grasp_gen.utils.point_cloud_utils import (  # type: ignore
        point_cloud_outlier_removal,
        point_cloud_outlier_removal_with_color,
    )

    points_t = torch.from_numpy(object_points)
    if point_colors is None:
        filtered_points_t, _removed_t = point_cloud_outlier_removal(points_t)
        return filtered_points_t.cpu().numpy(), None

    colors_t = torch.from_numpy(point_colors)
    filtered_points_t, _removed_t, filtered_colors_t, _ = point_cloud_outlier_removal_with_color(
        points_t,
        colors_t,
    )
    return filtered_points_t.cpu().numpy(), filtered_colors_t.cpu().numpy()


def _load_grasp_cfg(gripper_config: Path) -> Any:
    from grasp_gen.grasp_server import load_grasp_cfg

    return load_grasp_cfg(str(Path(gripper_config).expanduser().resolve()))


def _release_graspgen_sampler(sampler: Any | None) -> None:
    if sampler is not None:
        model = getattr(sampler, "model", None)
        if model is not None:
            try:
                model.cpu()
            except Exception:
                pass
            del model
        del sampler
    release_cuda_memory()


def run_graspgen_point_cloud_inference(
    object_points: np.ndarray,
    gripper_config: Path,
    *,
    grasp_threshold: float,
    num_grasps: int,
    topk_num_grasps: int,
    point_colors: np.ndarray | None = None,
    remove_outliers: bool = False,
    center_object: bool = False,
) -> RawGraspInference:
    """Run GraspGen on a point cloud and return grasps in the chosen local frame."""
    filtered_points, filtered_colors = _filter_object_point_cloud(
        object_points,
        point_colors,
        remove_outliers=remove_outliers,
    )
    if len(filtered_points) == 0:
        raise RuntimeError("Object point cloud empty after outlier removal.")

    if center_object:
        reference_center = filtered_points.mean(axis=0, dtype=np.float32)
        object_points_local = filtered_points - reference_center.reshape(1, 3)
    else:
        reference_center = np.zeros(3, dtype=np.float32)
        object_points_local = filtered_points

    cfg = _load_grasp_cfg(Path(gripper_config))
    sampler = None
    grasps_t = None
    conf_t = None
    try:
        from grasp_gen.grasp_server import GraspGenSampler

        sampler = GraspGenSampler(cfg)
        grasps_t, conf_t = GraspGenSampler.run_inference(
            object_points_local,
            sampler,
            grasp_threshold=grasp_threshold,
            num_grasps=num_grasps,
            topk_num_grasps=topk_num_grasps,
            remove_outliers=False,
        )
        if len(grasps_t) == 0:
            raise RuntimeError("No grasps returned by GraspGen inference.")

        grasps_local = grasps_t.detach().cpu().numpy()
        confidences = conf_t.detach().cpu().numpy()
        grasps_local[:, 3, 3] = 1.0
        object_to_local_transform = _translation_matrix(-reference_center)
        return RawGraspInference(
            cfg=cfg,
            object_points_local=np.asarray(object_points_local, dtype=np.float32),
            object_colors_local=None if filtered_colors is None else np.asarray(filtered_colors, dtype=np.uint8),
            object_reference_center=np.asarray(reference_center, dtype=np.float32),
            object_to_local_transform=object_to_local_transform,
            grasps_local=np.asarray(grasps_local, dtype=np.float32),
            confidences=np.asarray(confidences, dtype=np.float32),
        )
    finally:
        if grasps_t is not None:
            del grasps_t
        if conf_t is not None:
            del conf_t
        _release_graspgen_sampler(sampler)


def filter_grasps_by_collision(
    grasps_local: np.ndarray,
    scene_pc_local: np.ndarray | None,
    grasp_cfg: Any,
    *,
    collision_threshold: float,
    max_scene_points: int = 8192,
    num_collision_samples: int = 2000,
) -> CollisionFilterResult:
    """Filter local-frame grasps against a local-frame scene point cloud."""
    grasps_local = np.asarray(grasps_local, dtype=np.float32)
    if scene_pc_local is None or len(scene_pc_local) == 0:
        return CollisionFilterResult(
            collision_free_mask=np.ones(len(grasps_local), dtype=bool),
            scene_points_local=np.zeros((0, 3), dtype=np.float32),
            scene_points_used=0,
        )

    from grasp_gen.robot import get_gripper_info  # type: ignore
    from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps  # type: ignore

    scene_points_local = _as_point_cloud(
        scene_pc_local,
        name="scene_pc_local",
        allow_empty=True,
    )
    if len(scene_points_local) > max_scene_points:
        idx = np.random.choice(len(scene_points_local), max_scene_points, replace=False)
        scene_points_local = scene_points_local[idx]

    gripper_info = get_gripper_info(grasp_cfg.data.gripper_name)
    collision_free_mask = np.asarray(
        filter_colliding_grasps(
            scene_pc=scene_points_local,
            grasp_poses=grasps_local,
            gripper_collision_mesh=gripper_info.collision_mesh,
            collision_threshold=collision_threshold,
            num_collision_samples=num_collision_samples,
        ),
        dtype=bool,
    )
    return CollisionFilterResult(
        collision_free_mask=collision_free_mask,
        scene_points_local=np.asarray(scene_points_local, dtype=np.float32),
        scene_points_used=int(len(scene_points_local)),
    )


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
    raw = run_graspgen_point_cloud_inference(
        object_pc_local,
        gripper_config,
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        topk_num_grasps=topk_num_grasps,
        remove_outliers=True,
        center_object=False,
    )

    grasps, confidences = filter_by_approach_direction(raw.grasps_local, raw.confidences)
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

    collision_result = filter_grasps_by_collision(
        grasps,
        scene_pc_local,
        raw.cfg,
        collision_threshold=collision_threshold,
        max_scene_points=max_scene_points,
        num_collision_samples=num_collision_samples,
    )
    grasps = grasps[collision_result.collision_free_mask]
    confidences = confidences[collision_result.collision_free_mask]
    if len(grasps) == 0:
        raise RuntimeError("No collision-free grasps remain after obstacle filtering.")

    stats = {
        "num_grasps_after_approach_filter": total_after_approach,
        "num_grasps_after_front_filter": total_after_front,
        "num_grasps_after_pitch_filter": total_after_pitch,
        "num_grasps_after_collision_filter": int(len(grasps)),
        "num_scene_points_used_for_collision": collision_result.scene_points_used,
    }
    debug_data = {
        "all_grasps_local": all_grasps_local,
        "all_scores": all_scores,
        "all_local_z": all_local_z,
        "all_pitch_deg": all_pitch_deg,
        "collision_free_mask": collision_result.collision_free_mask,
        "collision_free_grasps_local": all_grasps_local[collision_result.collision_free_mask],
        "collision_free_scores": all_scores[collision_result.collision_free_mask],
        "collision_free_local_z": all_local_z[collision_result.collision_free_mask],
        "collision_free_pitch_deg": all_pitch_deg[collision_result.collision_free_mask],
        "object_pc_local": np.asarray(raw.object_points_local, dtype=float),
        "scene_pc_local": np.asarray(collision_result.scene_points_local, dtype=float),
        "grasp_inference_coordinate_frame": np.array("object_local"),
        "front_grasp_split_axis": np.array("object_local_z"),
        "max_grasp_local_z": np.array(float(max_local_z)),
        "grasp_pitch_convention": np.array("Rz(yaw)*Ry(pitch)*Rx(roll)"),
        "max_grasp_pitch_deg": np.array(float(max_pitch_deg)),
    }
    return grasps, confidences, stats, debug_data


def infer_grasps_from_mesh(
    mesh: Any,
    gripper_config: Path,
    grasp_threshold: float,
    num_grasps: int,
    topk_num_grasps: int,
    num_sample_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run GraspGen on mesh surface samples in a mesh-local frame."""
    mesh_points_local, _reference_center = sample_mesh_surface_local(mesh, num_sample_points)
    raw = run_graspgen_point_cloud_inference(
        mesh_points_local,
        gripper_config,
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        topk_num_grasps=topk_num_grasps,
        remove_outliers=False,
        center_object=False,
    )
    grasps, confidences = filter_by_approach_direction(raw.grasps_local, raw.confidences)
    if len(grasps) == 0:
        raise RuntimeError("No grasps remain after approach filtering.")
    return grasps, confidences


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
    """Run GraspGen and collision filtering entirely in mesh-local coordinates."""
    mesh_points_local, reference_center = sample_mesh_surface_local(mesh, num_sample_points)
    raw = run_graspgen_point_cloud_inference(
        mesh_points_local,
        gripper_config,
        grasp_threshold=grasp_threshold,
        num_grasps=num_grasps,
        topk_num_grasps=topk_num_grasps,
        remove_outliers=False,
        center_object=False,
    )
    grasps, confidences = filter_by_approach_direction(raw.grasps_local, raw.confidences)
    if len(grasps) == 0:
        raise RuntimeError("No grasps remain after approach filtering.")

    all_grasps_local = np.array(grasps, copy=True)
    all_scores = np.array(confidences, copy=True)
    total_after_approach = int(len(grasps))

    collision_result = filter_grasps_by_collision(
        grasps,
        scene_pc,
        raw.cfg,
        collision_threshold=collision_threshold,
        max_scene_points=max_scene_points,
        num_collision_samples=num_collision_samples,
    )
    grasps = grasps[collision_result.collision_free_mask]
    confidences = confidences[collision_result.collision_free_mask]
    if len(grasps) == 0:
        raise RuntimeError("No collision-free grasps remain after obstacle filtering.")

    stats = {
        "num_grasps_after_approach_filter": total_after_approach,
        "num_grasps_after_collision_filter": int(len(grasps)),
        "num_scene_points_used_for_collision": collision_result.scene_points_used,
    }
    debug_data = {
        "all_grasps": all_grasps_local,
        "all_scores": all_scores,
        "collision_free_mask": collision_result.collision_free_mask,
        "collision_free_grasps": all_grasps_local[collision_result.collision_free_mask],
        "collision_free_scores": all_scores[collision_result.collision_free_mask],
        "pc_object": np.asarray(mesh_points_local, dtype=float),
        "pc_object_raw": np.asarray(mesh_points_local, dtype=float),
        "pc_scene": np.asarray(collision_result.scene_points_local, dtype=float),
        "object_reference_center_mesh_local": np.asarray(reference_center, dtype=float),
        "saved_coordinate_frame": np.array("unity_local"),
        "collision_inference_coordinate_frame": np.array("unity_local"),
    }
    return grasps, confidences, stats, debug_data
