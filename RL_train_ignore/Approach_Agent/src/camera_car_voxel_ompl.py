from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.geometry.depth_backprojection import backproject_depth_to_points, decode_depth_png_bytes
from src.geometry.voxelization import WorkspaceBounds, crop_points_to_workspace, voxelize_points
from src.io.camera_capture import capture_rgbd_snapshot
from src.io.intrinsics import load_camera_intrinsics
from src.pybullet_ompl import BoxObstacleSpec, get_reset_end_effector_position, run_ompl_planning_test
from src.pybullet_smoke import APPROACH_AGENT_ROOT, _load_yaml, _resolve_input_path, _resolve_output_path

CAMERA_TO_PYBULLET_AXIS_MAPPING = "[x, y, z] -> [z, x, -y]"
CAMERA_TO_PYBULLET_ROTATION = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)
DEFAULT_VOXEL_SIZE_M = 0.0208008


@dataclass
class CameraCarVoxelOmplConfig:
    camera_name: str
    capture_timeout_sec: float
    amcl_topic: str
    intrinsics_path: str
    debug_npz_path: str | None
    debug_scene_pointcloud_key: str
    debug_target_grasp_key: str
    planner_config_path: str
    output_root_dir: str
    voxel_size_m: float
    min_depth_m: float
    max_depth_m: float
    flip_depth_vertical: bool
    pixel_stride: int
    max_voxel_obstacles: int | None
    obstacle_rgba: tuple[float, float, float, float]
    gripper_midpoint_camera_xyz: tuple[float, float, float]
    crop_to_workspace_camera: bool
    workspace_bounds_camera: WorkspaceBounds


@dataclass
class CameraCarVoxelOmplResult:
    capture_succeeded: bool = False
    intrinsics_loaded: bool = False
    depth_decode_succeeded: bool = False
    pointcloud_generated: bool = False
    voxelization_succeeded: bool = False
    planning_invoked: bool = False
    planning_passed: bool = False
    input_source: str | None = None
    camera_name: str | None = None
    intrinsics_path: str | None = None
    debug_npz_path: str | None = None
    target_position_source: str | None = None
    target_position_camera_xyz: list[float] | None = None
    target_position_pybullet_xyz: list[float] | None = None
    target_orientation_camera_xyzw: list[float] | None = None
    target_orientation_pybullet_xyzw: list[float] | None = None
    planner_config_path: str | None = None
    output_dir: str | None = None
    captured_rgb_path: str | None = None
    captured_depth_path: str | None = None
    voxel_npz_path: str | None = None
    report_json_path: str | None = None
    rgb_format: str | None = None
    depth_format: str | None = None
    depth_shape: list[int] | None = None
    depth_flipped_vertical: bool = False
    valid_depth_point_count: int = 0
    cropped_point_count: int = 0
    voxel_count: int = 0
    voxel_size_m: float | None = None
    voxel_frame: str | None = None
    camera_to_pybullet_axis_mapping: str | None = None
    crop_to_workspace_camera: bool = False
    workspace_bounds_camera: dict[str, list[float]] | None = None
    gripper_midpoint_camera_xyz: list[float] | None = None
    gripper_midpoint_pybullet_xyz: list[float] | None = None
    ee_anchor_pybullet_xyz: list[float] | None = None
    scene_translation_xyz: list[float] | None = None
    amcl_pose_world: dict[str, Any] | None = None
    planning_result: dict[str, Any] | None = None
    failure_bucket: str | None = None
    error: str | None = None


def _vector3(value: Sequence[Any], field_name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 values.")
    return (float(value[0]), float(value[1]), float(value[2]))


def _vector4(value: Sequence[Any], field_name: str) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError(f"{field_name} must contain exactly 4 values.")
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _is_null_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "null"}
    return False


def load_camera_car_voxel_ompl_config(config_path: Path) -> CameraCarVoxelOmplConfig:
    payload = _load_yaml(config_path)
    debug_npz_path = payload.get("debug_npz_path")
    workspace_payload = payload["workspace_bounds_camera"]
    max_voxel_obstacles = payload.get("max_voxel_obstacles")
    return CameraCarVoxelOmplConfig(
        camera_name=str(payload.get("camera_name", "Camera_Car")),
        capture_timeout_sec=float(payload.get("capture_timeout_sec", 15.0)),
        amcl_topic=str(payload.get("amcl_topic", "/amcl_pose")),
        intrinsics_path=str(_resolve_input_path(str(payload["intrinsics_path"]), config_path)),
        debug_npz_path=(
            None
            if _is_null_like(debug_npz_path)
            else str(_resolve_input_path(str(debug_npz_path), config_path))
        ),
        debug_scene_pointcloud_key=str(payload.get("debug_scene_pointcloud_key", "scene_pc_camera")),
        debug_target_grasp_key=str(payload.get("debug_target_grasp_key", "best_grasp_camera")),
        planner_config_path=str(_resolve_input_path(str(payload["planner_config_path"]), config_path)),
        output_root_dir=str(_resolve_output_path(str(payload.get("output_root_dir", "outputs/camera_car_voxel_ompl")), config_path)),
        voxel_size_m=float(payload.get("voxel_size_m", DEFAULT_VOXEL_SIZE_M)),
        min_depth_m=float(payload.get("min_depth_m", 0)),
        max_depth_m=float(payload.get("max_depth_m", 3.0)),
        flip_depth_vertical=bool(payload.get("flip_depth_vertical", True)),
        pixel_stride=int(payload.get("pixel_stride", 4)),
        max_voxel_obstacles=(None if max_voxel_obstacles in {None, 0} else int(max_voxel_obstacles)),
        obstacle_rgba=_vector4(payload.get("obstacle_rgba", [0.85, 0.2, 0.2, 0.55]), "obstacle_rgba"),
        gripper_midpoint_camera_xyz=_vector3(
            payload.get("gripper_midpoint_camera_xyz", [0.0, -0.0824, 0.023]),
            "gripper_midpoint_camera_xyz",
        ),
        crop_to_workspace_camera=bool(payload.get("crop_to_workspace_camera", False)),
        workspace_bounds_camera=WorkspaceBounds(
            min_xyz=_vector3(workspace_payload["min_xyz"], "workspace_bounds_camera.min_xyz"),
            max_xyz=_vector3(workspace_payload["max_xyz"], "workspace_bounds_camera.max_xyz"),
        ),
    )


def _choose_image_suffix(image_format: str, *, fallback_suffix: str) -> str:
    lowered = image_format.lower()
    if "jpeg" in lowered or "jpg" in lowered:
        return ".jpg"
    if "png" in lowered:
        return ".png"
    return fallback_suffix


def _timestamped_output_dir(output_root_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root_dir / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_capture_artifacts(
    output_dir: Path,
    *,
    rgb_bytes: bytes,
    rgb_format: str,
    depth_bytes: bytes,
    depth_format: str,
) -> tuple[Path, Path]:
    rgb_path = output_dir / f"capture_rgb{_choose_image_suffix(rgb_format, fallback_suffix='.bin')}"
    depth_path = output_dir / f"capture_depth{_choose_image_suffix(depth_format, fallback_suffix='.png')}"
    rgb_path.write_bytes(rgb_bytes)
    depth_path.write_bytes(depth_bytes)
    return rgb_path, depth_path


def _amcl_pose_to_dict(amcl_pose: Any | None) -> dict[str, Any] | None:
    if amcl_pose is None:
        return None
    payload = {
        "stamp_sec": amcl_pose.stamp_sec,
        "position_xyz": list(amcl_pose.position_xyz),
        "orientation_xyzw": list(amcl_pose.orientation_xyzw),
    }
    covariance = getattr(amcl_pose, "covariance", None)
    if covariance is not None:
        payload["covariance"] = [float(value) for value in covariance]
    return payload


def _load_debug_npz_array(debug_npz_path: Path, array_key: str) -> np.ndarray:
    with np.load(debug_npz_path, allow_pickle=True) as payload:
        if array_key not in payload:
            available_keys = ", ".join(sorted(payload.files))
            raise KeyError(
                f"Debug NPZ key '{array_key}' was not found in {debug_npz_path}. "
                f"Available keys: {available_keys}"
            )
        return np.asarray(payload[array_key], dtype=np.float32)


def _load_points_camera_from_debug_npz(debug_npz_path: Path, pointcloud_key: str) -> np.ndarray:
    points_camera = _load_debug_npz_array(debug_npz_path, pointcloud_key)
    if points_camera.ndim != 2 or points_camera.shape[1] != 3:
        raise ValueError(
            f"Debug NPZ key '{pointcloud_key}' must be an Nx3 point cloud, got shape {points_camera.shape}."
        )
    return points_camera


def _rotation_matrix_to_quaternion_xyzw(rotation_matrix: np.ndarray) -> tuple[float, float, float, float]:
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation_matrix must have shape (3, 3), got {rotation.shape}.")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([x, y, z, w], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def _load_target_grasp_pose_camera_from_debug_npz(
    debug_npz_path: Path,
    grasp_key: str,
) -> tuple[tuple[float, float, float], np.ndarray | None]:
    grasp_value = _load_debug_npz_array(debug_npz_path, grasp_key)
    if grasp_value.shape == (4, 4):
        target_xyz = grasp_value[:3, 3]
        target_rotation = np.asarray(grasp_value[:3, :3], dtype=np.float32)
    else:
        target_xyz = grasp_value.reshape(-1)
        target_rotation = None
    if target_xyz.shape != (3,):
        raise ValueError(
            f"Debug NPZ key '{grasp_key}' must contain either a 4x4 grasp pose or 3 target coordinates, "
            f"got shape {grasp_value.shape}."
        )
    return (
        (float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])),
        target_rotation,
    )


def _estimate_scene_view(positions_xyz: np.ndarray) -> tuple[list[float], float]:
    if len(positions_xyz) == 0:
        return [0.0, 0.0, 0.3], 1.0

    min_xyz = positions_xyz.min(axis=0)
    max_xyz = positions_xyz.max(axis=0)
    center = ((min_xyz + max_xyz) * 0.5).astype(np.float32)
    extent = np.maximum(max_xyz - min_xyz, 0.05)
    radius = float(np.linalg.norm(extent) * 0.8)
    return [float(center[0]), float(center[1]), float(center[2])], max(0.9, radius)


def _filter_points_by_camera_depth(
    points_camera: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    if len(points_camera) == 0:
        return np.empty((0, 3), dtype=np.float32)

    points_camera = np.asarray(points_camera, dtype=np.float32)
    depth_z = points_camera[:, 2]
    valid_mask = (depth_z >= float(min_depth_m)) & (depth_z <= float(max_depth_m))
    return np.asarray(points_camera[valid_mask], dtype=np.float32)


def _transform_camera_points_to_pybullet_basis(points_xyz: np.ndarray) -> np.ndarray:
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if len(points_xyz) == 0:
        return points_xyz.reshape(-1, 3)

    return np.asarray(points_xyz @ CAMERA_TO_PYBULLET_ROTATION.T, dtype=np.float32)


def _transform_camera_rotation_to_pybullet_basis(rotation_matrix_camera: np.ndarray) -> np.ndarray:
    rotation_matrix_camera = np.asarray(rotation_matrix_camera, dtype=np.float32)
    if rotation_matrix_camera.shape != (3, 3):
        raise ValueError(
            f"rotation_matrix_camera must have shape (3, 3), got {rotation_matrix_camera.shape}."
        )
    return np.asarray(CAMERA_TO_PYBULLET_ROTATION @ rotation_matrix_camera, dtype=np.float32)


def _align_points_to_ee_anchor(
    points_pybullet_basis: np.ndarray,
    *,
    gripper_midpoint_pybullet_xyz: Sequence[float],
    ee_anchor_pybullet_xyz: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    points_pybullet_basis = np.asarray(points_pybullet_basis, dtype=np.float32)
    gripper_midpoint = np.asarray(gripper_midpoint_pybullet_xyz, dtype=np.float32).reshape(1, 3)
    ee_anchor = np.asarray(ee_anchor_pybullet_xyz, dtype=np.float32).reshape(1, 3)
    scene_translation = (ee_anchor - gripper_midpoint).reshape(3)
    aligned_points = points_pybullet_basis + scene_translation.reshape(1, 3)
    return np.asarray(aligned_points, dtype=np.float32), np.asarray(scene_translation, dtype=np.float32)


def run_camera_car_voxel_ompl(
    config_path: Path,
    *,
    gui_override: bool = False,
    hold_seconds_override: float | None = None,
) -> CameraCarVoxelOmplResult:
    result = CameraCarVoxelOmplResult()

    try:
        config = load_camera_car_voxel_ompl_config(config_path)
    except Exception as exc:
        result.failure_bucket = "config"
        result.error = str(exc)
        return result

    result.camera_name = config.camera_name
    result.intrinsics_path = config.intrinsics_path
    result.debug_npz_path = config.debug_npz_path
    result.planner_config_path = config.planner_config_path
    result.voxel_size_m = config.voxel_size_m
    result.voxel_frame = "pybullet_axes_ee_aligned"
    result.camera_to_pybullet_axis_mapping = CAMERA_TO_PYBULLET_AXIS_MAPPING
    result.crop_to_workspace_camera = bool(config.crop_to_workspace_camera)
    result.workspace_bounds_camera = {
        "min_xyz": list(config.workspace_bounds_camera.min_xyz),
        "max_xyz": list(config.workspace_bounds_camera.max_xyz),
    }

    output_dir = _timestamped_output_dir(Path(config.output_root_dir))
    result.output_dir = str(output_dir)

    depth_metric_m: np.ndarray | None = None
    gripper_midpoint_camera_xyz = config.gripper_midpoint_camera_xyz
    target_position_camera_xyz: tuple[float, float, float] | None = None
    target_rotation_camera: np.ndarray | None = None

    if config.debug_npz_path is not None:
        result.input_source = "debug_npz"
        try:
            points_camera = _load_points_camera_from_debug_npz(
                Path(config.debug_npz_path),
                config.debug_scene_pointcloud_key,
            )
            target_position_camera_xyz, target_rotation_camera = _load_target_grasp_pose_camera_from_debug_npz(
                Path(config.debug_npz_path),
                config.debug_target_grasp_key,
            )
        except Exception as exc:
            result.failure_bucket = "debug_npz"
            result.error = str(exc)
            return result
        points_camera = _filter_points_by_camera_depth(
            points_camera,
            min_depth_m=config.min_depth_m,
            max_depth_m=config.max_depth_m,
        )
        result.pointcloud_generated = True
    else:
        result.input_source = "ros_depth"
        try:
            snapshot = capture_rgbd_snapshot(
                config.camera_name,
                timeout_sec=config.capture_timeout_sec,
                amcl_topic=config.amcl_topic,
            )
        except Exception as exc:
            result.failure_bucket = "capture"
            result.error = str(exc)
            return result

        result.capture_succeeded = True
        result.rgb_format = snapshot.rgb_format
        result.depth_format = snapshot.depth_format
        result.amcl_pose_world = _amcl_pose_to_dict(snapshot.amcl_pose)

        try:
            rgb_path, depth_path = _save_capture_artifacts(
                output_dir,
                rgb_bytes=snapshot.rgb_bytes,
                rgb_format=snapshot.rgb_format,
                depth_bytes=snapshot.depth_bytes,
                depth_format=snapshot.depth_format,
            )
            result.captured_rgb_path = str(rgb_path)
            result.captured_depth_path = str(depth_path)
        except Exception as exc:
            result.failure_bucket = "io"
            result.error = f"Failed to save capture artifacts: {exc}"
            return result

        try:
            intrinsics = load_camera_intrinsics(Path(config.intrinsics_path))
        except Exception as exc:
            result.failure_bucket = "config"
            result.error = f"Failed to load intrinsics: {exc}"
            return result
        result.intrinsics_loaded = True

        try:
            depth_metric_m = decode_depth_png_bytes(snapshot.depth_bytes)
        except Exception as exc:
            result.failure_bucket = "depth_decode"
            result.error = str(exc)
            return result

        result.depth_decode_succeeded = True
        if config.flip_depth_vertical:
            depth_metric_m = np.flipud(depth_metric_m).copy()
            result.depth_flipped_vertical = True
        result.depth_shape = [int(depth_metric_m.shape[0]), int(depth_metric_m.shape[1])]

        try:
            points_camera = backproject_depth_to_points(
                depth_metric_m,
                intrinsics.k,
                min_depth_m=config.min_depth_m,
                max_depth_m=config.max_depth_m,
                pixel_stride=config.pixel_stride,
            )
        except Exception as exc:
            result.failure_bucket = "pointcloud"
            result.error = str(exc)
            return result
        result.pointcloud_generated = True

    result.valid_depth_point_count = int(len(points_camera))
    if len(points_camera) == 0:
        result.failure_bucket = "pointcloud"
        result.error = "Depth backprojection produced zero valid points."
        return result

    result.gripper_midpoint_camera_xyz = [float(v) for v in gripper_midpoint_camera_xyz]
    if target_position_camera_xyz is not None:
        result.target_position_source = f"debug_npz:{config.debug_target_grasp_key}"
        result.target_position_camera_xyz = [float(v) for v in target_position_camera_xyz]
    if target_rotation_camera is not None:
        result.target_orientation_camera_xyzw = list(
            _rotation_matrix_to_quaternion_xyzw(target_rotation_camera)
        )
    try:
        ee_anchor_pybullet_xyz = get_reset_end_effector_position(Path(config.planner_config_path))
    except Exception as exc:
        result.failure_bucket = "planning_scene"
        result.error = f"Failed to compute reset EE anchor in PyBullet: {exc}"
        return result
    result.ee_anchor_pybullet_xyz = [float(v) for v in ee_anchor_pybullet_xyz]

    points_camera_for_voxel = points_camera
    if config.crop_to_workspace_camera:
        points_camera_for_voxel = crop_points_to_workspace(points_camera, config.workspace_bounds_camera)
    result.cropped_point_count = int(len(points_camera_for_voxel))
    if len(points_camera_for_voxel) == 0:
        result.failure_bucket = "voxelization"
        result.error = "Camera-frame point selection produced zero points."
        return result

    gripper_midpoint_pybullet = _transform_camera_points_to_pybullet_basis(
        np.asarray(gripper_midpoint_camera_xyz, dtype=np.float32).reshape(1, 3)
    ).reshape(3)
    result.gripper_midpoint_pybullet_xyz = [float(v) for v in gripper_midpoint_pybullet]
    points_camera_pybullet_basis = _transform_camera_points_to_pybullet_basis(points_camera_for_voxel)
    points_pybullet, scene_translation = _align_points_to_ee_anchor(
        points_camera_pybullet_basis,
        gripper_midpoint_pybullet_xyz=gripper_midpoint_pybullet,
        ee_anchor_pybullet_xyz=ee_anchor_pybullet_xyz,
    )
    result.scene_translation_xyz = [float(v) for v in scene_translation]
    target_position_pybullet_xyz: np.ndarray | None = None
    target_rotation_pybullet: np.ndarray | None = None
    target_orientation_pybullet_xyzw: tuple[float, float, float, float] | None = None
    if target_position_camera_xyz is not None:
        target_position_pybullet_xyz = (
            _transform_camera_points_to_pybullet_basis(
                np.asarray(target_position_camera_xyz, dtype=np.float32).reshape(1, 3)
            ).reshape(3)
            + scene_translation
        )
        result.target_position_pybullet_xyz = [float(v) for v in target_position_pybullet_xyz]
    if target_rotation_camera is not None:
        target_rotation_pybullet = _transform_camera_rotation_to_pybullet_basis(target_rotation_camera)
        target_orientation_pybullet_xyzw = _rotation_matrix_to_quaternion_xyzw(target_rotation_pybullet)
        result.target_orientation_pybullet_xyzw = list(target_orientation_pybullet_xyzw)

    voxel_centers_pybullet = voxelize_points(
        points_pybullet,
        voxel_size_m=config.voxel_size_m,
        max_voxels=config.max_voxel_obstacles,
    )
    result.voxel_count = int(len(voxel_centers_pybullet))
    result.voxelization_succeeded = True
    if len(voxel_centers_pybullet) == 0:
        result.failure_bucket = "voxelization"
        result.error = "Voxelization produced zero occupied voxels."
        return result

    voxel_npz_path = output_dir / "voxel_snapshot.npz"
    voxel_snapshot_payload: dict[str, np.ndarray] = {
        "points_camera": points_camera.astype(np.float32),
        "points_camera_selected": points_camera_for_voxel.astype(np.float32),
        "points_camera_selected_pybullet_basis": points_camera_pybullet_basis.astype(np.float32),
        "gripper_midpoint_camera_xyz": np.asarray(gripper_midpoint_camera_xyz, dtype=np.float32),
        "gripper_midpoint_pybullet_xyz": np.asarray(gripper_midpoint_pybullet, dtype=np.float32),
        "camera_to_pybullet_rotation_matrix": CAMERA_TO_PYBULLET_ROTATION.astype(np.float32),
        "ee_anchor_pybullet_xyz": np.asarray(ee_anchor_pybullet_xyz, dtype=np.float32),
        "scene_translation_xyz": np.asarray(scene_translation, dtype=np.float32),
        "points_pybullet": points_pybullet.astype(np.float32),
        "voxel_centers_pybullet": voxel_centers_pybullet.astype(np.float32),
    }
    if target_position_camera_xyz is not None:
        voxel_snapshot_payload["target_position_camera_xyz"] = np.asarray(target_position_camera_xyz, dtype=np.float32)
    if target_position_pybullet_xyz is not None:
        voxel_snapshot_payload["target_position_pybullet_xyz"] = np.asarray(target_position_pybullet_xyz, dtype=np.float32)
    if target_rotation_camera is not None:
        voxel_snapshot_payload["target_rotation_camera_matrix"] = np.asarray(target_rotation_camera, dtype=np.float32)
    if target_rotation_pybullet is not None:
        voxel_snapshot_payload["target_rotation_pybullet_matrix"] = np.asarray(target_rotation_pybullet, dtype=np.float32)
    if depth_metric_m is not None:
        voxel_snapshot_payload["depth_metric_m"] = depth_metric_m.astype(np.float32)
    np.savez_compressed(voxel_npz_path, **voxel_snapshot_payload)
    result.voxel_npz_path = str(voxel_npz_path)

    obstacle_specs = tuple(
        BoxObstacleSpec(
            size=(config.voxel_size_m, config.voxel_size_m, config.voxel_size_m),
            position=(float(center[0]), float(center[1]), float(center[2])),
            rgba=config.obstacle_rgba,
        )
        for center in voxel_centers_pybullet
    )

    scene_view_target, scene_view_distance = _estimate_scene_view(voxel_centers_pybullet)

    planner_debug_ppm = output_dir / "planner_debug.ppm"
    planner_frames_dir = output_dir / "planner_frames"
    planning_result = run_ompl_planning_test(
        Path(config.planner_config_path),
        gui_override=gui_override,
        hold_seconds_override=hold_seconds_override,
        save_debug_ppm_override=(None if gui_override else str(planner_debug_ppm)),
        save_animation_dir_override=(None if gui_override else str(planner_frames_dir)),
        obstacle_specs_override=obstacle_specs,
        target_position_override=(
            [float(v) for v in target_position_pybullet_xyz]
            if target_position_pybullet_xyz is not None
            else None
        ),
        target_orientation_override_xyzw=(
            list(target_orientation_pybullet_xyzw)
            if target_orientation_pybullet_xyzw is not None
            else None
        ),
        render_camera_target_position_override=scene_view_target,
        render_camera_distance_override=scene_view_distance,
        anchor_marker_position_override=ee_anchor_pybullet_xyz,
    )
    result.planning_invoked = True
    result.planning_passed = bool(planning_result.planning_test_passed)
    result.planning_result = asdict(planning_result)

    report_json_path = output_dir / "camera_car_voxel_ompl_report.json"
    result.report_json_path = str(report_json_path)
    report_json_path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")

    if not planning_result.planning_test_passed:
        result.failure_bucket = planning_result.failure_bucket or "planning"
        result.error = planning_result.error or "Voxel OMPL planning failed."
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build camera-frame voxels from Camera_Car input and run PyBullet + OMPL planning."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=APPROACH_AGENT_ROOT / "configs" / "camera_car_voxel_ompl.yaml",
        help="Path to the camera-car voxel OMPL YAML config.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open PyBullet in GUI mode for manual visualization.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=None,
        help="Keep the GUI window open for a few seconds before exit.",
    )
    args = parser.parse_args(argv)

    result = run_camera_car_voxel_ompl(
        args.config.resolve(),
        gui_override=bool(args.gui),
        hold_seconds_override=args.hold_seconds,
    )
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0 if result.planning_passed else 1


def cli_entrypoint() -> None:
    exit_code = main()
    os.sys.stdout.flush()
    os.sys.stderr.flush()
    os._exit(exit_code)
