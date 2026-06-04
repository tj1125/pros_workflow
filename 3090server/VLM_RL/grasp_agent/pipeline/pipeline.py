from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from grasp_agent.pipeline.config import (
    load_runtime_config,
    validate_required_paths,
    validate_runtime_device,
)
from grasp_agent.pipeline.constants import AGENT_ROOT
from tool.grasp.graspgen import (
    infer_grasps_from_point_cloud_with_collision,
    prepare_graspgen_runtime_imports,
)
from tool.vision.sam import sam_segment_with_bbox
from tool.vision.yolo import extract_detections, load_yolo_model, run_yolo


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


def _save_debug_npz(
    *,
    object_id: str,
    camera_name: str,
    depth_camera_x_mirrored: bool,
    image_bgr: np.ndarray,
    depth_m: np.ndarray,
    intrinsic_matrix: np.ndarray,
    bbox: BoundingBox,
    seg_mask_bool: np.ndarray,
    object_pc_camera: np.ndarray,
    object_pc_local: np.ndarray,
    scene_pc_camera: np.ndarray,
    scene_pc_local: np.ndarray | None,
    object_reference_center_camera: np.ndarray,
    gripper_midpoint_camera_xyz: np.ndarray,
    valid_grasps_local: np.ndarray,
    valid_grasps_camera: np.ndarray,
    valid_grasp_confidences: np.ndarray,
    valid_grasp_distance_to_gripper_midpoint_m: np.ndarray,
    best_grasp_local: np.ndarray,
    best_grasp_camera: np.ndarray,
    grasp_debug_npz: dict[str, np.ndarray],
) -> Path:
    output_dir = AGENT_ROOT / "data" / "debug_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latest_grasp_debug.npz"
    gripper_midpoint_camera_xyz = np.asarray(gripper_midpoint_camera_xyz, dtype=float)
    object_reference_center_camera = np.asarray(object_reference_center_camera, dtype=float)
    gripper_midpoint_local_xyz = gripper_midpoint_camera_xyz - object_reference_center_camera
    save_data = dict(grasp_debug_npz)
    save_data.update(
        {
            "object_id": np.array(object_id),
            "camera_name": np.array(camera_name),
            "depth_camera_x_mirrored": np.array(bool(depth_camera_x_mirrored)),
            "image_bgr": np.asarray(image_bgr, dtype=np.uint8),
            "depth_m": np.asarray(depth_m, dtype=np.float32),
            "intrinsic_matrix": np.asarray(intrinsic_matrix, dtype=float),
            "bbox_xyxy": np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2], dtype=float),
            "seg_mask_bool": np.asarray(seg_mask_bool, dtype=bool),
            "object_pc_camera": np.asarray(object_pc_camera, dtype=float),
            "object_pc_local": np.asarray(object_pc_local, dtype=float),
            "scene_pc_camera": np.asarray(scene_pc_camera, dtype=float),
            "scene_pc_local": (
                np.asarray(scene_pc_local, dtype=float)
                if scene_pc_local is not None
                else np.zeros((0, 3), dtype=float)
            ),
            "object_reference_center_camera": object_reference_center_camera,
            "gripper_midpoint_camera_xyz": gripper_midpoint_camera_xyz,
            "gripper_midpoint_local_xyz": gripper_midpoint_local_xyz,
            "gripper_midpoint_coordinate_frame_camera": np.array("camera"),
            "gripper_midpoint_coordinate_frame_local": np.array("object_local"),
            "valid_grasps_local": np.asarray(valid_grasps_local, dtype=float),
            "valid_grasps_camera": np.asarray(valid_grasps_camera, dtype=float),
            "valid_grasp_confidences": np.asarray(valid_grasp_confidences, dtype=float),
            "valid_grasp_distance_to_gripper_midpoint_m": np.asarray(
                valid_grasp_distance_to_gripper_midpoint_m,
                dtype=float,
            ),
            "best_grasp_local": np.asarray(best_grasp_local, dtype=float),
            "best_grasp_camera": np.asarray(best_grasp_camera, dtype=float),
            "valid_grasp_coordinate_frame_local": np.array("object_local"),
            "valid_grasp_coordinate_frame_camera": np.array("camera"),
            "best_grasp_coordinate_frame_local": np.array("object_local"),
            "best_grasp_coordinate_frame_camera": np.array("camera"),
        }
    )
    np.savez(str(output_path), **save_data)
    return output_path


def _load_intrinsic_matrix(path: Path) -> np.ndarray:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    camera_matrix = payload.get("camera_matrix")
    if not isinstance(camera_matrix, dict) or "data" not in camera_matrix:
        raise ValueError(f"camera_matrix.data not found in {path}")
    matrix = np.array(camera_matrix["data"], dtype=float).reshape(3, 3)
    return matrix


def _decode_rgb_image(rgb_bytes: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Failed to decode RGB image bytes.")
    return image


def _decode_depth_m(depth_bytes: bytes, depth_scale: float) -> np.ndarray:
    depth = cv2.imdecode(np.frombuffer(depth_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError("Failed to decode depth image bytes.")
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) / float(depth_scale)
    if depth.dtype in (np.float32, np.float64):
        return depth.astype(np.float32)
    raise RuntimeError(f"Unsupported depth dtype: {depth.dtype}")


def _rank_grasps_by_reference_point(
    grasps_local: np.ndarray,
    grasps_camera: np.ndarray,
    confidences: np.ndarray,
    reference_point_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prioritize grasps closest to a camera-frame reference point, then by confidence."""
    reference_point_camera = np.asarray(reference_point_camera, dtype=float)
    if reference_point_camera.shape != (3,):
        raise ValueError("Reference point must contain exactly three camera-frame coordinates.")

    reference_distances = np.linalg.norm(
        np.asarray(grasps_camera, dtype=float)[:, :3, 3] - reference_point_camera[None, :],
        axis=1,
    )
    sort_idx = np.lexsort((-np.asarray(confidences, dtype=float), reference_distances))
    return (
        np.asarray(grasps_local, dtype=float)[sort_idx],
        np.asarray(grasps_camera, dtype=float)[sort_idx],
        np.asarray(confidences, dtype=float)[sort_idx],
        reference_distances[sort_idx],
    )


def _bbox_center_xy(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _depth_at_image_xy(depth_m: np.ndarray, x: float, y: float, *, window_px: int = 7) -> float | None:
    height, width = depth_m.shape[:2]
    cx = int(round(float(x)))
    cy = int(round(float(y)))
    if cx < 0 or cy < 0 or cx >= width or cy >= height:
        return None
    radius = max(0, int(window_px) // 2)
    x1 = max(0, cx - radius)
    x2 = min(width, cx + radius + 1)
    y1 = max(0, cy - radius)
    y2 = min(height, cy + radius + 1)
    patch = np.asarray(depth_m[y1:y2, x1:x2], dtype=np.float32)
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _yaw_from_amcl_pose(amcl_pose: object) -> float | None:
    if not isinstance(amcl_pose, dict):
        return None
    try:
        if amcl_pose.get("yaw") is not None:
            return float(amcl_pose["yaw"])
        if amcl_pose.get("yaw_rad") is not None:
            return float(amcl_pose["yaw_rad"])
        qx = float(amcl_pose.get("qx", 0.0))
        qy = float(amcl_pose.get("qy", 0.0))
        qz = float(amcl_pose.get("qz", 0.0))
        qw = float(amcl_pose.get("qw", 1.0))
    except (TypeError, ValueError):
        return None
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _ros_map_origin_unity_xz(origin: object) -> tuple[float, float] | None:
    try:
        if isinstance(origin, dict):
            origin_x = origin.get("x", origin.get("ros_map_origin_unity_x"))
            origin_z = origin.get("z", origin.get("ros_map_origin_unity_z"))
            return float(origin_x), float(origin_z)
        if isinstance(origin, (list, tuple)) and len(origin) >= 2:
            return float(origin[0]), float(origin[1])
    except (TypeError, ValueError):
        return None
    return None


def _target_center_world_xyz(target_center_world: object) -> np.ndarray | None:
    if not isinstance(target_center_world, (list, tuple)) or len(target_center_world) < 3:
        return None
    try:
        target = np.asarray([float(target_center_world[0]), float(target_center_world[1]), float(target_center_world[2])], dtype=float)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(target)):
        return None
    return target


def _estimate_bbox_center_unity_world(
    *,
    bbox_xyxy: tuple[float, float, float, float],
    depth_m: np.ndarray,
    intrinsic_matrix: np.ndarray,
    mirror_depth_camera_x: bool,
    amcl_pose: object,
    ros_map_origin_unity: object,
    target_center_world: np.ndarray,
) -> dict[str, object] | None:
    if not isinstance(amcl_pose, dict):
        return None
    yaw = _yaw_from_amcl_pose(amcl_pose)
    origin = _ros_map_origin_unity_xz(ros_map_origin_unity)
    if yaw is None or origin is None:
        return None
    try:
        amcl_x = float(amcl_pose["x"])
        amcl_y = float(amcl_pose["y"])
    except (KeyError, TypeError, ValueError):
        return None

    u, v = _bbox_center_xy(bbox_xyxy)
    depth_v = float(depth_m.shape[0] - 1) - float(v)
    z_camera = _depth_at_image_xy(depth_m, u, depth_v)
    if z_camera is None:
        return None
    fx = float(intrinsic_matrix[0, 0])
    fy = float(intrinsic_matrix[1, 1])
    cx = float(intrinsic_matrix[0, 2])
    cy = float(intrinsic_matrix[1, 2])
    x_camera = (float(u) - cx) * z_camera / fx
    y_camera = (float(depth_v) - cy) * z_camera / fy
    if mirror_depth_camera_x:
        x_camera *= -1.0

    forward_m = float(z_camera)
    left_m = float(-x_camera)
    map_x = amcl_x + np.cos(yaw) * forward_m - np.sin(yaw) * left_m
    map_y = amcl_y + np.sin(yaw) * forward_m + np.cos(yaw) * left_m
    origin_x, origin_z = origin
    unity_x = float(map_y + origin_x)
    unity_z = float(origin_z - map_x)
    estimated = np.asarray([unity_x, float(target_center_world[1]), unity_z], dtype=float)
    planar_distance = float(np.linalg.norm(estimated[[0, 2]] - target_center_world[[0, 2]]))
    return {
        "bbox_center_px": [float(u), float(v)],
        "bbox_center_depth_px": [float(u), float(depth_v)],
        "depth_m": float(z_camera),
        "camera_point_xyz": [float(x_camera), float(y_camera), float(z_camera)],
        "ros_map_xy": [float(map_x), float(map_y)],
        "estimated_center_world": estimated.astype(float).tolist(),
        "distance_to_target_m": planar_distance,
    }


def _select_bbox_for_target(
    *,
    yolo_result: object,
    object_id: str,
    depth_m: np.ndarray,
    intrinsic_matrix: np.ndarray,
    mirror_depth_camera_x: bool,
    target_center_world: object | None,
    target_instance_key: str,
    amcl_pose: object | None,
    ros_map_origin_unity: object | None,
) -> tuple[tuple[float, float, float, float], float, dict[str, object]]:
    target_label = str(object_id).lower()
    detections = [detection for detection in extract_detections(yolo_result, conf_threshold=0.0) if detection.label == target_label]
    if not detections:
        available_labels = sorted({detection.label for detection in extract_detections(yolo_result, conf_threshold=0.0)})
        raise ValueError(f"No YOLO detection matched '{object_id}'. Available labels: {available_labels}")

    detections = sorted(detections, key=lambda detection: float(detection.conf), reverse=True)
    target = _target_center_world_xyz(target_center_world)
    candidate_records: list[dict[str, object]] = []
    for idx, detection in enumerate(detections, 1):
        estimate = None
        if target is not None and amcl_pose is not None and ros_map_origin_unity is not None:
            estimate = _estimate_bbox_center_unity_world(
                bbox_xyxy=detection.bbox_xyxy,
                depth_m=depth_m,
                intrinsic_matrix=intrinsic_matrix,
                mirror_depth_camera_x=mirror_depth_camera_x,
                amcl_pose=amcl_pose,
                ros_map_origin_unity=ros_map_origin_unity,
                target_center_world=target,
            )
        bbox_center_x, bbox_center_y = _bbox_center_xy(detection.bbox_xyxy)
        image_center_x = (float(depth_m.shape[1]) - 1.0) * 0.5
        record = {
            "index": idx,
            "label": detection.label,
            "confidence": float(detection.conf),
            "bbox_xyxy": [float(value) for value in detection.bbox_xyxy],
            "bbox_center_px": [float(bbox_center_x), float(bbox_center_y)],
            "image_center_x_px": float(image_center_x),
            "x_distance_to_image_center_px": abs(float(bbox_center_x) - float(image_center_x)),
        }
        if estimate is not None:
            record.update(estimate)
        candidate_records.append(record)

    ranked = [record for record in candidate_records if record.get("distance_to_target_m") is not None]
    if len(detections) == 1:
        selected_record = candidate_records[0]
        selection_mode = "single_detection"
    elif ranked:
        selected_record = min(ranked, key=lambda record: (float(record["distance_to_target_m"]), -float(record["confidence"])))
        selection_mode = "closest_to_target_center"
    else:
        selected_record = min(candidate_records, key=lambda record: (float(record["x_distance_to_image_center_px"]), -float(record["confidence"])))
        selection_mode = "closest_to_image_x_center_missing_target_context"

    selected_index = int(selected_record["index"]) - 1
    selected_detection = detections[selected_index]
    target_selection = {
        "selection_mode": selection_mode,
        "target_instance_key": target_instance_key,
        "target_center_world": target.astype(float).tolist() if target is not None else [],
        "candidate_count": len(detections),
        "selected_detection_index": int(selected_record["index"]),
        "selected_bbox_xyxy": selected_record.get("bbox_xyxy", []),
        "selected_estimated_center_world": selected_record.get("estimated_center_world", []),
        "selected_distance_to_target_m": selected_record.get("distance_to_target_m"),
        "candidates": candidate_records,
    }
    return selected_detection.bbox_xyxy, float(selected_detection.conf), target_selection


def _depth_to_point_cloud(
    depth_m: np.ndarray,
    intrinsic_matrix: np.ndarray,
    mask_bool: np.ndarray,
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
    mirror_x: bool = False,
) -> np.ndarray:
    fx = float(intrinsic_matrix[0, 0])
    fy = float(intrinsic_matrix[1, 1])
    cx = float(intrinsic_matrix[0, 2])
    cy = float(intrinsic_matrix[1, 2])

    valid = (
        mask_bool
        & np.isfinite(depth_m)
        & (depth_m > float(min_depth_m))
        & (depth_m < float(max_depth_m))
    )
    v_idx, u_idx = np.nonzero(valid)
    if len(u_idx) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if stride > 1:
        keep = np.arange(0, len(u_idx), int(stride))
        u_idx = u_idx[keep]
        v_idx = v_idx[keep]

    z = depth_m[v_idx, u_idx].astype(np.float32)
    x = (u_idx.astype(np.float32) - cx) * z / fx
    if mirror_x:
        x *= -1.0
    y = (v_idx.astype(np.float32) - cy) * z / fy
    return np.column_stack([x, y, z]).astype(np.float32)


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0 or voxel_size <= 0.0:
        return points
    buckets = np.floor(points / float(voxel_size)).astype(np.int32)
    _, keep_idx = np.unique(buckets, axis=0, return_index=True)
    return points[np.sort(keep_idx)]


def _flip_mask_for_depth_alignment(mask_bool: np.ndarray) -> np.ndarray:
    """Flip the SAM mask vertically before depth backprojection."""
    return np.flip(mask_bool, axis=0).copy()


def _mirror_camera_points_x(points_camera_xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(points_camera_xyz, dtype=float).copy()
    if points.shape[-1] != 3:
        raise ValueError(f"Expected camera points with last dimension 3, got shape {points.shape}.")
    points[..., 0] *= -1.0
    return points


def _runtime_bool(value: object, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    m = rotation
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        return [0.0, 0.0, 0.0, 1.0]
    return (quat / norm).tolist()


def _serialize_grasp_pose_camera(grasp_camera: np.ndarray) -> dict[str, object]:
    grasp_camera = np.asarray(grasp_camera, dtype=float)
    return {
        "frame": "camera",
        "position": grasp_camera[:3, 3].astype(float).tolist(),
        "rotation_matrix": grasp_camera[:3, :3].astype(float).tolist(),
        "quaternion_xyzw": _rotation_matrix_to_quaternion_xyzw(grasp_camera[:3, :3]),
        "matrix_4x4": grasp_camera.astype(float).tolist(),
    }


def _serialize_valid_grasp_pose_camera(
    grasp_camera: np.ndarray,
    *,
    grasp_confidence: float,
    grasp_distance_to_gripper_midpoint_m: float,
    rank: int,
) -> dict[str, object]:
    payload = _serialize_grasp_pose_camera(grasp_camera)
    payload.update(
        {
            "rank": int(rank),
            "grasp_confidence": float(grasp_confidence),
            "grasp_distance_to_gripper_midpoint_m": float(grasp_distance_to_gripper_midpoint_m),
            "grasp_distance_to_camera_m": float(np.linalg.norm(np.asarray(grasp_camera, dtype=float)[:3, 3])),
        }
    )
    return payload


def run_pipeline(
    config_path: Path,
    object_id: str,
    camera_name: str,
    rgb_bytes: bytes,
    depth_bytes: bytes,
    target_center_world: object | None = None,
    target_instance_key: str = "",
    amcl_pose: object | None = None,
    ros_map_origin_unity: object | None = None,
) -> dict[str, object]:
    cfg, _ = load_runtime_config(config_path)
    validate_required_paths(cfg)
    device = validate_runtime_device(cfg)
    prepare_graspgen_runtime_imports(Path(cfg["models"]["graspgen_root"]))

    image_bgr = _decode_rgb_image(rgb_bytes)
    runtime = cfg["runtime"]
    depth_m = _decode_depth_m(depth_bytes, depth_scale=float(runtime["depth_scale"]))
    intrinsic_matrix = _load_intrinsic_matrix(Path(cfg["camera"]["intrinsics_path"]))
    mirror_depth_camera_x = _runtime_bool(runtime.get("mirror_depth_camera_x"), True)

    yolo_model = load_yolo_model(Path(cfg["models"]["yolo_weights"]))
    yolo_result = run_yolo(
        yolo_model,
        image_bgr,
        conf_threshold=float(runtime["yolo_conf_threshold"]),
        device=device,
    )
    bbox_xyxy, detection_confidence, target_selection = _select_bbox_for_target(
        yolo_result=yolo_result,
        object_id=object_id,
        depth_m=depth_m,
        intrinsic_matrix=intrinsic_matrix,
        mirror_depth_camera_x=mirror_depth_camera_x,
        target_center_world=target_center_world,
        target_instance_key=target_instance_key,
        amcl_pose=amcl_pose,
        ros_map_origin_unity=ros_map_origin_unity,
    )
    bbox = BoundingBox(*bbox_xyxy)

    seg_mask_bool = sam_segment_with_bbox(
        image_bgr=image_bgr,
        bbox=bbox,
        model_type=str(cfg["runtime"]["sam_model_type"]),
        checkpoint=Path(cfg["models"]["sam_seg_checkpoint"]),
        device=device,
    )
    seg_mask_bool = _flip_mask_for_depth_alignment(seg_mask_bool)

    object_pc_camera = _depth_to_point_cloud(
        depth_m=depth_m,
        intrinsic_matrix=intrinsic_matrix,
        mask_bool=seg_mask_bool,
        stride=int(runtime["object_point_stride"]),
        min_depth_m=float(runtime["min_depth_m"]),
        max_depth_m=float(runtime["max_depth_m"]),
        mirror_x=mirror_depth_camera_x,
    )
    if len(object_pc_camera) == 0:
        raise RuntimeError("Target mask produced no valid depth points.")

    object_pc_camera = _voxel_downsample(
        object_pc_camera,
        voxel_size=float(runtime["object_voxel_size_m"]),
    )
    if len(object_pc_camera) < 32:
        raise RuntimeError(f"Too few object points after filtering: {len(object_pc_camera)}")

    scene_mask = np.ones_like(seg_mask_bool, dtype=bool) & ~seg_mask_bool
    scene_pc_camera = _depth_to_point_cloud(
        depth_m=depth_m,
        intrinsic_matrix=intrinsic_matrix,
        mask_bool=scene_mask,
        stride=int(runtime["scene_point_stride"]),
        min_depth_m=float(runtime["min_depth_m"]),
        max_depth_m=float(runtime["max_depth_m"]),
        mirror_x=mirror_depth_camera_x,
    )
    scene_pc_camera = _voxel_downsample(
        scene_pc_camera,
        voxel_size=float(runtime["scene_voxel_size_m"]),
    )

    object_reference_center_camera = object_pc_camera.mean(axis=0)
    object_pc_local = object_pc_camera - object_reference_center_camera[None, :]
    scene_pc_local = scene_pc_camera - object_reference_center_camera[None, :] if len(scene_pc_camera) else None

    grasps_local, confidences, grasp_stats, grasp_debug_npz = infer_grasps_from_point_cloud_with_collision(
        object_pc_local=object_pc_local,
        gripper_config=Path(cfg["models"]["gripper_config"]),
        grasp_threshold=float(runtime["grasp_threshold"]),
        num_grasps=int(runtime["num_grasps"]),
        topk_num_grasps=int(runtime["topk_num_grasps"]),
        scene_pc_local=scene_pc_local,
        collision_threshold=float(runtime["collision_threshold"]),
        max_local_z=float(runtime.get("max_grasp_local_z", 0.0)),
        max_pitch_deg=float(runtime.get("max_grasp_pitch_deg", runtime.get("max_grasp_y_rotation_deg", 30.0))),
        max_scene_points=int(runtime["max_collision_scene_points"]),
        num_collision_samples=int(runtime["num_collision_samples"]),
    )

    valid_grasps_local = np.asarray(grasps_local, dtype=float)
    valid_grasp_confidences = np.asarray(confidences, dtype=float)
    valid_grasps_camera = np.array(valid_grasps_local, copy=True)
    valid_grasps_camera[:, :3, 3] = (
        valid_grasps_camera[:, :3, 3] + object_reference_center_camera[None, :]
    )
    gripper_midpoint_camera_xyz = np.asarray(
        runtime.get("gripper_midpoint_camera_xyz", [0.0, -0.0824, 0.023]),
        dtype=float,
    )
    if gripper_midpoint_camera_xyz.shape != (3,):
        raise ValueError("runtime.gripper_midpoint_camera_xyz must contain exactly three values.")
    if mirror_depth_camera_x:
        gripper_midpoint_camera_xyz = _mirror_camera_points_x(gripper_midpoint_camera_xyz)
    (
        valid_grasps_local,
        valid_grasps_camera,
        valid_grasp_confidences,
        valid_grasp_distance_to_gripper_midpoint_m,
    ) = _rank_grasps_by_reference_point(
        valid_grasps_local,
        valid_grasps_camera,
        valid_grasp_confidences,
        gripper_midpoint_camera_xyz,
    )

    best_idx = 0
    best_grasp_local = np.array(valid_grasps_local[best_idx], dtype=float)
    best_grasp_camera = np.array(valid_grasps_camera[best_idx], dtype=float)
    valid_grasp_poses_camera = [
        _serialize_valid_grasp_pose_camera(
            grasp_camera,
            grasp_confidence=grasp_confidence,
            grasp_distance_to_gripper_midpoint_m=grasp_distance_to_gripper_midpoint_m,
            rank=rank,
        )
        for rank, (
            grasp_camera,
            grasp_confidence,
            grasp_distance_to_gripper_midpoint_m,
        ) in enumerate(
            zip(
                valid_grasps_camera,
                valid_grasp_confidences,
                valid_grasp_distance_to_gripper_midpoint_m,
            ),
            start=1,
        )
    ]
    debug_npz_path = _save_debug_npz(
        object_id=object_id,
        camera_name=camera_name or str(cfg["camera"]["camera_name"]),
        depth_camera_x_mirrored=mirror_depth_camera_x,
        image_bgr=image_bgr,
        depth_m=depth_m,
        intrinsic_matrix=intrinsic_matrix,
        bbox=bbox,
        seg_mask_bool=seg_mask_bool,
        object_pc_camera=object_pc_camera,
        object_pc_local=object_pc_local,
        scene_pc_camera=scene_pc_camera,
        scene_pc_local=scene_pc_local,
        object_reference_center_camera=object_reference_center_camera,
        gripper_midpoint_camera_xyz=gripper_midpoint_camera_xyz,
        valid_grasps_local=valid_grasps_local,
        valid_grasps_camera=valid_grasps_camera,
        valid_grasp_confidences=valid_grasp_confidences,
        valid_grasp_distance_to_gripper_midpoint_m=valid_grasp_distance_to_gripper_midpoint_m,
        best_grasp_local=best_grasp_local,
        best_grasp_camera=best_grasp_camera,
        grasp_debug_npz=grasp_debug_npz,
    )

    object_depth_values = object_pc_camera[:, 2]
    return {
        "object_id": object_id,
        "camera_name": camera_name or str(cfg["camera"]["camera_name"]),
        "target_instance_key": target_instance_key,
        "bbox_xyxy": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
        "target_selection": target_selection,
        "mask_area_px": int(seg_mask_bool.sum()),
        "detection_confidence": float(detection_confidence),
        "grasp_confidence": float(valid_grasp_confidences[best_idx]),
        "num_valid_grasps": int(len(valid_grasps_camera)),
        "depth_camera_x_mirrored": bool(mirror_depth_camera_x),
        "gripper_midpoint_camera_xyz": gripper_midpoint_camera_xyz.astype(float).tolist(),
        "grasp_distance_to_gripper_midpoint_m": float(
            valid_grasp_distance_to_gripper_midpoint_m[best_idx]
        ),
        "grasp_distance_to_camera_m": float(np.linalg.norm(best_grasp_camera[:3, 3])),
        "num_candidate_grasps": int(len(confidences)),
        "object_reference_center_camera": object_reference_center_camera.astype(float).tolist(),
        "depth_stats_m": {
            "min": float(np.min(object_depth_values)),
            "max": float(np.max(object_depth_values)),
            "median": float(np.median(object_depth_values)),
        },
        "num_object_points": int(len(object_pc_camera)),
        "num_scene_points": int(len(scene_pc_camera)),
        "grasp_debug_npz_path": str(debug_npz_path),
        "best_grasp_pose_camera": _serialize_grasp_pose_camera(best_grasp_camera),
        "valid_grasp_poses_camera": valid_grasp_poses_camera,
        **grasp_stats,
    }
