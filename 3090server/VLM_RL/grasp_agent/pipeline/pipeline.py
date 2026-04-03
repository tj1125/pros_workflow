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
from tool.vision.yolo import load_yolo_model, run_yolo, select_best_bbox


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
    best_grasp_local: np.ndarray,
    best_grasp_camera: np.ndarray,
    grasp_debug_npz: dict[str, np.ndarray],
) -> Path:
    output_dir = AGENT_ROOT / "data" / "debug_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latest_grasp_debug.npz"
    save_data = dict(grasp_debug_npz)
    save_data.update(
        {
            "object_id": np.array(object_id),
            "camera_name": np.array(camera_name),
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
            "object_reference_center_camera": np.asarray(object_reference_center_camera, dtype=float),
            "best_grasp_local": np.asarray(best_grasp_local, dtype=float),
            "best_grasp_camera": np.asarray(best_grasp_camera, dtype=float),
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


def _depth_to_point_cloud(
    depth_m: np.ndarray,
    intrinsic_matrix: np.ndarray,
    mask_bool: np.ndarray,
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
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
    y = (v_idx.astype(np.float32) - cy) * z / fy
    return np.column_stack([x, y, z]).astype(np.float32)


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0 or voxel_size <= 0.0:
        return points
    buckets = np.floor(points / float(voxel_size)).astype(np.int32)
    _, keep_idx = np.unique(buckets, axis=0, return_index=True)
    return points[np.sort(keep_idx)]


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


def run_pipeline(
    config_path: Path,
    object_id: str,
    camera_name: str,
    rgb_bytes: bytes,
    depth_bytes: bytes,
) -> dict[str, object]:
    cfg, _ = load_runtime_config(config_path)
    validate_required_paths(cfg)
    device = validate_runtime_device(cfg)
    prepare_graspgen_runtime_imports(Path(cfg["models"]["graspgen_root"]))

    image_bgr = _decode_rgb_image(rgb_bytes)
    depth_m = _decode_depth_m(depth_bytes, depth_scale=float(cfg["runtime"]["depth_scale"]))
    intrinsic_matrix = _load_intrinsic_matrix(Path(cfg["camera"]["intrinsics_path"]))

    yolo_model = load_yolo_model(Path(cfg["models"]["yolo_weights"]))
    yolo_result = run_yolo(
        yolo_model,
        image_bgr,
        conf_threshold=float(cfg["runtime"]["yolo_conf_threshold"]),
        device=device,
    )
    bbox_xyxy, detection_confidence = select_best_bbox(yolo_result, object_id)
    bbox = BoundingBox(*bbox_xyxy)

    seg_mask_bool = sam_segment_with_bbox(
        image_bgr=image_bgr,
        bbox=bbox,
        model_type=str(cfg["runtime"]["sam_model_type"]),
        checkpoint=Path(cfg["models"]["sam_seg_checkpoint"]),
        device=device,
    )

    runtime = cfg["runtime"]
    object_pc_camera = _depth_to_point_cloud(
        depth_m=depth_m,
        intrinsic_matrix=intrinsic_matrix,
        mask_bool=seg_mask_bool,
        stride=int(runtime["object_point_stride"]),
        min_depth_m=float(runtime["min_depth_m"]),
        max_depth_m=float(runtime["max_depth_m"]),
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
        max_scene_points=int(runtime["max_collision_scene_points"]),
        num_collision_samples=int(runtime["num_collision_samples"]),
    )

    best_idx = int(np.argmax(confidences))
    best_grasp_local = np.array(grasps_local[best_idx], dtype=float)
    best_grasp_camera = np.array(best_grasp_local, copy=True)
    best_grasp_camera[:3, 3] = best_grasp_camera[:3, 3] + object_reference_center_camera
    debug_npz_path = _save_debug_npz(
        object_id=object_id,
        camera_name=camera_name or str(cfg["camera"]["camera_name"]),
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
        best_grasp_local=best_grasp_local,
        best_grasp_camera=best_grasp_camera,
        grasp_debug_npz=grasp_debug_npz,
    )

    object_depth_values = object_pc_camera[:, 2]
    return {
        "object_id": object_id,
        "camera_name": camera_name or str(cfg["camera"]["camera_name"]),
        "bbox_xyxy": [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
        "mask_area_px": int(seg_mask_bool.sum()),
        "detection_confidence": float(detection_confidence),
        "grasp_confidence": float(confidences[best_idx]),
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
        "best_grasp_pose_camera": {
            "frame": "camera",
            "position": best_grasp_camera[:3, 3].astype(float).tolist(),
            "rotation_matrix": best_grasp_camera[:3, :3].astype(float).tolist(),
            "quaternion_xyzw": _rotation_matrix_to_quaternion_xyzw(best_grasp_camera[:3, :3]),
            "matrix_4x4": best_grasp_camera.astype(float).tolist(),
        },
        **grasp_stats,
    }
