from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from get_item_info_agent_no_sam3d.pipeline.config import (
    load_scene_config,
    prepare_runtime_imports,
    validate_required_paths,
    validate_runtime_device,
)
from get_item_info_agent_no_sam3d.pipeline.steps.goal_pose import compute_goal_pose
from get_item_info_agent_no_sam3d.pipeline.constants import AGENT_ROOT
from get_item_info_agent_no_sam3d.pipeline.steps.topic_input import (
    canonical_camera_name,
    normalize_label,
    parse_world_position_data,
)
from get_item_info_agent_no_sam3d.pipeline.types import BoundingBox, TopicObservation, WorldPositionObject
from tool.grasp.graspgen import (
    filter_by_approach_direction,
    filter_grasps_by_collision,
    run_graspgen_point_cloud_inference,
    transform_points,
)
from tool.runtime.memory import release_cuda_memory

LABEL_COLORS = {
    "apple": np.array([255, 140, 40], dtype=np.uint8),
    "box": np.array([150, 95, 45], dtype=np.uint8),
    "coffee": np.array([245, 245, 235], dtype=np.uint8),
    "cup": np.array([120, 80, 45], dtype=np.uint8),
    "doll": np.array([60, 220, 90], dtype=np.uint8),
    "gaobear": np.array([35, 35, 35], dtype=np.uint8),
    "hpb": np.array([25, 80, 160], dtype=np.uint8),
    "xbox": np.array([90, 90, 90], dtype=np.uint8),
}
HEIGHT_WEIGHT_RATIO_ANCHORS = np.array([0.70, 1.03, 2.28], dtype=np.float32)
HEIGHT_WEIGHT_VALUE_ANCHORS = np.array([0.65, 0.80, 0.90], dtype=np.float32)

def _save_visualization_npz(
    center_world: np.ndarray,
    debug_npz: dict[str, np.ndarray],
    primary_camera_id: str,
    target_label: str,
    objects: list[dict[str, object]],
) -> Path:
    output_dir = AGENT_ROOT / "data" / "debug_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latest_grasp_visualization.npz"
    save_data = dict(debug_npz)
    save_data["center_world"] = np.asarray(center_world, dtype=float)
    save_data["center_world_coordinate_frame"] = np.array("unity_world")
    save_data["primary_camera_id"] = np.array(primary_camera_id)
    save_data["target_label"] = np.array(target_label)
    save_data["objects_json"] = np.array(json.dumps(objects))
    np.savez(str(output_path), **save_data)
    return output_path


def _canonicalize_image_paths(image_paths_by_camera: dict[str, Path]) -> dict[str, Path]:
    canonical: dict[str, Path] = {}
    for camera_name, path in image_paths_by_camera.items():
        canonical[canonical_camera_name(camera_name)] = Path(path).expanduser().resolve()
    return canonical


def _normalize_selected_camera(selected_camera: str | None) -> str | None:
    if not selected_camera:
        return None
    text = str(selected_camera).strip()
    return canonical_camera_name(text) if text else None


def _aligned_world_to_cv(point_world_unity: np.ndarray) -> np.ndarray:
    point_world_cv = np.asarray(point_world_unity, dtype=np.float32).copy()
    if point_world_cv.ndim == 1:
        point_world_cv = point_world_cv.reshape(3)
        point_world_cv[2] *= -1.0
        return point_world_cv
    if point_world_cv.ndim == 2 and point_world_cv.shape[1] == 3:
        point_world_cv[:, 2] *= -1.0
        return point_world_cv
    raise ValueError(f"Expected shape (3,) or (N, 3), got {point_world_cv.shape}")


def _cv_world_to_unity(point_world_cv: np.ndarray) -> np.ndarray:
    point_world_unity = np.asarray(point_world_cv, dtype=np.float32).copy()
    if point_world_unity.ndim == 1:
        point_world_unity = point_world_unity.reshape(3)
        point_world_unity[2] *= -1.0
        return point_world_unity
    if point_world_unity.ndim == 2 and point_world_unity.shape[1] == 3:
        point_world_unity[:, 2] *= -1.0
        return point_world_unity
    raise ValueError(f"Expected shape (3,) or (N, 3), got {point_world_unity.shape}")


def _resolve_extrinsics_dir(camera_parameter_dir: Path) -> Path:
    candidate = camera_parameter_dir / "Extrinsic"
    if candidate.exists():
        return candidate
    candidate = camera_parameter_dir / "Extrinsic_orig"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"No Extrinsic directory found under {camera_parameter_dir}")


def _load_intrinsic_matrix(camera_id: str, camera_parameter_dir: Path) -> np.ndarray:
    import yaml

    path = camera_parameter_dir / "Intrinsics" / f"{camera_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    matrix = payload.get("camera_matrix") or payload.get("intrinsic_matrix") or payload.get("K")
    if matrix is None:
        raise ValueError(f"Intrinsic matrix key not found in {path}")
    data = matrix["data"] if isinstance(matrix, dict) and "data" in matrix else matrix
    return np.array(data, dtype=float).reshape(3, 3)


def _load_extrinsic_matrix(camera_id: str, camera_parameter_dir: Path) -> np.ndarray:
    import pickle

    extrinsics_dir = _resolve_extrinsics_dir(camera_parameter_dir)
    pkl_path = extrinsics_dir / f"{camera_id}.pkl"
    pkl_path_lower = extrinsics_dir / f"{camera_id.lower()}.pkl"
    if pkl_path.exists():
        with pkl_path.open("rb") as handle:
            payload = pickle.load(handle)
    elif pkl_path_lower.exists():
        with pkl_path_lower.open("rb") as handle:
            payload = pickle.load(handle)
    else:
        json_dir = extrinsics_dir / "json"
        json_matches = sorted(json_dir.glob(f"{camera_id}_*.json")) if json_dir.exists() else []
        if not json_matches and json_dir.exists():
            json_matches = sorted(json_dir.glob(f"{camera_id.lower()}_*.json"))
        if not json_matches:
            raise FileNotFoundError(f"Extrinsic file not found for {camera_id} in {extrinsics_dir}")
        payload = json.loads(json_matches[0].read_text(encoding="utf-8"))

    extrinsic = payload.get("extrinsic_matrix")
    if extrinsic is None:
        rotation = np.array(payload["rotation_matrix"], dtype=float)
        translation = np.array(payload["translation_vector"], dtype=float).reshape(3, 1)
        extrinsic = np.hstack([rotation, translation])
    extrinsic = np.array(extrinsic, dtype=float)
    if extrinsic.shape == (4, 4):
        extrinsic = extrinsic[:3, :]
    if extrinsic.shape != (3, 4):
        raise ValueError(f"Unexpected extrinsic shape for {camera_id}: {extrinsic.shape}")
    return extrinsic


def _camera_center_from_projection(proj: np.ndarray) -> np.ndarray:
    return -np.linalg.inv(proj[:, :3]) @ proj[:, 3]


def _load_camera_models(camera_ids: list[str], camera_parameter_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    models: dict[str, dict[str, np.ndarray]] = {}
    for camera_id in camera_ids:
        k = _load_intrinsic_matrix(camera_id, camera_parameter_dir)
        extrinsic = _load_extrinsic_matrix(camera_id, camera_parameter_dir)
        projection = k @ extrinsic
        models[camera_id] = {
            "K": k,
            "extrinsic": extrinsic,
            "P": projection,
            "camera_center": _camera_center_from_projection(projection),
        }
    return models


def _camera_to_world_cv(extrinsic: np.ndarray, points_cam: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    inv_rotation = np.linalg.inv(rotation)
    return (np.asarray(points_cam, dtype=np.float32) - translation.reshape(1, 3)) @ inv_rotation.T


def _project_point(projection: np.ndarray, point_world_cv: np.ndarray) -> np.ndarray:
    point_h = np.append(np.asarray(point_world_cv, dtype=np.float64).reshape(3), 1.0)
    uvw = np.asarray(projection, dtype=np.float64) @ point_h
    if abs(float(uvw[2])) < 1e-12:
        raise RuntimeError("Point projects to infinity.")
    return np.array([uvw[0] / uvw[2], uvw[1] / uvw[2]], dtype=np.float32)


def _ray_direction_from_projection(proj: np.ndarray, uv: np.ndarray) -> np.ndarray:
    inv_proj = np.linalg.inv(np.asarray(proj[:, :3], dtype=np.float64))
    direction = inv_proj @ np.array([float(uv[0]), float(uv[1]), 1.0], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    return direction.astype(np.float32)


def _normalize_vec(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        raise RuntimeError("Degenerate vector encountered.")
    return vec / norm


WORLD_UP_CV = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _visible_horizontal_direction(center_world_cv: np.ndarray, camera_center_world_cv: np.ndarray) -> np.ndarray:
    view = np.asarray(camera_center_world_cv, dtype=np.float32) - np.asarray(center_world_cv, dtype=np.float32)
    view[1] = 0.0
    view = _normalize_vec(view)
    side = np.cross(WORLD_UP_CV, view).astype(np.float32)
    return _normalize_vec(side)


def _local_image_axes(
    projection: np.ndarray,
    center_world_cv: np.ndarray,
    camera_center_world_cv: np.ndarray,
    axis_step_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_uv = _project_point(projection, center_world_cv)
    up_uv = _project_point(projection, np.asarray(center_world_cv, dtype=np.float32) + WORLD_UP_CV * axis_step_m)
    axis_y_img = _normalize_vec(up_uv - center_uv)

    side_world = _visible_horizontal_direction(center_world_cv, camera_center_world_cv)
    side_uv = _project_point(projection, np.asarray(center_world_cv, dtype=np.float32) + side_world * axis_step_m)
    axis_x_img = _normalize_vec(side_uv - center_uv)
    return center_uv, axis_x_img, axis_y_img


def _robust_axis_extent(
    mask: np.ndarray,
    center_uv: np.ndarray,
    axis_img: np.ndarray,
    lower: float,
    upper: float,
) -> float:
    ys, xs = np.where(mask)
    if len(xs) < 8:
        raise RuntimeError("Mask too small to measure.")
    offsets = np.column_stack(
        [
            xs.astype(np.float32) - float(center_uv[0]),
            ys.astype(np.float32) - float(center_uv[1]),
        ]
    )
    signed = (offsets @ np.asarray(axis_img, dtype=np.float32).reshape(2, 1)).reshape(-1)
    lo = float(np.percentile(signed, lower))
    hi = float(np.percentile(signed, upper))
    return max(0.0, hi - lo)


def _measure_camera_width_from_mask(
    mask: np.ndarray,
    center_uv: np.ndarray,
    axis_x_img: np.ndarray,
    axis_y_img: np.ndarray,
    object_height_m: float,
    lower: float,
    upper: float,
) -> dict[str, float]:
    height_px = _robust_axis_extent(mask, center_uv, axis_y_img, lower, upper)
    width_px = _robust_axis_extent(mask, center_uv, axis_x_img, lower, upper)
    if height_px <= 1e-6:
        raise RuntimeError("Measured SAM height is degenerate.")
    width_m = float(object_height_m * (width_px / height_px))
    return {
        "mask_height_px": float(height_px),
        "mask_width_px": float(width_px),
        "width_m_from_height_ratio": width_m,
    }


def _fit_vertical_line_xz(
    obj: WorldPositionObject,
    camera_models: dict[str, dict[str, np.ndarray]],
    eps: float = 1e-6,
) -> np.ndarray:
    eye = np.eye(2)
    a = np.zeros((2, 2), dtype=float)
    b = np.zeros(2, dtype=float)
    fallback_center_world_cv = _aligned_world_to_cv(obj.center_world_unity)
    for camera_id, obs in obj.observations.items():
        if camera_id not in camera_models:
            continue
        projection = camera_models[camera_id]["P"]
        camera_center_xz = _camera_center_from_projection(projection)[[0, 2]]
        ray_xz = _ray_direction_from_projection(projection, obs.bbox.center)[[0, 2]]
        norm_xz = float(np.linalg.norm(ray_xz))
        if norm_xz <= eps:
            continue
        ray_xz /= norm_xz
        m = eye - np.outer(ray_xz, ray_xz)
        a += m
        b += m @ camera_center_xz
    if np.linalg.matrix_rank(a) < 2 or np.linalg.cond(a) > 1 / eps:
        return fallback_center_world_cv[[0, 2]].copy()
    return np.linalg.solve(a, b)


def _project_pixel_to_vertical_line(
    projection: np.ndarray,
    uv: np.ndarray,
    vertical_line_xz: np.ndarray,
    eps: float = 1e-8,
) -> dict[str, object] | None:
    camera_center = _camera_center_from_projection(projection)
    ray_direction = _ray_direction_from_projection(projection, uv)
    denom = float(ray_direction[0] ** 2 + ray_direction[2] ** 2)
    if denom <= eps:
        return None
    tau = (
        (vertical_line_xz[0] - camera_center[0]) * ray_direction[0]
        + (vertical_line_xz[1] - camera_center[2]) * ray_direction[2]
    ) / denom
    point_on_ray = camera_center + tau * ray_direction
    point_on_vertical_line = np.array(
        [vertical_line_xz[0], point_on_ray[1], vertical_line_xz[1]],
        dtype=float,
    )
    residual_xz_m = float(np.linalg.norm(point_on_ray[[0, 2]] - vertical_line_xz))
    return {
        "tau": float(tau),
        "point_on_ray_world_cv": point_on_ray,
        "point_on_vertical_line_world_cv": point_on_vertical_line,
        "y_world_cv": float(point_on_vertical_line[1]),
        "residual_xz_m": residual_xz_m,
    }


def _estimate_height_on_vertical_line(
    obj: WorldPositionObject,
    camera_models: dict[str, dict[str, np.ndarray]],
) -> dict[str, object]:
    vertical_line_xz = _fit_vertical_line_xz(obj, camera_models)
    top_y_by_camera: dict[str, float] = {}
    bottom_y_by_camera: dict[str, float] = {}
    height_by_camera: dict[str, float] = {}
    for camera_id, obs in obj.observations.items():
        if camera_id not in camera_models:
            continue
        projection = camera_models[camera_id]["P"]
        bbox = obs.bbox
        top_uv = np.array([(bbox.x1 + bbox.x2) * 0.5, bbox.y1], dtype=np.float32)
        bottom_uv = np.array([(bbox.x1 + bbox.x2) * 0.5, bbox.y2], dtype=np.float32)
        center_fit = _project_pixel_to_vertical_line(projection, bbox.center, vertical_line_xz)
        top_fit = _project_pixel_to_vertical_line(projection, top_uv, vertical_line_xz)
        bottom_fit = _project_pixel_to_vertical_line(projection, bottom_uv, vertical_line_xz)
        if center_fit is None or top_fit is None or bottom_fit is None:
            continue
        top_y_by_camera[camera_id] = float(top_fit["y_world_cv"])
        bottom_y_by_camera[camera_id] = float(bottom_fit["y_world_cv"])
        height_by_camera[camera_id] = float(top_fit["y_world_cv"] - bottom_fit["y_world_cv"])

    if not height_by_camera:
        raise RuntimeError(f"Vertical-line height estimation failed for {obj.label}.")

    top_y_values = np.asarray(list(top_y_by_camera.values()), dtype=float)
    bottom_y_values = np.asarray(list(bottom_y_by_camera.values()), dtype=float)
    height_values = np.asarray(list(height_by_camera.values()), dtype=float)
    top_y_median = float(np.median(top_y_values))
    bottom_y_median = float(np.median(bottom_y_values))
    return {
        "vertical_line_xz_world_cv": np.array(vertical_line_xz, dtype=float),
        "top_y_by_camera": top_y_by_camera,
        "bottom_y_by_camera": bottom_y_by_camera,
        "height_by_camera": height_by_camera,
        "top_world_cv_median": np.array([vertical_line_xz[0], top_y_median, vertical_line_xz[1]], dtype=float),
        "bottom_world_cv_median": np.array([vertical_line_xz[0], bottom_y_median, vertical_line_xz[1]], dtype=float),
        "height_median": float(np.median(height_values)),
        "height_min": float(np.min(height_values)),
    }


def _height_estimates_from_bbox(
    obj: WorldPositionObject,
    camera_models: dict[str, dict[str, np.ndarray]],
) -> dict[str, float]:
    estimates: dict[str, float] = {}
    center_world_cv = _aligned_world_to_cv(obj.center_world_unity)
    for camera_id, obs in obj.observations.items():
        if camera_id not in camera_models:
            continue
        k = camera_models[camera_id]["K"]
        extrinsic = camera_models[camera_id]["extrinsic"]
        point_cam = extrinsic[:, :3] @ center_world_cv + extrinsic[:, 3]
        z_cam = float(point_cam[2])
        if z_cam <= 0:
            continue
        estimates[camera_id] = float(obs.bbox.height * z_cam / float(k[1, 1]))
    return estimates


def _choose_final_height_m(
    bbox_height_estimates: dict[str, float],
    vertical_line_height_m: float | None,
    min_valid_height_m: float,
    max_valid_height_m: float,
) -> float:
    candidates = list(bbox_height_estimates.values())
    if vertical_line_height_m is not None and min_valid_height_m <= vertical_line_height_m <= max_valid_height_m:
        candidates.append(vertical_line_height_m)
    if not candidates:
        raise RuntimeError("No valid height estimate could be computed.")
    return float(np.median(np.asarray(candidates, dtype=float)))


def _shape_ratio_height_weight(height_m: float, size_x_m: float, size_z_m: float) -> tuple[float, float]:
    footprint_sum = max(float(size_x_m) + float(size_z_m), 1e-6)
    ratio = float(height_m / footprint_sum)
    weight = float(
        np.interp(
            ratio,
            HEIGHT_WEIGHT_RATIO_ANCHORS.astype(np.float64),
            HEIGHT_WEIGHT_VALUE_ANCHORS.astype(np.float64),
            left=float(HEIGHT_WEIGHT_VALUE_ANCHORS[0]),
            right=float(HEIGHT_WEIGHT_VALUE_ANCHORS[-1]),
        )
    )
    return ratio, weight


def _camera_azimuth_about_object(center_world_cv: np.ndarray, camera_center_world_cv: np.ndarray) -> float:
    dx = float(camera_center_world_cv[0] - center_world_cv[0])
    dz = float(camera_center_world_cv[2] - center_world_cv[2])
    return float(math.atan2(dz, dx))


def _fit_rotated_rectangle_footprint(
    center_world_cv: np.ndarray,
    width_by_camera: dict[str, float],
    camera_models: dict[str, dict[str, np.ndarray]],
    max_aspect_ratio: float = 1.8,
) -> dict[str, object]:
    camera_ids = [camera_id for camera_id in width_by_camera if camera_id in camera_models]
    if not camera_ids:
        raise RuntimeError("No valid multi-camera width estimates available.")

    observed_widths = np.asarray([width_by_camera[camera_id] for camera_id in camera_ids], dtype=np.float64)
    azimuths = np.asarray(
        [
            _camera_azimuth_about_object(center_world_cv, camera_models[camera_id]["camera_center"])
            for camera_id in camera_ids
        ],
        dtype=np.float64,
    )
    median_width = float(np.median(observed_widths))
    if len(camera_ids) == 1:
        return {
            "size_x_m": median_width,
            "size_z_m": median_width,
            "yaw_rad": 0.0,
            "fit_rmse_m": 0.0,
            "max_aspect_ratio": float(max_aspect_ratio),
            "predicted_width_by_camera": {camera_ids[0]: median_width},
            "azimuth_rad_by_camera": {camera_ids[0]: float(azimuths[0])},
        }

    best_result: dict[str, object] | None = None
    phi_values = np.linspace(0.0, math.pi, 721, dtype=np.float64)
    min_size = max(0.02, 0.25 * median_width)
    max_aspect_ratio = max(float(max_aspect_ratio), 1.0)
    for phi in phi_values:
        basis = np.column_stack(
            [
                np.abs(np.sin(azimuths - phi)),
                np.abs(np.cos(azimuths - phi)),
            ]
        )
        solution, *_ = np.linalg.lstsq(basis, observed_widths, rcond=None)
        size_x = max(float(solution[0]), min_size)
        size_z = max(float(solution[1]), min_size)
        long_side = max(size_x, size_z)
        short_side = min(size_x, size_z)
        if short_side > 1e-6 and long_side / short_side > max_aspect_ratio:
            clamped_short = max((size_x + size_z) / (1.0 + max_aspect_ratio), min_size)
            clamped_long = max_aspect_ratio * clamped_short
            if size_x >= size_z:
                size_x, size_z = clamped_long, clamped_short
            else:
                size_x, size_z = clamped_short, clamped_long
        predicted = basis @ np.array([size_x, size_z], dtype=np.float64)
        rmse = float(np.sqrt(np.mean((predicted - observed_widths) ** 2)))
        if best_result is None or rmse < float(best_result["fit_rmse_m"]):
            best_result = {
                "size_x_m": size_x,
                "size_z_m": size_z,
                "yaw_rad": float(phi),
                "fit_rmse_m": rmse,
                "max_aspect_ratio": float(max_aspect_ratio),
                "predicted_width_by_camera": {
                    camera_id: float(width_value)
                    for camera_id, width_value in zip(camera_ids, predicted.tolist())
                },
                "azimuth_rad_by_camera": {
                    camera_id: float(azimuth_value)
                    for camera_id, azimuth_value in zip(camera_ids, azimuths.tolist())
                },
            }
    if best_result is None:
        raise RuntimeError("Failed to fit object footprint.")
    return best_result


def _build_sam_predictor(model_type: str, checkpoint: Path, device: str):
    from segment_anything import SamPredictor, sam_model_registry  # type: ignore

    if model_type not in sam_model_registry:
        raise ValueError(f"Unsupported SAM model type: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam = sam.to(device=device)
    predictor = SamPredictor(sam)
    return sam, predictor


def _clamp_bbox(bbox: BoundingBox, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    left = max(0, min(w, int(math.floor(bbox.x1))))
    top = max(0, min(h, int(math.floor(bbox.y1))))
    right = max(0, min(w, int(math.ceil(bbox.x2))))
    bottom = max(0, min(h, int(math.ceil(bbox.y2))))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid bbox after clamp: {(left, top, right, bottom)}")
    return left, top, right, bottom


def _bbox_mask(image_shape: tuple[int, int, int], bbox: BoundingBox) -> np.ndarray:
    left, top, right, bottom = _clamp_bbox(bbox, image_shape)
    mask = np.zeros(image_shape[:2], dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def _segment_objects_with_sam(
    cfg: dict[str, Any],
    image_paths_by_camera: dict[str, Path],
    objects: list[WorldPositionObject],
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    images_by_camera: dict[str, np.ndarray] = {}
    masks_by_object: dict[str, dict[str, np.ndarray]] = {}
    observations_by_camera: dict[str, list[tuple[str, TopicObservation]]] = {}

    for obj in objects:
        object_key = _object_key(obj)
        masks_by_object[object_key] = {}
        for camera_id, obs in obj.observations.items():
            if camera_id not in image_paths_by_camera:
                continue
            observations_by_camera.setdefault(camera_id, []).append((object_key, obs))

    sam = None
    predictor = None
    total_start = time.perf_counter()
    sam_model_load_s = 0.0
    per_camera_sam_inference_s: dict[str, float] = {}
    num_sam_predictions = 0
    try:
        model_load_start = time.perf_counter()
        sam, predictor = _build_sam_predictor(
            str(cfg["runtime"]["sam_model_type"]),
            Path(cfg["models"]["sam_seg_checkpoint"]),
            device,
        )
        sam_model_load_s = float(time.perf_counter() - model_load_start)
        for camera_id, object_observations in observations_by_camera.items():
            camera_start = time.perf_counter()
            image_bgr = cv2.imread(str(image_paths_by_camera[camera_id]), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Failed to read image: {image_paths_by_camera[camera_id]}")
            images_by_camera[camera_id] = image_bgr
            predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            for object_key, obs in object_observations:
                box = np.array([obs.bbox.x1, obs.bbox.y1, obs.bbox.x2, obs.bbox.y2], dtype=np.float32)
                try:
                    masks, scores, _ = predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box[None, :],
                        multimask_output=True,
                    )
                    if masks is None or len(masks) == 0:
                        raise RuntimeError("SAM returned no mask.")
                    best_idx = int(np.argmax(scores))
                    mask = masks[best_idx].astype(bool)
                    num_sam_predictions += 1
                except Exception:
                    mask = _bbox_mask(image_bgr.shape, obs.bbox)
                masks_by_object[object_key][camera_id] = mask
            per_camera_sam_inference_s[camera_id] = float(time.perf_counter() - camera_start)
    finally:
        if predictor is not None:
            del predictor
        if sam is not None:
            try:
                sam.cpu()
            except Exception:
                pass
            del sam
        release_cuda_memory()

    return images_by_camera, masks_by_object, {
        "sam_model_load_s": sam_model_load_s,
        "sam_total_s": float(time.perf_counter() - total_start),
        "sam_inference_total_s": float(sum(per_camera_sam_inference_s.values())),
        "sam_inference_by_camera_s": per_camera_sam_inference_s,
        "num_sam_predictions": int(num_sam_predictions),
    }


def _estimate_object_size_from_masks(
    obj: WorldPositionObject,
    camera_models: dict[str, dict[str, np.ndarray]],
    masks_by_object: dict[str, dict[str, np.ndarray]],
    runtime_cfg: dict[str, Any],
) -> dict[str, Any]:
    center_world_cv = _aligned_world_to_cv(obj.center_world_unity)
    center_world_unity = np.asarray(obj.center_world_unity, dtype=np.float32)
    vertical_line_fit = _estimate_height_on_vertical_line(obj, camera_models)
    vertical_line_height_m = (
        float(vertical_line_fit["height_min"])
        if str(runtime_cfg.get("vertical_line_height_stat", "min")) == "min"
        else float(vertical_line_fit["height_median"])
    )
    bbox_height_estimates = _height_estimates_from_bbox(obj, camera_models)
    base_height_m = _choose_final_height_m(
        bbox_height_estimates=bbox_height_estimates,
        vertical_line_height_m=vertical_line_height_m,
        min_valid_height_m=float(runtime_cfg.get("min_valid_height_m", 0.03)),
        max_valid_height_m=float(runtime_cfg.get("max_valid_height_m", 1.50)),
    )
    width_by_camera: dict[str, float] = {}
    per_camera_measurements: dict[str, dict[str, float | list[float]]] = {}
    object_key = _object_key(obj)

    for camera_id, obs in obj.observations.items():
        if camera_id not in camera_models or camera_id not in masks_by_object[object_key]:
            continue
        mask = masks_by_object[object_key][camera_id]
        center_uv, axis_x_img, axis_y_img = _local_image_axes(
            camera_models[camera_id]["P"],
            center_world_cv,
            camera_models[camera_id]["camera_center"],
            axis_step_m=float(runtime_cfg.get("axis_step_m", 0.05)),
        )
        measurement = _measure_camera_width_from_mask(
            mask,
            center_uv,
            axis_x_img,
            axis_y_img,
            object_height_m=float(base_height_m),
            lower=float(runtime_cfg.get("robust_lower_percentile", 5.0)),
            upper=float(runtime_cfg.get("robust_upper_percentile", 95.0)),
        )
        width_by_camera[camera_id] = float(measurement["width_m_from_height_ratio"])
        per_camera_measurements[camera_id] = {
            **measurement,
            "center_uv": center_uv.astype(float).tolist(),
            "axis_x_img": axis_x_img.astype(float).tolist(),
            "axis_y_img": axis_y_img.astype(float).tolist(),
            "bbox_xyxy": [obs.bbox.x1, obs.bbox.y1, obs.bbox.x2, obs.bbox.y2],
            "confidence": 1.0,
        }

    if not width_by_camera:
        raise RuntimeError(f"No usable mask-based width estimates for {obj.label}")

    footprint_fit = _fit_rotated_rectangle_footprint(
        center_world_cv,
        width_by_camera,
        camera_models,
        max_aspect_ratio=float(runtime_cfg.get("footprint_max_aspect_ratio", 1.8)),
    )
    size_x_m = float(footprint_fit["size_x_m"])
    size_z_m = float(footprint_fit["size_z_m"])
    shape_ratio_h_over_x_plus_z, shape_height_weight = _shape_ratio_height_weight(
        float(base_height_m),
        size_x_m,
        size_z_m,
    )
    applied_height_multiplier = float(shape_height_weight * float(runtime_cfg.get("height_scale", 1.0)))
    selected_height_m = float(base_height_m * applied_height_multiplier)
    size_xyz = np.array(
        [
            size_x_m,
            selected_height_m,
            size_z_m,
        ],
        dtype=np.float32,
    )
    yaw_rad_unity = float(footprint_fit["yaw_rad"])
    return {
        "center_world_unity": center_world_unity.astype(float).tolist(),
        "center_world_cv": center_world_cv.astype(float).tolist(),
        "reference_height_m": float(base_height_m),
        "reference_height_mm": float(base_height_m * 1000.0),
        "selected_height_before_scale_m": float(base_height_m),
        "selected_height_before_scale_mm": float(base_height_m * 1000.0),
        "selected_height_m": float(selected_height_m),
        "selected_height_mm": float(selected_height_m * 1000.0),
        "shape_ratio_h_over_x_plus_z": float(shape_ratio_h_over_x_plus_z),
        "shape_height_weight": float(shape_height_weight),
        "applied_height_multiplier": float(applied_height_multiplier),
        "height_scale": float(runtime_cfg.get("height_scale", 1.0)),
        "size_xyz_m": size_xyz.astype(float).tolist(),
        "size_xyz_mm": (size_xyz * 1000.0).astype(float).tolist(),
        "yaw_rad": float(yaw_rad_unity),
        "yaw_deg": float(np.degrees(float(footprint_fit["yaw_rad"]))),
        "sam_mask_width_by_camera_m": {camera_id: float(value) for camera_id, value in width_by_camera.items()},
        "predicted_width_by_camera_m": {
            camera_id: float(value)
            for camera_id, value in footprint_fit["predicted_width_by_camera"].items()
        },
        "footprint_max_aspect_ratio": float(footprint_fit["max_aspect_ratio"]),
        "camera_azimuth_rad_by_camera": {
            camera_id: -float(value)
            for camera_id, value in footprint_fit["azimuth_rad_by_camera"].items()
        },
        "footprint_fit_rmse_m": float(footprint_fit["fit_rmse_m"]),
        "bbox_height_estimates_m": {
            camera_id: float(value) for camera_id, value in bbox_height_estimates.items()
        },
        "vertical_line_height_median_m": float(vertical_line_fit["height_median"]),
        "vertical_line_height_min_m": float(vertical_line_fit["height_min"]),
        "vertical_line_height_selected_m": float(vertical_line_height_m),
        "per_camera_measurements": per_camera_measurements,
    }


def _grid_resolution_for_budget(size_xyz: np.ndarray, budget: int) -> tuple[int, int, int]:
    sx, sy, sz = [max(float(v), 1e-4) for v in size_xyz]
    volume = sx * sy * sz
    if volume <= 1e-12:
        return (4, 4, 4)
    scale = (max(int(budget), 64) / volume) ** (1.0 / 3.0)
    nx = max(3, int(round(sx * scale)))
    ny = max(3, int(round(sy * scale)))
    nz = max(3, int(round(sz * scale)))
    return nx, ny, nz


def _yaw_rotation_matrix(yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )


def _transform_local_points(local_points: np.ndarray, center_world: np.ndarray, yaw_rad: float) -> np.ndarray:
    rotation = _yaw_rotation_matrix(yaw_rad)
    return local_points @ rotation.T + center_world.reshape(1, 3)


def _sample_box_volume_local(size_xyz: np.ndarray, budget: int) -> np.ndarray:
    nx, ny, nz = _grid_resolution_for_budget(size_xyz, budget)
    xs = np.linspace(-size_xyz[0] * 0.5, size_xyz[0] * 0.5, nx, dtype=np.float32)
    ys = np.linspace(-size_xyz[1] * 0.5, size_xyz[1] * 0.5, ny, dtype=np.float32)
    zs = np.linspace(-size_xyz[2] * 0.5, size_xyz[2] * 0.5, nz, dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)


def _build_box_points_world(
    center_world: np.ndarray,
    size_xyz: np.ndarray,
    yaw_rad: float,
    point_budget: int,
) -> np.ndarray:
    local_pts = _sample_box_volume_local(size_xyz, point_budget)
    return _transform_local_points(local_pts, center_world, yaw_rad)


def _point_colors(label: str, count: int) -> np.ndarray:
    color = LABEL_COLORS.get(normalize_label(label), np.array([180, 180, 180], dtype=np.uint8))
    return np.repeat(color.reshape(1, 3), count, axis=0)


def _object_key(obj: WorldPositionObject) -> str:
    return f"{obj.topic_key}:{obj.label}:{obj.item_id}"


def run_pipeline(
    scene_config: Path,
    yolo_class_name: str,
    goal_output: Path,
    debug_save: bool,
    image_paths_by_camera: dict[str, Path],
    world_position_data: Any,
    primary_camera_id: str | None = None,
) -> dict:
    pipeline_start = time.perf_counter()
    topic_parse_s = 0.0
    camera_model_load_s = 0.0
    object_geometry_estimation_s = 0.0
    cfg, _ = load_scene_config(scene_config)
    validate_required_paths(cfg)
    prepare_runtime_imports(cfg)
    device = validate_runtime_device(cfg)

    target_label = normalize_label(yolo_class_name)
    canonical_images = _canonicalize_image_paths(image_paths_by_camera)
    if not canonical_images:
        raise RuntimeError("No input images were provided.")

    topic_parse_start = time.perf_counter()
    objects = parse_world_position_data(world_position_data)
    topic_parse_s = float(time.perf_counter() - topic_parse_start)
    target_candidates = [obj for obj in objects if obj.label == target_label]
    if not target_candidates:
        available = sorted({obj.label for obj in objects})
        raise RuntimeError(f"Target '{target_label}' not found in world_position_data. Available: {available}")
    target_obj = target_candidates[0]

    selected_primary = _normalize_selected_camera(primary_camera_id)
    if (
        selected_primary is None
        or selected_primary not in canonical_images
        or selected_primary not in target_obj.observations
    ):
        fallback_candidates = [
            camera_id for camera_id in target_obj.observations.keys() if camera_id in canonical_images
        ]
        if not fallback_candidates:
            raise RuntimeError("No uploaded image corresponds to a target camera observation.")
        selected_primary = fallback_candidates[0]

    camera_ids = sorted({camera_id for obj in objects for camera_id in obj.observations if camera_id in canonical_images})
    if not camera_ids:
        raise RuntimeError("No overlap between uploaded images and world_position_data camera observations.")
    camera_model_start = time.perf_counter()
    camera_models = _load_camera_models(camera_ids, Path(cfg["camera"]["camera_parameter_dir"]))
    camera_model_load_s = float(time.perf_counter() - camera_model_start)

    images_by_camera, masks_by_object, sam_timing = _segment_objects_with_sam(
        cfg,
        canonical_images,
        objects,
        device=device,
    )

    object_reports: list[dict[str, object]] = []
    target_report: dict[str, object] | None = None
    geometry_start = time.perf_counter()
    for obj in objects:
        size_report = _estimate_object_size_from_masks(
            obj,
            camera_models=camera_models,
            masks_by_object=masks_by_object,
            runtime_cfg=cfg["runtime"],
        )
        report = {
            "label": obj.label,
            "id": obj.item_id,
            "topic_key": obj.topic_key,
            "center_world_unity": size_report["center_world_unity"],
            "bbox_observations": {
                camera_id: [obs.bbox.x1, obs.bbox.y1, obs.bbox.x2, obs.bbox.y2]
                for camera_id, obs in obj.observations.items()
            },
            "selected_cameras": sorted(size_report["per_camera_measurements"].keys()),
            "reference_height_m": size_report["reference_height_m"],
            "reference_height_mm": float(size_report["reference_height_mm"]),
            "selected_height_before_scale_m": size_report["selected_height_before_scale_m"],
            "selected_height_before_scale_mm": size_report["selected_height_before_scale_mm"],
            "selected_height_m": size_report["selected_height_m"],
            "selected_height_mm": size_report["selected_height_mm"],
            "size_xyz_m": size_report["size_xyz_m"],
            "size_xyz_mm": size_report["size_xyz_mm"],
            "yaw_rad": size_report["yaw_rad"],
            "yaw_deg": size_report["yaw_deg"],
            "height_source": "geometry",
            "shape_ratio_h_over_x_plus_z": size_report["shape_ratio_h_over_x_plus_z"],
            "shape_height_weight": size_report["shape_height_weight"],
            "applied_height_multiplier": size_report["applied_height_multiplier"],
            "height_scale": size_report["height_scale"],
            "sam_mask_width_by_camera_m": size_report["sam_mask_width_by_camera_m"],
            "predicted_width_by_camera_m": size_report["predicted_width_by_camera_m"],
            "camera_azimuth_rad_by_camera": size_report["camera_azimuth_rad_by_camera"],
            "footprint_max_aspect_ratio": size_report["footprint_max_aspect_ratio"],
            "footprint_fit_rmse_m": size_report["footprint_fit_rmse_m"],
            "bbox_height_estimates_m": size_report["bbox_height_estimates_m"],
            "vertical_line_height_median_m": size_report["vertical_line_height_median_m"],
            "vertical_line_height_min_m": size_report["vertical_line_height_min_m"],
            "vertical_line_height_selected_m": size_report["vertical_line_height_selected_m"],
            "per_camera_measurements": size_report["per_camera_measurements"],
        }
        object_reports.append(report)

        if _object_key(obj) == _object_key(target_obj):
            target_report = report
    object_geometry_estimation_s = float(time.perf_counter() - geometry_start)

    if target_report is None:
        raise RuntimeError("Failed to build target report.")

    import trimesh.transformations as tra  # type: ignore

    target_center_world = np.asarray(target_report["center_world_unity"], dtype=np.float32)
    point_budget = int(
        cfg["runtime"].get(
            "point_budget",
            cfg["runtime"].get("obstacle_point_budget", cfg["runtime"].get("num_sample_points", 1800)),
        )
    )
    target_size_xyz = np.asarray(target_report["size_xyz_m"], dtype=np.float32)
    target_yaw_rad = float(target_report["yaw_rad"])
    object_points = _build_box_points_world(target_center_world, target_size_xyz, target_yaw_rad, point_budget)
    object_colors = _point_colors(target_label, len(object_points))

    obstacle_scene_parts: list[np.ndarray] = []
    obstacle_scene_colors: list[np.ndarray] = []
    for report in object_reports:
        if (
            str(report["topic_key"]) == str(target_report["topic_key"])
            and int(report["id"]) == int(target_report["id"])
            and str(report["label"]) == str(target_report["label"])
        ):
            continue
        center_world = np.asarray(report["center_world_unity"], dtype=np.float32)
        size_xyz = np.asarray(report["size_xyz_m"], dtype=np.float32)
        yaw_rad = float(report["yaw_rad"])
        obstacle_world = _build_box_points_world(center_world, size_xyz, yaw_rad, point_budget).astype(np.float32)
        obstacle_scene_parts.append(obstacle_world)
        obstacle_scene_colors.append(_point_colors(str(report["label"]), len(obstacle_world)))

    scene_pc_world = (
        np.concatenate(obstacle_scene_parts, axis=0).astype(np.float32)
        if obstacle_scene_parts
        else np.zeros((0, 3), dtype=np.float32)
    )
    scene_colors = (
        np.concatenate(obstacle_scene_colors, axis=0).astype(np.uint8)
        if obstacle_scene_colors
        else np.zeros((0, 3), dtype=np.uint8)
    )

    grasp_stage_start = time.perf_counter()
    gripper_config = Path(cfg["models"]["gripper_config"])
    used_outlier_filter = True
    try:
        raw_inference = run_graspgen_point_cloud_inference(
            object_points,
            gripper_config,
            grasp_threshold=-1.0,
            num_grasps=int(cfg["runtime"]["num_grasps"]),
            topk_num_grasps=int(cfg["runtime"]["topk_num_grasps"]),
            point_colors=object_colors,
            remove_outliers=True,
            center_object=True,
        )
    except RuntimeError as exc:
        if "empty after outlier removal" not in str(exc).lower():
            raise
        used_outlier_filter = False
        raw_inference = run_graspgen_point_cloud_inference(
            object_points,
            gripper_config,
            grasp_threshold=-1.0,
            num_grasps=int(cfg["runtime"]["num_grasps"]),
            topk_num_grasps=int(cfg["runtime"]["topk_num_grasps"]),
            point_colors=object_colors,
            remove_outliers=False,
            center_object=True,
        )
    graspgen_inference_s = float(time.perf_counter() - grasp_stage_start)

    pc_c = raw_inference.object_points_local
    obj_colors_c = raw_inference.object_colors_local
    grasps_c = raw_inference.grasps_local
    confidences = raw_inference.confidences
    t_center = raw_inference.object_to_local_transform
    grasp_cfg = raw_inference.cfg
    num_total_grasps = int(len(grasps_c))
    approach_filter_start = time.perf_counter()
    grasps_c, confidences = filter_by_approach_direction(grasps_c, confidences)
    approach_filter_s = float(time.perf_counter() - approach_filter_start)
    if len(grasps_c) == 0:
        raise RuntimeError("No grasps remain after approach filtering.")

    object_pc_raw_c = transform_points(object_points, t_center)
    if len(scene_pc_world) > 0:
        scene_raw_c = transform_points(scene_pc_world, t_center)
        collision_start = time.perf_counter()
        collision_result = filter_grasps_by_collision(
            grasps_c,
            scene_raw_c,
            grasp_cfg,
            collision_threshold=float(cfg["runtime"].get("collision_threshold", 0.02)),
            max_scene_points=int(cfg["runtime"].get("max_collision_scene_points", 8192)),
        )
        collision_filter_s = float(time.perf_counter() - collision_start)
        collision_free_mask = collision_result.collision_free_mask
        scene_c = collision_result.scene_points_local
    else:
        collision_free_mask = np.ones(len(grasps_c), dtype=bool)
        scene_c = np.zeros((0, 3), dtype=np.float32)
        scene_raw_c = np.zeros((0, 3), dtype=np.float32)
        collision_result = None
        collision_filter_s = 0.0

    all_grasps = np.asarray(grasps_c, dtype=np.float32).copy()
    all_confidences = np.asarray(confidences, dtype=np.float32).copy()
    grasps = grasps_c[collision_free_mask]
    confidences = confidences[collision_free_mask]
    if len(grasps) == 0:
        raise RuntimeError("No collision-free grasps remain after obstacle filtering.")

    grasp_stats = {
        "used_outlier_filter": bool(used_outlier_filter),
        "num_total_grasps": num_total_grasps,
        "num_collision_free_grasps": int(len(grasps)),
        "num_grasps_after_approach_filter": int(len(grasps_c)),
        "num_grasps_after_collision_filter": int(len(grasps)),
        "num_scene_points_used_for_collision": (
            int(collision_result.scene_points_used)
            if collision_result is not None
            else 0
        ),
        "surface_sample_s": 0.0,
        "graspgen_model_init_s": 0.0,
        "graspgen_inference_s": graspgen_inference_s,
        "approach_filter_s": approach_filter_s,
        "collision_filter_s": collision_filter_s,
        "grasp_stage_total_s": float(time.perf_counter() - grasp_stage_start),
    }
    grasp_debug_npz = {
        "all_grasps": all_grasps,
        "all_scores": all_confidences,
        "collision_free_mask": np.asarray(collision_free_mask, dtype=bool),
        "collision_free_grasps": np.asarray(grasps, dtype=np.float32),
        "collision_free_scores": np.asarray(confidences, dtype=np.float32),
        "pc_object": np.asarray(pc_c, dtype=np.float32),
        "pc_object_raw": np.asarray(object_pc_raw_c, dtype=np.float32),
        "pc_scene": np.asarray(scene_c, dtype=np.float32),
        "pc_scene_raw": np.asarray(scene_raw_c, dtype=np.float32),
        "pc_scene_colors": np.asarray(scene_colors, dtype=np.uint8),
        "target_label": np.array([target_label], dtype=object),
    }
    if obj_colors_c is not None:
        grasp_debug_npz["pc_object_colors"] = np.asarray(obj_colors_c, dtype=np.uint8)
    if object_colors is not None:
        grasp_debug_npz["pc_object_raw_colors"] = np.asarray(object_colors, dtype=np.uint8)

    goal_data = compute_goal_pose(target_center_world, grasps, confidences, cfg["map"])
    map_feasible_mask = np.asarray(goal_data["map_feasible_mask"], dtype=bool)
    map_feasible_grasps = np.asarray(grasps[map_feasible_mask], dtype=np.float32)
    map_feasible_scores = np.asarray(confidences[map_feasible_mask], dtype=np.float32)
    grasp_debug_npz["map_feasible_mask"] = map_feasible_mask
    grasp_debug_npz["map_feasible_grasps"] = map_feasible_grasps
    grasp_debug_npz["map_feasible_scores"] = map_feasible_scores
    grasp_debug_npz["goal_unity_candidates"] = np.asarray(goal_data["goal_unity_candidates"], dtype=np.float32)
    grasp_debug_npz["goal_ros_candidates"] = np.asarray(goal_data["goal_ros_candidates"], dtype=np.float32)
    grasp_debug_npz["collision_free_grasps_before_map_filter"] = np.asarray(grasps, dtype=np.float32)
    grasp_debug_npz["collision_free_scores_before_map_filter"] = np.asarray(confidences, dtype=np.float32)
    grasp_debug_npz["collision_free_grasps"] = map_feasible_grasps
    grasp_debug_npz["collision_free_scores"] = map_feasible_scores

    visualization_npz_path = _save_visualization_npz(
        center_world=target_center_world,
        debug_npz=grasp_debug_npz,
        primary_camera_id=selected_primary,
        target_label=target_label,
        objects=object_reports,
    )

    goal_output = goal_output.expanduser().resolve()
    goal_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "center_world": target_center_world.astype(float).tolist(),
        "center_world_coordinate_frame": "unity_world",
        "grasp_relative_coordinate_frame": "unity_local",
        "group_ranking": goal_data["group_ranking"],
        "goal_pose_path": str(goal_output),
        "primary_camera_id": selected_primary,
        "target_object": {
            "label": target_obj.label,
            "id": target_obj.item_id,
            "height_depth_m": None,
            "height_bbox_cap_m": None,
            "height_geometry_base_m": float(target_report["selected_height_before_scale_m"]),
            "height_selected_m": float(target_report["selected_height_m"]),
            "height_source": "geometry",
        },
        "objects": object_reports,
        "num_matched_objects": int(len(object_reports)),
        "num_obstacle_objects": int(sum(1 for obj in object_reports if obj["label"] != target_label)),
        "num_obstacle_scene_points": int(len(scene_pc_world)),
        "goal_pose_used_map_fallback": bool(goal_data.get("used_map_fallback", False)),
        "grasp_visualization_npz_path": (
            str(visualization_npz_path) if visualization_npz_path is not None else None
        ),
        "depth_affine_scale": None,
        "depth_affine_offset_m": None,
        "primary_depth_crop_box_xyxy": None,
        "timing": {
            "topic_parse_s": topic_parse_s,
            "camera_model_load_s": camera_model_load_s,
            "sam_model_load_s": float(sam_timing["sam_model_load_s"]),
            "sam_inference_total_s": float(sam_timing["sam_inference_total_s"]),
            "sam_inference_by_camera_s": sam_timing["sam_inference_by_camera_s"],
            "sam_total_s": float(sam_timing["sam_total_s"]),
            "num_sam_predictions": int(sam_timing["num_sam_predictions"]),
            "depth_model_load_s": 0.0,
            "depth_inference_s": 0.0,
            "depth_stage_total_s": 0.0,
            "object_geometry_estimation_s": object_geometry_estimation_s,
            "surface_sample_s": float(grasp_stats.get("surface_sample_s", 0.0)),
            "graspgen_model_init_s": float(grasp_stats.get("graspgen_model_init_s", 0.0)),
            "graspgen_inference_s": float(grasp_stats.get("graspgen_inference_s", 0.0)),
            "approach_filter_s": float(grasp_stats.get("approach_filter_s", 0.0)),
            "collision_filter_s": float(grasp_stats.get("collision_filter_s", 0.0)),
            "grasp_stage_total_s": float(grasp_stats.get("grasp_stage_total_s", 0.0)),
            "pipeline_total_s": float(time.perf_counter() - pipeline_start),
        },
        **grasp_stats,
    }
    goal_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if debug_save:
        debug_dir = goal_output.parent / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / "objects.json").write_text(json.dumps(object_reports, indent=2), encoding="utf-8")

    return payload
