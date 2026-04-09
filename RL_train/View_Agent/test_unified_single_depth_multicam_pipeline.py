#!/usr/bin/env python3
"""
Unified single-file pipeline:
  RGB images + camera params -> YOLO -> SAM -> single depth height ->
  multi-camera size fusion -> grasp + collision -> NPZ

This consolidates the previous two-stage flow into one script and records
per-model timing information.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh.transformations as tra  # type: ignore

from multicam_volume_utils import default_sam_checkpoint
from test_eye_level_grasp import (
    DEFAULT_CAMERA_PARAMETER_DIR,
    DEFAULT_FUSE_CAMERA_IDS,
    DEFAULT_IMAGE_DIR,
    build_primitive_obstacle,
    default_weights_path,
    fit_rotated_rectangle_footprint,
    load_or_run_obstacle_yolo,
    point_colors,
)
from test_graspgen import GRIPPER_CONFIG, filter_collisions, run_grasp_inference
from test_multicam_teddy_height import (
    Detection,
    choose_final_height_m,
    estimate_height_on_vertical_line,
    load_camera_models,
    match_target_detections,
    normalize_label,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TOOL_ROOT = REPO_ROOT / "3090server" / "VLM_RL"
if str(TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOL_ROOT))

SAM3D_CAMERA_ROOT = TOOL_ROOT / "get_item_info_agent" / "vendor" / "sam3d_runtime" / "Camera_3D_Localization"
for path in (SAM3D_CAMERA_ROOT, SAM3D_CAMERA_ROOT / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from segment_anything import SamPredictor, sam_model_registry  # type: ignore  # noqa: E402
from src.seg_crop_depth import infer_depth_image, load_depth_model, normalize_to_uint8  # type: ignore  # noqa: E402


DEFAULT_PRIMARY_CAMERA_ID = "Camera_Room1_12"
DEFAULT_LABELS = ("doll", "apple", "wine")
WORLD_UP_CV = np.array([0.0, 1.0, 0.0], dtype=np.float32)
DEFAULT_KNOWN_CENTERS_ALIGNED_WORLD = {
    "doll": np.array([2.36199999, 0.43299982, 8.4659996], dtype=np.float32),
    "apple": np.array([2.33899999, 0.407000005, 8.1079998], dtype=np.float32),
    "wine": np.array([2.31200004, 0.495999992, 8.78999996], dtype=np.float32),
}


def default_depth_weights_path() -> Path:
    return TOOL_ROOT / "models" / "depth" / "depth_anything_v2_vitb.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified single-depth + multi-camera SAM + grasp pipeline."
    )
    parser.add_argument("--target-label", default="doll")
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--camera-parameter-dir", type=Path, default=DEFAULT_CAMERA_PARAMETER_DIR)
    parser.add_argument("--camera-ids", nargs="+", default=list(DEFAULT_FUSE_CAMERA_IDS))
    parser.add_argument("--primary-camera-id", default=DEFAULT_PRIMARY_CAMERA_ID)
    parser.add_argument("--weights", type=Path, default=default_weights_path())
    parser.add_argument("--conf-thresh", type=float, default=0.20)
    parser.add_argument("--device", default="")
    parser.add_argument("--min-views", type=int, default=None)
    parser.add_argument("--sam-checkpoint", type=Path, default=default_sam_checkpoint())
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--depth-weights", type=Path, default=default_depth_weights_path())
    parser.add_argument("--depth-input-size", type=int, default=518)
    parser.add_argument("--robust-lower", type=float, default=5.0)
    parser.add_argument("--robust-upper", type=float, default=95.0)
    parser.add_argument("--axis-step-m", type=float, default=0.05)
    parser.add_argument("--min-valid-height-m", type=float, default=0.03)
    parser.add_argument("--max-valid-height-m", type=float, default=1.50)
    parser.add_argument("--vertical-line-height-stat", choices=("min", "median"), default="min")
    parser.add_argument("--disable-yolo-height-cap", action="store_true")
    parser.add_argument("--point-budget", type=int, default=1800)
    parser.add_argument("--num-grasps", type=int, default=200)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--collision-thresh", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def default_output_dir(image_dir: Path) -> Path:
    return image_dir / "unified_single_depth_multicam_pipeline_outputs"


def image_path_for_camera(image_dir: Path, camera_id: str) -> Path:
    return image_dir / f"{camera_id}_rgb.png"


def build_sam_predictor(model_type: str, checkpoint: Path, device: str) -> tuple[object, SamPredictor]:
    if model_type not in sam_model_registry:
        raise ValueError(f"Unsupported SAM model type: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam = sam.to(device=device)
    predictor = SamPredictor(sam)
    return sam, predictor


def clamp_bbox_xyxy(det: Detection, image_shape: tuple[int, int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    x1 = max(0, min(w, int(math.floor(det.x1))))
    y1 = max(0, min(h, int(math.floor(det.y1))))
    x2 = max(0, min(w, int(math.ceil(det.x2))))
    y2 = max(0, min(h, int(math.ceil(det.y2))))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid bbox after clamp: {(x1, y1, x2, y2)}")
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def segment_mask_for_detection(
    predictor: SamPredictor,
    image_bgr: np.ndarray,
    detection: Detection,
) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    bbox = clamp_bbox_xyxy(detection, image_bgr.shape)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=bbox[None, :],
        multimask_output=True,
    )
    if masks is None or len(masks) == 0:
        raise RuntimeError("SAM returned no masks.")
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool)


def project_point(projection: np.ndarray, point_world: np.ndarray) -> np.ndarray:
    x_h = np.append(np.asarray(point_world, dtype=np.float64).reshape(3), 1.0)
    uvw = np.asarray(projection, dtype=np.float64) @ x_h
    if abs(float(uvw[2])) < 1e-12:
        raise RuntimeError("Point projects to infinity.")
    return np.array([uvw[0] / uvw[2], uvw[1] / uvw[2]], dtype=np.float32)


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        raise RuntimeError("Degenerate vector encountered.")
    return v / n


def visible_horizontal_direction(center_world_cv: np.ndarray, camera_center_world_cv: np.ndarray) -> np.ndarray:
    view = np.asarray(camera_center_world_cv, dtype=np.float32) - np.asarray(center_world_cv, dtype=np.float32)
    view[1] = 0.0
    view = normalize_vec(view)
    side = np.cross(WORLD_UP_CV, view).astype(np.float32)
    return normalize_vec(side)


def local_image_axes(
    projection: np.ndarray,
    center_world_cv: np.ndarray,
    camera_center_world_cv: np.ndarray,
    axis_step_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_uv = project_point(projection, center_world_cv)
    up_uv = project_point(projection, np.asarray(center_world_cv, dtype=np.float32) + WORLD_UP_CV * axis_step_m)
    axis_y_img = normalize_vec(up_uv - center_uv)
    side_world = visible_horizontal_direction(center_world_cv, camera_center_world_cv)
    side_uv = project_point(projection, np.asarray(center_world_cv, dtype=np.float32) + side_world * axis_step_m)
    axis_x_img = normalize_vec(side_uv - center_uv)
    return center_uv, axis_x_img, axis_y_img


def robust_axis_extent(
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


def measure_camera_width_from_mask(
    mask: np.ndarray,
    center_uv: np.ndarray,
    axis_x_img: np.ndarray,
    axis_y_img: np.ndarray,
    reference_height_m: float,
    lower: float,
    upper: float,
) -> dict[str, float]:
    height_px = robust_axis_extent(mask, center_uv, axis_y_img, lower, upper)
    width_px = robust_axis_extent(mask, center_uv, axis_x_img, lower, upper)
    if height_px <= 1e-6:
        raise RuntimeError("Measured SAM height is degenerate.")
    width_m = float(reference_height_m * (width_px / height_px))
    return {
        "mask_height_px": float(height_px),
        "mask_width_px": float(width_px),
        "width_m_from_height_ratio": width_m,
    }


def cv_world_to_aligned_world(point_world_cv: np.ndarray) -> np.ndarray:
    point_world = np.asarray(point_world_cv, dtype=np.float32).copy()
    if point_world.ndim == 1:
        point_world = point_world.reshape(3)
        point_world[2] *= -1.0
        return point_world
    if point_world.ndim == 2 and point_world.shape[1] == 3:
        point_world[:, 2] *= -1.0
        return point_world
    raise ValueError(f"Expected (3,) or (N,3), got {point_world.shape}")


def aligned_world_to_cv(point_world_aligned: np.ndarray) -> np.ndarray:
    point_world = np.asarray(point_world_aligned, dtype=np.float32).copy()
    if point_world.ndim == 1:
        point_world = point_world.reshape(3)
        point_world[2] *= -1.0
        return point_world
    if point_world.ndim == 2 and point_world.shape[1] == 3:
        point_world[:, 2] *= -1.0
        return point_world
    raise ValueError(f"Expected (3,) or (N,3), got {point_world.shape}")


def known_center_world_cv(label: str, fallback_center_world_cv: np.ndarray) -> np.ndarray:
    normalized = normalize_label(label)
    if normalized in DEFAULT_KNOWN_CENTERS_ALIGNED_WORLD:
        return aligned_world_to_cv(DEFAULT_KNOWN_CENTERS_ALIGNED_WORLD[normalized])
    return np.asarray(fallback_center_world_cv, dtype=np.float32).reshape(3)


def reprojection_error_for_center(
    center_world_cv: np.ndarray,
    detections: dict[str, Detection],
    camera_models: dict[str, dict[str, np.ndarray]],
) -> float:
    errors = []
    for camera_id, det in detections.items():
        uv_proj = project_point(camera_models[camera_id]["P"], center_world_cv)
        errors.append(float(np.linalg.norm(uv_proj - det.center_uv)))
    return float(np.mean(errors)) if errors else 0.0


def union_bbox_from_detections(detections: list[Detection], image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    x1 = min(float(det.x1) for det in detections)
    y1 = min(float(det.y1) for det in detections)
    x2 = max(float(det.x2) for det in detections)
    y2 = max(float(det.y2) for det in detections)
    h, w = image_shape[:2]
    left = max(0, min(w, int(math.floor(x1))))
    top = max(0, min(h, int(math.floor(y1))))
    right = max(0, min(w, int(math.ceil(x2))))
    bottom = max(0, min(h, int(math.ceil(y2))))
    if right <= left or bottom <= top:
        raise RuntimeError("Union bbox is degenerate.")
    return left, top, right, bottom


def crop_intrinsic(k: np.ndarray, left: int, top: int) -> np.ndarray:
    k_crop = np.asarray(k, dtype=np.float32).copy()
    k_crop[0, 2] -= float(left)
    k_crop[1, 2] -= float(top)
    return k_crop


def sample_depth_value(depth_map: np.ndarray, uv: np.ndarray, radius: int = 2) -> float:
    x = int(round(float(uv[0])))
    y = int(round(float(uv[1])))
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(depth_map.shape[1], x + radius + 1)
    y2 = min(depth_map.shape[0], y + radius + 1)
    patch = np.asarray(depth_map[y1:y2, x1:x2], dtype=np.float32)
    valid = np.isfinite(patch) & (np.abs(patch) > 1e-6)
    if np.any(valid):
        return float(np.median(patch[valid]))
    value = float(depth_map[min(max(y, 0), depth_map.shape[0] - 1), min(max(x, 0), depth_map.shape[1] - 1)])
    if not np.isfinite(value) or abs(value) <= 1e-6:
        raise RuntimeError("No valid depth around calibration point.")
    return value


def fit_depth_affine(depth_rel_values: list[float], true_z_values: list[float]) -> tuple[float, float]:
    depth_rel = np.asarray(depth_rel_values, dtype=np.float64)
    true_z = np.asarray(true_z_values, dtype=np.float64)
    if len(depth_rel) == 0:
        raise RuntimeError("No depth calibration samples available.")
    if len(depth_rel) == 1:
        if abs(float(depth_rel[0])) <= 1e-8:
            raise RuntimeError("Single-point depth calibration failed due to zero relative depth.")
        return float(true_z[0] / depth_rel[0]), 0.0
    a, b = np.linalg.lstsq(
        np.column_stack([depth_rel, np.ones_like(depth_rel)]),
        true_z,
        rcond=None,
    )[0]
    return float(a), float(b)


def backproject_depth(depth_metric_m: np.ndarray, intrinsic_k: np.ndarray) -> np.ndarray:
    h, w = depth_metric_m.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    z = depth_metric_m.astype(np.float32)
    fx = float(intrinsic_k[0, 0])
    fy = float(intrinsic_k[1, 1])
    cx = float(intrinsic_k[0, 2])
    cy = float(intrinsic_k[1, 2])
    x = (xs.astype(np.float32) - cx) * z / fx
    y = (ys.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def camera_to_world_cv(extrinsic: np.ndarray, points_cam: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    inv_rotation = np.linalg.inv(rotation)
    return (np.asarray(points_cam, dtype=np.float32) - translation.reshape(1, 3)) @ inv_rotation.T


def bbox_height_estimates_from_center(
    detections: dict[str, Detection],
    center_world_cv: np.ndarray,
    camera_models: dict[str, dict[str, np.ndarray]],
) -> dict[str, float]:
    estimates: dict[str, float] = {}
    center_world_cv = np.asarray(center_world_cv, dtype=np.float32).reshape(3)
    for camera_id, det in detections.items():
        k = camera_models[camera_id]["K"]
        extrinsic = camera_models[camera_id]["extrinsic"]
        point_cam = extrinsic[:, :3] @ center_world_cv + extrinsic[:, 3]
        z_cam = float(point_cam[2])
        if abs(z_cam) <= 1e-8:
            continue
        estimates[camera_id] = float(det.bbox_height_px * abs(z_cam) / float(k[1, 1]))
    return estimates


def run_grasp_inference_no_filter(
    object_pc: np.ndarray,
    object_colors: np.ndarray | None,
    num_grasps: int,
    topk: int,
):
    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg  # type: ignore
    from grasp_gen.utils.meshcat_utils import get_color_from_score  # type: ignore

    pc_filtered = np.asarray(object_pc, dtype=np.float32)
    filtered_colors = None if object_colors is None else np.asarray(object_colors, dtype=np.uint8)

    cfg = load_grasp_cfg(str(GRIPPER_CONFIG))
    sampler = GraspGenSampler(cfg)
    grasps_t, conf_t = GraspGenSampler.run_inference(
        pc_filtered,
        sampler,
        grasp_threshold=-1.0,
        num_grasps=num_grasps,
        topk_num_grasps=topk,
        remove_outliers=False,
    )
    if len(grasps_t) == 0:
        raise RuntimeError("GraspGen returned no grasps.")

    grasps = grasps_t.cpu().numpy()
    conf = conf_t.cpu().numpy()
    grasps[:, 3, 3] = 1.0
    t_sub = tra.translation_matrix(-pc_filtered.mean(axis=0))
    pc_c = tra.transform_points(pc_filtered, t_sub)
    grasps_c = np.array([t_sub @ g for g in grasps])
    scores = get_color_from_score(conf, use_255_scale=True)
    return pc_c, filtered_colors, grasps_c, conf, scores, t_sub, cfg


def maybe_downsample_points(points: np.ndarray, max_points: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) <= max_points:
        return pts
    idx = np.linspace(0, len(pts) - 1, num=max_points, dtype=int)
    return pts[idx]


def robust_extent_xyz(points: np.ndarray, lower: float, upper: float) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    lo = np.percentile(pts, lower, axis=0)
    hi = np.percentile(pts, upper, axis=0)
    return (hi - lo).astype(np.float32)


def full_extent_xyz(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    return (pts.max(axis=0) - pts.min(axis=0)).astype(np.float32)


def robust_filter_points_world(
    points_world_aligned: np.ndarray,
    center_world_aligned: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world_aligned, dtype=np.float32)
    center = np.asarray(center_world_aligned, dtype=np.float32).reshape(1, 3)
    delta = points - center
    lo = np.percentile(delta, lower, axis=0).astype(np.float32)
    hi = np.percentile(delta, upper, axis=0).astype(np.float32)
    keep = np.all((delta >= lo.reshape(1, 3)) & (delta <= hi.reshape(1, 3)), axis=1)
    filtered = points[keep]
    if len(filtered) < 8:
        return points, lo, hi
    return filtered, lo, hi


def build_eye_level_pointcloud(
    points_world_aligned: np.ndarray,
    center_world_aligned: np.ndarray,
    camera_center_world_aligned: np.ndarray,
    robust_lower: float,
    robust_upper: float,
) -> dict[str, np.ndarray]:
    points = np.asarray(points_world_aligned, dtype=np.float32)
    center = np.asarray(center_world_aligned, dtype=np.float32).reshape(3)
    camera_center = np.asarray(camera_center_world_aligned, dtype=np.float32).reshape(3)
    if len(points) < 8:
        raise RuntimeError("Not enough points to build eye-level point cloud.")

    filtered_points, prefilter_lo, prefilter_hi = robust_filter_points_world(
        points,
        center,
        robust_lower,
        robust_upper,
    )
    centroid = filtered_points.mean(axis=0)

    # Eye-level should be defined by camera geometry, not by PCA.
    # Y is fixed to world up. The eye-level forward direction keeps only the
    # horizontal component of the camera-to-object viewing direction.
    y_axis = WORLD_UP_CV.copy()
    view_dir = center - camera_center
    view_dir_horizontal = view_dir - y_axis * float(np.dot(view_dir, y_axis))
    horizontal_distance_m = float(np.linalg.norm(view_dir_horizontal))
    if horizontal_distance_m <= 1e-8:
        raise RuntimeError("Degenerate eye-level transform: camera view is parallel to world up.")
    z_axis = normalize_vec(view_dir_horizontal)
    x_axis = normalize_vec(np.cross(y_axis, z_axis).astype(np.float32))
    z_axis = normalize_vec(np.cross(x_axis, y_axis).astype(np.float32))

    delta = filtered_points - center.reshape(1, 3)
    local = np.stack(
        [
            delta @ x_axis,
            delta @ y_axis,
            delta @ z_axis,
        ],
        axis=1,
    ).astype(np.float32)
    positioned = local + center.reshape(1, 3)

    # Virtual eye-level camera: same horizontal distance and horizontal viewing
    # direction as the original camera, but elevated to the object's center Y.
    eye_camera_center = center - z_axis * horizontal_distance_m
    delta_camera = filtered_points - eye_camera_center.reshape(1, 3)
    camera_frame = np.stack(
        [
            delta_camera @ x_axis,
            delta_camera @ y_axis,
            delta_camera @ z_axis,
        ],
        axis=1,
    ).astype(np.float32)
    object_center_camera_frame = np.array([0.0, 0.0, horizontal_distance_m], dtype=np.float32)
    camera_frame_centered = camera_frame - object_center_camera_frame.reshape(1, 3)

    return {
        "points_world_prefiltered": filtered_points,
        "points_eye_level_centered": local,
        "points_eye_level_world_positioned": positioned,
        "points_eye_level_camera_frame": camera_frame,
        "points_eye_level_camera_frame_centered": camera_frame_centered,
        "projected_front_xy": local[:, :2].astype(np.float32),
        "robust_extent_xyz_m": robust_extent_xyz(local, robust_lower, robust_upper),
        "full_extent_xyz_m": full_extent_xyz(local),
        "eye_level_axes_world": np.stack([x_axis, y_axis, z_axis], axis=0).astype(np.float32),
        "eye_level_pca_centroid_world_m": centroid.astype(np.float32),
        "eye_level_reference_center_world_m": center.astype(np.float32),
        "eye_level_reference_camera_center_world_m": camera_center.astype(np.float32),
        "eye_level_virtual_camera_center_world_m": eye_camera_center.astype(np.float32),
        "eye_level_object_center_camera_frame_m": object_center_camera_frame.astype(np.float32),
        "eye_level_prefilter_lower_delta_world_m": prefilter_lo.astype(np.float32),
        "eye_level_prefilter_upper_delta_world_m": prefilter_hi.astype(np.float32),
    }


def build_eye_level_comparison_payload(
    object_points_world_aligned: dict[str, np.ndarray],
    results: dict[str, dict[str, object]],
    primary_camera_center_aligned: np.ndarray,
    max_points_per_object: int,
    robust_lower: float,
    robust_upper: float,
    gap_m: float = 0.30,
) -> dict[str, object]:
    centers = [np.asarray(info["multicam_center_world_m"], dtype=np.float32) for info in results.values()]
    sizes = [np.asarray(info["size_xyz_m"], dtype=np.float32) for info in results.values()]
    if not centers:
        raise RuntimeError("No objects available for eye-level comparison export.")

    min_x = min(float(center[0] - size[0] * 0.5) for center, size in zip(centers, sizes))
    max_x = max(float(center[0] + size[0] * 0.5) for center, size in zip(centers, sizes))
    scene_width = max_x - min_x
    cluster_offset = np.array([scene_width + float(gap_m), 0.0, 0.0], dtype=np.float32)

    before_points = []
    before_colors = []
    before_filtered_points = []
    before_filtered_colors = []
    after_points = []
    after_colors = []
    kept_labels = []
    payload: dict[str, object] = {
        "comparison_offset_m": cluster_offset.astype(float).tolist(),
    }

    for label, info in results.items():
        if label not in object_points_world_aligned:
            continue
        center_world = np.asarray(info["multicam_center_world_m"], dtype=np.float32)
        pts_before = maybe_downsample_points(object_points_world_aligned[label], max_points_per_object).astype(np.float32)
        eye_level = build_eye_level_pointcloud(
            pts_before,
            center_world,
            primary_camera_center_aligned,
            robust_lower=robust_lower,
            robust_upper=robust_upper,
        )
        pts_before_filtered = np.asarray(eye_level["points_world_prefiltered"], dtype=np.float32)
        pts_after_local = np.asarray(eye_level["points_eye_level_centered"], dtype=np.float32)
        pts_after = np.asarray(eye_level["points_eye_level_world_positioned"], dtype=np.float32) + cluster_offset.reshape(1, 3)
        pts_after_camera = np.asarray(eye_level["points_eye_level_camera_frame"], dtype=np.float32)
        pts_after_camera_centered = np.asarray(eye_level["points_eye_level_camera_frame_centered"], dtype=np.float32)
        color_before = point_colors(label, len(pts_before))
        color_before_filtered = point_colors(label, len(pts_before_filtered))
        color_after = point_colors(label, len(pts_after))

        before_points.append(pts_before)
        before_colors.append(color_before)
        before_filtered_points.append(pts_before_filtered)
        before_filtered_colors.append(color_before_filtered)
        after_points.append(pts_after)
        after_colors.append(color_after)
        kept_labels.append(label)

        key = normalize_label(label).replace(" ", "_")
        payload[f"{key}_before_points"] = pts_before
        payload[f"{key}_before_colors"] = color_before
        payload[f"{key}_before_filtered_points"] = pts_before_filtered
        payload[f"{key}_before_filtered_colors"] = color_before_filtered
        payload[f"{key}_after_points"] = pts_after
        payload[f"{key}_after_colors"] = color_after
        payload[f"{key}_after_camera_frame_points"] = pts_after_camera
        payload[f"{key}_after_camera_frame_centered_points"] = pts_after_camera_centered
        payload[f"{key}_before_center_world_m"] = center_world.astype(float)
        payload[f"{key}_after_center_world_m"] = (center_world + cluster_offset).astype(float)
        payload[f"{key}_size_xyz_m"] = np.asarray(info["size_xyz_m"], dtype=np.float32).astype(float)
        payload[f"{key}_before_yaw_rad"] = np.array([float(info["yaw_rad"])], dtype=np.float32)
        payload[f"{key}_after_yaw_rad"] = np.array([0.0], dtype=np.float32)
        payload[f"{key}_eye_level_centered_points"] = pts_after_local
        payload[f"{key}_eye_level_axes_world"] = np.asarray(eye_level["eye_level_axes_world"], dtype=np.float32)
        payload[f"{key}_eye_level_reference_center_world_m"] = np.asarray(
            eye_level["eye_level_reference_center_world_m"], dtype=np.float32
        )
        payload[f"{key}_eye_level_reference_camera_center_world_m"] = np.asarray(
            eye_level["eye_level_reference_camera_center_world_m"], dtype=np.float32
        )
        payload[f"{key}_eye_level_virtual_camera_center_world_m"] = np.asarray(
            eye_level["eye_level_virtual_camera_center_world_m"], dtype=np.float32
        )
        payload[f"{key}_eye_level_object_center_camera_frame_m"] = np.asarray(
            eye_level["eye_level_object_center_camera_frame_m"], dtype=np.float32
        )
        payload[f"{key}_before_full_extent_xyz_m"] = full_extent_xyz(pts_before)
        payload[f"{key}_before_robust_extent_xyz_m"] = robust_extent_xyz(pts_before, robust_lower, robust_upper)
        payload[f"{key}_after_full_extent_xyz_m"] = np.asarray(eye_level["full_extent_xyz_m"], dtype=np.float32)
        payload[f"{key}_after_robust_extent_xyz_m"] = np.asarray(eye_level["robust_extent_xyz_m"], dtype=np.float32)

    if not before_points:
        raise RuntimeError("No depth point clouds available for eye-level comparison export.")

    payload["labels"] = np.array(kept_labels, dtype=object)
    payload["before_points"] = np.concatenate(before_points, axis=0).astype(np.float32)
    payload["before_colors"] = np.concatenate(before_colors, axis=0).astype(np.uint8)
    payload["before_filtered_points"] = np.concatenate(before_filtered_points, axis=0).astype(np.float32)
    payload["before_filtered_colors"] = np.concatenate(before_filtered_colors, axis=0).astype(np.uint8)
    payload["after_points"] = np.concatenate(after_points, axis=0).astype(np.float32)
    payload["after_colors"] = np.concatenate(after_colors, axis=0).astype(np.uint8)
    payload["comparison_points"] = np.concatenate([payload["before_points"], payload["after_points"]], axis=0)
    payload["comparison_colors"] = np.concatenate([payload["before_colors"], payload["after_colors"]], axis=0)
    return payload


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()

    image_dir = args.image_dir.expanduser().resolve()
    camera_parameter_dir = args.camera_parameter_dir.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(image_dir).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    target_label = normalize_label(args.target_label)
    labels = [normalize_label(label) for label in args.labels]
    if target_label not in labels:
        labels.insert(0, target_label)

    camera_ids = [str(camera_id) for camera_id in args.camera_ids]
    primary_camera_id = str(args.primary_camera_id)
    min_views = args.min_views if args.min_views is not None else len(camera_ids)
    if min_views < 2 or min_views > len(camera_ids):
        raise ValueError("Invalid --min-views for the selected camera ids.")
    if primary_camera_id not in camera_ids:
        raise ValueError("--primary-camera-id must be included in --camera-ids")
    if not args.sam_checkpoint.exists():
        raise FileNotFoundError(args.sam_checkpoint)
    if not args.depth_weights.exists():
        raise FileNotFoundError(args.depth_weights)

    timings: dict[str, object] = {}

    camera_load_start = time.perf_counter()
    camera_models = load_camera_models(camera_ids, camera_parameter_dir)
    timings["camera_model_load"] = float(time.perf_counter() - camera_load_start)

    yolo_start = time.perf_counter()
    detections_by_camera, yolo_cache = load_or_run_obstacle_yolo(
        base_dir=output_dir,
        camera_ids=camera_ids,
        image_dir=image_dir,
        weights=weights,
        conf_thresh=args.conf_thresh,
        device=args.device,
    )
    timings["yolo_inference"] = float(time.perf_counter() - yolo_start)
    timings["yolo_cache"] = yolo_cache

    match_start = time.perf_counter()
    matches = {
        label: match_target_detections(
            detections_by_camera=detections_by_camera,
            camera_models=camera_models,
            target_labels=[label],
            min_views=min_views,
        )
        for label in labels
    }
    timings["multicam_matching"] = float(time.perf_counter() - match_start)

    sam_model_start = time.perf_counter()
    sam_device = str(args.device or "cuda")
    sam_model, sam_predictor = build_sam_predictor(args.sam_model_type, args.sam_checkpoint, sam_device)
    timings["sam_model_load"] = float(time.perf_counter() - sam_model_start)

    image_cache: dict[str, np.ndarray] = {}
    masks_by_label: dict[str, dict[str, np.ndarray]] = {label: {} for label in labels}
    sam_inference_by_camera: dict[str, float] = {}
    sam_inference_total = 0.0
    try:
        for camera_id in camera_ids:
            if camera_id not in image_cache:
                image_path = image_path_for_camera(image_dir, camera_id)
                image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise RuntimeError(f"Failed to read image: {image_path}")
                image_cache[camera_id] = image_bgr
            image_bgr = image_cache[camera_id]
            camera_start = time.perf_counter()
            for label, match in matches.items():
                if camera_id not in match.detections:
                    continue
                masks_by_label[label][camera_id] = segment_mask_for_detection(
                    sam_predictor,
                    image_bgr,
                    match.detections[camera_id],
                )
            elapsed = float(time.perf_counter() - camera_start)
            sam_inference_by_camera[camera_id] = elapsed
            sam_inference_total += elapsed
    finally:
        try:
            del sam_predictor
        except Exception:
            pass
        try:
            sam_model.cpu()
        except Exception:
            pass
        del sam_model

    timings["sam_inference_total"] = sam_inference_total
    timings["sam_inference_by_camera"] = sam_inference_by_camera

    primary_visible_labels = [label for label, match in matches.items() if primary_camera_id in match.detections]
    if target_label not in primary_visible_labels:
        raise RuntimeError(f"Target '{target_label}' is not visible in primary camera {primary_camera_id}.")
    primary_image = image_cache[primary_camera_id]
    primary_dets = [matches[label].detections[primary_camera_id] for label in primary_visible_labels]
    crop_left, crop_top, crop_right, crop_bottom = union_bbox_from_detections(primary_dets, primary_image.shape)
    crop_bgr = primary_image[crop_top:crop_bottom, crop_left:crop_right]
    crop_k = crop_intrinsic(camera_models[primary_camera_id]["K"], crop_left, crop_top)

    depth_model_start = time.perf_counter()
    import torch

    depth_model = load_depth_model(args.depth_weights.expanduser().resolve(), torch.device(str(args.device or "cuda")))
    timings["depth_model_load"] = float(time.perf_counter() - depth_model_start)
    depth_infer_start = time.perf_counter()
    with torch.no_grad():
        depth_rel = infer_depth_image(
            depth_model,
            crop_bgr,
            input_size=int(args.depth_input_size),
        )
    timings["depth_inference"] = float(time.perf_counter() - depth_infer_start)
    if isinstance(depth_rel, torch.Tensor):
        depth_rel = depth_rel.detach().cpu().numpy()
    depth_rel = np.asarray(depth_rel, dtype=np.float32)
    try:
        depth_model.cpu()
    except Exception:
        pass
    del depth_model

    depth_calibration_start = time.perf_counter()
    depth_rel_values: list[float] = []
    true_z_values: list[float] = []
    for label in primary_visible_labels:
        det = matches[label].detections[primary_camera_id]
        center_uv_crop = det.center_uv - np.array([crop_left, crop_top], dtype=np.float32)
        depth_rel_values.append(sample_depth_value(depth_rel, center_uv_crop))
        center_world_cv = known_center_world_cv(label, matches[label].center_world_cv)
        point_cam = camera_models[primary_camera_id]["extrinsic"][:, :3] @ center_world_cv + camera_models[primary_camera_id]["extrinsic"][:, 3]
        true_z_values.append(float(point_cam[2]))
    affine_scale, affine_offset = fit_depth_affine(depth_rel_values, true_z_values)
    depth_metric = affine_scale * depth_rel + affine_offset
    points_cam_img = backproject_depth(depth_metric, crop_k)
    timings["depth_postprocess"] = float(time.perf_counter() - depth_calibration_start)

    depth_height_by_label: dict[str, float] = {}
    comparison_points_by_label: dict[str, np.ndarray] = {}
    for label in primary_visible_labels:
        mask_full = masks_by_label[label][primary_camera_id]
        mask_crop = mask_full[crop_top:crop_bottom, crop_left:crop_right]
        valid = np.isfinite(depth_metric) & (np.abs(depth_metric) > 1e-6) & mask_crop
        target_points_cam = points_cam_img[valid]
        if len(target_points_cam) == 0:
            continue
        target_points_world_cv = camera_to_world_cv(camera_models[primary_camera_id]["extrinsic"], target_points_cam)
        comparison_points_by_label[label] = cv_world_to_aligned_world(target_points_world_cv).astype(np.float32)
        ys = target_points_world_cv[:, 1]
        lower = float(np.percentile(ys, float(args.robust_lower)))
        upper = float(np.percentile(ys, float(args.robust_upper)))
        depth_height_by_label[label] = max(0.0, upper - lower)

    per_label_times: dict[str, float] = {}
    results: dict[str, dict[str, object]] = {}
    for label in labels:
        label_start = time.perf_counter()
        match = matches[label]
        center_world_cv = known_center_world_cv(label, match.center_world_cv)
        vertical_line_fit = estimate_height_on_vertical_line(match, camera_models)
        vertical_line_height_m = (
            float(vertical_line_fit["height_min"])
            if args.vertical_line_height_stat == "min"
            else float(vertical_line_fit["height_median"])
        )
        bbox_height_estimates = bbox_height_estimates_from_center(match.detections, center_world_cv, camera_models)
        yolo_height_m = choose_final_height_m(
            bbox_height_estimates=bbox_height_estimates,
            vertical_line_height_m=vertical_line_height_m,
            min_valid_height_m=float(args.min_valid_height_m),
            max_valid_height_m=float(args.max_valid_height_m),
        )
        if label in depth_height_by_label:
            selected_height_m = (
                min(float(depth_height_by_label[label]), float(yolo_height_m))
                if not args.disable_yolo_height_cap
                else float(depth_height_by_label[label])
            )
            height_source = (
                "yolo_cap"
                if (not args.disable_yolo_height_cap and yolo_height_m < depth_height_by_label[label])
                else "depth"
            )
        else:
            selected_height_m = float(yolo_height_m)
            height_source = "yolo_only"

        width_by_camera: dict[str, float] = {}
        per_camera: dict[str, dict[str, float | list[float]]] = {}
        for camera_id, det in match.detections.items():
            mask = masks_by_label[label][camera_id]
            center_uv, axis_x_img, axis_y_img = local_image_axes(
                camera_models[camera_id]["P"],
                center_world_cv,
                camera_models[camera_id]["camera_center"],
                axis_step_m=float(args.axis_step_m),
            )
            measurement = measure_camera_width_from_mask(
                mask,
                center_uv,
                axis_x_img,
                axis_y_img,
                reference_height_m=selected_height_m,
                lower=float(args.robust_lower),
                upper=float(args.robust_upper),
            )
            width_by_camera[camera_id] = float(measurement["width_m_from_height_ratio"])
            per_camera[camera_id] = {
                **measurement,
                "center_uv": center_uv.astype(float).tolist(),
                "axis_x_img": axis_x_img.astype(float).tolist(),
                "axis_y_img": axis_y_img.astype(float).tolist(),
                "bbox_xyxy": [float(det.x1), float(det.y1), float(det.x2), float(det.y2)],
                "confidence": float(det.conf),
            }

        footprint_fit = fit_rotated_rectangle_footprint(center_world_cv, width_by_camera, camera_models)
        size_xyz_m = np.array(
            [
                float(footprint_fit["size_x_m"]),
                float(selected_height_m),
                float(footprint_fit["size_z_m"]),
            ],
            dtype=np.float32,
        )
        results[label] = {
            "multicam_center_world_m": cv_world_to_aligned_world(center_world_cv).astype(float).tolist(),
            "size_xyz_m": size_xyz_m.astype(float).tolist(),
            "size_xyz_mm": (size_xyz_m * 1000.0).astype(float).tolist(),
            "yaw_rad": float(-float(footprint_fit["yaw_rad"])),
            "yaw_deg": float(np.degrees(-float(footprint_fit["yaw_rad"]))),
            "reprojection_error_px": float(
                reprojection_error_for_center(center_world_cv, match.detections, camera_models)
            ),
            "mean_confidence": float(match.mean_conf),
            "selected_cameras": list(match.detections.keys()),
            "sam_mask_width_by_camera_m": width_by_camera,
            "predicted_width_by_camera_m": {
                camera_id: float(value)
                for camera_id, value in footprint_fit["predicted_width_by_camera"].items()
            },
            "camera_azimuth_rad_by_camera": {
                camera_id: -float(value)
                for camera_id, value in footprint_fit["azimuth_rad_by_camera"].items()
            },
            "footprint_fit_rmse_m": float(footprint_fit["fit_rmse_m"]),
            "bbox_height_estimates_m": {camera_id: float(value) for camera_id, value in bbox_height_estimates.items()},
            "vertical_line_height_median_m": float(vertical_line_fit["height_median"]),
            "vertical_line_height_min_m": float(vertical_line_fit["height_min"]),
            "vertical_line_height_selected_m": float(vertical_line_height_m),
            "depth_height_m": float(depth_height_by_label[label]) if label in depth_height_by_label else None,
            "selected_height_m": float(selected_height_m),
            "selected_height_mm": float(selected_height_m * 1000.0),
            "height_source": height_source,
            "per_camera_measurements": per_camera,
        }
        per_label_times[label] = float(time.perf_counter() - label_start)

    object_points = []
    object_colors = None
    obstacle_points = []
    obstacle_colors = []
    obstacle_labels: list[str] = []
    object_geometry = {}
    target_num_points = 0
    for label, info in results.items():
        center_world = np.asarray(info["multicam_center_world_m"], dtype=np.float32)
        size_xyz = np.asarray(info["size_xyz_m"], dtype=np.float32)
        yaw_rad = float(info["yaw_rad"])
        pts = build_primitive_obstacle(
            center_world,
            size_xyz,
            mode="box",
            point_budget=int(args.point_budget),
            polygon_sides=8,
            yaw_rad=yaw_rad,
        )
        object_geometry[label] = {
            "center_world_m": center_world.astype(float).tolist(),
            "size_xyz_m": size_xyz.astype(float).tolist(),
            "size_xyz_mm": (size_xyz * 1000.0).astype(float).tolist(),
            "yaw_rad": yaw_rad,
            "yaw_deg": float(np.degrees(yaw_rad)),
            "num_points": int(len(pts)),
        }
        if label == target_label:
            object_points = pts.astype(np.float32)
            object_colors = point_colors(label, len(object_points))
            target_num_points = int(len(object_points))
        else:
            obstacle_points.append(pts.astype(np.float32))
            obstacle_colors.append(point_colors(label, len(pts)))
            obstacle_labels.append(label)

    if len(object_points) == 0:
        raise RuntimeError(f"Failed to build target primitive for '{target_label}'.")

    if obstacle_points:
        scene_pc = np.concatenate(obstacle_points, axis=0).astype(np.float32)
        scene_colors = np.concatenate(obstacle_colors, axis=0).astype(np.uint8)
    else:
        scene_pc = np.zeros((0, 3), dtype=np.float32)
        scene_colors = np.zeros((0, 3), dtype=np.uint8)

    grasp_start = time.perf_counter()
    used_outlier_filter = True
    try:
        pc_c, obj_colors_c, grasps_c, conf, scores, t_center, cfg = run_grasp_inference(
            object_points,
            object_colors,
            num_grasps=args.num_grasps,
            topk=args.topk,
        )
    except RuntimeError as exc:
        if "empty after outlier removal" not in str(exc).lower():
            raise
        used_outlier_filter = False
        pc_c, obj_colors_c, grasps_c, conf, scores, t_center, cfg = run_grasp_inference_no_filter(
            object_points,
            object_colors,
            num_grasps=args.num_grasps,
            topk=args.topk,
        )
    grasp_time_s = time.perf_counter() - grasp_start

    collision_start = time.perf_counter()
    if len(scene_pc) > 0:
        coll_mask, scene_c = filter_collisions(
            scene_pc,
            grasps_c,
            t_center,
            cfg,
            collision_threshold=args.collision_thresh,
        )
        scene_raw_c = tra.transform_points(scene_pc, t_center)
    else:
        coll_mask = np.ones(len(grasps_c), dtype=bool)
        scene_c = np.zeros((0, 3), dtype=np.float32)
        scene_raw_c = np.zeros((0, 3), dtype=np.float32)
    collision_time_s = time.perf_counter() - collision_start

    free_grasps = grasps_c[coll_mask]
    free_conf = conf[coll_mask]
    object_pc_raw_c = tra.transform_points(object_points, t_center)

    summary_json = output_dir / "unified_single_depth_multicam_pipeline_summary.json"
    summary_md = output_dir / "unified_single_depth_multicam_pipeline_summary.md"
    grasp_npz = output_dir / f"{target_label}_unified_single_depth_multicam_grasp_result.npz"
    grasp_report_json = output_dir / f"{target_label}_unified_single_depth_multicam_grasp_report.json"
    eye_level_compare_npz = output_dir / "unified_single_depth_multicam_eye_level_compare.npz"

    npz_write_start = time.perf_counter()
    save_data = dict(
        all_grasps=grasps_c,
        all_scores=conf,
        collision_free_mask=coll_mask,
        collision_free_grasps=free_grasps,
        collision_free_scores=free_conf,
        pc_object=pc_c,
        pc_object_raw=object_pc_raw_c,
        pc_scene=scene_c,
        pc_scene_raw=scene_raw_c,
        pc_scene_colors=scene_colors,
        target_label=np.array([target_label]),
        source_summary=np.array([str(summary_json)]),
        obstacle_labels=np.array(obstacle_labels, dtype=object),
    )
    if obj_colors_c is not None:
        save_data["pc_object_colors"] = obj_colors_c
    save_data["pc_object_raw_colors"] = object_colors
    np.savez_compressed(str(grasp_npz), **save_data)
    comparison_payload = build_eye_level_comparison_payload(
        comparison_points_by_label,
        results,
        primary_camera_center_aligned=cv_world_to_aligned_world(camera_models[primary_camera_id]["camera_center"]),
        max_points_per_object=int(args.point_budget),
        robust_lower=float(args.robust_lower),
        robust_upper=float(args.robust_upper),
    )
    np.savez_compressed(str(eye_level_compare_npz), **comparison_payload)
    npz_write_s = float(time.perf_counter() - npz_write_start)

    timings.update(
        {
            "depth_height_by_label": {label: float(value) for label, value in depth_height_by_label.items()},
            "per_label_size_estimation": per_label_times,
            "grasp_inference": float(grasp_time_s),
            "collision_filter": float(collision_time_s),
            "npz_write": npz_write_s,
            "total": float(time.perf_counter() - total_start),
        }
    )

    summary_payload = {
        "target_label": target_label,
        "camera_ids": camera_ids,
        "primary_camera_id": primary_camera_id,
        "yolo_cache": yolo_cache,
        "crop_box_xyxy": [int(crop_left), int(crop_top), int(crop_right), int(crop_bottom)],
        "depth_affine_scale": float(affine_scale),
        "depth_affine_offset_m": float(affine_offset),
        "timings_s": timings,
        "labels": results,
        "eye_level_compare_npz": str(eye_level_compare_npz),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Unified Single-Depth + Multi-Camera Pipeline Summary",
        "",
        "| Label | Width (mm) | Height (mm) | Depth (mm) | Yaw (deg) | Cameras | RMSE (mm) | Height Source |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for label in labels:
        info = results[label]
        width_mm, height_mm, depth_mm = info["size_xyz_mm"]
        md_lines.append(
            f"| {label} | {width_mm:.3f} | {height_mm:.3f} | {depth_mm:.3f} | "
            f"{info['yaw_deg']:.3f} | {', '.join(info['selected_cameras'])} | "
            f"{float(info['footprint_fit_rmse_m']) * 1000.0:.3f} | {info['height_source']} |"
        )
    summary_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    grasp_report = {
        "summary_json": str(summary_json),
        "target_label": target_label,
        "target_mode": "unified_single_depth_multicam_size_box",
        "used_outlier_filter": bool(used_outlier_filter),
        "object_geometry": object_geometry[target_label],
        "obstacle_geometry": {label: object_geometry[label] for label in obstacle_labels},
        "num_object_points_raw": target_num_points,
        "num_scene_points": int(len(scene_pc)),
        "num_total_grasps": int(len(grasps_c)),
        "num_collision_free_grasps": int(int(coll_mask.sum())),
        "collision_threshold": float(args.collision_thresh),
        "result_npz": str(grasp_npz),
        "timings_s": {
            "grasp_inference": float(grasp_time_s),
            "collision_filter": float(collision_time_s),
            "npz_write": float(npz_write_s),
            "total": float(time.perf_counter() - total_start),
        },
    }
    grasp_report_json.write_text(json.dumps(grasp_report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary_payload, "grasp_report": grasp_report}, indent=2))


if __name__ == "__main__":
    main()
