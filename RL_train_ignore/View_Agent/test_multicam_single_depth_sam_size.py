#!/usr/bin/env python3
"""
Estimate object size from:
  - one reference camera with live YOLO+SAM+Depth-derived height
  - multiple cameras with YOLO+SAM masks only for width/depth

Method:
  1. Run YOLO on all selected cameras and match the requested labels.
  2. Run SAM for the matched detections on all cameras.
  3. On one reference camera, run a single crop-level Depth Anything pass.
  4. Calibrate depth with known 3D object centers, then backproject to world.
  5. Convert each reference-camera object point cloud into eye-level coordinates
     and use its robust Y extent as the depth-derived height H.
  6. For each camera, measure SAM-mask extents around the known object center.
  7. Convert each camera's horizontal extent to meters using:
       width_m ~= H * (width_px / height_px)
  8. Fuse all camera widths with a rotated-rectangle footprint fit to obtain X/Z.
  9. Optionally cap the depth-derived height with multi-camera YOLO bbox height.
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
import torch

from multicam_volume_utils import default_sam_checkpoint
from test_eye_level_grasp import (
    DEFAULT_CAMERA_PARAMETER_DIR,
    DEFAULT_FUSE_CAMERA_IDS,
    DEFAULT_IMAGE_DIR,
    default_weights_path,
    fit_rotated_rectangle_footprint,
    load_or_run_obstacle_yolo,
)
from test_eye_level_grasp import yolo_cache_path as obstacle_yolo_cache_path
from test_multicam_teddy_height import (
    Detection,
    choose_final_height_m,
    estimate_height_on_vertical_line,
    height_estimates_from_bbox,
    load_camera_models,
    match_target_detections,
    normalize_label,
)
from test_unified_single_depth_multicam_pipeline import (
    backproject_depth,
    build_eye_level_comparison_payload,
    build_eye_level_pointcloud,
    camera_to_world_cv,
    crop_intrinsic,
    default_depth_weights_path,
    fit_depth_affine,
    known_center_world_cv,
    sample_depth_value,
    union_bbox_from_detections,
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
from src.seg_crop_depth import infer_depth_image, load_depth_model  # type: ignore  # noqa: E402


DEFAULT_REFERENCE_BASE_DIR = Path(
    "/home/tjchen/workspace/VLM_RL/RL_train/View_Agent/test_tmp/pics/Camera_Room1_test/"
    "four_camera_eye_level_experiment_outputs/Camera_Room1_12"
)
DEFAULT_LABELS = ("doll", "apple", "wine")
WORLD_UP_CV = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate object sizes from one live depth-derived height plus multi-camera SAM masks."
    )
    parser.add_argument("--reference-base-dir", type=Path, default=DEFAULT_REFERENCE_BASE_DIR)
    parser.add_argument("--reference-camera-id", default="Camera_Room1_12")
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--camera-parameter-dir", type=Path, default=DEFAULT_CAMERA_PARAMETER_DIR)
    parser.add_argument("--camera-ids", nargs="+", default=list(DEFAULT_FUSE_CAMERA_IDS))
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
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def default_output_dir(reference_base_dir: Path) -> Path:
    return reference_base_dir / "single_depth_multicam_sam_size_outputs"


def image_path_for_camera(image_dir: Path, camera_id: str) -> Path:
    return image_dir / f"{camera_id}_rgb.png"


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


def clamp_bbox_xyxy(det: Detection, image_shape: tuple[int, int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    x1 = max(0, min(w, int(math.floor(det.x1))))
    y1 = max(0, min(h, int(math.floor(det.y1))))
    x2 = max(0, min(w, int(math.ceil(det.x2))))
    y2 = max(0, min(h, int(math.ceil(det.y2))))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid bbox after clamp: {(x1, y1, x2, y2)}")
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def build_sam_predictor(model_type: str, checkpoint: Path, device: str) -> tuple[object, SamPredictor]:
    if model_type not in sam_model_registry:
        raise ValueError(f"Unsupported SAM model type: {model_type}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam = sam.to(device=device)
    predictor = SamPredictor(sam)
    return sam, predictor


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
    signed = offsets @ np.asarray(axis_img, dtype=np.float32).reshape(2, 1)
    signed = signed.reshape(-1)
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


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()

    reference_base_dir = args.reference_base_dir.expanduser().resolve()
    image_dir = args.image_dir.expanduser().resolve()
    camera_parameter_dir = args.camera_parameter_dir.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(reference_base_dir).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [normalize_label(label) for label in args.labels]
    camera_ids = [str(camera_id) for camera_id in args.camera_ids]
    min_views = args.min_views if args.min_views is not None else len(camera_ids)
    if min_views < 2 or min_views > len(camera_ids):
        raise ValueError("Invalid --min-views for the selected camera ids.")
    if not args.sam_checkpoint.exists():
        raise FileNotFoundError(args.sam_checkpoint)
    if not args.depth_weights.exists():
        raise FileNotFoundError(args.depth_weights)
    if args.reference_camera_id not in camera_ids:
        raise ValueError("--reference-camera-id must be included in --camera-ids")

    timings: dict[str, object] = {}

    camera_load_start = time.perf_counter()
    camera_models = load_camera_models(camera_ids, camera_parameter_dir)
    timings["camera_model_load"] = float(time.perf_counter() - camera_load_start)

    yolo_start = time.perf_counter()
    detections_by_camera, yolo_cache = load_or_run_obstacle_yolo(
        base_dir=reference_base_dir,
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
    results: dict[str, dict[str, object]] = {}
    label_times: dict[str, float] = {}
    sam_inference_by_camera: dict[str, float] = {}
    sam_inference_total = 0.0
    reference_height_m_by_label: dict[str, float] = {}
    reference_height_details: dict[str, dict[str, object]] = {}
    comparison_points_by_label: dict[str, np.ndarray] = {}
    primary_visible_labels: list[str] = []
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
                masks_by_label[label][camera_id] = segment_mask_for_detection(sam_predictor, image_bgr, match.detections[camera_id])
            elapsed = float(time.perf_counter() - camera_start)
            sam_inference_by_camera[camera_id] = elapsed
            sam_inference_total += elapsed
        timings["sam_inference_total"] = sam_inference_total
        timings["sam_inference_by_camera"] = sam_inference_by_camera

        primary_camera_id = str(args.reference_camera_id)
        primary_visible_labels = [label for label in labels if primary_camera_id in matches[label].detections]
        if not primary_visible_labels:
            raise RuntimeError(f"No requested labels are visible in reference camera {primary_camera_id}.")

        primary_image = image_cache[primary_camera_id]
        primary_detections = [matches[label].detections[primary_camera_id] for label in primary_visible_labels]
        crop_left, crop_top, crop_right, crop_bottom = union_bbox_from_detections(primary_detections, primary_image.shape)
        crop_bgr = primary_image[crop_top:crop_bottom, crop_left:crop_right]
        crop_k = crop_intrinsic(camera_models[primary_camera_id]["K"], crop_left, crop_top)

        depth_model_start = time.perf_counter()
        depth_model = load_depth_model(args.depth_weights.expanduser().resolve(), torch.device(str(args.device or "cuda")))
        timings["depth_model_load"] = float(time.perf_counter() - depth_model_start)

        depth_infer_start = time.perf_counter()
        with torch.no_grad():
            depth_rel = infer_depth_image(depth_model, crop_bgr, input_size=int(args.depth_input_size))
        timings["depth_inference"] = float(time.perf_counter() - depth_infer_start)
        if isinstance(depth_rel, torch.Tensor):
            depth_rel = depth_rel.detach().cpu().numpy()
        depth_rel = np.asarray(depth_rel, dtype=np.float32)
        try:
            depth_model.cpu()
        except Exception:
            pass
        del depth_model

        depth_postprocess_start = time.perf_counter()
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
        timings["depth_postprocess"] = float(time.perf_counter() - depth_postprocess_start)

        primary_camera_center_aligned = cv_world_to_aligned_world(camera_models[primary_camera_id]["camera_center"])
        for label in primary_visible_labels:
            center_world_cv = known_center_world_cv(label, matches[label].center_world_cv)
            center_world_aligned = cv_world_to_aligned_world(center_world_cv)
            mask_full = masks_by_label[label][primary_camera_id]
            mask_crop = mask_full[crop_top:crop_bottom, crop_left:crop_right]
            valid = np.isfinite(depth_metric) & (np.abs(depth_metric) > 1e-6) & mask_crop
            target_points_cam = points_cam_img[valid]
            if len(target_points_cam) == 0:
                raise RuntimeError(f"No valid depth points for label '{label}' in {primary_camera_id}.")
            target_points_world_cv = camera_to_world_cv(camera_models[primary_camera_id]["extrinsic"], target_points_cam)
            target_points_world_aligned = cv_world_to_aligned_world(target_points_world_cv)
            comparison_points_by_label[label] = np.asarray(target_points_world_aligned, dtype=np.float32)
            eye_level = build_eye_level_pointcloud(
                target_points_world_aligned,
                center_world_aligned,
                primary_camera_center_aligned,
                robust_lower=float(args.robust_lower),
                robust_upper=float(args.robust_upper),
            )
            reference_height_m = float(np.asarray(eye_level["robust_extent_xyz_m"], dtype=np.float32).reshape(3)[1])
            world_y = target_points_world_aligned[:, 1]
            world_y_lo = float(np.percentile(world_y, float(args.robust_lower)))
            world_y_hi = float(np.percentile(world_y, float(args.robust_upper)))
            reference_height_m_by_label[label] = reference_height_m
            reference_height_details[label] = {
                "reference_camera_id": primary_camera_id,
                "center_world_m": center_world_aligned.astype(float).tolist(),
                "crop_box_xyxy": [crop_left, crop_top, crop_right, crop_bottom],
                "depth_affine_scale": float(affine_scale),
                "depth_affine_offset_m": float(affine_offset),
                "num_depth_points": int(len(target_points_world_aligned)),
                "reference_eye_level_height_m": reference_height_m,
                "reference_eye_level_height_mm": float(reference_height_m * 1000.0),
                "reference_world_y_height_m": max(0.0, world_y_hi - world_y_lo),
                "reference_world_y_height_mm": max(0.0, world_y_hi - world_y_lo) * 1000.0,
                "reference_eye_level_robust_extent_xyz_m": np.asarray(
                    eye_level["robust_extent_xyz_m"], dtype=np.float32
                ).astype(float).tolist(),
            }

        for label in labels:
            label_start = time.perf_counter()
            match = matches[label]
            center_world_cv = known_center_world_cv(label, match.center_world_cv)
            if label not in reference_height_m_by_label:
                raise RuntimeError(
                    f"Reference camera {primary_camera_id} does not provide depth height for label '{label}'."
                )
            reference_height_m = float(reference_height_m_by_label[label])

            vertical_line_fit = estimate_height_on_vertical_line(match, camera_models)
            vertical_line_height_m = (
                float(vertical_line_fit["height_min"])
                if args.vertical_line_height_stat == "min"
                else float(vertical_line_fit["height_median"])
            )
            bbox_height_estimates = height_estimates_from_bbox(match, camera_models)
            yolo_height_m = choose_final_height_m(
                bbox_height_estimates=bbox_height_estimates,
                vertical_line_height_m=vertical_line_height_m,
                min_valid_height_m=float(args.min_valid_height_m),
                max_valid_height_m=float(args.max_valid_height_m),
            )
            constrained_height_m = (
                min(float(reference_height_m), float(yolo_height_m))
                if not args.disable_yolo_height_cap
                else float(reference_height_m)
            )

            width_by_camera: dict[str, float] = {}
            per_camera: dict[str, dict[str, float | list[float]]] = {}
            for camera_id, det in match.detections.items():
                image_bgr = image_cache[camera_id]
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
                    reference_height_m=constrained_height_m,
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
                    float(constrained_height_m),
                    float(footprint_fit["size_z_m"]),
                ],
                dtype=np.float32,
            )

            results[label] = {
                "reference_base_dir": str(reference_base_dir),
                "reference_camera_id": primary_camera_id,
                "reference_height_m": float(reference_height_m),
                "reference_height_mm": float(reference_height_m * 1000.0),
                "yolo_height_cap_enabled": bool(not args.disable_yolo_height_cap),
                "yolo_height_m": float(yolo_height_m),
                "yolo_height_mm": float(yolo_height_m * 1000.0),
                "selected_height_m": float(constrained_height_m),
                "selected_height_mm": float(constrained_height_m * 1000.0),
                "height_source": (
                    "yolo_cap"
                    if (not args.disable_yolo_height_cap and yolo_height_m < reference_height_m)
                    else "reference_depth"
                ),
                "reference_center_world_m": reference_height_details[label]["center_world_m"],
                "multicam_center_world_m": cv_world_to_aligned_world(center_world_cv).astype(float).tolist(),
                "size_xyz_m": size_xyz_m.astype(float).tolist(),
                "size_xyz_mm": (size_xyz_m * 1000.0).astype(float).tolist(),
                "yaw_rad": float(-float(footprint_fit["yaw_rad"])),
                "yaw_deg": float(np.degrees(-float(footprint_fit["yaw_rad"]))),
                "reprojection_error_px": float(match.reprojection_error_px),
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
                "bbox_height_estimates_m": {
                    camera_id: float(value) for camera_id, value in bbox_height_estimates.items()
                },
                "vertical_line_height_median_m": float(vertical_line_fit["height_median"]),
                "vertical_line_height_min_m": float(vertical_line_fit["height_min"]),
                "vertical_line_height_selected_m": float(vertical_line_height_m),
                "reference_height_details": reference_height_details[label],
                "per_camera_measurements": per_camera,
            }
            label_times[label] = float(time.perf_counter() - label_start)
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

    summary = {
        "reference_base_dir": str(reference_base_dir),
        "reference_camera_id": str(args.reference_camera_id),
        "primary_visible_labels": primary_visible_labels,
        "camera_ids": camera_ids,
        "yolo_cache": yolo_cache,
        "timings_s": timings,
        "labels": results,
    }

    report_json = output_dir / "single_depth_multicam_sam_size_summary.json"
    report_md = output_dir / "single_depth_multicam_sam_size_summary.md"
    report_npz = output_dir / "single_depth_multicam_sam_size_summary.npz"
    eye_level_compare_npz = output_dir / "single_depth_multicam_sam_size_eye_level_compare.npz"
    report_write_start = time.perf_counter()
    report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_lines = [
        "# Single-Depth + Multi-Camera SAM Size Summary",
        "",
        "| Label | Width (mm) | Height (mm) | Depth (mm) | Yaw (deg) | Cameras | RMSE (mm) |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for label in labels:
        info = results[label]
        width_mm, height_mm, depth_mm = info["size_xyz_mm"]
        md_lines.append(
            f"| {label} | {width_mm:.3f} | {height_mm:.3f} | {depth_mm:.3f} | "
            f"{info['yaw_deg']:.3f} | {', '.join(info['selected_cameras'])} | "
            f"{float(info['footprint_fit_rmse_m']) * 1000.0:.3f} |"
        )
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    npz_payload: dict[str, np.ndarray] = {
        "labels": np.array(labels, dtype=object),
        "camera_ids": np.array(camera_ids, dtype=object),
        "reference_camera_id": np.array([str(args.reference_camera_id)], dtype=object),
        "primary_visible_labels": np.array(primary_visible_labels, dtype=object),
        "centers_world_m": np.stack(
            [np.asarray(results[label]["multicam_center_world_m"], dtype=np.float32) for label in labels], axis=0
        ),
        "sizes_xyz_m": np.stack(
            [np.asarray(results[label]["size_xyz_m"], dtype=np.float32) for label in labels], axis=0
        ),
        "sizes_xyz_mm": np.stack(
            [np.asarray(results[label]["size_xyz_mm"], dtype=np.float32) for label in labels], axis=0
        ),
        "yaw_rad": np.array([float(results[label]["yaw_rad"]) for label in labels], dtype=np.float32),
        "yaw_deg": np.array([float(results[label]["yaw_deg"]) for label in labels], dtype=np.float32),
        "reference_height_m": np.array(
            [float(results[label]["reference_height_m"]) for label in labels], dtype=np.float32
        ),
        "yolo_height_m": np.array([float(results[label]["yolo_height_m"]) for label in labels], dtype=np.float32),
        "selected_height_m": np.array(
            [float(results[label]["selected_height_m"]) for label in labels], dtype=np.float32
        ),
        "height_source": np.array([str(results[label]["height_source"]) for label in labels], dtype=object),
    }
    for label in labels:
        key = normalize_label(label).replace(" ", "_")
        info = results[label]
        npz_payload[f"{key}_center_world_m"] = np.asarray(info["multicam_center_world_m"], dtype=np.float32)
        npz_payload[f"{key}_size_xyz_m"] = np.asarray(info["size_xyz_m"], dtype=np.float32)
        npz_payload[f"{key}_size_xyz_mm"] = np.asarray(info["size_xyz_mm"], dtype=np.float32)
        npz_payload[f"{key}_yaw_rad"] = np.array([float(info["yaw_rad"])], dtype=np.float32)
        npz_payload[f"{key}_yaw_deg"] = np.array([float(info["yaw_deg"])], dtype=np.float32)
        npz_payload[f"{key}_reference_height_m"] = np.array([float(info["reference_height_m"])], dtype=np.float32)
        npz_payload[f"{key}_yolo_height_m"] = np.array([float(info["yolo_height_m"])], dtype=np.float32)
        npz_payload[f"{key}_selected_height_m"] = np.array([float(info["selected_height_m"])], dtype=np.float32)
    np.savez_compressed(str(report_npz), **npz_payload)
    comparison_payload = build_eye_level_comparison_payload(
        comparison_points_by_label,
        results,
        primary_camera_center_aligned=cv_world_to_aligned_world(camera_models[str(args.reference_camera_id)]["camera_center"]),
        max_points_per_object=1800,
        robust_lower=float(args.robust_lower),
        robust_upper=float(args.robust_upper),
    )
    np.savez_compressed(str(eye_level_compare_npz), **comparison_payload)
    summary["timings_s"]["report_write"] = float(time.perf_counter() - report_write_start)
    summary["timings_s"]["total"] = float(time.perf_counter() - total_start)
    summary["summary_npz_path"] = str(report_npz)
    summary["eye_level_compare_npz"] = str(eye_level_compare_npz)
    report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
