#!/usr/bin/env python3
"""
Rebuild the old single-camera eye-level reference NPZ files that were used by:
  - test_multicam_single_depth_sam_size.py
  - test_eye_level_grasp.py

Method:
  1. Run YOLO on one reference camera image.
  2. Segment each requested object with SAM.
  3. Run one crop-level Depth Anything pass.
  4. Calibrate depth with known 3D world centers.
  5. Backproject object depth points to world.
  6. Convert each object point cloud into eye-level canonical coordinates.
  7. Save <label>_eye_level_points.npz in the legacy format.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from test_unified_single_depth_multicam_pipeline import (
    DEFAULT_CAMERA_PARAMETER_DIR,
    DEFAULT_IMAGE_DIR,
    DEFAULT_KNOWN_CENTERS_ALIGNED_WORLD,
    DEFAULT_PRIMARY_CAMERA_ID,
    backproject_depth,
    build_eye_level_pointcloud,
    build_sam_predictor,
    camera_to_world_cv,
    crop_intrinsic,
    cv_world_to_aligned_world,
    default_depth_weights_path,
    fit_depth_affine,
    image_path_for_camera,
    known_center_world_cv,
    load_camera_models,
    load_or_run_obstacle_yolo,
    sample_depth_value,
    segment_mask_for_detection,
    union_bbox_from_detections,
)
from test_eye_level_grasp import DEFAULT_BASE_DIR, default_weights_path
from test_multicam_teddy_height import Detection, normalize_label
from src.seg_crop_depth import infer_depth_image, load_depth_model  # type: ignore


DEFAULT_LABELS = ("doll", "apple", "wine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild legacy eye-level reference NPZ files from one depth camera.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--camera-parameter-dir", type=Path, default=DEFAULT_CAMERA_PARAMETER_DIR)
    parser.add_argument("--primary-camera-id", default=DEFAULT_PRIMARY_CAMERA_ID)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--weights", type=Path, default=default_weights_path())
    parser.add_argument("--conf-thresh", type=float, default=0.20)
    parser.add_argument("--device", default="")
    parser.add_argument("--sam-checkpoint", type=Path, default=None)
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--depth-weights", type=Path, default=default_depth_weights_path())
    parser.add_argument("--depth-input-size", type=int, default=518)
    parser.add_argument("--robust-lower", type=float, default=5.0)
    parser.add_argument("--robust-upper", type=float, default=95.0)
    parser.add_argument("--point-budget", type=int, default=1800)
    return parser.parse_args()


def select_detections_for_labels(
    detections_by_camera: dict[str, list[Detection]],
    camera_id: str,
    labels: list[str],
) -> dict[str, Detection]:
    dets = detections_by_camera.get(camera_id, [])
    selected: dict[str, Detection] = {}
    for label in labels:
        label_norm = normalize_label(label)
        candidates = [det for det in dets if normalize_label(det.label) == label_norm]
        if not candidates:
            raise RuntimeError(f"No YOLO detection for label '{label}' in {camera_id}.")
        selected[label_norm] = max(candidates, key=lambda det: float(det.conf))
    return selected


def default_output_dir(base_dir: Path, label: str) -> Path:
    return base_dir / f"{label}_eye_level_outputs"


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()

    base_dir = args.base_dir.expanduser().resolve()
    image_dir = args.image_dir.expanduser().resolve()
    camera_parameter_dir = args.camera_parameter_dir.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    primary_camera_id = str(args.primary_camera_id)
    labels = [normalize_label(label) for label in args.labels]
    sam_checkpoint = args.sam_checkpoint.expanduser().resolve() if args.sam_checkpoint else None
    if sam_checkpoint is None:
        from multicam_volume_utils import default_sam_checkpoint  # local import to avoid unused import at module level

        sam_checkpoint = default_sam_checkpoint()
    if not sam_checkpoint.exists():
        raise FileNotFoundError(sam_checkpoint)
    if not args.depth_weights.exists():
        raise FileNotFoundError(args.depth_weights)

    timings: dict[str, object] = {}

    camera_load_start = time.perf_counter()
    camera_models = load_camera_models([primary_camera_id], camera_parameter_dir)
    camera_model = camera_models[primary_camera_id]
    timings["camera_model_load"] = float(time.perf_counter() - camera_load_start)

    yolo_start = time.perf_counter()
    detections_by_camera, yolo_cache = load_or_run_obstacle_yolo(
        base_dir=base_dir,
        camera_ids=[primary_camera_id],
        image_dir=image_dir,
        weights=weights,
        conf_thresh=float(args.conf_thresh),
        device=str(args.device),
    )
    timings["yolo_inference"] = float(time.perf_counter() - yolo_start)
    timings["yolo_cache"] = yolo_cache

    selected_dets = select_detections_for_labels(detections_by_camera, primary_camera_id, labels)
    image_path = image_path_for_camera(image_dir, primary_camera_id)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    sam_model_start = time.perf_counter()
    sam_device = str(args.device or "cuda")
    sam_model, sam_predictor = build_sam_predictor(args.sam_model_type, sam_checkpoint, sam_device)
    timings["sam_model_load"] = float(time.perf_counter() - sam_model_start)

    sam_infer_start = time.perf_counter()
    masks_by_label: dict[str, np.ndarray] = {}
    try:
        for label, det in selected_dets.items():
            masks_by_label[label] = segment_mask_for_detection(sam_predictor, image_bgr, det)
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
    timings["sam_inference_total"] = float(time.perf_counter() - sam_infer_start)

    crop_left, crop_top, crop_right, crop_bottom = union_bbox_from_detections(list(selected_dets.values()), image_bgr.shape)
    crop_bgr = image_bgr[crop_top:crop_bottom, crop_left:crop_right]
    crop_k = crop_intrinsic(camera_model["K"], crop_left, crop_top)

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

    depth_calib_start = time.perf_counter()
    depth_rel_values: list[float] = []
    true_z_values: list[float] = []
    for label, det in selected_dets.items():
        center_uv_crop = det.center_uv - np.array([crop_left, crop_top], dtype=np.float32)
        depth_rel_values.append(sample_depth_value(depth_rel, center_uv_crop))
        center_world_cv = known_center_world_cv(label, np.zeros(3, dtype=np.float32))
        point_cam = camera_model["extrinsic"][:, :3] @ center_world_cv + camera_model["extrinsic"][:, 3]
        true_z_values.append(float(point_cam[2]))
    affine_scale, affine_offset = fit_depth_affine(depth_rel_values, true_z_values)
    depth_metric = affine_scale * depth_rel + affine_offset
    points_cam_img = backproject_depth(depth_metric, crop_k)
    timings["depth_postprocess"] = float(time.perf_counter() - depth_calib_start)

    primary_camera_center_aligned = cv_world_to_aligned_world(camera_model["camera_center"])
    per_label_times: dict[str, float] = {}
    summary_labels: dict[str, object] = {}

    for label in labels:
        label_start = time.perf_counter()
        mask_full = masks_by_label[label]
        mask_crop = mask_full[crop_top:crop_bottom, crop_left:crop_right]
        valid = np.isfinite(depth_metric) & (np.abs(depth_metric) > 1e-6) & mask_crop
        target_points_cam = points_cam_img[valid]
        if len(target_points_cam) == 0:
            raise RuntimeError(f"No valid depth points for label '{label}'.")

        target_points_world_cv = camera_to_world_cv(camera_model["extrinsic"], target_points_cam)
        target_points_world_aligned = cv_world_to_aligned_world(target_points_world_cv).astype(np.float32)
        center_world_aligned = np.asarray(DEFAULT_KNOWN_CENTERS_ALIGNED_WORLD[label], dtype=np.float32)
        eye_level = build_eye_level_pointcloud(
            target_points_world_aligned,
            center_world_aligned,
            primary_camera_center_aligned,
            robust_lower=float(args.robust_lower),
            robust_upper=float(args.robust_upper),
        )

        output_dir = default_output_dir(base_dir, label)
        output_dir.mkdir(parents=True, exist_ok=True)
        npz_path = output_dir / f"{label}_eye_level_points.npz"
        np.savez_compressed(
            str(npz_path),
            center_world_m=center_world_aligned.astype(np.float32),
            object_points_world_positioned=target_points_world_aligned.astype(np.float32),
            object_points_eye_level_centered=np.asarray(eye_level["points_eye_level_centered"], dtype=np.float32),
            object_points_eye_level_world_positioned=np.asarray(
                eye_level["points_eye_level_world_positioned"], dtype=np.float32
            ),
            projected_front_xy=np.asarray(eye_level["projected_front_xy"], dtype=np.float32),
            robust_extent_xyz_m=np.asarray(eye_level["robust_extent_xyz_m"], dtype=np.float32),
            full_extent_xyz_m=np.asarray(eye_level["full_extent_xyz_m"], dtype=np.float32),
            eye_level_axes_world=np.asarray(eye_level["eye_level_axes_world"], dtype=np.float32),
            eye_level_pca_centroid_world_m=np.asarray(eye_level["eye_level_pca_centroid_world_m"], dtype=np.float32),
            primary_camera_id=np.array([primary_camera_id]),
            crop_box_xyxy=np.array([crop_left, crop_top, crop_right, crop_bottom], dtype=np.int32),
            depth_affine_scale=np.array([affine_scale], dtype=np.float32),
            depth_affine_offset_m=np.array([affine_offset], dtype=np.float32),
        )

        summary_labels[label] = {
            "npz_path": str(npz_path),
            "center_world_m": center_world_aligned.astype(float).tolist(),
            "robust_extent_xyz_mm": (np.asarray(eye_level["robust_extent_xyz_m"], dtype=np.float32) * 1000.0)
            .astype(float)
            .tolist(),
            "full_extent_xyz_mm": (np.asarray(eye_level["full_extent_xyz_m"], dtype=np.float32) * 1000.0)
            .astype(float)
            .tolist(),
            "num_world_points": int(len(target_points_world_aligned)),
            "bbox_xyxy": [float(selected_dets[label].x1), float(selected_dets[label].y1), float(selected_dets[label].x2), float(selected_dets[label].y2)],
        }
        per_label_times[label] = float(time.perf_counter() - label_start)

    summary = {
        "base_dir": str(base_dir),
        "primary_camera_id": primary_camera_id,
        "crop_box_xyxy": [crop_left, crop_top, crop_right, crop_bottom],
        "depth_affine_scale": float(affine_scale),
        "depth_affine_offset_m": float(affine_offset),
        "timings_s": {
            **timings,
            "per_label": per_label_times,
            "total": float(time.perf_counter() - total_start),
        },
        "labels": summary_labels,
    }
    summary_path = base_dir / "rebuilt_eye_level_points_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
