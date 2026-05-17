#!/usr/bin/env python3
"""
Run GraspGen on an eye-level target object with multi-camera obstacle geometry.

Target object:
  - uses its own eye-level projected 3D point cloud or aligned SAM3D mesh

Obstacle scene:
  - can still use single-camera eye-level points for debug modes
  - for primitive modes, estimates obstacle center + size from multi-camera YOLO
    geometry without running a depth model on every camera

Output is compatible with visualize_grasps.py.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from test_graspgen import filter_collisions, run_grasp_inference
from test_multicam_teddy_height import (
    DEFAULT_CAMERA_PARAMETER_DIR,
    DEFAULT_IMAGE_DIR,
    Detection,
    choose_final_height_m,
    default_weights_path,
    estimate_height_on_vertical_line,
    height_estimates_from_bbox,
    load_camera_models,
    match_target_detections,
    run_yolo,
)


DEFAULT_BASE_DIR = Path(
    "/home/tjchen/workspace/VLM_RL/RL_train/View_Agent/test_tmp/pics/Camera_Room1_test/"
    "four_camera_eye_level_experiment_outputs/Camera_Room1_12"
)
DEFAULT_TARGET_LABEL = "doll"
DEFAULT_LABELS = ("doll", "apple", "wine")
DEFAULT_FUSE_CAMERA_IDS = (
    "Camera_Room1_12",
    "Camera_Room1_13",
    "Camera_Room1_14",
    "Camera_Room1_15",
)
LABEL_COLORS = {
    "doll": np.array([60, 220, 90], dtype=np.uint8),
    "apple": np.array([255, 140, 40], dtype=np.uint8),
    "wine": np.array([70, 160, 255], dtype=np.uint8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run grasp generation on eye-level object point clouds.")
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--target-label", default=DEFAULT_TARGET_LABEL)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--target-source", choices=("sam3d", "eye_level"), default="sam3d")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--camera-parameter-dir", type=Path, default=DEFAULT_CAMERA_PARAMETER_DIR)
    parser.add_argument("--weights", type=Path, default=default_weights_path())
    parser.add_argument("--conf-thresh", type=float, default=0.20)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--obstacle-mode",
        choices=("projection", "positioned_3d", "box", "cylinder", "polygon_prism"),
        default="box",
    )
    parser.add_argument("--fuse-camera-ids", nargs="+", default=list(DEFAULT_FUSE_CAMERA_IDS))
    parser.add_argument("--obstacle-min-views", type=int, default=None)
    parser.add_argument("--obstacle-min-valid-height-m", type=float, default=0.03)
    parser.add_argument("--obstacle-max-valid-height-m", type=float, default=1.50)
    parser.add_argument("--obstacle-height-offset-mm", type=float, default=0.0)
    parser.add_argument(
        "--obstacle-vertical-line-height-stat",
        choices=("min", "median"),
        default="median",
    )
    parser.add_argument("--obstacle-point-budget", type=int, default=1200)
    parser.add_argument("--polygon-sides", type=int, default=8)
    parser.add_argument("--num-grasps", type=int, default=200)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--collision-thresh", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_eye_level_npz(base_dir: Path, label: str) -> dict[str, np.ndarray]:
    npz_path = base_dir / f"{label}_eye_level_outputs" / f"{label}_eye_level_points.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    return dict(np.load(str(npz_path), allow_pickle=True))


def yolo_cache_path(base_dir: Path, camera_ids: list[str]) -> Path:
    tag = "_".join(camera_ids)
    return base_dir / f"obstacle_yolo_cache_{tag}.json"


def detections_to_jsonable(detections_by_camera: dict[str, list[Detection]]) -> dict[str, list[dict[str, object]]]:
    return {
        camera_id: [det.to_dict() for det in dets]
        for camera_id, dets in detections_by_camera.items()
    }


def detections_from_jsonable(payload: dict[str, list[dict[str, object]]]) -> dict[str, list[Detection]]:
    restored: dict[str, list[Detection]] = {}
    for camera_id, dets in payload.items():
        restored[camera_id] = [
            Detection(
                camera_id=str(det["camera_id"]),
                label=str(det["label"]),
                conf=float(det["confidence"]),
                x1=float(det["bbox_xyxy"][0]),
                y1=float(det["bbox_xyxy"][1]),
                x2=float(det["bbox_xyxy"][2]),
                y2=float(det["bbox_xyxy"][3]),
            )
            for det in dets
        ]
    return restored


def load_or_run_obstacle_yolo(
    *,
    base_dir: Path,
    camera_ids: list[str],
    image_dir: Path,
    weights: Path,
    conf_thresh: float,
    device: str,
) -> tuple[dict[str, list[Detection]], str]:
    cache_path = yolo_cache_path(base_dir, camera_ids)
    cache_key = {
        "camera_ids": camera_ids,
        "image_dir": str(image_dir),
        "weights": str(weights),
        "conf_thresh": float(conf_thresh),
        "device": str(device),
    }
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("cache_key") == cache_key:
                return detections_from_jsonable(payload["detections_by_camera"]), str(cache_path)
        except Exception:
            pass

    detections_by_camera = run_yolo(
        camera_ids=camera_ids,
        image_dir=image_dir,
        weights=weights,
        conf_thresh=conf_thresh,
        device=device,
    )
    cache_payload = {
        "cache_key": cache_key,
        "detections_by_camera": detections_to_jsonable(detections_by_camera),
    }
    cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")
    return detections_by_camera, str(cache_path)


def resolve_fuse_base_dirs(base_dir: Path, camera_ids: list[str]) -> list[Path]:
    parent = base_dir.parent
    dirs: list[Path] = []
    for camera_id in camera_ids:
        candidate = parent / str(camera_id)
        if candidate.exists():
            dirs.append(candidate)
    if not dirs:
        raise FileNotFoundError(f"No fuse camera directories found under {parent} for {camera_ids}")
    return dirs


def point_colors(label: str, count: int) -> np.ndarray:
    color = LABEL_COLORS.get(label, np.array([180, 180, 180], dtype=np.uint8))
    return np.repeat(color.reshape(1, 3), count, axis=0)


def cv_world_to_aligned_world(point_world_cv: np.ndarray) -> np.ndarray:
    point_world = np.asarray(point_world_cv, dtype=np.float32).reshape(3).copy()
    point_world[2] *= -1.0
    return point_world


def default_output_paths(base_dir: Path, target_label: str) -> tuple[Path, Path]:
    tag = target_label.replace(" ", "_")
    result_npz = base_dir / f"{tag}_eye_level_grasp_result.npz"
    report_json = base_dir / f"{tag}_eye_level_grasp_report.json"
    return result_npz, report_json


def build_projection_scene(data: dict[str, np.ndarray]) -> np.ndarray:
    center_world = np.asarray(data["center_world_m"], dtype=np.float32).reshape(3)
    projected_front_xy = np.asarray(data["projected_front_xy"], dtype=np.float32)
    points = np.column_stack(
        [
            projected_front_xy[:, 0] + center_world[0],
            projected_front_xy[:, 1] + center_world[1],
            np.full(len(projected_front_xy), center_world[2], dtype=np.float32),
        ]
    ).astype(np.float32)
    return points


def obstacle_size_xyz(data: dict[str, np.ndarray], size_source: str) -> np.ndarray:
    key = "robust_extent_xyz_m" if size_source == "robust" else "full_extent_xyz_m"
    if key not in data:
        raise KeyError(f"Missing obstacle size key: {key}")
    size_xyz = np.asarray(data[key], dtype=np.float32).reshape(3)
    return np.clip(size_xyz, 0.01, None)


def grid_resolution_for_budget(size_xyz: np.ndarray, budget: int) -> tuple[int, int, int]:
    sx, sy, sz = [max(float(v), 1e-4) for v in size_xyz]
    volume = sx * sy * sz
    if volume <= 1e-12:
        return (4, 4, 4)
    scale = (max(int(budget), 64) / volume) ** (1.0 / 3.0)
    nx = max(3, int(round(sx * scale)))
    ny = max(3, int(round(sy * scale)))
    nz = max(3, int(round(sz * scale)))
    return nx, ny, nz


def yaw_rotation_matrix(yaw_rad: float) -> np.ndarray:
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


def transform_local_points(local_points: np.ndarray, center_world: np.ndarray, yaw_rad: float) -> np.ndarray:
    rotation = yaw_rotation_matrix(yaw_rad)
    return local_points @ rotation.T + center_world.reshape(1, 3)


def sample_box_volume_local(size_xyz: np.ndarray, budget: int) -> np.ndarray:
    nx, ny, nz = grid_resolution_for_budget(size_xyz, budget)
    xs = np.linspace(-size_xyz[0] * 0.5, size_xyz[0] * 0.5, nx, dtype=np.float32)
    ys = np.linspace(-size_xyz[1] * 0.5, size_xyz[1] * 0.5, ny, dtype=np.float32)
    zs = np.linspace(-size_xyz[2] * 0.5, size_xyz[2] * 0.5, nz, dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)
    return pts


def sample_cylinder_volume_local(size_xyz: np.ndarray, budget: int) -> np.ndarray:
    diameter = max(float(size_xyz[0]), float(size_xyz[2]))
    radius = max(diameter * 0.5, 0.005)
    height = max(float(size_xyz[1]), 0.01)
    num_heights = max(6, int(round(math.sqrt(max(budget, 64)) / 2)))
    num_angles = max(16, int(round(math.sqrt(max(budget, 64)) * 2)))
    num_radii = max(5, int(round(math.sqrt(max(budget, 64)) / 2)))
    ys = np.linspace(-height * 0.5, height * 0.5, num_heights, dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * math.pi, num_angles, endpoint=False, dtype=np.float32)
    radii = np.linspace(0.0, radius, num_radii, dtype=np.float32)
    pts = []
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    for y in ys:
        for r in radii:
            ring = np.column_stack(
                [
                    r * cos_a,
                    np.full_like(cos_a, y, dtype=np.float32),
                    r * sin_a,
                ]
            )
            pts.append(ring)
    return np.vstack(pts).astype(np.float32)


def regular_polygon_vertices(rx: float, rz: float, sides: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False, dtype=np.float32)
    return np.column_stack([rx * np.cos(angles), rz * np.sin(angles)]).astype(np.float32)


def point_in_polygon(points_xz: np.ndarray, polygon_xz: np.ndarray) -> np.ndarray:
    x = points_xz[:, 0]
    z = points_xz[:, 1]
    px = polygon_xz[:, 0]
    pz = polygon_xz[:, 1]
    inside = np.zeros(len(points_xz), dtype=bool)
    j = len(polygon_xz) - 1
    for i in range(len(polygon_xz)):
        intersects = ((pz[i] > z) != (pz[j] > z)) & (
            x < (px[j] - px[i]) * (z - pz[i]) / (pz[j] - pz[i] + 1e-8) + px[i]
        )
        inside ^= intersects
        j = i
    return inside


def sample_polygon_prism_volume_local(size_xyz: np.ndarray, budget: int, sides: int) -> np.ndarray:
    rx = max(float(size_xyz[0]) * 0.5, 0.005)
    rz = max(float(size_xyz[2]) * 0.5, 0.005)
    height = max(float(size_xyz[1]), 0.01)
    nx, ny, nz = grid_resolution_for_budget(size_xyz, budget)
    xs = np.linspace(-rx, rx, nx, dtype=np.float32)
    ys = np.linspace(-height * 0.5, height * 0.5, ny, dtype=np.float32)
    zs = np.linspace(-rz, rz, nz, dtype=np.float32)
    xx, zz = np.meshgrid(xs, zs, indexing="xy")
    base_points = np.column_stack([xx.ravel(), zz.ravel()]).astype(np.float32)
    poly = regular_polygon_vertices(rx, rz, max(int(sides), 3))
    keep = point_in_polygon(base_points, poly)
    base_inside = base_points[keep]
    layers = []
    for y in ys:
        layer = np.column_stack(
            [
                base_inside[:, 0],
                np.full(len(base_inside), y, dtype=np.float32),
                base_inside[:, 1],
            ]
        )
        layers.append(layer)
    pts = np.vstack(layers).astype(np.float32)
    return pts


def build_primitive_obstacle(
    center_world: np.ndarray,
    size_xyz: np.ndarray,
    mode: str,
    point_budget: int,
    polygon_sides: int,
    yaw_rad: float = 0.0,
) -> np.ndarray:
    if mode == "box":
        local_pts = sample_box_volume_local(size_xyz, point_budget)
        return transform_local_points(local_pts, center_world, yaw_rad)
    if mode == "cylinder":
        local_pts = sample_cylinder_volume_local(size_xyz, point_budget)
        return transform_local_points(local_pts, center_world, yaw_rad)
    if mode == "polygon_prism":
        local_pts = sample_polygon_prism_volume_local(size_xyz, point_budget, polygon_sides)
        return transform_local_points(local_pts, center_world, yaw_rad)
    raise ValueError(f"Unsupported primitive mode: {mode}")


def bbox_width_estimates_from_height(match, height_m: float) -> dict[str, float]:
    estimates: dict[str, float] = {}
    if height_m <= 1e-6:
        return estimates
    for camera_id, det in match.detections.items():
        if det.bbox_height_px <= 1e-6:
            continue
        estimates[camera_id] = float(height_m * det.bbox_width_px / det.bbox_height_px)
    return estimates


def camera_azimuth_about_object(center_world_cv: np.ndarray, camera_center_world_cv: np.ndarray) -> float:
    dx = float(camera_center_world_cv[0] - center_world_cv[0])
    dz = float(camera_center_world_cv[2] - center_world_cv[2])
    return float(math.atan2(dz, dx))


def fit_rotated_rectangle_footprint(
    center_world_cv: np.ndarray,
    width_by_camera: dict[str, float],
    camera_models: dict[str, dict[str, np.ndarray]],
) -> dict[str, object]:
    camera_ids = [camera_id for camera_id in width_by_camera if camera_id in camera_models]
    if not camera_ids:
        raise RuntimeError("No valid multi-camera width estimates available.")

    observed_widths = np.asarray([width_by_camera[camera_id] for camera_id in camera_ids], dtype=np.float64)
    azimuths = np.asarray(
        [
            camera_azimuth_about_object(center_world_cv, camera_models[camera_id]["camera_center"])
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
            "predicted_width_by_camera": {camera_ids[0]: median_width},
            "azimuth_rad_by_camera": {camera_ids[0]: float(azimuths[0])},
        }

    best_result: dict[str, object] | None = None
    phi_values = np.linspace(0.0, math.pi, 721, dtype=np.float64)
    min_size = max(0.02, 0.25 * median_width)

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
        predicted = basis @ np.array([size_x, size_z], dtype=np.float64)
        rmse = float(np.sqrt(np.mean((predicted - observed_widths) ** 2)))
        if best_result is None or rmse < float(best_result["fit_rmse_m"]):
            best_result = {
                "size_x_m": size_x,
                "size_z_m": size_z,
                "yaw_rad": float(phi),
                "fit_rmse_m": rmse,
                "predicted_width_by_camera": {
                    camera_id: float(width_value)
                    for camera_id, width_value in zip(camera_ids, predicted.tolist())
                },
                "azimuth_rad_by_camera": {
                    camera_id: float(azimuth_value)
                    for camera_id, azimuth_value in zip(camera_ids, azimuths.tolist())
                },
            }

    assert best_result is not None
    return best_result


def estimate_multicam_obstacle_geometry(
    label: str,
    detections_by_camera: dict[str, list[object]],
    camera_models: dict[str, dict[str, np.ndarray]],
    min_views: int,
    min_valid_height_m: float,
    max_valid_height_m: float,
    height_offset_mm: float,
    vertical_line_height_stat: str,
) -> dict[str, object]:
    match = match_target_detections(
        detections_by_camera=detections_by_camera,
        camera_models=camera_models,
        target_labels=[label],
        min_views=min_views,
    )
    vertical_line_fit = estimate_height_on_vertical_line(match, camera_models)
    vertical_line_height_m = (
        float(vertical_line_fit["height_min"])
        if vertical_line_height_stat == "min"
        else float(vertical_line_fit["height_median"])
    )
    bbox_height_by_camera = height_estimates_from_bbox(match, camera_models)
    height_m = choose_final_height_m(
        bbox_height_estimates=bbox_height_by_camera,
        vertical_line_height_m=vertical_line_height_m,
        min_valid_height_m=min_valid_height_m,
        max_valid_height_m=max_valid_height_m,
    )
    height_m = max(0.02, height_m + height_offset_mm / 1000.0)

    width_by_camera = bbox_width_estimates_from_height(match, height_m)
    footprint_fit = fit_rotated_rectangle_footprint(match.center_world_cv, width_by_camera, camera_models)
    center_world = cv_world_to_aligned_world(match.center_world_cv)
    yaw_aligned_world = -float(footprint_fit["yaw_rad"])
    size_xyz = np.array(
        [
            float(footprint_fit["size_x_m"]),
            float(height_m),
            float(footprint_fit["size_z_m"]),
        ],
        dtype=np.float32,
    )
    return {
        "center_world_m": center_world,
        "size_xyz_m": size_xyz,
        "yaw_rad": yaw_aligned_world,
        "yaw_deg": float(np.degrees(yaw_aligned_world)),
        "reprojection_error_px": float(match.reprojection_error_px),
        "mean_confidence": float(match.mean_conf),
        "selected_cameras": list(match.detections.keys()),
        "bbox_height_by_camera_m": bbox_height_by_camera,
        "bbox_width_by_camera_m": width_by_camera,
        "vertical_line_height_m": vertical_line_height_m,
        "estimated_height_m": float(height_m),
        "footprint_fit_rmse_m": float(footprint_fit["fit_rmse_m"]),
        "predicted_width_by_camera_m": footprint_fit["predicted_width_by_camera"],
        "camera_azimuth_rad_by_camera": {
            camera_id: -float(value)
            for camera_id, value in footprint_fit["azimuth_rad_by_camera"].items()
        },
    }


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    base_dir = args.base_dir.expanduser().resolve()
    image_dir = args.image_dir.expanduser().resolve()
    camera_parameter_dir = args.camera_parameter_dir.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    target_label = str(args.target_label).lower()
    labels = [str(label).lower() for label in args.labels]
    if target_label not in labels:
        raise ValueError(f"Target label '{target_label}' must be included in --labels.")

    target_data = load_eye_level_npz(base_dir, target_label)
    loaded_obstacles = {}
    if args.obstacle_mode in {"projection", "positioned_3d"}:
        loaded_obstacles = {label: load_eye_level_npz(base_dir, label) for label in labels if label != target_label}

    camera_ids = [str(camera_id) for camera_id in args.fuse_camera_ids]
    obstacle_min_views = args.obstacle_min_views if args.obstacle_min_views is not None else len(camera_ids)
    if obstacle_min_views < 2:
        raise ValueError("--obstacle-min-views must be at least 2.")
    if obstacle_min_views > len(camera_ids):
        raise ValueError("--obstacle-min-views cannot exceed the number of --fuse-camera-ids.")

    obstacle_camera_models = None
    obstacle_detections_by_camera = None
    obstacle_yolo_time_s = 0.0
    obstacle_yolo_cache = None
    if args.obstacle_mode in {"box", "cylinder", "polygon_prism"}:
        obstacle_camera_models = load_camera_models(camera_ids, camera_parameter_dir)
        obstacle_yolo_start = time.perf_counter()
        obstacle_detections_by_camera, obstacle_yolo_cache = load_or_run_obstacle_yolo(
            base_dir=base_dir,
            camera_ids=camera_ids,
            image_dir=image_dir,
            weights=weights,
            conf_thresh=args.conf_thresh,
            device=args.device,
        )
        obstacle_yolo_time_s = time.perf_counter() - obstacle_yolo_start

    if args.target_source == "sam3d":
        sam3d_key = "sam3d_aligned_mesh_vertices_world_positioned"
        if sam3d_key not in target_data:
            raise KeyError(
                f"Target '{target_label}' does not contain {sam3d_key}. "
                "Run test_add_sam3d_rotation_align_to_npz.py first or use --target-source eye_level."
            )
        target_points_world = np.asarray(target_data[sam3d_key], dtype=np.float32)
    else:
        target_points_world = np.asarray(
            target_data["object_points_eye_level_world_positioned"], dtype=np.float32
        )
    if len(target_points_world) == 0:
        raise RuntimeError(f"Empty target points for label '{target_label}'.")
    target_colors = point_colors(target_label, len(target_points_world))

    obstacle_points = []
    obstacle_colors = []
    obstacle_labels = []
    obstacle_geometry = {}
    obstacle_geometry_start = time.perf_counter()
    for label in labels:
        if label == target_label:
            continue
        if args.obstacle_mode == "projection":
            pts = build_projection_scene(loaded_obstacles[label])
            obstacle_geometry[label] = {
                "mode": "projection",
                "used_camera_dirs": [base_dir.name],
            }
        elif args.obstacle_mode in {"box", "cylinder", "polygon_prism"}:
            assert obstacle_camera_models is not None
            assert obstacle_detections_by_camera is not None
            geometry = estimate_multicam_obstacle_geometry(
                label=label,
                detections_by_camera=obstacle_detections_by_camera,
                camera_models=obstacle_camera_models,
                min_views=obstacle_min_views,
                min_valid_height_m=args.obstacle_min_valid_height_m,
                max_valid_height_m=args.obstacle_max_valid_height_m,
                height_offset_mm=args.obstacle_height_offset_mm,
                vertical_line_height_stat=args.obstacle_vertical_line_height_stat,
            )
            pts = build_primitive_obstacle(
                geometry["center_world_m"],
                geometry["size_xyz_m"],
                mode=args.obstacle_mode,
                point_budget=args.obstacle_point_budget,
                polygon_sides=args.polygon_sides,
                yaw_rad=float(geometry["yaw_rad"]),
            )
            obstacle_geometry[label] = {
                "mode": args.obstacle_mode,
                "geometry_source": "multicam_bbox_geometry",
                "used_camera_ids": geometry["selected_cameras"],
                "center_world_m": np.asarray(geometry["center_world_m"], dtype=float).tolist(),
                "size_xyz_m": np.asarray(geometry["size_xyz_m"], dtype=float).tolist(),
                "size_xyz_mm": (np.asarray(geometry["size_xyz_m"], dtype=float) * 1000.0).tolist(),
                "yaw_rad": float(geometry["yaw_rad"]),
                "yaw_deg": float(geometry["yaw_deg"]),
                "estimated_height_m": float(geometry["estimated_height_m"]),
                "vertical_line_height_m": float(geometry["vertical_line_height_m"]),
                "bbox_height_by_camera_m": geometry["bbox_height_by_camera_m"],
                "bbox_width_by_camera_m": geometry["bbox_width_by_camera_m"],
                "predicted_width_by_camera_m": geometry["predicted_width_by_camera_m"],
                "camera_azimuth_rad_by_camera": geometry["camera_azimuth_rad_by_camera"],
                "reprojection_error_px": float(geometry["reprojection_error_px"]),
                "mean_confidence": float(geometry["mean_confidence"]),
                "footprint_fit_rmse_m": float(geometry["footprint_fit_rmse_m"]),
            }
        else:
            pts = np.asarray(loaded_obstacles[label]["object_points_eye_level_world_positioned"], dtype=np.float32)
            obstacle_geometry[label] = {
                "mode": "positioned_3d",
                "used_camera_dirs": [base_dir.name],
            }
        if len(pts) == 0:
            continue
        obstacle_points.append(pts)
        obstacle_colors.append(point_colors(label, len(pts)))
        obstacle_labels.append(label)
    obstacle_geometry_time_s = time.perf_counter() - obstacle_geometry_start

    if obstacle_points:
        scene_pc = np.concatenate(obstacle_points, axis=0).astype(np.float32)
        scene_colors = np.concatenate(obstacle_colors, axis=0).astype(np.uint8)
    else:
        scene_pc = np.zeros((0, 3), dtype=np.float32)
        scene_colors = np.zeros((0, 3), dtype=np.uint8)

    grasp_start = time.perf_counter()
    pc_c, obj_colors_c, grasps_c, conf, scores, t_center, cfg = run_grasp_inference(
        target_points_world,
        target_colors,
        num_grasps=args.num_grasps,
        topk=args.topk,
    )
    grasp_inference_time_s = time.perf_counter() - grasp_start

    import trimesh.transformations as tra  # type: ignore

    object_pc_raw_c = tra.transform_points(target_points_world, t_center)
    if len(scene_pc) > 0:
        collision_start = time.perf_counter()
        coll_mask, scene_c = filter_collisions(
            scene_pc,
            grasps_c,
            t_center,
            cfg,
            collision_threshold=args.collision_thresh,
        )
        collision_time_s = time.perf_counter() - collision_start
        scene_raw_c = tra.transform_points(scene_pc, t_center)
    else:
        coll_mask = np.ones(len(grasps_c), dtype=bool)
        scene_c = np.zeros((0, 3), dtype=np.float32)
        scene_raw_c = scene_c.copy()
        collision_time_s = 0.0

    free_grasps = grasps_c[coll_mask]
    free_conf = conf[coll_mask]

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = target_label.replace(" ", "_")
        result_npz = output_dir / f"{tag}_eye_level_grasp_result.npz"
        report_json = output_dir / f"{tag}_eye_level_grasp_report.json"
    else:
        result_npz, report_json = default_output_paths(base_dir, target_label)

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
        obstacle_labels=np.array(obstacle_labels, dtype=object),
    )
    if obj_colors_c is not None:
        save_data["pc_object_colors"] = obj_colors_c
    save_data["pc_object_raw_colors"] = target_colors

    result_write_start = time.perf_counter()
    np.savez_compressed(str(result_npz), **save_data)
    result_write_time_s = time.perf_counter() - result_write_start

    report = {
        "base_dir": str(base_dir),
        "image_dir": str(image_dir),
        "camera_parameter_dir": str(camera_parameter_dir),
        "weights": str(weights),
        "target_label": target_label,
        "target_source": args.target_source,
        "obstacle_mode": args.obstacle_mode,
        "fuse_camera_ids": [str(camera_id) for camera_id in args.fuse_camera_ids],
        "obstacle_yolo_cache": obstacle_yolo_cache,
        "obstacle_min_views": int(obstacle_min_views),
        "obstacle_vertical_line_height_stat": args.obstacle_vertical_line_height_stat,
        "obstacle_height_offset_mm": float(args.obstacle_height_offset_mm),
        "obstacle_point_budget": int(args.obstacle_point_budget),
        "polygon_sides": int(args.polygon_sides),
        "labels": labels,
        "obstacle_labels": obstacle_labels,
        "obstacle_geometry": obstacle_geometry,
        "num_object_points_raw": int(len(target_points_world)),
        "num_scene_points": int(len(scene_pc)),
        "num_total_grasps": int(len(grasps_c)),
        "num_collision_free_grasps": int(int(coll_mask.sum())),
        "collision_threshold": float(args.collision_thresh),
        "result_npz": str(result_npz),
        "timings_s": {
            "obstacle_yolo_inference": float(obstacle_yolo_time_s),
            "obstacle_geometry_estimation": float(obstacle_geometry_time_s),
            "grasp_inference": float(grasp_inference_time_s),
            "collision_filter": float(collision_time_s),
            "result_npz_write": float(result_write_time_s),
            "report_write": 0.0,
            "total": 0.0,
        },
    }
    report_write_start = time.perf_counter()
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["timings_s"]["report_write"] = float(time.perf_counter() - report_write_start)
    report["timings_s"]["total"] = float(time.perf_counter() - total_start)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
