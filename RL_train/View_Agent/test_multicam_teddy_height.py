#!/usr/bin/env python3
"""
Run YOLO on Camera_Room1_2/7/10, localize a target object in 3D, and estimate its height.

Default inputs:
  - test_tmp/pics/Camera_Room1_test/Camera_Room1_2_rgb.png
  - test_tmp/pics/Camera_Room1_test/Camera_Room1_7_rgb.png
  - test_tmp/pics/Camera_Room1_test/Camera_Room1_10_rgb.png

Outputs:
  - test_tmp/pics/Camera_Room1_test/<target>_height_outputs/*_annotated.png
  - test_tmp/pics/Camera_Room1_test/<target>_height_outputs/multicam_<target>_height_report.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from PIL import Image, ImageDraw


VIEW_AGENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VIEW_AGENT_ROOT.parent.parent

DEFAULT_IMAGE_DIR = VIEW_AGENT_ROOT / "test_tmp" / "pics" / "Camera_Room1_test"
DEFAULT_CAMERA_PARAMETER_DIR = (
    REPO_ROOT / "3090server" / "VLM_RL" / "get_item_info_agent" / "data" / "camera_parameter"
)
DEFAULT_CAMERA_IDS = ("Camera_Room1_2", "Camera_Room1_5", "Camera_Room1_7", "Camera_Room1_10")
DEFAULT_TARGET_LABELS = ("apple",)
YOLO_WEIGHT_CANDIDATES = (
    VIEW_AGENT_ROOT / "test_tmp" / "checkpoints" / "pure720.pt",
    REPO_ROOT / "3090server" / "VLM_RL" / "models" / "yolo" / "pure720.pt",
)


@dataclass(frozen=True)
class Detection:
    camera_id: str
    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center_uv(self) -> np.ndarray:
        return np.array([(self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5], dtype=float)

    @property
    def top_center_uv(self) -> np.ndarray:
        return np.array([(self.x1 + self.x2) * 0.5, self.y1], dtype=float)

    @property
    def bottom_center_uv(self) -> np.ndarray:
        return np.array([(self.x1 + self.x2) * 0.5, self.y2], dtype=float)

    @property
    def bbox_width_px(self) -> float:
        return float(self.x2 - self.x1)

    @property
    def bbox_height_px(self) -> float:
        return float(self.y2 - self.y1)

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "label": self.label,
            "confidence": self.conf,
            "bbox_xyxy": [self.x1, self.y1, self.x2, self.y2],
            "center_uv": self.center_uv.tolist(),
            "top_center_uv": self.top_center_uv.tolist(),
            "bottom_center_uv": self.bottom_center_uv.tolist(),
            "bbox_width_px": self.bbox_width_px,
            "bbox_height_px": self.bbox_height_px,
        }


@dataclass(frozen=True)
class CandidateMatch:
    detections: dict[str, Detection]
    center_world_cv: np.ndarray
    reprojection_error_px: float
    mean_conf: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-camera YOLO + 3D localization + target-object height estimation test."
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--camera-parameter-dir", type=Path, default=DEFAULT_CAMERA_PARAMETER_DIR)
    parser.add_argument("--camera-ids", nargs="+", default=list(DEFAULT_CAMERA_IDS))
    parser.add_argument("--target-labels", nargs="+", default=list(DEFAULT_TARGET_LABELS))
    parser.add_argument("--weights", type=Path, default=default_weights_path())
    parser.add_argument("--conf-thresh", type=float, default=0.20)
    parser.add_argument("--device", default="")
    parser.add_argument("--min-views", type=int, default=None)
    parser.add_argument("--min-valid-height-m", type=float, default=0.03)
    parser.add_argument("--max-valid-height-m", type=float, default=1.50)
    parser.add_argument("--height-offset-mm", type=float, default=-5.0)
    parser.add_argument("--vertical-line-height-stat", choices=("min", "median"), default="min")
    return parser.parse_args()


def default_weights_path() -> Path:
    for path in YOLO_WEIGHT_CANDIDATES:
        if path.exists():
            return path
    return YOLO_WEIGHT_CANDIDATES[0]


def normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())


def output_tag_from_labels(target_labels: Iterable[str]) -> str:
    first_label = next(iter(target_labels), "target")
    return normalize_label(first_label).replace(" ", "_")


def default_output_dir_for_labels(target_labels: Iterable[str]) -> Path:
    return DEFAULT_IMAGE_DIR / f"{output_tag_from_labels(target_labels)}_height_outputs"


def image_path_for_camera(image_dir: Path, camera_id: str) -> Path:
    return image_dir / f"{camera_id}_rgb.png"


def load_intrinsic_matrix(camera_id: str, intrinsics_dir: Path) -> np.ndarray:
    path = intrinsics_dir / f"{camera_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Intrinsic file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    matrix = payload.get("camera_matrix") or payload.get("intrinsic_matrix") or payload.get("K")
    if matrix is None:
        raise ValueError(f"Intrinsic matrix key not found in {path}")
    if isinstance(matrix, dict) and "data" in matrix:
        data = matrix["data"]
    else:
        data = matrix
    arr = np.array(data, dtype=float).reshape(3, 3)
    return arr


def load_extrinsic_matrix(camera_id: str, extrinsics_dir: Path) -> np.ndarray:
    pkl_path = extrinsics_dir / f"{camera_id}.pkl"
    pkl_path_lower = extrinsics_dir / f"{camera_id.lower()}.pkl"
    if pkl_path.exists():
        with pkl_path.open("rb") as handle:
            payload = pickle.load(handle)
    elif pkl_path_lower.exists():
        with pkl_path_lower.open("rb") as handle:
            payload = pickle.load(handle)
    else:
        json_matches = sorted((extrinsics_dir / "json").glob(f"{camera_id}_*.json"))
        if not json_matches:
            json_matches = sorted((extrinsics_dir / "json").glob(f"{camera_id.lower()}_*.json"))
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


def build_projection_matrix(camera_id: str, camera_parameter_dir: Path) -> np.ndarray:
    intrinsics = load_intrinsic_matrix(camera_id, camera_parameter_dir / "Intrinsics")
    extrinsic = load_extrinsic_matrix(camera_id, camera_parameter_dir / "Extrinsic")
    return intrinsics @ extrinsic


def camera_center_from_projection(proj: np.ndarray) -> np.ndarray:
    return -np.linalg.inv(proj[:, :3]) @ proj[:, 3]


def ray_direction_from_projection(proj: np.ndarray, uv: np.ndarray) -> np.ndarray:
    ray = np.linalg.inv(proj[:, :3]) @ np.array([uv[0], uv[1], 1.0], dtype=float)
    return ray / np.linalg.norm(ray)


def triangulate_point_multi_view(
    projections: dict[str, np.ndarray],
    pixels_by_camera: dict[str, np.ndarray],
    eps: float = 1e-6,
) -> np.ndarray:
    eye = np.eye(3)
    a = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)

    for camera_id, uv in pixels_by_camera.items():
        proj = projections[camera_id]
        center = camera_center_from_projection(proj)
        direction = ray_direction_from_projection(proj, uv)
        m = eye - np.outer(direction, direction)
        a += m
        b += m @ center

    if np.linalg.matrix_rank(a) == 3 and np.linalg.cond(a) <= 1 / eps:
        return np.linalg.solve(a, b)

    rows = []
    for camera_id, uv in pixels_by_camera.items():
        proj = projections[camera_id]
        rows.append(float(uv[0]) * proj[2, :] - proj[0, :])
        rows.append(float(uv[1]) * proj[2, :] - proj[1, :])
    _, _, vt = np.linalg.svd(np.vstack(rows))
    xh = vt[-1]
    if np.isclose(xh[3], 0.0):
        raise ValueError("Triangulation failed with zero homogeneous coordinate.")
    return xh[:3] / xh[3]


def project_point(projection: np.ndarray, point_world_cv: np.ndarray) -> np.ndarray:
    point_h = np.append(point_world_cv, 1.0)
    uvw = projection @ point_h
    return uvw[:2] / uvw[2]


def cv_world_to_unity(point_world_cv: np.ndarray) -> np.ndarray:
    point_world_unity = np.array(point_world_cv, dtype=float).copy()
    point_world_unity[2] *= -1.0
    return point_world_unity


def load_camera_models(camera_ids: Iterable[str], camera_parameter_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    models: dict[str, dict[str, np.ndarray]] = {}
    for camera_id in camera_ids:
        k = load_intrinsic_matrix(camera_id, camera_parameter_dir / "Intrinsics")
        extrinsic = load_extrinsic_matrix(camera_id, camera_parameter_dir / "Extrinsic")
        projection = k @ extrinsic
        models[camera_id] = {
            "K": k,
            "extrinsic": extrinsic,
            "P": projection,
            "camera_center": camera_center_from_projection(projection),
        }
    return models


def run_yolo(
    camera_ids: Iterable[str],
    image_dir: Path,
    weights: Path,
    conf_thresh: float,
    device: str,
) -> dict[str, list[Detection]]:
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'ultralytics'. Install it in the runtime environment before running this script."
        ) from exc

    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")

    model = YOLO(str(weights))
    detections: dict[str, list[Detection]] = {}

    for camera_id in camera_ids:
        image_path = image_path_for_camera(image_dir, camera_id)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        predict_kwargs = {"source": str(image_path), "conf": conf_thresh, "verbose": False}
        if device:
            predict_kwargs["device"] = device
        result = model.predict(**predict_kwargs)[0]
        names = {int(k): str(v) for k, v in result.names.items()}
        detections[camera_id] = []

        if result.boxes is None:
            continue

        for box in result.boxes:
            cls_id = int(box.cls.item())
            label = normalize_label(names.get(cls_id, str(cls_id)))
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections[camera_id].append(
                Detection(
                    camera_id=camera_id,
                    label=label,
                    conf=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    return detections


def match_target_detections(
    detections_by_camera: dict[str, list[Detection]],
    camera_models: dict[str, dict[str, np.ndarray]],
    target_labels: Iterable[str],
    min_views: int,
) -> CandidateMatch:
    target_aliases = {normalize_label(label) for label in target_labels}
    target_detections_by_camera = {
        camera_id: [det for det in dets if normalize_label(det.label) in target_aliases]
        for camera_id, dets in detections_by_camera.items()
    }
    available_cameras = [camera_id for camera_id, dets in target_detections_by_camera.items() if dets]
    if len(available_cameras) < max(2, min_views):
        available_labels = {
            camera_id: sorted({det.label for det in dets}) for camera_id, dets in detections_by_camera.items()
        }
        raise RuntimeError(
            "The target object was not detected in enough cameras. "
            f"Requested min_views={min_views}, detected_views={len(available_cameras)}. "
            f"Target labels={sorted(target_aliases)}. Seen labels={available_labels}"
        )

    projections = {camera_id: model["P"] for camera_id, model in camera_models.items()}
    candidates: list[CandidateMatch] = []

    max_views = len(available_cameras)
    for num_views in range(max_views, max(2, min_views) - 1, -1):
        for camera_subset in combinations(available_cameras, num_views):
            det_lists = [target_detections_by_camera[camera_id] for camera_id in camera_subset]
            for det_tuple in product(*det_lists):
                pixels = {camera_id: det.center_uv for camera_id, det in zip(camera_subset, det_tuple)}
                center_world_cv = triangulate_point_multi_view(projections, pixels)
                reprojection_errors = []
                for camera_id, det in zip(camera_subset, det_tuple):
                    uv_proj = project_point(projections[camera_id], center_world_cv)
                    reprojection_errors.append(float(np.linalg.norm(uv_proj - det.center_uv)))
                candidates.append(
                    CandidateMatch(
                        detections={camera_id: det for camera_id, det in zip(camera_subset, det_tuple)},
                        center_world_cv=center_world_cv,
                        reprojection_error_px=float(np.mean(reprojection_errors)),
                        mean_conf=float(np.mean([det.conf for det in det_tuple])),
                    )
                )

    if not candidates:
        raise RuntimeError("Unable to form any multi-view target candidate.")

    candidates.sort(
        key=lambda item: (
            -len(item.detections),
            item.reprojection_error_px,
            -item.mean_conf,
        )
    )
    return candidates[0]


def triangulate_optional(
    projections: dict[str, np.ndarray],
    pixels_by_camera: dict[str, np.ndarray],
) -> np.ndarray | None:
    if len(pixels_by_camera) < 2:
        return None
    try:
        return triangulate_point_multi_view(projections, pixels_by_camera)
    except Exception:
        return None


def fit_vertical_line_xz(
    match: CandidateMatch,
    camera_models: dict[str, dict[str, np.ndarray]],
    eps: float = 1e-6,
) -> np.ndarray:
    eye = np.eye(2)
    a = np.zeros((2, 2), dtype=float)
    b = np.zeros(2, dtype=float)

    for camera_id, det in match.detections.items():
        projection = camera_models[camera_id]["P"]
        camera_center_xz = camera_center_from_projection(projection)[[0, 2]]
        ray_xz = ray_direction_from_projection(projection, det.center_uv)[[0, 2]]
        norm_xz = float(np.linalg.norm(ray_xz))
        if norm_xz <= eps:
            continue
        ray_xz /= norm_xz
        m = eye - np.outer(ray_xz, ray_xz)
        a += m
        b += m @ camera_center_xz

    if np.linalg.matrix_rank(a) < 2 or np.linalg.cond(a) > 1 / eps:
        return match.center_world_cv[[0, 2]].copy()
    return np.linalg.solve(a, b)


def project_pixel_to_vertical_line(
    projection: np.ndarray,
    uv: np.ndarray,
    vertical_line_xz: np.ndarray,
    eps: float = 1e-8,
) -> dict[str, object] | None:
    camera_center = camera_center_from_projection(projection)
    ray_direction = ray_direction_from_projection(projection, uv)
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


def estimate_height_on_vertical_line(
    match: CandidateMatch,
    camera_models: dict[str, dict[str, np.ndarray]],
) -> dict[str, object]:
    vertical_line_xz = fit_vertical_line_xz(match, camera_models)
    top_y_by_camera: dict[str, float] = {}
    bottom_y_by_camera: dict[str, float] = {}
    height_by_camera: dict[str, float] = {}
    center_residual_xz_by_camera: dict[str, float] = {}
    top_residual_xz_by_camera: dict[str, float] = {}
    bottom_residual_xz_by_camera: dict[str, float] = {}

    for camera_id, det in match.detections.items():
        projection = camera_models[camera_id]["P"]
        center_fit = project_pixel_to_vertical_line(projection, det.center_uv, vertical_line_xz)
        top_fit = project_pixel_to_vertical_line(projection, det.top_center_uv, vertical_line_xz)
        bottom_fit = project_pixel_to_vertical_line(projection, det.bottom_center_uv, vertical_line_xz)
        if center_fit is None or top_fit is None or bottom_fit is None:
            continue

        center_residual_xz_by_camera[camera_id] = float(center_fit["residual_xz_m"])
        top_residual_xz_by_camera[camera_id] = float(top_fit["residual_xz_m"])
        bottom_residual_xz_by_camera[camera_id] = float(bottom_fit["residual_xz_m"])
        top_y_by_camera[camera_id] = float(top_fit["y_world_cv"])
        bottom_y_by_camera[camera_id] = float(bottom_fit["y_world_cv"])
        height_by_camera[camera_id] = float(top_fit["y_world_cv"] - bottom_fit["y_world_cv"])

    if not height_by_camera:
        raise RuntimeError("Vertical-line height estimation failed for all cameras.")

    top_y_values = np.asarray(list(top_y_by_camera.values()), dtype=float)
    bottom_y_values = np.asarray(list(bottom_y_by_camera.values()), dtype=float)
    height_values = np.asarray(list(height_by_camera.values()), dtype=float)

    top_y_median = float(np.median(top_y_values))
    bottom_y_median = float(np.median(bottom_y_values))
    height_median = float(np.median(height_values))
    height_min = float(np.min(height_values))

    return {
        "vertical_line_xz_world_cv": np.array(vertical_line_xz, dtype=float),
        "top_y_by_camera": top_y_by_camera,
        "bottom_y_by_camera": bottom_y_by_camera,
        "height_by_camera": height_by_camera,
        "center_residual_xz_by_camera": center_residual_xz_by_camera,
        "top_residual_xz_by_camera": top_residual_xz_by_camera,
        "bottom_residual_xz_by_camera": bottom_residual_xz_by_camera,
        "top_world_cv_median": np.array([vertical_line_xz[0], top_y_median, vertical_line_xz[1]], dtype=float),
        "bottom_world_cv_median": np.array([vertical_line_xz[0], bottom_y_median, vertical_line_xz[1]], dtype=float),
        "height_median": height_median,
        "height_min": height_min,
    }


def height_estimates_from_bbox(
    match: CandidateMatch,
    camera_models: dict[str, dict[str, np.ndarray]],
) -> dict[str, float]:
    estimates: dict[str, float] = {}
    for camera_id, det in match.detections.items():
        k = camera_models[camera_id]["K"]
        extrinsic = camera_models[camera_id]["extrinsic"]
        point_cam = extrinsic[:, :3] @ match.center_world_cv + extrinsic[:, 3]
        z_cam = float(point_cam[2])
        if z_cam <= 0:
            continue
        estimates[camera_id] = float(det.bbox_height_px * z_cam / float(k[1, 1]))
    return estimates


def choose_final_height_m(
    bbox_height_estimates: dict[str, float],
    vertical_line_height_m: float | None,
    min_valid_height_m: float,
    max_valid_height_m: float,
) -> float:
    candidates = list(bbox_height_estimates.values())
    if (
        vertical_line_height_m is not None
        and min_valid_height_m <= vertical_line_height_m <= max_valid_height_m
    ):
        candidates.append(vertical_line_height_m)
    if not candidates:
        raise RuntimeError("No valid height estimate could be computed.")
    return float(np.median(np.asarray(candidates, dtype=float)))


def annotate_and_save(
    image_path: Path,
    detection: Detection,
    output_path: Path,
    estimated_height_m: float,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    color = (34, 177, 76)
    label = f"{detection.label} {detection.conf:.2f} | {estimated_height_m * 1000.0:.1f} mm"

    draw.rectangle([detection.x1, detection.y1, detection.x2, detection.y2], outline=color, width=4)
    cx, cy = detection.center_uv.tolist()
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 80, 80))
    draw.text((detection.x1, max(0.0, detection.y1 - 16.0)), label, fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    args = parse_args()
    image_dir = args.image_dir.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    camera_parameter_dir = args.camera_parameter_dir.expanduser().resolve()
    camera_ids = [str(camera_id) for camera_id in args.camera_ids]
    target_labels = [str(label) for label in args.target_labels]
    min_views = args.min_views if args.min_views is not None else len(camera_ids)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir_for_labels(target_labels).resolve()
    )

    if min_views < 2:
        raise ValueError("--min-views must be at least 2.")
    if min_views > len(camera_ids):
        raise ValueError("--min-views cannot exceed the number of camera ids.")

    for camera_id in camera_ids:
        image_path = image_path_for_camera(image_dir, camera_id)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

    camera_models = load_camera_models(camera_ids, camera_parameter_dir)
    detections_by_camera = run_yolo(
        camera_ids=camera_ids,
        image_dir=image_dir,
        weights=weights,
        conf_thresh=args.conf_thresh,
        device=args.device,
    )
    match = match_target_detections(
        detections_by_camera=detections_by_camera,
        camera_models=camera_models,
        target_labels=target_labels,
        min_views=min_views,
    )

    selected_camera_ids = list(match.detections.keys())
    vertical_line_fit = estimate_height_on_vertical_line(match, camera_models)
    top_world_cv = vertical_line_fit["top_world_cv_median"]
    bottom_world_cv = vertical_line_fit["bottom_world_cv_median"]
    vertical_line_height_median = float(vertical_line_fit["height_median"])
    vertical_line_height_min = float(vertical_line_fit["height_min"])
    vertical_line_height_m = (
        vertical_line_height_min
        if args.vertical_line_height_stat == "min"
        else vertical_line_height_median
    )

    bbox_height_estimates = height_estimates_from_bbox(match, camera_models)
    raw_estimated_height_m = choose_final_height_m(
        bbox_height_estimates=bbox_height_estimates,
        vertical_line_height_m=vertical_line_height_m,
        min_valid_height_m=args.min_valid_height_m,
        max_valid_height_m=args.max_valid_height_m,
    )
    estimated_height_m = max(0.0, raw_estimated_height_m + args.height_offset_mm / 1000.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    for camera_id, det in match.detections.items():
        image_path = image_path_for_camera(image_dir, camera_id)
        annotated_path = output_dir / f"{camera_id}_annotated.png"
        annotate_and_save(
            image_path=image_path,
            detection=det,
            output_path=annotated_path,
            estimated_height_m=estimated_height_m,
        )

    report = {
        "camera_ids": camera_ids,
        "min_views": min_views,
        "selected_cameras": selected_camera_ids,
        "target_label_aliases": [normalize_label(label) for label in target_labels],
        "selected_label": next(iter(match.detections.values())).label,
        "weights": str(weights),
        "image_dir": str(image_dir),
        "camera_parameter_dir": str(camera_parameter_dir),
        "reprojection_error_px": match.reprojection_error_px,
        "mean_confidence": match.mean_conf,
        "center_world_cv": match.center_world_cv.tolist(),
        "center_world_unity": cv_world_to_unity(match.center_world_cv).tolist(),
        "vertical_line_xz_world_cv": [
            float(vertical_line_fit["vertical_line_xz_world_cv"][0]),
            float(vertical_line_fit["vertical_line_xz_world_cv"][1]),
        ],
        "vertical_line_xz_world_unity": [
            float(vertical_line_fit["vertical_line_xz_world_cv"][0]),
            float(-vertical_line_fit["vertical_line_xz_world_cv"][1]),
        ],
        "top_world_cv": top_world_cv.tolist(),
        "bottom_world_cv": bottom_world_cv.tolist(),
        "top_world_unity": cv_world_to_unity(top_world_cv).tolist(),
        "bottom_world_unity": cv_world_to_unity(bottom_world_cv).tolist(),
        "height_estimates_m": {
            "from_bbox_per_camera": bbox_height_estimates,
            "from_vertical_line_per_camera": vertical_line_fit["height_by_camera"],
            "vertical_line_top_y_per_camera": vertical_line_fit["top_y_by_camera"],
            "vertical_line_bottom_y_per_camera": vertical_line_fit["bottom_y_by_camera"],
            "vertical_line_center_residual_xz_m_per_camera": vertical_line_fit["center_residual_xz_by_camera"],
            "vertical_line_top_residual_xz_m_per_camera": vertical_line_fit["top_residual_xz_by_camera"],
            "vertical_line_bottom_residual_xz_m_per_camera": vertical_line_fit["bottom_residual_xz_by_camera"],
            "from_vertical_line_median": vertical_line_height_median,
            "from_vertical_line_min": vertical_line_height_min,
            "vertical_line_height_stat": args.vertical_line_height_stat,
            "from_vertical_line_selected": vertical_line_height_m,
            "final_median": raw_estimated_height_m,
            "height_offset_mm": args.height_offset_mm,
            "final_with_offset": estimated_height_m,
        },
        "estimated_height_m": estimated_height_m,
        "estimated_height_mm": estimated_height_m * 1000.0,
        "target_detections": {
            camera_id: det.to_dict() for camera_id, det in match.detections.items()
        },
        "all_detections": {
            camera_id: [det.to_dict() for det in dets] for camera_id, dets in detections_by_camera.items()
        },
        "annotated_images": {
            camera_id: str(output_dir / f"{camera_id}_annotated.png") for camera_id in match.detections
        },
    }

    report_path = output_dir / f"multicam_{output_tag_from_labels(target_labels)}_height_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "report_path": str(report_path),
            "estimated_height_m": estimated_height_m,
            "estimated_height_mm": estimated_height_m * 1000.0,
            "selected_label": report["selected_label"],
            "selected_cameras": selected_camera_ids,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
