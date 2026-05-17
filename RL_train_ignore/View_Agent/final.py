#!/usr/bin/env python3
"""
Self-contained end-to-end pipeline:
  multi-camera RGB -> YOLO -> SAM -> multi-view geometry size ->
  multi-camera size fusion -> optional grasp + collision filtering

This file intentionally does not import helper logic from other local Python
files in the repository. External runtime dependencies and model/data files are
still required at execution time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


VIEW_AGENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VIEW_AGENT_ROOT.parent.parent
TOOL_ROOT = REPO_ROOT / "3090server" / "VLM_RL"
AGENT_ROOT = TOOL_ROOT / "get_item_info_agent"
SAM3D_CAMERA_ROOT = AGENT_ROOT / "vendor" / "sam3d_runtime" / "Camera_3D_Localization"
GRASPGEN_VENDOR = AGENT_ROOT / "vendor" / "graspgen_runtime"

for path in (
    SAM3D_CAMERA_ROOT,
    SAM3D_CAMERA_ROOT / "src",
    GRASPGEN_VENDOR,
    GRASPGEN_VENDOR / "pointnet2_ops",
):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

os.environ.setdefault("GRASPGEN_NO_VIS", "1")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(AGENT_ROOT / ".cache" / "torch_extensions"))


PIPELINE_NAME = "multicam_geometry_sam_size_then_grasp"
WORLD_UP_CV = np.array([0.0, 1.0, 0.0], dtype=np.float32)

DEFAULT_IMAGE_DIR = VIEW_AGENT_ROOT / "test_tmp" / "pics" / "Camera_Room1_test"
DEFAULT_OUTPUT_DIR_NAME = "final_outputs"
DEFAULT_CAMERA_PARAMETER_DIR = AGENT_ROOT / "data" / "camera_parameter"
DEFAULT_CAMERA_IDS = (
    "Camera_Room1_12",
    "Camera_Room1_13",
    "Camera_Room1_14",
    "Camera_Room1_15",
)
DEFAULT_REFERENCE_CAMERA_ID = "Camera_Room1_12"
DEFAULT_LABELS = ("doll", "apple", "wine")
DEFAULT_TARGET_LABEL = "doll"

YOLO_WEIGHT_CANDIDATES = (
    TOOL_ROOT / "models" / "yolo" / "pure720.pt",
    VIEW_AGENT_ROOT / "test_tmp" / "checkpoints" / "pure720.pt",
)
SAM_CHECKPOINT_CANDIDATES = (
    TOOL_ROOT / "models" / "segmentation" / "sam_vit_b_01ec64.pth",
    TOOL_ROOT / "models" / "sam_vit_b_01ec64.pth",
)
GRIPPER_CONFIG = TOOL_ROOT / "models" / "graspgen_checkpoints" / "graspgen_robotiq_2f_140.yml"

DEFAULT_KNOWN_CENTERS_ALIGNED_WORLD = {
    "doll": np.array([2.36199999, 0.43299982, 8.4659996], dtype=np.float32),
    "apple": np.array([2.33899999, 0.407000005, 8.1079998], dtype=np.float32),
    "wine": np.array([2.31200004, 0.495999992, 8.78999996], dtype=np.float32),
}
LABEL_COLORS = {
    "doll": np.array([60, 220, 90], dtype=np.uint8),
    "apple": np.array([255, 140, 40], dtype=np.uint8),
    "wine": np.array([70, 160, 255], dtype=np.uint8),
}
HEIGHT_WEIGHT_RATIO_ANCHORS = np.array([0.70, 1.03, 2.28], dtype=np.float32)
HEIGHT_WEIGHT_VALUE_ANCHORS = np.array([0.65, 0.80, 0.90], dtype=np.float32)


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
            "bbox_width_px": float(self.x2 - self.x1),
            "bbox_height_px": self.bbox_height_px,
        }


@dataclass(frozen=True)
class CandidateMatch:
    detections: dict[str, Detection]
    center_world_cv: np.ndarray
    reprojection_error_px: float
    mean_conf: float


@dataclass(frozen=True)
class SizeStageResult:
    summary: dict[str, object]
    label_results: dict[str, dict[str, object]]
    output_paths: dict[str, Path]


@dataclass(frozen=True)
class GraspStageResult:
    report: dict[str, object]
    output_paths: dict[str, Path]


def normalize_label(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())


def default_output_dir(image_dir: Path) -> Path:
    return image_dir / DEFAULT_OUTPUT_DIR_NAME


def default_weights_path() -> Path:
    for path in YOLO_WEIGHT_CANDIDATES:
        if path.exists():
            return path
    return YOLO_WEIGHT_CANDIDATES[0]


def default_sam_checkpoint() -> Path:
    for path in SAM_CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    return SAM_CHECKPOINT_CANDIDATES[0]


def default_inference_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def image_path_for_camera(image_dir: Path, camera_id: str) -> Path:
    return image_dir / f"{camera_id}_rgb.png"


def output_tag(label: str) -> str:
    return normalize_label(label).replace(" ", "_")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-camera geometry + SAM size pipeline with optional grasp output."
    )
    parser.add_argument("--target-label", default=DEFAULT_TARGET_LABEL)
    parser.add_argument("--labels", nargs="+", default=list(DEFAULT_LABELS))
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--camera-parameter-dir", type=Path, default=DEFAULT_CAMERA_PARAMETER_DIR)
    parser.add_argument("--camera-ids", nargs="+", default=list(DEFAULT_CAMERA_IDS))
    parser.add_argument("--reference-camera-id", default=DEFAULT_REFERENCE_CAMERA_ID)
    parser.add_argument("--weights", type=Path, default=default_weights_path())
    parser.add_argument("--conf-thresh", type=float, default=0.20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-views", type=int, default=None)
    parser.add_argument("--sam-checkpoint", type=Path, default=default_sam_checkpoint())
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--robust-lower", type=float, default=5.0)
    parser.add_argument("--robust-upper", type=float, default=95.0)
    parser.add_argument("--axis-step-m", type=float, default=0.05)
    parser.add_argument("--min-valid-height-m", type=float, default=0.03)
    parser.add_argument("--max-valid-height-m", type=float, default=1.50)
    parser.add_argument("--vertical-line-height-stat", choices=("min", "median"), default="min")
    parser.add_argument("--footprint-max-aspect-ratio", type=float, default=1.8)
    parser.add_argument(
        "--height-scale",
        type=float,
        default=1.0,
        help="Additional manual multiplier applied after the shape-ratio height weight.",
    )
    parser.add_argument("--disable-yolo-height-cap", action="store_true")
    parser.add_argument("--skip-grasp", action="store_true")
    parser.add_argument("--point-budget", type=int, default=1800)
    parser.add_argument("--num-grasps", type=int, default=200)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--collision-thresh", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_intrinsic_matrix(camera_id: str, intrinsics_dir: Path) -> np.ndarray:
    path = intrinsics_dir / f"{camera_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Intrinsic file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    matrix = payload.get("camera_matrix") or payload.get("intrinsic_matrix") or payload.get("K")
    if matrix is None:
        raise ValueError(f"Intrinsic matrix key not found in {path}")
    data = matrix["data"] if isinstance(matrix, dict) and "data" in matrix else matrix
    return np.array(data, dtype=float).reshape(3, 3)


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
    point_h = np.append(np.asarray(point_world_cv, dtype=np.float64).reshape(3), 1.0)
    uvw = np.asarray(projection, dtype=np.float64) @ point_h
    if abs(float(uvw[2])) < 1e-12:
        raise RuntimeError("Point projects to infinity.")
    return np.array([uvw[0] / uvw[2], uvw[1] / uvw[2]], dtype=np.float32)


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
        raise ImportError("Missing dependency 'ultralytics'.") from exc

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


def detections_to_jsonable(detections_by_camera: dict[str, list[Detection]]) -> dict[str, list[dict[str, object]]]:
    return {camera_id: [det.to_dict() for det in dets] for camera_id, dets in detections_by_camera.items()}


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


def yolo_cache_path(base_dir: Path, camera_ids: list[str]) -> Path:
    return base_dir / f"obstacle_yolo_cache_{'_'.join(camera_ids)}.json"


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
    cache_path.write_text(json.dumps(cache_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return detections_by_camera, str(cache_path)


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

    candidates.sort(key=lambda item: (-len(item.detections), item.reprojection_error_px, -item.mean_conf))
    return candidates[0]


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
    for camera_id, det in match.detections.items():
        projection = camera_models[camera_id]["P"]
        center_fit = project_pixel_to_vertical_line(projection, det.center_uv, vertical_line_xz)
        top_fit = project_pixel_to_vertical_line(projection, det.top_center_uv, vertical_line_xz)
        bottom_fit = project_pixel_to_vertical_line(projection, det.bottom_center_uv, vertical_line_xz)
        if center_fit is None or top_fit is None or bottom_fit is None:
            continue
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
    if vertical_line_height_m is not None and min_valid_height_m <= vertical_line_height_m <= max_valid_height_m:
        candidates.append(vertical_line_height_m)
    if not candidates:
        raise RuntimeError("No valid height estimate could be computed.")
    return float(np.median(np.asarray(candidates, dtype=float)))


def shape_ratio_height_weight(height_m: float, size_x_m: float, size_z_m: float) -> tuple[float, float]:
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


def build_sam_predictor(model_type: str, checkpoint: Path, device: str):
    from segment_anything import SamPredictor, sam_model_registry  # type: ignore

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


def segment_mask_for_detection(predictor, image_bgr: np.ndarray, detection: Detection) -> np.ndarray:
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
    return masks[int(np.argmax(scores))].astype(bool)


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


def known_center_world_cv(label: str, fallback_center_world_cv: np.ndarray) -> np.ndarray:
    normalized = normalize_label(label)
    if normalized in DEFAULT_KNOWN_CENTERS_ALIGNED_WORLD:
        return aligned_world_to_cv(DEFAULT_KNOWN_CENTERS_ALIGNED_WORLD[normalized])
    return np.asarray(fallback_center_world_cv, dtype=np.float32).reshape(3)


def merge_xyxy_boxes(boxes: Iterable[Iterable[int | float]]) -> list[int]:
    box_list = [list(box) for box in boxes]
    if not box_list:
        raise RuntimeError("No crop boxes available for union crop.")
    left = int(min(float(box[0]) for box in box_list))
    top = int(min(float(box[1]) for box in box_list))
    right = int(max(float(box[2]) for box in box_list))
    bottom = int(max(float(box[3]) for box in box_list))
    if right <= left or bottom <= top:
        raise RuntimeError("Merged union crop box is degenerate.")
    return [left, top, right, bottom]


def bbox_from_mask(mask: np.ndarray, pad: int = 2) -> tuple[int, int, int, int]:
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if len(xs) == 0:
        raise RuntimeError("Mask bbox is empty.")
    h, w = mask.shape[:2]
    left = max(0, int(xs.min()) - int(pad))
    top = max(0, int(ys.min()) - int(pad))
    right = min(w, int(xs.max()) + int(pad) + 1)
    bottom = min(h, int(ys.max()) + int(pad) + 1)
    if right <= left or bottom <= top:
        raise RuntimeError("Mask bbox is degenerate.")
    return left, top, right, bottom


def camera_azimuth_about_object(center_world_cv: np.ndarray, camera_center_world_cv: np.ndarray) -> float:
    dx = float(camera_center_world_cv[0] - center_world_cv[0])
    dz = float(camera_center_world_cv[2] - center_world_cv[2])
    return float(math.atan2(dz, dx))


def fit_rotated_rectangle_footprint(
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
    assert best_result is not None
    return best_result


def point_colors(label: str, count: int) -> np.ndarray:
    color = LABEL_COLORS.get(normalize_label(label), np.array([180, 180, 180], dtype=np.uint8))
    return np.repeat(color.reshape(1, 3), count, axis=0)


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


def sample_box_volume_local(size_xyz: np.ndarray, budget: int) -> np.ndarray:
    nx, ny, nz = grid_resolution_for_budget(size_xyz, budget)
    xs = np.linspace(-size_xyz[0] * 0.5, size_xyz[0] * 0.5, nx, dtype=np.float32)
    ys = np.linspace(-size_xyz[1] * 0.5, size_xyz[1] * 0.5, ny, dtype=np.float32)
    zs = np.linspace(-size_xyz[2] * 0.5, size_xyz[2] * 0.5, nz, dtype=np.float32)
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="xy")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)


def build_box_pointcloud(
    center_world: np.ndarray,
    size_xyz: np.ndarray,
    point_budget: int,
    yaw_rad: float,
) -> np.ndarray:
    local_pts = sample_box_volume_local(np.asarray(size_xyz, dtype=np.float32), point_budget)
    rotation = yaw_rotation_matrix(yaw_rad)
    center = np.asarray(center_world, dtype=np.float32).reshape(1, 3)
    return local_pts @ rotation.T + center


def run_grasp_inference(
    object_pc: np.ndarray,
    object_colors: np.ndarray | None,
    num_grasps: int,
    topk: int,
):
    import torch
    import trimesh.transformations as tra  # type: ignore

    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg  # type: ignore
    from grasp_gen.utils.meshcat_utils import get_color_from_score  # type: ignore
    from grasp_gen.utils.point_cloud_utils import (  # type: ignore
        point_cloud_outlier_removal,
        point_cloud_outlier_removal_with_color,
    )

    filtered_colors = None
    if object_colors is not None:
        pc_t, removed_t, color_t, _ = point_cloud_outlier_removal_with_color(
            torch.from_numpy(object_pc),
            torch.from_numpy(object_colors),
        )
        filtered_colors = color_t.numpy()
    else:
        pc_t, removed_t = point_cloud_outlier_removal(torch.from_numpy(object_pc))
    pc_filtered = pc_t.numpy()
    if len(pc_filtered) == 0:
        raise RuntimeError("Object PC empty after outlier removal.")

    cfg = load_grasp_cfg(str(GRIPPER_CONFIG))
    sampler = GraspGenSampler(cfg)
    grasps_t, conf_t = GraspGenSampler.run_inference(
        pc_filtered,
        sampler,
        grasp_threshold=-1.0,
        num_grasps=num_grasps,
        topk_num_grasps=topk,
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


def run_grasp_inference_no_filter(
    object_pc: np.ndarray,
    object_colors: np.ndarray | None,
    num_grasps: int,
    topk: int,
):
    import trimesh.transformations as tra  # type: ignore

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


def filter_collisions(
    scene_pc: np.ndarray,
    grasps_c: np.ndarray,
    t_center: np.ndarray,
    cfg,
    collision_threshold: float = 0.02,
    max_scene_pts: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    import trimesh.transformations as tra  # type: ignore

    from grasp_gen.robot import get_gripper_info  # type: ignore
    from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps  # type: ignore

    scene_c = tra.transform_points(scene_pc, t_center)
    if len(scene_c) > max_scene_pts:
        idx = np.random.choice(len(scene_c), max_scene_pts, replace=False)
        scene_ds = scene_c[idx]
    else:
        scene_ds = scene_c
    gripper_info = get_gripper_info(cfg.data.gripper_name)
    mask = filter_colliding_grasps(
        scene_pc=scene_ds,
        grasp_poses=grasps_c,
        gripper_collision_mesh=gripper_info.collision_mesh,
        collision_threshold=collision_threshold,
    )
    return mask, scene_c


def build_size_markdown(labels: list[str], results: dict[str, dict[str, object]]) -> str:
    lines = [
        "# Multi-Camera Geometry + SAM Size Summary",
        "",
        "| Label | Width (mm) | Height (mm) | Depth (mm) | Yaw (deg) | Cameras | RMSE (mm) | Height Source |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for label in labels:
        info = results[label]
        width_mm, height_mm, depth_mm = info["size_xyz_mm"]
        lines.append(
            f"| {label} | {width_mm:.3f} | {height_mm:.3f} | {depth_mm:.3f} | "
            f"{info['yaw_deg']:.3f} | {', '.join(info['selected_cameras'])} | "
            f"{float(info['footprint_fit_rmse_m']) * 1000.0:.3f} | {info['height_source']} |"
        )
    return "\n".join(lines) + "\n"


def build_pointcloud_npz_payload(
    results: dict[str, dict[str, object]],
    reference_camera_id: str,
    crop_box_xyxy: list[int],
    box_point_budget: int,
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "labels": np.array(list(results.keys()), dtype=object),
        "reference_camera_id": np.array([reference_camera_id], dtype=object),
        "reference_axis_mode": np.array(["geometry"], dtype=object),
        "crop_box_xyxy": np.asarray(crop_box_xyxy, dtype=np.int32),
    }
    box_points_all = []
    box_colors_all = []

    for label, info in results.items():
        key = output_tag(label)
        center_world = np.asarray(
            info.get("center_world_unity", info["multicam_center_world_m"]),
            dtype=np.float32,
        )
        size_xyz = np.asarray(info["size_xyz_m"], dtype=np.float32)
        yaw_rad = float(info["yaw_rad"])
        box_points = build_box_pointcloud(
            center_world=center_world,
            size_xyz=size_xyz,
            point_budget=box_point_budget,
            yaw_rad=yaw_rad,
        ).astype(np.float32)
        color = point_colors(label, 1)[0]
        box_colors = np.repeat(color.reshape(1, 3), len(box_points), axis=0).astype(np.uint8)

        payload[f"{key}_estimated_box_points_world_m"] = box_points
        payload[f"{key}_estimated_box_points_colors"] = box_colors
        payload[f"{key}_center_world_m"] = center_world
        payload[f"{key}_size_xyz_m"] = size_xyz
        payload[f"{key}_yaw_rad"] = np.array([yaw_rad], dtype=np.float32)

        box_points_all.append(box_points)
        box_colors_all.append(box_colors)

    if box_points_all:
        payload["estimated_box_points_world_m"] = np.concatenate(box_points_all, axis=0).astype(np.float32)
        payload["estimated_box_points_colors"] = np.concatenate(box_colors_all, axis=0).astype(np.uint8)

    return payload


def run_size_stage(args: argparse.Namespace, output_dir: Path) -> SizeStageResult:
    stage_start = time.perf_counter()
    image_dir = args.image_dir.expanduser().resolve()
    camera_parameter_dir = args.camera_parameter_dir.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    target_label = normalize_label(args.target_label)
    labels = [normalize_label(label) for label in args.labels]
    if target_label not in labels:
        labels.insert(0, target_label)
    camera_ids = [str(camera_id) for camera_id in args.camera_ids]
    reference_camera_id = str(args.reference_camera_id)
    min_views = args.min_views if args.min_views is not None else len(camera_ids)

    if min_views < 2 or min_views > len(camera_ids):
        raise ValueError("Invalid --min-views for the selected camera ids.")
    if reference_camera_id not in camera_ids:
        raise ValueError("--reference-camera-id must be included in --camera-ids")
    if not args.sam_checkpoint.exists():
        raise FileNotFoundError(args.sam_checkpoint)

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

    sam_device = str(args.device or default_inference_device())
    sam_model_start = time.perf_counter()
    sam_model, sam_predictor = build_sam_predictor(args.sam_model_type, args.sam_checkpoint, sam_device)
    timings["sam_model_load"] = float(time.perf_counter() - sam_model_start)

    image_cache: dict[str, np.ndarray] = {}
    masks_by_label: dict[str, dict[str, np.ndarray]] = {label: {} for label in labels}
    sam_inference_by_camera: dict[str, float] = {}
    sam_inference_total = 0.0
    try:
        for camera_id in camera_ids:
            image_path = image_path_for_camera(image_dir, camera_id)
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            image_cache[camera_id] = image_bgr

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

    primary_visible_labels = [label for label in labels if reference_camera_id in matches[label].detections]
    if not primary_visible_labels:
        raise RuntimeError(f"No requested labels are visible in reference camera {reference_camera_id}.")
    if target_label not in primary_visible_labels:
        raise RuntimeError(f"Target '{target_label}' is not visible in reference camera {reference_camera_id}.")

    primary_image = image_cache[reference_camera_id]
    results: dict[str, dict[str, object]] = {}
    label_times: dict[str, float] = {}

    for label in labels:
        label_start = time.perf_counter()
        match = matches[label]
        center_world_cv = known_center_world_cv(label, match.center_world_cv)
        center_world_aligned = cv_world_to_aligned_world(center_world_cv)
        vertical_line_fit = estimate_height_on_vertical_line(match, camera_models)
        vertical_line_height_m = (
            float(vertical_line_fit["height_min"])
            if args.vertical_line_height_stat == "min"
            else float(vertical_line_fit["height_median"])
        )
        bbox_height_estimates = height_estimates_from_bbox(match, camera_models)
        base_height_m = choose_final_height_m(
            bbox_height_estimates=bbox_height_estimates,
            vertical_line_height_m=vertical_line_height_m,
            min_valid_height_m=float(args.min_valid_height_m),
            max_valid_height_m=float(args.max_valid_height_m),
        )

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
                reference_height_m=float(base_height_m),
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

        footprint_fit = fit_rotated_rectangle_footprint(
            center_world_cv,
            width_by_camera,
            camera_models,
            max_aspect_ratio=float(args.footprint_max_aspect_ratio),
        )
        size_x_m = float(footprint_fit["size_x_m"])
        size_z_m = float(footprint_fit["size_z_m"])
        shape_ratio_h_over_x_plus_z, shape_height_weight = shape_ratio_height_weight(
            float(base_height_m),
            size_x_m,
            size_z_m,
        )
        applied_height_multiplier = float(shape_height_weight * float(args.height_scale))
        selected_height_m = float(base_height_m * applied_height_multiplier)
        size_xyz_m = np.array([size_x_m, selected_height_m, size_z_m], dtype=np.float32)

        label_details: dict[str, object] = {
            "reference_camera_id": reference_camera_id,
            "height_estimation_mode": "geometry",
            "center_world_m": center_world_aligned.astype(float).tolist(),
            "center_world_unity": center_world_aligned.astype(float).tolist(),
            "center_world_cv": center_world_cv.astype(float).tolist(),
            "bbox_height_estimates_m": {
                camera_id: float(value) for camera_id, value in bbox_height_estimates.items()
            },
            "vertical_line_height_median_m": float(vertical_line_fit["height_median"]),
            "vertical_line_height_min_m": float(vertical_line_fit["height_min"]),
            "vertical_line_height_selected_m": float(vertical_line_height_m),
            "selected_geometry_height_before_scale_m": float(base_height_m),
            "selected_geometry_height_before_scale_mm": float(base_height_m * 1000.0),
            "selected_geometry_height_m": float(selected_height_m),
            "selected_geometry_height_mm": float(selected_height_m * 1000.0),
            "shape_ratio_h_over_x_plus_z": float(shape_ratio_h_over_x_plus_z),
            "shape_height_weight": float(shape_height_weight),
            "applied_height_multiplier": float(applied_height_multiplier),
            "height_scale": float(args.height_scale),
        }
        if reference_camera_id in masks_by_label[label]:
            label_details["crop_box_xyxy"] = [
                int(v) for v in bbox_from_mask(np.asarray(masks_by_label[label][reference_camera_id], dtype=bool))
            ]

        results[label] = {
            "reference_camera_id": reference_camera_id,
            "reference_height_m": float(base_height_m),
            "reference_height_mm": float(base_height_m * 1000.0),
            "reference_height_m_by_camera": {},
            "reference_height_mm_by_camera": {},
            "reference_world_y_height_m_by_camera": {},
            "reference_world_y_height_mm_by_camera": {},
            "yolo_height_cap_enabled": False,
            "yolo_height_m": float(base_height_m),
            "yolo_height_mm": float(base_height_m * 1000.0),
            "selected_height_before_scale_m": float(base_height_m),
            "selected_height_before_scale_mm": float(base_height_m * 1000.0),
            "selected_height_m": float(selected_height_m),
            "selected_height_mm": float(selected_height_m * 1000.0),
            "shape_ratio_h_over_x_plus_z": float(shape_ratio_h_over_x_plus_z),
            "shape_height_weight": float(shape_height_weight),
            "applied_height_multiplier": float(applied_height_multiplier),
            "height_scale": float(args.height_scale),
            "height_source": "geometry",
            "center_world_unity": center_world_aligned.astype(float).tolist(),
            "center_world_cv": center_world_cv.astype(float).tolist(),
            "reference_center_world_m": center_world_aligned.astype(float).tolist(),
            "multicam_center_world_m": center_world_aligned.astype(float).tolist(),
            "size_xyz_m": size_xyz_m.astype(float).tolist(),
            "size_xyz_mm": (size_xyz_m * 1000.0).astype(float).tolist(),
            "yaw_rad": float(float(footprint_fit["yaw_rad"]) + math.pi),
            "yaw_deg": float(np.degrees(float(footprint_fit["yaw_rad"])) + 180.0),
            "reprojection_error_px": float(match.reprojection_error_px),
            "mean_confidence": float(match.mean_conf),
            "selected_cameras": list(match.detections.keys()),
            "sam_mask_width_by_camera_m": width_by_camera,
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
            "reference_height_details": label_details,
            "reference_height_details_by_camera": {},
            "per_camera_measurements": per_camera,
        }
        label_times[label] = float(time.perf_counter() - label_start)

    summary_json = output_dir / "single_depth_multicam_sam_size_summary.json"
    summary_md = output_dir / "single_depth_multicam_sam_size_summary.md"
    pointcloud_npz = output_dir / "single_depth_multicam_sam_size_pointclouds.npz"
    crop_rgb_png = output_dir / "single_depth_multicam_sam_size_crop_rgb.png"
    union_crop_rgb_png = output_dir / "single_depth_multicam_sam_size_union_crop_rgb.png"

    target_mask_full = np.asarray(masks_by_label[target_label][reference_camera_id], dtype=bool)
    target_crop_box_xyxy = [int(v) for v in bbox_from_mask(target_mask_full)]
    target_left, target_top, target_right, target_bottom = target_crop_box_xyxy
    target_crop_bgr = primary_image[target_top:target_bottom, target_left:target_right].copy()
    target_crop_mask = target_mask_full[target_top:target_bottom, target_left:target_right]
    target_crop_bgr[~target_crop_mask] = 0

    union_crop_box_xyxy = merge_xyxy_boxes(
        [
            bbox_from_mask(np.asarray(masks_by_label[label][reference_camera_id], dtype=bool))
            for label in primary_visible_labels
        ]
    )
    union_left, union_top, union_right, union_bottom = union_crop_box_xyxy
    union_crop_bgr = primary_image[union_top:union_bottom, union_left:union_right].copy()
    union_mask = np.zeros((union_bottom - union_top, union_right - union_left), dtype=bool)
    for label in primary_visible_labels:
        mask_full = np.asarray(masks_by_label[label][reference_camera_id], dtype=bool)
        union_mask |= mask_full[union_top:union_bottom, union_left:union_right]
    union_crop_bgr[~union_mask] = 0

    report_write_start = time.perf_counter()
    summary_md.write_text(build_size_markdown(labels, results), encoding="utf-8")
    timings["report_write"] = float(time.perf_counter() - report_write_start)

    npz_write_start = time.perf_counter()
    pointcloud_payload = build_pointcloud_npz_payload(
        results=results,
        reference_camera_id=reference_camera_id,
        crop_box_xyxy=target_crop_box_xyxy,
        box_point_budget=int(args.point_budget),
    )
    np.savez_compressed(str(pointcloud_npz), **pointcloud_payload)

    crop_image_write_start = time.perf_counter()
    cv2.imwrite(str(crop_rgb_png), target_crop_bgr)
    cv2.imwrite(str(union_crop_rgb_png), union_crop_bgr)

    summary = {
        "pipeline_name": PIPELINE_NAME,
        "entrypoint": str(Path(__file__).resolve()),
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "target_label": target_label,
        "reference_camera_id": reference_camera_id,
        "reference_axis_mode": None,
        "size_estimation_mode": "geometry",
        "height_scale": float(args.height_scale),
        "footprint_max_aspect_ratio": float(args.footprint_max_aspect_ratio),
        "primary_visible_labels": primary_visible_labels,
        "camera_ids": camera_ids,
        "yolo_cache": yolo_cache,
        "crop_visualization_label": target_label,
        "crop_box_xyxy": target_crop_box_xyxy,
        "union_crop_box_xyxy": list(union_crop_box_xyxy),
        "union_crop_labels": list(primary_visible_labels),
        "depth_affine_scale": None,
        "depth_affine_offset_m": None,
        "pointcloud_npz": str(pointcloud_npz),
        "crop_rgbd_pointcloud_npz": None,
        "union_crop_pointcloud_npz": None,
        "full_scene_pointcloud_npz": None,
        "crop_rgb_png": str(crop_rgb_png),
        "crop_depth_relative_vis_png": None,
        "crop_depth_metric_vis_png": None,
        "crop_depth_metric_mm_u16_png": None,
        "union_crop_rgb_png": str(union_crop_rgb_png),
        "union_crop_depth_relative_vis_png": None,
        "union_crop_depth_metric_vis_png": None,
        "union_crop_depth_metric_mm_u16_png": None,
        "timings_s": {
            **timings,
            "per_label_size_estimation": label_times,
            "pointcloud_npz_write": float(time.perf_counter() - npz_write_start),
            "crop_rgbd_pointcloud_npz_write": 0.0,
            "union_crop_pointcloud_npz_write": 0.0,
            "full_scene_pointcloud_npz_write": 0.0,
            "crop_depth_image_write": float(time.perf_counter() - crop_image_write_start),
            "report_write": float(timings["report_write"]),
            "total": 0.0,
        },
        "labels": results,
    }
    summary["timings_s"]["total"] = float(time.perf_counter() - stage_start)
    write_json(summary_json, summary)

    return SizeStageResult(
        summary=summary,
        label_results=results,
        output_paths={
            "summary_json": summary_json,
            "summary_markdown": summary_md,
            "pointcloud_npz": pointcloud_npz,
            "crop_rgbd_pointcloud_npz": None,
            "union_crop_pointcloud_npz": None,
            "full_scene_pointcloud_npz": None,
            "crop_rgb_png": crop_rgb_png,
            "crop_depth_relative_vis_png": None,
            "crop_depth_metric_vis_png": None,
            "crop_depth_metric_mm_u16_png": None,
            "union_crop_rgb_png": union_crop_rgb_png,
            "union_crop_depth_relative_vis_png": None,
            "union_crop_depth_metric_vis_png": None,
            "union_crop_depth_metric_mm_u16_png": None,
        },
    )


def run_grasp_stage(
    args: argparse.Namespace,
    output_dir: Path,
    summary_json: Path,
    label_results: dict[str, dict[str, object]],
) -> GraspStageResult:
    import trimesh.transformations as tra  # type: ignore

    stage_start = time.perf_counter()
    target_label = normalize_label(args.target_label)
    if target_label not in label_results:
        raise ValueError(f"Target label '{target_label}' not found in size summary.")

    object_points = np.zeros((0, 3), dtype=np.float32)
    object_colors = None
    target_num_points = 0
    obstacle_points = []
    obstacle_colors = []
    obstacle_labels: list[str] = []
    object_geometry: dict[str, dict[str, object]] = {}

    for label, info in label_results.items():
        center_world = np.asarray(
            info.get("center_world_unity", info["multicam_center_world_m"]),
            dtype=np.float32,
        )
        size_xyz = np.asarray(info["size_xyz_m"], dtype=np.float32)
        yaw_rad = float(info["yaw_rad"])
        pts = build_box_pointcloud(
            center_world=center_world,
            size_xyz=size_xyz,
            point_budget=int(args.point_budget),
            yaw_rad=yaw_rad,
        )
        object_geometry[label] = {
            "center_world_unity": center_world.astype(float).tolist(),
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
        raise RuntimeError(f"Failed to build target box point cloud for '{target_label}'.")

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

    object_pc_raw_c = tra.transform_points(object_points, t_center)
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
        scene_raw_c = np.zeros((0, 3), dtype=np.float32)
        collision_time_s = 0.0

    free_grasps = grasps_c[coll_mask]
    free_conf = conf[coll_mask]

    tag = output_tag(target_label)
    grasp_npz = output_dir / f"{tag}_single_depth_multicam_sam_size_grasp_result.npz"
    grasp_report_json = output_dir / f"{tag}_single_depth_multicam_sam_size_grasp_report.json"

    result_write_start = time.perf_counter()
    save_data = {
        "all_grasps": grasps_c,
        "all_scores": conf,
        "collision_free_mask": coll_mask,
        "collision_free_grasps": free_grasps,
        "collision_free_scores": free_conf,
        "pc_object": pc_c,
        "pc_object_raw": object_pc_raw_c,
        "pc_scene": scene_c,
        "pc_scene_raw": scene_raw_c,
        "pc_scene_colors": scene_colors,
        "target_label": np.array([target_label]),
        "source_summary": np.array([str(summary_json)]),
        "obstacle_labels": np.array(obstacle_labels, dtype=object),
    }
    if obj_colors_c is not None:
        save_data["pc_object_colors"] = obj_colors_c
    if object_colors is not None:
        save_data["pc_object_raw_colors"] = object_colors
    np.savez_compressed(str(grasp_npz), **save_data)
    result_write_time_s = time.perf_counter() - result_write_start

    report = {
        "summary_json": str(summary_json),
        "target_label": target_label,
        "target_mode": "multicam_geometry_sam_size_box",
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
            "result_npz_write": float(result_write_time_s),
            "report_write": 0.0,
            "total": 0.0,
        },
    }
    report_write_start = time.perf_counter()
    write_json(grasp_report_json, report)
    report["timings_s"]["report_write"] = float(time.perf_counter() - report_write_start)
    report["timings_s"]["total"] = float(time.perf_counter() - stage_start)
    write_json(grasp_report_json, report)

    return GraspStageResult(
        report=report,
        output_paths={
            "grasp_report_json": grasp_report_json,
            "grasp_result_npz": grasp_npz,
        },
    )


def build_final_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    size_stage: SizeStageResult,
    grasp_stage: GraspStageResult | None,
    total_time_s: float,
) -> dict[str, object]:
    target_label = normalize_label(args.target_label)
    description_path = output_dir / "current_pipeline_description_zh.json"

    def path_or_none(value: object) -> str | None:
        if value is None:
            return None
        return str(value)

    outputs = {
        "size_summary_json": str(size_stage.output_paths["summary_json"]),
        "size_summary_markdown": str(size_stage.output_paths["summary_markdown"]),
        "pointcloud_npz": str(size_stage.output_paths["pointcloud_npz"]),
        "crop_rgbd_pointcloud_npz": path_or_none(size_stage.output_paths["crop_rgbd_pointcloud_npz"]),
        "union_crop_pointcloud_npz": path_or_none(size_stage.output_paths["union_crop_pointcloud_npz"]),
        "full_scene_pointcloud_npz": path_or_none(size_stage.output_paths["full_scene_pointcloud_npz"]),
        "crop_rgb_png": path_or_none(size_stage.output_paths["crop_rgb_png"]),
        "crop_depth_relative_vis_png": path_or_none(size_stage.output_paths["crop_depth_relative_vis_png"]),
        "crop_depth_metric_vis_png": path_or_none(size_stage.output_paths["crop_depth_metric_vis_png"]),
        "crop_depth_metric_mm_u16_png": path_or_none(size_stage.output_paths["crop_depth_metric_mm_u16_png"]),
        "union_crop_rgb_png": path_or_none(size_stage.output_paths["union_crop_rgb_png"]),
        "union_crop_depth_relative_vis_png": path_or_none(size_stage.output_paths["union_crop_depth_relative_vis_png"]),
        "union_crop_depth_metric_vis_png": path_or_none(size_stage.output_paths["union_crop_depth_metric_vis_png"]),
        "union_crop_depth_metric_mm_u16_png": path_or_none(size_stage.output_paths["union_crop_depth_metric_mm_u16_png"]),
        "final_manifest_json": str(output_dir / "final_manifest.json"),
        "pipeline_description_json": str(description_path) if description_path.exists() else None,
        "grasp_report_json": (
            str(grasp_stage.output_paths["grasp_report_json"]) if grasp_stage is not None else None
        ),
        "grasp_result_npz": (
            str(grasp_stage.output_paths["grasp_result_npz"]) if grasp_stage is not None else None
        ),
    }
    return {
        "pipeline_name": PIPELINE_NAME,
        "entrypoint": str(Path(__file__).resolve()),
        "target_label": target_label,
        "skip_grasp": bool(args.skip_grasp),
        "inputs": {
            "image_dir": str(args.image_dir.expanduser().resolve()),
            "camera_parameter_dir": str(args.camera_parameter_dir.expanduser().resolve()),
            "camera_ids": [str(camera_id) for camera_id in args.camera_ids],
            "reference_camera_id": str(args.reference_camera_id),
            "labels": [normalize_label(label) for label in args.labels],
        },
        "parameters": {
            "conf_thresh": float(args.conf_thresh),
            "size_estimation_mode": "geometry",
            "height_scale": float(args.height_scale),
            "footprint_max_aspect_ratio": float(args.footprint_max_aspect_ratio),
            "robust_lower": float(args.robust_lower),
            "robust_upper": float(args.robust_upper),
            "axis_step_m": float(args.axis_step_m),
            "min_valid_height_m": float(args.min_valid_height_m),
            "max_valid_height_m": float(args.max_valid_height_m),
            "vertical_line_height_stat": str(args.vertical_line_height_stat),
            "disable_yolo_height_cap": bool(args.disable_yolo_height_cap),
            "point_budget": int(args.point_budget),
            "num_grasps": int(args.num_grasps),
            "topk": int(args.topk),
            "collision_thresh": float(args.collision_thresh),
        },
        "outputs": outputs,
        "size_stage": {
            "summary_json": str(size_stage.output_paths["summary_json"]),
            "timings_s": size_stage.summary["timings_s"],
            "labels": size_stage.label_results,
        },
        "grasp_stage": (
            grasp_stage.report
            if grasp_stage is not None
            else {
                "skipped": True,
                "reason": "--skip-grasp was set",
                "target_label": target_label,
            }
        ),
        "timings_s": {
            "size_stage_total": float(size_stage.summary["timings_s"]["total"]),
            "grasp_stage_total": (
                float(grasp_stage.report["timings_s"]["total"]) if grasp_stage is not None else 0.0
            ),
            "total": float(total_time_s),
        },
    }


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(args.image_dir.expanduser().resolve()).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    size_stage = run_size_stage(args, output_dir)
    grasp_stage = None if args.skip_grasp else run_grasp_stage(
        args,
        output_dir,
        size_stage.output_paths["summary_json"],
        size_stage.label_results,
    )
    manifest = build_final_manifest(
        args=args,
        output_dir=output_dir,
        size_stage=size_stage,
        grasp_stage=grasp_stage,
        total_time_s=float(time.perf_counter() - total_start),
    )
    manifest_path = output_dir / "final_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
