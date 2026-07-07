from __future__ import annotations

import json
import logging
import math
import pickle
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np
import scipy.io
import yaml

from get_item_info_agent.pipeline.steps.obstacles import obstacle_cylinder_dimensions
from get_item_info_agent.pipeline.types import BoundingBox
from tool.vision.yolo import (
    extract_detections as extract_yolo_detections,
    load_yolo_model,
    run_yolo,
    select_best_bbox,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    det_id: str
    camera_id: str
    label: str
    conf: float
    bbox: BoundingBox

    @property
    def center_uv(self) -> np.ndarray:
        return self.bbox.center

    @property
    def top_left_uv(self) -> np.ndarray:
        return np.array(self.bbox.top_left, dtype=float)

    @property
    def top_right_uv(self) -> np.ndarray:
        return np.array(self.bbox.top_right, dtype=float)

    @property
    def bottom_left_uv(self) -> np.ndarray:
        return np.array(self.bbox.bottom_left, dtype=float)

    @property
    def bottom_right_uv(self) -> np.ndarray:
        return np.array(self.bbox.bottom_right, dtype=float)


@dataclass(frozen=True)
class MatchedObject:
    label: str
    detections: dict[str, Detection]
    center_world_cv: np.ndarray
    reprojection_error_px: float
    mean_conf: float


def detect_best_bbox(model, image_path: Path, class_name: str) -> tuple[BoundingBox, np.ndarray, float]:
    """Run YOLO inference and return the highest-confidence bbox for the target class."""
    result = run_yolo(model, image_path)
    if result.boxes is None or len(result.boxes) == 0:
        raise ValueError(f"No YOLO boxes predicted for image: {image_path}")

    best_bbox, best_conf = select_best_bbox(result, class_name)
    x1, y1, x2, y2 = [float(v) for v in best_bbox]
    return BoundingBox(x1, y1, x2, y2), result.orig_img.copy(), float(best_conf)


def detect_all_bboxes(
    model,
    image_path: Path,
    camera_id: str,
    conf_threshold: float,
) -> tuple[list[Detection], np.ndarray]:
    """Run YOLO inference and return all detections above the threshold."""
    result = run_yolo(model, image_path, conf_threshold=conf_threshold)
    image_bgr = result.orig_img.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return [], image_bgr

    detections: list[Detection] = []
    for idx, yolo_det in enumerate(
        extract_yolo_detections(result, conf_threshold=conf_threshold),
        start=1,
    ):
        x1, y1, x2, y2 = [float(v) for v in yolo_det.bbox_xyxy]
        detections.append(
            Detection(
                det_id=f"{camera_id}:{idx}",
                camera_id=camera_id,
                label=yolo_det.label,
                conf=float(yolo_det.conf),
                bbox=BoundingBox(x1, y1, x2, y2),
            )
        )
    return detections, image_bgr


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _camera_name_candidates(camera_id: str) -> list[str]:
    raw = str(camera_id)
    if raw.startswith("Camera_Room"):
        suffix = raw[len("Camera_Room") :]
    else:
        suffix = raw
    return list(
        dict.fromkeys(
            [
                raw,
                suffix,
                f"Camera_Room{suffix}",
                f"Camera_Room1_{suffix}",
                f"meta_{suffix}",
                f"unity_camera_{suffix}",
            ]
        )
    )


def _reshape_matrix(data, expected_shape: tuple[int, int], source_path: Path, key: str) -> np.ndarray:
    if isinstance(data, dict) and "data" in data:
        rows = int(data.get("rows", expected_shape[0]))
        cols = int(data.get("cols", expected_shape[1]))
        matrix = np.array(data["data"], dtype=float).reshape(rows, cols)
    else:
        matrix = np.array(data, dtype=float)
    if matrix.shape != expected_shape:
        raise ValueError(f"Invalid {key} shape in {source_path}: {matrix.shape}")
    return matrix


def _intrinsic_candidates(camera_id: str, intrinsics_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for root in (intrinsics_dir, intrinsics_dir / "yaml"):
        for stem in _camera_name_candidates(camera_id):
            paths.append(root / f"{stem}.yaml")
    suffix = str(camera_id)
    if suffix.startswith("Camera_Room"):
        suffix = suffix[len("Camera_Room") :]
    paths.append(intrinsics_dir / f"meta_{suffix}.mat")
    return _dedupe_paths(paths)


def _extrinsic_pkl_candidates(camera_id: str, extrinsics_dir: Path) -> list[Path]:
    return _dedupe_paths([extrinsics_dir / f"{stem}.pkl" for stem in _camera_name_candidates(camera_id)])


def _extrinsic_json_candidates(camera_id: str, extrinsics_dir: Path) -> list[Path]:
    json_dir = extrinsics_dir / "json"
    matches: list[Path] = []
    for stem in _camera_name_candidates(camera_id):
        matches.extend(sorted(json_dir.glob(f"{stem}_*.json")))
    return _dedupe_paths(matches)


def load_intrinsic_matrix(camera_id: str, intrinsics_dir: Path) -> np.ndarray:
    """Load the 3x3 intrinsic matrix for the given camera from YAML or legacy MAT."""
    for path in _intrinsic_candidates(camera_id, intrinsics_dir):
        if not path.exists():
            continue
        if path.suffix == ".yaml":
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            for key in ("camera_matrix", "intrinsic_matrix", "K"):
                if key in payload:
                    return _reshape_matrix(payload[key], (3, 3), path, key)
            raise ValueError(f"Intrinsic matrix key not found in {path}")
        if path.suffix == ".mat":
            mat = scipy.io.loadmat(path)
            for key in ("intrinsic_matrix", "K"):
                if key in mat:
                    return _reshape_matrix(mat[key], (3, 3), path, key)
            raise ValueError(f"Intrinsic matrix key not found in {path}")
    raise FileNotFoundError(f"Intrinsic matrix not found for camera '{camera_id}' in {intrinsics_dir}")


def load_extrinsic_matrix(camera_id: str, extrinsics_dir: Path) -> np.ndarray:
    """Load the 3x4 extrinsic matrix for the given camera from PKL or JSON."""
    payload = None
    source_path: Path | None = None
    for path in _extrinsic_pkl_candidates(camera_id, extrinsics_dir):
        if path.exists():
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            source_path = path
            break
    if payload is None:
        for path in _extrinsic_json_candidates(camera_id, extrinsics_dir):
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                source_path = path
                break
    if payload is None or source_path is None:
        raise FileNotFoundError(f"Extrinsic matrix not found for camera '{camera_id}' in {extrinsics_dir}")

    extrinsic = payload.get("extrinsic_matrix")
    if extrinsic is None:
        rotation = np.array(payload["rotation_matrix"], dtype=float)
        translation = np.array(payload["translation_vector"], dtype=float).reshape(3, 1)
        extrinsic = np.hstack([rotation, translation])

    extrinsic = np.array(extrinsic, dtype=float)
    if extrinsic.shape == (4, 4):
        extrinsic = extrinsic[:3, :]
    if extrinsic.shape != (3, 4):
        raise ValueError(f"Invalid extrinsic shape in {source_path}: {extrinsic.shape}")

    return extrinsic


def build_projection_matrix(camera_id: str, camera_parameter_dir: Path) -> np.ndarray:
    """Compute the 3×4 projection matrix P = K @ [R | t]."""
    intrinsics = load_intrinsic_matrix(camera_id, camera_parameter_dir / "Intrinsics")
    extrinsic = load_extrinsic_matrix(camera_id, camera_parameter_dir / "Extrinsic")
    return intrinsics @ extrinsic


def camera_center_from_projection(proj: np.ndarray) -> np.ndarray:
    """Recover the camera center C = -M^{-1} p4 from a projection matrix."""
    m = proj[:, :3]
    p4 = proj[:, 3]
    return -np.linalg.inv(m) @ p4


def ray_direction_from_projection(proj: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Compute the unit ray direction for a pixel coordinate uv."""
    ray = np.linalg.inv(proj[:, :3]) @ np.array([uv[0], uv[1], 1.0], dtype=float)
    return ray / np.linalg.norm(ray)


def triangulate_point_svd(p1: np.ndarray, uv1: np.ndarray, p2: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    """Triangulate a 3D point from two projection matrices and image coordinates via SVD."""
    u1, v1 = float(uv1[0]), float(uv1[1])
    u2, v2 = float(uv2[0]), float(uv2[1])
    a = np.vstack(
        [
            u1 * p1[2, :] - p1[0, :],
            v1 * p1[2, :] - p1[1, :],
            u2 * p2[2, :] - p2[0, :],
            v2 * p2[2, :] - p2[1, :],
        ]
    )
    _, _, vt = np.linalg.svd(a)
    xh = vt[-1]
    if np.isclose(xh[3], 0.0):
        raise ValueError("Triangulation failed with zero homogeneous coordinate.")
    return xh[:3] / xh[3]


def triangulate_point_rays(
    p1: np.ndarray,
    uv1: np.ndarray,
    p2: np.ndarray,
    uv2: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Triangulate using ray midpoint; falls back to SVD for ill-conditioned systems."""
    c1 = camera_center_from_projection(p1)
    c2 = camera_center_from_projection(p2)
    d1 = ray_direction_from_projection(p1, uv1)
    d2 = ray_direction_from_projection(p2, uv2)
    eye = np.eye(3)
    a = (eye - np.outer(d1, d1)) + (eye - np.outer(d2, d2))
    b = (eye - np.outer(d1, d1)) @ c1 + (eye - np.outer(d2, d2)) @ c2
    if np.linalg.cond(a) > 1 / eps:
        return triangulate_point_svd(p1, uv1, p2, uv2)
    return np.linalg.solve(a, b)


def triangulate_point_multi_view(
    projections: dict[str, np.ndarray],
    pixels_by_camera: dict[str, np.ndarray],
    eps: float = 1e-6,
) -> np.ndarray:
    """Triangulate from 2+ camera rays; falls back to SVD when the ray system is ill-conditioned."""
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


def cv_world_to_unity(point: np.ndarray) -> np.ndarray:
    """Convert a world point from OpenCV convention (z-forward) to Unity (z-backward)."""
    out = point.copy()
    out[2] *= -1.0
    return out


def load_camera_models(camera_ids: list[str], camera_parameter_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    models: dict[str, dict[str, np.ndarray]] = {}
    for camera_id in camera_ids:
        intrinsics = load_intrinsic_matrix(camera_id, camera_parameter_dir / "Intrinsics")
        extrinsic = load_extrinsic_matrix(camera_id, camera_parameter_dir / "Extrinsic")
        projection = intrinsics @ extrinsic
        models[camera_id] = {
            "K": intrinsics,
            "extrinsic": extrinsic,
            "P": projection,
            "camera_center": camera_center_from_projection(projection),
        }
    return models


def fallback_height_from_bbox(match: MatchedObject, camera_models: dict[str, dict[str, np.ndarray]]) -> float:
    estimates = []
    for camera_id, det in match.detections.items():
        k = camera_models[camera_id]["K"]
        extrinsic = camera_models[camera_id]["extrinsic"]
        point_cam = extrinsic[:, :3] @ match.center_world_cv + extrinsic[:, 3]
        z_cam = float(point_cam[2])
        if z_cam <= 0:
            continue
        estimates.append(det.bbox.height * z_cam / float(k[1, 1]))
    if not estimates:
        return 0.05
    return float(max(estimates))


def fallback_length_from_bbox(match: MatchedObject, camera_models: dict[str, dict[str, np.ndarray]]) -> float:
    estimates = []
    for camera_id, det in match.detections.items():
        k = camera_models[camera_id]["K"]
        extrinsic = camera_models[camera_id]["extrinsic"]
        point_cam = extrinsic[:, :3] @ match.center_world_cv + extrinsic[:, 3]
        z_cam = float(point_cam[2])
        if z_cam <= 0:
            continue
        estimates.append(det.bbox.width * z_cam / float(k[0, 0]))
    if not estimates:
        return 0.05
    return float(max(estimates))


def estimate_length_height(
    match: MatchedObject,
    camera_models: dict[str, dict[str, np.ndarray]],
    height_scale: float,
) -> tuple[float, float, float]:
    projections = {camera_id: info["P"] for camera_id, info in camera_models.items()}

    fallback_length = fallback_length_from_bbox(match, camera_models)
    fallback_height_raw = fallback_height_from_bbox(match, camera_models)

    corner_pixels = {
        "top_left": {camera_id: det.top_left_uv for camera_id, det in match.detections.items()},
        "top_right": {camera_id: det.top_right_uv for camera_id, det in match.detections.items()},
        "bottom_left": {camera_id: det.bottom_left_uv for camera_id, det in match.detections.items()},
        "bottom_right": {camera_id: det.bottom_right_uv for camera_id, det in match.detections.items()},
    }

    corner_points: dict[str, np.ndarray] = {}
    for name, pixels in corner_pixels.items():
        if len(pixels) < 2:
            continue
        try:
            corner_points[name] = triangulate_point_multi_view(projections, pixels)
        except Exception as exc:
            logger.debug("Corner triangulation failed for %s/%s: %s", match.label, name, exc)

    length_candidates = []
    if "top_left" in corner_points and "top_right" in corner_points:
        length_candidates.append(float(np.linalg.norm(corner_points["top_right"] - corner_points["top_left"])))
    if "bottom_left" in corner_points and "bottom_right" in corner_points:
        length_candidates.append(float(np.linalg.norm(corner_points["bottom_right"] - corner_points["bottom_left"])))

    height_candidates = []
    if "top_left" in corner_points and "bottom_left" in corner_points:
        height_candidates.append(float(np.linalg.norm(corner_points["bottom_left"] - corner_points["top_left"])))
    if "top_right" in corner_points and "bottom_right" in corner_points:
        height_candidates.append(float(np.linalg.norm(corner_points["bottom_right"] - corner_points["top_right"])))

    length_m = max(length_candidates) if length_candidates else fallback_length
    height_raw_m = max(height_candidates) if height_candidates else fallback_height_raw

    if length_m < 0.03:
        length_m = fallback_length
    if height_raw_m < 0.03:
        height_raw_m = fallback_height_raw

    length_m = max(float(length_m), 0.03)
    height_raw_m = max(float(height_raw_m), 0.03)
    height_scaled_m = height_raw_m * float(height_scale)
    return length_m, height_raw_m, max(float(height_scaled_m), 0.03)


def match_objects_multi_view(
    detections_by_camera: dict[str, list[Detection]],
    camera_models: dict[str, dict[str, np.ndarray]],
    preferred_camera_id: str | None = None,
) -> list[MatchedObject]:
    labels = sorted({det.label for dets in detections_by_camera.values() for det in dets})
    projections = {camera_id: info["P"] for camera_id, info in camera_models.items()}
    matches: list[MatchedObject] = []

    for label in labels:
        per_cam = {
            camera_id: [det for det in dets if det.label == label]
            for camera_id, dets in detections_by_camera.items()
        }
        available_cameras = [camera_id for camera_id, dets in per_cam.items() if dets]
        if len(available_cameras) < 2:
            continue

        candidates = []
        max_views = min(3, len(available_cameras))
        for num_views in range(max_views, 1, -1):
            for camera_subset in combinations(available_cameras, num_views):
                for det_tuple in product(*(per_cam[camera_id] for camera_id in camera_subset)):
                    pixels = {
                        camera_id: det.center_uv for camera_id, det in zip(camera_subset, det_tuple)
                    }
                    point_world_cv = triangulate_point_multi_view(projections, pixels)
                    errors = []
                    for camera_id, det in zip(camera_subset, det_tuple):
                        uv_proj = project_point(projections[camera_id], point_world_cv)
                        errors.append(float(np.linalg.norm(uv_proj - det.center_uv)))
                    candidates.append(
                        {
                            "detections": {camera_id: det for camera_id, det in zip(camera_subset, det_tuple)},
                            "center_world_cv": point_world_cv,
                            "mean_error": float(np.mean(errors)),
                            "mean_conf": float(np.mean([det.conf for det in det_tuple])),
                        }
                    )

        candidates.sort(
            key=lambda item: (
                -len(item["detections"]),
                0 if preferred_camera_id and preferred_camera_id in item["detections"] else 1,
                item["mean_error"],
                -item["mean_conf"],
            )
        )

        used_det_ids: set[str] = set()
        for candidate in candidates:
            det_ids = {det.det_id for det in candidate["detections"].values()}
            if det_ids & used_det_ids:
                continue
            used_det_ids |= det_ids
            matches.append(
                MatchedObject(
                    label=label,
                    detections=candidate["detections"],
                    center_world_cv=np.asarray(candidate["center_world_cv"], dtype=float),
                    reprojection_error_px=float(candidate["mean_error"]),
                    mean_conf=float(candidate["mean_conf"]),
                )
            )

    matches.sort(key=lambda item: (-len(item.detections), item.label, -item.mean_conf, item.reprojection_error_px))
    return matches


def choose_target_match(
    matches: list[MatchedObject],
    target_label: str,
    primary_camera_id: str | None = None,
) -> MatchedObject:
    target_label = target_label.lower()
    targets = [match for match in matches if match.label == target_label]
    if not targets:
        raise RuntimeError(f"Target label '{target_label}' was not matched across at least two cameras.")

    if primary_camera_id:
        visible_targets = [match for match in targets if primary_camera_id in match.detections]
        if not visible_targets:
            raise RuntimeError(
                f"Target '{target_label}' was not detected in the selected primary camera '{primary_camera_id}'."
            )
        targets = visible_targets

    return max(
        targets,
        key=lambda item: (len(item.detections), item.mean_conf, -item.reprojection_error_px),
    )


def select_best_camera_pair(
    cfg: dict,
    image_paths_by_camera: dict[str, Path],
    yolo_class_name: str,
    preferred_camera: str | None = None,
) -> tuple[str, str]:
    """Pick a camera pair from the uploaded group, preferring the user-selected camera."""
    if len(image_paths_by_camera) < 2:
        raise ValueError("At least two camera images are required to select a stereo pair.")

    yolo = load_yolo_model(Path(cfg["models"]["yolo_weights"]))
    detections: dict[str, dict] = {}

    for camera_name, image_path in image_paths_by_camera.items():
        try:
            bbox, _, conf = detect_best_bbox(yolo, image_path, yolo_class_name)
            detections[camera_name] = {
                "bbox": bbox,
                "conf": float(conf),
            }
        except Exception as exc:
            logger.info(
                "Pair selection: skipping camera=%s class=%s reason=%s",
                camera_name,
                yolo_class_name,
                exc,
            )

    if len(detections) < 2:
        raise ValueError(
            f"Need detections from at least 2 cameras in the selected group for '{yolo_class_name}'."
        )

    if preferred_camera in detections:
        partner_names = [camera_name for camera_name in detections if camera_name != preferred_camera]
        if partner_names:
            partner_name = max(
                partner_names,
                key=lambda camera_name: (detections[camera_name]["conf"], str(camera_name)),
            )
            logger.info(
                "Pair selection: preferred=%s partner=%s confs=(%.3f, %.3f)",
                preferred_camera,
                partner_name,
                detections[preferred_camera]["conf"],
                detections[partner_name]["conf"],
            )
            return preferred_camera, partner_name

    pair_candidates: list[tuple[int, float, float, str, str]] = []
    ordered_names = sorted(detections)
    for idx, camera_a_name in enumerate(ordered_names):
        for camera_b_name in ordered_names[idx + 1 :]:
            conf_a = detections[camera_a_name]["conf"]
            conf_b = detections[camera_b_name]["conf"]
            pair_candidates.append(
                (
                    1 if preferred_camera and preferred_camera in (camera_a_name, camera_b_name) else 0,
                    conf_a + conf_b,
                    max(conf_a, conf_b),
                    camera_a_name,
                    camera_b_name,
                )
            )

    _, _, _, camera_a_name, camera_b_name = max(pair_candidates)
    if preferred_camera == camera_b_name and preferred_camera != camera_a_name:
        camera_a_name, camera_b_name = camera_b_name, camera_a_name
    elif detections[camera_b_name]["conf"] > detections[camera_a_name]["conf"]:
        camera_a_name, camera_b_name = camera_b_name, camera_a_name

    logger.info(
        "Pair selection: fallback pair=(%s, %s) confs=(%.3f, %.3f)",
        camera_a_name,
        camera_b_name,
        detections[camera_a_name]["conf"],
        detections[camera_b_name]["conf"],
    )
    return camera_a_name, camera_b_name


def run_detection_and_triangulation(
    cfg: dict,
    image_a_path: Path,
    image_b_path: Path,
    yolo_class_name: str,
    camera_a_id: str | None = None,
    camera_b_id: str | None = None,
) -> dict:
    """Legacy two-view detection+triangulation path kept for backward compatibility."""
    runtime = cfg["runtime"]
    camera_cfg = cfg["camera"]

    yolo = load_yolo_model(Path(cfg["models"]["yolo_weights"]))
    bbox_a, image_a_bgr, _ = detect_best_bbox(yolo, image_a_path, yolo_class_name)
    bbox_b, _, _ = detect_best_bbox(yolo, image_b_path, yolo_class_name)

    camera_a_name = str(camera_a_id or camera_cfg["camera_a"])
    camera_b_name = str(camera_b_id or camera_cfg["camera_b"])

    proj_a = build_projection_matrix(camera_a_name, Path(camera_cfg["camera_parameter_dir"]))
    proj_b = build_projection_matrix(camera_b_name, Path(camera_cfg["camera_parameter_dir"]))

    top_left_world = triangulate_point_rays(proj_a, np.array(bbox_a.top_left), proj_b, np.array(bbox_b.top_left))
    bottom_left_world = triangulate_point_rays(
        proj_a,
        np.array(bbox_a.bottom_left),
        proj_b,
        np.array(bbox_b.bottom_left),
    )
    center_world = triangulate_point_rays(proj_a, bbox_a.center, proj_b, bbox_b.center)

    top_left_world = cv_world_to_unity(top_left_world)
    bottom_left_world = cv_world_to_unity(bottom_left_world)
    center_world = cv_world_to_unity(center_world)

    target_height = float(np.linalg.norm(top_left_world - bottom_left_world)) * float(runtime["height_scale"])

    return {
        "bbox_a": bbox_a,
        "bbox_b": bbox_b,
        "image_a_bgr": image_a_bgr,
        "center_world": center_world,
        "target_height": target_height,
        "camera_a_id": camera_a_name,
        "camera_b_id": camera_b_name,
    }


def run_detection_and_triangulation_multi_view(
    cfg: dict,
    image_paths_by_camera: dict[str, Path],
    yolo_class_name: str,
    primary_camera_id: str | None = None,
) -> dict:
    """Run YOLO on all uploaded images, triangulate all detectable objects, and pick a primary view."""
    if len(image_paths_by_camera) < 2:
        raise ValueError("At least two camera images are required for 3D localization.")
    if primary_camera_id and primary_camera_id not in image_paths_by_camera:
        raise ValueError(f"Primary camera '{primary_camera_id}' was not included in the uploaded images.")

    runtime = cfg["runtime"]
    camera_cfg = cfg["camera"]
    yolo_conf_threshold = float(runtime.get("yolo_conf_threshold", 0.25))
    height_scale = float(runtime.get("height_scale", 1.0))

    yolo = load_yolo_model(Path(cfg["models"]["yolo_weights"]))
    detections_by_camera: dict[str, list[Detection]] = {}
    images_by_camera: dict[str, np.ndarray] = {}
    for camera_id, image_path in image_paths_by_camera.items():
        detections, image_bgr = detect_all_bboxes(
            yolo,
            image_path,
            camera_id=camera_id,
            conf_threshold=yolo_conf_threshold,
        )
        detections_by_camera[camera_id] = detections
        images_by_camera[camera_id] = image_bgr

    camera_ids = list(image_paths_by_camera.keys())
    camera_models = load_camera_models(camera_ids, Path(camera_cfg["camera_parameter_dir"]))
    matches = match_objects_multi_view(
        detections_by_camera,
        camera_models,
        preferred_camera_id=primary_camera_id,
    )
    target_match = choose_target_match(matches, yolo_class_name, primary_camera_id=primary_camera_id)

    resolved_primary_camera = primary_camera_id
    if not resolved_primary_camera:
        resolved_primary_camera = max(
            target_match.detections.keys(),
            key=lambda camera_id: target_match.detections[camera_id].conf,
        )
    if resolved_primary_camera not in target_match.detections:
        raise RuntimeError(
            f"Target '{yolo_class_name}' has no detection in primary camera '{resolved_primary_camera}'."
        )

    object_reports = []
    target_report = None
    for match in matches:
        center_world_unity = cv_world_to_unity(match.center_world_cv)
        length_m, height_raw_m, height_m = estimate_length_height(match, camera_models, height_scale=height_scale)
        obstacle_diameter_m, obstacle_height_m = obstacle_cylinder_dimensions(length_m, height_m)
        report = {
            "label": match.label,
            "used_cameras": sorted(match.detections.keys()),
            "mean_conf": float(match.mean_conf),
            "reprojection_error_px": float(match.reprojection_error_px),
            "center_world_cv": match.center_world_cv.tolist(),
            "center_world_unity": center_world_unity.tolist(),
            "length_m": float(length_m),
            "length_mm": float(length_m * 1000.0),
            "height_raw_m": float(height_raw_m),
            "height_m": float(height_m),
            "height_mm": float(height_m * 1000.0),
            "obstacle_shape": "cylinder",
            "obstacle_diameter_m": float(obstacle_diameter_m),
            "obstacle_height_m": float(obstacle_height_m),
            "bboxes_by_camera": {
                camera_id: [
                    float(det.bbox.x1),
                    float(det.bbox.y1),
                    float(det.bbox.x2),
                    float(det.bbox.y2),
                ]
                for camera_id, det in match.detections.items()
            },
            "bbox_conf_by_camera": {
                camera_id: float(det.conf)
                for camera_id, det in match.detections.items()
            },
        }
        object_reports.append(report)
        if match is target_match:
            target_report = report

    if target_report is None:
        raise RuntimeError(f"Failed to build object report for target '{yolo_class_name}'.")

    primary_bbox = target_match.detections[resolved_primary_camera].bbox
    return {
        "primary_camera_id": resolved_primary_camera,
        "primary_bbox": primary_bbox,
        "primary_image_bgr": images_by_camera[resolved_primary_camera],
        "center_world": np.array(target_report["center_world_unity"], dtype=float),
        "target_height": float(target_report["height_m"]),
        "target_length": float(target_report["length_m"]),
        "target_object": target_report,
        "objects": object_reports,
        "num_matched_objects": len(object_reports),
        "camera_ids": camera_ids,
    }
