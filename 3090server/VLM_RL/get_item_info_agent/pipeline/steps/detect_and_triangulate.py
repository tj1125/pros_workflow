from __future__ import annotations

import json
import logging
import math
import pickle
from pathlib import Path

import numpy as np
import scipy.io
import yaml

from get_item_info_agent.pipeline.types import BoundingBox

logger = logging.getLogger(__name__)


def load_yolo_model(weights_path: Path):
    """Load a YOLO model from the given weights file."""
    from ultralytics import YOLO  # type: ignore

    return YOLO(str(weights_path))


def detect_best_bbox(model, image_path: Path, class_name: str) -> tuple[BoundingBox, np.ndarray, float]:
    """Run YOLO inference and return the highest-confidence bbox for the target class."""
    results = model(str(image_path))
    if not results:
        raise RuntimeError(f"No YOLO result for image: {image_path}")

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        raise ValueError(f"No YOLO boxes predicted for image: {image_path}")

    best = None
    best_conf = -np.inf
    target = class_name.lower()
    name_lookup = {int(k): v for k, v in result.names.items()}

    for box in result.boxes:
        cls_id = int(box.cls.item())
        label = name_lookup.get(cls_id, str(cls_id)).lower()
        if label != target:
            continue
        conf = float(box.conf.item())
        if conf > best_conf:
            best_conf = conf
            best = box.xyxy.cpu().numpy()[0]

    if best is None:
        raise ValueError(f"No class '{class_name}' detection in {image_path}")

    x1, y1, x2, y2 = map(float, best)
    return BoundingBox(x1, y1, x2, y2), result.orig_img.copy(), best_conf


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


def cv_world_to_unity(point: np.ndarray) -> np.ndarray:
    """Convert a world point from OpenCV convention (z-forward) to Unity (z-backward)."""
    out = point.copy()
    out[2] *= -1.0
    return out


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
    """Run YOLO detection on both images and triangulate the 3D world position."""
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
