from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import scipy.io

from get_item_info_agent.pipeline.types import BoundingBox


def load_yolo_model(weights_path: Path):
    """Load a YOLO model from the given weights file."""
    from ultralytics import YOLO  # type: ignore

    return YOLO(str(weights_path))


def detect_best_bbox(model, image_path: Path, class_name: str) -> tuple[BoundingBox, np.ndarray]:
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
    return BoundingBox(x1, y1, x2, y2), result.orig_img.copy()


def load_intrinsic_matrix(camera_id: str, intrinsics_dir: Path) -> np.ndarray:
    """Load the 3×3 intrinsic matrix for the given camera from a .mat file."""
    mat_path = intrinsics_dir / f"meta_{camera_id}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"Intrinsic matrix not found: {mat_path}")

    mat = scipy.io.loadmat(mat_path)
    intrinsic = None
    for key in ("intrinsic_matrix", "K"):
        if key in mat:
            intrinsic = np.array(mat[key], dtype=float)
            break

    if intrinsic is None or intrinsic.shape != (3, 3):
        raise ValueError(f"Invalid intrinsic matrix in {mat_path}")
    return intrinsic


def load_extrinsic_matrix(camera_id: str, extrinsics_dir: Path) -> np.ndarray:
    """Load the 3×4 extrinsic matrix for the given camera from a .pkl file."""
    pkl_path = extrinsics_dir / f"unity_camera_{camera_id}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Extrinsic matrix not found: {pkl_path}")

    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)

    extrinsic = payload.get("extrinsic_matrix")
    if extrinsic is None:
        rotation = np.array(payload["rotation_matrix"], dtype=float)
        translation = np.array(payload["translation_vector"], dtype=float).reshape(3, 1)
        extrinsic = np.hstack([rotation, translation])

    extrinsic = np.array(extrinsic, dtype=float)
    if extrinsic.shape == (4, 4):
        extrinsic = extrinsic[:3, :]
    if extrinsic.shape != (3, 4):
        raise ValueError(f"Invalid extrinsic shape in {pkl_path}: {extrinsic.shape}")

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


def run_detection_and_triangulation(
    cfg: dict,
    image_a_path: Path,
    image_b_path: Path,
    yolo_class_name: str,
) -> dict:
    """Run YOLO detection on both images and triangulate the 3D world position."""
    runtime = cfg["runtime"]
    camera_cfg = cfg["camera"]

    yolo = load_yolo_model(Path(cfg["models"]["yolo_weights"]))
    bbox_a, image_a_bgr = detect_best_bbox(yolo, image_a_path, yolo_class_name)
    bbox_b, _ = detect_best_bbox(yolo, image_b_path, yolo_class_name)

    proj_a = build_projection_matrix(str(camera_cfg["camera_a"]), Path(camera_cfg["camera_parameter_dir"]))
    proj_b = build_projection_matrix(str(camera_cfg["camera_b"]), Path(camera_cfg["camera_parameter_dir"]))

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
    }
