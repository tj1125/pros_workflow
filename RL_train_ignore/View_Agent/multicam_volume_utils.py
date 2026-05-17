#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageColor, ImageDraw
from scipy import ndimage

from test_multicam_teddy_height import (
    CandidateMatch,
    Detection,
    cv_world_to_unity,
    default_weights_path,
    height_estimates_from_bbox,
    image_path_for_camera,
    load_camera_models,
    match_target_detections,
    normalize_label,
    output_tag_from_labels,
    project_point,
    run_yolo,
    triangulate_optional,
)


VIEW_AGENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = VIEW_AGENT_ROOT.parent.parent

DEFAULT_IMAGE_DIR = VIEW_AGENT_ROOT / "test_tmp" / "pics" / "Camera_Room1_test"
DEFAULT_CAMERA_PARAMETER_DIR = (
    REPO_ROOT / "3090server" / "VLM_RL" / "get_item_info_agent" / "data" / "camera_parameter"
)
DEFAULT_CAMERA_IDS = ("Camera_Room1_2", "Camera_Room1_5", "Camera_Room1_7", "Camera_Room1_10")
DEFAULT_TARGET_LABELS = ("doll",)
SAM_CHECKPOINT_CANDIDATES = (
    REPO_ROOT / "3090server" / "VLM_RL" / "models" / "segmentation" / "sam_vit_b_01ec64.pth",
    REPO_ROOT / "3090server" / "VLM_RL" / "models" / "sam_vit_b_01ec64.pth",
)


@dataclass(frozen=True)
class PreparedScene:
    image_dir: Path
    camera_parameter_dir: Path
    camera_ids: list[str]
    target_labels: list[str]
    weights: Path
    min_views: int
    camera_models: dict[str, dict[str, np.ndarray]]
    detections_by_camera: dict[str, list[Detection]]
    match: CandidateMatch
    projections: dict[str, np.ndarray]
    image_size: tuple[int, int]


@dataclass(frozen=True)
class SearchBounds:
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    rough_width_m: float
    rough_height_m: float
    triangulated_height_m: float | None
    voxel_size_m: float

    @property
    def size_xyz(self) -> np.ndarray:
        return self.max_xyz - self.min_xyz


@dataclass(frozen=True)
class VolumeResult:
    occupied_points_world_cv: np.ndarray
    bounds_min_xyz: np.ndarray
    bounds_max_xyz: np.ndarray
    size_xyz_m: np.ndarray
    centroid_world_cv: np.ndarray
    height_m: float
    volume_m3: float
    occupied_voxel_count: int
    voxel_size_m: float

    @property
    def centroid_world_unity(self) -> np.ndarray:
        return cv_world_to_unity(self.centroid_world_cv)

    @property
    def bounds_min_unity(self) -> np.ndarray:
        return cv_world_to_unity(self.bounds_min_xyz)

    @property
    def bounds_max_unity(self) -> np.ndarray:
        return cv_world_to_unity(self.bounds_max_xyz)


def default_sam_checkpoint() -> Path:
    for path in SAM_CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    return SAM_CHECKPOINT_CANDIDATES[0]


def default_torch_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_scene(
    image_dir: Path,
    camera_parameter_dir: Path,
    camera_ids: Iterable[str],
    target_labels: Iterable[str],
    weights: Path,
    conf_thresh: float,
    device: str,
    min_views: int | None,
) -> PreparedScene:
    camera_ids_list = [str(camera_id) for camera_id in camera_ids]
    target_labels_list = [str(label) for label in target_labels]
    required_views = len(camera_ids_list) if min_views is None else int(min_views)
    if required_views < 2:
        raise ValueError("min_views must be at least 2.")
    if required_views > len(camera_ids_list):
        raise ValueError("min_views cannot exceed the number of camera ids.")

    for camera_id in camera_ids_list:
        image_path = image_path_for_camera(image_dir, camera_id)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

    camera_models = load_camera_models(camera_ids_list, camera_parameter_dir)
    detections_by_camera = run_yolo(
        camera_ids=camera_ids_list,
        image_dir=image_dir,
        weights=weights,
        conf_thresh=conf_thresh,
        device=device,
    )
    match = match_target_detections(
        detections_by_camera=detections_by_camera,
        camera_models=camera_models,
        target_labels=target_labels_list,
        min_views=required_views,
    )

    with Image.open(image_path_for_camera(image_dir, camera_ids_list[0])) as image:
        image_size = image.size

    projections = {camera_id: camera_models[camera_id]["P"] for camera_id in camera_ids_list}
    return PreparedScene(
        image_dir=image_dir,
        camera_parameter_dir=camera_parameter_dir,
        camera_ids=camera_ids_list,
        target_labels=target_labels_list,
        weights=weights,
        min_views=required_views,
        camera_models=camera_models,
        detections_by_camera=detections_by_camera,
        match=match,
        projections=projections,
        image_size=image_size,
    )


def _metric_width_estimates(match: CandidateMatch, camera_models: dict[str, dict[str, np.ndarray]]) -> dict[str, float]:
    estimates: dict[str, float] = {}
    for camera_id, det in match.detections.items():
        k = camera_models[camera_id]["K"]
        extrinsic = camera_models[camera_id]["extrinsic"]
        point_cam = extrinsic[:, :3] @ match.center_world_cv + extrinsic[:, 3]
        z_cam = float(point_cam[2])
        if z_cam <= 0:
            continue
        estimates[camera_id] = float(det.bbox_width_px * z_cam / float(k[0, 0]))
    return estimates


def estimate_search_bounds(
    scene: PreparedScene,
    voxel_size_m: float,
    margin_scale: float,
    min_half_xy_m: float,
    min_half_y_m: float,
) -> SearchBounds:
    bbox_height_m = height_estimates_from_bbox(scene.match, scene.camera_models)
    bbox_width_m = _metric_width_estimates(scene.match, scene.camera_models)

    rough_height_m = float(np.median(list(bbox_height_m.values()))) if bbox_height_m else 0.18
    rough_width_m = float(np.median(list(bbox_width_m.values()))) if bbox_width_m else max(0.08, rough_height_m * 0.5)

    top_world_cv = triangulate_optional(
        scene.projections,
        {camera_id: det.top_center_uv for camera_id, det in scene.match.detections.items()},
    )
    bottom_world_cv = triangulate_optional(
        scene.projections,
        {camera_id: det.bottom_center_uv for camera_id, det in scene.match.detections.items()},
    )
    triangulated_height_m = None
    if top_world_cv is not None and bottom_world_cv is not None:
        triangulated_height_m = float(np.linalg.norm(top_world_cv - bottom_world_cv))
        rough_height_m = max(rough_height_m, triangulated_height_m)

    half_x = max(min_half_xy_m, rough_width_m * 0.75) * margin_scale
    half_y = max(min_half_y_m, rough_height_m * 0.65) * margin_scale
    half_z = max(min_half_xy_m, rough_width_m * 0.75) * margin_scale

    extent = np.array([half_x, half_y, half_z], dtype=float)
    min_xyz = scene.match.center_world_cv - extent
    max_xyz = scene.match.center_world_cv + extent
    return SearchBounds(
        min_xyz=min_xyz,
        max_xyz=max_xyz,
        rough_width_m=rough_width_m,
        rough_height_m=rough_height_m,
        triangulated_height_m=triangulated_height_m,
        voxel_size_m=voxel_size_m,
    )


def build_voxel_grid(bounds: SearchBounds) -> tuple[np.ndarray, tuple[int, int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    step = float(bounds.voxel_size_m)
    xs = np.arange(bounds.min_xyz[0], bounds.max_xyz[0] + step * 0.5, step, dtype=float)
    ys = np.arange(bounds.min_xyz[1], bounds.max_xyz[1] + step * 0.5, step, dtype=float)
    zs = np.arange(bounds.min_xyz[2], bounds.max_xyz[2] + step * 0.5, step, dtype=float)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3), (len(xs), len(ys), len(zs)), (xs, ys, zs)


def project_world_points(camera_model: dict[str, np.ndarray], points_world_cv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points_h = np.concatenate([points_world_cv, np.ones((len(points_world_cv), 1), dtype=float)], axis=1)
    uvw = points_h @ camera_model["P"].T
    uv = uvw[:, :2] / uvw[:, 2:3]
    points_cam = points_world_cv @ camera_model["extrinsic"][:, :3].T + camera_model["extrinsic"][:, 3]
    return uv, points_cam[:, 2]


def keep_largest_component(mask_3d: np.ndarray) -> np.ndarray:
    if not mask_3d.any():
        return mask_3d
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labels, num_labels = ndimage.label(mask_3d, structure=structure)
    if num_labels <= 1:
        return mask_3d
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    keep_label = int(np.argmax(counts))
    return labels == keep_label


def carve_volume_with_bboxes(
    scene: PreparedScene,
    points_world_cv: np.ndarray,
    grid_shape: tuple[int, int, int],
    bbox_padding_px: float,
    keep_only_largest_component: bool = True,
) -> np.ndarray:
    occupied = np.ones(len(points_world_cv), dtype=bool)
    image_w, image_h = scene.image_size
    for camera_id, det in scene.match.detections.items():
        active_idx = np.flatnonzero(occupied)
        uv, depth = project_world_points(scene.camera_models[camera_id], points_world_cv[active_idx])
        u = uv[:, 0]
        v = uv[:, 1]
        inside = (
            (depth > 0.0)
            & (u >= 0.0)
            & (u <= image_w - 1)
            & (v >= 0.0)
            & (v <= image_h - 1)
            & (u >= det.x1 - bbox_padding_px)
            & (u <= det.x2 + bbox_padding_px)
            & (v >= det.y1 - bbox_padding_px)
            & (v <= det.y2 + bbox_padding_px)
        )
        occupied[:] = False
        occupied[active_idx[inside]] = True
        if not occupied.any():
            break

    occupied_3d = occupied.reshape(grid_shape)
    return keep_largest_component(occupied_3d) if keep_only_largest_component else occupied_3d


def carve_volume_with_masks(
    scene: PreparedScene,
    points_world_cv: np.ndarray,
    grid_shape: tuple[int, int, int],
    masks_by_camera: dict[str, np.ndarray],
    keep_only_largest_component: bool = True,
) -> np.ndarray:
    occupied = np.ones(len(points_world_cv), dtype=bool)
    image_w, image_h = scene.image_size
    for camera_id, det in scene.match.detections.items():
        active_idx = np.flatnonzero(occupied)
        uv, depth = project_world_points(scene.camera_models[camera_id], points_world_cv[active_idx])
        u = np.rint(uv[:, 0]).astype(int)
        v = np.rint(uv[:, 1]).astype(int)
        valid = (
            (depth > 0.0)
            & (u >= 0)
            & (u < image_w)
            & (v >= 0)
            & (v < image_h)
        )
        inside = np.zeros(len(active_idx), dtype=bool)
        if np.any(valid):
            inside_valid = masks_by_camera[camera_id][v[valid], u[valid]]
            inside[valid] = inside_valid
        occupied[:] = False
        occupied[active_idx[inside]] = True
        if not occupied.any():
            break

    occupied_3d = occupied.reshape(grid_shape)
    return keep_largest_component(occupied_3d) if keep_only_largest_component else occupied_3d


def summarize_occupied_volume(
    occupied_3d: np.ndarray,
    points_world_cv: np.ndarray,
    voxel_size_m: float,
) -> VolumeResult:
    flat = occupied_3d.reshape(-1)
    if not np.any(flat):
        raise RuntimeError("Voxel carving produced an empty volume. Try increasing padding, dilation, or search bounds.")

    occupied_points = points_world_cv[flat]
    bounds_min = occupied_points.min(axis=0) - voxel_size_m * 0.5
    bounds_max = occupied_points.max(axis=0) + voxel_size_m * 0.5
    size_xyz = bounds_max - bounds_min
    centroid = occupied_points.mean(axis=0)
    return VolumeResult(
        occupied_points_world_cv=occupied_points,
        bounds_min_xyz=bounds_min,
        bounds_max_xyz=bounds_max,
        size_xyz_m=size_xyz,
        centroid_world_cv=centroid,
        height_m=float(size_xyz[1]),
        volume_m3=float(len(occupied_points) * (voxel_size_m ** 3)),
        occupied_voxel_count=int(len(occupied_points)),
        voxel_size_m=float(voxel_size_m),
    )


def draw_projection_overlay(
    image_path: Path,
    detection: Detection,
    output_path: Path,
    info_text: str,
    projected_points_uv: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> None:
    image = Image.open(image_path).convert("RGBA")
    canvas = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([detection.x1, detection.y1, detection.x2, detection.y2], outline=(34, 177, 76, 255), width=4)
    cx, cy = detection.center_uv.tolist()
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=(255, 80, 80, 255))
    draw.text((detection.x1, max(0.0, detection.y1 - 16.0)), info_text, fill=(34, 177, 76, 255))

    if mask is not None:
        mask_img = Image.fromarray((mask.astype(np.uint8) * 120), mode="L")
        color = Image.new("RGBA", image.size, ImageColor.getrgb("#ff4444") + (0,))
        color.putalpha(mask_img)
        canvas = Image.alpha_composite(canvas, color)

    if projected_points_uv is not None and len(projected_points_uv) > 0:
        for u, v in projected_points_uv:
            if 0 <= u < image.size[0] and 0 <= v < image.size[1]:
                draw.point((float(u), float(v)), fill=(80, 160, 255, 255))

    out = Image.alpha_composite(image, canvas).convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def sample_projected_points_for_camera(
    scene: PreparedScene,
    camera_id: str,
    occupied_points_world_cv: np.ndarray,
    max_points: int = 2500,
) -> np.ndarray:
    if len(occupied_points_world_cv) == 0:
        return np.zeros((0, 2), dtype=float)
    stride = max(1, len(occupied_points_world_cv) // max_points)
    sample = occupied_points_world_cv[::stride]
    uv, depth = project_world_points(scene.camera_models[camera_id], sample)
    valid = depth > 0.0
    return uv[valid]


def save_volume_npz(output_path: Path, volume: VolumeResult, occupied_3d: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        occupied_points_world_cv=volume.occupied_points_world_cv,
        occupied_points_world_unity=np.asarray([cv_world_to_unity(p) for p in volume.occupied_points_world_cv]),
        bounds_min_xyz=volume.bounds_min_xyz,
        bounds_max_xyz=volume.bounds_max_xyz,
        centroid_world_cv=volume.centroid_world_cv,
        centroid_world_unity=volume.centroid_world_unity,
        size_xyz_m=volume.size_xyz_m,
        occupied_mask=occupied_3d,
        voxel_size_m=np.array([volume.voxel_size_m], dtype=float),
    )


def save_json(output_path: Path, payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
