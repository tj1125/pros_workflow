#!/usr/bin/env python3
"""
Visualize single-depth + multi-camera SAM size estimates in MeshCat.

Supports either:
  - summary JSON
  - summary NPZ
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

from test_multicam_teddy_height import DEFAULT_CAMERA_PARAMETER_DIR, load_camera_models


AGENT_ROOT = Path("/home/tjchen/workspace/VLM_RL/3090server/VLM_RL/get_item_info_agent")
GRASPGEN_VENDOR = AGENT_ROOT / "vendor/graspgen_runtime"

for p in [str(GRASPGEN_VENDOR), str(GRASPGEN_VENDOR / "pointnet2_ops")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from grasp_gen.utils.meshcat_utils import create_visualizer, visualize_mesh, visualize_pointcloud  # type: ignore


DEFAULT_SUMMARY_JSON = Path(
    "/home/tjchen/workspace/VLM_RL/RL_train/View_Agent/test_tmp/pics/Camera_Room1_test/"
    "four_camera_eye_level_experiment_outputs/Camera_Room1_12/"
    "single_depth_multicam_sam_size_outputs/single_depth_multicam_sam_size_summary.json"
)
LABEL_COLORS = {
    "doll": [60, 220, 90],
    "apple": [255, 140, 40],
    "wine": [70, 160, 255],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize size estimates from single-depth + multi-camera SAM summary.")
    parser.add_argument("summary_path", nargs="?", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--camera-parameter-dir", type=Path, default=DEFAULT_CAMERA_PARAMETER_DIR)
    parser.add_argument("--show-cameras", action="store_true")
    parser.add_argument("--show-centers", action="store_true")
    parser.add_argument("--center-point-size", type=float, default=0.02)
    parser.add_argument("--camera-point-size", type=float, default=0.02)
    return parser.parse_args()


def yaw_rotation_matrix(yaw_rad: float) -> np.ndarray:
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    return np.array(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def translation_matrix(translation: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float32)
    t[:3, 3] = np.asarray(translation, dtype=np.float32).reshape(3)
    return t


def cv_world_to_aligned_world(point_world_cv: np.ndarray) -> np.ndarray:
    point_world = np.asarray(point_world_cv, dtype=np.float32).reshape(3).copy()
    point_world[2] *= -1.0
    return point_world


def build_box_mesh(size_xyz_m: np.ndarray, center_world_m: np.ndarray, yaw_rad: float) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=np.asarray(size_xyz_m, dtype=np.float32))
    transform = translation_matrix(center_world_m) @ yaw_rotation_matrix(yaw_rad)
    mesh.apply_transform(transform)
    return mesh


def load_payload(summary_path: Path) -> tuple[dict[str, dict[str, object]], list[str]]:
    if summary_path.suffix.lower() == ".json":
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        return payload["labels"], [str(camera_id) for camera_id in payload.get("camera_ids", [])]

    if summary_path.suffix.lower() == ".npz":
        arr = np.load(str(summary_path), allow_pickle=True)
        labels = [str(v) for v in arr["labels"].tolist()]
        camera_ids = [str(v) for v in arr["camera_ids"].tolist()] if "camera_ids" in arr.files else []
        label_payload: dict[str, dict[str, object]] = {}
        for idx, label in enumerate(labels):
            key = label.replace(" ", "_")
            center = np.asarray(arr[f"{key}_center_world_m"], dtype=np.float32)
            size = np.asarray(arr[f"{key}_size_xyz_m"], dtype=np.float32)
            yaw = float(np.asarray(arr[f"{key}_yaw_rad"], dtype=np.float32).reshape(-1)[0])
            label_payload[label] = {
                "multicam_center_world_m": center.astype(float).tolist(),
                "size_xyz_m": size.astype(float).tolist(),
                "yaw_rad": yaw,
            }
        return label_payload, camera_ids

    raise ValueError(f"Unsupported summary format: {summary_path}")


def main() -> None:
    args = parse_args()
    summary_path = args.summary_path.expanduser().resolve()
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    labels, camera_ids = load_payload(summary_path)

    vis = create_visualizer()
    vis.delete()
    print(f"Summary: {summary_path}")
    print(f"MeshCat URL: {vis.url()}")

    for label, info in labels.items():
        color = LABEL_COLORS.get(label, [180, 180, 180])
        center_world = np.asarray(info["multicam_center_world_m"], dtype=np.float32)
        size_xyz = np.asarray(info["size_xyz_m"], dtype=np.float32)
        yaw_rad = float(info["yaw_rad"])
        mesh = build_box_mesh(size_xyz, center_world, yaw_rad)
        visualize_mesh(vis, f"objects/{label}/box", mesh, color=color)
        if args.show_centers:
            visualize_pointcloud(
                vis,
                f"objects/{label}/center",
                center_world.reshape(1, 3),
                color,
                size=float(args.center_point_size),
            )

    if args.show_cameras and camera_ids:
        models = load_camera_models(camera_ids, args.camera_parameter_dir.expanduser().resolve())
        cam_points = []
        cam_colors = []
        for camera_id in camera_ids:
            cam_center = cv_world_to_aligned_world(models[camera_id]["camera_center"])
            cam_points.append(cam_center.reshape(1, 3))
            cam_colors.append(np.array([[220, 220, 220]], dtype=np.uint8))
            print(f"{camera_id}: {cam_center.tolist()}")
        if cam_points:
            visualize_pointcloud(
                vis,
                "cameras/centers",
                np.concatenate(cam_points, axis=0).astype(np.float32),
                np.concatenate(cam_colors, axis=0).astype(np.uint8),
                size=float(args.camera_point_size),
            )

    print("Visualization ready.")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
