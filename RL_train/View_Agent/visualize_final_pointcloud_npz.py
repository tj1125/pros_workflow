#!/usr/bin/env python3
"""
Visualize final.py point-cloud NPZ outputs in MeshCat.

Supports viewing:
  - raw reference-depth point clouds
  - prefiltered point clouds
  - estimated box point clouds
  - optional PCA/local axes stored in the NPZ
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


AGENT_ROOT = Path("/home/tjchen/workspace/VLM_RL/3090server/VLM_RL/get_item_info_agent")
GRASPGEN_VENDOR = AGENT_ROOT / "vendor" / "graspgen_runtime"

for path in (GRASPGEN_VENDOR, GRASPGEN_VENDOR / "pointnet2_ops"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


DEFAULT_POINTCLOUD_NPZ = Path(
    "/home/tjchen/workspace/VLM_RL/RL_train/View_Agent/test_tmp/pics/Camera_Room1_test/"
    "final_outputs/single_depth_multicam_sam_size_pointclouds.npz"
)
LABEL_COLORS = {
    "doll": np.array([60, 220, 90], dtype=np.uint8),
    "apple": np.array([255, 140, 40], dtype=np.uint8),
    "wine": np.array([70, 160, 255], dtype=np.uint8),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize final point-cloud NPZ in MeshCat.")
    parser.add_argument("pointcloud_npz", nargs="?", type=Path, default=DEFAULT_POINTCLOUD_NPZ)
    parser.add_argument("--mode", choices=("raw", "prefiltered", "box", "dense", "labels", "all"), default="all")
    parser.add_argument("--label", default="", help="Show only one label, e.g. doll/apple/wine.")
    parser.add_argument("--point-size", type=float, default=0.008)
    parser.add_argument(
        "--stage-offset",
        type=float,
        default=0.6,
        help="X offset between raw/prefiltered/box clouds when visualizing the final pointcloud schema in --mode all.",
    )
    parser.add_argument("--frame", choices=("world", "camera"), default="world")
    parser.add_argument(
        "--filtered-labels",
        action="store_true",
        help="For crop RGBD NPZ, visualize per-label kNN-filtered clouds if available.",
    )
    parser.add_argument("--show-centers", action="store_true")
    parser.add_argument("--show-axes", action="store_true")
    parser.add_argument("--axis-length", type=float, default=0.15)
    return parser.parse_args()


def label_key(label: str) -> str:
    return " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split()).replace(" ", "_")


def axis_points(axis_length: float, num_points: int = 60) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, axis_length, num_points, dtype=np.float32)
    zeros = np.zeros_like(xs)
    pts_x = np.stack([xs, zeros, zeros], axis=1)
    pts_y = np.stack([zeros, xs, zeros], axis=1)
    pts_z = np.stack([zeros, zeros, xs], axis=1)
    pts = np.concatenate([pts_x, pts_y, pts_z], axis=0)
    colors = np.concatenate(
        [
            np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (num_points, 1)),
            np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (num_points, 1)),
            np.tile(np.array([[0, 128, 255]], dtype=np.uint8), (num_points, 1)),
        ],
        axis=0,
    )
    return pts, colors


def transform_local_to_world(local_points: np.ndarray, axes_world: np.ndarray, center_world: np.ndarray) -> np.ndarray:
    return local_points @ np.asarray(axes_world, dtype=np.float32) + np.asarray(center_world, dtype=np.float32).reshape(1, 3)


def payload_schema(payload: np.lib.npyio.NpzFile) -> str:
    if "full_scene_rgbd_schema" in payload.files:
        return "full_scene_rgbd"
    if "segmented_crop_rgbd_schema" in payload.files:
        return "segmented_crop_rgbd"
    if "crop_rgb_uint8" in payload.files and "pointcloud_world_m" in payload.files:
        return "crop_rgbd"
    return "final_pointcloud"


def repeat_color(color: np.ndarray, count: int) -> np.ndarray:
    return np.repeat(np.asarray(color, dtype=np.uint8).reshape(1, 3), count, axis=0)


def label_entries(payload: np.lib.npyio.NpzFile) -> list[tuple[str, str]]:
    if "labels" in payload.files:
        labels = [str(x) for x in payload["labels"].tolist()]
        return [(label, label_key(label)) for label in labels]
    prefixes = sorted({key[: -len("_crop_mask")] for key in payload.files if key.endswith("_crop_mask")})
    return [(prefix, prefix) for prefix in prefixes]


def filter_label_entries(entries: list[tuple[str, str]], label: str) -> list[tuple[str, str]]:
    if not label:
        return entries
    selected = label_key(label)
    filtered = [(display, prefix) for display, prefix in entries if label_key(display) == selected or prefix == selected]
    if not filtered:
        raise KeyError(f"Label not found: {label}")
    return filtered


def stage_suffixes(mode: str) -> list[tuple[str, str, str]]:
    mapping = {
        "raw": [("raw", "raw_points_world_m", "raw_points_colors")],
        "prefiltered": [("prefiltered", "prefiltered_points_world_m", "prefiltered_points_colors")],
        "box": [("box", "estimated_box_points_world_m", "estimated_box_points_colors")],
        "all": [
            ("raw", "raw_points_world_m", "raw_points_colors"),
            ("prefiltered", "prefiltered_points_world_m", "prefiltered_points_colors"),
            ("box", "estimated_box_points_world_m", "estimated_box_points_colors"),
        ],
    }
    return mapping[mode]


def visualize_final_pointcloud_payload(payload, args, vis, visualize_pointcloud) -> None:
    entries = filter_label_entries(label_entries(payload), args.label)
    point_size = float(args.point_size)
    stages = stage_suffixes(args.mode)
    stage_offset = float(args.stage_offset)

    for stage_idx, (stage_name, points_suffix, colors_suffix) in enumerate(stages):
        shift = (
            np.array([stage_idx * stage_offset, 0.0, 0.0], dtype=np.float32)
            if len(stages) > 1
            else np.zeros(3, dtype=np.float32)
        )
        for display_label, prefix in entries:
            points_key = f"{prefix}_{points_suffix}"
            colors_key = f"{prefix}_{colors_suffix}"
            if points_key not in payload.files:
                continue
            points = np.asarray(payload[points_key], dtype=np.float32) + shift.reshape(1, 3)
            colors = (
                np.asarray(payload[colors_key], dtype=np.uint8)
                if colors_key in payload.files
                else repeat_color(np.array([180, 180, 180], dtype=np.uint8), len(points))
            )
            visualize_pointcloud(
                vis,
                f"{stage_name}/{prefix}/points",
                points,
                colors,
                size=point_size,
            )

            if args.show_centers and f"{prefix}_center_world_m" in payload.files:
                center = np.asarray(payload[f"{prefix}_center_world_m"], dtype=np.float32).reshape(1, 3)
                center = center + shift.reshape(1, 3)
                visualize_pointcloud(
                    vis,
                    f"{stage_name}/{prefix}/center",
                    center,
                    colors[0:1],
                    size=point_size * 2.0,
                )

            if args.show_axes and f"{prefix}_reference_axes_world" in payload.files and f"{prefix}_center_world_m" in payload.files:
                axes_world = np.asarray(payload[f"{prefix}_reference_axes_world"], dtype=np.float32)
                center_world = np.asarray(payload[f"{prefix}_center_world_m"], dtype=np.float32)
                local_axes_pts, local_axes_colors = axis_points(float(args.axis_length))
                world_axes_pts = transform_local_to_world(local_axes_pts, axes_world, center_world) + shift.reshape(1, 3)
                visualize_pointcloud(
                    vis,
                    f"{stage_name}/{prefix}/axes",
                    world_axes_pts,
                    local_axes_colors,
                    size=point_size * 1.5,
                )

    if len(stages) > 1:
        print(f"Stage offsets on +X: raw=0, prefiltered={stage_offset:.3f}, box={stage_offset * 2.0:.3f}")


def visualize_crop_rgbd_payload(payload, args, vis, visualize_pointcloud) -> None:
    entries = filter_label_entries(label_entries(payload), args.label)
    point_size = float(args.point_size)
    frame_suffix = "world_m" if args.frame == "world" else "camera_m"

    if args.mode not in {"dense", "labels", "all"}:
        raise ValueError("For crop RGBD NPZ, use --mode dense, --mode labels, or --mode all.")

    if args.mode in {"dense", "all"}:
        dense_key = f"pointcloud_{frame_suffix}"
        if dense_key not in payload.files:
            raise KeyError(f"Missing key in NPZ: {dense_key}")
        dense_points = np.asarray(payload[dense_key], dtype=np.float32)
        dense_colors = np.asarray(payload["pointcloud_colors_rgb_uint8"], dtype=np.uint8)
        visualize_pointcloud(
            vis,
            "crop/dense",
            dense_points,
            dense_colors,
            size=point_size,
        )
        if args.show_centers and len(dense_points) > 0:
            dense_center = dense_points.mean(axis=0, keepdims=True).astype(np.float32)
            visualize_pointcloud(
                vis,
                "crop/dense_center",
                dense_center,
                repeat_color(np.array([255, 255, 255], dtype=np.uint8), 1),
                size=point_size * 2.0,
            )

    if args.mode in {"labels", "all"}:
        for display_label, prefix in entries:
            if args.filtered_labels:
                label_points_key = f"{prefix}_pointcloud_knn_filtered_{frame_suffix}"
                label_colors_key = f"{prefix}_pointcloud_knn_filtered_colors_rgb_uint8"
                if label_points_key not in payload.files:
                    label_points_key = f"{prefix}_pointcloud_inlier90_{frame_suffix}"
                    label_colors_key = f"{prefix}_pointcloud_inlier90_colors_rgb_uint8"
                if label_points_key not in payload.files:
                    label_points_key = f"{prefix}_pointcloud_{frame_suffix}"
                    label_colors_key = f"{prefix}_pointcloud_colors_rgb_uint8"
            else:
                label_points_key = f"{prefix}_pointcloud_{frame_suffix}"
                label_colors_key = f"{prefix}_pointcloud_colors_rgb_uint8"
            if label_points_key not in payload.files:
                continue
            points = np.asarray(payload[label_points_key], dtype=np.float32)
            color = LABEL_COLORS.get(label_key(display_label), np.array([255, 255, 0], dtype=np.uint8))
            colors = (
                np.asarray(payload[label_colors_key], dtype=np.uint8)
                if label_colors_key in payload.files
                else repeat_color(color, len(points))
            )
            visualize_pointcloud(
                vis,
                f"labels/{prefix}/points",
                points,
                colors,
                size=point_size * 1.15,
            )

            if args.show_centers and len(points) > 0:
                if args.frame == "world" and f"{prefix}_center_world_m" in payload.files:
                    center = np.asarray(payload[f"{prefix}_center_world_m"], dtype=np.float32).reshape(1, 3)
                else:
                    center = points.mean(axis=0, keepdims=True).astype(np.float32)
                visualize_pointcloud(
                    vis,
                    f"labels/{prefix}/center",
                    center,
                    repeat_color(color, 1),
                    size=point_size * 2.2,
                )

            if args.show_axes and args.frame == "world" and f"{prefix}_reference_axes_world" in payload.files:
                if f"{prefix}_center_world_m" in payload.files:
                    center_world = np.asarray(payload[f"{prefix}_center_world_m"], dtype=np.float32)
                elif len(points) > 0:
                    center_world = points.mean(axis=0).astype(np.float32)
                else:
                    continue
                axes_world = np.asarray(payload[f"{prefix}_reference_axes_world"], dtype=np.float32)
                local_axes_pts, local_axes_colors = axis_points(float(args.axis_length))
                world_axes_pts = transform_local_to_world(local_axes_pts, axes_world, center_world)
                visualize_pointcloud(
                    vis,
                    f"labels/{prefix}/axes",
                    world_axes_pts,
                    local_axes_colors,
                    size=point_size * 1.5,
                )


def visualize_segmented_crop_rgbd_payload(payload, args, vis, visualize_pointcloud) -> None:
    entries = filter_label_entries(label_entries(payload), args.label)
    point_size = float(args.point_size)
    frame_suffix = "world_m" if args.frame == "world" else "camera_m"

    if args.mode not in {"dense", "labels", "all"}:
        raise ValueError("For segmented crop RGBD NPZ, use --mode dense, --mode labels, or --mode all.")

    if args.mode in {"dense", "all"}:
        for display_label, prefix in entries:
            dense_key = f"{prefix}_dense_pointcloud_{frame_suffix}"
            color_key = f"{prefix}_dense_pointcloud_colors_rgb_uint8"
            if dense_key not in payload.files:
                continue
            points = np.asarray(payload[dense_key], dtype=np.float32)
            colors = (
                np.asarray(payload[color_key], dtype=np.uint8)
                if color_key in payload.files
                else repeat_color(LABEL_COLORS.get(label_key(display_label), np.array([200, 200, 200], dtype=np.uint8)), len(points))
            )
            visualize_pointcloud(
                vis,
                f"dense/{prefix}/points",
                points,
                colors,
                size=point_size,
            )
            if args.show_centers and len(points) > 0:
                center = points.mean(axis=0, keepdims=True).astype(np.float32)
                visualize_pointcloud(
                    vis,
                    f"dense/{prefix}/center",
                    center,
                    repeat_color(np.array([255, 255, 255], dtype=np.uint8), 1),
                    size=point_size * 2.0,
                )

    if args.mode in {"labels", "all"}:
        for display_label, prefix in entries:
            if args.filtered_labels:
                label_points_key = f"{prefix}_pointcloud_knn_filtered_{frame_suffix}"
                label_colors_key = f"{prefix}_pointcloud_knn_filtered_colors_rgb_uint8"
                if label_points_key not in payload.files:
                    label_points_key = f"{prefix}_pointcloud_{frame_suffix}"
                    label_colors_key = f"{prefix}_pointcloud_colors_rgb_uint8"
            else:
                label_points_key = f"{prefix}_pointcloud_{frame_suffix}"
                label_colors_key = f"{prefix}_pointcloud_colors_rgb_uint8"
            if label_points_key not in payload.files:
                continue
            points = np.asarray(payload[label_points_key], dtype=np.float32)
            color = LABEL_COLORS.get(label_key(display_label), np.array([255, 255, 0], dtype=np.uint8))
            colors = (
                np.asarray(payload[label_colors_key], dtype=np.uint8)
                if label_colors_key in payload.files
                else repeat_color(color, len(points))
            )
            visualize_pointcloud(
                vis,
                f"labels/{prefix}/points",
                points,
                colors,
                size=point_size * 1.15,
            )
            if args.show_centers and len(points) > 0:
                if args.frame == "world" and f"{prefix}_center_world_m" in payload.files:
                    center = np.asarray(payload[f"{prefix}_center_world_m"], dtype=np.float32).reshape(1, 3)
                else:
                    center = points.mean(axis=0, keepdims=True).astype(np.float32)
                visualize_pointcloud(
                    vis,
                    f"labels/{prefix}/center",
                    center,
                    repeat_color(color, 1),
                    size=point_size * 2.2,
                )
            if args.show_axes and args.frame == "world" and f"{prefix}_reference_axes_world" in payload.files:
                if f"{prefix}_center_world_m" in payload.files:
                    center_world = np.asarray(payload[f"{prefix}_center_world_m"], dtype=np.float32)
                elif len(points) > 0:
                    center_world = points.mean(axis=0).astype(np.float32)
                else:
                    continue
                axes_world = np.asarray(payload[f"{prefix}_reference_axes_world"], dtype=np.float32)
                local_axes_pts, local_axes_colors = axis_points(float(args.axis_length))
                world_axes_pts = transform_local_to_world(local_axes_pts, axes_world, center_world)
                visualize_pointcloud(
                    vis,
                    f"labels/{prefix}/axes",
                    world_axes_pts,
                    local_axes_colors,
                    size=point_size * 1.5,
                )


def visualize_full_scene_rgbd_payload(payload, args, vis, visualize_pointcloud) -> None:
    entries = filter_label_entries(label_entries(payload), args.label)
    point_size = float(args.point_size)
    frame_suffix = "world_m" if args.frame == "world" else "camera_m"

    if args.mode not in {"dense", "labels", "all"}:
        raise ValueError("For full-scene RGBD NPZ, use --mode dense, --mode labels, or --mode all.")

    if args.mode in {"dense", "all"}:
        dense_key = f"pointcloud_{frame_suffix}"
        if dense_key not in payload.files:
            raise KeyError(f"Missing key in NPZ: {dense_key}")
        dense_points = np.asarray(payload[dense_key], dtype=np.float32)
        dense_colors = np.asarray(payload["pointcloud_colors_rgb_uint8"], dtype=np.uint8)
        visualize_pointcloud(
            vis,
            "scene/dense",
            dense_points,
            dense_colors,
            size=point_size,
        )
        if args.show_centers and len(dense_points) > 0:
            dense_center = dense_points.mean(axis=0, keepdims=True).astype(np.float32)
            visualize_pointcloud(
                vis,
                "scene/dense_center",
                dense_center,
                repeat_color(np.array([255, 255, 255], dtype=np.uint8), 1),
                size=point_size * 2.0,
            )

    if args.mode in {"labels", "all"}:
        for display_label, prefix in entries:
            if args.filtered_labels:
                label_points_key = f"{prefix}_pointcloud_knn_filtered_{frame_suffix}"
                label_colors_key = f"{prefix}_pointcloud_knn_filtered_colors_rgb_uint8"
                if label_points_key not in payload.files:
                    label_points_key = f"{prefix}_pointcloud_{frame_suffix}"
                    label_colors_key = f"{prefix}_pointcloud_colors_rgb_uint8"
            else:
                label_points_key = f"{prefix}_pointcloud_{frame_suffix}"
                label_colors_key = f"{prefix}_pointcloud_colors_rgb_uint8"
            if label_points_key not in payload.files:
                continue
            points = np.asarray(payload[label_points_key], dtype=np.float32)
            color = LABEL_COLORS.get(label_key(display_label), np.array([255, 255, 0], dtype=np.uint8))
            colors = (
                np.asarray(payload[label_colors_key], dtype=np.uint8)
                if label_colors_key in payload.files
                else repeat_color(color, len(points))
            )
            visualize_pointcloud(
                vis,
                f"labels/{prefix}/points",
                points,
                colors,
                size=point_size * 1.15,
            )
            if args.show_centers and len(points) > 0:
                if args.frame == "world" and f"{prefix}_center_world_m" in payload.files:
                    center = np.asarray(payload[f"{prefix}_center_world_m"], dtype=np.float32).reshape(1, 3)
                else:
                    center = points.mean(axis=0, keepdims=True).astype(np.float32)
                visualize_pointcloud(
                    vis,
                    f"labels/{prefix}/center",
                    center,
                    repeat_color(color, 1),
                    size=point_size * 2.2,
                )
            if args.show_axes and args.frame == "world" and f"{prefix}_reference_axes_world" in payload.files:
                if f"{prefix}_center_world_m" in payload.files:
                    center_world = np.asarray(payload[f"{prefix}_center_world_m"], dtype=np.float32)
                elif len(points) > 0:
                    center_world = points.mean(axis=0).astype(np.float32)
                else:
                    continue
                axes_world = np.asarray(payload[f"{prefix}_reference_axes_world"], dtype=np.float32)
                local_axes_pts, local_axes_colors = axis_points(float(args.axis_length))
                world_axes_pts = transform_local_to_world(local_axes_pts, axes_world, center_world)
                visualize_pointcloud(
                    vis,
                    f"labels/{prefix}/axes",
                    world_axes_pts,
                    local_axes_colors,
                    size=point_size * 1.5,
                )


def main() -> None:
    args = parse_args()
    from grasp_gen.utils.meshcat_utils import create_visualizer, visualize_pointcloud  # type: ignore

    pointcloud_npz = args.pointcloud_npz.expanduser().resolve()
    if not pointcloud_npz.exists():
        raise FileNotFoundError(pointcloud_npz)

    payload = np.load(pointcloud_npz, allow_pickle=True)
    schema = payload_schema(payload)

    vis = create_visualizer()
    vis.delete()
    print(f"Pointcloud NPZ: {pointcloud_npz}")
    print(f"Schema: {schema}")
    print(f"MeshCat URL: {vis.url()}")

    if schema == "full_scene_rgbd":
        visualize_full_scene_rgbd_payload(payload, args, vis, visualize_pointcloud)
    elif schema == "segmented_crop_rgbd":
        visualize_segmented_crop_rgbd_payload(payload, args, vis, visualize_pointcloud)
    elif schema == "crop_rgbd":
        visualize_crop_rgbd_payload(payload, args, vis, visualize_pointcloud)
    else:
        if args.frame != "world":
            raise ValueError("--frame camera is only supported for crop RGBD NPZ.")
        visualize_final_pointcloud_payload(payload, args, vis, visualize_pointcloud)

    print("Visualization ready.")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
