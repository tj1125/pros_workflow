#!/usr/bin/env python3
"""
Visualize before/after eye-level comparison point clouds from the unified pipeline.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


AGENT_ROOT = Path("/home/tjchen/workspace/VLM_RL/3090server/VLM_RL/get_item_info_agent")
GRASPGEN_VENDOR = AGENT_ROOT / "vendor/graspgen_runtime"

for p in [str(GRASPGEN_VENDOR), str(GRASPGEN_VENDOR / "pointnet2_ops")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from grasp_gen.utils.meshcat_utils import create_visualizer, make_frame, visualize_pointcloud  # type: ignore


DEFAULT_COMPARE_NPZ = Path(
    "/home/tjchen/workspace/VLM_RL/RL_train/View_Agent/test_tmp/pics/Camera_Room1_test/"
    "unified_single_depth_multicam_pipeline_outputs/unified_single_depth_multicam_eye_level_compare.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize before/after eye-level comparison NPZ in MeshCat.")
    parser.add_argument("compare_npz", nargs="?", type=Path, default=DEFAULT_COMPARE_NPZ)
    parser.add_argument("--point-size", type=float, default=0.01)
    parser.add_argument("--show-combined", action="store_true")
    parser.add_argument("--before-filtered-only", action="store_true")
    parser.add_argument("--after-only", action="store_true")
    parser.add_argument("--after-camera-frame-only", action="store_true")
    parser.add_argument("--center-before-filtered", action="store_true")
    parser.add_argument("--center-after", action="store_true")
    parser.add_argument("--show-centers", action="store_true")
    parser.add_argument("--label", default="", help="Show only one label, e.g. doll/apple/wine.")
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


def main() -> None:
    args = parse_args()
    compare_npz = args.compare_npz.expanduser().resolve()
    if not compare_npz.exists():
        raise FileNotFoundError(compare_npz)

    payload = np.load(compare_npz, allow_pickle=True)
    vis = create_visualizer()
    vis.delete()
    print(f"Comparison NPZ: {compare_npz}")
    print(f"MeshCat URL: {vis.url()}")

    point_size = float(args.point_size)
    labels = [str(x) for x in payload["labels"].tolist()]
    selected_labels = labels
    if args.label:
        selected = label_key(args.label)
        selected_labels = [label for label in labels if label_key(label) == selected]
        if not selected_labels:
            raise KeyError(f"Label not found: {args.label}")

    def collect_points(prefix: str) -> tuple[np.ndarray, np.ndarray]:
        pts_list = []
        color_list = []
        for label in selected_labels:
            key = label_key(label)
            pts_list.append(np.asarray(payload[f"{key}_{prefix}_points"], dtype=np.float32))
            color_list.append(np.asarray(payload[f"{key}_{prefix}_colors"], dtype=np.uint8))
        return np.concatenate(pts_list, axis=0), np.concatenate(color_list, axis=0)

    after_points, after_colors = collect_points("after")
    before_filtered_points, before_filtered_colors = collect_points("before_filtered")
    before_filtered_shift = np.zeros(3, dtype=np.float32)
    if args.center_before_filtered:
        before_filtered_shift = -before_filtered_points.mean(axis=0)
        before_filtered_points = before_filtered_points + before_filtered_shift.reshape(1, 3)
    after_shift = np.zeros(3, dtype=np.float32)
    if args.center_after:
        after_shift = -after_points.mean(axis=0)
        after_points = after_points + after_shift.reshape(1, 3)

    if args.show_combined:
        visualize_pointcloud(
            vis,
            "comparison/all",
            np.asarray(payload["comparison_points"], dtype=np.float32),
            np.asarray(payload["comparison_colors"], dtype=np.uint8),
            size=point_size,
        )
    elif args.before_filtered_only:
        visualize_pointcloud(
            vis,
            "comparison/before_filtered",
            before_filtered_points,
            before_filtered_colors,
            size=point_size,
        )
    elif args.after_camera_frame_only:
        pts_list = []
        color_list = []
        for label in selected_labels:
            key = label_key(label)
            pts_list.append(np.asarray(payload[f"{key}_after_camera_frame_centered_points"], dtype=np.float32))
            color_list.append(np.asarray(payload[f"{key}_after_colors"], dtype=np.uint8))
        camera_points = np.concatenate(pts_list, axis=0)
        camera_colors = np.concatenate(color_list, axis=0)
        visualize_pointcloud(
            vis,
            "comparison/after_camera_frame",
            camera_points,
            camera_colors,
            size=point_size,
        )
    elif args.after_only:
        visualize_pointcloud(
            vis,
            "comparison/after",
            after_points,
            after_colors,
            size=point_size,
        )
    else:
        before_points, before_colors = collect_points("before")
        visualize_pointcloud(
            vis,
            "comparison/before",
            before_points,
            before_colors,
            size=point_size,
        )
        visualize_pointcloud(
            vis,
            "comparison/after",
            after_points,
            after_colors,
            size=point_size,
        )

    if args.show_centers:
        for label in selected_labels:
            key = label_key(label)
            after_center = np.asarray(payload[f"{key}_after_center_world_m"], dtype=np.float32).reshape(1, 3)
            if args.center_after:
                after_center = after_center + after_shift.reshape(1, 3)
            after_color = np.asarray(payload[f"{key}_after_colors"], dtype=np.uint8)[0:1]
            if args.before_filtered_only:
                before_filtered_center = np.asarray(payload[f"{key}_before_center_world_m"], dtype=np.float32).reshape(1, 3)
                if args.center_before_filtered:
                    before_filtered_center = before_filtered_center + before_filtered_shift.reshape(1, 3)
                before_filtered_color = np.asarray(payload[f"{key}_before_filtered_colors"], dtype=np.uint8)[0:1]
                visualize_pointcloud(
                    vis,
                    f"centers/{key}/before_filtered",
                    before_filtered_center,
                    before_filtered_color,
                    size=point_size * 2.0,
                )
            elif args.after_camera_frame_only:
                after_color = np.asarray(payload[f"{key}_after_colors"], dtype=np.uint8)[0:1]
                visualize_pointcloud(
                    vis,
                    f"centers/{key}/after_camera_frame",
                    np.zeros((1, 3), dtype=np.float32),
                    after_color,
                    size=point_size * 2.5,
                )
            elif not args.after_only:
                before_center = np.asarray(payload[f"{key}_before_center_world_m"], dtype=np.float32).reshape(1, 3)
                before_color = np.asarray(payload[f"{key}_before_colors"], dtype=np.uint8)[0:1]
                visualize_pointcloud(vis, f"centers/{key}/before", before_center, before_color, size=point_size * 2.0)
            if not args.before_filtered_only and not args.after_camera_frame_only:
                visualize_pointcloud(vis, f"centers/{key}/after", after_center, after_color, size=point_size * 2.0)

    if args.show_axes:
        if args.after_camera_frame_only:
            make_frame(vis, "comparison/axes_frame", h=float(args.axis_length), radius=point_size * 0.5)
        else:
            pts_axis, colors_axis = axis_points(float(args.axis_length))
            visualize_pointcloud(vis, "comparison/axes", pts_axis, colors_axis, size=point_size * 1.5)

    print("Visualization ready.")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
