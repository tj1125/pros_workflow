#!/usr/bin/env python3
"""
Run GraspGen from the single-depth + multi-camera SAM size summary.

Target object:
  - represented as a primitive box from the summary size + yaw

Obstacle objects:
  - represented as primitive boxes from the same summary

Output is compatible with visualize_grasps.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh.transformations as tra  # type: ignore

from test_eye_level_grasp import build_primitive_obstacle, point_colors
from test_graspgen import GRIPPER_CONFIG, filter_collisions, run_grasp_inference


DEFAULT_SUMMARY_JSON = Path(
    "/home/tjchen/workspace/VLM_RL/RL_train/View_Agent/test_tmp/pics/Camera_Room1_test/"
    "four_camera_eye_level_experiment_outputs/Camera_Room1_12/"
    "single_depth_multicam_sam_size_outputs/single_depth_multicam_sam_size_summary.json"
)
DEFAULT_TARGET_LABEL = "doll"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grasp generation from single-depth + multi-camera SAM size summary."
    )
    parser.add_argument("summary_json", nargs="?", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--target-label", default=DEFAULT_TARGET_LABEL)
    parser.add_argument("--point-budget", type=int, default=1800)
    parser.add_argument("--num-grasps", type=int, default=200)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--collision-thresh", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def default_output_paths(summary_json: Path, target_label: str) -> tuple[Path, Path]:
    output_dir = summary_json.parent
    tag = target_label.replace(" ", "_")
    result_npz = output_dir / f"{tag}_single_depth_multicam_sam_size_grasp_result.npz"
    report_json = output_dir / f"{tag}_single_depth_multicam_sam_size_grasp_report.json"
    return result_npz, report_json


def run_grasp_inference_no_filter(
    object_pc: np.ndarray,
    object_colors: np.ndarray | None,
    num_grasps: int,
    topk: int,
):
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


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    summary_json = args.summary_json.expanduser().resolve()
    if not summary_json.exists():
        raise FileNotFoundError(summary_json)

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    labels_data = payload["labels"]
    target_label = str(args.target_label).lower()
    if target_label not in labels_data:
        raise ValueError(f"Target label '{target_label}' not found in summary.")

    object_points = []
    object_colors = None
    obstacle_points = []
    obstacle_colors = []
    obstacle_labels: list[str] = []
    object_geometry = {}
    target_num_points = 0

    for label, info in labels_data.items():
        center_world = np.asarray(info["multicam_center_world_m"], dtype=np.float32)
        size_xyz = np.asarray(info["size_xyz_m"], dtype=np.float32)
        yaw_rad = float(info["yaw_rad"])
        pts = build_primitive_obstacle(
            center_world,
            size_xyz,
            mode="box",
            point_budget=int(args.point_budget),
            polygon_sides=8,
            yaw_rad=yaw_rad,
        )
        object_geometry[label] = {
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
        raise RuntimeError(f"Failed to build target primitive for '{target_label}'.")

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

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = target_label.replace(" ", "_")
        result_npz = output_dir / f"{tag}_single_depth_multicam_sam_size_grasp_result.npz"
        report_json = output_dir / f"{tag}_single_depth_multicam_sam_size_grasp_report.json"
    else:
        result_npz, report_json = default_output_paths(summary_json, target_label)

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
        source_summary=np.array([str(summary_json)]),
        obstacle_labels=np.array(obstacle_labels, dtype=object),
    )
    if obj_colors_c is not None:
        save_data["pc_object_colors"] = obj_colors_c
    save_data["pc_object_raw_colors"] = object_colors

    result_write_start = time.perf_counter()
    np.savez_compressed(str(result_npz), **save_data)
    result_write_time_s = time.perf_counter() - result_write_start

    report = {
        "summary_json": str(summary_json),
        "target_label": target_label,
        "target_mode": "single_depth_multicam_sam_size_box",
        "used_outlier_filter": bool(used_outlier_filter),
        "object_geometry": object_geometry[target_label],
        "obstacle_geometry": {label: object_geometry[label] for label in obstacle_labels},
        "num_object_points_raw": target_num_points,
        "num_scene_points": int(len(scene_pc)),
        "num_total_grasps": int(len(grasps_c)),
        "num_collision_free_grasps": int(int(coll_mask.sum())),
        "collision_threshold": float(args.collision_thresh),
        "result_npz": str(result_npz),
        "timings_s": {
            "grasp_inference": float(grasp_time_s),
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
