from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from get_item_info_agent.pipeline.adapters.graspgen_adapter import infer_grasps_from_mesh_with_collision
from get_item_info_agent.pipeline.config import (
    load_scene_config,
    prepare_runtime_imports,
    resolve_input_images,
    validate_required_paths,
    validate_runtime_device,
)
from get_item_info_agent.pipeline.constants import AGENT_ROOT
from get_item_info_agent.pipeline.steps.detect_and_triangulate import (
    run_detection_and_triangulation,
    run_detection_and_triangulation_multi_view,
)
from get_item_info_agent.pipeline.steps.goal_pose import compute_goal_pose
from get_item_info_agent.pipeline.steps.mesh_align import align_mesh_with_depth, reconstruct_mesh_with_sam3d
from get_item_info_agent.pipeline.steps.obstacles import build_cylindrical_obstacle_scene
from get_item_info_agent.pipeline.steps.sam_and_depth import build_sam3d_inputs, infer_depth_u8, sam_segment_with_bbox


def _save_visualization_npz(
    center_world: np.ndarray,
    debug_npz: dict[str, np.ndarray],
    primary_camera_id: str,
    target_label: str,
) -> Path:
    output_dir = AGENT_ROOT / "data" / "debug_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latest_grasp_visualization.npz"
    save_data = dict(debug_npz)
    save_data["center_world"] = np.asarray(center_world, dtype=float)
    save_data["center_world_coordinate_frame"] = np.array("unity_world")
    save_data["primary_camera_id"] = np.array(primary_camera_id)
    save_data["target_label"] = np.array(target_label)
    np.savez(str(output_path), **save_data)
    return output_path


def run_pipeline(
    scene_config: Path,
    yolo_class_name: str,
    image_a: Path | None,
    image_b: Path | None,
    goal_output: Path,
    debug_save: bool,
    camera_a_id: str | None = None,
    camera_b_id: str | None = None,
    image_paths_by_camera: dict[str, Path] | None = None,
    primary_camera_id: str | None = None,
) -> dict:
    """Execute the full perception pipeline and write goal_pose.json."""
    cfg, _ = load_scene_config(scene_config)
    validate_required_paths(cfg)
    prepare_runtime_imports(cfg)

    runtime = cfg["runtime"]
    device = validate_runtime_device(cfg)
    target_label = yolo_class_name.lower()

    if image_paths_by_camera:
        tri = run_detection_and_triangulation_multi_view(
            cfg,
            image_paths_by_camera=image_paths_by_camera,
            yolo_class_name=target_label,
            primary_camera_id=primary_camera_id,
        )
        primary_bbox = tri["primary_bbox"]
        primary_image_bgr = tri["primary_image_bgr"]
        center_world = tri["center_world"]
        target_height = tri["target_height"]
        object_reports = tri["objects"]
        resolved_primary_camera = str(tri["primary_camera_id"])
    else:
        image_a_path, image_b_path = resolve_input_images(cfg, image_a, image_b)
        tri = run_detection_and_triangulation(
            cfg,
            image_a_path,
            image_b_path,
            target_label,
            camera_a_id=camera_a_id,
            camera_b_id=camera_b_id,
        )
        primary_bbox = tri["bbox_a"]
        primary_image_bgr = tri["image_a_bgr"]
        center_world = tri["center_world"]
        target_height = tri["target_height"]
        object_reports = []
        resolved_primary_camera = str(tri["camera_a_id"])

    seg_mask_bool = sam_segment_with_bbox(
        primary_image_bgr,
        primary_bbox,
        model_type=str(runtime["sam_model_type"]),
        checkpoint=Path(cfg["models"]["sam_seg_checkpoint"]),
        device=device,
    )

    sam3d_image_rgb, sam3d_mask_bool = build_sam3d_inputs(
        primary_image_bgr,
        seg_mask_bool,
        primary_bbox,
        base_ratio=float(runtime["base_ratio"]),
    )

    depth_u8 = infer_depth_u8(
        sam3d_image_rgb,
        sam3d_mask_bool,
        depth_model_path=Path(cfg["models"]["depthanything_weights"]),
        device=device,
        input_size=int(runtime["depth_input_size"]),
    )

    mesh = reconstruct_mesh_with_sam3d(
        Path(cfg["models"]["sam3d_config"]),
        sam3d_image_rgb,
        sam3d_mask_bool,
        sam_seed=int(runtime["sam_seed"]),
        device=device,
    )

    aligned_mesh, best_angle_y, align_metrics, align_mask_u8 = align_mesh_with_depth(
        mesh,
        depth_u8,
        cfg["alignment"],
        device=device,
        target_y=target_height,
    )

    obstacle_scene = build_cylindrical_obstacle_scene(
        object_reports,
        target_label=target_label,
        target_center_world=center_world,
        samples_per_object=int(runtime.get("obstacle_surface_samples", 144)),
    )

    grasps, confidences, grasp_stats, grasp_debug_npz = infer_grasps_from_mesh_with_collision(
        aligned_mesh,
        Path(cfg["models"]["gripper_config"]),
        grasp_threshold=float(runtime["grasp_threshold"]),
        num_grasps=int(runtime["num_grasps"]),
        topk_num_grasps=int(runtime["topk_num_grasps"]),
        num_sample_points=int(runtime["num_sample_points"]),
        scene_pc=obstacle_scene,
        collision_threshold=float(runtime.get("collision_threshold", 0.02)),
        max_scene_points=int(runtime.get("max_collision_scene_points", 8192)),
        num_collision_samples=int(runtime.get("num_collision_samples", 2000)),
    )
    visualization_npz_path = _save_visualization_npz(
        center_world=center_world,
        debug_npz=grasp_debug_npz,
        primary_camera_id=resolved_primary_camera,
        target_label=target_label,
    )

    goal_data = compute_goal_pose(center_world, grasps, confidences, cfg["map"])

    goal_output = goal_output.expanduser().resolve()
    goal_output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "center_world": center_world.tolist(),
        "center_world_coordinate_frame": "unity_world",
        "grasp_relative_coordinate_frame": "unity_local",
        "group_ranking": goal_data["group_ranking"],
        "goal_pose_path": str(goal_output),
        "primary_camera_id": resolved_primary_camera,
        "target_object": tri.get("target_object"),
        "objects": object_reports,
        "num_matched_objects": int(tri.get("num_matched_objects", len(object_reports))),
        "num_obstacle_objects": int(sum(1 for obj in object_reports if obj.get("label") != target_label)),
        "num_obstacle_scene_points": int(len(obstacle_scene)),
        "mesh_alignment_angle_y": float(best_angle_y),
        "mesh_alignment_metrics": align_metrics,
        "goal_pose_used_map_fallback": bool(goal_data.get("used_map_fallback", False)),
        "grasp_visualization_npz_path": str(visualization_npz_path),
        **grasp_stats,
    }
    goal_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if debug_save:
        debug_dir = goal_output.parent / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / "seg_mask.png"), (seg_mask_bool.astype("uint8") * 255))
        cv2.imwrite(str(debug_dir / "sam3d_input_rgb.png"), cv2.cvtColor(sam3d_image_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(debug_dir / "sam3d_input_mask.png"), (sam3d_mask_bool.astype("uint8") * 255))
        cv2.imwrite(str(debug_dir / "depth_u8.png"), depth_u8)
        cv2.imwrite(str(debug_dir / "align_mask_u8.png"), align_mask_u8)
        (debug_dir / "objects.json").write_text(json.dumps(object_reports, indent=2), encoding="utf-8")
        try:
            aligned_mesh.export(str(debug_dir / "aligned_mesh.obj"))
        except Exception:
            pass

    return payload
