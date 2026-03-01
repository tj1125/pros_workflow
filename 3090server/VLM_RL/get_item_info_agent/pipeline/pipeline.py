from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2

from pipeline.adapters.graspgen_adapter import infer_grasps_from_mesh
from pipeline.config import (
    load_scene_config,
    prepare_runtime_imports,
    resolve_input_images,
    validate_required_paths,
    validate_runtime_device,
)
from pipeline.steps.detect_and_triangulate import run_detection_and_triangulation
from pipeline.steps.goal_pose import compute_goal_pose
from pipeline.steps.mesh_align import align_mesh_with_depth, reconstruct_mesh_with_sam3d, scale_mesh_to_target_y
from pipeline.steps.sam_and_depth import build_sam3d_inputs, infer_depth_u8, sam_segment_with_bbox


def run_pipeline(
    scene_config: Path,
    yolo_class_name: str,
    image_a: Path | None,
    image_b: Path | None,
    goal_output: Path,
    debug_save: bool,
) -> dict:
    """Execute the full perception pipeline and write goal_pose.json.

    Returns a compact summary dict with keys: goal_pose_path, center_world, group_ranking.
    """
    cfg, cfg_path = load_scene_config(scene_config)
    validate_required_paths(cfg)
    prepare_runtime_imports(cfg)

    runtime = cfg["runtime"]
    device = validate_runtime_device(cfg)

    image_a_path, image_b_path = resolve_input_images(cfg, image_a, image_b)

    tri = run_detection_and_triangulation(cfg, image_a_path, image_b_path, yolo_class_name)
    bbox_a = tri["bbox_a"]
    image_a_bgr = tri["image_a_bgr"]
    center_world = tri["center_world"]
    target_height = tri["target_height"]

    seg_mask_bool = sam_segment_with_bbox(
        image_a_bgr,
        bbox_a,
        model_type=str(runtime["sam_model_type"]),
        checkpoint=Path(cfg["models"]["sam_seg_checkpoint"]),
        device=device,
    )

    sam3d_image_rgb, sam3d_mask_bool = build_sam3d_inputs(
        image_a_bgr,
        seg_mask_bool,
        bbox_a,
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
    )

    scaled_mesh = scale_mesh_to_target_y(mesh, target_height)
    aligned_mesh, best_angle_y, align_metrics, align_mask_u8 = align_mesh_with_depth(
        scaled_mesh,
        depth_u8,
        cfg["alignment"],
        device=device,
    )

    grasps, confidences = infer_grasps_from_mesh(
        aligned_mesh,
        Path(cfg["models"]["gripper_config"]),
        grasp_threshold=float(runtime["grasp_threshold"]),
        num_grasps=int(runtime["num_grasps"]),
        topk_num_grasps=int(runtime["topk_num_grasps"]),
        num_sample_points=int(runtime["num_sample_points"]),
    )

    goal_data = compute_goal_pose(center_world, grasps, confidences, cfg["map"])

    goal_output = goal_output.expanduser().resolve()
    goal_output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "center_world": center_world.tolist(),
        "group_ranking": goal_data["group_ranking"],
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
        try:
            aligned_mesh.export(str(debug_dir / "aligned_mesh.obj"))
        except Exception:
            pass

    return {
        "center_world": payload["center_world"],
        "group_ranking": payload["group_ranking"],
        "goal_pose_path": str(goal_output),
    }
