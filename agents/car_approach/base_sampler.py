"""Slim base sampler with point-cloud voxels in PyBullet and no motion planner."""

from __future__ import annotations

import argparse
import ast
import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import sample_logic
from .debug_log import debug_stage
from .src.geometry import coordinate_transforms as coord
from .src.geometry import map_free_space
from .src.io.depth_pointcloud import DepthPointcloudConfig, DepthPointcloudResult, capture_depth_pointcloud, depth_png_bytes_to_pointcloud


APPROACH_AGENT_DIR = Path(__file__).resolve().parent
VLM_RL_ROOT_DIR = APPROACH_AGENT_DIR.parents[1]
DEFAULT_CONFIG_PATH = Path("configs/car_approach.yaml")


@dataclass(frozen=True)
class BaseSamplerRunConfig:
    config_path: Path = DEFAULT_CONFIG_PATH
    grasp_json_path: Path | None = None
    grasp_result_payload: dict[str, object] | None = None
    pointcloud_xyz: object | None = None
    depth_png_bytes: bytes | None = None
    target_mask: object | None = None
    target_bbox_xyxy: object | None = None


@dataclass(frozen=True)
class RobotSceneConfig:
    urdf_path: Path
    base_height_m: float
    base_orientation_euler_deg: tuple[float, float, float]
    joint_reset_deg: tuple[float, ...]
    joint_bounds_deg: tuple[tuple[float, float], ...]
    ee_link_index: int
    controllable_joints: int
    position_tolerance_m: float
    orientation_tolerance_deg: float
    ik_max_iterations: int
    ik_residual_threshold: float
    voxel_collision_threshold_m: float
    voxel_size_m: float


def prepare_scene_pointcloud_and_voxels(
    run_config: BaseSamplerRunConfig,
    config: dict[str, object],
    config_path: Path,
    scene_config: RobotSceneConfig,
) -> tuple[np.ndarray, str, dict[str, object], np.ndarray]:
    debug_stage("base_sampler", "階段 1：準備點雲輸入，可能來自 payload depth、live depth 或 pointcloud payload")
    pointcloud, pointcloud_source, pointcloud_info = _pointcloud_from_run_config(
        run_config,
        config,
        config_path,
    )
    debug_stage(
        "base_sampler",
        "階段 1 完成：點雲已準備並轉成 PyBullet frame",
        source=pointcloud_source,
        points=len(pointcloud),
        input_frame=pointcloud_info.get("pointcloud_input_frame"),
        target_mask_applied=pointcloud_info.get("target_mask_applied"),
        target_mask_source=pointcloud_info.get("target_mask_source"),
        target_bbox_applied=pointcloud_info.get("target_bbox_applied"),
        target_bbox_source=pointcloud_info.get("target_bbox_source"),
        target_exclusion_applied=pointcloud_info.get("target_exclusion_applied"),
        excluded_mask_pixel_count=pointcloud_info.get("excluded_mask_pixel_count"),
    )

    debug_stage("base_sampler", "階段 2：根據 voxel_edge_length_m 將點雲 voxelize")
    voxels = coord.voxelize_points(
        pointcloud,
        voxel_size_m=scene_config.voxel_size_m,
        max_voxels=_positive_int_or_none(config.get("max_voxels")),
    )
    debug_stage(
        "base_sampler",
        "階段 2 完成：voxel 已準備放入 PyBullet",
        voxel_count=len(voxels),
        voxel_edge_m=scene_config.voxel_size_m,
    )
    return pointcloud, pointcloud_source, pointcloud_info, voxels


def run_base_sampling(run_config: BaseSamplerRunConfig | None = None) -> dict[str, object]:
    run_config = run_config or BaseSamplerRunConfig()
    started_at = time.time()
    resolved_config_path = _resolve_approach_path(run_config.config_path)
    config = _load_sampler_config(resolved_config_path)
    scene_config = _robot_scene_config(config, resolved_config_path)
    debug_stage("base_sampler", "讀取 car_approach 設定完成", config=str(resolved_config_path))

    try:
        pointcloud, pointcloud_source, pointcloud_info, voxels = prepare_scene_pointcloud_and_voxels(
            run_config,
            config,
            resolved_config_path,
            scene_config,
        )
    except Exception as exc:
        debug_stage("base_sampler", "階段 1/2 失敗：點雲或 voxel 準備失敗", error=str(exc))
        return _failure("pointcloud_failed", f"Point cloud/voxel preparation failed: {exc}", started_at)

    debug_stage("base_sampler", "階段 3：讀取 grasp poses")
    grasp_candidates, grasp_source = _load_grasps(run_config, config)
    if not grasp_candidates:
        debug_stage("base_sampler", "階段 3 失敗：沒有 grasp pose 可用", source=grasp_source)
        return _failure("no_grasp", "No grasp pose is available.", started_at)
    debug_stage("base_sampler", "階段 3 完成：grasp poses 讀取完成", source=grasp_source, grasp_count=len(grasp_candidates))

    debug_stage("base_sampler", "階段 4：grasp pose position/rotation 轉成 PyBullet frame")
    grasp_candidates = _transform_grasp_candidates_to_pybullet(
        grasp_candidates[: max(1, int(config["max_grasps"]))],
        config,
    )
    debug_stage("base_sampler", "階段 4 完成：grasp poses 已在 PyBullet frame", grasp_count=len(grasp_candidates))

    debug_stage("base_sampler", "階段 5：根據 grasp approach axis sample base pose")
    samples = sample_logic.sample_base_points(
        grasp_candidates,
        min_backoff_m=float(config["min_backoff_m"]),
        max_backoff_m=float(config["max_backoff_m"]),
        step_m=float(config["backoff_step_m"]),
        yaw_span_deg=float(config["yaw_span_deg"]),
        yaw_step_deg=float(config["yaw_step_deg"]),
        max_samples=int(config["max_samples"]),
    )
    debug_stage("base_sampler", "階段 5 完成：base pose sample 產生", sample_count=len(samples))
    if not samples:
        debug_stage("base_sampler", "階段 5 失敗：沒有產生任何 base pose sample")
        return _failure("no_sample", "No base sample was generated.", started_at)

    debug_stage("base_sampler", "階段 6：PyBullet base pose 轉 ROS map pose，並做地圖可行性檢查")
    samples, ros_map_stats = _annotate_and_filter_samples_for_ros_map(
        samples,
        config,
        resolved_config_path,
        pointcloud_info,
    )
    debug_stage(
        "base_sampler",
        "階段 6 完成：ROS map 檢查結束",
        map_status=ros_map_stats.get("map_status"),
        feasible=ros_map_stats.get("feasible_sample_count"),
        rejected=ros_map_stats.get("rejected_sample_count"),
    )
    if not samples:
        debug_stage(
            "base_sampler",
            "階段 6 失敗：沒有 sample 通過 ROS map 檢查",
            phase=ros_map_stats.get("failure_phase"),
            message=ros_map_stats.get("message"),
        )
        return _failure(
            str(ros_map_stats.get("failure_phase", "no_ros_map_feasible_sample")),
            str(ros_map_stats.get("message", "No sampled base pose passed ROS map feasibility checks.")),
            started_at,
            extra={"ros_map_stats": ros_map_stats},
        )

    debug_stage("base_sampler", "階段 7：在 PyBullet 檢查碰撞、IK 可達、夾爪角度 tolerance")
    evaluation = _evaluate_samples_in_pybullet(
        samples,
        scene_config=scene_config,
        voxel_centers=voxels,
        voxel_size_m=scene_config.voxel_size_m,
    )
    evaluated_samples = evaluation["samples"]
    selected = evaluation["selected_solution"]
    closest = evaluation.get("closest_solution")
    evaluation_summary = _evaluation_summary(evaluated_samples, scene_config=scene_config)
    debug_stage(
        "base_sampler",
        "階段 7 完成：PyBullet 評估結束",
        evaluated=len(evaluated_samples),
        selected=selected is not None,
        closest_solution=closest is not None,
        **evaluation_summary,
    )

    if selected is None:
        phase = "no_feasible_sample"
        if closest is None:
            message = _no_collision_free_message(evaluation_summary)
        else:
            message = _no_feasible_with_closest_message(evaluation_summary)
        debug_stage(
            "base_sampler",
            "階段 7 失敗：沒有 IK feasible，回傳不碰撞的 closest_solution 供 debug",
            closest_available=closest is not None,
            **evaluation_summary,
        )
    else:
        debug_stage(
            "base_sampler",
            "階段 7 成功：找到 selected base pose",
            sample_index=selected.get("sample_index"),
            pos_err=selected.get("ee_position_error_m"),
            ori_err_deg=selected.get("ee_orientation_error_deg"),
        )
        phase = "sampled"
        message = "Base sample selected by PyBullet IK/collision checks."
    success = selected is not None
    return {
        "success": bool(success),
        "status_code": "BASE_SAMPLE_SUCCESS" if success else "BASE_SAMPLE_FAIL",
        "phase": phase,
        "message": message,
        "next_agent": None,
        "grasp_source": grasp_source,
        "grasp_count": len(grasp_candidates),
        "sample_count": len(samples),
        "evaluated_sample_count": len(evaluated_samples),
        "pointcloud_source": pointcloud_source,
        "pointcloud_input_frame": pointcloud_info.get("pointcloud_input_frame"),
        "pointcloud_frame": pointcloud_info.get("pointcloud_frame"),
        "pointcloud_point_count": int(len(pointcloud)),
        "voxel_count": int(len(voxels)),
        "depth_shape": pointcloud_info.get("depth_shape"),
        "depth_format": pointcloud_info.get("depth_format"),
        "camera_name": pointcloud_info.get("camera_name"),
        "depth_camera_x_mirrored": pointcloud_info.get("depth_camera_x_mirrored"),
        "capture_amcl_pose": pointcloud_info.get("capture_amcl_pose"),
        "target_bbox_flip_y_for_depth_alignment": pointcloud_info.get("target_bbox_flip_y_for_depth_alignment"),
        "target_mask_applied": pointcloud_info.get("target_mask_applied"),
        "target_mask_source": pointcloud_info.get("target_mask_source"),
        "target_bbox_applied": pointcloud_info.get("target_bbox_applied"),
        "target_bbox_source": pointcloud_info.get("target_bbox_source"),
        "target_bbox_xyxy": pointcloud_info.get("target_bbox_xyxy"),
        "excluded_bbox_xyxy": pointcloud_info.get("excluded_bbox_xyxy"),
        "excluded_bbox_pixel_count": pointcloud_info.get("excluded_bbox_pixel_count"),
        "target_exclusion_applied": pointcloud_info.get("target_exclusion_applied"),
        "target_exclusion_source": pointcloud_info.get("target_exclusion_source"),
        "excluded_mask_shape": pointcloud_info.get("excluded_mask_shape"),
        "excluded_mask_pixel_count": pointcloud_info.get("excluded_mask_pixel_count"),
        "grasp_target_gripper_z_offset_m": float(config.get("grasp_target_gripper_z_offset_m", 0.0)),
        "sampling_summary": {
            **evaluation_summary,
            "target_mask_applied": bool(pointcloud_info.get("target_mask_applied", False)),
            "target_mask_source": pointcloud_info.get("target_mask_source"),
            "target_bbox_applied": bool(pointcloud_info.get("target_bbox_applied", False)),
            "target_bbox_source": pointcloud_info.get("target_bbox_source"),
            "target_bbox_xyxy": pointcloud_info.get("target_bbox_xyxy"),
            "target_bbox_flip_y_for_depth_alignment": pointcloud_info.get("target_bbox_flip_y_for_depth_alignment"),
            "excluded_bbox_xyxy": pointcloud_info.get("excluded_bbox_xyxy"),
            "excluded_bbox_pixel_count": pointcloud_info.get("excluded_bbox_pixel_count"),
            "target_exclusion_applied": bool(pointcloud_info.get("target_exclusion_applied", False)),
            "target_exclusion_source": pointcloud_info.get("target_exclusion_source"),
            "excluded_mask_shape": pointcloud_info.get("excluded_mask_shape"),
            "excluded_mask_pixel_count": pointcloud_info.get("excluded_mask_pixel_count"),
            "voxel_count": int(len(voxels)),
            "pointcloud_point_count": int(len(pointcloud)),
            "grasp_target_gripper_z_offset_m": float(config.get("grasp_target_gripper_z_offset_m", 0.0)),
            "position_tolerance_m": float(scene_config.position_tolerance_m),
            "orientation_tolerance_deg": float(scene_config.orientation_tolerance_deg),
        },
        "ros_map_stats": ros_map_stats,
        "voxel_size_m": scene_config.voxel_size_m,
        "selected_solution": selected or {},
        "closest_solution": closest or {},
        "closest_solution_available": closest is not None,
        "rank_closest_solutions": evaluation.get("rank_closest_solutions", []),
        "selected_solution_source": "rank_order_feasible" if selected is not None else "",
        "fallback_to_closest_solution": False,
        "candidate_solutions": evaluated_samples,
        "elapsed_sec": time.time() - started_at,
    }


def _evaluate_samples_in_pybullet(
    samples: list[dict[str, object]],
    *,
    scene_config: RobotSceneConfig,
    voxel_centers: np.ndarray,
    voxel_size_m: float,
) -> dict[str, object]:
    import pybullet as p
    import pybullet_data

    debug_stage("base_sampler", "PyBullet 評估：啟動 DIRECT scene", samples=len(samples), voxel_count=len(voxel_centers))
    client_id = p.connect(p.DIRECT)
    if client_id < 0:
        raise RuntimeError("PyBullet DIRECT connection failed.")
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.8)
        p.loadURDF("plane.urdf")
        obstacle_ids = _add_voxel_obstacles(p, voxel_centers, voxel_size_m)
        debug_stage("base_sampler", "PyBullet 評估：voxel obstacle 已放進 scene", obstacle_count=len(obstacle_ids))
        for search_path in _urdf_search_paths(scene_config.urdf_path):
            p.setAdditionalSearchPath(str(search_path))
        robot_id = p.loadURDF(
            str(scene_config.urdf_path),
            useFixedBase=True,
            basePosition=[0.0, 0.0, scene_config.base_height_m],
            baseOrientation=p.getQuaternionFromEuler(coord.deg_sequence_to_rad(scene_config.base_orientation_euler_deg)),
        )
        joint_ids = _controllable_joint_ids(p, robot_id, scene_config.controllable_joints)
        debug_stage("base_sampler", "PyBullet 評估：URDF 已載入，準備逐一檢查 sample", joint_count=len(joint_ids))
        if len(joint_ids) != scene_config.controllable_joints:
            raise RuntimeError(
                f"Expected {scene_config.controllable_joints} controllable joints, found {len(joint_ids)}."
            )

        evaluated: list[dict[str, object]] = []
        selected: dict[str, object] | None = None
        closest: dict[str, object] | None = None
        rank_closest: dict[int, dict[str, object]] = {}
        for sample in samples:
            record = _evaluate_sample(p, robot_id, joint_ids, obstacle_ids, scene_config, sample)
            evaluated.append(record)
            rank = _optional_int(record.get("grasp_rank"))
            if bool(record.get("collision_free", False)):
                if closest is None or _closest_solution_sort_key(record) < _closest_solution_sort_key(closest):
                    closest = record
                if rank is not None and (
                    rank not in rank_closest
                    or _closest_solution_sort_key(record) < _closest_solution_sort_key(rank_closest[rank])
                ):
                    rank_closest[rank] = record
            if record.get("ik_feasible"):
                selected = record
                debug_stage(
                    "base_sampler",
                    "PyBullet 評估：rank order 找到第一個 feasible，停止後續 grasp 測試",
                    grasp_rank=record.get("grasp_rank"),
                    grasp_index=record.get("grasp_index"),
                    sample_index=record.get("sample_index"),
                    position_error=record.get("ee_position_error_m"),
                    orientation_error=record.get("ee_orientation_error_deg"),
                )
                break
        return {
            "samples": evaluated,
            "selected_solution": selected,
            "closest_solution": closest,
            "rank_closest_solutions": [rank_closest[key] for key in sorted(rank_closest)],
        }
    finally:
        p.disconnect(client_id)


def _evaluation_summary(
    evaluated_samples: list[dict[str, object]],
    *,
    scene_config: RobotSceneConfig,
) -> dict[str, object]:
    total = len(evaluated_samples)
    collision_free = [record for record in evaluated_samples if bool(record.get("collision_free", False))]
    position_ok = [
        record
        for record in evaluated_samples
        if _finite_float_or_none(record.get("ee_position_error_m")) is not None
        and float(record["ee_position_error_m"]) <= scene_config.position_tolerance_m
    ]
    orientation_ok = [
        record
        for record in evaluated_samples
        if _finite_float_or_none(record.get("ee_orientation_error_deg")) is not None
        and float(record["ee_orientation_error_deg"]) <= scene_config.orientation_tolerance_deg
    ]
    feasible = [record for record in evaluated_samples if bool(record.get("ik_feasible", False))]
    best_position = min(
        (_finite_float_or_none(record.get("ee_position_error_m")) for record in evaluated_samples),
        default=None,
    )
    best_orientation = min(
        (_finite_float_or_none(record.get("ee_orientation_error_deg")) for record in evaluated_samples),
        default=None,
    )
    best_collision_free_position = min(
        (_finite_float_or_none(record.get("ee_position_error_m")) for record in collision_free),
        default=None,
    )
    grasp_rank_summaries = _grasp_rank_evaluation_summaries(evaluated_samples)
    return {
        "evaluated_sample_count": int(total),
        "feasible_count": int(len(feasible)),
        "collision_free_count": int(len(collision_free)),
        "collision_rejected_count": int(total - len(collision_free)),
        "position_ok_count": int(len(position_ok)),
        "orientation_ok_count": int(len(orientation_ok)),
        "best_position_error_m": best_position,
        "best_orientation_error_deg": best_orientation,
        "best_collision_free_position_error_m": best_collision_free_position,
        "evaluated_grasp_ranks": [item["grasp_rank"] for item in grasp_rank_summaries],
        "evaluated_grasp_rank_count": len(grasp_rank_summaries),
        "grasp_rank_summaries": grasp_rank_summaries,
        "dominant_rejection_reason": _dominant_rejection_reason(
            total=total,
            collision_free_count=len(collision_free),
            position_ok_count=len(position_ok),
            orientation_ok_count=len(orientation_ok),
        ),
    }


def _grasp_rank_evaluation_summaries(evaluated_samples: list[dict[str, object]]) -> list[dict[str, object]]:
    by_rank: dict[int, list[dict[str, object]]] = {}
    for record in evaluated_samples:
        rank = _optional_int(record.get("grasp_rank"))
        if rank is None:
            continue
        by_rank.setdefault(rank, []).append(record)

    summaries: list[dict[str, object]] = []
    for rank in sorted(by_rank):
        records = by_rank[rank]
        collision_free = [record for record in records if bool(record.get("collision_free", False))]
        feasible = [record for record in records if bool(record.get("ik_feasible", False))]
        best_position = min((_finite_float_or_none(record.get("ee_position_error_m")) for record in records), default=None)
        best_collision_free_position = min(
            (_finite_float_or_none(record.get("ee_position_error_m")) for record in collision_free),
            default=None,
        )
        summaries.append(
            {
                "grasp_rank": int(rank),
                "evaluated_sample_count": int(len(records)),
                "feasible_count": int(len(feasible)),
                "collision_free_count": int(len(collision_free)),
                "best_position_error_m": best_position,
                "best_collision_free_position_error_m": best_collision_free_position,
            }
        )
    return summaries


def _dominant_rejection_reason(
    *,
    total: int,
    collision_free_count: int,
    position_ok_count: int,
    orientation_ok_count: int,
) -> str:
    if total <= 0:
        return "no_evaluated_samples"
    if collision_free_count <= 0:
        return "all_samples_collide_with_voxels"
    if position_ok_count <= 0:
        return "all_samples_fail_position_tolerance"
    if orientation_ok_count <= 0:
        return "all_samples_fail_orientation_tolerance"
    return "mixed_ik_or_collision_rejection"


def _no_collision_free_message(summary: dict[str, object]) -> str:
    return (
        "No sampled base pose passed PyBullet IK/collision checks; "
        f"all {int(summary.get('evaluated_sample_count', 0) or 0)} evaluated samples collided with voxel obstacles, "
        "so no collision-free closest_solution is available."
    )


def _no_feasible_with_closest_message(summary: dict[str, object]) -> str:
    return (
        "No sampled base pose passed PyBullet IK/collision checks; "
        f"collision_free={int(summary.get('collision_free_count', 0) or 0)}, "
        f"position_ok={int(summary.get('position_ok_count', 0) or 0)}, "
        f"orientation_ok={int(summary.get('orientation_ok_count', 0) or 0)}. "
        "Returning collision-free closest_solution."
    )


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _finite_float_or_none(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _closest_solution_sort_key(solution: dict[str, object]) -> tuple[float, float, int, int]:
    orientation_error = solution.get("ee_orientation_error_deg")
    return (
        float(solution.get("ee_position_error_m", float("inf"))),
        float("inf") if orientation_error is None else float(orientation_error),
        int(solution.get("grasp_rank", 0)),
        int(solution.get("sample_index", 0)),
    )


def _evaluate_sample(
    p: Any,
    robot_id: int,
    joint_ids: list[int],
    obstacle_ids: list[int],
    scene_config: RobotSceneConfig,
    sample: dict[str, object],
) -> dict[str, object]:
    base_xyz = [float(v) for v in sample["pb_base_link_xyz"]]
    base_xyz[2] = scene_config.base_height_m
    base_yaw = float(sample["pb_base_link_yaw_rad"])
    target_xyz = np.asarray(sample["target_xyz"], dtype=np.float64).reshape(3)
    target_rotation = np.asarray(sample["target_rotation_matrix"], dtype=np.float64).reshape(3, 3)
    target_quat = coord.matrix_to_quat_xyzw(target_rotation)
    reset_rad = coord.deg_sequence_to_rad(scene_config.joint_reset_deg)

    p.resetBasePositionAndOrientation(robot_id, base_xyz, p.getQuaternionFromEuler([0.0, 0.0, base_yaw]))
    _set_joint_positions(p, robot_id, joint_ids, reset_rad)
    ik = p.calculateInverseKinematics(
        robot_id,
        scene_config.ee_link_index,
        targetPosition=target_xyz.astype(float).tolist(),
        targetOrientation=target_quat,
        maxNumIterations=scene_config.ik_max_iterations,
        residualThreshold=scene_config.ik_residual_threshold,
    )
    joint_solution = [float(v) for v in ik[: len(joint_ids)]]
    joint_solution = _clip_to_bounds(joint_solution, scene_config.joint_bounds_deg)
    _set_joint_positions(p, robot_id, joint_ids, joint_solution)
    p.performCollisionDetection()

    ee_state = p.getLinkState(robot_id, scene_config.ee_link_index, computeForwardKinematics=True)
    ee_position = np.asarray(ee_state[4], dtype=np.float64)
    ee_quat = [float(v) for v in ee_state[5]]
    error_xyz = target_xyz - ee_position
    position_error = float(np.linalg.norm(error_xyz))
    orientation_error = coord.quat_angle_error_deg(target_quat, ee_quat)
    collision_free = not _robot_collides(p, robot_id, obstacle_ids, scene_config.voxel_collision_threshold_m)
    feasible = (
        position_error <= scene_config.position_tolerance_m
        and orientation_error <= scene_config.orientation_tolerance_deg
        and collision_free
    )

    record = dict(sample)
    record.update(
        {
            "pb_base_link_xyz": base_xyz,
            "pb_base_link_yaw_rad": base_yaw,
            "pb_base_link_yaw_deg": coord.rad_to_deg(base_yaw),
            "ik_feasible": bool(feasible),
            "collision_free": bool(collision_free),
            "ee_position_error_m": position_error,
            "ee_orientation_error_deg": orientation_error,
            "ik_error_xyz_m": error_xyz.astype(float).tolist(),
            "final_ee_position_xyz": ee_position.astype(float).tolist(),
            "final_ee_orientation_xyzw": ee_quat,
            "ik_joint_solution_rad": joint_solution,
            "ik_joint_solution_deg": coord.rad_sequence_to_deg(joint_solution),
        }
    )
    return record


def _add_voxel_obstacles(p: Any, voxel_centers: np.ndarray, voxel_size_m: float) -> list[int]:
    centers = np.asarray(voxel_centers, dtype=np.float64).reshape(-1, 3)
    if len(centers) == 0:
        return []
    half_extents = [float(voxel_size_m) * 0.5] * 3
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.8, 0.2, 0.2, 0.45])
    return [
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=center.astype(float).tolist(),
        )
        for center in centers
    ]


def _load_grasps(
    run_config: BaseSamplerRunConfig,
    config: dict[str, object],
) -> tuple[list[sample_logic.GraspPoseCandidate], str]:
    if run_config.grasp_result_payload is not None:
        candidates, _ = sample_logic.load_grasp_candidates_from_result_payload(
            run_config.grasp_result_payload,
            source_label="grasp_result_payload",
        )
        return candidates, "payload"
    grasp_json_path = run_config.grasp_json_path
    if grasp_json_path is None and config.get("grasp_json_path"):
        grasp_json_path = Path(str(config["grasp_json_path"]))
    candidates, _, resolved_path = sample_logic.load_grasp_candidates_from_result_json(grasp_json_path)
    return candidates, str(resolved_path)


def _pointcloud_from_run_config(
    run_config: BaseSamplerRunConfig,
    config: dict[str, object],
    config_path: Path,
) -> tuple[np.ndarray, str, dict[str, object]]:
    target_mask, target_mask_source = _target_mask_from_run_config(run_config)
    target_bbox_xyxy, target_bbox_source = _target_bbox_from_run_config(run_config)

    if run_config.pointcloud_xyz is not None:
        debug_stage("base_sampler", "點雲來源：使用外部傳入的 pointcloud payload")
        input_frame = str(config.get("pointcloud_frame", "pybullet"))
        points = _finite_pointcloud(run_config.pointcloud_xyz)
        return _transform_pointcloud_to_pybullet(points, input_frame, config), "payload", {
            "pointcloud_input_frame": input_frame,
            "pointcloud_frame": "pybullet",
            "target_mask_applied": False,
            "target_mask_source": target_mask_source,
            "target_bbox_applied": False,
            "target_bbox_source": target_bbox_source,
            "target_bbox_xyxy": _jsonable_bbox_xyxy(target_bbox_xyxy),
            "target_exclusion_applied": False,
            "target_exclusion_source": None,
            "target_mask_message": "target mask/bbox cannot be applied to precomputed pointcloud_xyz",
        }

    if run_config.depth_png_bytes is not None or bool(config.get("capture_depth_image", False)):
        depth_config = _depth_pointcloud_config(
            config,
            config_path,
            exclude_mask=target_mask,
            exclude_bbox_xyxy=target_bbox_xyxy,
        )
        if run_config.depth_png_bytes is not None:
            debug_stage("base_sampler", "點雲來源：使用外部傳入的 depth PNG bytes")
            result = depth_png_bytes_to_pointcloud(
                run_config.depth_png_bytes,
                depth_config,
                source="payload_depth_png",
            )
        else:
            debug_stage("base_sampler", "點雲來源：呼叫 ROS live depth capture")
            result = capture_depth_pointcloud(depth_config)
        info = _depth_pointcloud_info(result)
        info["target_mask_source"] = target_mask_source
        info["target_bbox_source"] = target_bbox_source
        info["target_bbox_xyxy"] = _jsonable_bbox_xyxy(target_bbox_xyxy)
        info["target_exclusion_source"] = _target_exclusion_source(
            target_mask_source if info.get("target_mask_applied") else None,
            target_bbox_source if info.get("target_bbox_applied") else None,
        )
        return _transform_pointcloud_to_pybullet(result.points_xyz, result.pointcloud_frame, config), result.source, info

    debug_stage("base_sampler", "點雲來源：沒有收到點雲或深度圖，使用空點雲")
    return np.empty((0, 3), dtype=np.float32), "none", {
        "pointcloud_input_frame": "none",
        "pointcloud_frame": "pybullet",
        "target_mask_applied": False,
        "target_mask_source": target_mask_source,
        "target_bbox_applied": False,
        "target_bbox_source": target_bbox_source,
        "target_bbox_xyxy": _jsonable_bbox_xyxy(target_bbox_xyxy),
        "target_exclusion_applied": False,
        "target_exclusion_source": None,
    }


_MASK_KEYS = (
    "target_mask",
    "target_object_mask",
    "object_mask",
    "object_segmentation_mask",
    "segmentation_mask",
    "mask",
)
_MASK_BASE64_KEYS = (
    "target_mask_base64",
    "target_mask_png_base64",
    "target_object_mask_base64",
    "target_object_mask_png_base64",
    "object_mask_base64",
    "object_mask_png_base64",
    "segmentation_mask_base64",
    "segmentation_mask_png_base64",
    "mask_base64",
    "mask_png_base64",
)
_BBOX_XYXY_KEYS = (
    "bbox_xyxy",
    "target_bbox_xyxy",
    "object_bbox_xyxy",
    "box_xyxy",
)


def _target_mask_from_run_config(run_config: BaseSamplerRunConfig) -> tuple[object | None, str | None]:
    if run_config.target_mask is not None:
        return run_config.target_mask, "run_config.target_mask"
    payload = run_config.grasp_result_payload
    if not isinstance(payload, dict):
        return None, None
    for source, mapping in _candidate_mask_mappings(payload):
        mask = _mask_from_mapping(mapping)
        if mask is not None:
            return mask, source
    return None, None


def _candidate_mask_mappings(payload: dict[str, object]) -> list[tuple[str, dict[str, object]]]:
    mappings: list[tuple[str, dict[str, object]]] = [("grasp_result_payload", payload)]
    raw_result = payload.get("raw_result")
    if isinstance(raw_result, dict):
        mappings.append(("grasp_result_payload.raw_result", raw_result))
    nested_result = payload.get("result")
    if isinstance(nested_result, dict):
        mappings.append(("grasp_result_payload.result", nested_result))
        nested_raw = nested_result.get("raw_result")
        if isinstance(nested_raw, dict):
            mappings.append(("grasp_result_payload.result.raw_result", nested_raw))
    return mappings


def _mask_from_mapping(mapping: dict[str, object]) -> object | None:
    for key in _MASK_KEYS:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    for key in _MASK_BASE64_KEYS:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return base64.b64decode(value)
    return None


def _target_bbox_from_run_config(run_config: BaseSamplerRunConfig) -> tuple[tuple[float, float, float, float] | None, str | None]:
    if run_config.target_bbox_xyxy is not None:
        bbox = _bbox_xyxy_from_value(run_config.target_bbox_xyxy)
        if bbox is not None:
            return bbox, "run_config.target_bbox_xyxy"
    payload = run_config.grasp_result_payload
    if not isinstance(payload, dict):
        return None, None
    for source, mapping in _candidate_mask_mappings(payload):
        bbox = _bbox_xyxy_from_mapping(mapping)
        if bbox is not None:
            return bbox, source
    return None, None


def _bbox_xyxy_from_mapping(mapping: dict[str, object]) -> tuple[float, float, float, float] | None:
    for key in _BBOX_XYXY_KEYS:
        bbox = _bbox_xyxy_from_value(mapping.get(key))
        if bbox is not None:
            return bbox
    return None


def _bbox_xyxy_from_value(value: object) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        ordered = (value.get("x1"), value.get("y1"), value.get("x2"), value.get("y2"))
        if all(v is not None for v in ordered):
            value = ordered
        else:
            ordered = (value.get("xmin"), value.get("ymin"), value.get("xmax"), value.get("ymax"))
            if all(v is not None for v in ordered):
                value = ordered
    try:
        bbox = np.asarray(value, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if len(bbox) != 4 or not np.all(np.isfinite(bbox)):
        return None
    return tuple(float(v) for v in bbox)


def _jsonable_bbox_xyxy(value: object | None) -> list[float] | None:
    bbox = _bbox_xyxy_from_value(value)
    if bbox is None:
        return None
    return [float(v) for v in bbox]


def _target_exclusion_source(mask_source: str | None, bbox_source: str | None) -> str | None:
    sources = []
    if mask_source:
        sources.append(f"mask:{mask_source}")
    if bbox_source:
        sources.append(f"bbox:{bbox_source}")
    return "+".join(sources) if sources else None


def _finite_pointcloud(points_xyz: object) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    return points.astype(np.float32)


def _transform_pointcloud_to_pybullet(points_xyz: np.ndarray, input_frame: str, config: dict[str, object]) -> np.ndarray:
    frame = str(input_frame or "pybullet").lower()
    if frame in {"pybullet", "pb", "scene", "world"}:
        debug_stage("base_sampler", "點雲座標：輸入已是 PyBullet frame，略過座標轉換", points=len(points_xyz))
        return np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if frame == "camera":
        debug_stage("base_sampler", "點雲座標：camera frame 轉 PyBullet scene frame", points=len(points_xyz), camera_pb=_camera_position_pb_xyz(config))
        return coord.camera_points_to_pybullet_world(
            points_xyz,
            _camera_position_pb_xyz(config),
        )
    raise ValueError(f"Unsupported pointcloud frame: {input_frame}")


def _transform_grasp_candidates_to_pybullet(
    candidates: list[sample_logic.GraspPoseCandidate],
    config: dict[str, object],
) -> list[sample_logic.GraspPoseCandidate]:
    frame = str(config.get("grasp_pose_frame", "camera")).lower()
    if frame in {"pybullet", "pb", "scene", "world"}:
        debug_stage("base_sampler", "grasp 座標：輸入已是 PyBullet frame，略過座標轉換", grasp_count=len(candidates))
        return candidates
    if frame != "camera":
        raise ValueError(f"Unsupported grasp pose frame: {frame}")

    offset_m = float(config.get("grasp_target_gripper_z_offset_m", 0.0))
    debug_stage(
        "base_sampler",
        "grasp 座標：camera frame position/rotation 轉 PyBullet scene frame",
        grasp_count=len(candidates),
        camera_pb=_camera_position_pb_xyz(config),
        grasp_target_gripper_z_offset_m=offset_m,
    )
    transformed: list[sample_logic.GraspPoseCandidate] = []
    for candidate in candidates:
        camera_rotation = np.asarray(candidate.rotation_matrix, dtype=np.float64).reshape(3, 3)
        target_camera_position = _camera_grasp_target_position(candidate, config)
        position, rotation = coord.camera_grasp_pose_to_pybullet(
            target_camera_position,
            camera_rotation,
            camera_position_pb_xyz=_camera_position_pb_xyz(config),
        )
        transformed.append(
            sample_logic.GraspPoseCandidate(
                index=candidate.index,
                rank=candidate.rank,
                grasp_confidence=candidate.grasp_confidence,
                position_xyz=position,
                rotation_matrix=rotation,
            )
        )
    return transformed


def _camera_grasp_target_position(
    candidate: sample_logic.GraspPoseCandidate,
    config: dict[str, object],
) -> np.ndarray:
    camera_position = np.asarray(candidate.position_xyz, dtype=np.float64).reshape(3)
    camera_rotation = np.asarray(candidate.rotation_matrix, dtype=np.float64).reshape(3, 3)
    offset_m = float(config.get("grasp_target_gripper_z_offset_m", 0.0))
    return (camera_position + camera_rotation[:, 2] * offset_m).astype(np.float64)


def _camera_position_pb_xyz(config: dict[str, object]) -> tuple[float, float, float]:
    arm_base = np.asarray(_arm_base_link_pb_xyz(config), dtype=np.float64).reshape(3)
    camera_offset = np.asarray(_camera_from_arm_base_pb_xyz(config), dtype=np.float64).reshape(3)
    base_yaw = _initial_arm_base_yaw_rad(config)
    rotation = coord.yaw_rotation_matrix(base_yaw)
    camera_position = arm_base + rotation @ camera_offset
    return tuple(float(v) for v in camera_position)


def _camera_from_arm_base_pb_xyz(config: dict[str, object]) -> tuple[float, float, float]:
    raw = config.get("camera_from_arm_base_pb_xyz", config.get("camera_from_arm_base_pb_xy", [0.0, -0.198544, 0.353028]))
    return tuple(float(v) for v in raw)


def _initial_arm_base_yaw_rad(config: dict[str, object]) -> float:
    euler_deg = config.get("base_orientation_euler_deg", [0.0, 0.0, 0.0])
    return float(coord.deg_to_rad(list(euler_deg)[2]))


def _arm_base_link_pb_xyz(config: dict[str, object]) -> tuple[float, float, float]:
    return tuple(float(v) for v in config.get("arm_base_link_pb_xyz", [-0.001193, -0.001505, 0.039689]))


def _car_center_from_arm_base_pb_xy(config: dict[str, object]) -> tuple[float, float]:
    return tuple(float(v) for v in config.get("car_center_from_arm_base_pb_xy", [0.0, -0.1285]))


def _depth_pointcloud_config(
    config: dict[str, object],
    config_path: Path,
    *,
    exclude_mask: object | None = None,
    exclude_bbox_xyxy: object | None = None,
) -> DepthPointcloudConfig:
    raw_intrinsics_path = config.get("intrinsics_path")
    if not raw_intrinsics_path:
        raise ValueError("intrinsics_path is required when depth image input is enabled.")
    return DepthPointcloudConfig(
        camera_name=str(config.get("camera_name", "Camera_Car")),
        intrinsics_path=_resolve_input_path(str(raw_intrinsics_path), config_path.parent),
        min_depth_m=float(config.get("min_depth_m", 0.19)),
        max_depth_m=float(config.get("max_depth_m", 1.5)),
        pixel_stride=int(config.get("pixel_stride", 1)),
        mirror_camera_x=_config_bool(config.get("mirror_depth_camera_x"), True),
        capture_timeout_sec=float(config.get("capture_timeout_sec", 15.0)),
        amcl_topic=str(config.get("amcl_topic", "/amcl_pose")),
        pre_capture_amcl_timeout_sec=float(config.get("pre_capture_amcl_timeout_sec", 2.0)),
        exclude_mask=exclude_mask,
        exclude_bbox_xyxy=exclude_bbox_xyxy,
        exclude_bbox_flip_y=_config_bool(config.get("target_bbox_flip_y_for_depth_alignment"), True),
    )


def _depth_pointcloud_info(result: DepthPointcloudResult) -> dict[str, object]:
    info: dict[str, object] = {
        "pointcloud_input_frame": result.pointcloud_frame,
        "pointcloud_frame": "pybullet",
        "depth_shape": list(result.depth_shape),
        "depth_format": result.depth_format,
        "camera_name": result.camera_name,
        "depth_camera_x_mirrored": bool(result.depth_camera_x_mirrored),
        "target_mask_applied": result.target_mask_shape is not None,
        "target_mask_pixel_count": int(result.target_mask_pixel_count),
        "target_bbox_applied": result.excluded_bbox_xyxy is not None,
        "target_bbox_flip_y_for_depth_alignment": bool(result.excluded_bbox_flip_y),
        "excluded_bbox_xyxy": None if result.excluded_bbox_xyxy is None else list(result.excluded_bbox_xyxy),
        "excluded_bbox_pixel_count": int(result.excluded_bbox_pixel_count),
        "target_exclusion_applied": result.excluded_mask_shape is not None,
        "excluded_mask_shape": None if result.excluded_mask_shape is None else list(result.excluded_mask_shape),
        "excluded_mask_pixel_count": int(result.excluded_mask_pixel_count),
    }
    if result.depth_metric_m is not None:
        info["_depth_metric_m"] = result.depth_metric_m
    if result.amcl_pose is not None:
        info["capture_amcl_pose"] = result.amcl_pose
    return info


def _annotate_and_filter_samples_for_ros_map(
    samples: list[dict[str, object]],
    config: dict[str, object],
    config_path: Path,
    pointcloud_info: dict[str, object],
    *,
    return_rejected: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    reference_amcl_pose = pointcloud_info.get("capture_amcl_pose") or config.get("reference_amcl_pose")
    arm_base_link_pb_xyz = _arm_base_link_pb_xyz(config)
    car_center_from_arm_base_pb_xy = _car_center_from_arm_base_pb_xy(config)
    reference_pb_yaw_rad = _initial_arm_base_yaw_rad(config)
    map_checker, map_status = _load_map_checker(config, config_path, reference_amcl_pose)
    if map_status == "skipped_no_reference_amcl_pose":
        debug_stage("base_sampler", "ROS map 檢查：沒有 AMCL reference，依規則直接 fail")
        return [], {
            "input_sample_count": len(samples),
            "feasible_sample_count": 0,
            "rejected_sample_count": len(samples),
            "map_status": map_status,
            "has_reference_amcl_pose": False,
            "failure_phase": "missing_amcl_for_ros_map",
            "message": "No AMCL pose is available, so ROS map alignment cannot be checked.",
        }

    annotated: list[dict[str, object]] = []
    rejected = 0
    for sample in samples:
        local_xy = (float(sample["pb_base_link_xyz"][0]), float(sample["pb_base_link_xyz"][1]))
        local_yaw = float(sample["pb_base_link_yaw_rad"])
        ros_amcl_pose, ros_base_pose = coord.local_pybullet_base_to_ros_map_poses(
            local_xy,
            local_yaw,
            reference_amcl_pose=reference_amcl_pose,
            arm_base_link_pb_xyz=arm_base_link_pb_xyz,
            car_center_from_arm_base_pb_xy=car_center_from_arm_base_pb_xy,
            reference_pb_yaw_rad=reference_pb_yaw_rad,
        )
        map_feasible = True
        map_reason = map_status
        if map_checker is not None and ros_amcl_pose is not None:
            map_feasible = map_free_space.ros_map_pose_is_clear(map_checker, ros_amcl_pose)
            map_reason = "clear" if map_feasible else "blocked"

        record = dict(sample)
        record["pybullet_goal_pose"] = dict(sample["goal_pose"])
        record["ros_map_amcl_pose"] = ros_amcl_pose
        record["ros_map_base_link_pose"] = ros_base_pose
        record["reference_amcl_pose"] = reference_amcl_pose
        record["arm_base_link_pb_xyz"] = list(arm_base_link_pb_xyz)
        record["reference_pb_yaw_rad"] = float(reference_pb_yaw_rad)
        record["reference_pb_yaw_deg"] = float(coord.rad_to_deg(reference_pb_yaw_rad))
        record["car_center_from_arm_base_pb_xy"] = list(car_center_from_arm_base_pb_xy)
        record["ros_map_feasible"] = bool(map_feasible)
        record["ros_map_check"] = map_reason
        record["goal_pose"] = ros_amcl_pose if ros_amcl_pose is not None else ros_base_pose
        if map_feasible:
            annotated.append(record)
        else:
            rejected += 1
            if return_rejected:
                annotated.append(record)

    return annotated, {
        "input_sample_count": len(samples),
        "feasible_sample_count": len(samples) - rejected,
        "rejected_sample_count": rejected,
        "returned_sample_count": len(annotated),
        "map_status": map_status,
        "has_reference_amcl_pose": reference_amcl_pose is not None,
        "returned_rejected_samples": bool(return_rejected),
    }


def _load_map_checker(
    config: dict[str, object],
    config_path: Path,
    reference_amcl_pose: object | None,
) -> tuple[map_free_space.MapFreeSpace | None, str]:
    if not bool(config.get("enable_ros_map_check", False)):
        return None, "disabled"
    raw_map_path = config.get("map_yaml_path")
    if not raw_map_path:
        return None, "missing_map_yaml_path"
    if reference_amcl_pose is None and not bool(config.get("allow_ros_map_check_without_reference", False)):
        return None, "skipped_no_reference_amcl_pose"
    map_path = _resolve_input_path(str(raw_map_path), config_path.parent)
    return (
        map_free_space.build_map_free_space(
            map_path,
            vehicle_length_x_m=float(config.get("vehicle_base_length_x_m", 0.33)),
            vehicle_length_y_m=float(config.get("vehicle_base_length_y_m", 0.35)),
        ),
        "loaded",
    )


def _robot_scene_config(config: dict[str, object], config_path: Path) -> RobotSceneConfig:
    bounds = tuple(tuple(float(v) for v in pair) for pair in config["joint_bounds_deg"])
    return RobotSceneConfig(
        urdf_path=_resolve_input_path(str(config["urdf_path"]), config_path.parent),
        base_height_m=float(config.get("base_height_m", _arm_base_link_pb_xyz(config)[2])),
        base_orientation_euler_deg=tuple(float(v) for v in config["base_orientation_euler_deg"]),
        joint_reset_deg=tuple(float(v) for v in config["joint_reset_deg"]),
        joint_bounds_deg=bounds,
        ee_link_index=int(config["ee_link_index"]),
        controllable_joints=int(config["controllable_joints"]),
        position_tolerance_m=float(config["position_tolerance_m"]),
        orientation_tolerance_deg=float(config["orientation_tolerance_deg"]),
        ik_max_iterations=int(config["ik_max_iterations"]),
        ik_residual_threshold=float(config["ik_residual_threshold"]),
        voxel_collision_threshold_m=float(config["voxel_collision_threshold_m"]),
        voxel_size_m=_voxel_size_m(config),
    )


def load_car_approach_config(config_path: Path = DEFAULT_CONFIG_PATH) -> tuple[Path, dict[str, object]]:
    resolved_path = _resolve_approach_path(config_path)
    return resolved_path, _load_sampler_config(resolved_path)


def _load_sampler_config(config_path: Path) -> dict[str, object]:
    payload = _load_yaml(config_path)
    defaults = {
        "grasp_json_path": None,
        "pointcloud_frame": "pybullet",
        "grasp_pose_frame": "camera",
        "grasp_target_gripper_z_offset_m": 0.1,
        "camera_from_arm_base_pb_xy": [0.0, -0.198544, 0.353028],
        "arm_base_link_pb_xyz": [-0.001193, -0.001505, 0.039689],
        "car_center_from_arm_base_pb_xy": [0.0, -0.1285],
        "reference_amcl_pose": None,
        "enable_ros_map_check": True,
        "allow_ros_map_check_without_reference": False,
        "map_yaml_path": "tools/car_control/src/nav_goal_bridge_pkg/config/keepout_map.yaml",
        "vehicle_base_length_x_m": 0.33,
        "vehicle_base_length_y_m": 0.35,
        "capture_depth_image": False,
        "camera_name": "Camera_Car",
        "intrinsics_path": None,
        "capture_timeout_sec": 15.0,
        "pre_capture_amcl_timeout_sec": 2.0,
        "min_depth_m": 0.19,
        "max_depth_m": 1.5,
        "pixel_stride": 1,
        "mirror_depth_camera_x": True,
        "target_bbox_flip_y_for_depth_alignment": True,
        "voxel_edge_length_m": 0.03,
        "voxel_size_m": None,
        "max_voxels": 0,
        "voxel_collision_threshold_m": 0.0,
        "max_grasps": 10,
        "max_samples": 160,
        "min_backoff_m": 0.10,
        "max_backoff_m": 0.60,
        "backoff_step_m": 0.02,
        "yaw_span_deg": 20.0,
        "yaw_step_deg": 5.0,
        "ik_max_iterations": 5000,
        "ik_residual_threshold": 1e-4,
    }
    return {**defaults, **payload}



def _controllable_joint_ids(p: Any, robot_id: int, expected_count: int) -> list[int]:
    joint_ids: list[int] = []
    for index in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, index)
        joint_type = info[2]
        joint_name = info[1].decode("utf-8")
        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC) and joint_name != "Revolute 6":
            joint_ids.append(index)
    return joint_ids[: int(expected_count)]


def _set_joint_positions(p: Any, robot_id: int, joint_ids: list[int], joint_positions: list[float]) -> None:
    for joint_id, value in zip(joint_ids, joint_positions):
        p.resetJointState(robot_id, joint_id, targetValue=float(value), targetVelocity=0.0)


def _clip_to_bounds(values: list[float], bounds_deg: tuple[tuple[float, float], ...]) -> list[float]:
    result = []
    for value, (lower_deg, upper_deg) in zip(values, bounds_deg):
        lower = coord.deg_to_rad(lower_deg)
        upper = coord.deg_to_rad(upper_deg)
        result.append(min(max(float(value), lower), upper))
    return result


def _robot_collides(p: Any, robot_id: int, obstacle_ids: list[int], threshold_m: float) -> bool:
    for obstacle_id in obstacle_ids:
        for point in p.getClosestPoints(robot_id, obstacle_id, distance=max(float(threshold_m), 0.0)):
            if float(point[8]) <= float(threshold_m):
                return True
    return False


def _urdf_search_paths(urdf_path: Path) -> list[Path]:
    return [path for path in (urdf_path.parent, urdf_path.parent.parent) if path.exists()]


def _voxel_size_m(config: dict[str, object]) -> float:
    raw_value = config.get("voxel_edge_length_m")
    if raw_value is None:
        raw_value = config.get("voxel_size_m", 0.03)
    result = float(raw_value)
    if result <= 0.0:
        raise ValueError(f"voxel edge length must be positive, got {result}.")
    return result


def _config_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _positive_int_or_none(value: object) -> int | None:
    result = int(value or 0)
    return result if result > 0 else None


def _load_yaml(path: Path) -> dict[str, object]:
    try:
        import yaml  # type: ignore[import]

        with Path(path).open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
    except ImportError:
        payload = _parse_flat_yaml(Path(path))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def _parse_flat_yaml(path: Path) -> dict[str, object]:
    payload: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        value = raw_value.strip()
        if value.lower() in {"", "null", "none"}:
            payload[key.strip()] = None
            continue
        try:
            payload[key.strip()] = ast.literal_eval(value)
        except Exception:
            payload[key.strip()] = value
    return payload


def _resolve_approach_path(path: Path) -> Path:
    return _resolve_input_path(str(path), APPROACH_AGENT_DIR)


def _resolve_input_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    candidates = (base_dir / path, APPROACH_AGENT_DIR / path, VLM_RL_ROOT_DIR / path, Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (base_dir / path).resolve()


def _failure(
    phase: str,
    message: str,
    started_at: float,
    *,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    result = {
        "success": False,
        "status_code": "BASE_SAMPLE_FAIL",
        "phase": phase,
        "message": message,
        "next_agent": None,
        "elapsed_sec": time.time() - started_at,
    }
    if extra:
        result.update(extra)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a base pose with voxel obstacles in PyBullet.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--grasp-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_base_sampling(
        BaseSamplerRunConfig(
            config_path=args.config,
            grasp_json_path=args.grasp_json,
        )
    )
    debug_stage(
        "base_sampler",
        "CLI 執行結果",
        success=bool(result.get("success")),
        phase=result.get("phase"),
        message=result.get("message"),
    )


if __name__ == "__main__":
    main()
