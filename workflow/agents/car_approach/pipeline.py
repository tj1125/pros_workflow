"""End-to-end car approach pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import base_sampler, move_arm, move_car
from .debug_log import debug_stage


APPROACH_AGENT_DIR = Path(__file__).resolve().parent
WORKFLOW_ROOT_DIR = APPROACH_AGENT_DIR.parents[1]
REPO_ROOT_DIR = WORKFLOW_ROOT_DIR.parent
DEFAULT_CONFIG_PATH = base_sampler.DEFAULT_CONFIG_PATH


@dataclass(frozen=True)
class ApproachPipelineRunConfig:
    config_path: Path = DEFAULT_CONFIG_PATH
    grasp_json_path: Path | None = None
    grasp_result_payload: dict[str, object] | None = None
    pointcloud_xyz: object | None = None
    depth_png_bytes: bytes | None = None
    target_mask: object | None = None


def run_approach_pipeline(run_config: ApproachPipelineRunConfig | None = None) -> dict[str, object]:
    run_config = run_config or ApproachPipelineRunConfig()
    started_at = time.time()
    config_path, config = base_sampler.load_car_approach_config(run_config.config_path)
    debug_stage("pipeline", "開始 car_approach 總流程", config=str(config_path))

    debug_stage("pipeline", "階段 1：開始 base sampling，準備點雲、voxel、grasp、base pose 檢查")
    try:
        sampling_result = base_sampler.run_base_sampling(
            base_sampler.BaseSamplerRunConfig(
                config_path=run_config.config_path,
                grasp_json_path=run_config.grasp_json_path,
                grasp_result_payload=run_config.grasp_result_payload,
                pointcloud_xyz=run_config.pointcloud_xyz,
                depth_png_bytes=run_config.depth_png_bytes,
                target_mask=run_config.target_mask,
            )
        )
    except Exception as exc:
        sampling_result = {
            "success": False,
            "status_code": "BASE_SAMPLE_FAIL",
            "phase": "base_sampling_exception",
            "message": str(exc),
        }
        debug_stage("pipeline", "階段 1 失敗：base sampling 發生例外", error=str(exc))
        return _failure_pipeline_result(
            failure_stage="base_sampling",
            phase="base_sampling_exception",
            message=str(exc),
            started_at=started_at,
            config=config,
            config_path=config_path,
            sampling_result=sampling_result,
        )
    fallback_to_closest = False
    selected = sampling_result.get("selected_solution")
    if not bool(sampling_result.get("success", False)):
        closest_fallback = _closest_fallback_solution(sampling_result)
        if closest_fallback is not None:
            fallback_to_closest = True
            selected = closest_fallback
            sampling_result = dict(sampling_result)
            sampling_result["selected_solution"] = closest_fallback
            sampling_result["selected_solution_source"] = "closest_solution_fallback_no_ik"
            sampling_result["fallback_to_closest_solution"] = True
            debug_stage(
                "pipeline",
                "階段 1 fallback：沒有 IK feasible，改用不碰撞的 closest_solution 開車",
                phase=sampling_result.get("phase"),
                sample_index=closest_fallback.get("sample_index"),
                position_error=closest_fallback.get("ee_position_error_m"),
                orientation_error=closest_fallback.get("ee_orientation_error_deg"),
            )
        else:
            debug_stage(
                "pipeline",
                "階段 1 失敗：base sampling 沒有找到可執行解",
                phase=sampling_result.get("phase"),
                message=sampling_result.get("message"),
            )
            return _failure_pipeline_result(
                failure_stage="base_sampling",
                phase=str(sampling_result.get("phase", "base_sampling_failed")),
                message=str(sampling_result.get("message", "Base sampling failed.")),
                started_at=started_at,
                config=config,
                config_path=config_path,
                sampling_result=sampling_result,
            )

    if not fallback_to_closest:
        selected = sampling_result.get("selected_solution")

    if not isinstance(selected, dict) or not selected:
        debug_stage(
            "pipeline",
            "階段 1 失敗：base sampler 成功但沒有 selected_solution",
            phase=sampling_result.get("phase"),
            message=sampling_result.get("message"),
        )
        return _failure_pipeline_result(
            failure_stage="base_sampling",
            phase="no_selected_solution",
            message="Base sampler succeeded without a selected solution.",
            started_at=started_at,
            config=config,
            config_path=config_path,
            sampling_result=sampling_result,
        )

    goal_pose = selected.get("goal_pose")
    if not isinstance(goal_pose, dict) or not goal_pose:
        debug_stage("pipeline", "階段 1 失敗：selected solution 沒有 goal_pose", fallback=fallback_to_closest)
        return _failure_pipeline_result(
            failure_stage="base_sampling",
            phase="missing_selected_goal_pose",
            message="Selected base solution does not include a goal_pose.",
            started_at=started_at,
            config=config,
            config_path=config_path,
            sampling_result=sampling_result,
        )

    nav_config = _navigation_config(config)
    execute_in_place = bool(sampling_result.get("execute_in_place", False)) and not fallback_to_closest
    if execute_in_place:
        debug_stage(
            "pipeline",
            "階段 2 跳過：當前位置已有可行 grasp，不開車，直接執行手臂",
            goal_pose=goal_pose,
        )
        nav_result = None
    else:
        debug_stage(
            "pipeline",
            "階段 2：把 selected base pose 丟給 move_car 開車",
            fallback=fallback_to_closest,
            goal_pose=goal_pose,
        )
        nav_result = move_car.drive_to_pose_by_rule(goal_pose, config=nav_config)
        if not bool(nav_result.get("success", False)):
            debug_stage(
                "pipeline",
                "階段 2 失敗：車子沒有抵達 selected base pose",
                phase=nav_result.get("phase"),
                final_amcl=nav_result.get("final_amcl_pose"),
            )
            nav_failure_phase = "closest_navigation_failed" if fallback_to_closest else "navigation_failed"
            nav_failure_message = (
                "Collision-free closest_solution selected, but the car did not reach it."
                if fallback_to_closest
                else "Base pose selected, but the car did not reach it."
            )
            return _failure_pipeline_result(
                failure_stage="navigation",
                phase=nav_failure_phase,
                message=nav_failure_message,
                started_at=started_at,
                config=config,
                config_path=config_path,
                sampling_result=sampling_result,
                nav_result=nav_result,
                nav_config=nav_config,
            )

        debug_stage(
            "pipeline",
            "階段 2 完成：車子已抵達，準備重算 IK 並執行手臂流程",
            fallback=fallback_to_closest,
            final_amcl=nav_result.get("final_amcl_pose"),
        )
    debug_stage("pipeline", "階段 3：開始 move_arm arrival grasp sequence", execute_in_place=execute_in_place)
    arm_result = move_arm.run_arrival_grasp_sequence(
        selected,
        nav_result=nav_result,
        config=_arm_motion_config(config, config_path),
    )
    debug_stage(
        "pipeline",
        "階段 3 完成：手臂流程結束",
        success=arm_result.get("success"),
        phase=arm_result.get("phase"),
    )

    debug_stage("pipeline", "cleanup：不論手臂結果，先把手臂回到 config reset joints", arm_phase=arm_result.get("phase"))
    arm_reset_result = _try_reset_arm_to_config(
        config,
        config_path,
        reason=f"post_arm:{arm_result.get('phase', 'unknown')}",
    )
    debug_stage(
        "pipeline",
        "cleanup：手臂 reset joints 結束",
        success=arm_reset_result.get("success"),
        phase=arm_reset_result.get("phase"),
    )

    return_pose = _return_pose_from_nav_result(nav_result)
    debug_stage("pipeline", "階段 4：開始 move_car 回到原位", return_pose=return_pose)
    car_return_result = _try_return_car_to_initial(nav_result, nav_config, reason="post_arm:return_to_initial")
    debug_stage(
        "pipeline",
        "階段 4 完成：回程流程結束",
        success=car_return_result.get("success"),
        phase=car_return_result.get("phase"),
    )

    arm_success = bool(arm_result.get("success", False))
    arm_reset_success = bool(arm_reset_result.get("success", False))
    car_return_success = bool(car_return_result.get("success", False))
    success = arm_success and arm_reset_success and car_return_success
    if success:
        phase = "done"
        message = "Base reached, arm grasp sequence finished, arm reset joints were commanded, and car returned to the original pose."
        failed_stage = None
        failed_phase = None
    elif not arm_success:
        phase = "arm_sequence_failed"
        message = "Base reached, but the arm grasp sequence failed; cleanup reset and car return were still attempted."
        failed_stage = "arm_sequence"
        failed_phase = str(arm_result.get("phase", "failed"))
    elif not arm_reset_success:
        phase = "arm_reset_failed"
        message = "Arm sequence finished, but cleanup reset joints failed; car return was still attempted."
        failed_stage = "arm_reset"
        failed_phase = str(arm_reset_result.get("phase", "failed"))
    else:
        phase = "return_failed"
        message = "Arm sequence finished and arm reset was attempted, but the car did not return to the original pose."
        failed_stage = "car_return"
        failed_phase = str(car_return_result.get("phase", "failed"))
    debug_stage("pipeline", "car_approach 總流程結束", success=success, phase=phase, failed_stage=failed_stage, failed_phase=failed_phase)

    return _pipeline_result(
        success=success,
        phase=phase,
        message=message,
        started_at=started_at,
        sampling_result=sampling_result,
        nav_result=nav_result,
        arm_result=arm_result,
        arm_reset_result=arm_reset_result,
        car_return_result=car_return_result,
        failed_stage=failed_stage,
        failed_phase=failed_phase,
    )


def _failure_pipeline_result(
    *,
    failure_stage: str,
    phase: str,
    message: str,
    started_at: float,
    config: dict[str, object],
    config_path: Path,
    sampling_result: dict[str, object],
    nav_result: dict[str, object] | None = None,
    arm_result: dict[str, object] | None = None,
    nav_config: move_car.RuleNavigationConfig | None = None,
) -> dict[str, object]:
    debug_stage("pipeline", "cleanup：流程失敗，開始保底收尾", failed_stage=failure_stage, failed_phase=phase)
    arm_reset_result = _try_reset_arm_to_config(config, config_path, reason=f"failure:{failure_stage}:{phase}")
    car_return_result = _try_return_car_to_initial(nav_result, nav_config, reason=f"failure:{failure_stage}:{phase}")
    debug_stage(
        "pipeline",
        "cleanup：流程失敗收尾結束",
        arm_reset_success=arm_reset_result.get("success"),
        car_return_success=car_return_result.get("success"),
    )
    return _pipeline_result(
        success=False,
        phase=phase,
        message=message,
        started_at=started_at,
        sampling_result=sampling_result,
        nav_result=nav_result,
        arm_result=arm_result,
        arm_reset_result=arm_reset_result,
        car_return_result=car_return_result,
        failed_stage=failure_stage,
        failed_phase=phase,
    )


def _try_reset_arm_to_config(config: dict[str, object], config_path: Path, *, reason: str) -> dict[str, object]:
    try:
        arm_config = _arm_motion_config(config, config_path)
        return move_arm.reset_arm_to_config_pose(config=arm_config, reason=reason)
    except Exception as exc:
        debug_stage("pipeline", "cleanup：手臂 reset joints 發生例外", reason=reason, error=str(exc))
        return {"success": False, "phase": "arm_reset_exception", "message": str(exc), "reason": reason}


def _try_return_car_to_initial(
    nav_result: dict[str, object] | None,
    nav_config: move_car.RuleNavigationConfig | None,
    *,
    reason: str,
) -> dict[str, object]:
    if nav_result is None:
        return {
            "success": True,
            "skipped": True,
            "phase": "not_started_no_car_motion",
            "message": "Car did not start navigation before this failure.",
            "reason": reason,
        }
    return_pose = _return_pose_from_nav_result(nav_result)
    if return_pose is None:
        return {
            "success": False,
            "skipped": True,
            "phase": "missing_initial_pose",
            "message": "Cannot return to the original pose because navigation did not report initial_amcl_pose.",
            "reason": reason,
        }
    if nav_config is None:
        return {
            "success": False,
            "phase": "missing_navigation_config",
            "message": "Cannot return to the original pose because navigation config is unavailable.",
            "target_pose": return_pose,
            "reason": reason,
        }
    try:
        result = move_car.drive_back_to_pose_by_rule(return_pose, config=nav_config)
        result["reason"] = reason
        return result
    except Exception as exc:
        debug_stage("pipeline", "cleanup：車子回原位發生例外", reason=reason, error=str(exc))
        return {"success": False, "phase": "car_return_exception", "message": str(exc), "target_pose": return_pose, "reason": reason}


def _pipeline_result(
    *,
    success: bool,
    phase: str,
    message: str,
    started_at: float,
    sampling_result: dict[str, object],
    nav_result: dict[str, object] | None = None,
    arm_result: dict[str, object] | None = None,
    arm_reset_result: dict[str, object] | None = None,
    car_return_result: dict[str, object] | None = None,
    failed_stage: str | None = None,
    failed_phase: str | None = None,
) -> dict[str, object]:
    nav_payload = nav_result or _not_started("navigation")
    arm_payload = arm_result or _not_started("arm_sequence")
    arm_reset_payload = arm_reset_result or _not_started("arm_reset")
    car_return_payload = car_return_result or _not_started("car_return")
    return {
        "success": bool(success),
        "status_code": "APPROACH_SUCCESS" if success else "APPROACH_FAIL",
        "phase": phase,
        "message": message,
        "failed_stage": None if success else failed_stage,
        "failed_phase": None if success else (failed_phase or phase),
        "next_agent": None,
        "sampling_result": sampling_result,
        "selected_solution": sampling_result.get("selected_solution", {}),
        "closest_solution": sampling_result.get("closest_solution", {}),
        "selected_solution_source": sampling_result.get("selected_solution_source", ""),
        "fallback_to_closest_solution": bool(sampling_result.get("fallback_to_closest_solution", False)),
        "sampling_summary": sampling_result.get("sampling_summary", {}),
        "nav_result": nav_payload,
        "arm_result": arm_payload,
        "arm_reset_result": arm_reset_payload,
        "car_return_result": car_return_payload,
        "cleanup_result": {
            "arm_reset": arm_reset_payload,
            "car_return": car_return_payload,
        },
        "elapsed_sec": time.time() - started_at,
    }


def _closest_fallback_solution(sampling_result: dict[str, object]) -> dict[str, object] | None:
    if str(sampling_result.get("phase", "")) != "no_feasible_sample":
        return None
    closest = sampling_result.get("closest_solution")
    if not isinstance(closest, dict) or not closest:
        return None
    if not bool(closest.get("collision_free", False)):
        return None
    goal_pose = closest.get("goal_pose")
    if not isinstance(goal_pose, dict) or not goal_pose:
        return None
    fallback = dict(closest)
    fallback["selected_solution_source"] = "closest_solution_fallback_no_ik"
    fallback["fallback_to_closest_solution"] = True
    return fallback


def _not_started(phase: str) -> dict[str, object]:
    return {"success": False, "skipped": True, "phase": phase, "message": "not started"}


def _return_pose_from_nav_result(nav_result: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(nav_result, dict):
        return None
    initial_pose = nav_result.get("initial_amcl_pose")
    if not isinstance(initial_pose, dict) or initial_pose.get("x") is None or initial_pose.get("y") is None:
        return None
    return initial_pose


def _navigation_config(config: dict[str, object]) -> move_car.RuleNavigationConfig:
    return move_car.RuleNavigationConfig(
        amcl_topic=str(config.get("amcl_topic", move_car.DEFAULT_AMCL_TOPIC)),
        front_wheel_topic=str(config.get("front_wheel_topic", move_car.DEFAULT_FRONT_WHEEL_TOPIC)),
        rear_wheel_topic=str(config.get("rear_wheel_topic", move_car.DEFAULT_REAR_WHEEL_TOPIC)),
        xy_tolerance_m=float(config.get("xy_tolerance_m", 0.03)),
        face_target_yaw_tolerance_rad=float(config.get("face_target_yaw_tolerance_rad", 0.08)),
        drive_heading_tolerance_rad=float(config.get("drive_heading_tolerance_rad", 0.14)),
        final_yaw_tolerance_rad=float(config.get("final_yaw_tolerance_rad", 0.08)),
        command_period_sec=float(config.get("command_period_sec", 0.1)),
        amcl_wait_timeout_sec=float(config.get("amcl_wait_timeout_sec", 5.0)),
        amcl_stale_timeout_sec=float(config.get("amcl_stale_timeout_sec", 1.0)),
        max_duration_sec=float(config.get("max_duration_sec", 120.0)),
    )


def _arm_motion_config(config: dict[str, object], config_path: Path) -> move_arm.ArmMotionConfig:
    return move_arm.ArmMotionConfig(
        urdf_path=_resolve_input_path(str(config["urdf_path"]), config_path.parent),
        base_height_m=float(config["base_height_m"]),
        base_orientation_euler_deg=tuple(float(v) for v in config["base_orientation_euler_deg"]),
        joint_reset_deg=tuple(float(v) for v in config["joint_reset_deg"]),
        joint_bounds_deg=tuple(tuple(float(v) for v in pair) for pair in config["joint_bounds_deg"]),
        ee_link_index=int(config["ee_link_index"]),
        controllable_joints=int(config["controllable_joints"]),
        position_tolerance_m=float(config["position_tolerance_m"]),
        ik_max_iterations=int(config["ik_max_iterations"]),
        ik_residual_threshold=float(config["ik_residual_threshold"]),
        gripper_joint_index=int(config.get("gripper_joint_index", 4)),
        gripper_open_deg=float(config.get("gripper_open_deg", 60.0)),
        gripper_close_deg=float(config.get("gripper_close_deg", 10.0)),
        arm_topic=str(config.get("arm_topic", move_arm.DEFAULT_ARM_TOPIC)),
        joint_state_topic=str(config.get("joint_state_topic", move_arm.DEFAULT_JOINT_STATE_TOPIC)),
        joint_state_groups=_joint_state_groups(config.get("joint_state_groups")),
        joint_state_wait_sec=float(config.get("arm_joint_state_wait_sec", 5.0)),
        command_timeout_sec=float(config.get("arm_command_timeout_sec", 5.0)),
        command_tolerance_rad=float(config.get("arm_command_tolerance_rad", 0.08)),
        republish_interval_sec=float(config.get("arm_republish_interval_sec", 0.02)),
        waypoint_steps=int(config.get("arm_waypoint_steps", 3)),
        reset_hold_sec=float(config.get("arm_reset_hold_sec", 3.0)),
        gripper_close_timeout_sec=float(config.get("gripper_close_timeout_sec", 2.0)),
        gripper_settle_hold_sec=float(config.get("gripper_settle_hold_sec", 1.0)),
        gripper_close_settle_hold_sec=float(config.get("gripper_close_settle_hold_sec", config.get("gripper_settle_hold_sec", 1.0))),
        publisher_match_timeout_sec=float(config.get("arm_publisher_match_timeout_sec", 1.0)),
        cube_z_distance_topic=str(config.get("cube_z_distance_topic", move_arm.DEFAULT_CUBE_Z_DISTANCE_TOPIC)),
        cube_z_distance_final_waypoint_stop_threshold_m=float(config.get("cube_z_distance_final_waypoint_stop_threshold_m", 0.025)),
        cube_z_distance_final_verify_threshold_m=float(config.get("cube_z_distance_final_verify_threshold_m", 0.04)),
        cube_z_distance_poll_interval_sec=float(config.get("cube_z_distance_poll_interval_sec", 0.02)),
        cube_z_distance_verify_timeout_sec=float(config.get("cube_z_distance_verify_timeout_sec", 1.0)),
    )


def _joint_state_groups(raw_groups: object) -> tuple[tuple[str, ...], ...]:
    if not isinstance(raw_groups, list):
        return move_arm.DEFAULT_JOINT_STATE_GROUPS
    groups: list[tuple[str, ...]] = []
    for raw_group in raw_groups:
        if isinstance(raw_group, list):
            group = tuple(str(name).strip() for name in raw_group if str(name).strip())
        else:
            group = (str(raw_group).strip(),)
        if group:
            groups.append(group)
    return tuple(groups) if groups else move_arm.DEFAULT_JOINT_STATE_GROUPS


def _resolve_input_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    candidates = (
        base_dir / path,
        APPROACH_AGENT_DIR / path,
        WORKFLOW_ROOT_DIR / path,
        REPO_ROOT_DIR / path,
        Path.cwd() / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (base_dir / path).resolve()
