"""Finish the grasp directly from car_approach after the base arrives."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CAR_ARM_FINISH_ENABLED = True
CAR_ARM_FINISH_NODE_NAME = "approach_agent_car_arm_finish"
CAR_ARM_FINISH_GRIPPER_JOINT_INDEX = 4
CAR_ARM_FINISH_GRIPPER_OPEN_DEG = 60.0
CAR_ARM_FINISH_GRIPPER_CLOSE_DEG = 10.0
CAR_ARM_FINISH_WRIST_JOINT_INDEX = 3
CAR_ARM_FINISH_TARGET_GRASP_YAW_REFERENCE_PERIOD_DEG = 180.0
CAR_ARM_FINISH_TARGET_GRASP_YAW_TO_WRIST_SIGN = 1.0
CAR_ARM_FINISH_WRIST_YAW_MIN_DEG = 60.0
CAR_ARM_FINISH_WRIST_YAW_MAX_DEG = 120.0
CAR_ARM_FINISH_ARM_ACTION_SERVER_NAME = "arm_action_server"
CAR_ARM_FINISH_ACTION_SERVER_WAIT_SEC = 5.0
CAR_ARM_FINISH_ACTION_RESULT_TIMEOUT_SEC = 45.0
CAR_ARM_FINISH_GRASP_TARGET_OFFSET_X_M = 0.04
CAR_ARM_FINISH_GRASP_TARGET_OFFSET_Y_M = 0.0
CAR_ARM_FINISH_GRASP_TARGET_OFFSET_Z_M = -0.135
CAR_ARM_FINISH_GRASP_TARGET_HORIZONTAL_DEEPER_M = 0.0
CAR_ARM_FINISH_GRASP_TARGET_TRAJECTORY_STEPS = 5
CAR_ARM_FINISH_GRASP_TARGET_WAYPOINT_SLEEP_SEC = 0.1
CAR_ARM_FINISH_GRASP_TARGET_TOLERANCE_M = 0.03
CAR_ARM_FINISH_JOINT_STATE_WAIT_SEC = 5.0
CAR_ARM_FINISH_JOINT_COMMAND_TIMEOUT_SEC = 5.0
CAR_ARM_FINISH_JOINT_COMMAND_TOLERANCE_RAD = 0.08
CAR_ARM_FINISH_JOINT_COMMAND_REPUBLISH_INTERVAL_SEC = 0.1
CAR_ARM_FINISH_GRIPPER_CLOSE_DELAY_SEC = 3.0
CAR_ARM_FINISH_AFTER_GRIPPER_CLOSE_INIT_POSE_DELAY_SEC = 1.0
CAR_ARM_FINISH_ACTION_EXECUTION_MODEL = "tools_arm_action_server_car_grasp_sequence"


def env_flag(name: str, default: bool) -> bool:
    default_value = "1" if default else "0"
    return os.getenv(name, default_value).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }


def car_arm_finish_enabled() -> bool:
    return env_flag("APPROACH_AGENT_CAR_FINISH_ARM_ON_ARRIVAL", CAR_ARM_FINISH_ENABLED)


def run_car_arm_finish_sequence(
    solution: dict[str, object],
    *,
    visualization_records: Sequence[dict[str, object]],
    planning_config: Any,
    arm_config: dict[str, object] | None,
    arm_base_target: dict[str, object] | None = None,
    planner_config_path: Path | None = None,
    p_mod: Any | None = None,
    pybullet_data: Any | None = None,
) -> dict[str, object]:
    """Open the gripper, reach the PB target offset, close, then call init_pose."""

    if not car_arm_finish_enabled():
        return {
            "success": False,
            "skipped": True,
            "phase": "disabled",
            "message": "car_approach direct arm finish is disabled.",
        }

    try:
        target_record = _selected_visualization_record(
            solution,
            visualization_records,
        )
        motion_solution, metadata = _build_motion_solution(
            solution,
            target_record=target_record,
            planning_config=planning_config,
            arm_config=arm_config,
            arm_base_target=arm_base_target,
        )
        gripper_index = int(metadata["preopened_gripper_joint_index"])
        gripper_open_rad = float(metadata["preopened_gripper_target_rad"])
        wrist_index = int(metadata.get("target_grasp_wrist_joint_index", CAR_ARM_FINISH_WRIST_JOINT_INDEX))
        wrist_target_rad = _wrist_target_rad_from_metadata(
            metadata,
            motion_solution,
            wrist_index=wrist_index,
        )
        gripper_close_rad = math.radians(
            float(
                os.getenv(
                    "APPROACH_AGENT_CAR_GRIPPER_CLOSE_DEG",
                    str(CAR_ARM_FINISH_GRIPPER_CLOSE_DEG),
                )
            )
        )
        (
            grasp_target_base_position_xyz,
            grasp_target_base_position_source,
            grasp_target_frame_metadata,
        ) = _grasp_target_base_position_xyz(
            motion_solution,
            target_record=target_record,
            planning_config=planning_config,
            arm_base_target=arm_base_target,
        )
        grasp_target_offset_xyz = _grasp_target_offset_xyz_from_environment()
        car_grasp_target_position_xyz, offset_metadata = _apply_grasp_target_offset_xyz(
            grasp_target_base_position_xyz,
            grasp_target_offset_xyz,
            motion_solution=motion_solution,
            planning_config=planning_config,
            arm_config=arm_config,
            p_mod=p_mod,
            pybullet_data=pybullet_data,
        )
        car_grasp_offset_position_xyz = car_grasp_target_position_xyz
        horizontal_deeper_m = _grasp_target_horizontal_deeper_m_from_environment()
        car_grasp_target_position_xyz, horizontal_depth_metadata = _apply_grasp_target_horizontal_depth_xyz(
            car_grasp_offset_position_xyz,
            horizontal_deeper_m,
        )
        metadata.update(
            {
                "pre_close_ee_offset_enabled": False,
                "pre_close_ee_offset_skipped": True,
                "pre_close_ee_offset_sequence": [],
                "pre_close_ee_offset_replaced_by": "car_grasp_sequence_target_offset_xyz",
                "car_grasp_sequence_base_position_source": grasp_target_base_position_source,
                "car_grasp_sequence_base_position_xyz": grasp_target_base_position_xyz,
                "car_grasp_sequence_offset_xyz": grasp_target_offset_xyz,
                "car_grasp_sequence_offset_position_xyz": car_grasp_offset_position_xyz,
                "car_grasp_sequence_target_position_xyz": car_grasp_target_position_xyz,
                "return_to_start_after_gripper_close_requested": True,
                "return_to_start_path_source": "arm_action_server.init_pose",
                **grasp_target_frame_metadata,
                **offset_metadata,
                **horizontal_depth_metadata,
            }
        )
        action_config = _action_finish_config_from_environment()
        publish_result = _run_car_grasp_sequence_action(
            car_grasp_target_position_xyz,
            action_config=action_config,
            gripper_joint_index=gripper_index,
            gripper_open_rad=gripper_open_rad,
            wrist_joint_index=wrist_index,
            wrist_target_rad=wrist_target_rad,
            gripper_close_rad=gripper_close_rad,
        )
    except Exception as exc:
        return {
            "success": False,
            "skipped": False,
            "phase": "action_sequence_failed",
            "message": str(exc),
        }

    execution_sequence = [
        "calculate_car_grasp_sequence_target",
        "open_gripper_60deg",
        "rotate_wrist_to_target_grasp_yaw",
        "send_car_grasp_sequence_action",
        "close_gripper_10deg",
        "return_to_init_pose",
    ]

    result: dict[str, object] = {
        **metadata,
        **publish_result,
        "success": bool(publish_result.get("success", False)),
        "skipped": False,
        "phase": "published" if bool(publish_result.get("success", False)) else "action_sequence_failed",
        "source": "approach_agent_car_approach_direct_arm_finish",
        "arm_motion_skipped": False,
        "target_pose_source": (
            "selected car_approach grasp target plus car_grasp_sequence offset "
            "and horizontal depth"
        ),
        "execution_sequence": execution_sequence,
    }
    if bool(result["success"]):
        print(
            "[base_approach] car_approach direct arm finish completed: "
            f"open_finger={float(metadata['preopened_gripper_target_deg']):.2f}deg "
            f"close_finger={float(result.get('gripper_close_deg', CAR_ARM_FINISH_GRIPPER_CLOSE_DEG)):.2f}deg "
            f"wrist={float(metadata.get('target_grasp_wrist_target_deg', float('nan'))):.2f}deg "
            f"car_grasp_target={result.get('car_grasp_sequence_target_position_xyz')} "
            f"init_pose={bool(result.get('init_pose_success', False))}",
            flush=True,
        )
    return result


def _grasp_target_base_position_xyz(
    solution: dict[str, object],
    *,
    target_record: dict[str, object] | None,
    planning_config: Any,
    arm_base_target: dict[str, object] | None,
) -> tuple[list[float], str, dict[str, object]]:
    planning_world_position: list[float] | None = None
    planning_world_source = ""
    if isinstance(target_record, dict) and target_record.get("target_pb") is not None:
        planning_world_position = _float_xyz(target_record["target_pb"], label="target_record.target_pb")
        planning_world_source = "target_record.target_pb"
    else:
        for key in (
            "target_pb",
            "target_position_pybullet_xyz",
            "final_ee_position_xyz",
        ):
            if solution.get(key) is not None:
                planning_world_position = _float_xyz(solution[key], label=f"solution.{key}")
                planning_world_source = f"solution.{key}"
                break

    if planning_world_position is None:
        raise KeyError("grasp_target needs target_record.target_pb or solution.final_ee_position_xyz.")

    base_pose = _grasp_target_base_pose_for_action(
        solution,
        planning_config=planning_config,
        arm_base_target=arm_base_target,
    )
    if base_pose is None:
        return (
            planning_world_position,
            planning_world_source,
            {
                "car_grasp_sequence_coordinate_frame": "car_approach_planning_pb_world_untransformed",
                "car_grasp_sequence_planning_world_position_xyz": planning_world_position,
                "car_grasp_sequence_planning_world_position_source": planning_world_source,
                "car_grasp_sequence_selected_base_transform_applied": False,
            },
        )

    base_xyz, base_yaw_rad, base_pose_source, base_pose_metadata = base_pose
    action_position = _planning_pb_world_position_to_tools_base_local_pb(
        planning_world_position,
        base_xyz=base_xyz,
        base_yaw_rad=base_yaw_rad,
        planning_config=planning_config,
    )
    return (
        action_position,
        f"{planning_world_source}->{base_pose_source}",
        {
            "car_grasp_sequence_coordinate_frame": "tools_pb_world_aligned_to_arrived_arm_base",
            "car_grasp_sequence_planning_world_position_xyz": planning_world_position,
            "car_grasp_sequence_planning_world_position_source": planning_world_source,
            "car_grasp_sequence_base_pose_source": base_pose_source,
            "car_grasp_sequence_selected_base_transform_applied": True,
            "car_grasp_sequence_base_xyz": base_xyz,
            "car_grasp_sequence_base_yaw_rad": float(base_yaw_rad),
            "car_grasp_sequence_base_yaw_deg": math.degrees(float(base_yaw_rad)),
            **base_pose_metadata,
        },
    )


def _grasp_target_base_pose_for_action(
    solution: dict[str, object],
    *,
    planning_config: Any,
    arm_base_target: dict[str, object] | None,
) -> tuple[list[float], float, str, dict[str, object]] | None:
    yaw_compensation = (
        arm_base_target.get("yaw_compensation")
        if isinstance(arm_base_target, dict)
        else None
    )
    if isinstance(yaw_compensation, dict) and bool(yaw_compensation.get("applied", False)):
        final_xyz_raw = yaw_compensation.get("final_base_link_local_pb_xyz")
        final_yaw_rad = _optional_float(yaw_compensation.get("final_base_link_local_pb_yaw_rad"))
        if final_xyz_raw is not None and final_yaw_rad is not None:
            final_xyz = _float_xyz(
                final_xyz_raw,
                label="arm_base_target.yaw_compensation.final_base_link_local_pb_xyz",
            )
            return (
                final_xyz,
                float(final_yaw_rad),
                "nav_result.final_amcl_pose_actual_base_local_pb",
                {
                    "car_grasp_sequence_nav_error_compensation_applied": True,
                    "car_grasp_sequence_planned_base_xyz": yaw_compensation.get("planned_base_link_local_pb_xyz"),
                    "car_grasp_sequence_final_base_xyz": final_xyz,
                    "car_grasp_sequence_vehicle_position_error_xyz_m": yaw_compensation.get(
                        "vehicle_position_error_from_planned_xyz_m"
                    ),
                    "car_grasp_sequence_vehicle_position_error_xy_m": yaw_compensation.get(
                        "vehicle_position_error_from_planned_xy_m"
                    ),
                    "car_grasp_sequence_vehicle_position_error_norm_m": yaw_compensation.get(
                        "vehicle_position_error_from_planned_norm_m"
                    ),
                    "car_grasp_sequence_vehicle_yaw_error_rad": yaw_compensation.get(
                        "vehicle_yaw_error_from_planned_rad"
                    ),
                    "car_grasp_sequence_vehicle_yaw_error_deg": yaw_compensation.get(
                        "vehicle_yaw_error_from_planned_deg"
                    ),
                },
            )

    base_xyz_raw = solution.get("pb_base_link_xyz")
    base_yaw_rad = _optional_float(solution.get("pb_base_link_yaw_rad"))
    if base_xyz_raw is None or base_yaw_rad is None:
        return None

    base_xyz = _float_xyz(base_xyz_raw, label="solution.pb_base_link_xyz")
    try:
        base_xyz[2] = float(planning_config.initial_height)
    except (AttributeError, TypeError, ValueError):
        pass
    return (
        base_xyz,
        float(base_yaw_rad),
        "selected_solution.planned_base_local_pb",
        {
            "car_grasp_sequence_nav_error_compensation_applied": False,
            "car_grasp_sequence_planned_base_xyz": base_xyz,
            "car_grasp_sequence_final_base_xyz": None,
        },
    )


def _planning_pb_world_position_to_tools_base_local_pb(
    position_xyz: Sequence[Any],
    *,
    base_xyz: Sequence[Any],
    base_yaw_rad: float,
    planning_config: Any,
) -> list[float]:
    position = np.asarray(_float_xyz(position_xyz, label="position_xyz"), dtype=np.float64)
    base = np.asarray(_float_xyz(base_xyz, label="base_xyz"), dtype=np.float64)
    delta = position - base
    cos_yaw = math.cos(-float(base_yaw_rad))
    sin_yaw = math.sin(-float(base_yaw_rad))
    local_x = (cos_yaw * float(delta[0])) - (sin_yaw * float(delta[1]))
    local_y = (sin_yaw * float(delta[0])) + (cos_yaw * float(delta[1]))
    try:
        base_height = float(planning_config.initial_height)
    except (AttributeError, TypeError, ValueError):
        base_height = float(base[2])
    return [
        float(local_x),
        float(local_y),
        float(base_height + float(delta[2])),
    ]


def _grasp_target_offset_xyz_from_environment() -> list[float]:
    return [
        float(os.getenv("APPROACH_AGENT_CAR_GRASP_TARGET_OFFSET_X_M", str(CAR_ARM_FINISH_GRASP_TARGET_OFFSET_X_M))),
        float(os.getenv("APPROACH_AGENT_CAR_GRASP_TARGET_OFFSET_Y_M", str(CAR_ARM_FINISH_GRASP_TARGET_OFFSET_Y_M))),
        float(os.getenv("APPROACH_AGENT_CAR_GRASP_TARGET_OFFSET_Z_M", str(CAR_ARM_FINISH_GRASP_TARGET_OFFSET_Z_M))),
    ]


def _grasp_target_offset_frame_from_environment() -> str:
    return (
        os.getenv("APPROACH_AGENT_CAR_GRASP_TARGET_OFFSET_FRAME", "base_local").strip().lower()
        or "base_local"
    )


def _grasp_target_horizontal_deeper_m_from_environment() -> float:
    return float(
        os.getenv(
            "APPROACH_AGENT_CAR_GRASP_TARGET_HORIZONTAL_DEEPER_M",
            str(CAR_ARM_FINISH_GRASP_TARGET_HORIZONTAL_DEEPER_M),
        )
    )


def _apply_grasp_target_horizontal_depth_xyz(
    target_position_xyz: Sequence[Any],
    depth_m: float,
) -> tuple[list[float], dict[str, object]]:
    target_position = np.asarray(
        _float_xyz(target_position_xyz, label="target_position_xyz"),
        dtype=np.float64,
    )
    depth = float(depth_m)
    if not math.isfinite(depth):
        raise ValueError("horizontal grasp target depth must be finite.")

    metadata: dict[str, object] = {
        "car_grasp_sequence_horizontal_depth_offset_m": float(depth),
        "car_grasp_sequence_target_position_before_horizontal_depth_xyz": (
            target_position.astype(float).tolist()
        ),
    }
    if depth == 0.0:
        return target_position.astype(float).tolist(), {
            **metadata,
            "car_grasp_sequence_horizontal_depth_offset_applied": False,
            "car_grasp_sequence_horizontal_depth_offset_vector_xyz": [0.0, 0.0, 0.0],
        }

    horizontal_xy = target_position[:2]
    horizontal_norm = float(np.linalg.norm(horizontal_xy))
    if horizontal_norm <= 1e-9:
        return target_position.astype(float).tolist(), {
            **metadata,
            "car_grasp_sequence_horizontal_depth_offset_applied": False,
            "car_grasp_sequence_horizontal_depth_offset_skip_reason": "target_xy_norm_is_zero",
            "car_grasp_sequence_horizontal_depth_offset_vector_xyz": [0.0, 0.0, 0.0],
        }

    horizontal_axis = np.asarray(
        [
            float(horizontal_xy[0]) / horizontal_norm,
            float(horizontal_xy[1]) / horizontal_norm,
            0.0,
        ],
        dtype=np.float64,
    )
    offset_vector = horizontal_axis * depth
    return (target_position + offset_vector).astype(float).tolist(), {
        **metadata,
        "car_grasp_sequence_horizontal_depth_offset_applied": True,
        "car_grasp_sequence_horizontal_depth_axis_xyz": horizontal_axis.astype(float).tolist(),
        "car_grasp_sequence_horizontal_depth_offset_vector_xyz": offset_vector.astype(float).tolist(),
    }


def _apply_grasp_target_offset_xyz(
    target_position_xyz: Sequence[Any],
    offset_xyz: Sequence[Any],
    *,
    motion_solution: dict[str, object],
    planning_config: Any,
    arm_config: dict[str, object] | None,
    p_mod: Any | None,
    pybullet_data: Any | None,
) -> tuple[list[float], dict[str, object]]:
    target_position = np.asarray(
        _float_xyz(target_position_xyz, label="target_position_xyz"),
        dtype=np.float64,
    )
    offset = np.asarray(_float_xyz(offset_xyz, label="offset_xyz"), dtype=np.float64)
    offset_frame = _grasp_target_offset_frame_from_environment()
    metadata: dict[str, object] = {
        "car_grasp_sequence_offset_frame": offset_frame,
    }

    if not np.any(np.abs(offset) > 0.0):
        return target_position.astype(float).tolist(), {
            **metadata,
            "car_grasp_sequence_offset_vector_xyz": [0.0, 0.0, 0.0],
        }

    if offset_frame not in {"base", "base_local", "tools_pb_world", "world"}:
        raise ValueError(
            "APPROACH_AGENT_CAR_GRASP_TARGET_OFFSET_FRAME must be one of "
            "base, base_local, tools_pb_world, or world."
        )

    offset_vector = offset
    return (target_position + offset_vector).astype(float).tolist(), {
        **metadata,
        "car_grasp_sequence_offset_vector_xyz": offset_vector.astype(float).tolist(),
    }


def _wrist_target_rad_from_metadata(
    metadata: dict[str, object],
    motion_solution: dict[str, object],
    *,
    wrist_index: int,
) -> float:
    metadata_target = _optional_float(metadata.get("target_grasp_wrist_target_rad"))
    if metadata_target is not None:
        return float(metadata_target)

    joint_positions = _goal_joint_rad_from_solution(motion_solution)
    if 0 <= int(wrist_index) < len(joint_positions):
        return float(joint_positions[int(wrist_index)])

    raise ValueError(
        f"Cannot command wrist joint {int(wrist_index)}; "
        f"goal vector length is {len(joint_positions)}."
    )


def _action_finish_config_from_environment() -> dict[str, object]:
    tolerance_rad = _joint_command_tolerance_rad_from_environment()
    return {
        "joint_state_wait_sec": max(
            0.0,
            float(
                os.getenv(
                    "APPROACH_AGENT_CAR_JOINT_STATE_WAIT_SEC",
                    str(CAR_ARM_FINISH_JOINT_STATE_WAIT_SEC),
                )
            ),
        ),
        "joint_command_timeout_sec": max(
            0.0,
            float(
                os.getenv(
                    "APPROACH_AGENT_CAR_JOINT_COMMAND_TIMEOUT_SEC",
                    str(CAR_ARM_FINISH_JOINT_COMMAND_TIMEOUT_SEC),
                )
            ),
        ),
        "joint_command_tolerance_rad": max(0.0, float(tolerance_rad)),
        "joint_command_republish_interval_sec": max(
            0.0,
            float(
                os.getenv(
                    "APPROACH_AGENT_CAR_JOINT_COMMAND_REPUBLISH_INTERVAL_SEC",
                    str(CAR_ARM_FINISH_JOINT_COMMAND_REPUBLISH_INTERVAL_SEC),
                )
            ),
        ),
        "gripper_close_delay_sec": max(
            0.0,
            float(
                os.getenv(
                    "APPROACH_AGENT_ARM_GRIPPER_CLOSE_DELAY_SEC",
                    str(CAR_ARM_FINISH_GRIPPER_CLOSE_DELAY_SEC),
                )
            ),
        ),
        "init_pose_delay_sec": max(
            0.0,
            float(
                os.getenv(
                    "APPROACH_AGENT_CAR_AFTER_GRIPPER_CLOSE_INIT_POSE_DELAY_SEC",
                    str(CAR_ARM_FINISH_AFTER_GRIPPER_CLOSE_INIT_POSE_DELAY_SEC),
                )
            ),
        ),
    }


def _joint_command_tolerance_rad_from_environment() -> float:
    tolerance_deg = os.getenv("APPROACH_AGENT_CAR_JOINT_COMMAND_TOLERANCE_DEG")
    if tolerance_deg is not None and tolerance_deg.strip():
        return math.radians(float(tolerance_deg))
    return float(
        os.getenv(
            "APPROACH_AGENT_CAR_JOINT_COMMAND_TOLERANCE_RAD",
            str(CAR_ARM_FINISH_JOINT_COMMAND_TOLERANCE_RAD),
        )
    )


def _run_car_grasp_sequence_action(
    target_position_xyz: Sequence[Any],
    *,
    action_config: dict[str, object],
    gripper_joint_index: int,
    gripper_open_rad: float,
    wrist_joint_index: int,
    wrist_target_rad: float,
    gripper_close_rad: float,
) -> dict[str, object]:
    import rclpy
    from action_interface.action import ArmGoal
    from rclpy.action import ActionClient
    from rclpy.node import Node

    target_position = _float_xyz(
        target_position_xyz,
        label="car_grasp_sequence target_position_xyz",
    )
    action_name = (
        os.getenv("APPROACH_AGENT_CAR_ARM_ACTION_SERVER_NAME", CAR_ARM_FINISH_ARM_ACTION_SERVER_NAME).strip()
        or CAR_ARM_FINISH_ARM_ACTION_SERVER_NAME
    )
    server_wait_sec = float(
        os.getenv(
            "APPROACH_AGENT_CAR_ARM_ACTION_SERVER_WAIT_SEC",
            str(CAR_ARM_FINISH_ACTION_SERVER_WAIT_SEC),
        )
    )
    action_timeout_sec = float(
        os.getenv(
            "APPROACH_AGENT_CAR_ARM_ACTION_RESULT_TIMEOUT_SEC",
            str(CAR_ARM_FINISH_ACTION_RESULT_TIMEOUT_SEC),
        )
    )
    trajectory_steps = int(
        os.getenv(
            "APPROACH_AGENT_CAR_GRASP_TARGET_TRAJECTORY_STEPS",
            str(CAR_ARM_FINISH_GRASP_TARGET_TRAJECTORY_STEPS),
        )
    )
    waypoint_sleep_sec = float(
        os.getenv(
            "APPROACH_AGENT_CAR_GRASP_TARGET_WAYPOINT_SLEEP_SEC",
            str(CAR_ARM_FINISH_GRASP_TARGET_WAYPOINT_SLEEP_SEC),
        )
    )
    goal_tolerance_m = float(
        os.getenv(
            "APPROACH_AGENT_CAR_GRASP_TARGET_TOLERANCE_M",
            str(CAR_ARM_FINISH_GRASP_TARGET_TOLERANCE_M),
        )
    )
    joint_state_wait_sec = max(0.0, float(action_config["joint_state_wait_sec"]))
    joint_command_timeout_sec = max(0.0, float(action_config["joint_command_timeout_sec"]))
    joint_command_tolerance_rad = max(0.0, float(action_config["joint_command_tolerance_rad"]))
    joint_command_republish_interval_sec = max(
        0.0,
        float(action_config["joint_command_republish_interval_sec"]),
    )
    close_delay_sec = max(0.0, float(action_config["gripper_close_delay_sec"]))
    init_pose_delay_sec = max(0.0, float(action_config["init_pose_delay_sec"]))

    owns_rclpy = False
    node = None
    try:
        if not rclpy.ok():
            rclpy.init(args=None)
            owns_rclpy = True

        node = Node(CAR_ARM_FINISH_NODE_NAME)
        action_client = ActionClient(node, ArmGoal, action_name)

        if not action_client.wait_for_server(timeout_sec=max(0.0, server_wait_sec)):
            return {
                "success": False,
                "execution_model": CAR_ARM_FINISH_ACTION_EXECUTION_MODEL,
                "action_server": action_name,
                "car_grasp_sequence_target_position_xyz": target_position,
                "message": f"Arm action server '{action_name}' was not available.",
            }

        print(
            "[base_approach] car arm finish: sending car_grasp_sequence action "
            f"target={target_position} wrist={math.degrees(float(wrist_target_rad)):.2f}deg",
            flush=True,
        )
        sequence_result = _send_arm_action_goal(
            rclpy,
            node,
            action_client,
            ArmGoal,
            mode="car_grasp_sequence",
            target_position=target_position,
            trajectory_steps=max(1, trajectory_steps),
            waypoint_sleep_sec=max(0.0, waypoint_sleep_sec),
            goal_tolerance_m=max(0.0, goal_tolerance_m),
            wrist_joint_index=int(wrist_joint_index),
            wrist_target_rad=float(wrist_target_rad),
            gripper_joint_index=int(gripper_joint_index),
            gripper_open_rad=float(gripper_open_rad),
            gripper_close_rad=float(gripper_close_rad),
            joint_state_wait_sec=float(joint_state_wait_sec),
            joint_command_timeout_sec=float(joint_command_timeout_sec),
            joint_command_tolerance_rad=float(joint_command_tolerance_rad),
            joint_command_republish_interval_sec=float(joint_command_republish_interval_sec),
            gripper_close_delay_sec=float(close_delay_sec),
            init_pose_delay_sec=float(init_pose_delay_sec),
            timeout_sec=max(0.0, action_timeout_sec),
        )
        success = bool(sequence_result.get("success", False))
        return {
            "success": success,
            "execution_model": CAR_ARM_FINISH_ACTION_EXECUTION_MODEL,
            "action_server": action_name,
            "car_grasp_sequence_success": success,
            "car_grasp_sequence_result": sequence_result,
            "joint_command_timeout_sec": float(joint_command_timeout_sec),
            "joint_command_tolerance_rad": float(joint_command_tolerance_rad),
            "joint_command_tolerance_deg": math.degrees(float(joint_command_tolerance_rad)),
            "joint_command_republish_interval_sec": float(joint_command_republish_interval_sec),
            "joint_state_wait_sec": float(joint_state_wait_sec),
            "car_grasp_sequence_target_position_xyz": target_position,
            "car_grasp_sequence_trajectory_steps": max(1, trajectory_steps),
            "car_grasp_sequence_waypoint_sleep_sec": max(0.0, waypoint_sleep_sec),
            "car_grasp_sequence_tolerance_m": max(0.0, goal_tolerance_m),
            "target_move_success": success,
            "gripper_open_success": success,
            "wrist_success": success,
            "wrist_joint_index": int(wrist_joint_index),
            "wrist_target_rad": float(wrist_target_rad),
            "wrist_target_deg": math.degrees(float(wrist_target_rad)),
            "gripper_close_success": success,
            "gripper_joint_index": int(gripper_joint_index),
            "gripper_open_rad": float(gripper_open_rad),
            "gripper_open_deg": math.degrees(float(gripper_open_rad)),
            "gripper_close_rad": float(gripper_close_rad),
            "gripper_close_deg": math.degrees(float(gripper_close_rad)),
            "gripper_close_delay_sec": float(close_delay_sec),
            "init_pose_delay_sec": float(init_pose_delay_sec),
            "init_pose_success": success,
            "return_to_start_published": success,
            "message": str(sequence_result.get("message", "")),
        }
    finally:
        if node is not None:
            node.destroy_node()
        if owns_rclpy and rclpy.ok():
            rclpy.shutdown()


def _send_arm_action_goal(
    rclpy_module: Any,
    node: Any,
    action_client: Any,
    arm_goal_type: Any,
    *,
    mode: str,
    target_position: Sequence[Any] | None = None,
    trajectory_steps: int = 0,
    waypoint_sleep_sec: float = 0.0,
    goal_tolerance_m: float = 0.0,
    wrist_joint_index: int | None = None,
    wrist_target_rad: float | None = None,
    gripper_joint_index: int | None = None,
    gripper_open_rad: float | None = None,
    gripper_close_rad: float | None = None,
    joint_state_wait_sec: float = 0.0,
    joint_command_timeout_sec: float = 0.0,
    joint_command_tolerance_rad: float = 0.0,
    joint_command_republish_interval_sec: float = 0.0,
    gripper_close_delay_sec: float = 0.0,
    init_pose_delay_sec: float = 0.0,
    timeout_sec: float,
) -> dict[str, object]:
    goal_msg = arm_goal_type.Goal()
    goal_msg.mode = str(mode)
    if target_position is not None:
        goal_msg.target_position = _float_xyz(target_position, label=f"{mode}.target_position")
    goal_msg.trajectory_steps = int(trajectory_steps)
    goal_msg.waypoint_sleep_sec = float(waypoint_sleep_sec)
    goal_msg.goal_tolerance_m = float(goal_tolerance_m)
    if wrist_joint_index is not None:
        goal_msg.wrist_joint_index = int(wrist_joint_index)
    if wrist_target_rad is not None:
        goal_msg.wrist_target_rad = float(wrist_target_rad)
    if gripper_joint_index is not None:
        goal_msg.gripper_joint_index = int(gripper_joint_index)
    if gripper_open_rad is not None:
        goal_msg.gripper_open_rad = float(gripper_open_rad)
    if gripper_close_rad is not None:
        goal_msg.gripper_close_rad = float(gripper_close_rad)
    goal_msg.joint_state_wait_sec = float(joint_state_wait_sec)
    goal_msg.joint_command_timeout_sec = float(joint_command_timeout_sec)
    goal_msg.joint_command_tolerance_rad = float(joint_command_tolerance_rad)
    goal_msg.joint_command_republish_interval_sec = float(joint_command_republish_interval_sec)
    goal_msg.gripper_close_delay_sec = float(gripper_close_delay_sec)
    goal_msg.init_pose_delay_sec = float(init_pose_delay_sec)

    send_future = action_client.send_goal_async(goal_msg)
    rclpy_module.spin_until_future_complete(node, send_future, timeout_sec=timeout_sec)
    if not send_future.done():
        return {
            "success": False,
            "accepted": False,
            "mode": str(mode),
            "message": f"Timed out while sending {mode} goal.",
        }

    goal_handle = send_future.result()
    if goal_handle is None:
        return {
            "success": False,
            "accepted": False,
            "mode": str(mode),
            "message": f"{mode} goal returned no handle.",
        }
    if not bool(goal_handle.accepted):
        return {
            "success": False,
            "accepted": False,
            "mode": str(mode),
            "message": f"{mode} goal was rejected.",
        }

    result_future = goal_handle.get_result_async()
    rclpy_module.spin_until_future_complete(node, result_future, timeout_sec=timeout_sec)
    if not result_future.done():
        return {
            "success": False,
            "accepted": True,
            "mode": str(mode),
            "message": f"Timed out waiting for {mode} result.",
        }

    result_response = result_future.result()
    result = getattr(result_response, "result", None)
    status = getattr(result_response, "status", None)
    return {
        "success": bool(getattr(result, "success", False)),
        "accepted": True,
        "mode": str(mode),
        "status": None if status is None else int(status),
        "message": str(getattr(result, "message", "")),
    }


def _goal_joint_rad_from_solution(solution: dict[str, object]) -> list[float]:
    raw_rad = solution.get("ik_joint_solution_rad")
    if isinstance(raw_rad, list):
        return _float_sequence(raw_rad, label="solution.ik_joint_solution_rad")
    raw_deg = solution.get("ik_joint_solution_deg")
    if isinstance(raw_deg, list):
        return [math.radians(value) for value in _float_sequence(raw_deg, label="solution.ik_joint_solution_deg")]
    raise KeyError("solution needs ik_joint_solution_rad or ik_joint_solution_deg.")


def _float_xyz(values: Any, *, label: str) -> list[float]:
    try:
        result = [float(value) for value in values]
    except TypeError as exc:
        raise ValueError(f"{label} must be a numeric [x, y, z] sequence.") from exc
    if len(result) != 3:
        raise ValueError(f"{label} must contain exactly 3 values.")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values.")
    return result


def _float_sequence(values: Sequence[Any], *, label: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{label} must not be empty.")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values.")
    return result


def _build_motion_solution(
    solution: dict[str, object],
    *,
    target_record: dict[str, object] | None,
    planning_config: Any,
    arm_config: dict[str, object] | None,
    arm_base_target: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    goal_joint_positions = _goal_joint_rad_from_solution(solution)
    metadata: dict[str, object] = {
        "raw_goal_joint_positions_rad": [float(value) for value in goal_joint_positions],
        "raw_goal_joint_positions_deg": [math.degrees(float(value)) for value in goal_joint_positions],
    }

    if isinstance(arm_base_target, dict) and arm_base_target.get("command_joint_position_rad") is not None:
        joint_index = int(arm_base_target.get("joint_index", 0))
        if 0 <= joint_index < len(goal_joint_positions):
            locked_joint_rad = float(arm_base_target["command_joint_position_rad"])
            goal_joint_positions[joint_index] = locked_joint_rad
            metadata["arm_base_target_applied"] = True
            metadata["arm_base_target"] = dict(arm_base_target)
            metadata["locked_arm_base_joint"] = True
            metadata["locked_arm_base_joint_index"] = int(joint_index)
            metadata["locked_arm_base_joint_rad"] = float(locked_joint_rad)
            metadata["locked_arm_base_joint_deg"] = math.degrees(float(locked_joint_rad))
            metadata["locked_arm_base_joint_source"] = "arm_base_target.command_joint_position_rad"
        else:
            metadata["arm_base_target_applied"] = False
            metadata["arm_base_target_error"] = (
                f"joint index {joint_index} is outside goal vector length {len(goal_joint_positions)}"
            )
    else:
        metadata["arm_base_target_applied"] = False
        if isinstance(arm_base_target, dict):
            metadata["arm_base_target"] = dict(arm_base_target)

    if not bool(metadata.get("locked_arm_base_joint", False)):
        fallback_joint_index = int(os.getenv("APPROACH_AGENT_CAR_ARM_BASE_JOINT_INDEX", "0"))
        if 0 <= fallback_joint_index < len(goal_joint_positions):
            locked_joint_rad = float(goal_joint_positions[fallback_joint_index])
            metadata["locked_arm_base_joint"] = True
            metadata["locked_arm_base_joint_index"] = int(fallback_joint_index)
            metadata["locked_arm_base_joint_rad"] = float(locked_joint_rad)
            metadata["locked_arm_base_joint_deg"] = math.degrees(float(locked_joint_rad))
            metadata["locked_arm_base_joint_source"] = "selected_solution.ik_joint_solution_rad"
        else:
            raise ValueError(
                f"Cannot lock arm base joint {fallback_joint_index}; "
                f"goal vector length is {len(goal_joint_positions)}."
            )

    goal_joint_positions, wrist_metadata = _apply_wrist_from_target_grasp_yaw(
        goal_joint_positions,
        solution=solution,
        target_record=target_record,
        planning_config=planning_config,
        arm_config=arm_config,
    )
    metadata.update(wrist_metadata)

    gripper_index = int(
        os.getenv(
            "APPROACH_AGENT_CAR_GRIPPER_JOINT_INDEX",
            str(CAR_ARM_FINISH_GRIPPER_JOINT_INDEX),
        )
    )
    gripper_open_deg = float(
        os.getenv(
            "APPROACH_AGENT_CAR_GRIPPER_OPEN_DEG",
            str(CAR_ARM_FINISH_GRIPPER_OPEN_DEG),
        )
    )
    gripper_open_rad = math.radians(gripper_open_deg)
    gripper_lower_deg, gripper_upper_deg = _joint_limit_deg_from_configs(
        joint_index=gripper_index,
        planning_config=planning_config,
        arm_config=arm_config,
    )
    if 0 <= gripper_index < len(goal_joint_positions):
        goal_joint_positions[gripper_index] = gripper_open_rad
        metadata["preopened_gripper_applied"] = True
        metadata["preopened_gripper_limit_min_deg"] = gripper_lower_deg
        metadata["preopened_gripper_limit_max_deg"] = gripper_upper_deg
        if gripper_upper_deg is not None and gripper_open_deg > float(gripper_upper_deg):
            metadata["preopened_gripper_limit_warning"] = (
                f"requested finger open angle {gripper_open_deg:.2f}deg exceeds "
                f"configured upper limit {float(gripper_upper_deg):.2f}deg; "
                "tools/car_control may clamp the command."
            )
    else:
        metadata["preopened_gripper_applied"] = False
        metadata["preopened_gripper_error"] = (
            f"gripper joint index {gripper_index} is outside goal vector length {len(goal_joint_positions)}"
        )

    motion_solution = dict(solution)
    motion_solution["ik_joint_solution_rad"] = [float(value) for value in goal_joint_positions]
    motion_solution["ik_joint_solution_deg"] = [math.degrees(float(value)) for value in goal_joint_positions]
    motion_solution["car_approach_direct_arm_finish"] = True
    motion_solution["locked_arm_base_joint"] = bool(metadata.get("locked_arm_base_joint", False))
    motion_solution["locked_arm_base_joint_index"] = metadata.get("locked_arm_base_joint_index")
    motion_solution["locked_arm_base_joint_rad"] = metadata.get("locked_arm_base_joint_rad")
    motion_solution["locked_arm_base_joint_deg"] = metadata.get("locked_arm_base_joint_deg")
    motion_solution["locked_arm_base_joint_source"] = metadata.get("locked_arm_base_joint_source")
    motion_solution["preopened_gripper_joint_index"] = gripper_index
    motion_solution["preopened_gripper_target_deg"] = gripper_open_deg
    motion_solution["preopened_gripper_target_rad"] = gripper_open_rad

    metadata.update(
        {
            "target_joint_positions_rad": [float(value) for value in goal_joint_positions],
            "target_joint_positions_deg": [math.degrees(float(value)) for value in goal_joint_positions],
            "preopened_gripper_joint_index": gripper_index,
            "preopened_gripper_target_deg": gripper_open_deg,
            "preopened_gripper_target_rad": gripper_open_rad,
        }
    )
    return motion_solution, metadata


def _selected_visualization_record(
    solution: dict[str, object],
    visualization_records: Sequence[dict[str, object]],
) -> dict[str, object] | None:
    target_order = _optional_int(solution.get("target_sample_order"))
    if target_order is not None:
        for record in visualization_records:
            if _optional_int(record.get("target_sample_order")) == target_order:
                return dict(record)

    selected_rank = _optional_int(solution.get("grasp_rank"))
    if selected_rank is not None:
        for record in visualization_records:
            if _optional_int(record.get("rank")) == selected_rank:
                return dict(record)

    for record in visualization_records:
        if bool(record.get("selected_as_best", False)):
            return dict(record)
    return None


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _apply_wrist_from_target_grasp_yaw(
    goal_joint_positions: list[float],
    *,
    solution: dict[str, object],
    target_record: dict[str, object] | None,
    planning_config: Any,
    arm_config: dict[str, object] | None,
) -> tuple[list[float], dict[str, object]]:
    wrist_index = int(CAR_ARM_FINISH_WRIST_JOINT_INDEX)
    adjusted_positions = [float(value) for value in goal_joint_positions]
    if not (0 <= wrist_index < len(adjusted_positions)):
        return adjusted_positions, {
            "target_grasp_wrist_yaw_applied": False,
            "target_grasp_wrist_yaw_error": (
                f"wrist joint index {wrist_index} is outside joint vector length {len(adjusted_positions)}"
            ),
        }

    try:
        wrist_reset_deg = float(planning_config.joint_reset_deg[wrist_index])
    except (AttributeError, IndexError, TypeError, ValueError):
        wrist_reset_deg = 90.0

    target_yaw_rad, target_yaw_source = _target_grasp_yaw_rad(solution, target_record)
    if target_yaw_rad is None:
        return adjusted_positions, {
            "target_grasp_wrist_yaw_applied": False,
            "target_grasp_wrist_yaw_error": "target grasp yaw is unavailable.",
        }

    target_yaw_normalized_deg = float(math.degrees(target_yaw_rad) % 360.0)
    yaw_reference_rad, yaw_delta_from_reference_rad = _nearest_half_turn_reference_and_delta_rad(target_yaw_rad)
    yaw_delta_from_reference_deg = math.degrees(yaw_delta_from_reference_rad)
    wrist_unclamped_deg = wrist_reset_deg + (
        float(CAR_ARM_FINISH_TARGET_GRASP_YAW_TO_WRIST_SIGN) * yaw_delta_from_reference_deg
    )
    lower_deg, upper_deg = _joint_limit_deg_from_configs(
        joint_index=wrist_index,
        planning_config=planning_config,
        arm_config=arm_config,
    )
    wrist_yaw_min_deg = float(
        os.getenv(
            "APPROACH_AGENT_ARM_WRIST_YAW_MIN_DEG",
            str(CAR_ARM_FINISH_WRIST_YAW_MIN_DEG),
        )
    )
    wrist_yaw_max_deg = float(
        os.getenv(
            "APPROACH_AGENT_ARM_WRIST_YAW_MAX_DEG",
            str(CAR_ARM_FINISH_WRIST_YAW_MAX_DEG),
        )
    )
    if wrist_yaw_min_deg > wrist_yaw_max_deg:
        wrist_yaw_min_deg, wrist_yaw_max_deg = wrist_yaw_max_deg, wrist_yaw_min_deg

    command_lower_deg = wrist_yaw_min_deg if lower_deg is None else max(float(lower_deg), wrist_yaw_min_deg)
    command_upper_deg = wrist_yaw_max_deg if upper_deg is None else min(float(upper_deg), wrist_yaw_max_deg)
    wrist_target_deg = _clamp_deg(wrist_unclamped_deg, command_lower_deg, command_upper_deg)
    adjusted_positions[wrist_index] = math.radians(wrist_target_deg)

    return adjusted_positions, {
        "target_grasp_wrist_yaw_applied": True,
        "target_grasp_wrist_yaw_source": target_yaw_source,
        "target_grasp_wrist_joint_index": wrist_index,
        "target_grasp_yaw_reference_period_deg": float(CAR_ARM_FINISH_TARGET_GRASP_YAW_REFERENCE_PERIOD_DEG),
        "target_grasp_yaw_nearest_reference_rad": float(yaw_reference_rad),
        "target_grasp_yaw_nearest_reference_deg": float(math.degrees(yaw_reference_rad) % 360.0),
        "target_grasp_yaw_rad": float(target_yaw_rad),
        "target_grasp_yaw_deg": float(math.degrees(target_yaw_rad)),
        "target_grasp_yaw_normalized_deg": target_yaw_normalized_deg,
        "target_grasp_yaw_delta_from_nearest_reference_rad": float(yaw_delta_from_reference_rad),
        "target_grasp_yaw_delta_from_nearest_reference_deg": float(yaw_delta_from_reference_deg),
        "target_grasp_wrist_reset_deg": float(wrist_reset_deg),
        "target_grasp_wrist_sign": float(CAR_ARM_FINISH_TARGET_GRASP_YAW_TO_WRIST_SIGN),
        "target_grasp_wrist_unclamped_deg": float(wrist_unclamped_deg),
        "target_grasp_wrist_target_deg": float(wrist_target_deg),
        "target_grasp_wrist_target_rad": float(math.radians(wrist_target_deg)),
        "target_grasp_wrist_limit_min_deg": lower_deg,
        "target_grasp_wrist_limit_max_deg": upper_deg,
        "target_grasp_wrist_yaw_min_deg": float(wrist_yaw_min_deg),
        "target_grasp_wrist_yaw_max_deg": float(wrist_yaw_max_deg),
        "target_grasp_wrist_command_min_deg": float(command_lower_deg),
        "target_grasp_wrist_command_max_deg": float(command_upper_deg),
        "target_grasp_wrist_clamped": not math.isclose(
            float(wrist_target_deg),
            float(wrist_unclamped_deg),
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
    }


def _target_grasp_yaw_rad(
    solution: dict[str, object],
    target_record: dict[str, object] | None,
) -> tuple[float | None, str]:
    if isinstance(target_record, dict):
        raw_rot = target_record.get("target_rot_pb")
        if raw_rot is not None:
            try:
                target_rot_pb = np.asarray(raw_rot, dtype=np.float64).reshape(3, 3)
                return _axis_yaw_xy(target_rot_pb[:, 0], fallback_yaw_rad=0.0), "visualization_record.target_rot_pb[:,0]"
            except Exception:
                pass

    for key in (
        "target_grasp_pose_yaw_rad",
        "target_yaw_rad",
        "desired_pb_base_link_yaw_rad",
    ):
        raw_yaw = solution.get(key)
        if raw_yaw is not None:
            try:
                return _wrap_angle_rad(float(raw_yaw)), f"solution.{key}"
            except (TypeError, ValueError):
                pass

    raw_direction = solution.get("backoff_direction_pb_xy")
    if isinstance(raw_direction, list) and len(raw_direction) >= 2:
        try:
            return _axis_yaw_xy(
                np.asarray([float(raw_direction[0]), float(raw_direction[1]), 0.0], dtype=np.float64),
                fallback_yaw_rad=0.0,
            ), "solution.backoff_direction_pb_xy"
        except (TypeError, ValueError):
            pass

    return None, ""


def _axis_yaw_xy(axis_xyz: np.ndarray, fallback_yaw_rad: float = 0.0) -> float:
    axis = np.asarray(axis_xyz, dtype=np.float64).reshape(3)
    axis_xy = axis[:2]
    axis_norm = float(np.linalg.norm(axis_xy))
    if axis_norm <= 1e-6:
        return float(fallback_yaw_rad)
    return float(math.atan2(float(axis_xy[1]), float(axis_xy[0])))


def _nearest_half_turn_reference_and_delta_deg(angle_rad: float) -> tuple[float, float]:
    angle_deg = float(math.degrees(float(angle_rad)) % 360.0)
    reference_candidates_deg = (0.0, 180.0, 360.0)

    def _sort_key(reference_deg: float) -> tuple[float, int, float]:
        delta_deg = angle_deg - float(reference_deg)
        return (abs(delta_deg), 1 if delta_deg < 0.0 else 0, float(reference_deg))

    reference_deg = min(reference_candidates_deg, key=_sort_key)
    return float(reference_deg), float(angle_deg - reference_deg)


def _nearest_half_turn_reference_and_delta_rad(angle_rad: float) -> tuple[float, float]:
    reference_deg, delta_deg = _nearest_half_turn_reference_and_delta_deg(angle_rad)
    return math.radians(reference_deg), math.radians(delta_deg)


def _wrap_angle_rad(angle_rad: float) -> float:
    return float((float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi)


def _joint_limit_deg_from_configs(
    *,
    joint_index: int,
    planning_config: Any,
    arm_config: dict[str, object] | None,
) -> tuple[float | None, float | None]:
    lower_deg: float | None = None
    upper_deg: float | None = None

    try:
        planning_lower, planning_upper = planning_config.joint_bounds_deg[int(joint_index)]
        lower_deg = float(planning_lower)
        upper_deg = float(planning_upper)
    except (AttributeError, IndexError, TypeError, ValueError):
        pass

    try:
        joint_config = arm_config["joints"][int(joint_index)] if arm_config is not None else None
        if isinstance(joint_config, dict):
            arm_lower = joint_config.get("min_angle")
            arm_upper = joint_config.get("max_angle")
            if arm_lower is not None:
                lower_deg = float(arm_lower) if lower_deg is None else max(lower_deg, float(arm_lower))
            if arm_upper is not None:
                upper_deg = float(arm_upper) if upper_deg is None else min(upper_deg, float(arm_upper))
    except (KeyError, TypeError, ValueError):
        pass

    return lower_deg, upper_deg


def _clamp_deg(value_deg: float, lower_deg: float | None, upper_deg: float | None) -> float:
    clamped = float(value_deg)
    if lower_deg is not None:
        clamped = max(clamped, float(lower_deg))
    if upper_deg is not None:
        clamped = min(clamped, float(upper_deg))
    return clamped
