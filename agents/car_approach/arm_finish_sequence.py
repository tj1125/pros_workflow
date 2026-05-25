"""Finish the grasp directly from car_approach after the base arrives."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CAR_ARM_FINISH_ENABLED = True
CAR_ARM_FINISH_NODE_NAME = "approach_agent_car_arm_finish"
CAR_ARM_FINISH_GRIPPER_JOINT_INDEX = 4
CAR_ARM_FINISH_GRIPPER_OPEN_DEG = 60.0
CAR_ARM_FINISH_GRIPPER_CLOSE_DEG = 10.0
CAR_ARM_FINISH_JOINT_STATE_WAIT_SEC = 5.0
CAR_ARM_FINISH_JOINT_COMMAND_TIMEOUT_SEC = 5.0
CAR_ARM_FINISH_JOINT_COMMAND_TOLERANCE_RAD = 0.08
CAR_ARM_FINISH_JOINT_COMMAND_REPUBLISH_INTERVAL_SEC = 0.1
CAR_ARM_FINISH_AFTER_GRIPPER_CLOSE_INIT_POSE_DELAY_SEC = 1.0
CAR_ARM_FINISH_DIRECT_WAYPOINT_STEPS = 5
CAR_ARM_FINISH_DIRECT_INIT_POSE_SETTLE_SEC = 3.0
CAR_ARM_FINISH_DIRECT_REPUBLISH_INTERVAL_SEC = 0.1
CAR_ARM_FINISH_DIRECT_EXECUTION_MODEL = "approach_agent_direct_joint_interpolation_publish"


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
    """Publish an arrival-adjusted PyBullet IK solution through joint-space waypoints."""
    _ = planner_config_path

    if not car_arm_finish_enabled():
        return {
            "success": False,
            "skipped": True,
            "phase": "disabled",
            "message": "car_approach direct arm finish is disabled.",
        }

    try:
        target_record = _selected_visualization_record(solution, visualization_records)
        motion_solution, metadata = _build_motion_solution(
            solution,
            target_record=target_record,
            planning_config=planning_config,
            arm_config=arm_config,
            arm_base_target=arm_base_target,
            p_mod=p_mod,
            pybullet_data=pybullet_data,
        )
        gripper_index = int(metadata["preopened_gripper_joint_index"])
        gripper_open_rad = float(metadata["preopened_gripper_target_rad"])
        gripper_close_rad = math.radians(
            float(
                os.getenv(
                    "APPROACH_AGENT_CAR_GRIPPER_CLOSE_DEG",
                    str(CAR_ARM_FINISH_GRIPPER_CLOSE_DEG),
                )
            )
        )
        metadata.update(
            {
                "pre_close_ee_offset_enabled": False,
                "pre_close_ee_offset_skipped": True,
                "pre_close_ee_offset_sequence": [],
                "return_to_start_after_gripper_close_requested": True,
                "return_to_start_path_source": "direct_joint_interpolation_reset_pose",
                "arm_finish_execution_mode": "direct_joint_interpolation_publish",
            }
        )
        direct_config = _direct_joint_finish_config_from_environment()
        publish_result = _run_direct_joint_interpolation_sequence(
            motion_solution,
            direct_config=direct_config,
            planning_config=planning_config,
            arm_config=arm_config,
            gripper_joint_index=gripper_index,
            gripper_open_rad=gripper_open_rad,
            gripper_close_rad=gripper_close_rad,
        )
    except Exception as exc:
        return {
            "success": False,
            "skipped": False,
            "phase": "direct_joint_sequence_failed",
            "message": str(exc),
        }

    execution_sequence = [
        "transform_selected_target_to_arrived_base_local_pb",
        "recompute_ik_in_car_approach_pybullet",
        "interpolate_from_current_or_reset_joint_state",
        "publish_joint_waypoints_until_joint_states_reach_tolerance",
        "close_gripper_10deg_hold_before_init_pose",
        "return_to_init_pose",
    ]
    target_pose_source = "arrival-adjusted car_approach PyBullet IK joint solution"

    result: dict[str, object] = {
        **metadata,
        **publish_result,
        "success": bool(publish_result.get("success", False)),
        "skipped": False,
        "phase": "published" if bool(publish_result.get("success", False)) else "direct_joint_sequence_failed",
        "source": "approach_agent_car_approach_arrival_adjusted_arm_finish",
        "arm_motion_skipped": False,
        "target_pose_source": target_pose_source,
        "execution_sequence": execution_sequence,
    }
    if bool(result["success"]):
        print(
            "[base_approach] car_approach arrival-adjusted arm finish completed: "
            f"open_finger={float(metadata['preopened_gripper_target_deg']):.2f}deg "
            f"close_finger={float(result.get('gripper_close_deg', CAR_ARM_FINISH_GRIPPER_CLOSE_DEG)):.2f}deg "
            f"goal_deg={[round(value, 2) for value in result.get('direct_joint_pregrasp_positions_deg', [])]} "
            f"init_pose={bool(result.get('init_pose_success', False))}",
            flush=True,
        )
    return result


def _direct_joint_waypoint_steps_from_environment() -> int:
    return max(
        1,
        int(
            os.getenv(
                "APPROACH_AGENT_CAR_DIRECT_WAYPOINT_STEPS",
                str(CAR_ARM_FINISH_DIRECT_WAYPOINT_STEPS),
            )
        ),
    )


def _linear_interpolated_joint_positions(
    start_positions_rad: Sequence[Any],
    goal_positions_rad: Sequence[Any],
    *,
    steps: int,
) -> list[list[float]]:
    start = np.asarray(_float_sequence(start_positions_rad, label="start_positions_rad"), dtype=np.float64)
    goal = np.asarray(_float_sequence(goal_positions_rad, label="goal_positions_rad"), dtype=np.float64)
    if start.shape != goal.shape:
        raise ValueError(
            "start_positions_rad and goal_positions_rad must have the same length: "
            f"{start.size} vs {goal.size}."
        )
    step_count = max(1, int(steps))
    return [
        ((1.0 - alpha) * start + alpha * goal).astype(float).tolist()
        for alpha in (float(index) / float(step_count) for index in range(1, step_count + 1))
    ]


def _direct_joint_publish_topic_from_config(arm_config: dict[str, object] | None) -> str:
    topic = "/robot_arm"
    try:
        global_config = arm_config.get("global", {}) if isinstance(arm_config, dict) else {}
        configured_topic = global_config.get("arm_topic") if isinstance(global_config, dict) else None
        if configured_topic:
            topic = str(configured_topic).strip() or topic
    except Exception:
        pass
    return topic


def _direct_joint_state_topic_from_config(arm_config: dict[str, object] | None) -> str:
    topic = "/joint_states"
    try:
        global_config = arm_config.get("global", {}) if isinstance(arm_config, dict) else {}
        configured_topic = global_config.get("joint_state_topic") if isinstance(global_config, dict) else None
        if configured_topic:
            topic = str(configured_topic).strip() or topic
    except Exception:
        pass
    return topic


def _direct_joint_state_groups_from_config(
    arm_config: dict[str, object] | None,
) -> list[tuple[str, ...]]:
    try:
        global_config = arm_config.get("global", {}) if isinstance(arm_config, dict) else {}
        raw_names = global_config.get("joint_state_names", []) if isinstance(global_config, dict) else []
    except Exception:
        raw_names = []
    groups: list[tuple[str, ...]] = []
    if not isinstance(raw_names, list):
        return groups
    for raw_name in raw_names:
        if isinstance(raw_name, list):
            group = tuple(str(name).strip() for name in raw_name if str(name).strip())
        else:
            group = (str(raw_name).strip(),)
        if group:
            groups.append(group)
    return groups


def _direct_joint_publish_float_env(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _direct_joint_publish_float_env_any(names: Sequence[str], default: float) -> float:
    for name in names:
        raw = os.getenv(str(name))
        if raw is None:
            continue
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite float.")
        return value
    return float(default)


def _direct_joint_finish_config_from_environment() -> dict[str, float]:
    return {
        "joint_state_wait_sec": _direct_joint_publish_float_env_any(
            (
                "APPROACH_AGENT_CAR_DIRECT_JOINT_STATE_WAIT_SEC",
                "APPROACH_AGENT_CAR_ARM_JOINT_STATE_WAIT_SEC",
            ),
            CAR_ARM_FINISH_JOINT_STATE_WAIT_SEC,
        ),
        "joint_command_timeout_sec": _direct_joint_publish_float_env_any(
            (
                "APPROACH_AGENT_CAR_DIRECT_JOINT_COMMAND_TIMEOUT_SEC",
                "APPROACH_AGENT_CAR_ARM_JOINT_COMMAND_TIMEOUT_SEC",
            ),
            CAR_ARM_FINISH_JOINT_COMMAND_TIMEOUT_SEC,
        ),
        "joint_command_tolerance_rad": _direct_joint_publish_float_env_any(
            (
                "APPROACH_AGENT_CAR_DIRECT_JOINT_COMMAND_TOLERANCE_RAD",
                "APPROACH_AGENT_CAR_ARM_JOINT_COMMAND_TOLERANCE_RAD",
            ),
            CAR_ARM_FINISH_JOINT_COMMAND_TOLERANCE_RAD,
        ),
        "joint_command_republish_interval_sec": _direct_joint_publish_float_env_any(
            (
                "APPROACH_AGENT_CAR_DIRECT_REPUBLISH_INTERVAL_SEC",
                "APPROACH_AGENT_CAR_ARM_JOINT_COMMAND_REPUBLISH_INTERVAL_SEC",
            ),
            CAR_ARM_FINISH_JOINT_COMMAND_REPUBLISH_INTERVAL_SEC,
        ),
        "init_pose_delay_sec": _direct_joint_publish_float_env_any(
            (
                "APPROACH_AGENT_CAR_DIRECT_INIT_POSE_DELAY_SEC",
                "APPROACH_AGENT_CAR_ARM_INIT_POSE_DELAY_SEC",
            ),
            CAR_ARM_FINISH_AFTER_GRIPPER_CLOSE_INIT_POSE_DELAY_SEC,
        ),
    }


def _direct_joint_reset_positions_rad(
    *,
    planning_config: Any,
    arm_config: dict[str, object] | None,
    joint_count: int,
) -> list[float]:
    reset_deg: list[float] = []
    try:
        joints_reset = arm_config.get("joints_reset", {}) if isinstance(arm_config, dict) else {}
        reset_deg = [float(joints_reset[index]) for index in range(int(joint_count))]
    except Exception:
        reset_deg = []
    if len(reset_deg) != int(joint_count):
        try:
            reset_deg = [float(value) for value in planning_config.joint_reset_deg[: int(joint_count)]]
        except Exception:
            reset_deg = []
    if len(reset_deg) != int(joint_count):
        raise ValueError(f"Cannot build reset pose for {int(joint_count)} joints.")
    return [math.radians(value) for value in reset_deg]


def _publish_joint_positions_once(publisher: Any, joint_positions_rad: Sequence[Any]) -> None:
    from trajectory_msgs.msg import JointTrajectoryPoint

    positions = [float(value) for value in joint_positions_rad]
    zero_vec = [0.0] * len(positions)
    message = JointTrajectoryPoint()
    message.positions = positions
    message.velocities = zero_vec
    message.accelerations = zero_vec
    message.effort = zero_vec
    message.time_from_start.sec = 0
    message.time_from_start.nanosec = 0
    publisher.publish(message)


def _map_joint_state_positions(
    msg: Any,
    *,
    joint_state_groups: Sequence[tuple[str, ...]],
    joint_count: int,
) -> list[float] | None:
    try:
        raw_positions = [float(value) for value in msg.position]
    except Exception:
        return None
    if len(raw_positions) < int(joint_count):
        return None
    if joint_state_groups and getattr(msg, "name", None):
        name_to_position = {
            str(name): float(raw_positions[index])
            for index, name in enumerate(msg.name)
            if index < len(raw_positions)
        }
        mapped_positions: list[float] = []
        for group in joint_state_groups[: int(joint_count)]:
            if not all(name in name_to_position for name in group):
                return None
            group_positions = [name_to_position[name] for name in group]
            mapped_positions.append(sum(group_positions) / float(len(group_positions)))
    else:
        mapped_positions = raw_positions[: int(joint_count)]
    if len(mapped_positions) < int(joint_count):
        return None
    if not all(math.isfinite(value) for value in mapped_positions[: int(joint_count)]):
        return None
    return [float(value) for value in mapped_positions[: int(joint_count)]]


def _joint_angle_error_rad(current_rad: float, target_rad: float) -> float:
    error = abs(float(current_rad) - float(target_rad))
    wrapped_error = abs((error + math.pi) % (2.0 * math.pi) - math.pi)
    return min(error, wrapped_error)


def _joint_errors_rad(current_positions: Sequence[Any], target_positions: Sequence[Any]) -> list[float]:
    current = [float(value) for value in current_positions]
    target = [float(value) for value in target_positions]
    return [
        _joint_angle_error_rad(current_value, target_value)
        for current_value, target_value in zip(current, target)
    ]


def _wait_for_joint_state_positions(
    rclpy_module: Any,
    node: Any,
    get_latest_positions: Any,
    *,
    timeout_sec: float,
) -> list[float] | None:
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while time.monotonic() < deadline:
        positions = get_latest_positions()
        if positions is not None:
            return positions
        rclpy_module.spin_once(node, timeout_sec=0.05)
    return get_latest_positions()


def _publish_waypoint_until_reached(
    rclpy_module: Any,
    node: Any,
    publisher: Any,
    get_latest_positions: Any,
    target_positions_rad: Sequence[Any],
    *,
    phase: str,
    tolerance_rad: float,
    timeout_sec: float,
    republish_interval_sec: float,
) -> dict[str, object]:
    target_positions = [float(value) for value in target_positions_rad]
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    interval = max(0.001, float(republish_interval_sec))
    next_publish_time = -float("inf")
    published_count = 0
    last_errors: list[float] = []
    last_positions: list[float] | None = None

    while True:
        now = time.monotonic()
        if published_count == 0 or now >= next_publish_time:
            _publish_joint_positions_once(publisher, target_positions)
            published_count += 1
            next_publish_time = now + interval

        rclpy_module.spin_once(node, timeout_sec=0.02)
        current_positions = get_latest_positions()
        if current_positions is not None:
            last_positions = [float(value) for value in current_positions]
            last_errors = _joint_errors_rad(last_positions, target_positions)
            if last_errors and all(error <= float(tolerance_rad) for error in last_errors):
                return {
                    "success": True,
                    "phase": str(phase),
                    "published_count": int(published_count),
                    "target_positions_rad": target_positions,
                    "target_positions_deg": [math.degrees(value) for value in target_positions],
                    "actual_positions_rad": last_positions,
                    "actual_positions_deg": [math.degrees(value) for value in last_positions],
                    "joint_errors_rad": last_errors,
                    "joint_errors_deg": [math.degrees(value) for value in last_errors],
                    "max_error_rad": max(last_errors),
                    "max_error_deg": math.degrees(max(last_errors)),
                    "message": "joint waypoint reached tolerance",
                }

        if time.monotonic() >= deadline:
            break

    return {
        "success": False,
        "phase": str(phase),
        "published_count": int(published_count),
        "target_positions_rad": target_positions,
        "target_positions_deg": [math.degrees(value) for value in target_positions],
        "actual_positions_rad": last_positions,
        "actual_positions_deg": None if last_positions is None else [math.degrees(value) for value in last_positions],
        "joint_errors_rad": last_errors,
        "joint_errors_deg": [math.degrees(value) for value in last_errors],
        "max_error_rad": max(last_errors) if last_errors else None,
        "max_error_deg": math.degrees(max(last_errors)) if last_errors else None,
        "message": "timed out waiting for joint waypoint tolerance",
    }


def _publish_joint_positions_for_duration(
    rclpy_module: Any,
    node: Any,
    publisher: Any,
    get_latest_positions: Any,
    target_positions_rad: Sequence[Any],
    *,
    phase: str,
    duration_sec: float,
    republish_interval_sec: float,
) -> dict[str, object]:
    target_positions = [float(value) for value in target_positions_rad]
    hold_sec = max(0.0, float(duration_sec))
    deadline = time.monotonic() + hold_sec
    interval = max(0.001, float(republish_interval_sec))
    next_publish_time = -float("inf")
    published_count = 0
    last_errors: list[float] = []
    last_positions: list[float] | None = None

    while True:
        now = time.monotonic()
        if published_count == 0 or now >= next_publish_time:
            _publish_joint_positions_once(publisher, target_positions)
            published_count += 1
            next_publish_time = now + interval

        rclpy_module.spin_once(node, timeout_sec=0.02)
        current_positions = get_latest_positions()
        if current_positions is not None:
            last_positions = [float(value) for value in current_positions]
            last_errors = _joint_errors_rad(last_positions, target_positions)

        if time.monotonic() >= deadline and published_count > 0:
            break

    return {
        "success": True,
        "phase": str(phase),
        "published_count": int(published_count),
        "hold_sec": float(hold_sec),
        "target_positions_rad": target_positions,
        "target_positions_deg": [math.degrees(value) for value in target_positions],
        "actual_positions_rad": last_positions,
        "actual_positions_deg": None if last_positions is None else [math.degrees(value) for value in last_positions],
        "joint_errors_rad": last_errors,
        "joint_errors_deg": [math.degrees(value) for value in last_errors],
        "max_error_rad": max(last_errors) if last_errors else None,
        "max_error_deg": math.degrees(max(last_errors)) if last_errors else None,
        "message": "joint command held for duration",
    }


def _run_direct_joint_interpolation_sequence(
    motion_solution: dict[str, object],
    *,
    direct_config: dict[str, object],
    planning_config: Any,
    arm_config: dict[str, object] | None,
    gripper_joint_index: int,
    gripper_open_rad: float,
    gripper_close_rad: float,
) -> dict[str, object]:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectoryPoint

    pregrasp_positions = _goal_joint_rad_from_solution(motion_solution)
    joint_count = len(pregrasp_positions)
    if joint_count <= 0:
        raise ValueError("Cannot direct-publish an empty joint solution.")
    if not all(math.isfinite(float(value)) for value in pregrasp_positions):
        raise ValueError("Cannot direct-publish non-finite joint positions.")

    gripper_index = int(gripper_joint_index)
    if not (0 <= gripper_index < joint_count):
        raise ValueError(
            f"gripper joint index {gripper_index} is outside joint vector length {joint_count}."
        )

    pregrasp_positions = [float(value) for value in pregrasp_positions]
    pregrasp_positions[gripper_index] = float(gripper_open_rad)
    close_positions = list(pregrasp_positions)
    close_positions[gripper_index] = float(gripper_close_rad)
    reset_positions = _direct_joint_reset_positions_rad(
        planning_config=planning_config,
        arm_config=arm_config,
        joint_count=joint_count,
    )

    arm_topic = _direct_joint_publish_topic_from_config(arm_config)
    joint_state_topic = _direct_joint_state_topic_from_config(arm_config)
    joint_state_groups = _direct_joint_state_groups_from_config(arm_config)
    waypoint_steps = _direct_joint_waypoint_steps_from_environment()
    joint_state_wait_sec = max(0.0, float(direct_config["joint_state_wait_sec"]))
    waypoint_timeout_sec = max(0.0, float(direct_config["joint_command_timeout_sec"]))
    tolerance_rad = max(0.0, float(direct_config["joint_command_tolerance_rad"]))
    republish_interval_sec = max(
        0.001,
        _direct_joint_publish_float_env(
            "APPROACH_AGENT_CAR_DIRECT_REPUBLISH_INTERVAL_SEC",
            float(direct_config.get("joint_command_republish_interval_sec", 0.0))
            or CAR_ARM_FINISH_DIRECT_REPUBLISH_INTERVAL_SEC,
        ),
    )
    after_gripper_close_delay_sec = max(0.0, float(direct_config["init_pose_delay_sec"]))
    init_settle_sec = max(
        0.0,
        _direct_joint_publish_float_env(
            "APPROACH_AGENT_CAR_DIRECT_INIT_POSE_SETTLE_SEC",
            CAR_ARM_FINISH_DIRECT_INIT_POSE_SETTLE_SEC,
        ),
    )

    owns_rclpy = False
    node = None
    latest_joint_positions: list[float] | None = None

    def get_latest_positions() -> list[float] | None:
        return None if latest_joint_positions is None else list(latest_joint_positions)

    def on_joint_state(msg: JointState) -> None:
        nonlocal latest_joint_positions
        mapped = _map_joint_state_positions(
            msg,
            joint_state_groups=joint_state_groups,
            joint_count=joint_count,
        )
        if mapped is not None:
            latest_joint_positions = mapped

    phases: list[dict[str, object]] = []
    try:
        if not rclpy.ok():
            rclpy.init(args=None)
            owns_rclpy = True
        node = Node(f"{CAR_ARM_FINISH_NODE_NAME}_direct")
        publisher = node.create_publisher(JointTrajectoryPoint, arm_topic, 10)
        node.create_subscription(JointState, joint_state_topic, on_joint_state, 10)
        time.sleep(0.2)

        start_positions = _wait_for_joint_state_positions(
            rclpy,
            node,
            get_latest_positions,
            timeout_sec=joint_state_wait_sec,
        )
        if start_positions is None:
            start_positions = list(reset_positions)
            start_source = "reset_pose_fallback_no_initial_joint_state"
        else:
            start_source = "joint_states"

        waypoints = _linear_interpolated_joint_positions(
            start_positions,
            pregrasp_positions,
            steps=waypoint_steps,
        )
        print(
            "[base_approach] car arm finish: publishing joint-space waypoints "
            f"steps={len(waypoints)} topic={arm_topic} joint_state_topic={joint_state_topic} "
            f"goal_deg={[round(math.degrees(value), 2) for value in pregrasp_positions]}",
            flush=True,
        )
        for waypoint_index, waypoint in enumerate(waypoints, start=1):
            result = _publish_waypoint_until_reached(
                rclpy,
                node,
                publisher,
                get_latest_positions,
                waypoint,
                phase=f"waypoint_{waypoint_index:02d}",
                tolerance_rad=tolerance_rad,
                timeout_sec=waypoint_timeout_sec,
                republish_interval_sec=republish_interval_sec,
            )
            phases.append(result)
            if not bool(result.get("success", False)):
                break

        target_move_success = bool(phases) and all(
            bool(phase.get("success", False)) for phase in phases
        )
        if target_move_success:
            close_result = _publish_joint_positions_for_duration(
                rclpy,
                node,
                publisher,
                get_latest_positions,
                close_positions,
                phase="close_gripper",
                duration_sec=after_gripper_close_delay_sec,
                republish_interval_sec=republish_interval_sec,
            )
            phases.append(close_result)
        else:
            phases.append(
                {
                    "success": False,
                    "skipped": True,
                    "phase": "close_gripper",
                    "published_count": 0,
                    "message": "close gripper skipped because a pregrasp waypoint did not reach tolerance",
                }
            )

        should_publish_reset = any(
            int(phase.get("published_count", 0) or 0) > 0
            for phase in phases
            if str(phase.get("phase", "")) != "init_pose"
        )
        if should_publish_reset:
            reset_result = _publish_joint_positions_for_duration(
                rclpy,
                node,
                publisher,
                get_latest_positions,
                reset_positions,
                phase="init_pose",
                duration_sec=init_settle_sec,
                republish_interval_sec=republish_interval_sec,
            )
            reset_result["delay_sec"] = float(after_gripper_close_delay_sec)
            reset_result["after_gripper_close_delay_sec"] = float(after_gripper_close_delay_sec)
            reset_result["settle_sec"] = float(init_settle_sec)
            reset_result["joint_state_tolerance_wait_skipped"] = True
            reset_result["message"] = (
                "init pose command republished for settle duration; skipped joint state tolerance wait"
            )
            phases.append(reset_result)
    finally:
        if node is not None:
            node.destroy_node()
        if owns_rclpy and rclpy.ok():
            rclpy.shutdown()

    target_move_success = bool(phases) and all(
        bool(phase.get("success", False))
        for phase in phases
        if str(phase.get("phase", "")).startswith("waypoint_")
    )
    close_phase = next((phase for phase in phases if phase.get("phase") == "close_gripper"), None)
    init_phase = next((phase for phase in phases if phase.get("phase") == "init_pose"), None)
    gripper_close_success = bool(close_phase and close_phase.get("success", False))
    init_pose_success = bool(init_phase and init_phase.get("success", False))
    success = bool(target_move_success and gripper_close_success and init_pose_success)
    failed_phase = next((phase for phase in phases if not bool(phase.get("success", False))), None)
    phase_summary = ", ".join(
        f"{phase['phase']}:{phase.get('published_count', 0)}pub:{'ok' if phase.get('success') else 'fail'}"
        for phase in phases
    )
    message = (
        f"direct joint interpolation {'completed' if success else 'failed'}; {phase_summary}"
    )
    if failed_phase is not None:
        message += f"; failed_phase={failed_phase.get('phase')}: {failed_phase.get('message')}"

    return {
        "success": success,
        "execution_model": CAR_ARM_FINISH_DIRECT_EXECUTION_MODEL,
        "published_joint_directly": True,
        "direct_joint_interpolation": True,
        "direct_joint_publish_topic": arm_topic,
        "direct_joint_state_topic": joint_state_topic,
        "direct_joint_state_groups": [list(group) for group in joint_state_groups],
        "direct_joint_start_source": start_source if 'start_source' in locals() else "unavailable",
        "direct_joint_waypoint_steps": int(waypoint_steps),
        "direct_joint_phases": phases,
        "direct_joint_pregrasp_positions_rad": pregrasp_positions,
        "direct_joint_pregrasp_positions_deg": [math.degrees(value) for value in pregrasp_positions],
        "direct_joint_close_positions_rad": close_positions,
        "direct_joint_close_positions_deg": [math.degrees(value) for value in close_positions],
        "direct_joint_reset_positions_rad": reset_positions,
        "direct_joint_reset_positions_deg": [math.degrees(value) for value in reset_positions],
        "joint_command_timeout_sec": float(waypoint_timeout_sec),
        "joint_command_tolerance_rad": float(tolerance_rad),
        "joint_command_tolerance_deg": math.degrees(float(tolerance_rad)),
        "joint_command_republish_interval_sec": float(republish_interval_sec),
        "joint_state_wait_sec": float(joint_state_wait_sec),
        "after_gripper_close_delay_sec": float(after_gripper_close_delay_sec),
        "gripper_close_hold_sec": float(after_gripper_close_delay_sec),
        "gripper_joint_index": int(gripper_index),
        "gripper_open_rad": float(gripper_open_rad),
        "gripper_open_deg": math.degrees(float(gripper_open_rad)),
        "gripper_close_rad": float(gripper_close_rad),
        "gripper_close_deg": math.degrees(float(gripper_close_rad)),
        "gripper_close_success": bool(gripper_close_success),
        "target_move_success": bool(target_move_success),
        "wrist_success": bool(target_move_success),
        "gripper_open_success": bool(target_move_success),
        "init_pose_success": bool(init_pose_success),
        "return_to_start_published": bool(init_pose_success),
        "continued_to_init_pose": bool(init_pose_success),
        "cube_z_distance_required_for_success": False,
        "cube_z_distance_verified": False,
        "cube_z_distance_success": False,
        "message": message,
    }


def _goal_joint_rad_from_solution(solution: dict[str, object]) -> list[float]:
    raw_rad = solution.get("ik_joint_solution_rad")
    if isinstance(raw_rad, list):
        return _float_sequence(raw_rad, label="solution.ik_joint_solution_rad")
    raw_deg = solution.get("ik_joint_solution_deg")
    if isinstance(raw_deg, list):
        return [math.radians(value) for value in _float_sequence(raw_deg, label="solution.ik_joint_solution_deg")]
    raise KeyError("solution needs ik_joint_solution_rad or ik_joint_solution_deg.")


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


def _target_position_xyz_from_record_or_solution(
    solution: dict[str, object],
    target_record: dict[str, object] | None,
) -> tuple[list[float], str]:
    if isinstance(target_record, dict) and target_record.get("target_pb") is not None:
        return _float_xyz(target_record["target_pb"], label="target_record.target_pb"), "target_record.target_pb"
    for key in ("target_pb", "target_position_pybullet_xyz", "final_ee_position_xyz"):
        if solution.get(key) is not None:
            return _float_xyz(solution[key], label=f"solution.{key}"), f"solution.{key}"
    raise KeyError("arrival-adjusted IK needs target_record.target_pb or solution.final_ee_position_xyz.")


def _arrived_base_pose_for_ik(
    solution: dict[str, object],
    *,
    planning_config: Any,
    arm_base_target: dict[str, object] | None,
) -> tuple[list[float], float, str, dict[str, object]]:
    yaw_compensation = (
        arm_base_target.get("yaw_compensation")
        if isinstance(arm_base_target, dict)
        else None
    )
    if isinstance(yaw_compensation, dict) and bool(yaw_compensation.get("applied", False)):
        final_xyz = _float_xyz(
            yaw_compensation["final_base_link_local_pb_xyz"],
            label="arm_base_target.yaw_compensation.final_base_link_local_pb_xyz",
        )
        final_yaw_rad = float(yaw_compensation["final_base_link_local_pb_yaw_rad"])
        if not math.isfinite(final_yaw_rad):
            raise ValueError("final_base_link_local_pb_yaw_rad must be finite.")
        return (
            final_xyz,
            final_yaw_rad,
            "nav_result.final_amcl_pose_actual_base_local_pb",
            {
                "arrival_adjusted_ik_nav_error_compensation_applied": True,
                "arrival_adjusted_ik_planned_base_xyz": yaw_compensation.get("planned_base_link_local_pb_xyz"),
                "arrival_adjusted_ik_final_base_xyz": final_xyz,
                "arrival_adjusted_ik_vehicle_position_error_xyz_m": yaw_compensation.get(
                    "vehicle_position_error_from_planned_xyz_m"
                ),
                "arrival_adjusted_ik_vehicle_position_error_xy_m": yaw_compensation.get(
                    "vehicle_position_error_from_planned_xy_m"
                ),
                "arrival_adjusted_ik_vehicle_position_error_norm_m": yaw_compensation.get(
                    "vehicle_position_error_from_planned_norm_m"
                ),
                "arrival_adjusted_ik_vehicle_yaw_error_rad": yaw_compensation.get(
                    "vehicle_yaw_error_from_planned_rad"
                ),
                "arrival_adjusted_ik_vehicle_yaw_error_deg": yaw_compensation.get(
                    "vehicle_yaw_error_from_planned_deg"
                ),
            },
        )

    base_xyz = _float_xyz(solution["pb_base_link_xyz"], label="solution.pb_base_link_xyz")
    base_xyz[2] = float(planning_config.initial_height)
    base_yaw_rad = float(solution["pb_base_link_yaw_rad"])
    if not math.isfinite(base_yaw_rad):
        raise ValueError("solution.pb_base_link_yaw_rad must be finite.")
    return (
        base_xyz,
        base_yaw_rad,
        "selected_solution.planned_base_local_pb",
        {
            "arrival_adjusted_ik_nav_error_compensation_applied": False,
            "arrival_adjusted_ik_planned_base_xyz": base_xyz,
            "arrival_adjusted_ik_final_base_xyz": None,
        },
    )


def _planning_pb_world_position_to_base_local_pb(
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
    return [
        float(local_x),
        float(local_y),
        float(planning_config.initial_height) + float(delta[2]),
    ]


def _urdf_search_paths(urdf_path: str) -> list[str]:
    urdf_dir = Path(urdf_path).resolve().parent
    candidates = [urdf_dir.parent, urdf_dir.parent.parent]
    result: list[str] = []
    for candidate in candidates:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in result:
                result.append(candidate_str)
    return result


def _controllable_joint_ids(p_module: Any, robot_id: int, expected_joint_count: int) -> list[int]:
    joint_ids: list[int] = []
    for joint_index in range(p_module.getNumJoints(robot_id)):
        joint_info = p_module.getJointInfo(robot_id, joint_index)
        joint_name = joint_info[1].decode("utf-8")
        joint_type = joint_info[2]
        if joint_type in (p_module.JOINT_REVOLUTE, p_module.JOINT_PRISMATIC) and joint_name != "Revolute 6":
            joint_ids.append(joint_index)
    return joint_ids[: int(expected_joint_count)]


def _recompute_arrival_adjusted_ik(
    target_position_base_xyz: Sequence[Any],
    *,
    planning_config: Any,
    arm_config: dict[str, object] | None,
    p_mod: Any | None,
    pybullet_data: Any | None,
) -> tuple[list[float], dict[str, object]]:
    if p_mod is None:
        import pybullet as p_mod  # type: ignore[no-redef]
    if pybullet_data is None:
        import pybullet_data as pybullet_data  # type: ignore[no-redef]

    target_position = _float_xyz(target_position_base_xyz, label="arrival_adjusted_ik_target_position_base_xyz")
    expected_joint_count = int((arm_config or {}).get("pybullet", {}).get("controllable_joints", len(planning_config.joint_reset_deg)))
    ik_max_iterations = int((arm_config or {}).get("pybullet", {}).get("ik_max_num_iterations", 200))
    ik_residual_threshold = float((arm_config or {}).get("pybullet", {}).get("ik_residual_threshold", 1e-5))
    time_step = float((arm_config or {}).get("pybullet", {}).get("time_step", 1e-3))

    client_id = p_mod.connect(p_mod.DIRECT)
    if client_id < 0:
        raise RuntimeError("PyBullet DIRECT unavailable for arrival-adjusted IK.")
    try:
        p_mod.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_mod.resetSimulation()
        p_mod.setGravity(0.0, 0.0, -9.8)
        p_mod.setTimeStep(time_step)
        p_mod.loadURDF("plane.urdf")
        for search_path in _urdf_search_paths(planning_config.urdf_path):
            p_mod.setAdditionalSearchPath(search_path)
        base_orientation_rad = [math.radians(value) for value in planning_config.base_orientation_euler_deg]
        robot_id = p_mod.loadURDF(
            planning_config.urdf_path,
            useFixedBase=True,
            basePosition=[0.0, 0.0, float(planning_config.initial_height)],
            baseOrientation=p_mod.getQuaternionFromEuler(base_orientation_rad),
        )
        joint_ids = _controllable_joint_ids(p_mod, robot_id, expected_joint_count)
        if len(joint_ids) != expected_joint_count:
            raise RuntimeError(
                f"Expected {expected_joint_count} controllable joints, found {len(joint_ids)}."
            )
        reset_positions_rad = [math.radians(float(value)) for value in planning_config.joint_reset_deg]
        for joint_id, joint_position in zip(joint_ids, reset_positions_rad):
            p_mod.resetJointState(robot_id, joint_id, targetValue=float(joint_position), targetVelocity=0.0)
        p_mod.performCollisionDetection()

        ik_solution = p_mod.calculateInverseKinematics(
            robot_id,
            int(planning_config.ee_link_index),
            targetPosition=target_position,
            maxNumIterations=ik_max_iterations,
            residualThreshold=ik_residual_threshold,
        )
        if len(ik_solution) < expected_joint_count:
            raise RuntimeError("PyBullet IK returned fewer joint values than required.")
        joint_positions = [float(value) for value in ik_solution[:expected_joint_count]]
        for joint_id, joint_position in zip(joint_ids, joint_positions):
            p_mod.resetJointState(robot_id, joint_id, targetValue=float(joint_position), targetVelocity=0.0)
        p_mod.performCollisionDetection()
        ee_state = p_mod.getLinkState(robot_id, int(planning_config.ee_link_index), computeForwardKinematics=True)
        final_position = [float(value) for value in ee_state[4]]
        error_xyz = [float(target - actual) for target, actual in zip(target_position, final_position)]
        position_error_m = float(np.linalg.norm(np.asarray(error_xyz, dtype=np.float64)))
        return joint_positions, {
            "arrival_adjusted_ik_recomputed": True,
            "arrival_adjusted_ik_target_position_base_xyz": target_position,
            "arrival_adjusted_ik_joint_solution_rad": joint_positions,
            "arrival_adjusted_ik_joint_solution_deg": [math.degrees(value) for value in joint_positions],
            "arrival_adjusted_ik_final_ee_position_xyz": final_position,
            "arrival_adjusted_ik_error_xyz_m": error_xyz,
            "arrival_adjusted_ik_position_error_m": position_error_m,
            "arrival_adjusted_ik_position_only": True,
            "arrival_adjusted_ik_ee_link_index": int(planning_config.ee_link_index),
        }
    finally:
        try:
            p_mod.disconnect(client_id)
        except Exception:
            pass


def _build_motion_solution(
    solution: dict[str, object],
    *,
    target_record: dict[str, object] | None,
    planning_config: Any,
    arm_config: dict[str, object] | None,
    arm_base_target: dict[str, object] | None,
    p_mod: Any | None,
    pybullet_data: Any | None,
) -> tuple[dict[str, object], dict[str, object]]:
    raw_goal_joint_positions = _goal_joint_rad_from_solution(solution)
    target_position_xyz, target_position_source = _target_position_xyz_from_record_or_solution(
        solution,
        target_record,
    )
    base_xyz, base_yaw_rad, base_pose_source, base_metadata = _arrived_base_pose_for_ik(
        solution,
        planning_config=planning_config,
        arm_base_target=arm_base_target,
    )
    target_position_base_xyz = _planning_pb_world_position_to_base_local_pb(
        target_position_xyz,
        base_xyz=base_xyz,
        base_yaw_rad=base_yaw_rad,
        planning_config=planning_config,
    )
    goal_joint_positions, ik_metadata = _recompute_arrival_adjusted_ik(
        target_position_base_xyz,
        planning_config=planning_config,
        arm_config=arm_config,
        p_mod=p_mod,
        pybullet_data=pybullet_data,
    )
    metadata: dict[str, object] = {
        "raw_goal_joint_positions_rad": [float(value) for value in raw_goal_joint_positions],
        "raw_goal_joint_positions_deg": [math.degrees(float(value)) for value in raw_goal_joint_positions],
        "arrival_adjusted_ik_target_position_source": target_position_source,
        "arrival_adjusted_ik_planning_world_target_xyz": target_position_xyz,
        "arrival_adjusted_ik_base_pose_source": base_pose_source,
        "arrival_adjusted_ik_base_xyz": base_xyz,
        "arrival_adjusted_ik_base_yaw_rad": float(base_yaw_rad),
        "arrival_adjusted_ik_base_yaw_deg": math.degrees(float(base_yaw_rad)),
        "selected_ik_joint_solution_preserved": False,
        "selected_ik_joint_solution_compensated_after_arrival": True,
        "selected_ik_wrist_angle_preserved": False,
        "selected_ik_preservation_reason": "arrival_adjusted_target_position_recomputed_ik_in_car_approach",
        "arm_base_target_applied": False,
        **base_metadata,
        **ik_metadata,
    }

    fallback_joint_index = int(os.getenv("APPROACH_AGENT_CAR_ARM_BASE_JOINT_INDEX", "0"))
    if 0 <= fallback_joint_index < len(goal_joint_positions):
        locked_joint_rad = float(goal_joint_positions[fallback_joint_index])
        metadata["locked_arm_base_joint"] = True
        metadata["locked_arm_base_joint_index"] = int(fallback_joint_index)
        metadata["locked_arm_base_joint_rad"] = float(locked_joint_rad)
        metadata["locked_arm_base_joint_deg"] = math.degrees(float(locked_joint_rad))
        metadata["locked_arm_base_joint_source"] = "arrival_adjusted_ik_joint_solution_rad"
    else:
        raise ValueError(
            f"Cannot read arm base joint {fallback_joint_index}; "
            f"goal vector length is {len(goal_joint_positions)}."
        )

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
    motion_solution["arrival_adjusted_ik_recomputed"] = True
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
