"""Joint command publishing and feedback helpers for car approach arm motion."""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import numpy as np

from .debug_log import debug_stage
from .src.geometry import coordinate_transforms as coord


DEFAULT_ARM_TOPIC = "/robot_arm"
DEFAULT_JOINT_STATE_TOPIC = "/joint_states"
DEFAULT_JOINT_STATE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("base_link2_v2_1",),
    ("link3_v1_1",),
    ("link4_v1_1",),
    ("robot_ver7_link_4_v1_1",),
    ("robot_ver7_grap1_2_v1_1", "robot_ver7_grap2_2_v1_1"),
)


def _float_sequence(values: Sequence[Any], *, label: str) -> list[float]:
    result = [float(value) for value in values]
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain finite values.")
    return result


def _linear_interpolated_joint_positions(start_positions_rad: Sequence[Any], goal_positions_rad: Sequence[Any], *, steps: int) -> list[list[float]]:
    start = np.asarray(_float_sequence(start_positions_rad, label="start_positions_rad"), dtype=np.float64)
    goal = np.asarray(_float_sequence(goal_positions_rad, label="goal_positions_rad"), dtype=np.float64)
    if start.shape != goal.shape:
        raise ValueError("start_positions_rad and goal_positions_rad must have the same length.")
    count = max(1, int(steps))
    return [((1.0 - alpha) * start + alpha * goal).astype(float).tolist() for alpha in (i / count for i in range(1, count + 1))]


def _publisher_subscription_count(publisher: Any) -> int | None:
    try:
        return int(publisher.get_subscription_count())
    except Exception:
        return None


def _wait_for_publisher_subscription(
    rclpy_module: Any,
    node: Any,
    publisher: Any,
    *,
    timeout_sec: float,
) -> int | None:
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    last_count = _publisher_subscription_count(publisher)
    while time.monotonic() < deadline:
        if last_count is None or last_count > 0:
            return last_count
        rclpy_module.spin_once(node, timeout_sec=0.05)
        last_count = _publisher_subscription_count(publisher)
    return last_count


def _publish_joint_positions_once(publisher: Any, joint_positions_rad: Sequence[Any]) -> int | None:
    from trajectory_msgs.msg import JointTrajectoryPoint

    positions = [float(value) for value in joint_positions_rad]
    message = JointTrajectoryPoint()
    message.positions = positions
    message.velocities = [0.0] * len(positions)
    message.accelerations = [0.0] * len(positions)
    message.effort = [0.0] * len(positions)
    message.time_from_start.sec = 0
    message.time_from_start.nanosec = 0
    subscription_count = _publisher_subscription_count(publisher)
    publisher.publish(message)
    return subscription_count


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
    ignored_joint_indices: Sequence[int] = (),
    early_stop_reader: Any = None,
    early_stop_threshold_m: float = 0.0,
    early_stop_since_time_sec: float | None = None,
    early_stop_poll_interval_sec: float = 0.02,
) -> dict[str, object]:
    target = [float(value) for value in target_positions_rad]
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    interval = max(0.001, float(republish_interval_sec))
    next_publish_at = -float("inf")
    published_count = 0
    last_positions: list[float] | None = None
    last_errors: list[float] = []
    last_subscription_count: int | None = _publisher_subscription_count(publisher)
    stop_threshold = max(0.0, float(early_stop_threshold_m))
    poll_interval = max(0.001, float(early_stop_poll_interval_sec))
    last_poll_at = -float("inf")
    while time.monotonic() < deadline:
        now = time.monotonic()
        if published_count == 0 or now >= next_publish_at:
            last_subscription_count = _publish_joint_positions_once(publisher, target)
            published_count += 1
            next_publish_at = now + interval
        rclpy_module.spin_once(node, timeout_sec=0.02)
        current = get_latest_positions()
        if current is not None:
            last_positions = [float(value) for value in current]
            last_errors = _joint_errors_rad(last_positions, target, ignored_joint_indices=ignored_joint_indices)
            if last_errors and all(error <= float(tolerance_rad) for error in last_errors):
                return _joint_phase_result(
                    True,
                    phase,
                    published_count,
                    target,
                    last_positions,
                    last_errors,
                    "joint waypoint reached tolerance",
                    ignored_joint_indices=ignored_joint_indices,
                    publisher_subscription_count=last_subscription_count,
                )
        if early_stop_reader is not None and stop_threshold > 0.0 and now - last_poll_at >= poll_interval:
            last_poll_at = now
            distance_m = _read_fresh_scalar(early_stop_reader, since_time_sec=early_stop_since_time_sec)
            if distance_m is not None and distance_m < stop_threshold:
                result = _joint_phase_result(
                    True,
                    phase,
                    published_count,
                    target,
                    last_positions,
                    last_errors,
                    "joint waypoint early-stopped by cube_z_distance",
                    ignored_joint_indices=ignored_joint_indices,
                    publisher_subscription_count=last_subscription_count,
                )
                result.update(
                    {
                        "early_stopped_by_cube_z_distance": True,
                        "early_stop_distance_m": float(distance_m),
                        "early_stop_threshold_m": float(stop_threshold),
                    }
                )
                return result
    return _joint_phase_result(
        False,
        phase,
        published_count,
        target,
        last_positions,
        last_errors,
        "timed out waiting for joint waypoint tolerance",
        ignored_joint_indices=ignored_joint_indices,
        publisher_subscription_count=last_subscription_count,
    )



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
    ignored_joint_indices: Sequence[int] = (),
) -> dict[str, object]:
    target = [float(value) for value in target_positions_rad]
    deadline = time.monotonic() + max(0.0, float(duration_sec))
    interval = max(0.001, float(republish_interval_sec))
    next_publish_at = -float("inf")
    published_count = 0
    last_positions: list[float] | None = None
    last_errors: list[float] = []
    last_subscription_count: int | None = _publisher_subscription_count(publisher)
    while True:
        now = time.monotonic()
        if published_count == 0 or now >= next_publish_at:
            last_subscription_count = _publish_joint_positions_once(publisher, target)
            published_count += 1
            next_publish_at = now + interval
        rclpy_module.spin_once(node, timeout_sec=0.02)
        current = get_latest_positions()
        if current is not None:
            last_positions = [float(value) for value in current]
            last_errors = _joint_errors_rad(last_positions, target, ignored_joint_indices=ignored_joint_indices)
        if time.monotonic() >= deadline and published_count > 0:
            break
    result = _joint_phase_result(
        True,
        phase,
        published_count,
        target,
        last_positions,
        last_errors,
        "joint command held for duration",
        ignored_joint_indices=ignored_joint_indices,
        publisher_subscription_count=last_subscription_count,
    )
    result["hold_sec"] = float(duration_sec)
    return result



def _publish_until_joint_leaves_position(
    rclpy_module: Any,
    node: Any,
    publisher: Any,
    get_latest_positions: Any,
    target_positions_rad: Sequence[Any],
    *,
    phase: str,
    joint_index: int,
    reference_position_rad: float,
    leave_tolerance_rad: float,
    timeout_sec: float,
    republish_interval_sec: float,
    ignored_joint_indices: Sequence[int] = (),
) -> dict[str, object]:
    target = [float(value) for value in target_positions_rad]
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    interval = max(0.001, float(republish_interval_sec))
    next_publish_at = -float("inf")
    published_count = 0
    last_positions: list[float] | None = None
    last_errors: list[float] = []
    last_subscription_count: int | None = _publisher_subscription_count(publisher)
    ref = float(reference_position_rad)
    joint_index = int(joint_index)
    tolerance = max(0.0, float(leave_tolerance_rad))
    left_reference = False
    while time.monotonic() < deadline:
        now = time.monotonic()
        if published_count == 0 or now >= next_publish_at:
            last_subscription_count = _publish_joint_positions_once(publisher, target)
            published_count += 1
            next_publish_at = now + interval
        rclpy_module.spin_once(node, timeout_sec=0.02)
        current = get_latest_positions()
        if current is None:
            continue
        last_positions = [float(value) for value in current]
        last_errors = _joint_errors_rad(last_positions, target, ignored_joint_indices=ignored_joint_indices)
        if 0 <= joint_index < len(last_positions):
            gripper_error_from_open = coord.joint_angle_error_rad(last_positions[joint_index], ref)
            if gripper_error_from_open > tolerance:
                left_reference = True
                result = _joint_phase_result(
                    True,
                    phase,
                    published_count,
                    target,
                    last_positions,
                    last_errors,
                    "joint left reference position",
                    ignored_joint_indices=ignored_joint_indices,
                    publisher_subscription_count=last_subscription_count,
                )
                result.update(
                    {
                        "reference_position_rad": ref,
                        "reference_position_deg": coord.rad_to_deg(ref),
                        "reference_joint_index": joint_index,
                        "reference_error_rad": float(gripper_error_from_open),
                        "reference_error_deg": coord.rad_to_deg(gripper_error_from_open),
                        "condition_met": True,
                        "timed_out": False,
                    }
                )
                return result
    result = _joint_phase_result(
        True,
        phase,
        published_count,
        target,
        last_positions,
        last_errors,
        "timed out waiting for joint to leave reference position; continuing",
        ignored_joint_indices=ignored_joint_indices,
        publisher_subscription_count=last_subscription_count,
    )
    reference_error = None
    if last_positions is not None and 0 <= joint_index < len(last_positions):
        reference_error = coord.joint_angle_error_rad(last_positions[joint_index], ref)
    result.update(
        {
            "reference_position_rad": ref,
            "reference_position_deg": coord.rad_to_deg(ref),
            "reference_joint_index": joint_index,
            "reference_error_rad": reference_error,
            "reference_error_deg": None if reference_error is None else coord.rad_to_deg(reference_error),
            "condition_met": left_reference,
            "timed_out": True,
        }
    )
    return result


def _read_fresh_scalar(reader: Any, *, since_time_sec: float | None) -> float | None:
    try:
        reading = reader(since_time_sec=since_time_sec)
    except TypeError:
        reading = reader()
    except Exception:
        return None
    if reading is None:
        return None
    value = reading[0] if isinstance(reading, tuple) else reading
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _debug_joint_phase_result(
    stage_message: str,
    result: dict[str, object],
    *,
    waypoint_index: int | None = None,
    waypoint_total: int | None = None,
) -> None:
    debug_stage(
        "move_arm",
        stage_message,
        phase=result.get("phase"),
        success=result.get("success"),
        waypoint_index=waypoint_index,
        waypoint_total=waypoint_total,
        published_count=result.get("published_count"),
        target_deg=_round_float_sequence(result.get("target_positions_deg")),
        actual_deg=_round_float_sequence(result.get("actual_positions_deg")),
        error_deg=_round_float_sequence(result.get("joint_errors_deg")),
        max_error_deg=_round_float(result.get("max_error_deg")),
        ignored_joint_indices=result.get("ignored_joint_indices"),
        publisher_subscription_count=result.get("publisher_subscription_count"),
        message=result.get("message"),
    )


def _round_float(value: Any, *, digits: int = 3) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return round(numeric, digits)


def _round_float_sequence(values: Any, *, digits: int = 3) -> list[float] | None:
    if values is None or isinstance(values, (str, bytes)):
        return None
    try:
        result = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in result):
        return None
    return [round(value, digits) for value in result]


def _rad_sequence_to_deg_or_none(values: Sequence[Any] | None) -> list[float] | None:
    return None if values is None else coord.rad_sequence_to_deg(values)


def _joint_phase_result(
    success: bool,
    phase: str,
    published_count: int,
    target: list[float],
    actual: list[float] | None,
    errors: list[float],
    message: str,
    *,
    ignored_joint_indices: Sequence[int] = (),
    publisher_subscription_count: int | None = None,
) -> dict[str, object]:
    return {
        "success": bool(success),
        "phase": phase,
        "published_count": int(published_count),
        "target_positions_rad": target,
        "target_positions_deg": coord.rad_sequence_to_deg(target),
        "actual_positions_rad": actual,
        "actual_positions_deg": None if actual is None else coord.rad_sequence_to_deg(actual),
        "joint_errors_rad": errors,
        "joint_errors_deg": coord.rad_sequence_to_deg(errors),
        "max_error_rad": max(errors) if errors else None,
        "max_error_deg": coord.rad_to_deg(max(errors)) if errors else None,
        "ignored_joint_indices": [int(index) for index in ignored_joint_indices],
        "publisher_subscription_count": publisher_subscription_count,
        "message": message,
    }


def _wait_for_joint_state_positions(rclpy_module: Any, node: Any, get_latest_positions: Any, *, timeout_sec: float) -> list[float] | None:
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while time.monotonic() < deadline:
        positions = get_latest_positions()
        if positions is not None:
            return positions
        rclpy_module.spin_once(node, timeout_sec=0.05)
    return get_latest_positions()


def _map_joint_state_positions(
    msg: Any,
    *,
    joint_state_groups: Sequence[tuple[str, ...]],
    joint_count: int,
    ignored_joint_indices: Sequence[int] = (),
) -> list[float] | None:
    try:
        raw_positions = [float(value) for value in msg.position]
    except Exception:
        return None
    count = int(joint_count)
    ignored = {int(index) for index in ignored_joint_indices}
    if len(raw_positions) < count and any(index not in ignored for index in range(len(raw_positions), count)):
        return None
    if joint_state_groups and getattr(msg, "name", None):
        name_to_position: dict[str, float] = {}
        for index, name in enumerate(msg.name):
            if index >= len(raw_positions):
                break
            name_to_position.setdefault(str(name), raw_positions[index])
        mapped: list[float] = []
        for index, group in enumerate(joint_state_groups[:count]):
            if all(name in name_to_position for name in group):
                values = [name_to_position[name] for name in group]
                mapped.append(sum(values) / float(len(values)))
            elif index in ignored:
                mapped.append(raw_positions[index] if index < len(raw_positions) else 0.0)
            else:
                return None
        for index in range(len(mapped), count):
            if index in ignored:
                mapped.append(raw_positions[index] if index < len(raw_positions) else 0.0)
            else:
                return None
    else:
        mapped = [
            raw_positions[index] if index < len(raw_positions) else 0.0
            for index in range(count)
        ]
    return mapped if len(mapped) >= count and all(math.isfinite(v) for v in mapped) else None


def _joint_errors_rad(
    current_positions: Sequence[Any],
    target_positions: Sequence[Any],
    *,
    ignored_joint_indices: Sequence[int] = (),
) -> list[float]:
    ignored = {int(index) for index in ignored_joint_indices}
    return [
        coord.joint_angle_error_rad(float(cur), float(tgt))
        for index, (cur, tgt) in enumerate(zip(current_positions, target_positions))
        if index not in ignored
    ]

