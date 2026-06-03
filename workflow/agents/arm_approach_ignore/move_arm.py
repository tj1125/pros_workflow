"""Publish Approach Agent arm motions through the ROS control packages.

This module intentionally keeps the interpolation on the agent side for now:
it expands a start/goal joint target into small joint-space points, then sends
each point as a structured ``arm_control_signal`` command.  The ROS control
packages remain the only owner of the final ``/robot_arm`` publisher.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_ARM_TOPIC = "/robot_arm"
DEFAULT_ARM_CONTROL_TOPIC = "arm_control_signal"
DEFAULT_COMMAND_PERIOD_SEC = 0.5
DEFAULT_HOLD_FINAL_COUNT = 3
DEFAULT_HOLD_FINAL_INTERVAL_SEC = 0.05
DEFAULT_INTERPOLATION_STEPS = 10
DEFAULT_PUBLISH_START = False
DEFAULT_WAIT_FOR_SUBSCRIBERS_SEC = 2.0
DEFAULT_CLOSE_GRIPPER_ON_ARRIVAL = True
DEFAULT_GRIPPER_JOINT_INDEX = 4
DEFAULT_GRIPPER_CLOSE_DEG = 10.0
DEFAULT_GRIPPER_CLOSE_DELAY_SEC = 0.2
DEFAULT_FORWARD_BEFORE_GRIPPER_CLOSE = True
DEFAULT_FORWARD_BEFORE_GRIPPER_CLOSE_SETTLE_SEC = DEFAULT_COMMAND_PERIOD_SEC
DEFAULT_RETURN_TO_START_AFTER_GRIPPER_CLOSE = False
DEFAULT_RETURN_TO_START_DELAY_SEC = 1
ARM_COMMAND_SET_JOINT_POSITIONS_RAD = "set_joint_positions_rad"
ARM_COMMAND_SOURCE = "approach_agent_move_arm"
EXECUTION_MODEL = "agent_interpolates_tools_step_targets"


@dataclass(frozen=True)
class ArmMoveConfig:
    """Runtime settings for agent-side joint interpolation publishing."""

    arm_topic: str = DEFAULT_ARM_TOPIC
    control_topic: str = DEFAULT_ARM_CONTROL_TOPIC
    interpolation_steps: int = DEFAULT_INTERPOLATION_STEPS
    command_period_sec: float = DEFAULT_COMMAND_PERIOD_SEC
    wait_for_subscribers_sec: float = DEFAULT_WAIT_FOR_SUBSCRIBERS_SEC
    hold_final_count: int = DEFAULT_HOLD_FINAL_COUNT
    hold_final_interval_sec: float = DEFAULT_HOLD_FINAL_INTERVAL_SEC
    publish_start: bool = DEFAULT_PUBLISH_START
    close_gripper_on_arrival: bool = DEFAULT_CLOSE_GRIPPER_ON_ARRIVAL
    gripper_joint_index: int = DEFAULT_GRIPPER_JOINT_INDEX
    gripper_close_rad: float = math.radians(DEFAULT_GRIPPER_CLOSE_DEG)
    gripper_close_delay_sec: float = DEFAULT_GRIPPER_CLOSE_DELAY_SEC
    forward_before_gripper_close: bool = DEFAULT_FORWARD_BEFORE_GRIPPER_CLOSE
    forward_before_gripper_close_settle_sec: float = DEFAULT_FORWARD_BEFORE_GRIPPER_CLOSE_SETTLE_SEC
    return_to_start_after_gripper_close: bool = DEFAULT_RETURN_TO_START_AFTER_GRIPPER_CLOSE
    return_to_start_delay_sec: float = DEFAULT_RETURN_TO_START_DELAY_SEC
    node_name: str = "approach_agent_move_arm"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_planner_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "pybullet_ompl.yaml"


def _default_arm_config_path() -> Path:
    return _repo_root() / "src" / "arm_control_pkg" / "config" / "arm_config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return payload


def _read_arm_topic_from_config(arm_config_path: Path | None = None) -> str:
    path = arm_config_path or _default_arm_config_path()
    try:
        payload = _load_yaml(path)
        topic = str(payload.get("global", {}).get("arm_topic", "")).strip()
    except Exception:
        topic = ""
    return topic or DEFAULT_ARM_TOPIC


def _validate_control_topic(topic: str | None) -> str:
    topic = (topic or DEFAULT_ARM_CONTROL_TOPIC).strip()
    if topic in {DEFAULT_ARM_TOPIC, DEFAULT_ARM_TOPIC.lstrip("/")}:
        raise ValueError(
            "move_arm control_topic must point to tools' arm_control_signal topic; "
            f"{DEFAULT_ARM_TOPIC} is published by arm_control_pkg."
        )
    return topic or DEFAULT_ARM_CONTROL_TOPIC


def _float_sequence(values: Sequence[Any], *, label: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{label} must not be empty.")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values.")
    return result


def _radians_from_degrees(values_deg: Sequence[Any], *, label: str) -> list[float]:
    return [math.radians(value) for value in _float_sequence(values_deg, label=label)]


def _env_flag(name: str, default: bool) -> bool:
    default_value = "1" if default else "0"
    return os.getenv(name, default_value).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }


def _joint_reset_rad_from_planning_config(planning_config: Any | None) -> list[float] | None:
    if planning_config is None:
        return None
    if isinstance(planning_config, dict):
        joint_reset_deg = planning_config.get("joint_reset_deg")
    else:
        joint_reset_deg = getattr(planning_config, "joint_reset_deg", None)
    if joint_reset_deg is None:
        return None
    return _radians_from_degrees(joint_reset_deg, label="planning_config.joint_reset_deg")


def _interpolation_steps_from_planning_config(planning_config: Any | None) -> int:
    # ``path_interpolation_states`` belongs to PyBullet/OMPL planning and is too
    # dense for ROS command publishing, so arm motion uses its own lightweight
    # default unless explicitly overridden by APPROACH_AGENT_ARM_INTERPOLATION_STEPS.
    return DEFAULT_INTERPOLATION_STEPS


def _load_default_joint_reset_rad(planner_config_path: Path | None = None) -> list[float]:
    path = planner_config_path or _default_planner_config_path()
    payload = _load_yaml(path)
    joint_reset_deg = payload.get("joint_reset_deg")
    if joint_reset_deg is None:
        raise KeyError(f"joint_reset_deg is missing from {path}")
    return _radians_from_degrees(joint_reset_deg, label=f"{path}.joint_reset_deg")


def _goal_joint_rad_from_solution(solution: dict[str, Any]) -> list[float]:
    raw_rad = solution.get("ik_joint_solution_rad")
    if raw_rad is not None:
        return _float_sequence(raw_rad, label="solution.ik_joint_solution_rad")

    raw_deg = solution.get("ik_joint_solution_deg")
    if raw_deg is not None:
        return _radians_from_degrees(raw_deg, label="solution.ik_joint_solution_deg")

    raise KeyError("solution needs ik_joint_solution_rad or ik_joint_solution_deg.")


def build_linear_joint_trajectory(
    start_joint_positions_rad: Sequence[Any],
    goal_joint_positions_rad: Sequence[Any],
    *,
    interpolation_steps: int,
    include_start: bool = True,
) -> list[list[float]]:
    """Return joint-space waypoints from ``start`` to ``goal`` in radians."""

    start = _float_sequence(start_joint_positions_rad, label="start_joint_positions_rad")
    goal = _float_sequence(goal_joint_positions_rad, label="goal_joint_positions_rad")
    if len(start) != len(goal):
        raise ValueError(
            "start and goal joint vectors must have the same length: "
            f"{len(start)} vs {len(goal)}."
        )

    step_count = max(1, int(interpolation_steps))
    first_index = 0 if include_start else 1
    return [
        [
            float(start_value + (goal_value - start_value) * (float(index) / float(step_count)))
            for start_value, goal_value in zip(start, goal)
        ]
        for index in range(first_index, step_count + 1)
    ]


def _make_set_joint_positions_command(point_rad: Sequence[float], elapsed_sec: float, sequence_index: int) -> Any:
    from std_msgs.msg import String

    msg = String()
    msg.data = json.dumps(
        {
            "command": ARM_COMMAND_SET_JOINT_POSITIONS_RAD,
            "positions": [float(value) for value in point_rad],
            "time_from_start_sec": float(elapsed_sec),
            "sequence_index": int(sequence_index),
            "source": ARM_COMMAND_SOURCE,
        },
        separators=(",", ":"),
    )
    return msg


def _wait_for_subscribers(node: Any, publisher: Any, topic: str, timeout_sec: float) -> None:
    import rclpy

    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while publisher.get_subscription_count() <= 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    subscriber_count = publisher.get_subscription_count()
    if subscriber_count <= 0:
        print(f"[move_arm] warning: publishing {topic} with no matched subscribers.", flush=True)
    else:
        print(f"[move_arm] matched {subscriber_count} subscriber(s) on {topic}.", flush=True)


def publish_joint_trajectory(
    trajectory_rad: Sequence[Sequence[Any]],
    *,
    config: ArmMoveConfig,
    post_arrival_joint_positions_rad: Sequence[Any] | None = None,
    post_arrival_joint_positions_sequence_rad: Sequence[Sequence[Any]] | None = None,
    return_trajectory_rad: Sequence[Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Publish pre-interpolated joint points to tools via ``arm_control_signal``."""

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    points = [
        _float_sequence(point, label=f"trajectory_rad[{index}]")
        for index, point in enumerate(trajectory_rad)
    ]
    if not points:
        raise ValueError("trajectory_rad must contain at least one point.")
    return_points = (
        [
            _float_sequence(point, label=f"return_trajectory_rad[{index}]")
            for index, point in enumerate(return_trajectory_rad)
        ]
        if return_trajectory_rad is not None
        else []
    )
    if post_arrival_joint_positions_sequence_rad is not None:
        post_arrival_points = [
            _float_sequence(point, label=f"post_arrival_joint_positions_sequence_rad[{index}]")
            for index, point in enumerate(post_arrival_joint_positions_sequence_rad)
        ]
    elif post_arrival_joint_positions_rad is not None:
        post_arrival_points = [
            _float_sequence(
                post_arrival_joint_positions_rad,
                label="post_arrival_joint_positions_rad",
            )
        ]
    else:
        post_arrival_points = []
    for index, post_arrival_point in enumerate(post_arrival_points):
        if len(post_arrival_point) != len(points[-1]):
            raise ValueError(
                "post-arrival joint positions must have the same length as trajectory points: "
                f"post_arrival[{index}]={len(post_arrival_point)} vs trajectory={len(points[-1])}."
            )

    control_topic = _validate_control_topic(config.control_topic)
    owns_rclpy = False
    node = None
    publisher = None
    published_count = 0
    try:
        if not rclpy.ok():
            rclpy.init(args=None)
            owns_rclpy = True

        node = Node(config.node_name)
        publisher = node.create_publisher(String, control_topic, 10)
        _wait_for_subscribers(
            node,
            publisher,
            control_topic,
            config.wait_for_subscribers_sec,
        )

        period = max(0.0, float(config.command_period_sec))
        print(
            "[move_arm] publishing interpolated joint targets: "
            f"control_topic={control_topic} tools_arm_topic={config.arm_topic} "
            f"points={len(points)} period={period:.3f}s",
            flush=True,
        )
        for index, point in enumerate(points):
            publisher.publish(
                _make_set_joint_positions_command(
                    point,
                    index * period,
                    index,
                )
            )
            published_count += 1
            rclpy.spin_once(node, timeout_sec=0.0)
            if period > 0.0 and index < len(points) - 1:
                time.sleep(period)

        final_point = points[-1]
        hold_count = max(0, int(config.hold_final_count))
        hold_interval = max(0.0, float(config.hold_final_interval_sec))
        for hold_index in range(hold_count):
            sequence_index = len(points) + hold_index
            publisher.publish(
                _make_set_joint_positions_command(
                    final_point,
                    ((len(points) - 1) * period) + ((hold_index + 1) * hold_interval),
                    sequence_index,
                )
            )
            published_count += 1
            rclpy.spin_once(node, timeout_sec=0.0)
            if hold_interval > 0.0 and hold_index < hold_count - 1:
                time.sleep(hold_interval)

        post_arrival_forward_published = False
        post_arrival_forward_point: list[float] | None = None
        post_arrival_forward_points: list[list[float]] = []
        close_base_point = final_point
        close_base_elapsed = ((len(points) - 1) * period) + (hold_count * hold_interval)
        if config.forward_before_gripper_close and post_arrival_points:
            settle_sec = max(0.0, float(config.forward_before_gripper_close_settle_sec))
            for post_arrival_index, post_arrival_point in enumerate(post_arrival_points):
                sequence_index = len(points) + hold_count + post_arrival_index
                publisher.publish(
                    _make_set_joint_positions_command(
                        post_arrival_point,
                        close_base_elapsed,
                        sequence_index,
                    )
                )
                published_count += 1
                post_arrival_forward_published = True
                post_arrival_forward_point = list(post_arrival_point)
                post_arrival_forward_points.append(post_arrival_forward_point)
                close_base_point = post_arrival_forward_point
                rclpy.spin_once(node, timeout_sec=0.0)
                if settle_sec > 0.0:
                    time.sleep(settle_sec)
                    close_base_elapsed += settle_sec

        gripper_close_published = False
        gripper_close_point: list[float] | None = None
        close_delay = 0.0
        if config.close_gripper_on_arrival:
            gripper_index = int(config.gripper_joint_index)
            if 0 <= gripper_index < len(close_base_point):
                close_delay = max(0.0, float(config.gripper_close_delay_sec))
                if close_delay > 0.0:
                    time.sleep(close_delay)
                gripper_close_point = list(close_base_point)
                gripper_close_point[gripper_index] = float(config.gripper_close_rad)
                sequence_index = len(points) + hold_count + len(post_arrival_forward_points)
                publisher.publish(
                    _make_set_joint_positions_command(
                        gripper_close_point,
                        close_base_elapsed + close_delay,
                        sequence_index,
                    )
                )
                published_count += 1
                gripper_close_published = True
                rclpy.spin_once(node, timeout_sec=0.0)
            else:
                print(
                    "[move_arm] warning: gripper close skipped because "
                    f"joint index {gripper_index} is outside final point length {len(close_base_point)}.",
                    flush=True,
                )

        return_to_start_published = False
        return_final_point: list[float] | None = None
        if (
            config.return_to_start_after_gripper_close
            and gripper_close_published
            and return_points
        ):
            return_delay = max(0.0, float(config.return_to_start_delay_sec))
            if return_delay > 0.0:
                time.sleep(return_delay)
            return_base_elapsed = close_base_elapsed + close_delay + return_delay
            for return_index, return_point in enumerate(return_points):
                sequence_index = (
                    len(points)
                    + hold_count
                    + len(post_arrival_forward_points)
                    + 1
                    + return_index
                )
                publisher.publish(
                    _make_set_joint_positions_command(
                        return_point,
                        return_base_elapsed + (return_index * period),
                        sequence_index,
                    )
                )
                published_count += 1
                return_to_start_published = True
                return_final_point = list(return_point)
                rclpy.spin_once(node, timeout_sec=0.0)
                if period > 0.0 and return_index < len(return_points) - 1:
                    time.sleep(period)

        print(
            "[move_arm] interpolated joint targets published: "
            f"commands={published_count} final_deg="
            f"{[round(math.degrees(value), 2) for value in final_point]} "
            f"forward_before_close={post_arrival_forward_published} "
            f"gripper_closed={gripper_close_published} "
            f"returned_to_start={return_to_start_published}",
            flush=True,
        )
        return {
            "success": True,
            "topic": control_topic,
            "control_topic": control_topic,
            "tools_arm_topic": config.arm_topic,
            "execution_model": EXECUTION_MODEL,
            "arm_command": ARM_COMMAND_SET_JOINT_POSITIONS_RAD,
            "trajectory_point_count": len(points),
            "commands_published": published_count,
            "final_joint_positions_rad": final_point,
            "final_joint_positions_deg": [math.degrees(value) for value in final_point],
            "post_arrival_forward_published": post_arrival_forward_published,
            "post_arrival_forward_point_count": len(post_arrival_forward_points),
            "post_arrival_forward_joint_positions_sequence_rad": post_arrival_forward_points,
            "post_arrival_forward_joint_positions_sequence_deg": [
                [math.degrees(value) for value in point] for point in post_arrival_forward_points
            ],
            "post_arrival_forward_joint_positions_rad": post_arrival_forward_point,
            "post_arrival_forward_joint_positions_deg": (
                None
                if post_arrival_forward_point is None
                else [math.degrees(value) for value in post_arrival_forward_point]
            ),
            "post_arrival_forward_settle_sec": float(config.forward_before_gripper_close_settle_sec),
            "gripper_close_published": gripper_close_published,
            "gripper_joint_index": int(config.gripper_joint_index),
            "gripper_close_rad": float(config.gripper_close_rad),
            "gripper_close_deg": math.degrees(float(config.gripper_close_rad)),
            "gripper_closed_joint_positions_rad": gripper_close_point,
            "gripper_closed_joint_positions_deg": (
                None if gripper_close_point is None else [math.degrees(value) for value in gripper_close_point]
            ),
            "return_to_start_published": return_to_start_published,
            "return_trajectory_point_count": len(return_points),
            "return_final_joint_positions_rad": return_final_point,
            "return_final_joint_positions_deg": (
                None if return_final_point is None else [math.degrees(value) for value in return_final_point]
            ),
        }
    finally:
        if node is not None:
            node.destroy_node()
        if owns_rclpy and rclpy.ok():
            rclpy.shutdown()


def config_from_environment(
    *,
    planning_config: Any | None = None,
    arm_topic: str | None = None,
    control_topic: str | None = None,
    arm_config_path: Path | None = None,
) -> ArmMoveConfig:
    topic = (
        arm_topic
        or os.getenv("APPROACH_AGENT_ARM_TOPIC", "").strip()
        or _read_arm_topic_from_config(arm_config_path)
    )
    command_topic = (
        control_topic
        or os.getenv("APPROACH_AGENT_ARM_CONTROL_TOPIC", "").strip()
        or DEFAULT_ARM_CONTROL_TOPIC
    )
    command_topic = _validate_control_topic(command_topic)
    default_steps = _interpolation_steps_from_planning_config(planning_config)
    interpolation_steps = int(os.getenv("APPROACH_AGENT_ARM_INTERPOLATION_STEPS", str(default_steps)))
    command_period_sec = float(os.getenv("APPROACH_AGENT_ARM_COMMAND_PERIOD_SEC", str(DEFAULT_COMMAND_PERIOD_SEC)))
    wait_for_subscribers_sec = float(
        os.getenv("APPROACH_AGENT_ARM_WAIT_FOR_SUBSCRIBERS_SEC", str(DEFAULT_WAIT_FOR_SUBSCRIBERS_SEC))
    )
    hold_final_count = int(os.getenv("APPROACH_AGENT_ARM_HOLD_FINAL_COUNT", str(DEFAULT_HOLD_FINAL_COUNT)))
    hold_final_interval_sec = float(
        os.getenv("APPROACH_AGENT_ARM_HOLD_FINAL_INTERVAL_SEC", str(DEFAULT_HOLD_FINAL_INTERVAL_SEC))
    )
    publish_start = _env_flag("APPROACH_AGENT_ARM_PUBLISH_START", DEFAULT_PUBLISH_START)
    close_gripper_on_arrival = _env_flag(
        "APPROACH_AGENT_ARM_CLOSE_GRIPPER_ON_ARRIVAL",
        DEFAULT_CLOSE_GRIPPER_ON_ARRIVAL,
    )
    gripper_joint_index = int(os.getenv("APPROACH_AGENT_ARM_GRIPPER_JOINT_INDEX", str(DEFAULT_GRIPPER_JOINT_INDEX)))
    gripper_close_deg = float(os.getenv("APPROACH_AGENT_ARM_GRIPPER_CLOSE_DEG", str(DEFAULT_GRIPPER_CLOSE_DEG)))
    gripper_close_delay_sec = float(
        os.getenv("APPROACH_AGENT_ARM_GRIPPER_CLOSE_DELAY_SEC", str(DEFAULT_GRIPPER_CLOSE_DELAY_SEC))
    )
    forward_before_gripper_close = _env_flag(
        "APPROACH_AGENT_ARM_FORWARD_BEFORE_GRIPPER_CLOSE",
        DEFAULT_FORWARD_BEFORE_GRIPPER_CLOSE,
    )
    forward_before_gripper_close_settle_sec = float(
        os.getenv(
            "APPROACH_AGENT_ARM_FORWARD_BEFORE_GRIPPER_CLOSE_SETTLE_SEC",
            str(DEFAULT_FORWARD_BEFORE_GRIPPER_CLOSE_SETTLE_SEC),
        )
    )
    return_to_start_after_gripper_close = _env_flag(
        "APPROACH_AGENT_ARM_RETURN_TO_START_AFTER_GRIPPER_CLOSE",
        DEFAULT_RETURN_TO_START_AFTER_GRIPPER_CLOSE,
    )
    return_to_start_delay_sec = float(
        os.getenv("APPROACH_AGENT_ARM_RETURN_TO_START_DELAY_SEC", str(DEFAULT_RETURN_TO_START_DELAY_SEC))
    )
    return ArmMoveConfig(
        arm_topic=topic,
        control_topic=command_topic,
        interpolation_steps=max(1, interpolation_steps),
        command_period_sec=max(0.0, command_period_sec),
        wait_for_subscribers_sec=max(0.0, wait_for_subscribers_sec),
        hold_final_count=max(0, hold_final_count),
        hold_final_interval_sec=max(0.0, hold_final_interval_sec),
        publish_start=publish_start,
        close_gripper_on_arrival=close_gripper_on_arrival,
        gripper_joint_index=gripper_joint_index,
        gripper_close_rad=math.radians(gripper_close_deg),
        gripper_close_delay_sec=max(0.0, gripper_close_delay_sec),
        forward_before_gripper_close=forward_before_gripper_close,
        forward_before_gripper_close_settle_sec=max(0.0, forward_before_gripper_close_settle_sec),
        return_to_start_after_gripper_close=return_to_start_after_gripper_close,
        return_to_start_delay_sec=max(0.0, return_to_start_delay_sec),
    )


def move_arm_for_solution(
    solution: dict[str, Any],
    *,
    planning_config: Any | None = None,
    config: ArmMoveConfig | None = None,
    start_joint_positions_rad: Sequence[Any] | None = None,
    planner_config_path: Path | None = None,
) -> dict[str, Any]:
    """Interpolate from the current/planning start joints to the selected IK goal."""

    goal_joint_positions_rad = _goal_joint_rad_from_solution(solution)
    start_joint_positions = (
        _float_sequence(start_joint_positions_rad, label="start_joint_positions_rad")
        if start_joint_positions_rad is not None
        else _joint_reset_rad_from_planning_config(planning_config)
    )
    if start_joint_positions is None:
        start_joint_positions = _load_default_joint_reset_rad(planner_config_path)

    move_config = config or config_from_environment(planning_config=planning_config)
    trajectory = build_linear_joint_trajectory(
        start_joint_positions,
        goal_joint_positions_rad,
        interpolation_steps=move_config.interpolation_steps,
        include_start=move_config.publish_start,
    )
    return_trajectory: list[list[float]] | None = None
    if move_config.return_to_start_after_gripper_close:
        gripper_index = int(move_config.gripper_joint_index)
        if 0 <= gripper_index < len(goal_joint_positions_rad) and gripper_index < len(start_joint_positions):
            closed_goal = list(goal_joint_positions_rad)
            closed_start = list(start_joint_positions)
            closed_goal[gripper_index] = float(move_config.gripper_close_rad)
            closed_start[gripper_index] = float(move_config.gripper_close_rad)
            return_trajectory = build_linear_joint_trajectory(
                closed_goal,
                closed_start,
                interpolation_steps=move_config.interpolation_steps,
                include_start=False,
            )
    post_arrival_joint_positions_rad = solution.get("post_arrival_forward_joint_positions_rad")
    post_arrival_joint_positions_sequence_rad = (
        solution.get("post_arrival_joint_positions_sequence_rad")
        or solution.get("post_arrival_forward_joint_positions_sequence_rad")
    )
    return publish_joint_trajectory(
        trajectory,
        config=move_config,
        post_arrival_joint_positions_rad=post_arrival_joint_positions_rad,
        post_arrival_joint_positions_sequence_rad=post_arrival_joint_positions_sequence_rad,
        return_trajectory_rad=return_trajectory,
    )


def _load_solution(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("selected_solution", "solution", "goal_pose_solution"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if "ik_joint_solution_rad" in payload or "ik_joint_solution_deg" in payload:
            return payload
    raise ValueError(f"No selected IK solution found in {path}")


def _parse_float_list(raw_values: Sequence[str] | None) -> list[float] | None:
    if not raw_values:
        return None
    values: list[float] = []
    for raw_value in raw_values:
        for part in str(raw_value).replace(",", " ").split():
            values.append(float(part))
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish selected Approach_Agent IK motion as high-level arm_control_signal "
            "commands for arm_control_pkg to execute on /robot_arm."
        )
    )
    parser.add_argument("--solution-json", type=Path, default=None)
    parser.add_argument("--goal-rad", nargs="*", default=None)
    parser.add_argument("--goal-deg", nargs="*", default=None)
    parser.add_argument("--start-rad", nargs="*", default=None)
    parser.add_argument("--start-deg", nargs="*", default=None)
    parser.add_argument("--planner-config", type=Path, default=_default_planner_config_path())
    parser.add_argument("--arm-config", type=Path, default=_default_arm_config_path())
    parser.add_argument("--control-topic", "--topic", dest="control_topic", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--period-sec", type=float, default=None)
    parser.add_argument("--wait-subscribers-sec", type=float, default=None)
    parser.add_argument("--hold-final-count", type=int, default=None)
    parser.add_argument("--hold-final-interval-sec", type=float, default=None)
    parser.add_argument("--publish-start", action="store_true")
    parser.add_argument("--no-publish-start", action="store_true")
    parser.add_argument("--no-close-gripper", action="store_true")
    parser.add_argument("--gripper-joint-index", type=int, default=None)
    parser.add_argument("--gripper-close-deg", type=float, default=None)
    parser.add_argument("--gripper-close-delay-sec", type=float, default=None)
    parser.add_argument("--return-to-start", action="store_true")
    parser.add_argument("--no-return-to-start", action="store_true")
    parser.add_argument("--return-to-start-delay-sec", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    goal_rad = _parse_float_list(args.goal_rad)
    goal_deg = _parse_float_list(args.goal_deg)
    if args.solution_json is not None:
        solution = _load_solution(args.solution_json)
    elif goal_rad is not None:
        solution = {"ik_joint_solution_rad": goal_rad}
    elif goal_deg is not None:
        solution = {"ik_joint_solution_deg": goal_deg}
    else:
        raise SystemExit("Provide --solution-json, --goal-rad, or --goal-deg.")

    planner_payload = _load_yaml(args.planner_config)
    env_config = config_from_environment(
        planning_config=planner_payload,
        control_topic=args.control_topic,
        arm_config_path=args.arm_config,
    )
    config = ArmMoveConfig(
        arm_topic=env_config.arm_topic,
        control_topic=env_config.control_topic,
        interpolation_steps=max(1, args.steps if args.steps is not None else env_config.interpolation_steps),
        command_period_sec=max(
            0.0,
            args.period_sec if args.period_sec is not None else env_config.command_period_sec,
        ),
        wait_for_subscribers_sec=max(
            0.0,
            (
                args.wait_subscribers_sec
                if args.wait_subscribers_sec is not None
                else env_config.wait_for_subscribers_sec
            ),
        ),
        hold_final_count=max(
            0,
            args.hold_final_count if args.hold_final_count is not None else env_config.hold_final_count,
        ),
        hold_final_interval_sec=max(
            0.0,
            (
                args.hold_final_interval_sec
                if args.hold_final_interval_sec is not None
                else env_config.hold_final_interval_sec
            ),
        ),
        publish_start=(args.publish_start or env_config.publish_start) and not args.no_publish_start,
        close_gripper_on_arrival=env_config.close_gripper_on_arrival and not args.no_close_gripper,
        gripper_joint_index=(
            args.gripper_joint_index
            if args.gripper_joint_index is not None
            else env_config.gripper_joint_index
        ),
        gripper_close_rad=math.radians(
            args.gripper_close_deg
            if args.gripper_close_deg is not None
            else math.degrees(env_config.gripper_close_rad)
        ),
        gripper_close_delay_sec=max(
            0.0,
            (
                args.gripper_close_delay_sec
                if args.gripper_close_delay_sec is not None
                else env_config.gripper_close_delay_sec
            ),
        ),
        forward_before_gripper_close=env_config.forward_before_gripper_close,
        forward_before_gripper_close_settle_sec=env_config.forward_before_gripper_close_settle_sec,
        return_to_start_after_gripper_close=(
            (args.return_to_start or env_config.return_to_start_after_gripper_close)
            and not args.no_return_to_start
        ),
        return_to_start_delay_sec=max(
            0.0,
            (
                args.return_to_start_delay_sec
                if args.return_to_start_delay_sec is not None
                else env_config.return_to_start_delay_sec
            ),
        ),
    )

    start_rad = _parse_float_list(args.start_rad)
    start_deg = _parse_float_list(args.start_deg)
    if start_rad is not None and start_deg is not None:
        raise SystemExit("Use only one of --start-rad or --start-deg.")
    start_joint_positions_rad = (
        start_rad
        if start_rad is not None
        else None if start_deg is None else [math.radians(value) for value in start_deg]
    )
    move_arm_for_solution(
        solution,
        planning_config=planner_payload,
        config=config,
        start_joint_positions_rad=start_joint_positions_rad,
        planner_config_path=args.planner_config,
    )


if __name__ == "__main__":
    main()
