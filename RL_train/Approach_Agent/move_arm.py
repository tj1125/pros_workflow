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
DEFAULT_COMMAND_PERIOD_SEC = 0.05
DEFAULT_HOLD_FINAL_COUNT = 3
DEFAULT_HOLD_FINAL_INTERVAL_SEC = 0.05
DEFAULT_INTERPOLATION_STEPS = 96
DEFAULT_WAIT_FOR_SUBSCRIBERS_SEC = 2.0


@dataclass(frozen=True)
class ArmMoveConfig:
    arm_topic: str = DEFAULT_ARM_TOPIC
    interpolation_steps: int = DEFAULT_INTERPOLATION_STEPS
    command_period_sec: float = DEFAULT_COMMAND_PERIOD_SEC
    wait_for_subscribers_sec: float = DEFAULT_WAIT_FOR_SUBSCRIBERS_SEC
    hold_final_count: int = DEFAULT_HOLD_FINAL_COUNT
    hold_final_interval_sec: float = DEFAULT_HOLD_FINAL_INTERVAL_SEC
    publish_start: bool = True
    node_name: str = "approach_agent_move_arm"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_planner_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "pybullet_ompl.yaml"


def _default_arm_config_path() -> Path:
    return _repo_root() / "tools" / "car_control" / "src" / "arm_control_pkg" / "config" / "arm_config.yaml"


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


def _float_sequence(values: Sequence[Any], *, label: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{label} must not be empty.")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values.")
    return result


def _radians_from_degrees(values_deg: Sequence[Any], *, label: str) -> list[float]:
    return [math.radians(value) for value in _float_sequence(values_deg, label=label)]


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
    if planning_config is None:
        return DEFAULT_INTERPOLATION_STEPS
    if isinstance(planning_config, dict):
        raw_value = planning_config.get("path_interpolation_states", DEFAULT_INTERPOLATION_STEPS)
    else:
        raw_value = getattr(planning_config, "path_interpolation_states", DEFAULT_INTERPOLATION_STEPS)
    return max(1, int(raw_value))


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


def _duration_from_seconds(duration_msg: Any, seconds: float) -> None:
    total_nanoseconds = max(0, int(round(float(seconds) * 1_000_000_000.0)))
    duration_msg.sec = int(total_nanoseconds // 1_000_000_000)
    duration_msg.nanosec = int(total_nanoseconds % 1_000_000_000)


def _make_joint_trajectory_point(point_rad: Sequence[float], elapsed_sec: float) -> Any:
    from trajectory_msgs.msg import JointTrajectoryPoint

    msg = JointTrajectoryPoint()
    msg.positions = [float(value) for value in point_rad]
    msg.velocities = []
    msg.accelerations = []
    msg.effort = []
    _duration_from_seconds(msg.time_from_start, elapsed_sec)
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
) -> dict[str, Any]:
    import rclpy
    from rclpy.node import Node
    from trajectory_msgs.msg import JointTrajectoryPoint

    points = [
        _float_sequence(point, label=f"trajectory_rad[{index}]")
        for index, point in enumerate(trajectory_rad)
    ]
    if not points:
        raise ValueError("trajectory_rad must contain at least one point.")

    owns_rclpy = False
    node = None
    publisher = None
    published_count = 0
    try:
        if not rclpy.ok():
            rclpy.init(args=None)
            owns_rclpy = True

        node = Node(config.node_name)
        publisher = node.create_publisher(JointTrajectoryPoint, config.arm_topic, 10)
        _wait_for_subscribers(
            node,
            publisher,
            config.arm_topic,
            config.wait_for_subscribers_sec,
        )

        period = max(0.0, float(config.command_period_sec))
        print(
            "[move_arm] publishing arm trajectory: "
            f"topic={config.arm_topic} points={len(points)} period={period:.3f}s",
            flush=True,
        )
        for index, point in enumerate(points):
            publisher.publish(_make_joint_trajectory_point(point, index * period))
            published_count += 1
            rclpy.spin_once(node, timeout_sec=0.0)
            if period > 0.0 and index < len(points) - 1:
                time.sleep(period)

        final_point = points[-1]
        hold_count = max(0, int(config.hold_final_count))
        hold_interval = max(0.0, float(config.hold_final_interval_sec))
        for hold_index in range(hold_count):
            publisher.publish(
                _make_joint_trajectory_point(
                    final_point,
                    (len(points) + hold_index) * max(period, hold_interval),
                )
            )
            published_count += 1
            rclpy.spin_once(node, timeout_sec=0.0)
            if hold_interval > 0.0 and hold_index < hold_count - 1:
                time.sleep(hold_interval)

        print(
            "[move_arm] arm trajectory published: "
            f"commands={published_count} final_deg="
            f"{[round(math.degrees(value), 2) for value in final_point]}",
            flush=True,
        )
        return {
            "success": True,
            "topic": config.arm_topic,
            "trajectory_point_count": len(points),
            "commands_published": published_count,
            "final_joint_positions_rad": final_point,
            "final_joint_positions_deg": [math.degrees(value) for value in final_point],
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
    arm_config_path: Path | None = None,
) -> ArmMoveConfig:
    topic = (
        arm_topic
        or os.getenv("APPROACH_AGENT_ARM_TOPIC", "").strip()
        or _read_arm_topic_from_config(arm_config_path)
    )
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
    publish_start = os.getenv("APPROACH_AGENT_ARM_PUBLISH_START", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }
    return ArmMoveConfig(
        arm_topic=topic,
        interpolation_steps=max(1, interpolation_steps),
        command_period_sec=max(0.0, command_period_sec),
        wait_for_subscribers_sec=max(0.0, wait_for_subscribers_sec),
        hold_final_count=max(0, hold_final_count),
        hold_final_interval_sec=max(0.0, hold_final_interval_sec),
        publish_start=publish_start,
    )


def move_arm_for_solution(
    solution: dict[str, Any],
    *,
    planning_config: Any | None = None,
    config: ArmMoveConfig | None = None,
    start_joint_positions_rad: Sequence[Any] | None = None,
    planner_config_path: Path | None = None,
) -> dict[str, Any]:
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
    return publish_joint_trajectory(trajectory, config=move_config)


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
        description="Publish the selected Approach_Agent IK trajectory to Unity on /robot_arm."
    )
    parser.add_argument("--solution-json", type=Path, default=None)
    parser.add_argument("--goal-rad", nargs="*", default=None)
    parser.add_argument("--goal-deg", nargs="*", default=None)
    parser.add_argument("--start-rad", nargs="*", default=None)
    parser.add_argument("--start-deg", nargs="*", default=None)
    parser.add_argument("--planner-config", type=Path, default=_default_planner_config_path())
    parser.add_argument("--arm-config", type=Path, default=_default_arm_config_path())
    parser.add_argument("--topic", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--period-sec", type=float, default=None)
    parser.add_argument("--wait-subscribers-sec", type=float, default=None)
    parser.add_argument("--hold-final-count", type=int, default=None)
    parser.add_argument("--hold-final-interval-sec", type=float, default=None)
    parser.add_argument("--no-publish-start", action="store_true")
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
        arm_topic=args.topic,
        arm_config_path=args.arm_config,
    )
    config = ArmMoveConfig(
        arm_topic=env_config.arm_topic,
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
        publish_start=env_config.publish_start and not args.no_publish_start,
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
