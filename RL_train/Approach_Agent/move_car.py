from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any


DEFAULT_INITIAL_POSE_TOPIC = "/initialpose"
DEFAULT_FRAME_ID = "map"
DEFAULT_AMCL_TOPIC = "/amcl_pose"
DEFAULT_FRONT_WHEEL_TOPIC = "car_C_front_wheel"
DEFAULT_REAR_WHEEL_TOPIC = "car_C_rear_wheel"

RULE_ACTION_MAPPINGS: dict[str, tuple[float, float, float, float]] = {
    "FORWARD_SLOW": (4.0, 4.0, 4.0, 4.0),
    "COUNTERCLOCKWISE_ROTATION_SLOW": (-5.85, 6.5, -7.8, 7.15),
    "CLOCKWISE_ROTATION_SLOW": (6.5, -5.85, 7.15, -7.8),
    "STOP": (0.0, 0.0, 0.0, 0.0),
}


@dataclass(frozen=True)
class GoalPose2D:
    x: float
    y: float
    yaw_rad: float
    z: float = 0.0


@dataclass(frozen=True)
class RuleNavigationConfig:
    amcl_topic: str = DEFAULT_AMCL_TOPIC
    initial_pose_topic: str = DEFAULT_INITIAL_POSE_TOPIC
    initial_pose_frame_id: str = DEFAULT_FRAME_ID
    initial_pose_publish_count: int = 5
    initial_pose_interval_sec: float = 0.1
    initial_pose_wait_for_subscribers_sec: float = 2.0
    front_wheel_topic: str = DEFAULT_FRONT_WHEEL_TOPIC
    rear_wheel_topic: str = DEFAULT_REAR_WHEEL_TOPIC
    xy_tolerance_m: float = 0.03
    face_target_yaw_tolerance_rad: float = 0.08
    drive_heading_tolerance_rad: float = 0.14
    final_yaw_tolerance_rad: float = 0.08
    slow_approach_distance_m: float = 0.12
    command_period_sec: float = 0.1
    amcl_wait_timeout_sec: float = 5.0
    amcl_stale_timeout_sec: float = 1.0
    max_duration_sec: float = 120.0
    stop_repeat: int = 5
    stop_interval_sec: float = 0.03
    log_interval_sec: float = 1.0


def _wrap_angle_rad(angle_rad: float) -> float:
    return float((float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi)


def _yaw_from_pose_like(pose: Any) -> float:
    if isinstance(pose, dict):
        if "yaw_rad" in pose:
            return _wrap_angle_rad(float(pose["yaw_rad"]))
        if "yaw" in pose:
            return _wrap_angle_rad(float(pose["yaw"]))
        if "yaw_deg" in pose:
            return _wrap_angle_rad(math.radians(float(pose["yaw_deg"])))
        if "qz" in pose and "qw" in pose:
            return _wrap_angle_rad(2.0 * math.atan2(float(pose["qz"]), float(pose["qw"])))
        raise KeyError("pose dict needs yaw_rad, yaw, yaw_deg, or qz/qw.")

    if hasattr(pose, "yaw_rad"):
        return _wrap_angle_rad(float(getattr(pose, "yaw_rad")))
    if hasattr(pose, "yaw"):
        return _wrap_angle_rad(float(getattr(pose, "yaw")))
    raise AttributeError("pose object needs yaw_rad or yaw.")


def goal_pose_from_ros_map_pose(pose: Any, *, z: float = 0.0) -> GoalPose2D:
    if pose is None:
        raise ValueError("ROS map pose is None.")
    if isinstance(pose, dict):
        return GoalPose2D(
            x=float(pose["x"]),
            y=float(pose["y"]),
            z=float(pose.get("z", z)),
            yaw_rad=_yaw_from_pose_like(pose),
        )
    return GoalPose2D(
        x=float(getattr(pose, "x")),
        y=float(getattr(pose, "y")),
        z=float(getattr(pose, "z", z)),
        yaw_rad=_yaw_from_pose_like(pose),
    )


def goal_pose_from_solution(
    solution: dict[str, Any],
    *,
    prefer_amcl_pose: bool = True,
) -> GoalPose2D:
    pose_keys = (
        ("ros_map_amcl_pose", "ros_map_base_link_pose")
        if prefer_amcl_pose
        else ("ros_map_base_link_pose", "ros_map_amcl_pose")
    )
    last_error: Exception | None = None
    for key in pose_keys:
        pose = solution.get(key)
        if pose is None:
            continue
        try:
            return goal_pose_from_ros_map_pose(pose)
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise ValueError(f"Selected solution has no usable ROS map goal pose: {last_error}") from last_error
    raise ValueError(f"Selected solution has none of {pose_keys}.")


def goal_pose_to_nav_dict(goal_pose: GoalPose2D) -> dict[str, float]:
    yaw = _wrap_angle_rad(goal_pose.yaw_rad)
    return {
        "x": float(goal_pose.x),
        "y": float(goal_pose.y),
        "z": float(goal_pose.z),
        "qx": 0.0,
        "qy": 0.0,
        "qz": float(math.sin(yaw / 2.0)),
        "qw": float(math.cos(yaw / 2.0)),
        "yaw": float(yaw),
    }


def goal_pose_dict_from_solution(
    solution: dict[str, Any],
    *,
    prefer_amcl_pose: bool = True,
) -> dict[str, float]:
    return goal_pose_to_nav_dict(
        goal_pose_from_solution(solution, prefer_amcl_pose=prefer_amcl_pose)
    )


def _goal_pose_dict_from_any(
    pose: Any,
    *,
    prefer_amcl_pose: bool = True,
) -> dict[str, float]:
    if isinstance(pose, dict) and ("ros_map_amcl_pose" in pose or "ros_map_base_link_pose" in pose):
        return goal_pose_dict_from_solution(pose, prefer_amcl_pose=prefer_amcl_pose)
    if isinstance(pose, dict) and {"x", "y", "qz", "qw"}.issubset(pose.keys()):
        return {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "z": float(pose.get("z", 0.0)),
            "qx": float(pose.get("qx", 0.0)),
            "qy": float(pose.get("qy", 0.0)),
            "qz": float(pose["qz"]),
            "qw": float(pose["qw"]),
            "yaw": _yaw_from_pose_like(pose),
        }
    return goal_pose_to_nav_dict(goal_pose_from_ros_map_pose(pose))


def _initial_pose_dict_from_any(pose: Any | None) -> dict[str, Any] | None:
    if pose is None:
        return None
    if isinstance(pose, dict):
        if {"x", "y"}.issubset(pose.keys()):
            if {"qx", "qy", "qz", "qw"}.issubset(pose.keys()):
                qx = float(pose["qx"])
                qy = float(pose["qy"])
                qz = float(pose["qz"])
                qw = float(pose["qw"])
            else:
                yaw = _yaw_from_pose_like(pose)
                qx = 0.0
                qy = 0.0
                qz = math.sin(yaw / 2.0)
                qw = math.cos(yaw / 2.0)
            return {
                "x": float(pose["x"]),
                "y": float(pose["y"]),
                "z": float(pose.get("z", 0.0)),
                "qx": qx,
                "qy": qy,
                "qz": qz,
                "qw": qw,
                "covariance": pose.get("covariance"),
            }
        if "position_xyz" in pose and "orientation_xyzw" in pose:
            position_xyz = pose["position_xyz"]
            orientation_xyzw = pose["orientation_xyzw"]
            return {
                "x": float(position_xyz[0]),
                "y": float(position_xyz[1]),
                "z": float(position_xyz[2]),
                "qx": float(orientation_xyzw[0]),
                "qy": float(orientation_xyzw[1]),
                "qz": float(orientation_xyzw[2]),
                "qw": float(orientation_xyzw[3]),
                "covariance": pose.get("covariance"),
            }

    if hasattr(pose, "position_xyz") and hasattr(pose, "orientation_xyzw"):
        position_xyz = getattr(pose, "position_xyz")
        orientation_xyzw = getattr(pose, "orientation_xyzw")
        return {
            "x": float(position_xyz[0]),
            "y": float(position_xyz[1]),
            "z": float(position_xyz[2]),
            "qx": float(orientation_xyzw[0]),
            "qy": float(orientation_xyzw[1]),
            "qz": float(orientation_xyzw[2]),
            "qw": float(orientation_xyzw[3]),
            "covariance": getattr(pose, "covariance", None),
        }

    ros_pose = getattr(getattr(pose, "pose", None), "pose", None)
    if ros_pose is not None:
        position = ros_pose.position
        orientation = ros_pose.orientation
        return {
            "x": float(position.x),
            "y": float(position.y),
            "z": float(position.z),
            "qx": float(orientation.x),
            "qy": float(orientation.y),
            "qz": float(orientation.z),
            "qw": float(orientation.w),
            "covariance": getattr(getattr(pose, "pose", None), "covariance", None),
        }

    raise ValueError("initial_pose must be a dict, AmclPoseSnapshot, or PoseWithCovarianceStamped-like object.")


def _wait_for_subscribers(node: Any, publisher: Any, topic: str, timeout_sec: float) -> None:
    wait_deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while publisher.get_subscription_count() <= 0 and time.monotonic() < wait_deadline:
        import rclpy

        rclpy.spin_once(node, timeout_sec=0.05)

    subscriber_count = publisher.get_subscription_count()
    if subscriber_count <= 0:
        print(f"[move_car] warning: publishing {topic} with no matched subscribers.", flush=True)
    else:
        print(f"[move_car] matched {subscriber_count} subscriber(s) on {topic}.", flush=True)


def _pose_from_amcl_msg(amcl_msg: Any) -> GoalPose2D:
    pose = amcl_msg.pose.pose
    return GoalPose2D(
        x=float(pose.position.x),
        y=float(pose.position.y),
        z=float(pose.position.z),
        yaw_rad=_yaw_from_pose_like(
            {
                "qz": float(pose.orientation.z),
                "qw": float(pose.orientation.w),
            }
        ),
    )


def _action_velocities(action: str) -> tuple[float, float, float, float]:
    try:
        return RULE_ACTION_MAPPINGS[action]
    except KeyError as exc:
        raise ValueError(f"Unknown rule navigation action: {action}") from exc


def _publish_wheel_action(
    front_wheel_publisher: Any,
    rear_wheel_publisher: Any,
    action: str,
) -> None:
    from std_msgs.msg import Float32MultiArray

    velocities = _action_velocities(action)
    front_msg = Float32MultiArray()
    rear_msg = Float32MultiArray()
    front_msg.data = [float(velocities[0]), float(velocities[1])]
    rear_msg.data = [float(velocities[2]), float(velocities[3])]
    front_wheel_publisher.publish(front_msg)
    rear_wheel_publisher.publish(rear_msg)


def _publish_stop_burst(
    front_wheel_publisher: Any,
    rear_wheel_publisher: Any,
    *,
    repeat: int,
    interval_sec: float,
) -> None:
    for index in range(max(1, int(repeat))):
        _publish_wheel_action(front_wheel_publisher, rear_wheel_publisher, "STOP")
        if index < max(1, int(repeat)) - 1:
            time.sleep(max(0.0, float(interval_sec)))


def _rotation_action_for_error(yaw_error_rad: float, *, final_alignment: bool = False) -> str:
    if yaw_error_rad < 0.0:
        return "CLOCKWISE_ROTATION_SLOW"
    return "COUNTERCLOCKWISE_ROTATION_SLOW"


def _distance_to_target(current_pose: GoalPose2D, target_pose: dict[str, float]) -> float:
    return math.hypot(float(target_pose["x"]) - current_pose.x, float(target_pose["y"]) - current_pose.y)


def _target_heading_error(current_pose: GoalPose2D, target_pose: dict[str, float]) -> float:
    target_heading = math.atan2(float(target_pose["y"]) - current_pose.y, float(target_pose["x"]) - current_pose.x)
    return _wrap_angle_rad(target_heading - current_pose.yaw_rad)


def drive_to_pose_by_rule(
    pose: Any,
    *,
    prefer_amcl_pose: bool = True,
    initial_pose: Any | None = None,
    config: RuleNavigationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RuleNavigationConfig()
    target_pose = _goal_pose_dict_from_any(pose, prefer_amcl_pose=prefer_amcl_pose)
    initial_pose_payload = _initial_pose_dict_from_any(initial_pose) if initial_pose is not None else None

    import rclpy
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Float32MultiArray

    owns_rclpy = False
    node = None
    front_wheel_publisher = None
    rear_wheel_publisher = None
    latest_amcl_msg: PoseWithCovarianceStamped | None = None
    latest_amcl_received_at = 0.0

    def _amcl_callback(msg: PoseWithCovarianceStamped) -> None:
        nonlocal latest_amcl_msg, latest_amcl_received_at
        latest_amcl_msg = msg
        latest_amcl_received_at = time.monotonic()

    try:
        if not rclpy.ok():
            rclpy.init(args=None)
            owns_rclpy = True

        node = Node("approach_agent_rule_navigator")
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        front_wheel_publisher = node.create_publisher(
            Float32MultiArray,
            cfg.front_wheel_topic,
            10,
        )
        rear_wheel_publisher = node.create_publisher(
            Float32MultiArray,
            cfg.rear_wheel_topic,
            10,
        )
        initial_pose_publisher = (
            node.create_publisher(PoseWithCovarianceStamped, cfg.initial_pose_topic, qos)
            if initial_pose_payload is not None and int(cfg.initial_pose_publish_count) > 0
            else None
        )
        amcl_subscription = node.create_subscription(
            PoseWithCovarianceStamped,
            cfg.amcl_topic,
            _amcl_callback,
            10,
        )

        def _make_initial_pose_stamped() -> PoseWithCovarianceStamped:
            if initial_pose_payload is None:
                raise RuntimeError("initial_pose_payload is not available.")
            msg = PoseWithCovarianceStamped()
            msg.header.frame_id = cfg.initial_pose_frame_id
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.pose.pose.position.x = float(initial_pose_payload["x"])
            msg.pose.pose.position.y = float(initial_pose_payload["y"])
            msg.pose.pose.position.z = float(initial_pose_payload["z"])
            msg.pose.pose.orientation.x = float(initial_pose_payload["qx"])
            msg.pose.pose.orientation.y = float(initial_pose_payload["qy"])
            msg.pose.pose.orientation.z = float(initial_pose_payload["qz"])
            msg.pose.pose.orientation.w = float(initial_pose_payload["qw"])
            covariance = initial_pose_payload.get("covariance")
            if covariance is not None and len(covariance) == 36:
                msg.pose.covariance = [float(value) for value in covariance]
            return msg

        def _wait_for_fresh_amcl(timeout_sec: float) -> GoalPose2D | None:
            deadline = time.monotonic() + max(0.0, float(timeout_sec))
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                if latest_amcl_msg is None:
                    continue
                if time.monotonic() - latest_amcl_received_at <= max(0.1, cfg.amcl_stale_timeout_sec):
                    return _pose_from_amcl_msg(latest_amcl_msg)
            return None

        def _spin_until_next_command(next_publish_at: float) -> None:
            while rclpy.ok() and time.monotonic() < next_publish_at:
                rclpy.spin_once(node, timeout_sec=min(0.05, max(0.0, next_publish_at - time.monotonic())))

        def _current_pose_or_wait() -> GoalPose2D | None:
            if latest_amcl_msg is not None and time.monotonic() - latest_amcl_received_at <= cfg.amcl_stale_timeout_sec:
                return _pose_from_amcl_msg(latest_amcl_msg)
            _publish_stop_burst(
                front_wheel_publisher,
                rear_wheel_publisher,
                repeat=cfg.stop_repeat,
                interval_sec=cfg.stop_interval_sec,
            )
            print("[move_car] waiting for fresh /amcl_pose before continuing rule navigation.", flush=True)
            return _wait_for_fresh_amcl(cfg.amcl_wait_timeout_sec)

        def _run_phase(
            phase_name: str,
            choose_action,
            is_done,
            *,
            timeout_deadline: float | None,
        ) -> tuple[bool, GoalPose2D | None]:
            next_log_at = 0.0
            while rclpy.ok():
                if timeout_deadline is not None and time.monotonic() >= timeout_deadline:
                    return False, None
                current_pose = _current_pose_or_wait()
                if current_pose is None:
                    return False, None
                distance_m = _distance_to_target(current_pose, target_pose)
                heading_error = _target_heading_error(current_pose, target_pose) if distance_m > 1e-6 else 0.0
                final_yaw_error = _wrap_angle_rad(float(target_pose["yaw"]) - current_pose.yaw_rad)
                if time.monotonic() >= next_log_at:
                    print(
                        f"[move_car] rule_nav {phase_name}: "
                        f"current=({current_pose.x:.3f}, {current_pose.y:.3f}, yaw={current_pose.yaw_rad:.3f}) "
                        f"target=({target_pose['x']:.3f}, {target_pose['y']:.3f}, yaw={target_pose['yaw']:.3f}) "
                        f"dist={distance_m:.3f}m "
                        f"heading_err={heading_error:.3f}rad "
                        f"final_yaw_err={final_yaw_error:.3f}rad",
                        flush=True,
                    )
                    next_log_at = time.monotonic() + max(0.2, cfg.log_interval_sec)
                if is_done(current_pose, distance_m, heading_error, final_yaw_error):
                    _publish_stop_burst(
                        front_wheel_publisher,
                        rear_wheel_publisher,
                        repeat=cfg.stop_repeat,
                        interval_sec=cfg.stop_interval_sec,
                    )
                    return True, current_pose
                action = choose_action(current_pose, distance_m, heading_error, final_yaw_error)
                _publish_wheel_action(front_wheel_publisher, rear_wheel_publisher, action)
                next_publish_at = time.monotonic() + max(0.02, float(cfg.command_period_sec))
                _spin_until_next_command(next_publish_at)
            return False, None

        print(
            "[move_car] rule navigation target: "
            f"x={target_pose['x']:.3f} y={target_pose['y']:.3f} yaw={target_pose['yaw']:.3f}. "
            "No /goal_pose or Nav2 topics will be used.",
            flush=True,
        )
        if initial_pose_payload is None:
            print("[move_car] /initialpose skipped: no capture-time /amcl_pose snapshot.", flush=True)
        elif initial_pose_publisher is not None:
            _wait_for_subscribers(
                node,
                initial_pose_publisher,
                cfg.initial_pose_topic,
                cfg.initial_pose_wait_for_subscribers_sec,
            )
            initial_count = max(1, int(cfg.initial_pose_publish_count))
            initial_interval = max(0.0, float(cfg.initial_pose_interval_sec))
            for index in range(initial_count):
                initial_pose_publisher.publish(_make_initial_pose_stamped())
                published_count = index + 1
                if published_count <= 3 or published_count % 10 == 0:
                    print(
                        f"[move_car] published {cfg.initial_pose_topic} #{published_count}: "
                        "source=capture-time /amcl_pose "
                        f"x={initial_pose_payload['x']:.3f} "
                        f"y={initial_pose_payload['y']:.3f} "
                        f"subscribers={initial_pose_publisher.get_subscription_count()}",
                        flush=True,
                    )
                rclpy.spin_once(node, timeout_sec=0.01)
                if index < initial_count - 1 and initial_interval > 0.0:
                    time.sleep(initial_interval)

        first_pose = _wait_for_fresh_amcl(cfg.amcl_wait_timeout_sec)
        if first_pose is None:
            _publish_stop_burst(
                front_wheel_publisher,
                rear_wheel_publisher,
                repeat=cfg.stop_repeat,
                interval_sec=cfg.stop_interval_sec,
            )
            return {
                "success": False,
                "phase": "wait_amcl",
                "target_pose": target_pose,
                "message": "No fresh /amcl_pose received for rule navigation.",
            }

        timeout = max(0.0, float(cfg.max_duration_sec))
        timeout_deadline = None if timeout <= 0.0 else time.monotonic() + timeout

        face_done, _ = _run_phase(
            "face_target",
            lambda current, distance, heading_error, final_yaw_error: _rotation_action_for_error(heading_error),
            lambda current, distance, heading_error, final_yaw_error: (
                distance <= cfg.xy_tolerance_m or abs(heading_error) <= cfg.face_target_yaw_tolerance_rad
            ),
            timeout_deadline=timeout_deadline,
        )
        if not face_done:
            _publish_stop_burst(
                front_wheel_publisher,
                rear_wheel_publisher,
                repeat=cfg.stop_repeat,
                interval_sec=cfg.stop_interval_sec,
            )
            return {
                "success": False,
                "phase": "face_target",
                "target_pose": target_pose,
                "message": "Rule navigation failed while facing the target point.",
            }

        drive_done, _ = _run_phase(
            "drive_to_target",
            lambda current, distance, heading_error, final_yaw_error: (
                _rotation_action_for_error(heading_error)
                if abs(heading_error) > cfg.drive_heading_tolerance_rad
                else "FORWARD_SLOW"
            ),
            lambda current, distance, heading_error, final_yaw_error: distance <= cfg.xy_tolerance_m,
            timeout_deadline=timeout_deadline,
        )
        if not drive_done:
            _publish_stop_burst(
                front_wheel_publisher,
                rear_wheel_publisher,
                repeat=cfg.stop_repeat,
                interval_sec=cfg.stop_interval_sec,
            )
            return {
                "success": False,
                "phase": "drive_to_target",
                "target_pose": target_pose,
                "message": "Rule navigation failed while driving to target position.",
            }

        align_done, final_pose = _run_phase(
            "align_target_yaw",
            lambda current, distance, heading_error, final_yaw_error: _rotation_action_for_error(
                final_yaw_error,
                final_alignment=True,
            ),
            lambda current, distance, heading_error, final_yaw_error: abs(final_yaw_error) <= cfg.final_yaw_tolerance_rad,
            timeout_deadline=timeout_deadline,
        )
        if not align_done:
            _publish_stop_burst(
                front_wheel_publisher,
                rear_wheel_publisher,
                repeat=cfg.stop_repeat,
                interval_sec=cfg.stop_interval_sec,
            )
            return {
                "success": False,
                "phase": "align_target_yaw",
                "target_pose": target_pose,
                "message": "Rule navigation failed while aligning target yaw.",
            }

        final_distance = _distance_to_target(final_pose, target_pose) if final_pose is not None else None
        final_yaw_error = (
            abs(_wrap_angle_rad(float(target_pose["yaw"]) - final_pose.yaw_rad))
            if final_pose is not None
            else None
        )
        print(
            "[move_car] rule navigation reached target: "
            f"distance={final_distance if final_distance is not None else float('nan'):.3f}m "
            f"yaw_error={final_yaw_error if final_yaw_error is not None else float('nan'):.3f}rad.",
            flush=True,
        )
        node.destroy_subscription(amcl_subscription)
        return {
            "success": True,
            "phase": "done",
            "target_pose": target_pose,
            "final_distance_m": final_distance,
            "final_yaw_error_rad": final_yaw_error,
        }
    finally:
        if front_wheel_publisher is not None and rear_wheel_publisher is not None:
            try:
                _publish_stop_burst(
                    front_wheel_publisher,
                    rear_wheel_publisher,
                    repeat=cfg.stop_repeat,
                    interval_sec=cfg.stop_interval_sec,
                )
            except Exception:
                pass
        if node is not None:
            node.destroy_node()
        if owns_rclpy and rclpy.ok():
            rclpy.shutdown()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the car to a ROS map pose with rule-based wheel commands.")
    parser.add_argument("--x", type=float, required=True, help="Goal x in the ROS map frame.")
    parser.add_argument("--y", type=float, required=True, help="Goal y in the ROS map frame.")
    parser.add_argument("--yaw-rad", type=float, default=None, help="Goal yaw in radians.")
    parser.add_argument("--yaw-deg", type=float, default=None, help="Goal yaw in degrees.")
    parser.add_argument("--z", type=float, default=0.0, help="Goal z in the ROS map frame.")
    parser.add_argument("--xy-tolerance-m", type=float, default=0.03, help="Stop distance tolerance.")
    parser.add_argument("--face-yaw-tolerance-rad", type=float, default=0.08, help="Initial face-target yaw tolerance.")
    parser.add_argument("--final-yaw-tolerance-rad", type=float, default=0.08, help="Final target yaw tolerance.")
    parser.add_argument("--max-duration-sec", type=float, default=120.0, help="Rule navigation timeout; 0 disables.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.yaw_rad is None and args.yaw_deg is None:
        yaw_rad = 0.0
    elif args.yaw_rad is not None:
        yaw_rad = float(args.yaw_rad)
    else:
        yaw_rad = math.radians(float(args.yaw_deg))
    result = drive_to_pose_by_rule(
        GoalPose2D(x=float(args.x), y=float(args.y), z=float(args.z), yaw_rad=yaw_rad),
        config=RuleNavigationConfig(
            xy_tolerance_m=float(args.xy_tolerance_m),
            face_target_yaw_tolerance_rad=float(args.face_yaw_tolerance_rad),
            final_yaw_tolerance_rad=float(args.final_yaw_tolerance_rad),
            max_duration_sec=float(args.max_duration_sec),
        ),
    )
    print(f"[move_car] rule navigation result = {result}", flush=True)


if __name__ == "__main__":
    main()
