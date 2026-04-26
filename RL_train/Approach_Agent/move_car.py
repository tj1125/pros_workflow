from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GoalPose2D:
    x: float
    y: float
    yaw_rad: float
    z: float = 0.0


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


def publish_goal_pose(
    pose: Any,
    *,
    topic: str = "/goal_pose",
    frame_id: str = "map",
    publish_count: int = 10,
    interval_sec: float = 0.1,
    prefer_amcl_pose: bool = True,
) -> dict[str, float]:
    goal_pose = _goal_pose_dict_from_any(pose, prefer_amcl_pose=prefer_amcl_pose)

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

    owns_rclpy = False
    node = None
    try:
        if not rclpy.ok():
            rclpy.init(args=None)
            owns_rclpy = True

        node = Node("approach_agent_goal_pose_publisher")
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        publisher = node.create_publisher(PoseStamped, topic, qos)
        count = max(1, int(publish_count))
        interval = max(0.0, float(interval_sec))

        for index in range(count):
            msg = PoseStamped()
            msg.header.frame_id = frame_id
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.pose.position.x = float(goal_pose["x"])
            msg.pose.position.y = float(goal_pose["y"])
            msg.pose.position.z = float(goal_pose["z"])
            msg.pose.orientation.x = float(goal_pose["qx"])
            msg.pose.orientation.y = float(goal_pose["qy"])
            msg.pose.orientation.z = float(goal_pose["qz"])
            msg.pose.orientation.w = float(goal_pose["qw"])
            publisher.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.01)
            if index < count - 1 and interval > 0.0:
                time.sleep(interval)
        return goal_pose
    finally:
        if node is not None:
            node.destroy_node()
        if owns_rclpy and rclpy.ok():
            rclpy.shutdown()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a ROS map goal to /goal_pose.")
    parser.add_argument("--x", type=float, required=True, help="Goal x in the ROS map frame.")
    parser.add_argument("--y", type=float, required=True, help="Goal y in the ROS map frame.")
    parser.add_argument("--yaw-rad", type=float, default=None, help="Goal yaw in radians.")
    parser.add_argument("--yaw-deg", type=float, default=None, help="Goal yaw in degrees.")
    parser.add_argument("--z", type=float, default=0.0, help="Goal z in the ROS map frame.")
    parser.add_argument("--topic", default="/goal_pose", help="PoseStamped topic to publish.")
    parser.add_argument("--frame-id", default="map", help="PoseStamped frame_id.")
    parser.add_argument("--publish-count", type=int, default=10, help="Number of PoseStamped messages to publish.")
    parser.add_argument("--interval-sec", type=float, default=0.1, help="Delay between repeated publish attempts.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.yaw_rad is None and args.yaw_deg is None:
        yaw_rad = 0.0
    elif args.yaw_rad is not None:
        yaw_rad = float(args.yaw_rad)
    else:
        yaw_rad = math.radians(float(args.yaw_deg))
    goal_pose = publish_goal_pose(
        GoalPose2D(x=float(args.x), y=float(args.y), z=float(args.z), yaw_rad=yaw_rad),
        topic=args.topic,
        frame_id=args.frame_id,
        publish_count=int(args.publish_count),
        interval_sec=float(args.interval_sec),
    )
    print(f"[move_car] published /goal_pose = {goal_pose}", flush=True)


if __name__ == "__main__":
    main()
