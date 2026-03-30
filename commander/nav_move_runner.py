"""
ROS-side pose publisher and navigation observer used by Orchestrator.nav_move_node.

Input:  JSON payload via --payload.
Output: single-line JSON result on stdout.
"""

import argparse
import json
import time
from math import atan2, degrees, hypot
from typing import Any, Dict, Optional

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String


def _yaw_deg_from_quaternion(z: float, w: float) -> float:
    return degrees(2.0 * atan2(z, w))


def _normalize_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


class NavMoveRunner(Node):
    def __init__(self, payload: Dict[str, Any]) -> None:
        super().__init__("nav_move_runner")
        self.payload = payload
        self.events = []

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        status_topic = str(payload.get("status_topic", "/nav_move/status"))

        self.goal_pose_pub = self.create_publisher(PoseStamped, "/goal_pose", latched_qos)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            "/initialpose",
            latched_qos,
        )
        self.status_pub = self.create_publisher(String, status_topic, 10)

        self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self._amcl_pose_callback,
            10,
        )
        self.create_subscription(Path, "/plan", self._plan_callback, 10)
        self.create_subscription(
            Path,
            "/received_global_plan",
            self._plan_callback,
            10,
        )

        self._last_amcl_pose_received_at: Optional[Time] = None
        self._last_amcl_pose_msg: Optional[PoseWithCovarianceStamped] = None
        self._last_plan_msg: Optional[Path] = None
        self._plan_ready = False

    def _emit_event(self, event: str, detail: str = "") -> None:
        payload = {
            "event": event,
            "detail": detail,
            "attempt": int(self.payload.get("attempt", 0)),
            "rank": int(self.payload.get("rank", 0)),
            "source": self.payload.get("source", ""),
            "timestamp": time.time(),
        }
        self.events.append(payload)
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)

    def _amcl_pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self._last_amcl_pose_msg = msg
        self._last_amcl_pose_received_at = self.get_clock().now()

    def _plan_callback(self, msg: Path) -> None:
        if msg.poses:
            self._last_plan_msg = msg
            self._plan_ready = True

    def _stamp_with_now(self, msg: PoseStamped | PoseWithCovarianceStamped) -> None:
        msg.header.stamp = self.get_clock().now().to_msg()

    def _make_goal_pose(self) -> PoseStamped:
        goal = self.payload.get("goal_pose", {})
        msg = PoseStamped()
        msg.header.frame_id = "map"
        self._stamp_with_now(msg)
        msg.pose.position.x = float(goal.get("x", 0.0))
        msg.pose.position.y = float(goal.get("y", 0.0))
        msg.pose.position.z = float(goal.get("z", 0.0))
        msg.pose.orientation.x = float(goal.get("qx", 0.0))
        msg.pose.orientation.y = float(goal.get("qy", 0.0))
        msg.pose.orientation.z = float(goal.get("qz", 0.0))
        msg.pose.orientation.w = float(goal.get("qw", 1.0))
        return msg

    def _make_initial_pose(self) -> PoseWithCovarianceStamped:
        initial = self.payload.get("initial_pose", {})
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        self._stamp_with_now(msg)
        msg.pose.pose.position.x = float(initial.get("x", 0.0))
        msg.pose.pose.position.y = float(initial.get("y", 0.0))
        msg.pose.pose.position.z = float(initial.get("z", 0.0))
        msg.pose.pose.orientation.x = float(initial.get("qx", 0.0))
        msg.pose.pose.orientation.y = float(initial.get("qy", 0.0))
        msg.pose.pose.orientation.z = float(initial.get("qz", 0.0))
        msg.pose.pose.orientation.w = float(initial.get("qw", 1.0))
        covariance = initial.get("covariance", [])
        if isinstance(covariance, list) and len(covariance) == 36:
            msg.pose.covariance = [float(v) for v in covariance]
        return msg

    def _wait_for_amcl_pose(self, since: Time, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() <= deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._last_amcl_pose_received_at and self._last_amcl_pose_received_at >= since:
                return True
        return False

    def _wait_for_plan(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() <= deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._plan_ready:
                return True
        return False

    def _goal_heading_error(
        self,
        car_orientation: tuple[float, float],
        goal_orientation: tuple[float, float],
    ) -> float:
        car_yaw = _yaw_deg_from_quaternion(car_orientation[0], car_orientation[1])
        goal_yaw = _yaw_deg_from_quaternion(goal_orientation[0], goal_orientation[1])
        return _normalize_angle_deg(goal_yaw - car_yaw)

    def _observe_until_arrival(
        self,
        goal_pose_msg: PoseStamped,
        *,
        arrival_timeout: float,
        goal_tolerance_m: float,
        goal_heading_tolerance_deg: float,
    ) -> Dict[str, Any]:
        goal_position_msg = goal_pose_msg.pose.position
        goal_orientation_msg = goal_pose_msg.pose.orientation
        goal_position = (float(goal_position_msg.x), float(goal_position_msg.y))
        goal_orientation = (float(goal_orientation_msg.z), float(goal_orientation_msg.w))
        deadline = time.monotonic() + arrival_timeout
        last_detail = ""

        while time.monotonic() <= deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._last_amcl_pose_msg is None:
                continue

            car_position_msg = self._last_amcl_pose_msg.pose.pose.position
            car_orientation_msg = self._last_amcl_pose_msg.pose.pose.orientation
            car_position = (float(car_position_msg.x), float(car_position_msg.y))
            car_orientation = (float(car_orientation_msg.z), float(car_orientation_msg.w))

            distance_to_goal = hypot(
                car_position[0] - goal_position[0],
                car_position[1] - goal_position[1],
            )
            heading_error = self._goal_heading_error(car_orientation, goal_orientation)
            detail = (
                f"distance={distance_to_goal:.2f} "
                f"goal_heading_error={heading_error:.1f}"
            )

            if detail != last_detail:
                self._emit_event("tracking", detail)
                last_detail = detail

            if (
                distance_to_goal <= goal_tolerance_m
                and abs(heading_error) <= goal_heading_tolerance_deg
            ):
                return {
                    "success": True,
                    "message": "goal pose reached",
                }

        return {
            "success": False,
            "message": "arrival timeout",
        }

    def run(self) -> Dict[str, Any]:
        plan_timeout = float(self.payload.get("plan_timeout_sec", 8.0))
        arrival_timeout = float(self.payload.get("arrival_timeout_sec", 120.0))
        initial_pose_timeout = float(self.payload.get("initial_pose_timeout_sec", 5.0))
        publish_initialpose = bool(self.payload.get("publish_initialpose", False))
        warmup_publish_count = int(self.payload.get("warmup_publish_count", 3))
        warmup_sleep = float(self.payload.get("publish_interval_sec", 0.2))
        goal_tolerance_m = float(self.payload.get("goal_tolerance_m", 0.08))
        goal_heading_tolerance_deg = float(
            self.payload.get("goal_heading_tolerance_deg", 5.0)
        )

        goal_pose_msg = self._make_goal_pose()
        initial_pose_msg = self._make_initial_pose()
        self._plan_ready = False
        self._last_plan_msg = None

        warmup_detail = "Warm-up publishing /goal_pose"
        if publish_initialpose:
            warmup_detail = "Warm-up publishing /initialpose and /goal_pose"
        self._emit_event("publishing_poses", warmup_detail)

        initial_pose_sent_at = None
        for idx in range(max(1, warmup_publish_count)):
            if publish_initialpose and idx == 0:
                initial_pose_sent_at = self.get_clock().now()
                self.initial_pose_pub.publish(initial_pose_msg)
            self.goal_pose_pub.publish(goal_pose_msg)
            rclpy.spin_once(self, timeout_sec=min(warmup_sleep, 0.1))
            if warmup_sleep > 0.1:
                time.sleep(max(0.0, warmup_sleep - 0.1))

        if publish_initialpose:
            self._emit_event("localization_wait", "Waiting for /amcl_pose after one-shot /initialpose")
            if not self._wait_for_amcl_pose(
                initial_pose_sent_at or self.get_clock().now(),
                initial_pose_timeout,
            ):
                self._emit_event("attempt_failed", "initial pose was not acknowledged by AMCL")
                return {
                    "success": False,
                    "plan_ready": False,
                    "message": "initial pose was not acknowledged by AMCL",
                    "events": self.events,
                }

        self._emit_event(
            "plan_wait",
            "Waiting for tools-side Nav2 bridge to publish a global plan",
        )
        if not self._wait_for_plan(plan_timeout):
            self._emit_event("attempt_failed", "failed to observe global plan")
            return {
                "success": False,
                "plan_ready": False,
                "message": "failed to observe global plan",
                "events": self.events,
            }

        self._emit_event(
            "plan_ready",
            f"path_poses={len(self._last_plan_msg.poses) if self._last_plan_msg else 0}",
        )
        nav_result = self._observe_until_arrival(
            goal_pose_msg,
            arrival_timeout=arrival_timeout,
            goal_tolerance_m=goal_tolerance_m,
            goal_heading_tolerance_deg=goal_heading_tolerance_deg,
        )
        if nav_result["success"]:
            self._emit_event("arrived", nav_result["message"])
            return {
                "success": True,
                "plan_ready": True,
                "message": nav_result["message"],
                "events": self.events,
            }

        self._emit_event("attempt_failed", nav_result["message"])
        return {
            "success": False,
            "plan_ready": True,
            "message": nav_result["message"],
            "events": self.events,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True, help="JSON payload")
    args = parser.parse_args()

    payload = json.loads(args.payload)
    rclpy.init()
    node = NavMoveRunner(payload)
    try:
        result = node.run()
        print(json.dumps(result, ensure_ascii=False))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
