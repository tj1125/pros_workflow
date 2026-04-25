"""
ROS-side pose publisher and navigation observer used by Orchestrator.nav_move_node.

Input:  JSON payload via --payload.
Output: single-line JSON result on stdout.
"""

import argparse
import json
import math
import time
from math import atan2, hypot
from typing import Any, Dict, Optional

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String

from .nav_settings import (
    goal_heading_tolerance_rad_default,
    goal_tolerance_m_default,
)


def _yaw_rad_from_quaternion(z: float, w: float) -> float:
    return 2.0 * atan2(z, w)


def _normalize_angle_rad(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class NavMoveRunner(Node):
    def __init__(self, payload: Dict[str, Any]) -> None:
        super().__init__("nav_move_runner")
        self.payload = payload
        self.events = []
        status_topic = str(payload.get("status_topic", "/nav_move/status"))

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

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
        self.create_subscription(
            String,
            str(payload.get("nav_result_topic", "/auto_nav/result")),
            self._auto_nav_result_callback,
            10,
        )

        self._last_amcl_pose_received_at: Optional[Time] = None
        self._last_amcl_pose_msg: Optional[PoseWithCovarianceStamped] = None
        self._last_plan_msg: Optional[Path] = None
        self._last_auto_nav_result: Optional[Dict[str, Any]] = None
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

    def _auto_nav_result_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return
        self._last_auto_nav_result = payload

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

    def _goal_heading_error(
        self,
        car_orientation: tuple[float, float],
        goal_orientation: tuple[float, float],
    ) -> float:
        car_yaw = _yaw_rad_from_quaternion(car_orientation[0], car_orientation[1])
        goal_yaw = _yaw_rad_from_quaternion(goal_orientation[0], goal_orientation[1])
        return _normalize_angle_rad(goal_yaw - car_yaw)

    def _auto_nav_result(self) -> Optional[Dict[str, Any]]:
        if self._last_auto_nav_result is None:
            return None

        success = bool(self._last_auto_nav_result.get("success", False))
        message = str(
            self._last_auto_nav_result.get("message")
            or ("navigation completed" if success else "navigation failed")
        )
        return {
            "success": success,
            "message": message,
        }

    @staticmethod
    def _goal_pose_is_satisfied(
        *,
        distance_to_goal: float,
        heading_error: float,
        goal_tolerance_m: float,
        goal_heading_tolerance_rad: float,
    ) -> bool:
        return (
            distance_to_goal <= goal_tolerance_m
            and abs(heading_error) <= goal_heading_tolerance_rad
        )

    def _observe_until_arrival(
        self,
        goal_pose_msg: PoseStamped,
        *,
        arrival_timeout: float,
        goal_tolerance_m: float,
        goal_heading_tolerance_rad: float,
    ) -> Dict[str, Any]:
        """Track AMCL pose, but treat /auto_nav/result as the authoritative finish."""
        goal_position_msg = goal_pose_msg.pose.position
        goal_orientation_msg = goal_pose_msg.pose.orientation
        goal_position = (float(goal_position_msg.x), float(goal_position_msg.y))
        goal_orientation = (float(goal_orientation_msg.z), float(goal_orientation_msg.w))
        deadline = time.monotonic() + arrival_timeout
        last_detail = ""
        last_distance_to_goal: Optional[float] = None
        last_heading_error: Optional[float] = None
        pose_goal_reached_event_sent = False

        while time.monotonic() <= deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

            auto_nav_result = self._auto_nav_result()
            if auto_nav_result is not None:
                return auto_nav_result

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
            last_distance_to_goal = distance_to_goal
            last_heading_error = heading_error
            detail = (
                f"distance={distance_to_goal:.2f} "
                f"goal_heading_error={heading_error:.3f}rad"
            )

            if detail != last_detail:
                self._emit_event("tracking", detail)
                last_detail = detail

            if self._goal_pose_is_satisfied(
                distance_to_goal=distance_to_goal,
                heading_error=heading_error,
                goal_tolerance_m=goal_tolerance_m,
                goal_heading_tolerance_rad=goal_heading_tolerance_rad,
            ):
                if not pose_goal_reached_event_sent:
                    self._emit_event(
                        "pose_goal_reached_waiting_auto_nav",
                        "goal pose tolerance met; waiting for /auto_nav/result",
                    )
                    pose_goal_reached_event_sent = True
                continue

        if pose_goal_reached_event_sent:
            timeout_reason = "waiting for /auto_nav/result"
        elif last_distance_to_goal is None or last_heading_error is None:
            timeout_reason = "no amcl pose"
        else:
            distance_ok = last_distance_to_goal <= goal_tolerance_m
            heading_ok = abs(last_heading_error) <= goal_heading_tolerance_rad
            if not distance_ok and not heading_ok:
                timeout_reason = "distance and heading not met"
            elif not distance_ok:
                timeout_reason = "distance not met"
            elif not heading_ok:
                timeout_reason = "heading not met"
            else:
                timeout_reason = "goal checker conditions not satisfied"

        return {
            "success": False,
            "message": f"arrival timeout: {timeout_reason}",
        }

    def run(self) -> Dict[str, Any]:
        plan_timeout = float(self.payload.get("plan_timeout_sec", 8.0))
        arrival_timeout = float(self.payload.get("arrival_timeout_sec", 120.0))
        initial_pose_timeout = float(self.payload.get("initial_pose_timeout_sec", 5.0))
        publish_initialpose = bool(self.payload.get("publish_initialpose", False))
        warmup_publish_count = int(self.payload.get("warmup_publish_count", 3))
        warmup_sleep = float(self.payload.get("publish_interval_sec", 0.2))
        goal_tolerance_m = float(
            self.payload.get("goal_tolerance_m", goal_tolerance_m_default())
        )
        legacy_goal_heading_tolerance_deg = self.payload.get("goal_heading_tolerance_deg")
        goal_heading_tolerance_rad = float(
            self.payload.get(
                "goal_heading_tolerance_rad",
                math.radians(float(legacy_goal_heading_tolerance_deg))
                if legacy_goal_heading_tolerance_deg is not None
                else goal_heading_tolerance_rad_default(),
            )
        )

        goal_pose_msg = self._make_goal_pose()
        initial_pose_msg = self._make_initial_pose()
        self._plan_ready = False
        self._last_plan_msg = None
        self._last_auto_nav_result = None

        self._emit_event(
            "plan_wait",
            "Hard looping /goal_pose (and /initialpose) every 0.1s until global plan is ready...",
        )

        deadline = time.monotonic() + plan_timeout
        while time.monotonic() <= deadline:
            if publish_initialpose:
                self._stamp_with_now(initial_pose_msg)
                self.initial_pose_pub.publish(initial_pose_msg)
            
            self._stamp_with_now(goal_pose_msg)
            self.goal_pose_pub.publish(goal_pose_msg)
            
            rclpy.spin_once(self, timeout_sec=0.01)
            if self._plan_ready:
                break
                
            time.sleep(0.1)

        if not self._plan_ready:
            self._emit_event("attempt_failed", "failed to observe global plan after hard loop")
            return {
                "success": False,
                "plan_ready": False,
                "message": "failed to observe global plan after hard loop",
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
            goal_heading_tolerance_rad=goal_heading_tolerance_rad,
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
