"""
ROS-side navigation runner used by Orchestrator.nav_move_node.

Input:  JSON payload via --payload.
Output: single-line JSON result on stdout.
"""

import argparse
import json
import time
from typing import Any, Dict

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String


class NavMoveRunner(Node):
    def __init__(self, payload: Dict[str, Any]) -> None:
        super().__init__("nav_move_runner")
        self.payload = payload
        self.events = []

        self.goal_pose_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )
        status_topic = str(payload.get("status_topic", "/nav_move/status"))
        self.status_pub = self.create_publisher(String, status_topic, 10)
        self.plan_sub = self.create_subscription(Path, "/plan", self._plan_callback, 10)
        self.nav_action = ActionClient(self, NavigateToPose, "/navigate_to_pose")

        self._plan_ready = False
        self._plan_msg_counter = 0
        self._plan_baseline_counter = 0
        self._goal_sent = False

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

    def _plan_callback(self, msg: Path) -> None:
        self._plan_msg_counter += 1
        if not self._goal_sent:
            return
        if self._plan_ready:
            return
        if len(msg.poses) == 0:
            return
        if self._plan_msg_counter <= self._plan_baseline_counter:
            return

        self._plan_ready = True
        self._emit_event("plan_ready", f"path_poses={len(msg.poses)}")

    def _make_goal_pose(self) -> PoseStamped:
        goal = self.payload.get("goal_pose", {})
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
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
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
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

    def _wait_for_server(self, timeout_sec: float = 5.0) -> bool:
        start = time.monotonic()
        while time.monotonic() - start <= timeout_sec:
            if self.nav_action.wait_for_server(timeout_sec=0.2):
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return False

    def run(self) -> Dict[str, Any]:
        publish_interval = float(self.payload.get("publish_interval_sec", 0.5))
        plan_timeout = float(self.payload.get("plan_timeout_sec", 8.0))
        arrival_timeout = float(self.payload.get("arrival_timeout_sec", 120.0))
        publish_initialpose = bool(self.payload.get("publish_initialpose", False))

        goal_pose_msg = self._make_goal_pose()
        initial_pose_msg = self._make_initial_pose()

        # Pre-publish to allow DDS discovery before checking action server
        self._emit_event("publishing_poses", "Pre-publishing /initialpose and /goal_pose")
        for _ in range(10):
            if publish_initialpose:
                self.initial_pose_pub.publish(initial_pose_msg)
            self.goal_pose_pub.publish(goal_pose_msg)
            rclpy.spin_once(self, timeout_sec=0.1)

        if not self._wait_for_server():
            self._emit_event("attempt_failed", "navigate_to_pose action server unavailable")
            return {
                "success": False,
                "plan_ready": False,
                "message": "navigate_to_pose action server unavailable",
                "events": self.events,
            }

        goal_request = NavigateToPose.Goal()
        goal_request.pose = goal_pose_msg
        self._plan_ready = False
        self._plan_baseline_counter = self._plan_msg_counter

        send_goal_future = self.nav_action.send_goal_async(goal_request)
        send_goal_start = time.monotonic()
        while not send_goal_future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() - send_goal_start > 5.0:
                self._emit_event("attempt_failed", "send_goal timeout")
                return {
                    "success": False,
                    "plan_ready": False,
                    "message": "send_goal timeout",
                    "events": self.events,
                }

        goal_handle = send_goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._emit_event("attempt_failed", "NavigateToPose goal rejected")
            return {
                "success": False,
                "plan_ready": False,
                "message": "NavigateToPose goal rejected",
                "events": self.events,
            }

        self._goal_sent = True
        self._emit_event("goal_publishing", "start publishing /initialpose and /goal_pose")
        result_future = goal_handle.get_result_async()

        last_pub = 0.0
        plan_start = time.monotonic()
        while not self._plan_ready:
            now = time.monotonic()
            if now - last_pub >= publish_interval:
                if publish_initialpose:
                    self.initial_pose_pub.publish(initial_pose_msg)
                self.goal_pose_pub.publish(goal_pose_msg)
                last_pub = now
            rclpy.spin_once(self, timeout_sec=0.05)

            if result_future.done():
                break
            if now - plan_start > plan_timeout:
                goal_handle.cancel_goal_async()
                self._emit_event("attempt_failed", "plan generation timeout")
                return {
                    "success": False,
                    "plan_ready": False,
                    "message": "plan generation timeout",
                    "events": self.events,
                }

        arrival_start = time.monotonic()
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() - arrival_start > arrival_timeout:
                goal_handle.cancel_goal_async()
                self._emit_event("attempt_failed", "arrival timeout")
                return {
                    "success": False,
                    "plan_ready": self._plan_ready,
                    "message": "arrival timeout",
                    "events": self.events,
                }

        result = result_future.result()
        status = int(result.status)
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._emit_event("arrived", "NavigateToPose succeeded")
            return {
                "success": True,
                "plan_ready": self._plan_ready,
                "message": "NavigateToPose succeeded",
                "events": self.events,
            }

        self._emit_event("attempt_failed", f"NavigateToPose failed status={status}")
        return {
            "success": False,
            "plan_ready": self._plan_ready,
            "message": f"NavigateToPose failed status={status}",
            "events": self.events,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one blocking nav_move attempt.")
    parser.add_argument("--payload", required=True, help="JSON payload")
    args = parser.parse_args()

    payload = json.loads(args.payload)
    rclpy.init(args=None)
    node = NavMoveRunner(payload)
    try:
        result = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print(json.dumps(result))


if __name__ == "__main__":
    main()
