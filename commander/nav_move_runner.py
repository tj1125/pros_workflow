"""
ROS-side navigation runner used by Orchestrator.nav_move_node.

Input:  JSON payload via --payload.
Output: single-line JSON result on stdout.
"""

import argparse
import json
import time
from typing import Any, Dict, Optional

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import ComputePathToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String

from commander.pros_nav import ProsPathFollower


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
        self.goal_pose_pub = self.create_publisher(PoseStamped, "/goal_pose", latched_qos)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            "/initialpose",
            latched_qos,
        )
        self.plan_pub = self.create_publisher(Path, "/plan", latched_qos)
        self.received_plan_pub = self.create_publisher(Path, "/received_global_plan", latched_qos)
        status_topic = str(payload.get("status_topic", "/nav_move/status"))
        self.status_pub = self.create_publisher(String, status_topic, 10)

        self.amcl_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self._amcl_pose_callback,
            10,
        )
        self.planner_action = ActionClient(self, ComputePathToPose, "/compute_path_to_pose")

        self._last_amcl_pose_received_at: Optional[Time] = None
        self._last_amcl_pose_msg: Optional[PoseWithCovarianceStamped] = None
        self._current_path: Optional[Path] = None
        self._plan_ready = False

    @staticmethod
    def _use_latest_available_transform(msg: PoseStamped | PoseWithCovarianceStamped) -> None:
        msg.header.stamp.sec = 0
        msg.header.stamp.nanosec = 0

    def _stamp_with_now(self, msg: PoseStamped | PoseWithCovarianceStamped) -> None:
        msg.header.stamp = self.get_clock().now().to_msg()

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

    def _make_goal_pose(self, stamp_mode: str = "latest") -> PoseStamped:
        goal = self.payload.get("goal_pose", {})
        msg = PoseStamped()
        msg.header.frame_id = "map"
        if stamp_mode == "now":
            self._stamp_with_now(msg)
        else:
            self._use_latest_available_transform(msg)
        msg.pose.position.x = float(goal.get("x", 0.0))
        msg.pose.position.y = float(goal.get("y", 0.0))
        msg.pose.position.z = float(goal.get("z", 0.0))
        msg.pose.orientation.x = float(goal.get("qx", 0.0))
        msg.pose.orientation.y = float(goal.get("qy", 0.0))
        msg.pose.orientation.z = float(goal.get("qz", 0.0))
        msg.pose.orientation.w = float(goal.get("qw", 1.0))
        return msg

    def _make_initial_pose(self, stamp_mode: str = "latest") -> PoseWithCovarianceStamped:
        initial = self.payload.get("initial_pose", {})
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        if stamp_mode == "now":
            self._stamp_with_now(msg)
        else:
            self._use_latest_available_transform(msg)
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

    def _wait_for_action_server(self, client: ActionClient, timeout_sec: float) -> bool:
        start = time.monotonic()
        while time.monotonic() - start <= timeout_sec:
            if client.wait_for_server(timeout_sec=0.2):
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return False

    def _wait_for_amcl_pose(self, since: Time, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() <= deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._last_amcl_pose_received_at and self._last_amcl_pose_received_at >= since:
                return True
        return False

    def _clear_plan_topics(self) -> None:
        empty = Path()
        empty.header.frame_id = "map"
        empty.header.stamp = self.get_clock().now().to_msg()
        self.plan_pub.publish(empty)
        self.received_plan_pub.publish(empty)
        self._current_path = None

    def _publish_plan(self, path: Path, event_name: str = "plan_ready") -> None:
        path.header.stamp = self.get_clock().now().to_msg()
        if not path.header.frame_id:
            path.header.frame_id = "map"
        self.plan_pub.publish(path)
        self.received_plan_pub.publish(path)
        self._current_path = path
        self._plan_ready = len(path.poses) > 0
        self._emit_event(event_name, f"path_poses={len(path.poses)}")

    def _compute_preview_path(self, goal_pose_msg: PoseStamped, timeout_sec: float) -> Optional[Path]:
        goal_request = ComputePathToPose.Goal()
        goal_request.goal = goal_pose_msg
        goal_request.use_start = False
        goal_request.planner_id = ""

        send_goal_future = self.planner_action.send_goal_async(goal_request)
        send_goal_start = time.monotonic()
        while not send_goal_future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() - send_goal_start > 5.0:
                self._emit_event("plan_preview_unavailable", "compute_path_to_pose send_goal timeout")
                return None

        goal_handle = send_goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._emit_event("plan_preview_unavailable", "compute_path_to_pose goal rejected")
            return None

        result_future = goal_handle.get_result_async()
        planning_start = time.monotonic()
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() - planning_start > timeout_sec:
                goal_handle.cancel_goal_async()
                self._emit_event("plan_preview_unavailable", "path preview timeout")
                return None

        result = result_future.result()
        if int(result.status) != GoalStatus.STATUS_SUCCEEDED:
            self._emit_event("plan_preview_unavailable", f"preview status={int(result.status)}")
            return None

        path = result.result.path
        if not path.poses:
            self._emit_event("plan_preview_unavailable", "preview path empty")
            return None

        return path

    def _follow_plan(
        self,
        goal_pose_topic_msg: PoseStamped,
        goal_pose_action_msg: PoseStamped,
        *,
        arrival_timeout: float,
        plan_timeout: float,
        replan_period_sec: float,
        follow_control_hz: float,
        goal_tolerance_m: float,
        goal_heading_tolerance_deg: float,
        min_target_distance_m: float,
    ) -> Dict[str, Any]:
        active_path = self._current_path
        if active_path is None or not active_path.poses:
            return {
                "success": False,
                "message": "global plan unavailable",
            }

        goal = self.payload.get("goal_pose", {}) or {}
        target_heading_point = None
        if "face_target_x" in goal and "face_target_y" in goal:
            target_heading_point = (
                float(goal["face_target_x"]),
                float(goal["face_target_y"]),
            )

        follower = ProsPathFollower(
            self,
            goal_tolerance_m=goal_tolerance_m,
            goal_heading_tolerance_deg=goal_heading_tolerance_deg,
            min_target_distance_m=min_target_distance_m,
            target_heading_point=target_heading_point,
        )
        control_period = 1.0 / max(1.0, follow_control_hz)
        deadline = time.monotonic() + arrival_timeout
        next_replan_at = time.monotonic() + max(0.5, replan_period_sec)

        self._emit_event("following_plan", "Following /received_global_plan with pros-style controller")

        last_action = None
        last_detail = ""
        while time.monotonic() <= deadline:
            loop_start = time.monotonic()
            rclpy.spin_once(self, timeout_sec=min(0.05, control_period))

            follow_step = follower.follow_step(
                self._last_amcl_pose_msg,
                goal_pose_topic_msg,
                self._current_path,
            )
            if follow_step.arrived:
                follower.stop()
                return {
                    "success": True,
                    "message": "pros-style path follower reached goal",
                }

            if (
                not follow_step.waiting_for_data
                and (
                    follow_step.action_key != last_action
                    or follow_step.detail != last_detail
                )
            ):
                self._emit_event(
                    "tracking",
                    f"action={follow_step.action_key} distance={follow_step.distance_to_goal:.2f} {follow_step.detail}",
                )
                last_action = follow_step.action_key
                last_detail = follow_step.detail

            now = time.monotonic()
            if now >= next_replan_at:
                refreshed_path = self._compute_preview_path(goal_pose_action_msg, plan_timeout)
                if refreshed_path is not None:
                    self._publish_plan(refreshed_path, event_name="plan_updated")
                next_replan_at = now + max(0.5, replan_period_sec)

            remaining = control_period - (time.monotonic() - loop_start)
            if remaining > 0.0:
                time.sleep(remaining)

        follower.stop()
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
        action_server_timeout = float(self.payload.get("action_server_timeout_sec", 8.0))
        replan_period_sec = float(self.payload.get("replan_period_sec", 1.5))
        follow_control_hz = float(self.payload.get("follow_control_hz", 10.0))
        goal_tolerance_m = float(self.payload.get("goal_tolerance_m", 0.08))
        goal_heading_tolerance_deg = float(
            self.payload.get("goal_heading_tolerance_deg", 5.0)
        )
        min_target_distance_m = float(self.payload.get("path_target_distance_m", 0.5))

        goal_pose_topic_msg = self._make_goal_pose(stamp_mode="now")
        goal_pose_action_msg = self._make_goal_pose(stamp_mode="latest")
        initial_pose_msg = self._make_initial_pose(stamp_mode="now")

        self._plan_ready = False
        self._clear_plan_topics()

        warmup_detail = "Warm-up publishing /goal_pose"
        if publish_initialpose:
            warmup_detail = "Warm-up publishing /initialpose and /goal_pose"
        self._emit_event("publishing_poses", warmup_detail)
        initial_pose_sent_at = None
        for idx in range(max(1, warmup_publish_count)):
            if publish_initialpose and idx == 0:
                initial_pose_sent_at = self.get_clock().now()
                self.initial_pose_pub.publish(initial_pose_msg)
            self.goal_pose_pub.publish(goal_pose_topic_msg)
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

        if not self._wait_for_action_server(self.planner_action, action_server_timeout):
            self._emit_event("attempt_failed", "compute_path_to_pose action server unavailable")
            return {
                "success": False,
                "plan_ready": False,
                "message": "compute_path_to_pose action server unavailable",
                "events": self.events,
            }

        preview_path = self._compute_preview_path(goal_pose_action_msg, plan_timeout)
        if preview_path is None:
            self._emit_event("attempt_failed", "failed to compute global plan")
            return {
                "success": False,
                "plan_ready": False,
                "message": "failed to compute global plan",
                "events": self.events,
            }

        self._publish_plan(preview_path)
        nav_result = self._follow_plan(
            goal_pose_topic_msg,
            goal_pose_action_msg,
            arrival_timeout=arrival_timeout,
            plan_timeout=plan_timeout,
            replan_period_sec=replan_period_sec,
            follow_control_hz=follow_control_hz,
            goal_tolerance_m=goal_tolerance_m,
            goal_heading_tolerance_deg=goal_heading_tolerance_deg,
            min_target_distance_m=min_target_distance_m,
        )

        if nav_result["success"]:
            self._emit_event("arrived", nav_result["message"])
            return {
                "success": True,
                "plan_ready": self._plan_ready,
                "message": nav_result["message"],
                "events": self.events,
            }

        self._emit_event("attempt_failed", nav_result["message"])
        return {
            "success": False,
            "plan_ready": self._plan_ready,
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
