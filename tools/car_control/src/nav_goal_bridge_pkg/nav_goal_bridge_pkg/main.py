from __future__ import annotations

import json
import time
from functools import partial
from typing import Any, Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String


class NavGoalBridge(Node):
    def __init__(self) -> None:
        super().__init__("nav_goal_bridge_node")
        self.declare_parameter("duplicate_goal_window_sec", 1.0)
        self.declare_parameter("tick_period_sec", 0.2)

        self._duplicate_goal_window_sec = float(
            self.get_parameter("duplicate_goal_window_sec").value
        )
        self._tick_period_sec = float(self.get_parameter("tick_period_sec").value)

        self._navigate_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")

        self._received_plan_pub = self.create_publisher(Path, "/received_global_plan", 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._goal_callback, 10)
        self.create_subscription(Path, "/plan", self._plan_callback, 10)
        self.create_subscription(String, "/auto_nav/result", self._auto_nav_result_callback, 10)
        self.create_timer(self._tick_period_sec, self._tick)

        self._last_goal_key: Optional[tuple[float, ...]] = None
        self._last_goal_received_at = 0.0
        self._active_nav2_goal: Optional[Any] = None
        self._pending_nav2_goal: Optional[tuple[int, PoseStamped]] = None
        self._nav2_goal_sent = False
        self._goal_sequence = 0
        self._active_goal_sequence = 0
        self._navigation_completed = False

        self.get_logger().info("nav_goal_bridge_node ready: /goal_pose -> /navigate_to_pose")

    @staticmethod
    def _pose_key(msg: PoseStamped) -> tuple[float, float, float, float]:
        return (
            round(float(msg.pose.position.x), 3),
            round(float(msg.pose.position.y), 3),
            round(float(msg.pose.orientation.z), 3),
            round(float(msg.pose.orientation.w), 3),
        )

    def _goal_callback(self, msg: PoseStamped) -> None:
        goal_key = self._pose_key(msg)
        now = time.monotonic()
        if (
            goal_key == self._last_goal_key
            and now - self._last_goal_received_at <= self._duplicate_goal_window_sec
        ):
            return

        self._goal_sequence += 1
        self._last_goal_key = goal_key
        self._last_goal_received_at = now
        self._navigation_completed = False
        self._clear_received_global_plan()
        self._cancel_active_nav2_goal()
        self._queue_nav2_goal(goal_sequence=self._goal_sequence, pose=self._copy_pose(msg))
        self.get_logger().info(
            "Accepted new goal: "
            f"pose=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f})"
        )

    def _queue_nav2_goal(self, *, goal_sequence: int, pose: PoseStamped) -> None:
        self._pending_nav2_goal = (goal_sequence, pose)
        self._nav2_goal_sent = False

    def _plan_callback(self, msg: Path) -> None:
        if self._navigation_completed:
            return
        self._received_plan_pub.publish(msg)

    def _clear_received_global_plan(self) -> None:
        empty = Path()
        empty.header.frame_id = "map"
        self._received_plan_pub.publish(empty)

    def _auto_nav_result_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict) or not bool(payload.get("success", False)):
            return
        message = str(payload.get("message", ""))
        if message.startswith("Navigation goal reached successfully."):
            self._navigation_completed = True
            self._clear_received_global_plan()

    def _cancel_active_nav2_goal(self) -> None:
        if self._active_nav2_goal is not None:
            self._active_nav2_goal.cancel_goal_async()
            self._active_nav2_goal = None

    def _tick(self) -> None:
        if self._pending_nav2_goal is not None and not self._nav2_goal_sent:
            self._try_send_nav2_goal()

    def _try_send_nav2_goal(self) -> None:
        if self._pending_nav2_goal is None:
            return
        if not self._navigate_client.wait_for_server(timeout_sec=0.1):
            return

        goal_sequence, pose = self._pending_nav2_goal
        goal = NavigateToPose.Goal()
        goal.pose = pose
        future = self._navigate_client.send_goal_async(goal)
        future.add_done_callback(partial(self._handle_nav2_goal_response, goal_sequence))
        self._nav2_goal_sent = True
        self.get_logger().info(
            f"Forwarded goal {goal_sequence} to /navigate_to_pose"
        )

    def _handle_nav2_goal_response(self, goal_sequence: int, future) -> None:
        if goal_sequence != self._goal_sequence:
            return

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._nav2_goal_sent = False
            self.get_logger().warn(
                f"/navigate_to_pose rejected goal {goal_sequence}; will retry"
            )
            return

        self._active_nav2_goal = goal_handle
        self._active_goal_sequence = goal_sequence
        self._pending_nav2_goal = None
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(partial(self._handle_nav2_result, goal_sequence))

    def _handle_nav2_result(self, goal_sequence: int, future) -> None:
        try:
            result = future.result()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.get_logger().error(f"/navigate_to_pose result error: {exc}")
            return

        if goal_sequence != self._active_goal_sequence:
            return

        self._active_nav2_goal = None
        status = int(result.status)
        self.get_logger().info(f"/navigate_to_pose finished goal {goal_sequence} with status={status}")

    @staticmethod
    def _copy_pose(msg: PoseStamped) -> PoseStamped:
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose
        return pose

def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavGoalBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
