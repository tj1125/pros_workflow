from __future__ import annotations

import time
from typing import Optional

import rclpy
from action_interface.action import NavGoal
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient, ClientGoalHandle
from rclpy.node import Node


class NavGoalBridge(Node):
    def __init__(self) -> None:
        super().__init__("nav_goal_bridge_node")
        self.declare_parameter("duplicate_goal_window_sec", 1.0)
        self.declare_parameter("tick_period_sec", 0.2)

        self._duplicate_goal_window_sec = float(
            self.get_parameter("duplicate_goal_window_sec").value
        )

        self._last_goal_key: Optional[tuple[float, float, float, float]] = None
        self._last_goal_received_at = 0.0
        self._pending_goal: Optional[PoseStamped] = None
        self._current_goal_key: Optional[tuple[float, float, float, float]] = None
        self._latest_amcl_pose: Optional[PoseWithCovarianceStamped] = None
        self._latest_plan: Optional[Path] = None
        self._nav2_goal_sent = False
        self._car_goal_sent = False
        self._active_nav2_goal: Optional[ClientGoalHandle] = None
        self._active_car_goal: Optional[ClientGoalHandle] = None

        self._navigate_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self._car_nav_client = ActionClient(self, NavGoal, "nav_action_server")

        self.create_subscription(PoseStamped, "/goal_pose", self._goal_callback, 10)
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self._amcl_callback,
            10,
        )
        self.create_subscription(Path, "/plan", self._plan_callback, 10)
        self.create_subscription(
            Path,
            "/received_global_plan",
            self._received_plan_callback,
            10,
        )
        self._received_plan_pub = self.create_publisher(Path, "/received_global_plan", 10)
        self.create_timer(
            float(self.get_parameter("tick_period_sec").value),
            self._tick,
        )

        self.get_logger().info(
            "nav_goal_bridge_node ready: /goal_pose -> /navigate_to_pose + nav_action_server"
        )

    @staticmethod
    def _goal_key(msg: PoseStamped) -> tuple[float, float, float, float]:
        return (
            round(float(msg.pose.position.x), 3),
            round(float(msg.pose.position.y), 3),
            round(float(msg.pose.orientation.z), 3),
            round(float(msg.pose.orientation.w), 3),
        )

    def _goal_callback(self, msg: PoseStamped) -> None:
        goal_key = self._goal_key(msg)
        now = time.monotonic()
        if (
            goal_key == self._last_goal_key
            and now - self._last_goal_received_at <= self._duplicate_goal_window_sec
        ):
            return

        self._last_goal_key = goal_key
        self._last_goal_received_at = now
        self._pending_goal = msg
        self._current_goal_key = goal_key
        self._latest_plan = None
        self._nav2_goal_sent = False
        self._car_goal_sent = False
        self._clear_received_global_plan()
        self._cancel_active_goals()
        self.get_logger().info(
            "Received new goal_pose: "
            f"x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}"
        )

    def _amcl_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self._latest_amcl_pose = msg

    def _plan_callback(self, msg: Path) -> None:
        self._latest_plan = msg
        self._received_plan_pub.publish(msg)

    def _received_plan_callback(self, msg: Path) -> None:
        if msg.poses:
            self._latest_plan = msg

    def _clear_received_global_plan(self) -> None:
        empty = Path()
        empty.header.frame_id = "map"
        self._received_plan_pub.publish(empty)

    def _cancel_active_goals(self) -> None:
        if self._active_nav2_goal is not None:
            self._active_nav2_goal.cancel_goal_async()
            self._active_nav2_goal = None
        if self._active_car_goal is not None:
            self._active_car_goal.cancel_goal_async()
            self._active_car_goal = None

    def _tick(self) -> None:
        if self._pending_goal is None:
            return

        if not self._nav2_goal_sent:
            self._try_send_nav2_goal()
            return

        if self._car_goal_sent:
            return

        if self._latest_amcl_pose is None:
            return

        if self._latest_plan is None or not self._latest_plan.poses:
            return

        self._try_send_car_goal()

    def _try_send_nav2_goal(self) -> None:
        if self._pending_goal is None:
            return
        if not self._navigate_client.wait_for_server(timeout_sec=0.1):
            return

        goal = NavigateToPose.Goal()
        goal.pose = self._pending_goal
        future = self._navigate_client.send_goal_async(goal)
        future.add_done_callback(self._handle_nav2_goal_response)
        self._nav2_goal_sent = True
        self.get_logger().info("Forwarded /goal_pose to /navigate_to_pose")

    def _try_send_car_goal(self) -> None:
        if not self._car_nav_client.wait_for_server(timeout_sec=0.1):
            return

        goal = NavGoal.Goal()
        goal.mode = "Manual_Nav"
        future = self._car_nav_client.send_goal_async(goal)
        future.add_done_callback(self._handle_car_goal_response)
        self._car_goal_sent = True
        self.get_logger().info("Triggered pros car nav_action_server with mode=Manual_Nav")

    def _handle_nav2_goal_response(self, future) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._nav2_goal_sent = False
            self.get_logger().warn("/navigate_to_pose rejected the goal; will retry")
            return

        self._active_nav2_goal = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._handle_nav2_result)

    def _handle_car_goal_response(self, future) -> None:
        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._car_goal_sent = False
            self.get_logger().warn("nav_action_server rejected the goal; will retry")
            return

        self._active_car_goal = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._handle_car_result)

    def _handle_nav2_result(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.get_logger().error(f"/navigate_to_pose result error: {exc}")
            return

        self.get_logger().info(
            f"/navigate_to_pose finished with status={int(result.status)}"
        )
        self._active_nav2_goal = None

    def _handle_car_result(self, future) -> None:
        try:
            result = future.result()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.get_logger().error(f"nav_action_server result error: {exc}")
            return

        self.get_logger().info(
            f"nav_action_server finished with status={int(result.status)}"
        )
        self._active_car_goal = None


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
