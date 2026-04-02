from __future__ import annotations

import math
import time
from functools import partial
from typing import Optional

import rclpy
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient, ClientGoalHandle
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String, UInt32


class NavGoalBridge(Node):
    APPROACH = "APPROACH"
    ALIGN = "ALIGN"
    ALIGN_COMPLETE = "ALIGN_COMPLETE"

    def __init__(self) -> None:
        super().__init__("nav_goal_bridge_node")
        self.declare_parameter("duplicate_goal_window_sec", 1.0)
        self.declare_parameter("tick_period_sec", 0.2)
        self.declare_parameter("target_point_timeout_sec", 2.0)
        self.declare_parameter("nav2_success_timeout_sec", 2.0)

        self._duplicate_goal_window_sec = float(
            self.get_parameter("duplicate_goal_window_sec").value
        )
        self._target_point_timeout_sec = float(
            self.get_parameter("target_point_timeout_sec").value
        )
        self._nav2_success_timeout_sec = float(
            self.get_parameter("nav2_success_timeout_sec").value
        )
        self._tick_period_sec = float(self.get_parameter("tick_period_sec").value)

        nav_share_dir = get_package_share_directory("vlm_rl_nav")
        self._phase1_bt_xml = (
            f"{nav_share_dir}/config/navigate_to_pose_phase1.xml"
        )
        self._phase2_bt_xml = (
            f"{nav_share_dir}/config/navigate_to_pose_phase2.xml"
        )

        self._navigate_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._received_plan_pub = self.create_publisher(Path, "/received_global_plan", 10)
        self._active_target_point_pub = self.create_publisher(
            PointStamped,
            "/nav_active_target_point",
            latched_qos,
        )
        self._active_mission_id_pub = self.create_publisher(
            UInt32,
            "/nav_active_mission_id",
            latched_qos,
        )

        self.create_subscription(PoseStamped, "/goal_pose", self._goal_callback, latched_qos)
        self.create_subscription(PointStamped, "/target_point", self._target_point_callback, 10)
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self._amcl_callback,
            10,
        )
        self.create_subscription(Path, "/plan", self._plan_callback, 10)
        self.create_subscription(Path, "/received_global_plan", self._received_plan_callback, 10)
        self.create_subscription(String, "/manual_nav/status", self._nav_status_callback, 10)
        self.create_timer(self._tick_period_sec, self._tick)

        self._last_goal_key: Optional[tuple[float, ...]] = None
        self._last_goal_received_at = 0.0
        self._latest_target_point: Optional[PointStamped] = None
        self._latest_target_received_at = 0.0
        self._active_target_point: Optional[PointStamped] = None
        self._latest_amcl_pose: Optional[PoseWithCovarianceStamped] = None
        self._active_nav2_goal: Optional[ClientGoalHandle] = None
        self._pending_nav2_goal: Optional[tuple[int, str, PoseStamped, str]] = None
        self._nav2_goal_sent = False
        self._mission_id = 0
        self._phase = self.APPROACH
        self._manual_align_complete_at: Optional[float] = None
        self._nav2_success_timeout_logged = False
        self._phase2_nav_succeeded = False

        self.get_logger().info(
            "nav_goal_bridge_node ready: /goal_pose + /target_point -> /navigate_to_pose"
        )

    @staticmethod
    def _pose_key(msg: PoseStamped) -> tuple[float, float, float, float]:
        return (
            round(float(msg.pose.position.x), 3),
            round(float(msg.pose.position.y), 3),
            round(float(msg.pose.orientation.z), 3),
            round(float(msg.pose.orientation.w), 3),
        )

    @staticmethod
    def _point_key(msg: PointStamped) -> tuple[float, float, float]:
        return (
            round(float(msg.point.x), 3),
            round(float(msg.point.y), 3),
            round(float(msg.point.z), 3),
        )

    @staticmethod
    def _yaw_to_quaternion(yaw: float) -> tuple[float, float]:
        return math.sin(yaw / 2.0), math.cos(yaw / 2.0)

    def _target_point_callback(self, msg: PointStamped) -> None:
        if msg.header.frame_id != "map":
            self.get_logger().warn(
                f"Ignoring /target_point with unsupported frame_id='{msg.header.frame_id}'"
            )
            return

        self._latest_target_point = msg
        self._latest_target_received_at = time.monotonic()

    def _goal_callback(self, msg: PoseStamped) -> None:
        target_point = self._get_fresh_target_point()
        if target_point is None:
            self.get_logger().warn(
                "Rejecting /goal_pose because /target_point is missing or stale"
            )
            return

        goal_key = self._pose_key(msg) + self._point_key(target_point)
        now = time.monotonic()
        if (
            goal_key == self._last_goal_key
            and now - self._last_goal_received_at <= self._duplicate_goal_window_sec
        ):
            return

        self._mission_id += 1
        self._last_goal_key = goal_key
        self._last_goal_received_at = now
        self._active_target_point = self._copy_point(target_point)
        self._phase = self.APPROACH
        self._manual_align_complete_at = None
        self._nav2_success_timeout_logged = False
        self._phase2_nav_succeeded = False

        self._publish_mission_context()
        self._clear_received_global_plan()
        self._cancel_active_nav2_goal()
        self._queue_nav2_goal(
            mission_id=self._mission_id,
            phase=self.APPROACH,
            pose=self._copy_pose(msg),
            behavior_tree=self._phase1_bt_xml,
        )
        self.get_logger().info(
            "Accepted new mission: "
            f"goal=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}) "
            f"target=({target_point.point.x:.3f}, {target_point.point.y:.3f})"
        )

    def _publish_mission_context(self) -> None:
        if self._active_target_point is None:
            return

        mission_msg = UInt32()
        mission_msg.data = self._mission_id
        self._active_target_point_pub.publish(self._active_target_point)
        self._active_mission_id_pub.publish(mission_msg)

    def _queue_nav2_goal(
        self,
        *,
        mission_id: int,
        phase: str,
        pose: PoseStamped,
        behavior_tree: str,
    ) -> None:
        self._pending_nav2_goal = (mission_id, phase, pose, behavior_tree)
        self._nav2_goal_sent = False

    def _amcl_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self._latest_amcl_pose = msg

    def _plan_callback(self, msg: Path) -> None:
        self._received_plan_pub.publish(msg)

    def _received_plan_callback(self, msg: Path) -> None:
        if msg.poses:
            self.get_logger().debug(
                f"received_global_plan updated with {len(msg.poses)} poses"
            )

    def _nav_status_callback(self, msg: String) -> None:
        mission_id, phase = self._parse_nav_status(msg.data)
        if mission_id != self._mission_id:
            return

        if phase == "APPROACH_COMPLETE" and self._phase == self.APPROACH:
            self._start_phase2_navigation()
            return

        if phase == self.ALIGN_COMPLETE:
            self._manual_align_complete_at = time.monotonic()

    def _parse_nav_status(self, payload: str) -> tuple[int, str]:
        try:
            mission_text, phase = payload.split(":", maxsplit=1)
            return int(mission_text), phase
        except ValueError:
            self.get_logger().warn(f"Ignoring malformed manual_nav status: {payload}")
            return -1, ""

    def _start_phase2_navigation(self) -> None:
        if self._latest_amcl_pose is None or self._active_target_point is None:
            self.get_logger().warn(
                "Cannot enter phase 2 because AMCL pose or target point is unavailable"
            )
            return

        current_position = self._latest_amcl_pose.pose.pose.position
        target_point = self._active_target_point.point
        yaw = math.atan2(
            target_point.y - current_position.y,
            target_point.x - current_position.x,
        )
        qz, qw = self._yaw_to_quaternion(yaw)

        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = current_position.x
        pose.pose.position.y = current_position.y
        pose.pose.position.z = current_position.z
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        self._phase = self.ALIGN
        self._manual_align_complete_at = None
        self._nav2_success_timeout_logged = False
        self._phase2_nav_succeeded = False
        self._cancel_active_nav2_goal()
        self._queue_nav2_goal(
            mission_id=self._mission_id,
            phase=self.ALIGN,
            pose=pose,
            behavior_tree=self._phase2_bt_xml,
        )
        self.get_logger().info(
            "Phase 2 queued: "
            f"pose=({pose.pose.position.x:.3f}, {pose.pose.position.y:.3f}) "
            f"yaw={math.degrees(yaw):.1f}deg"
        )

    def _get_fresh_target_point(self) -> Optional[PointStamped]:
        if self._latest_target_point is None:
            return None
        if time.monotonic() - self._latest_target_received_at > self._target_point_timeout_sec:
            return None
        return self._latest_target_point

    def _clear_received_global_plan(self) -> None:
        empty = Path()
        empty.header.frame_id = "map"
        self._received_plan_pub.publish(empty)

    def _cancel_active_nav2_goal(self) -> None:
        if self._active_nav2_goal is not None:
            self._active_nav2_goal.cancel_goal_async()
            self._active_nav2_goal = None

    def _tick(self) -> None:
        if self._pending_nav2_goal is not None and not self._nav2_goal_sent:
            self._try_send_nav2_goal()

        if (
            self._manual_align_complete_at is not None
            and not self._phase2_nav_succeeded
            and not self._nav2_success_timeout_logged
            and time.monotonic() - self._manual_align_complete_at > self._nav2_success_timeout_sec
        ):
            self._nav2_success_timeout_logged = True
            self.get_logger().error(
                "manual_nav completed final alignment, but Nav2 did not report success "
                f"within {self._nav2_success_timeout_sec:.1f}s. "
                "Check tolerance consistency between manual_nav and Nav2 goal checkers."
            )

    def _try_send_nav2_goal(self) -> None:
        if self._pending_nav2_goal is None:
            return
        if not self._navigate_client.wait_for_server(timeout_sec=0.1):
            return

        mission_id, phase, pose, behavior_tree = self._pending_nav2_goal
        goal = NavigateToPose.Goal()
        goal.pose = pose
        goal.behavior_tree = behavior_tree
        future = self._navigate_client.send_goal_async(goal)
        future.add_done_callback(
            partial(self._handle_nav2_goal_response, mission_id, phase)
        )
        self._nav2_goal_sent = True
        self.get_logger().info(
            f"Forwarded mission {mission_id} {phase} goal to /navigate_to_pose"
        )

    def _handle_nav2_goal_response(self, mission_id: int, phase: str, future) -> None:
        if mission_id != self._mission_id:
            return

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self._nav2_goal_sent = False
            self.get_logger().warn(
                f"/navigate_to_pose rejected mission {mission_id} {phase}; will retry"
            )
            return

        self._active_nav2_goal = goal_handle
        self._pending_nav2_goal = None
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            partial(self._handle_nav2_result, mission_id, phase)
        )

    def _handle_nav2_result(self, mission_id: int, phase: str, future) -> None:
        try:
            result = future.result()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.get_logger().error(f"/navigate_to_pose result error: {exc}")
            return

        if mission_id != self._mission_id:
            return

        self._active_nav2_goal = None
        status = int(result.status)
        if phase == self.ALIGN and status == GoalStatus.STATUS_SUCCEEDED:
            self._phase2_nav_succeeded = True

        self.get_logger().info(
            f"/navigate_to_pose finished mission {mission_id} {phase} with status={status}"
        )

    @staticmethod
    def _copy_pose(msg: PoseStamped) -> PoseStamped:
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose
        return pose

    @staticmethod
    def _copy_point(msg: PointStamped) -> PointStamped:
        point = PointStamped()
        point.header = msg.header
        point.point = msg.point
        return point


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
