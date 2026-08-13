import functools
import math
import threading
import time

from action_interface.action import ArmGoal
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_msgs.msg import Float32


class ArmActionServer(Node):
    def __init__(self, arm_commute_node, arm_auto_controller):
        super().__init__("arm_action_server_node")
        self._callback_group = ReentrantCallbackGroup()
        self._cube_z_distance_lock = threading.Lock()
        self._latest_cube_z_distance_m = None
        self._latest_cube_z_distance_received_at = None
        self._action_server = ActionServer(
            self,
            ArmGoal,
            "arm_action_server",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self._callback_group,
        )
        self._cube_z_distance_sub = self.create_subscription(
            Float32,
            "/cube_z_distance",
            self._cube_z_distance_callback,
            10,
            callback_group=self._callback_group,
        )
        self.arm_commute_node = arm_commute_node
        self.arm_auto_controller = arm_auto_controller

    def _cube_z_distance_callback(self, msg):
        try:
            distance_m = float(msg.data)
        except (TypeError, ValueError):
            return
        if not math.isfinite(distance_m):
            return
        with self._cube_z_distance_lock:
            self._latest_cube_z_distance_m = distance_m
            self._latest_cube_z_distance_received_at = time.monotonic()

    def latest_cube_z_distance_m(self, *, since_time_sec=None):
        with self._cube_z_distance_lock:
            if self._latest_cube_z_distance_m is None:
                return None
            received_at = float(self._latest_cube_z_distance_received_at or 0.0)
            if since_time_sec is not None and received_at < float(since_time_sec):
                return None
            return float(self._latest_cube_z_distance_m), received_at

    def goal_callback(self, goal_request):
        self.get_logger().info(f"Received arm action request: {goal_request.mode}")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("Received arm action cancel request")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        mode = goal_handle.request.mode
        self.get_logger().info(f"Executing arm action in mode: {mode}")
        arm_auto_method = self._select_arm_auto_method(goal_handle.request)

        if arm_auto_method is None:
            result = ArmGoal.Result()
            result.success = False
            result.message = f"Unknown mode: {mode}"
            goal_handle.abort()
            return result

        if goal_handle.is_cancel_requested:
            result = ArmGoal.Result(success=False, message="Canceled by user")
            goal_handle.canceled()
            return result

        result = arm_auto_method()
        if not isinstance(result, ArmGoal.Result):
            result = ArmGoal.Result(success=False, message=f"Mode {mode} returned no result")

        feedback = ArmGoal.Feedback()
        feedback.distance_to_goal = 0.0
        goal_handle.publish_feedback(feedback)

        if result.success:
            self.get_logger().info(f"Arm action {mode} succeeded: {result.message}")
            goal_handle.succeed()
        else:
            self.get_logger().error(f"Arm action {mode} failed: {result.message}")
            goal_handle.abort()
        return result

    def _select_arm_auto_method(self, goal_request):
        mode = str(goal_request.mode)
        if mode == "wave":
            return self.arm_auto_controller.arm_wave
        if mode == "catch":
            return self.arm_auto_controller.catch
        if mode == "arm_ik_move":
            return self.arm_auto_controller.arm_ik_move
        if mode == "test":
            return self.arm_auto_controller.test
        if mode == "look_up":
            return self.arm_auto_controller.look_up
        if mode == "init_pose":
            return self.arm_auto_controller.init_pose
        if mode == "open_gripper":
            return self.arm_auto_controller.open_gripper
        if mode == "close_gripper":
            return self.arm_auto_controller.close_gripper
        if mode == "set_joint_position":
            return functools.partial(
                self.arm_auto_controller.set_joint_position_rad,
                joint_index=int(getattr(goal_request, "joint_index", 0)),
                position_rad=float(getattr(goal_request, "joint_position_rad", 0.0)),
                settle_sec=max(0.0, float(getattr(goal_request, "joint_settle_sec", 0.0))),
            )
        if mode == "car_grasp_sequence":
            return functools.partial(
                self.arm_auto_controller.car_grasp_sequence,
                target_position=list(getattr(goal_request, "target_position", [])),
                wrist_target_rad=float(getattr(goal_request, "wrist_target_rad", 0.0)),
                wrist_joint_index=int(getattr(goal_request, "wrist_joint_index", 3)),
                gripper_joint_index=int(getattr(goal_request, "gripper_joint_index", 4)),
                gripper_open_rad=float(getattr(goal_request, "gripper_open_rad", 0.0)),
                gripper_close_rad=float(getattr(goal_request, "gripper_close_rad", 0.0)),
                steps=int(getattr(goal_request, "trajectory_steps", 0)),
                waypoint_sleep_sec=float(getattr(goal_request, "waypoint_sleep_sec", 0.0)),
                goal_tolerance_m=float(getattr(goal_request, "goal_tolerance_m", 0.0)),
                joint_state_wait_sec=float(getattr(goal_request, "joint_state_wait_sec", 0.0)),
                joint_command_timeout_sec=float(
                    getattr(goal_request, "joint_command_timeout_sec", 0.0)
                ),
                joint_command_tolerance_rad=float(
                    getattr(goal_request, "joint_command_tolerance_rad", 0.0)
                ),
                joint_command_republish_interval_sec=float(
                    getattr(goal_request, "joint_command_republish_interval_sec", 0.0)
                ),
                gripper_close_delay_sec=float(
                    getattr(goal_request, "gripper_close_delay_sec", 0.0)
                ),
                init_pose_delay_sec=float(getattr(goal_request, "init_pose_delay_sec", 0.0)),
                cube_z_distance_reader=self.latest_cube_z_distance_m,
            )
        if mode in ["up", "down", "right", "left"]:
            return functools.partial(
                self.arm_auto_controller.move_end_effector_direction, direction=mode
            )
        if mode in ["forward", "backward"]:
            return functools.partial(
                self.arm_auto_controller.move_forward_backward, direction=mode
            )
        if mode in ["elbow_forward", "elbow_backward", "elbow_left", "elbow_right"]:
            return functools.partial(
                self.arm_auto_controller.move_elbow_direction, direction=mode
            )
        return None
