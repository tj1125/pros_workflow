import functools

from action_interface.action import ArmGoal
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node


class ArmActionServer(Node):
    def __init__(self, arm_commute_node, arm_auto_controller):
        super().__init__("arm_action_server_node")
        self._action_server = ActionServer(
            self,
            ArmGoal,
            "arm_action_server",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )
        self.arm_commute_node = arm_commute_node
        self.arm_auto_controller = arm_auto_controller

    def goal_callback(self, goal_request):
        self.get_logger().info(f"Received arm action request: {goal_request.mode}")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("Received arm action cancel request")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        mode = goal_handle.request.mode
        self.get_logger().info(f"Executing arm action in mode: {mode}")
        arm_auto_method = self._select_arm_auto_method(mode)

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
            goal_handle.succeed()
        else:
            goal_handle.abort()
        return result

    def _select_arm_auto_method(self, mode: str):
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
