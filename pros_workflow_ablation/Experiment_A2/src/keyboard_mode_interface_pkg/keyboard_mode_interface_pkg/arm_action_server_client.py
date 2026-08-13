from action_interface.action import ArmGoal
from keyboard_mode_interface_pkg.base_action_client import BaseActionClient


class ArmActionClient(BaseActionClient):
    def __init__(self, node):
        """
        Initializes the ArmActionClient.

        Args:
            node: The rclpy.node.Node instance to use for communication.
        """
        super().__init__(
            node=node,
            action_type=ArmGoal,
            server_name="arm_action_server",
            client_name="ArmActionClient",
        )

    def _create_goal_msg(
        self,
        mode,
        *,
        target_position=None,
        trajectory_steps=0,
        waypoint_sleep_sec=0.0,
        goal_tolerance_m=0.0,
    ):
        goal_msg = ArmGoal.Goal()
        goal_msg.mode = mode
        if target_position is not None:
            goal_msg.target_position = [float(value) for value in target_position]
        goal_msg.trajectory_steps = int(trajectory_steps)
        goal_msg.waypoint_sleep_sec = float(waypoint_sleep_sec)
        goal_msg.goal_tolerance_m = float(goal_tolerance_m)
        return goal_msg

    # Public API methods with descriptive names
    def send_arm_mode(self, mode):
        """Send an arm mode goal"""
        return self.send_goal(mode)

    def send_arm_reach_goal(
        self,
        target_position,
        *,
        trajectory_steps=50,
        waypoint_sleep_sec=0.1,
        goal_tolerance_m=0.03,
    ):
        """Move the gripper end effector to a PyBullet/world-frame XYZ target."""
        return self.send_goal(
            "reach_goal",
            target_position=target_position,
            trajectory_steps=trajectory_steps,
            waypoint_sleep_sec=waypoint_sleep_sec,
            goal_tolerance_m=goal_tolerance_m,
        )

    def cancel_arm(self):
        """Cancel the current arm goal if one exists"""
        return self.cancel_goal()
