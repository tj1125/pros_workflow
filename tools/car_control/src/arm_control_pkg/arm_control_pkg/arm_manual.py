import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ManualControlNode(Node):
    def __init__(self, arm_commute_node, arm_angle_control_node, arm_params):
        super().__init__("manual_arm_control_node")
        self.arm_commute_node = arm_commute_node
        self.arm_angle_control_node = arm_angle_control_node
        self.arm_params = arm_params.get_arm_params()

        self.subscription = self.create_subscription(
            String, "arm_control_signal", self.arm_control_signal_callback, 10
        )

    def arm_control_signal_callback(self, msg):
        mode_str, key_str = self.parse_control_signal(msg.data)
        if mode_str is None or key_str is None or not mode_str.isdigit():
            return

        index = int(mode_str)
        key = key_str.lower()
        angle_step = self.arm_params["global"]["angle_step"]

        if key == "i":
            self.arm_angle_control_node.arm_increase_decrease(index, angle_step)
        elif key == "k":
            self.arm_angle_control_node.arm_increase_decrease(index, -angle_step)
        elif key == "b":
            self.arm_angle_control_node.arm_default_change()
        else:
            return

        self.arm_commute_node.publish_arm_angle()

    def parse_control_signal(self, signal_str: str):
        parts = [s.strip() for s in signal_str.split(":")]
        if len(parts) >= 2:
            return parts[0], parts[1].lower()
        return None, None
