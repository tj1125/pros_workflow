import json
import math

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
        if self._handle_structured_control_signal(msg.data):
            return

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

    def _handle_structured_control_signal(self, signal_str: str) -> bool:
        try:
            payload = json.loads(signal_str)
        except json.JSONDecodeError:
            return False

        if not isinstance(payload, dict):
            self.get_logger().warn("Ignoring structured arm_control_signal: payload must be an object.")
            return True

        command = str(payload.get("command", "")).strip().lower()
        if command in {"set_joint_positions_rad", "set_arm_angles_rad"}:
            raw_positions = payload.get("positions", payload.get("joint_positions_rad"))
            joint_positions = self._positions_to_degrees(raw_positions, radians=True)
        elif command in {"set_joint_positions_deg", "set_arm_angles_deg"}:
            raw_positions = payload.get("positions", payload.get("joint_positions_deg"))
            joint_positions = self._positions_to_degrees(raw_positions, radians=False)
        elif command in {"set_joint_position_rad", "set_joint_angle_rad"}:
            return self._handle_single_joint_position(payload, radians=True)
        elif command in {"set_joint_position_deg", "set_joint_angle_deg"}:
            return self._handle_single_joint_position(payload, radians=False)
        elif command in {"reset", "default"}:
            self.arm_angle_control_node.arm_default_change()
            self.arm_commute_node.publish_arm_angle()
            return True
        else:
            self.get_logger().warn(f"Ignoring unknown structured arm command: {command}")
            return True

        if joint_positions is None:
            return True

        expected_count = int(self.arm_params["global"]["joints_count"])
        if len(joint_positions) != expected_count:
            self.get_logger().error(
                "Ignoring structured arm command with unexpected joint count: "
                f"{len(joint_positions)} != {expected_count}"
            )
            return True

        self.arm_angle_control_node.arm_all_change(joint_positions)
        self.arm_commute_node.publish_arm_angle()
        return True

    def _positions_to_degrees(self, raw_positions, *, radians: bool):
        if not isinstance(raw_positions, list):
            self.get_logger().error("Structured arm command needs a positions list.")
            return None

        try:
            values = [float(value) for value in raw_positions]
        except (TypeError, ValueError):
            self.get_logger().error("Structured arm command positions must be numeric.")
            return None

        if radians:
            return [math.degrees(value) for value in values]
        return values

    def _handle_single_joint_position(self, payload, *, radians: bool) -> bool:
        raw_index = payload.get("index", payload.get("joint_index"))
        raw_position = payload.get(
            "position",
            payload.get("angle", payload.get("value")),
        )
        try:
            joint_index = int(raw_index)
            joint_position = float(raw_position)
        except (TypeError, ValueError):
            self.get_logger().error("Single-joint arm command needs numeric index and position.")
            return True

        expected_count = int(self.arm_params["global"]["joints_count"])
        if joint_index < 0 or joint_index >= expected_count:
            self.get_logger().error(
                "Ignoring single-joint arm command with invalid index: "
                f"{joint_index} not in [0, {expected_count - 1}]"
            )
            return True

        if radians:
            joint_position = math.degrees(joint_position)

        self.arm_angle_control_node.arm_index_change(joint_index, joint_position)
        self.arm_commute_node.publish_arm_angle()
        return True

    def parse_control_signal(self, signal_str: str):
        parts = [s.strip() for s in signal_str.split(":")]
        if len(parts) >= 2:
            return parts[0], parts[1].lower()
        return None, None
