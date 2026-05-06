import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint


class ArmCummuteNode(Node):
    def __init__(self, arm_params, arm_angle_control):
        super().__init__("arm_commute_node")
        self.arm_angle_control = arm_angle_control
        self.arm_params = arm_params.get_arm_params()

        self.arm_pub = self.create_publisher(
            JointTrajectoryPoint, self.arm_params["global"]["arm_topic"], 10
        )
        self.latest_joint_positions_rad = None
        self.latest_joint_group_positions_rad = None
        self.joint_state_groups = self._joint_state_groups_from_config()
        joint_state_topic = str(
            self.arm_params["global"].get("joint_state_topic", "/joint_states")
        ).strip() or "/joint_states"
        joint_state_topics = [joint_state_topic]
        if joint_state_topic == "/joint_states":
            joint_state_topics.append("/joint_state")
        self.joint_state_subscriptions = [
            self.create_subscription(
                JointState,
                topic,
                self.joint_state_callback,
                10,
            )
            for topic in dict.fromkeys(joint_state_topics)
        ]

        self.latest_imu_data = None
        self.imu_sub = self.create_subscription(
            Imu,
            self.arm_params["global"]["imu_receive_topic"],
            self.imu_callback,
            10,
        )

        self.object_coordinates = {}
        self.yolo_object_offset_sub = self.create_subscription(
            String,
            self.arm_params["global"]["yolo_object_offset_receive_topic"],
            self.yolo_object_offset_callback,
            10,
        )

    def imu_callback(self, msg: Imu):
        self.latest_imu_data = msg

    def _joint_state_groups_from_config(self):
        raw_names = self.arm_params["global"].get("joint_state_names", [])
        groups = []
        if not isinstance(raw_names, list):
            return groups
        for raw_name in raw_names:
            if isinstance(raw_name, list):
                group = tuple(str(name).strip() for name in raw_name if str(name).strip())
            else:
                group = (str(raw_name).strip(),)
            if group:
                groups.append(group)
        return groups

    def _map_joint_state_positions(self, msg: JointState, joint_positions):
        if not self.joint_state_groups or not msg.name:
            return list(joint_positions), [[float(value)] for value in joint_positions]

        name_to_position = {
            str(name): float(joint_positions[index])
            for index, name in enumerate(msg.name)
            if index < len(joint_positions)
        }
        mapped_positions = []
        group_positions = []
        for group in self.joint_state_groups:
            if not all(name in name_to_position for name in group):
                return None, None
            positions = [name_to_position[name] for name in group]
            group_positions.append(positions)
            mapped_positions.append(sum(positions) / len(positions))
        return mapped_positions, group_positions

    def joint_state_callback(self, msg: JointState):
        try:
            joint_positions = [float(value) for value in msg.position]
        except (TypeError, ValueError):
            return
        if not joint_positions or not all(math.isfinite(value) for value in joint_positions):
            return
        mapped_positions, group_positions = self._map_joint_state_positions(
            msg, joint_positions
        )
        if mapped_positions is None or group_positions is None:
            return
        if not all(math.isfinite(value) for value in mapped_positions):
            return
        self.latest_joint_positions_rad = mapped_positions
        self.latest_joint_group_positions_rad = group_positions

    def get_latest_joint_positions_rad(self, *, min_joint_count=0):
        if self.latest_joint_positions_rad is None:
            return None
        if len(self.latest_joint_positions_rad) < int(min_joint_count):
            return None
        return list(self.latest_joint_positions_rad)

    def get_latest_joint_group_positions_rad(self, joint_index):
        if self.latest_joint_group_positions_rad is None:
            return None
        joint_index = int(joint_index)
        if joint_index < 0 or joint_index >= len(self.latest_joint_group_positions_rad):
            return None
        return list(self.latest_joint_group_positions_rad[joint_index])

    def get_latest_imu_data(self):
        if self.latest_imu_data is None:
            return None
        orientation = self.latest_imu_data.orientation
        return [orientation.x, orientation.y, orientation.z, orientation.w]

    def yolo_object_offset_callback(self, msg: String):
        try:
            object_list = json.loads(msg.data)
            new_coordinates = {}
            for item in object_list:
                if isinstance(item, dict) and "label" in item and "offset_flu" in item:
                    label = item["label"]
                    coordinates = item["offset_flu"]
                    if isinstance(coordinates, list) and len(coordinates) == 3:
                        try:
                            new_coordinates[label] = [float(c) for c in coordinates]
                        except (ValueError, TypeError):
                            self.get_logger().warn(
                                f"Invalid coordinate format for label '{label}': {coordinates}"
                            )
            self.object_coordinates = new_coordinates
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"Failed to decode JSON string: {exc}")
        except Exception as exc:  # pragma: no cover - runtime protection
            self.get_logger().error(f"Error processing YOLO offset message: {exc}")

    def get_latest_object_coordinates(self, label: str = None):
        if label is None:
            return self.object_coordinates
        return self.object_coordinates.get(label, None)

    def degrees_to_radians(self, degree_positions):
        try:
            positions_array = np.array(degree_positions, dtype=float)
            return np.deg2rad(positions_array).tolist()
        except (ValueError, TypeError):
            radian_positions = []
            for pos in degree_positions:
                try:
                    radian_positions.append(float(pos) * math.pi / 180.0)
                except (ValueError, TypeError):
                    self.get_logger().error(f"Invalid angle value: {pos}")
                    radian_positions.append(0.0)
            return radian_positions

    def publish_arm_angle(self):
        joint_positions = self.arm_angle_control.get_arm_angles()
        msg = JointTrajectoryPoint()
        positions = self.degrees_to_radians(joint_positions)
        zero_vec = [0.0] * len(positions)
        msg.positions = positions
        msg.velocities = zero_vec
        msg.accelerations = zero_vec
        msg.effort = zero_vec
        msg.time_from_start.sec = 0
        msg.time_from_start.nanosec = 0
        self.arm_pub.publish(msg)
