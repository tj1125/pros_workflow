import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
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
