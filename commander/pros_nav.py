from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
from typing import Optional

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


WHEEL_SPEED = 20.0


ACTION_MAPPINGS = {
    "FORWARD": [WHEEL_SPEED, WHEEL_SPEED, WHEEL_SPEED, WHEEL_SPEED],
    "COUNTERCLOCKWISE_ROTATION": [-WHEEL_SPEED, WHEEL_SPEED, -WHEEL_SPEED, WHEEL_SPEED],
    "CLOCKWISE_ROTATION": [WHEEL_SPEED, -WHEEL_SPEED, WHEEL_SPEED, -WHEEL_SPEED],
    "STOP": [0.0, 0.0, 0.0, 0.0],
}


def _yaw_deg_from_quaternion(z: float, w: float) -> float:
    return degrees(2.0 * atan2(z, w))


def _normalize_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def calculate_diff_angle(
    car_position: tuple[float, float],
    car_orientation: tuple[float, float],
    target_point: tuple[float, float],
) -> float:
    car_yaw = _yaw_deg_from_quaternion(car_orientation[0], car_orientation[1])
    target_yaw = degrees(
        atan2(target_point[1] - car_position[1], target_point[0] - car_position[0])
    )
    return _normalize_angle_deg(target_yaw - car_yaw)


def calculate_goal_heading_error(
    car_orientation: tuple[float, float],
    goal_orientation: tuple[float, float],
) -> float:
    car_yaw = _yaw_deg_from_quaternion(car_orientation[0], car_orientation[1])
    goal_yaw = _yaw_deg_from_quaternion(goal_orientation[0], goal_orientation[1])
    return _normalize_angle_deg(goal_yaw - car_yaw)


@dataclass
class FollowStep:
    action_key: str
    distance_to_goal: float
    arrived: bool = False
    waiting_for_data: bool = False
    detail: str = ""


class ProsPathFollower:
    def __init__(
        self,
        node: Node,
        *,
        goal_tolerance_m: float = 0.35,
        goal_heading_tolerance_deg: float = 5.0,
        min_target_distance_m: float = 0.5,
    ) -> None:
        self.node = node
        self.goal_tolerance_m = goal_tolerance_m
        self.goal_heading_tolerance_deg = goal_heading_tolerance_deg
        self.min_target_distance_m = min_target_distance_m
        self.front_pub = node.create_publisher(Float32MultiArray, "/car_C_front_wheel", 10)
        self.rear_pub = node.create_publisher(Float32MultiArray, "/car_C_rear_wheel", 10)
        self._current_plan_key: Optional[tuple[int, int, int, float, float]] = None
        self._path_index = 0

    def stop(self) -> None:
        self.publish_action("STOP")

    def publish_action(self, action_key: str) -> None:
        velocities = ACTION_MAPPINGS.get(action_key, ACTION_MAPPINGS["STOP"])

        front_msg = Float32MultiArray()
        rear_msg = Float32MultiArray()
        front_msg.data = velocities[0:2]
        rear_msg.data = velocities[2:4]

        self.front_pub.publish(front_msg)
        self.rear_pub.publish(rear_msg)

    def reset_path_tracking(self) -> None:
        self._current_plan_key = None
        self._path_index = 0

    def _plan_key(self, path: Path) -> tuple[int, int, int, float, float]:
        if not path.poses:
            return (0, 0, 0, 0.0, 0.0)
        last_pose = path.poses[-1].pose.position
        return (
            len(path.poses),
            int(path.header.stamp.sec),
            int(path.header.stamp.nanosec),
            round(float(last_pose.x), 3),
            round(float(last_pose.y), 3),
        )

    def _ensure_path_tracking(self, path: Path) -> None:
        plan_key = self._plan_key(path)
        if plan_key != self._current_plan_key:
            self._current_plan_key = plan_key
            self._path_index = 0

    def _get_next_target_point(
        self,
        car_position: tuple[float, float],
        path: Path,
    ) -> Optional[tuple[float, float]]:
        if not path.poses:
            return None

        self._ensure_path_tracking(path)

        for idx in range(self._path_index, len(path.poses)):
            point = path.poses[idx].pose.position
            distance = hypot(car_position[0] - point.x, car_position[1] - point.y)
            if distance >= self.min_target_distance_m:
                self._path_index = idx
                return (float(point.x), float(point.y))

        last_pose = path.poses[-1].pose.position
        self._path_index = max(0, len(path.poses) - 1)
        return (float(last_pose.x), float(last_pose.y))

    @staticmethod
    def _choose_action(diff_angle: float) -> str:
        if -10.0 < diff_angle < 10.0:
            return "FORWARD"
        if -180.0 < diff_angle <= -10.0:
            return "CLOCKWISE_ROTATION"
        if 10.0 <= diff_angle < 180.0:
            return "COUNTERCLOCKWISE_ROTATION"
        return "STOP"

    @staticmethod
    def _choose_rotation_action(diff_angle: float) -> str:
        if diff_angle <= -1.0:
            return "CLOCKWISE_ROTATION"
        if diff_angle >= 1.0:
            return "COUNTERCLOCKWISE_ROTATION"
        return "STOP"

    def follow_step(
        self,
        amcl_pose_msg: Optional[PoseWithCovarianceStamped],
        goal_pose_msg: PoseStamped,
        path: Optional[Path],
    ) -> FollowStep:
        if amcl_pose_msg is None:
            self.stop()
            return FollowStep(
                action_key="STOP",
                distance_to_goal=float("inf"),
                waiting_for_data=True,
                detail="waiting_for_amcl_pose",
            )

        if path is None or not path.poses:
            self.stop()
            return FollowStep(
                action_key="STOP",
                distance_to_goal=float("inf"),
                waiting_for_data=True,
                detail="waiting_for_global_plan",
            )

        car_position_msg = amcl_pose_msg.pose.pose.position
        car_orientation_msg = amcl_pose_msg.pose.pose.orientation
        goal_position_msg = goal_pose_msg.pose.position
        goal_orientation_msg = goal_pose_msg.pose.orientation

        car_position = (float(car_position_msg.x), float(car_position_msg.y))
        car_orientation = (float(car_orientation_msg.z), float(car_orientation_msg.w))
        goal_position = (float(goal_position_msg.x), float(goal_position_msg.y))
        goal_orientation = (float(goal_orientation_msg.z), float(goal_orientation_msg.w))

        distance_to_goal = hypot(
            car_position[0] - goal_position[0],
            car_position[1] - goal_position[1],
        )
        goal_heading_error = calculate_goal_heading_error(car_orientation, goal_orientation)
        if distance_to_goal < self.goal_tolerance_m:
            if abs(goal_heading_error) <= self.goal_heading_tolerance_deg:
                self.stop()
                return FollowStep(
                    action_key="STOP",
                    distance_to_goal=distance_to_goal,
                    arrived=True,
                    detail=(
                        "goal_reached "
                        f"heading_error={goal_heading_error:.1f}"
                    ),
                )

            action_key = self._choose_rotation_action(goal_heading_error)
            self.publish_action(action_key)
            return FollowStep(
                action_key=action_key,
                distance_to_goal=distance_to_goal,
                detail=(
                    "final_heading_align "
                    f"heading_error={goal_heading_error:.1f}"
                ),
            )

        target_point = self._get_next_target_point(car_position, path)
        if target_point is None:
            self.stop()
            return FollowStep(
                action_key="STOP",
                distance_to_goal=distance_to_goal,
                waiting_for_data=True,
                detail="target_point_unavailable",
            )

        diff_angle = calculate_diff_angle(car_position, car_orientation, target_point)
        action_key = self._choose_action(diff_angle)
        self.publish_action(action_key)
        return FollowStep(
            action_key=action_key,
            distance_to_goal=distance_to_goal,
            detail=(
                f"target=({target_point[0]:.2f},{target_point[1]:.2f}) "
                f"diff_angle={diff_angle:.1f} goal_heading_error={goal_heading_error:.1f}"
            ),
        )
