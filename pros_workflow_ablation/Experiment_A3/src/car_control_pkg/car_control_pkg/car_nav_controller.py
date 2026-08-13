from __future__ import annotations

import math
from typing import Optional

from action_interface.action import NavGoal

from car_control_pkg.nav2_utils import (
    cal_distance,
    calculate_diff_angle,
    calculate_goal_heading_error,
)


class NavigationController:
    def __init__(self, car_control_node):
        self.car_control_node = car_control_node
        self.approach_stop_xy_tolerance_m = float(
            self.car_control_node.get_parameter("approach_stop_xy_tolerance_m").value
        )
        self.align_stop_yaw_tolerance_rad = float(
            self.car_control_node.get_parameter("align_stop_yaw_tolerance_rad").value
        )
        self.slow_approach_distance_m = float(
            self.car_control_node.get_parameter("slow_approach_distance_m").value
        )
        self.reset_index()
        self._active_goal_key = None

    def _get_context(self):
        car_position_msg, car_orientation_msg = (
            self.car_control_node.get_car_position_and_orientation()
        )
        goal_position_msg = self.car_control_node.get_goal_pose()
        goal_orientation_msg = self.car_control_node.get_goal_orientation()

        if not car_position_msg or not goal_position_msg or not goal_orientation_msg:
            self.car_control_node.publish_control("STOP")
            message = (
                "Cannot obtain car position data (localization lost/stale)"
                if not car_position_msg
                else "No goal pose defined for navigation"
            )
            return NavGoal.Result(success=False, message=message)

        car_position = [car_position_msg.x, car_position_msg.y]
        car_orientation = [car_orientation_msg.z, car_orientation_msg.w]
        goal_position = [goal_position_msg.x, goal_position_msg.y]
        goal_orientation = [goal_orientation_msg.z, goal_orientation_msg.w]
        return car_position, car_orientation, goal_position, goal_orientation

    def reset_index(self):
        self.index = 0
        self.final_alignment_active = False

    @staticmethod
    def _goal_key(goal_position, goal_orientation):
        return (
            round(float(goal_position[0]), 3),
            round(float(goal_position[1]), 3),
            round(float(goal_orientation[0]), 3),
            round(float(goal_orientation[1]), 3),
        )

    def manual_nav(self):
        result = self._get_context()
        if isinstance(result, NavGoal.Result):
            return result

        car_position, car_orientation, goal_position, goal_orientation = result
        goal_key = self._goal_key(goal_position, goal_orientation)
        if goal_key != self._active_goal_key:
            self.reset_index()
            self._active_goal_key = goal_key

        target_distance = cal_distance(car_position, goal_position)
        if self.final_alignment_active:
            return self._run_final_heading_alignment(
                car_orientation=car_orientation,
                goal_orientation=goal_orientation,
                target_distance=target_distance,
            )

        if target_distance <= self.approach_stop_xy_tolerance_m:
            self.final_alignment_active = True
            return self._run_final_heading_alignment(
                car_orientation=car_orientation,
                goal_orientation=goal_orientation,
                target_distance=target_distance,
            )

        path_points = self.car_control_node.get_path_points()
        if not path_points:
            self.car_control_node.publish_control("STOP")
            return None

        target_point = self.get_next_target_point(
            car_position=car_position, path_points=path_points
        )
        if target_point is None:
            self.car_control_node.publish_control("STOP")
            return NavGoal.Result(
                success=False, message="No valid target point found on global plan"
            )

        diff_angle = calculate_diff_angle(car_position, car_orientation, target_point)
        action_key = self.choose_path_action(
            diff_angle,
            force_slow=target_distance <= self.slow_approach_distance_m,
        )
        self.car_control_node.publish_control(action_key)
        return None

    def _run_final_heading_alignment(
        self,
        *,
        car_orientation,
        goal_orientation,
        target_distance: float,
    ):
        heading_error = calculate_goal_heading_error(car_orientation, goal_orientation)
        if abs(heading_error) <= self.align_stop_yaw_tolerance_rad:
            self.final_alignment_active = False
            self.car_control_node.publish_stop_burst()
            return NavGoal.Result(
                success=True,
                message=(
                    "Navigation goal reached successfully. "
                    f"Final distance {target_distance:.3f} m, "
                    f"heading error {heading_error:.3f} rad"
                ),
            )

        action_key = self.choose_final_align_action(heading_error)
        self.car_control_node.publish_control(action_key)
        return None

    @staticmethod
    def choose_path_action(diff_angle, force_slow=False):
        abs_diff = abs(diff_angle)
        heading_tolerance_deg = 10.0

        if force_slow:
            if abs_diff <= heading_tolerance_deg:
                return "FORWARD_SLOW"
            if -180 < diff_angle < -heading_tolerance_deg:
                return "CLOCKWISE_ROTATION_SLOW"
            if heading_tolerance_deg < diff_angle < 180:
                return "COUNTERCLOCKWISE_ROTATION_SLOW"
            return "STOP"

        if abs_diff <= heading_tolerance_deg:
            return "FORWARD"
        if -180 < diff_angle < -heading_tolerance_deg:
            return "CLOCKWISE_ROTATION"
        if heading_tolerance_deg < diff_angle < 180:
            return "COUNTERCLOCKWISE_ROTATION"
        return "STOP"

    @staticmethod
    def choose_rotation_action(diff_angle):
        abs_diff = abs(diff_angle)
        if abs_diff <= math.radians(3.0):
            return (
                "CLOCKWISE_ROTATION_SLOW"
                if diff_angle < 0
                else "COUNTERCLOCKWISE_ROTATION_SLOW"
            )
        if abs_diff <= math.radians(10.0):
            return (
                "CLOCKWISE_ROTATION_MEDIAN"
                if diff_angle < 0
                else "COUNTERCLOCKWISE_ROTATION_MEDIAN"
            )
        return "CLOCKWISE_ROTATION" if diff_angle < 0 else "COUNTERCLOCKWISE_ROTATION"

    @staticmethod
    def choose_final_align_action(diff_angle):
        return (
            "CLOCKWISE_ROTATION_SLOW"
            if diff_angle < 0
            else "COUNTERCLOCKWISE_ROTATION_SLOW"
        )

    def get_next_target_point(
        self, car_position, path_points, min_required_distance=0.2
    ) -> Optional[list[float]]:
        """
        Return the next path point at least `min_required_distance` from the car.
        Falls back to the final path point if all intermediate points are too close.
        """
        logger = self.car_control_node.get_logger()

        if not path_points:
            logger.error("Error: No path points available!")
            return None

        for idx in range(self.index, len(path_points)):
            point = path_points[idx]
            try:
                pos = point["position"]
                target_x, target_y = pos[0], pos[1]
            except (KeyError, IndexError, TypeError) as exc:
                logger.error(f"Invalid path point format at index {idx}: {exc}")
                continue

            distance_to_target = cal_distance(car_position, (target_x, target_y))
            if distance_to_target >= min_required_distance:
                self.index = idx
                return [target_x, target_y]

        try:
            last_point = path_points[-1]
            pos = last_point["position"]
            self.index = len(path_points) - 1
            return [pos[0], pos[1]]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error(f"Invalid format for last path point: {exc}")

        logger.warning("No valid target point found.")
        return None
