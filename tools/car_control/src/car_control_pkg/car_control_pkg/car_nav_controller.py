from __future__ import annotations

from typing import Optional

from action_interface.action import NavGoal

from car_control_pkg.nav2_utils import cal_distance, calculate_diff_angle


class NavigationController:
    def __init__(self, car_control_node):
        self.car_control_node = car_control_node
        self.approach_stop_xy_tolerance_m = float(
            self.car_control_node.get_parameter("approach_stop_xy_tolerance_m").value
        )
        self.reset_index()

    def _get_context(self):
        car_position_msg, car_orientation_msg = (
            self.car_control_node.get_car_position_and_orientation()
        )
        goal_position_msg = self.car_control_node.get_goal_pose()

        if not car_position_msg or not goal_position_msg:
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
        return car_position, car_orientation, goal_position

    def reset_index(self):
        self.index = 0

    def manual_nav(self):
        result = self._get_context()
        if isinstance(result, NavGoal.Result):
            return result

        car_position, car_orientation, goal_position = result
        path_points = self.car_control_node.get_path_points()
        if not path_points:
            self.car_control_node.publish_control("STOP")
            return NavGoal.Result(
                success=False, message="No path points available for navigation"
            )

        target_distance = cal_distance(car_position, goal_position)
        if target_distance <= self.approach_stop_xy_tolerance_m:
            self.car_control_node.publish_control("STOP")
            return NavGoal.Result(
                success=True,
                message="Navigation goal reached successfully. Final distance",
            )

        target_point = self.get_next_target_point(
            car_position=car_position, path_points=path_points
        )
        if target_point is None:
            self.car_control_node.publish_control("STOP")
            return NavGoal.Result(
                success=False, message="No valid target point found on global plan"
            )

        diff_angle = calculate_diff_angle(car_position, car_orientation, target_point)
        action_key = self.choose_path_action(diff_angle)
        self.car_control_node.publish_control(action_key)
        return None

    @staticmethod
    def choose_path_action(diff_angle):
        if -10 < diff_angle < 10:
            return "FORWARD"
        if -180 < diff_angle <= -10:
            return "CLOCKWISE_ROTATION"
        if 10 <= diff_angle < 180:
            return "COUNTERCLOCKWISE_ROTATION"
        return "STOP"

    def get_next_target_point(
        self, car_position, path_points, min_required_distance=0.5
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
