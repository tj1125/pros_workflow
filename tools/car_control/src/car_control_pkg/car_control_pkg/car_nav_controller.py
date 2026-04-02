from __future__ import annotations

from typing import Optional

from action_interface.action import NavGoal

from car_control_pkg.nav2_utils import cal_distance, calculate_diff_angle


class NavigationController:
    APPROACH = "APPROACH"
    ALIGN = "ALIGN"

    def __init__(self, car_control_node):
        self.car_control_node = car_control_node
        self.approach_stop_xy_tolerance_m = float(
            self.car_control_node.get_parameter("approach_stop_xy_tolerance_m").value
        )
        self.align_stop_yaw_tolerance_deg = float(
            self.car_control_node.get_parameter("align_stop_yaw_tolerance_deg").value
        )
        self.align_stable_cycles = int(
            self.car_control_node.get_parameter("align_stable_cycles").value
        )
        self.reset_index()
        self._reset_phase_state()

    def _reset_phase_state(self) -> None:
        self.phase = self.APPROACH
        self._align_stable_count = 0
        self._mission_id = -1
        self.car_control_node.publish_nav_phase(self.APPROACH)

    def _sync_mission_state(self) -> None:
        mission_id = self.car_control_node.get_active_mission_id()
        if mission_id != self._mission_id:
            self._mission_id = mission_id
            self.reset_index()
            self.phase = self.APPROACH
            self._align_stable_count = 0
            self.car_control_node.publish_nav_phase(self.APPROACH)

    def _get_context(self):
        car_position_msg, car_orientation_msg = (
            self.car_control_node.get_car_position_and_orientation()
        )
        goal_position_msg = self.car_control_node.get_goal_pose()
        active_target_point = self.car_control_node.get_active_target_point()

        if not car_position_msg or not goal_position_msg:
            self.car_control_node.publish_control("STOP")
            message = (
                "Cannot obtain car position data (localization lost/stale)"
                if not car_position_msg
                else "No goal pose defined for navigation"
            )
            return NavGoal.Result(success=False, message=message)

        if active_target_point is None:
            self.car_control_node.publish_control("STOP")
            return NavGoal.Result(
                success=False, message="No active target point defined for navigation"
            )

        car_position = [car_position_msg.x, car_position_msg.y]
        car_orientation = [car_orientation_msg.z, car_orientation_msg.w]
        goal_position = [goal_position_msg.x, goal_position_msg.y]
        return car_position, car_orientation, goal_position, active_target_point

    def reset_index(self):
        self.index = 0

    def manual_nav(self):
        self._sync_mission_state()
        result = self._get_context()
        if isinstance(result, NavGoal.Result):
            return result

        car_position, car_orientation, goal_position, active_target_point = result

        if self.phase == self.APPROACH:
            return self._run_approach_phase(
                car_position=car_position,
                car_orientation=car_orientation,
                goal_position=goal_position,
            )

        return self._run_align_phase(
            car_position=car_position,
            car_orientation=car_orientation,
            active_target_point=active_target_point,
        )

    def _run_approach_phase(self, car_position, car_orientation, goal_position):
        path_points = self.car_control_node.get_path_points()
        if not path_points:
            self.car_control_node.publish_control("STOP")
            return NavGoal.Result(
                success=False, message="No path points available for navigation"
            )

        target_distance = cal_distance(car_position, goal_position)
        if target_distance <= self.approach_stop_xy_tolerance_m:
            self.phase = self.ALIGN
            self._align_stable_count = 0
            self.car_control_node.publish_control("STOP")
            self.car_control_node.publish_nav_phase("APPROACH_COMPLETE")
            self.car_control_node.publish_nav_phase(self.ALIGN)
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
        action_key = self.choose_path_action(diff_angle)
        self.car_control_node.publish_control(action_key)
        return None

    def _run_align_phase(self, car_position, car_orientation, active_target_point):
        yaw_error = calculate_diff_angle(
            car_position, car_orientation, active_target_point[:2]
        )

        if abs(yaw_error) <= self.align_stop_yaw_tolerance_deg:
            self._align_stable_count += 1
            self.car_control_node.publish_control("STOP")
            if self._align_stable_count >= self.align_stable_cycles:
                self.car_control_node.publish_nav_phase("ALIGN_COMPLETE")
                return NavGoal.Result(
                    success=True,
                    message=(
                        "Navigation goal reached successfully. "
                        f"Final yaw error {yaw_error:.2f} deg"
                    ),
                )
            return None

        self._align_stable_count = 0
        action_key = self.choose_rotation_action(yaw_error)
        self.car_control_node.publish_control(action_key)
        self.car_control_node.publish_nav_phase(self.ALIGN)
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

    @staticmethod
    def choose_rotation_action(diff_angle):
        if diff_angle < 0:
            return "CLOCKWISE_ROTATION"
        if diff_angle > 0:
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
