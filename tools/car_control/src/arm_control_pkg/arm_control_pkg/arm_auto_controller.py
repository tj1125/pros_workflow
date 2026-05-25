import json
import math
import os
import time

from action_interface.action import ArmGoal


class ArmAutoController:
    def __init__(
        self, arm_params, arm_commute_node, pybulletRobotController, arm_agnle_control
    ):
        self.arm_params = arm_params.get_arm_params()
        self.pybullet_robot_controller = pybulletRobotController
        self.arm_commute_node = arm_commute_node
        self.arm_agnle_control = arm_agnle_control
        self.depth = 100.0

    def catch(self):
        while self.depth > 0.3:
            try:
                data = self.arm_commute_node.get_latest_object_coordinates(label="ball")
                if data is None:
                    continue
                self.depth = data[0]
            except Exception:
                continue

        while True:
            if self.follow_obj(label="ball") is True:
                break

        self.depth = 100.0
        data = self.arm_commute_node.get_latest_object_coordinates(label="ball")
        if data is None:
            return ArmGoal.Result(success=False, message="No object detected")

        depth = data[0]
        obj_pos = self.pybullet_robot_controller.markPointInFrontOfEndEffector(
            distance=depth + 0.05, z_offset=0.1
        )
        robot_angle = self.pybullet_robot_controller.generateInterpolatedTrajectory(
            target_position=obj_pos, steps=10
        )
        for angle in robot_angle:
            self.move_real_and_virtual(radian=angle)
            time.sleep(0.1)
        self.grap()
        time.sleep(1.0)
        self.init_pose(grap=True)
        return ArmGoal.Result(success=True, message="success")

    def car2_position(self):
        return None

    def arm_wave(self):
        return ArmGoal.Result(success=False, message="wave is not implemented")

    def arm_ik_move(self):
        target_position = self.pybullet_robot_controller.offset_from_end_effector(
            x_offset=0.0,
            y_offset=0.1,
            z_offset=0.1,
        )
        if target_position is None:
            return ArmGoal.Result(success=False, message="Failed to compute IK target")

        self.pybullet_robot_controller.setJointPosition(
            position=self.pybullet_robot_controller.solveInversePositionKinematics(
                target_position
            )[: len(self.pybullet_robot_controller.controllable_joints)]
        )
        self.pybullet_robot_controller.draw_link_axes(link_name="camera_1")
        return ArmGoal.Result(success=True, message="success")

    def radians_to_degrees(self, radians_list):
        if not isinstance(radians_list, (list, tuple)):
            return []
        try:
            return [math.degrees(rad) for rad in radians_list]
        except TypeError:
            return []

    def _gripper_joint_index(self):
        return 4

    def _wrist_joint_index(self):
        return 3

    def _expected_joint_count(self):
        return int(self.arm_params["global"]["joints_count"])

    def _current_joint_positions_rad(self):
        expected_count = self._expected_joint_count()
        latest_joint_positions = self.arm_commute_node.get_latest_joint_positions_rad(
            min_joint_count=expected_count
        )
        if latest_joint_positions:
            return [float(value) for value in latest_joint_positions[:expected_count]]
        return [
            math.radians(float(joint_angle))
            for joint_angle in self.arm_agnle_control.get_arm_angles()
        ]

    def _with_preserved_grasp_joints(self, radian):
        joint_positions = list(radian)
        current_joint_positions = self._current_joint_positions_rad()
        for joint_index in (self._wrist_joint_index(), self._gripper_joint_index()):
            if (
                0 <= joint_index < len(joint_positions)
                and joint_index < len(current_joint_positions)
            ):
                joint_positions[joint_index] = current_joint_positions[joint_index]
        return joint_positions

    def _sync_virtual_robot_to_current_joint_positions(self):
        current_joint_positions = self._current_joint_positions_rad()
        joint_count = len(self.pybullet_robot_controller.controllable_joints)
        if len(current_joint_positions) >= joint_count:
            self.pybullet_robot_controller.setJointPosition(
                position=current_joint_positions[:joint_count]
            )

    def _joint_reset_positions_rad(self):
        return [
            math.radians(float(self.arm_params["joints_reset"][index]))
            for index in range(self._expected_joint_count())
        ]

    def _wait_for_joint_state_positions(self, *, min_joint_count, timeout_sec):
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        expected_count = self._expected_joint_count()
        while time.monotonic() < deadline:
            positions = self.arm_commute_node.get_latest_joint_positions_rad(
                min_joint_count=int(min_joint_count)
            )
            if positions:
                return [float(value) for value in positions[:expected_count]]
            time.sleep(0.02)
        return None

    def _publish_joint_positions_rad(self, positions_rad):
        positions = [float(value) for value in positions_rad[: self._expected_joint_count()]]
        joint_count = len(self.pybullet_robot_controller.controllable_joints)
        if len(positions) >= joint_count:
            self.pybullet_robot_controller.setJointPosition(position=positions[:joint_count])
        self.arm_agnle_control.joint_positions = self.radians_to_degrees(positions)
        self._publish_robot_arm_positions_rad(positions)

    def _publish_robot_arm_positions_rad(self, positions_rad):
        from trajectory_msgs.msg import JointTrajectoryPoint

        positions = [float(value) for value in positions_rad]
        zero_vec = [0.0] * len(positions)
        message = JointTrajectoryPoint()
        message.positions = positions
        message.velocities = zero_vec
        message.accelerations = zero_vec
        message.effort = zero_vec
        message.time_from_start.sec = 0
        message.time_from_start.nanosec = 0
        self.arm_commute_node.arm_pub.publish(message)

    def _joint_angle_error_rad(self, current_rad, target_rad):
        error = abs(float(current_rad) - float(target_rad))
        wrapped_error = abs((error + math.pi) % (2.0 * math.pi) - math.pi)
        return min(error, wrapped_error)

    def _joint_command_errors_rad(self, current_positions, target_positions, tracked_indices):
        max_errors = {}
        group_errors = {}
        for joint_index in tracked_indices:
            group_positions = self.arm_commute_node.get_latest_joint_group_positions_rad(
                joint_index
            )
            if group_positions:
                errors = [
                    self._joint_angle_error_rad(position, target_positions[joint_index])
                    for position in group_positions
                ]
                group_errors[joint_index] = errors
                max_errors[joint_index] = max(errors)
            else:
                error = self._joint_angle_error_rad(
                    current_positions[joint_index],
                    target_positions[joint_index],
                )
                group_errors[joint_index] = [error]
                max_errors[joint_index] = error
        return max_errors, group_errors

    def _publish_joint_updates_and_wait(
        self,
        *,
        phase,
        joint_updates_rad,
        min_joint_count,
        tolerance_rad,
        timeout_sec,
        republish_interval_sec,
        cube_z_distance_reader=None,
        cube_z_distance_stop_threshold_m=0.0,
        cube_z_distance_since_time_sec=None,
        cube_z_distance_poll_interval_sec=0.1,
    ):
        current_positions = self._wait_for_joint_state_positions(
            min_joint_count=min_joint_count,
            timeout_sec=timeout_sec,
        )
        if current_positions is None:
            return {
                "success": False,
                "phase": str(phase),
                "message": "joint state was not available before publishing command",
            }

        target_positions = list(current_positions)
        for raw_index, raw_position in joint_updates_rad.items():
            joint_index = int(raw_index)
            if joint_index < 0 or joint_index >= len(target_positions):
                return {
                    "success": False,
                    "phase": str(phase),
                    "message": f"joint index {joint_index} is outside joint count {len(target_positions)}",
                }
            target_positions[joint_index] = float(raw_position)

        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        tracked_indices = sorted(int(index) for index in joint_updates_rad.keys())
        published_count = 0
        last_publish_time_sec = -float("inf")
        last_errors = {}
        stop_threshold = max(0.0, float(cube_z_distance_stop_threshold_m))
        poll_interval_sec = max(0.001, float(cube_z_distance_poll_interval_sec))
        last_cube_z_distance_poll_sec = -float("inf")
        since_time_sec = (
            float(cube_z_distance_since_time_sec)
            if cube_z_distance_since_time_sec is not None
            else time.monotonic()
        )

        while True:
            now = time.monotonic()
            if (
                published_count == 0
                or (
                    republish_interval_sec > 0.0
                    and now - last_publish_time_sec >= float(republish_interval_sec)
                )
            ):
                self._publish_joint_positions_rad(target_positions)
                published_count += 1
                last_publish_time_sec = now

            if (
                stop_threshold > 0.0
                and now - last_cube_z_distance_poll_sec >= poll_interval_sec
            ):
                last_cube_z_distance_poll_sec = now
                cube_z_distance_m = self._read_fresh_cube_z_distance(
                    cube_z_distance_reader,
                    since_time_sec=since_time_sec,
                )
                if cube_z_distance_m is not None and cube_z_distance_m < stop_threshold:
                    return {
                        "success": True,
                        "phase": str(phase),
                        "published_count": int(published_count),
                        "joint_errors_rad": last_errors,
                        "max_error_rad": max(last_errors.values()) if last_errors else None,
                        "early_stopped_by_cube_z_distance": True,
                        "early_stop_distance_m": float(cube_z_distance_m),
                        "early_stop_threshold_m": float(stop_threshold),
                        "message": (
                            "joint command early-stopped by cube_z_distance: "
                            f"early_stop_distance_m={cube_z_distance_m:.4f} < "
                            f"threshold_m={stop_threshold:.4f}"
                        ),
                    }

            current_positions = self.arm_commute_node.get_latest_joint_positions_rad(
                min_joint_count=len(target_positions)
            )
            if current_positions:
                current_positions = [float(value) for value in current_positions[: len(target_positions)]]
                last_errors, group_errors = self._joint_command_errors_rad(
                    current_positions, target_positions, tracked_indices
                )
                if all(error <= float(tolerance_rad) for error in last_errors.values()):
                    return {
                        "success": True,
                        "phase": str(phase),
                        "published_count": int(published_count),
                        "joint_errors_rad": last_errors,
                        "joint_group_errors_rad": group_errors,
                        "max_error_rad": max(last_errors.values()) if last_errors else 0.0,
                        "message": "joint command reached tolerance",
                    }

            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)

        return {
            "success": False,
            "phase": str(phase),
            "published_count": int(published_count),
            "joint_errors_rad": last_errors,
            "max_error_rad": max(last_errors.values()) if last_errors else None,
            "message": (
                f"timed out waiting for {phase} to reach "
                f"{float(tolerance_rad):.4f} rad tolerance"
            ),
        }

    def _verify_car_grasp_cube_z_distance(
        self,
        cube_z_distance_reader,
        *,
        since_time_sec,
        threshold_m=0.08,
    ):
        threshold = max(0.0, float(threshold_m))
        if cube_z_distance_reader is None:
            return {
                "success": False,
                "phase": "verify_cube_z_distance",
                "message": "grasp verification failed: no /cube_z_distance reader is configured",
            }
        try:
            reading = cube_z_distance_reader(since_time_sec=since_time_sec)
        except Exception as exc:
            return {
                "success": False,
                "phase": "verify_cube_z_distance",
                "message": f"grasp verification failed: /cube_z_distance read error: {exc}",
            }
        if reading is None:
            return {
                "success": False,
                "phase": "verify_cube_z_distance",
                "threshold_m": threshold,
                "message": "grasp verification failed: no fresh /cube_z_distance received after reset pose",
            }

        try:
            distance_m = float(reading[0])
        except (TypeError, ValueError, IndexError):
            return {
                "success": False,
                "phase": "verify_cube_z_distance",
                "threshold_m": threshold,
                "message": f"grasp verification failed: invalid /cube_z_distance reading: {reading}",
            }
        if not math.isfinite(distance_m):
            return {
                "success": False,
                "phase": "verify_cube_z_distance",
                "threshold_m": threshold,
                "cube_z_distance_m": distance_m,
                "message": "grasp verification failed: /cube_z_distance is not finite",
            }

        success = distance_m < threshold
        comparator = "<" if success else ">="
        return {
            "success": bool(success),
            "phase": "verify_cube_z_distance",
            "threshold_m": threshold,
            "cube_z_distance_m": float(distance_m),
            "message": (
                f"grasp verification {'succeeded' if success else 'failed'}: "
                f"cube_z_distance={distance_m:.4f}m {comparator} {threshold:.4f}m"
            ),
        }

    def _cube_z_distance_early_stop_threshold_m(self):
        raw_threshold = os.getenv(
            "APPROACH_AGENT_CAR_GRASP_EARLY_STOP_CUBE_Z_DISTANCE_M",
            "0.03",
        )
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError):
            return 0.03
        return threshold if math.isfinite(threshold) and threshold > 0.0 else 0.0

    def _cube_z_distance_early_stop_enabled(self):
        raw_enabled = os.getenv(
            "APPROACH_AGENT_CAR_GRASP_EARLY_STOP_ON_CUBE_Z_DISTANCE",
            "1",
        )
        return str(raw_enabled).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
            "",
        }

    def _cube_z_distance_early_stop_poll_interval_sec(self):
        raw_interval = os.getenv(
            "APPROACH_AGENT_CAR_GRASP_EARLY_STOP_POLL_INTERVAL_SEC",
            "0.1",
        )
        try:
            interval_sec = float(raw_interval)
        except (TypeError, ValueError):
            return 0.1
        return interval_sec if math.isfinite(interval_sec) and interval_sec > 0.0 else 0.1

    def _read_fresh_cube_z_distance(self, cube_z_distance_reader, *, since_time_sec):
        if cube_z_distance_reader is None:
            return None
        try:
            reading = cube_z_distance_reader(since_time_sec=since_time_sec)
        except Exception:
            return None
        if reading is None:
            return None
        try:
            distance_m = float(reading[0])
        except (TypeError, ValueError, IndexError):
            return None
        if not math.isfinite(distance_m):
            return None
        return distance_m

    def _wait_or_stop_on_cube_z_distance(
        self,
        cube_z_distance_reader,
        *,
        since_time_sec,
        duration_sec,
        threshold_m,
        poll_interval_sec,
        waypoint_index,
        waypoint_count,
    ):
        deadline = time.monotonic() + max(0.0, float(duration_sec))
        while True:
            cube_z_distance_m = self._read_fresh_cube_z_distance(
                cube_z_distance_reader,
                since_time_sec=since_time_sec,
            )
            if cube_z_distance_m is not None and cube_z_distance_m < float(threshold_m):
                return ArmGoal.Result(
                    success=True,
                    message=(
                        "target move early-stopped by cube_z_distance: "
                        f"early_stop_distance_m={cube_z_distance_m:.4f} < "
                        f"threshold_m={float(threshold_m):.4f} during "
                        f"high-frequency check at waypoint "
                        f"{int(waypoint_index)}/{int(waypoint_count)}"
                    ),
                )
            remaining_sec = deadline - time.monotonic()
            if remaining_sec <= 0.0:
                return None
            time.sleep(min(max(0.001, float(poll_interval_sec)), remaining_sec))

    def set_joint_position_rad(self, joint_index, position_rad, settle_sec=0.2):
        try:
            joint_index = int(joint_index)
            position_rad = float(position_rad)
        except (TypeError, ValueError):
            return ArmGoal.Result(
                success=False,
                message="set_joint_position needs numeric joint_index and position_rad",
            )

        expected_count = int(self.arm_params["global"]["joints_count"])
        if joint_index < 0 or joint_index >= expected_count:
            return ArmGoal.Result(
                success=False,
                message=(
                    f"set_joint_position joint_index {joint_index} is outside "
                    f"[0, {expected_count - 1}]"
                ),
            )
        if not math.isfinite(position_rad):
            return ArmGoal.Result(
                success=False,
                message="set_joint_position position_rad must be finite",
            )

        position_deg = math.degrees(position_rad)
        self.arm_agnle_control.arm_index_change(joint_index, position_deg)
        current_positions_rad = [
            math.radians(float(joint_angle))
            for joint_angle in self.arm_agnle_control.get_arm_angles()
        ]
        self.pybullet_robot_controller.setJointPosition(position=current_positions_rad)
        self.arm_commute_node.publish_arm_angle()

        if settle_sec > 0.0:
            time.sleep(max(0.0, float(settle_sec)))

        actual_position_deg = self.arm_agnle_control.get_arm_angles()[joint_index]
        return ArmGoal.Result(
            success=True,
            message=(
                f"set_joint_position success: joint {joint_index}="
                f"{float(actual_position_deg):.2f}deg"
            ),
        )

    def open_gripper(self):
        return self.set_joint_position_rad(
            self._gripper_joint_index(),
            math.radians(60.0),
            settle_sec=0.2,
        )

    def close_gripper(self):
        return self.set_joint_position_rad(
            self._gripper_joint_index(),
            math.radians(10.0),
            settle_sec=0.2,
        )

    def _move_to_target_position(
        self,
        target,
        *,
        steps,
        waypoint_sleep_sec,
        goal_tolerance_m,
        joint_command_timeout_sec,
        joint_command_tolerance_rad,
        joint_command_republish_interval_sec,
        cube_z_distance_reader=None,
        cube_z_distance_stop_threshold_m=0.0,
    ):
        step_count = max(1, int(steps)) if int(steps) > 0 else 50
        sleep_sec = (
            max(0.0, float(waypoint_sleep_sec))
            if waypoint_sleep_sec > 0
            else 0.1
        )
        tolerance = (
            max(0.0, float(goal_tolerance_m)) if goal_tolerance_m > 0 else 0.03
        )

        self._sync_virtual_robot_to_current_joint_positions()
        trajectory = self.pybullet_robot_controller.generateInterpolatedTrajectory(
            target_position=target,
            steps=step_count,
        )
        if not trajectory:
            return ArmGoal.Result(
                success=False,
                message=f"PB could not generate IK trajectory to {target}",
            )

        expected_count = self._expected_joint_count()
        final_target_positions = None
        move_start_time_sec = time.monotonic()
        stop_threshold = max(0.0, float(cube_z_distance_stop_threshold_m))
        poll_interval_sec = self._cube_z_distance_early_stop_poll_interval_sec()
        early_stop_start_waypoint = max(1, len(trajectory) - 1)
        for waypoint_index, joint_positions in enumerate(trajectory, start=1):
            target_positions = self._with_preserved_grasp_joints(joint_positions)
            if len(target_positions) < expected_count:
                return ArmGoal.Result(
                    success=False,
                    message=(
                        f"waypoint {waypoint_index}/{len(trajectory)} has "
                        f"{len(target_positions)} joints; expected {expected_count}"
                    ),
                )
            final_target_positions = [float(value) for value in target_positions[:expected_count]]
            self._publish_joint_positions_rad(final_target_positions)
            high_frequency_stop_enabled = (
                stop_threshold > 0.0 and waypoint_index >= early_stop_start_waypoint
            )
            if high_frequency_stop_enabled:
                stop_result = self._wait_or_stop_on_cube_z_distance(
                    cube_z_distance_reader,
                    since_time_sec=move_start_time_sec,
                    duration_sec=sleep_sec,
                    threshold_m=stop_threshold,
                    poll_interval_sec=poll_interval_sec,
                    waypoint_index=waypoint_index,
                    waypoint_count=len(trajectory),
                )
                if stop_result is not None:
                    return stop_result
            elif sleep_sec > 0.0:
                time.sleep(sleep_sec)

        if final_target_positions is None:
            return ArmGoal.Result(success=False, message="PB generated an empty trajectory")

        final_joint_result = self._publish_joint_updates_and_wait(
            phase="move_to_target_final",
            joint_updates_rad={
                joint_index: final_target_positions[joint_index]
                for joint_index in range(expected_count)
            },
            min_joint_count=expected_count,
            tolerance_rad=joint_command_tolerance_rad,
            timeout_sec=joint_command_timeout_sec,
            republish_interval_sec=joint_command_republish_interval_sec,
            cube_z_distance_reader=cube_z_distance_reader,
            cube_z_distance_stop_threshold_m=stop_threshold,
            cube_z_distance_since_time_sec=move_start_time_sec,
            cube_z_distance_poll_interval_sec=poll_interval_sec,
        )
        if not final_joint_result["success"]:
            return ArmGoal.Result(
                success=False,
                message=(
                    "final target joints failed: "
                    f"{final_joint_result['message']}; "
                    f"max_error_rad={final_joint_result.get('max_error_rad')}"
                ),
            )
        if bool(final_joint_result.get("early_stopped_by_cube_z_distance", False)):
            return ArmGoal.Result(success=True, message=str(final_joint_result["message"]))

        final_position = self.pybullet_robot_controller.solveForwardPositonKinematics(
            self.pybullet_robot_controller.getJointStates()[0]
        )[0:3]
        distance_to_goal = math.sqrt(
            sum(
                (float(current) - float(goal)) ** 2
                for current, goal in zip(final_position, target)
            )
        )
        if distance_to_goal > tolerance:
            return ArmGoal.Result(
                success=False,
                message=(
                    "PB final EE distance "
                    f"{distance_to_goal:.4f}m exceeds tolerance {tolerance:.4f}m"
                ),
            )

        return ArmGoal.Result(
            success=True,
            message=(
                f"target move success: {len(trajectory)} waypoints, "
                f"final PB EE distance {distance_to_goal:.4f}m"
            ),
        )

    def car_grasp_sequence(
        self,
        target_position,
        *,
        wrist_target_rad,
        wrist_joint_index=3,
        gripper_joint_index=4,
        gripper_open_rad=math.radians(60.0),
        gripper_close_rad=math.radians(10.0),
        steps=5,
        waypoint_sleep_sec=0.1,
        goal_tolerance_m=0.03,
        joint_state_wait_sec=5.0,
        joint_command_timeout_sec=5.0,
        joint_command_tolerance_rad=0.05,
        joint_command_republish_interval_sec=0.1,
        gripper_close_delay_sec=3.0,
        init_pose_delay_sec=1.0,
        cube_z_distance_reader=None,
    ):
        try:
            wrist_index = int(wrist_joint_index)
            gripper_index = int(gripper_joint_index)
            wrist_target = float(wrist_target_rad)
            target = [float(value) for value in target_position]
        except (TypeError, ValueError):
            return ArmGoal.Result(
                success=False,
                message="car_grasp_sequence needs numeric target_position and wrist_target_rad",
            )
        if len(target) != 3 or not all(math.isfinite(value) for value in target):
            return ArmGoal.Result(
                success=False,
                message="car_grasp_sequence target_position must contain exactly three finite values",
            )
        if not math.isfinite(wrist_target):
            return ArmGoal.Result(
                success=False,
                message="car_grasp_sequence wrist_target_rad must be finite",
            )

        expected_count = self._expected_joint_count()
        if not (0 <= wrist_index < expected_count) or not (0 <= gripper_index < expected_count):
            return ArmGoal.Result(
                success=False,
                message=(
                    "car_grasp_sequence joint index out of range: "
                    f"wrist={wrist_index}, gripper={gripper_index}, count={expected_count}"
                ),
            )

        open_rad = float(gripper_open_rad) if gripper_open_rad > 0.0 else math.radians(60.0)
        close_rad = float(gripper_close_rad) if gripper_close_rad > 0.0 else math.radians(10.0)
        joint_wait = max(0.0, float(joint_state_wait_sec)) if joint_state_wait_sec > 0 else 5.0
        joint_timeout = max(0.0, float(joint_command_timeout_sec)) if joint_command_timeout_sec > 0 else 5.0
        joint_tolerance = (
            max(0.0, float(joint_command_tolerance_rad))
            if joint_command_tolerance_rad > 0
            else 0.05
        )
        republish_interval = (
            max(0.0, float(joint_command_republish_interval_sec))
            if joint_command_republish_interval_sec > 0
            else 0.1
        )
        early_stop_threshold_m = (
            self._cube_z_distance_early_stop_threshold_m()
            if self._cube_z_distance_early_stop_enabled()
            else 0.0
        )
        min_joint_count = max(wrist_index, gripper_index) + 1

        phases = []
        warnings = []

        def add_warning(phase, message):
            warning = {
                "success": False,
                "phase": str(phase),
                "message": str(message),
            }
            warnings.append(warning)
            try:
                self.arm_commute_node.get_logger().warn(
                    f"car_grasp_sequence warning: {phase}: {message}"
                )
            except Exception:
                pass

        try:
            initial_positions = self._wait_for_joint_state_positions(
                min_joint_count=min_joint_count,
                timeout_sec=joint_wait,
            )
        except Exception as exc:
            initial_positions = None
            add_warning("joint_state_wait", f"exception: {exc}")

        if initial_positions is None:
            add_warning(
                "joint_state_wait",
                f"no joint_states with {min_joint_count} joints",
            )

        def publish_init_pose_after_delay():
            init_delay = max(0.0, float(init_pose_delay_sec))
            if init_delay > 0.0:
                time.sleep(init_delay)
            reset_positions = self._joint_reset_positions_rad()
            self._publish_joint_positions_rad(reset_positions)
            reset_command_time_sec = time.monotonic()
            init_settle_sec = 3.0
            time.sleep(init_settle_sec)
            return (
                {
                    "success": True,
                    "phase": "init_pose",
                    "published_count": 1,
                    "waited_sec": float(init_settle_sec),
                    "message": "init pose command published; skipped joint state tolerance wait",
                },
                reset_command_time_sec,
            )

        try:
            open_result = self._publish_joint_updates_and_wait(
                phase="open_gripper",
                joint_updates_rad={gripper_index: open_rad},
                min_joint_count=min_joint_count,
                tolerance_rad=joint_tolerance,
                timeout_sec=joint_timeout,
                republish_interval_sec=republish_interval,
            )
        except Exception as exc:
            open_result = {
                "success": False,
                "phase": "open_gripper",
                "published_count": 0,
                "message": f"exception: {exc}",
            }
        phases.append(open_result)
        if not open_result["success"]:
            add_warning("open_gripper", open_result["message"])

        try:
            wrist_result = self._publish_joint_updates_and_wait(
                phase="wrist",
                joint_updates_rad={wrist_index: wrist_target},
                min_joint_count=min_joint_count,
                tolerance_rad=joint_tolerance,
                timeout_sec=joint_timeout,
                republish_interval_sec=republish_interval,
            )
        except Exception as exc:
            wrist_result = {
                "success": False,
                "phase": "wrist",
                "published_count": 0,
                "message": f"exception: {exc}",
            }
        phases.append(wrist_result)
        if not wrist_result["success"]:
            add_warning("wrist", wrist_result["message"])

        try:
            move_result = self._move_to_target_position(
                target,
                steps=steps,
                waypoint_sleep_sec=waypoint_sleep_sec,
                goal_tolerance_m=goal_tolerance_m,
                joint_command_timeout_sec=joint_timeout,
                joint_command_tolerance_rad=joint_tolerance,
                joint_command_republish_interval_sec=republish_interval,
                cube_z_distance_reader=cube_z_distance_reader,
                cube_z_distance_stop_threshold_m=early_stop_threshold_m,
            )
        except Exception as exc:
            move_result = ArmGoal.Result(
                success=False,
                message=f"exception: {exc}",
            )
        move_result_message = str(getattr(move_result, "message", ""))
        if not move_result.success:
            add_warning("move_to_target", move_result_message)

        try:
            close_current_positions = self._wait_for_joint_state_positions(
                min_joint_count=min_joint_count,
                timeout_sec=joint_timeout,
            )
        except Exception as exc:
            add_warning("close_gripper", f"joint state read exception: {exc}")
            close_current_positions = None
        if close_current_positions is None:
            add_warning(
                "close_gripper",
                "joint state was not available before publishing command; using fallback joint positions",
            )
            try:
                close_current_positions = self._current_joint_positions_rad()
            except Exception as exc:
                add_warning("close_gripper", f"fallback joint position exception: {exc}")
                close_current_positions = []

        if len(close_current_positions) > gripper_index:
            try:
                close_target_positions = list(close_current_positions)
                close_target_positions[gripper_index] = close_rad
                self._publish_joint_positions_rad(close_target_positions)
                close_delay = max(0.0, float(gripper_close_delay_sec))
                if close_delay > 0.0:
                    time.sleep(close_delay)
                close_result = {
                    "success": True,
                    "phase": "close_gripper",
                    "published_count": 1,
                    "waited_sec": float(close_delay),
                    "message": "close gripper command published; skipped finger tolerance wait",
                }
            except Exception as exc:
                close_result = {
                    "success": False,
                    "phase": "close_gripper",
                    "published_count": 0,
                    "message": f"exception: {exc}",
                }
                add_warning("close_gripper", close_result["message"])
        else:
            close_result = {
                "success": False,
                "phase": "close_gripper",
                "published_count": 0,
                "message": (
                    "close gripper skipped: fallback joint position count "
                    f"{len(close_current_positions)} <= gripper index {gripper_index}"
                ),
            }
            add_warning("close_gripper", close_result["message"])
        phases.append(close_result)

        try:
            init_result, reset_command_time_sec = publish_init_pose_after_delay()
        except Exception as exc:
            reset_command_time_sec = time.monotonic()
            init_result = {
                "success": False,
                "phase": "init_pose",
                "published_count": 0,
                "message": f"exception: {exc}",
            }
            add_warning("init_pose", init_result["message"])
        phases.append(init_result)

        grasp_verify_result = self._verify_car_grasp_cube_z_distance(
            cube_z_distance_reader,
            since_time_sec=reset_command_time_sec,
            threshold_m=0.08,
        )
        phases.append(grasp_verify_result)

        phase_summary = ", ".join(
            f"{phase['phase']}:{phase.get('published_count', 0)}pub"
            for phase in phases
        )
        warning_summary = " | ".join(
            f"{warning['phase']}: {warning['message']}" for warning in warnings
        )
        message = (
            f"{grasp_verify_result['message']}; "
            f"continued_to_init_pose={init_result['success']}; "
            f"move_to_target={move_result_message}; "
            f"{phase_summary}"
        )
        if warning_summary:
            message += f"; arm_warnings={warning_summary}"
        trajectory_debug = getattr(
            self.pybullet_robot_controller,
            "last_interpolated_trajectory_debug",
            {},
        )
        if isinstance(trajectory_debug, dict) and trajectory_debug:
            try:
                message += "; car_grasp_sequence_ik_debug_json=" + json.dumps(
                    trajectory_debug,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            except Exception:
                pass
        return ArmGoal.Result(
            success=bool(grasp_verify_result["success"]),
            message=message,
        )

    def grap(self):
        self.arm_agnle_control.arm_index_change(4, 10)
        self.arm_commute_node.publish_arm_angle()

    def init_pose(self, grap=False):
        angle = self.arm_agnle_control.arm_default_change()
        if grap:
            self.arm_agnle_control.arm_index_change(4, 30)
            self.arm_commute_node.publish_arm_angle()
            time.sleep(1.0)
        self.arm_commute_node.publish_arm_angle()
        joints_reset_radians = [math.radians(joint_angle) for joint_angle in angle]
        self.pybullet_robot_controller.setJointPosition(position=joints_reset_radians)
        return ArmGoal.Result(success=True, message="success")

    def test(self):
        return ArmGoal.Result(success=True, message="success")

    def look_up(self):
        self.arm_agnle_control.arm_index_change(2, 140)
        self.arm_commute_node.publish_arm_angle()
        return ArmGoal.Result(success=True, message="success")

    def _is_at_target(
        self,
        depth: float,
        y: float,
        z: float,
        target_depth: float,
        depth_thresh: float,
        lateral_thresh: float,
    ) -> bool:
        return abs(y) <= 0.02 and abs(z) <= 0.07

    def follow_obj(self, label="ball", target_depth=0.3):
        depth_threshold = 0.05
        lateral_threshold = 0.05
        x_adjust_factor = 0.3
        y_adjust_factor = 0.3
        z_adjust_factor = 0.3

        data = self.arm_commute_node.get_latest_object_coordinates(label=label)
        if not data or len(data) < 3:
            return ArmGoal.Result(success=False, message="No object detected")
        current_depth, obj_y, obj_z = data

        if self._is_at_target(
            current_depth,
            obj_y,
            obj_z,
            target_depth,
            depth_threshold,
            lateral_threshold,
        ):
            return True

        depth_diff = current_depth - target_depth
        x_offset = depth_diff * x_adjust_factor
        y_offset = obj_y * y_adjust_factor
        z_offset = obj_z * z_adjust_factor

        target_pos = self.pybullet_robot_controller.offset_from_end_effector(
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset,
            visualize=True,
            mark_color=[0, 1, 0],
        )
        if target_pos is None:
            return ArmGoal.Result(success=False, message="Failed to compute target")
        target_pos[0] = 0.2

        traj = self.pybullet_robot_controller.generateInterpolatedTrajectory(
            target_position=target_pos, steps=10
        )
        if not traj:
            return True

        for angle in traj:
            self.move_real_and_virtual(radian=angle)
            time.sleep(0.05)

            new_data = self.arm_commute_node.get_latest_object_coordinates(label=label)
            if new_data and len(new_data) >= 3:
                nd, ny, nz = new_data
                if self._is_at_target(
                    nd, ny, nz, target_depth, depth_threshold, lateral_threshold
                ):
                    return True

    def ik_move_func(self):
        imu_data = self.arm_commute_node.get_latest_imu_data()
        obj_position_data = self.arm_commute_node.get_latest_object_coordinates(
            label="fire"
        )
        if imu_data is None or obj_position_data is None:
            return

        extrinsics = self.pybullet_robot_controller.calculate_imu_extrinsics(
            imu_world_quaternion=imu_data, link_name="camera_1", visualize=False
        )
        obj_pos_in_pybullet = self.pybullet_robot_controller.transform_object_to_world(
            T_world_to_imu=extrinsics,
            object_coords_imu=obj_position_data,
            visualize=True,
        )
        is_close_pos = self.pybullet_robot_controller.is_link_close_to_position(
            link_name="base_link", target_position=obj_pos_in_pybullet, threshold=0.8
        )
        if is_close_pos:
            robot_angle = self.pybullet_robot_controller.generateInterpolatedTrajectory(
                target_position=obj_pos_in_pybullet, steps=10
            )
            for angle in robot_angle:
                self.move_real_and_virtual(radian=angle)
                time.sleep(0.2)

    def move_real_and_virtual(self, radian, *, preserve_grasp_joints=False):
        joint_positions = self._with_preserved_grasp_joints(radian) if preserve_grasp_joints else list(radian)
        self.pybullet_robot_controller.setJointPosition(position=joint_positions)
        degree = self.radians_to_degrees(joint_positions)
        self.arm_agnle_control.arm_all_change(degree)
        self.arm_commute_node.publish_arm_angle()

    def move_forward_backward(self, direction="forward", distance=0.1):
        actual_distance = distance if direction == "forward" else -abs(distance)
        z_offset = 0.05 if direction == "forward" else -0.05

        obj_pos = self.pybullet_robot_controller.markPointInFrontOfEndEffector(
            distance=actual_distance, z_offset=z_offset
        )
        robot_angle = self.pybullet_robot_controller.generateInterpolatedTrajectory(
            target_position=obj_pos, steps=5
        )
        for angle in robot_angle:
            self.move_real_and_virtual(radian=angle)
            time.sleep(0.1)
        return ArmGoal.Result(success=True, message=f"Successfully moved {direction}")

    def move_end_effector_direction(self, direction="up"):
        pos = self.pybullet_robot_controller.move_ee_relative_example(
            direction=direction,
            distance=0.05,
        )
        if pos is None:
            return ArmGoal.Result(success=False, message=f"Invalid direction: {direction}")
        robot_angle = self.pybullet_robot_controller.generateInterpolatedTrajectory(
            target_position=pos, steps=5
        )
        for angle in robot_angle:
            self.move_real_and_virtual(radian=angle)
            time.sleep(0.1)
        return ArmGoal.Result(success=True, message="success")

    def move_elbow_direction(self, direction="elbow_forward"):
        self.pybullet_robot_controller.set_end_effector("elbow")
        target_direction = direction.replace("elbow_", "")

        if target_direction == "left":
            target_direction = "right"
        elif target_direction == "right":
            target_direction = "left"
        elif target_direction == "forward":
            target_direction = "backward"
        elif target_direction == "backward":
            target_direction = "forward"

        pos = self.pybullet_robot_controller.move_ee_relative_example(
            direction=target_direction,
            distance=0.05,
            execute_move=False,
        )

        if pos is not None:
            robot_angle_traj = (
                self.pybullet_robot_controller.generateInterpolatedTrajectory(
                    target_position=pos, steps=5
                )
            )
            for angle in robot_angle_traj:
                self.move_real_and_virtual(radian=angle)
                time.sleep(0.1)

        self.pybullet_robot_controller.set_end_effector("gripper")
        return ArmGoal.Result(success=True, message="success")
