import math
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

    def move_real_and_virtual(self, radian):
        self.pybullet_robot_controller.setJointPosition(position=radian)
        degree = self.radians_to_degrees(radian)
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
