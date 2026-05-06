import math
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R

try:
    from ament_index_python.packages import (
        PackageNotFoundError,
        get_package_share_directory,
    )
except ImportError:  # pragma: no cover - available in ROS runtime
    PackageNotFoundError = Exception
    get_package_share_directory = None


class PybulletRobotController:
    def __init__(self, arm_params, arm_angle_control_node):
        self.arm_params = arm_params.get_arm_params()
        self.arm_angle_control_node = arm_angle_control_node
        self.urdf_path = self._resolve_urdf_path()
        self.robot_id = None
        self.num_joints = None
        self.expected_controllable_joint_count = int(
            self.arm_params["pybullet"]["controllable_joints"]
        )
        self.controllable_joints = []
        self.end_eff_indices = self.arm_params["pybullet"]["end_eff_index"]
        self.end_eff_index = self.end_eff_indices[0]
        self.ik_max_num_iterations = int(
            self.arm_params["pybullet"].get("ik_max_num_iterations", 200)
        )
        self.ik_residual_threshold = float(
            self.arm_params["pybullet"].get("ik_residual_threshold", 1e-5)
        )
        self.time_step = float(self.arm_params["pybullet"]["time_step"])
        self.previous_ee_position = None
        self.initial_height = float(self.arm_params["pybullet"]["initial_height"])
        self.base_orientation_euler_deg = self._base_orientation_euler_deg_from_config()
        self.mimic_pairs = {}
        self.marker_ids = []
        self.transformed_object_marker_ids = []
        self.target_marker_ids = []
        self.front_marker_ids = []
        self.link_axes_lines = []

        self.createWorld(
            GUI=self.arm_params["pybullet"]["gui"],
            view_world=self.arm_params["pybullet"]["view_world"],
        )
        self.set_initial_joint_positions()

    def _resolve_urdf_path(self) -> str:
        urdf_name = self.arm_params["pybullet"]["urdf_name"]
        candidates = []

        if get_package_share_directory is not None:
            try:
                share_dir = Path(get_package_share_directory("robot_description"))
                candidates.append(share_dir / "urdf" / urdf_name)
            except PackageNotFoundError:
                pass

        candidates.append(
            Path(__file__).resolve().parents[2] / "robot_description" / "urdf" / urdf_name
        )

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        raise FileNotFoundError(f"Unable to locate robot description URDF: {urdf_name}")

    def _base_orientation_euler_deg_from_config(self) -> list[float]:
        raw_orientation = self.arm_params["pybullet"].get(
            "base_orientation_euler_deg",
            [0.0, 0.0, 0.0],
        )
        try:
            orientation = [float(value) for value in raw_orientation]
        except (TypeError, ValueError) as exc:
            raise ValueError("pybullet.base_orientation_euler_deg must be numeric [roll, pitch, yaw].") from exc
        if len(orientation) != 3:
            raise ValueError("pybullet.base_orientation_euler_deg must contain exactly three values.")
        return orientation

    def _get_urdf_search_paths(self) -> list[str]:
        urdf_dir = Path(self.urdf_path).resolve().parent
        package_dir = urdf_dir.parent
        candidates = [package_dir, package_dir.parent]
        search_paths = []

        for candidate in candidates:
            if candidate.exists():
                candidate_path = str(candidate)
                if candidate_path not in search_paths:
                    search_paths.append(candidate_path)

        return search_paths

    def set_end_effector(self, ee_type: str):
        if ee_type == "gripper":
            self.end_eff_index = self.end_eff_indices[0]
        elif ee_type == "elbow":
            self.end_eff_index = self.end_eff_indices[1]

    def markPointInFrontOfEndEffector(
        self, distance=0.3, z_offset=0.1, color=[0, 1, 1], visualize=True
    ):
        if visualize:
            for mid in self.front_marker_ids:
                try:
                    p.removeUserDebugItem(mid)
                except Exception:
                    pass
            self.front_marker_ids.clear()

        try:
            ee_state = p.getLinkState(
                self.robot_id, self.end_eff_index, computeForwardKinematics=True
            )
            position = np.array(ee_state[0])
            orientation = ee_state[1]
            rot_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
            forward_direction = rot_matrix[:, 0]
            target_point = position + forward_direction * distance
            target_point[2] += z_offset
        except Exception:
            return None

        if visualize:
            line_length = 0.05
            self.front_marker_ids.append(
                p.addUserDebugLine(
                    (target_point - np.array([line_length, 0, 0])).tolist(),
                    (target_point + np.array([line_length, 0, 0])).tolist(),
                    color,
                    lineWidth=2,
                )
            )
            self.front_marker_ids.append(
                p.addUserDebugLine(
                    (target_point - np.array([0, line_length, 0])).tolist(),
                    (target_point + np.array([0, line_length, 0])).tolist(),
                    color,
                    lineWidth=2,
                )
            )
            self.front_marker_ids.append(
                p.addUserDebugLine(
                    (target_point - np.array([0, 0, line_length])).tolist(),
                    (target_point + np.array([0, 0, line_length])).tolist(),
                    color,
                    lineWidth=2,
                )
            )

        return list(target_point)

    def draw_link_axes(self, link_name=None, axis_length=0.2):
        for lid in self.link_axes_lines:
            try:
                p.removeUserDebugItem(lid)
            except Exception:
                pass
        self.link_axes_lines.clear()

        if link_name is None:
            link_idx = self.end_eff_index
        else:
            link_idx = self._find_link_index(link_name)
            if link_idx is None:
                link_idx = self.end_eff_index

        ls = p.getLinkState(self.robot_id, link_idx)
        pos, orn = np.array(ls[0]), ls[1]
        r_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        x_axis, y_axis, z_axis = r_mat[:, 0], r_mat[:, 1], r_mat[:, 2]

        self.link_axes_lines.append(
            p.addUserDebugLine(pos.tolist(), (pos + x_axis * axis_length).tolist(), [1, 0, 0], lineWidth=3)
        )
        self.link_axes_lines.append(
            p.addUserDebugLine(pos.tolist(), (pos + y_axis * axis_length).tolist(), [0, 1, 0], lineWidth=3)
        )
        self.link_axes_lines.append(
            p.addUserDebugLine(pos.tolist(), (pos + z_axis * axis_length).tolist(), [0, 0, 1], lineWidth=3)
        )
        self.link_axes_lines.append(
            p.addUserDebugPoints([pos.tolist()], [[1, 1, 0]], pointSize=30)
        )

    def is_link_close_to_position(self, link_name, target_position, threshold):
        if not isinstance(target_position, (list, tuple)) or len(target_position) < 3:
            return False
        if not isinstance(threshold, (int, float)) or threshold < 0:
            return False

        target_pos_np = np.array(target_position[0:3])
        current_link_pos_np = None

        if link_name == "base_link":
            try:
                base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                current_link_pos_np = np.array(base_pos)
            except Exception:
                return False
        else:
            joint_idx = self._find_link_index(link_name)
            if joint_idx is None:
                return False
            try:
                link_state = p.getLinkState(
                    self.robot_id, joint_idx, computeForwardKinematics=True
                )
                current_link_pos_np = np.array(link_state[0])
            except Exception:
                return False

        distance = np.linalg.norm(current_link_pos_np - target_pos_np)
        return distance < threshold

    def _find_link_index(self, link_name):
        for jid in range(self.num_joints):
            try:
                info = p.getJointInfo(self.robot_id, jid)
                if info[12].decode("utf-8") == link_name:
                    return jid
            except Exception:
                continue
        return None

    def calculate_ee_relative_target_positions(self, distance):
        try:
            ee_state = p.getLinkState(
                self.robot_id, self.end_eff_index, computeForwardKinematics=True
            )
            current_position = np.array(ee_state[0])
            current_orientation = ee_state[1]
            rotation_matrix = np.array(
                p.getMatrixFromQuaternion(current_orientation)
            ).reshape(3, 3)
        except Exception:
            return None

        local_x_axis = rotation_matrix[:, 0]
        local_y_axis = rotation_matrix[:, 1]
        local_z_axis = rotation_matrix[:, 2]

        return {
            "up": (current_position + local_z_axis * distance).tolist(),
            "down": (current_position - local_z_axis * distance).tolist(),
            "left": (current_position + local_y_axis * distance).tolist(),
            "right": (current_position - local_y_axis * distance).tolist(),
            "forward": (current_position + local_x_axis * distance).tolist(),
            "backward": (current_position - local_x_axis * distance).tolist(),
        }

    def move_ee_relative_example(
        self, direction, distance, visualize=True, execute_move=False
    ):
        target_positions = self.calculate_ee_relative_target_positions(distance)
        if not target_positions or direction not in target_positions:
            return None

        target_pos_world = target_positions[direction]
        if visualize:
            self.markTarget(target_pos_world, color=[0, 1, 1])

        if execute_move:
            joint_angles = self.solveInversePositionKinematics(target_pos_world)
            if joint_angles:
                self.setJointPosition(joint_angles[: len(self.controllable_joints)])

        return target_pos_world

    def offset_from_end_effector(
        self, x_offset, y_offset, z_offset, visualize=False, mark_color=[1, 0, 1]
    ):
        ee_state = p.getLinkState(self.robot_id, self.end_eff_index)
        position = np.array(ee_state[0])
        orientation = ee_state[1]
        rot_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

        local_x_axis = rot_matrix[:, 0]
        local_y_axis = rot_matrix[:, 1]
        local_z_axis = rot_matrix[:, 2]
        offset_vector = (
            x_offset * local_x_axis + y_offset * local_y_axis + z_offset * local_z_axis
        )
        new_position = position + offset_vector

        if visualize:
            self.markTarget(new_position, color=mark_color)
        return list(new_position)

    def getJointStates(self):
        joint_states = p.getJointStates(self.robot_id, self.controllable_joints)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    def move_end_effector_laterally(self, distance=0.3):
        ee_state = p.getLinkState(self.robot_id, self.end_eff_index)
        position = np.array(ee_state[0])
        orientation = ee_state[1]
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        right_direction = rotation_matrix[:, 1]
        target_position = position + right_direction * distance
        target_pose = list(target_position) + list(p.getEulerFromQuaternion(orientation))
        joint_angles = self.solveInversePositionKinematics(target_pose)
        if joint_angles:
            return joint_angles
        return None

    def generateInterpolatedTrajectory(self, target_position, steps=50):
        current_position = self.solveForwardPositonKinematics(self.getJointStates()[0])[0:3]
        step_vector = (np.array(target_position) - np.array(current_position)) / steps
        self.markTarget(target_position)

        joint_angles_in_radians = []
        for i in range(steps):
            intermediate_position = np.array(current_position) + (i + 1) * step_vector
            joint_angles = self.solveInversePositionKinematics(intermediate_position)
            if joint_angles and len(joint_angles) >= len(self.controllable_joints):
                joint_angles_in_radians.append(
                    joint_angles[: len(self.controllable_joints)]
                )
            else:
                break

        return joint_angles_in_radians

    def calculate_imu_extrinsics(
        self, imu_world_quaternion, link_name, visualize=False, axis_length=0.1
    ):
        for mid in self.marker_ids:
            try:
                p.removeUserDebugItem(mid)
            except Exception:
                pass
        self.marker_ids.clear()

        link_idx = self._find_link_index(link_name)
        if link_idx is None:
            return None

        ls = p.getLinkState(self.robot_id, link_idx, computeForwardKinematics=True)
        link_origin_world = np.array(ls[4])
        link_orn = ls[5]

        rot_link = R.from_quat(link_orn)
        _, _, yaw_l = rot_link.as_euler("xyz", False)
        rot_imu = R.from_quat(imu_world_quaternion)
        roll_i, pitch_i, _ = rot_imu.as_euler("xyz", False)

        fused = R.from_euler("xyz", [roll_i, pitch_i, yaw_l], False)
        r_world_imu_fused = fused.as_matrix()
        r_imu_world = r_world_imu_fused.T
        t_imu_world = -r_imu_world @ link_origin_world

        t_world_to_imu = np.eye(4)
        t_world_to_imu[:3, :3] = r_imu_world
        t_world_to_imu[:3, 3] = t_imu_world

        if visualize:
            x_ax = r_world_imu_fused[:, 0] * axis_length
            y_ax = r_world_imu_fused[:, 1] * axis_length
            z_ax = r_world_imu_fused[:, 2] * axis_length
            origin = link_origin_world.tolist()
            self.marker_ids.append(
                p.addUserDebugLine(origin, (link_origin_world + x_ax).tolist(), [1, 0, 0], 4)
            )
            self.marker_ids.append(
                p.addUserDebugLine(origin, (link_origin_world + y_ax).tolist(), [0, 1, 0], 4)
            )
            self.marker_ids.append(
                p.addUserDebugLine(origin, (link_origin_world + z_ax).tolist(), [0, 0, 1], 4)
            )

        return t_world_to_imu

    def solveInversePositionKinematics(self, end_eff_pose):
        if len(end_eff_pose) == 6:
            return p.calculateInverseKinematics(
                self.robot_id,
                self.end_eff_index,
                targetPosition=end_eff_pose[0:3],
                targetOrientation=p.getQuaternionFromEuler(end_eff_pose[3:6]),
                maxNumIterations=self.ik_max_num_iterations,
                residualThreshold=self.ik_residual_threshold,
            )
        return p.calculateInverseKinematics(
            self.robot_id,
            self.end_eff_index,
            targetPosition=end_eff_pose[0:3],
            maxNumIterations=self.ik_max_num_iterations,
            residualThreshold=self.ik_residual_threshold,
        )

    def set_initial_joint_positions(self):
        initial_positions_deg = self.arm_angle_control_node.get_arm_angles()
        initial_positions_rad = [math.radians(angle) for angle in initial_positions_deg]
        self.setJointPosition(initial_positions_rad)

    def setJointPosition(self, position, kp=1.0, kv=1.0):
        zero_vec = [0.0] * len(self.controllable_joints)
        p.setJointMotorControlArray(
            self.robot_id,
            self.controllable_joints,
            p.POSITION_CONTROL,
            targetPositions=position,
            targetVelocities=zero_vec,
            positionGains=[kp] * len(self.controllable_joints),
            velocityGains=[kv] * len(self.controllable_joints),
        )
        for _ in range(100):
            p.stepSimulation()

    def transform_object_to_world(
        self,
        T_world_to_imu,
        object_coords_imu,
        visualize=False,
        marker_color=[0, 1, 1],
        text_color=[1, 1, 1],
    ):
        if not isinstance(T_world_to_imu, np.ndarray) or T_world_to_imu.shape != (4, 4):
            return None

        try:
            object_coords_imu = np.array(object_coords_imu, dtype=float)
            if object_coords_imu.shape != (3,):
                raise ValueError
        except Exception:
            return None

        try:
            t_imu_to_world = np.linalg.inv(T_world_to_imu)
            p_imu_homogeneous = np.append(object_coords_imu, 1.0)
            p_world_homogeneous = t_imu_to_world @ p_imu_homogeneous
            object_coords_world = p_world_homogeneous[:3]
        except Exception:
            return None

        for mid in self.transformed_object_marker_ids:
            try:
                p.removeUserDebugItem(mid)
            except Exception:
                pass
        self.transformed_object_marker_ids.clear()

        if visualize:
            self.markTarget(object_coords_world, color=marker_color)
            self.transformed_object_marker_ids.extend(self.target_marker_ids)
            text_position = object_coords_world + np.array([0, 0, 0.05])
            text_id = p.addUserDebugText(
                text=f"Obj: ({object_coords_world[0]:.2f}, {object_coords_world[1]:.2f}, {object_coords_world[2]:.2f})",
                textPosition=text_position.tolist(),
                textColorRGB=text_color,
                textSize=1.0,
            )
            self.transformed_object_marker_ids.append(text_id)

        return list(object_coords_world)

    def markTarget(self, target_position, color=[1, 0, 0]):
        for line_id in self.target_marker_ids:
            try:
                p.removeUserDebugItem(line_id)
            except Exception:
                pass
        self.target_marker_ids.clear()

        line_length = 0.1
        self.target_marker_ids.append(
            p.addUserDebugLine(
                [target_position[0] - line_length, target_position[1], target_position[2]],
                [target_position[0] + line_length, target_position[1], target_position[2]],
                color,
                lineWidth=3,
            )
        )
        self.target_marker_ids.append(
            p.addUserDebugLine(
                [target_position[0], target_position[1] - line_length, target_position[2]],
                [target_position[0], target_position[1] + line_length, target_position[2]],
                color,
                lineWidth=3,
            )
        )
        self.target_marker_ids.append(
            p.addUserDebugLine(
                [target_position[0], target_position[1], target_position[2] - line_length],
                [target_position[0], target_position[1], target_position[2] + line_length],
                color,
                lineWidth=3,
            )
        )

    def solveForwardPositonKinematics(self, joint_pos):
        del joint_pos
        ee_state = p.getLinkState(self.robot_id, self.end_eff_index)
        link_trn, link_rot = ee_state[0], ee_state[1]
        return list(link_trn) + list(p.getEulerFromQuaternion(link_rot))

    def createWorld(self, GUI=True, view_world=False):
        if GUI:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.time_step, numSolverIterations=100, numSubSteps=10
        )
        p.setRealTimeSimulation(True)
        p.loadURDF("plane.urdf")

        for search_path in self._get_urdf_search_paths():
            p.setAdditionalSearchPath(search_path)

        rotation = p.getQuaternionFromEuler(
            [math.radians(value) for value in self.base_orientation_euler_deg]
        )
        self.robot_id = p.loadURDF(
            self.urdf_path,
            useFixedBase=True,
            basePosition=[0, 0, self.initial_height],
            baseOrientation=rotation,
        )

        self.num_joints = p.getNumJoints(self.robot_id)
        mimic_joint_names = ["Revolute 6"]

        for jid in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, jid)
            joint_type = info[2]
            joint_name = info[1].decode("utf-8")
            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                if joint_name not in mimic_joint_names:
                    self.controllable_joints.append(jid)

        if len(self.controllable_joints) != self.expected_controllable_joint_count:
            self.controllable_joints = self.controllable_joints[
                : self.expected_controllable_joint_count
            ]

        if view_world:
            while True:  # pragma: no cover - debug visualization loop
                p.stepSimulation()
                time.sleep(self.time_step)
