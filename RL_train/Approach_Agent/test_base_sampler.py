import time
import math
import numpy as np
import pybullet as p
import scipy.spatial.transform as st
from pathlib import Path
from src.pybullet_ompl import load_planning_config, _find_controllable_joints, _set_joint_positions_direct, _add_debug_axes
from src.pybullet_smoke import _load_python_dependencies, _degrees_to_radians, _load_arm_config
from scripts.run_base_pose_sampling import _load_grasp_debug_data, load_config

def main():
    config_path = Path("configs/base_pose_sampling.yaml")
    cfg = load_config(config_path)
    planning_config = load_planning_config(Path(cfg["planner_config_path"]))
    arm_config = _load_arm_config()
    _, p, pybullet_data = _load_python_dependencies()

    target_pb, target_rot_pb, voxels_pb = _load_grasp_debug_data(
        Path(cfg["grasp_debug_npz_path"]),
        Path(cfg["planner_config_path"])
    )

    client_id = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.loadURDF("plane.urdf")

    # Draw initial scene
    voxel_size = 0.05
    half_extents = [voxel_size / 2.0] * 3
    col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.8, 0.2, 0.2, 0.8])
    for v in voxels_pb:
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_shape, basePosition=v.tolist())

    target_pos = target_pb.tolist()
    targ_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1.0, 0.8, 0.0, 1.0])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=targ_vis, basePosition=target_pos)
    target_quat_pb = st.Rotation.from_matrix(target_rot_pb).as_quat()
    _add_debug_axes(p, target_pos, orientation_xyzw=target_quat_pb.tolist(), label="TARGET")

    # Load robot
    robot_id = p.loadURDF(
        planning_config.urdf_path,
        useFixedBase=True,
        basePosition=[0.0, 0.0, planning_config.initial_height],
        baseOrientation=p.getQuaternionFromEuler([0,0,0]),
    )
    expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])
    controllable_joint_ids, _ = _find_controllable_joints(p, robot_id, expected_joint_count)

    # SAMPLING LOGIC
    print("--- Starting Base Pose Sampling ---")
        
    valid_candidates = []
    joint_reset_rad = _degrees_to_radians(planning_config.joint_reset_deg)

    for i in range(500):
        # 1. 看目標夾取pose座標系的座標，沿著-x方向找 pb 座標系的 xy base_link 點
        distance = np.random.uniform(0.20, 0.48)
        # 為了模擬採樣，我們在 -X 的射線上加入微小偏移
        angle_offset = np.random.uniform(math.radians(-10), math.radians(10))
        
        # Target local X axis (approach axis)
        v_x = target_rot_pb[:, 0]
        v_x_xy = np.array([v_x[0], v_x[1]])
        if np.linalg.norm(v_x_xy) > 1e-4:
            v_x_xy = v_x_xy / np.linalg.norm(v_x_xy)
        else:
            v_x_xy = np.array([1.0, 0.0])
            
        ray_angle = np.arctan2(v_x_xy[1], v_x_xy[0])
        theta = ray_angle + angle_offset
        
        # Base link ideal positions in PB
        pb_bx = target_pos[0] - distance * np.cos(theta)
        pb_by = target_pos[1] - distance * np.sin(theta)
        
        # Heading faces the target
        pb_yaw = np.arctan2(target_pos[1] - pb_by, target_pos[0] - pb_bx)
        
        # 2. 換算到 ros map 的 xy 點及 rotation
        # PB Base -> PB AMCL -> ROS MAP
        # rigorous 4x4 matrix transformation
        T_base_pb = np.eye(4)
        T_base_pb[:3, :3] = st.Rotation.from_euler('z', pb_yaw).as_matrix()
        T_base_pb[0, 3] = pb_bx
        T_base_pb[1, 3] = pb_by
        
        # base_link = amcl_pose + (0, 0.1288, 0.071) in car local frame
        T_offset = np.eye(4)
        T_offset[0, 3] = 0.0
        T_offset[1, 3] = 0.1288
        T_offset[2, 3] = 0.071
        
        # P_amcl = P_base * inv(T_offset)
        T_amcl_pb = T_base_pb @ np.linalg.inv(T_offset)
        amcl_pb_x = T_amcl_pb[0, 3]
        amcl_pb_y = T_amcl_pb[1, 3]
        amcl_pb_yaw = st.Rotation.from_matrix(T_amcl_pb[:3, :3]).as_euler('zyx')[0]
        
        # PB AMCL -> ROS MAP AMCL
        # Keeps original Yaw as requested
        ros_amcl_x = amcl_pb_y
        ros_amcl_y = -amcl_pb_x
        ros_amcl_yaw = amcl_pb_yaw 
        
        # [此處您的車子會判斷是否可達]
        # 若可達...
        
        # 3. 把可達的 ros map 點轉換成 pb 座標系的 base_link 點及 rotation
        # ROS MAP AMCL -> PB AMCL
        # Keeping Yaw consistent
        new_amcl_pb_x = -ros_amcl_y
        new_amcl_pb_y = ros_amcl_x
        new_amcl_pb_yaw = ros_amcl_yaw
        
        T_amcl_pb_new = np.eye(4)
        T_amcl_pb_new[:3, :3] = st.Rotation.from_euler('z', new_amcl_pb_yaw).as_matrix()
        T_amcl_pb_new[0, 3] = new_amcl_pb_x
        T_amcl_pb_new[1, 3] = new_amcl_pb_y
        
        # PB AMCL -> PB BASE
        T_base_pb_new = T_amcl_pb_new @ T_offset
        final_pb_bx = T_base_pb_new[0, 3]
        final_pb_by = T_base_pb_new[1, 3]
        final_pb_yaw = st.Rotation.from_matrix(T_base_pb_new[:3, :3]).as_euler('zyx')[0]
        
        # 4. 最後在 pb 座標系判斷 ik, ompl 可行與否
        p.resetBasePositionAndOrientation(robot_id, [final_pb_bx, final_pb_by, planning_config.initial_height], p.getQuaternionFromEuler([0, 0, final_pb_yaw]))
        
        _set_joint_positions_direct(p, robot_id, controllable_joint_ids, joint_reset_rad)
        ee_link_idx = planning_config.ee_link_index
        for _ in range(3):
            ik_joint_poses = p.calculateInverseKinematics(
                robot_id, ee_link_idx, target_pos, target_quat_pb.tolist(),
                maxNumIterations=5000, residualThreshold=1e-4
            )
            _set_joint_positions_direct(p, robot_id, controllable_joint_ids, ik_joint_poses[:len(controllable_joint_ids)])
            
        p.performCollisionDetection()
        final_state = p.getLinkState(robot_id, ee_link_idx, computeForwardKinematics=True)
        final_pos = np.array(final_state[4])
        dist_error = np.linalg.norm(np.array(target_pos) - final_pos)
        
        if dist_error < 0.015:
            pb_yaw_deg = math.degrees(final_pb_yaw)
            ros_yaw_deg = math.degrees(ros_amcl_yaw)
            
            print(f"[Sample {i}] REACHABLE!", flush=True)
            print(f"  ├─ [PB Base Link] x={final_pb_bx:.3f}, y={final_pb_by:.3f}, PB Yaw={pb_yaw_deg:.1f}°", flush=True)
            print(f"  └─ [ROS MAP AMCL] x={ros_amcl_x:.3f}, y={ros_amcl_y:.3f}, ROS Map Yaw={ros_yaw_deg:.1f}°\n", flush=True)
            
            valid_candidates.append((final_pb_bx, final_pb_by, final_pb_yaw))
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.01], rgbaColor=[0, 1, 0, 0.5])
            p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[final_pb_bx, final_pb_by, 0.01])
        else:
            if i % 10 == 0:
                print(f"[Sample {i}] IK Failed. Dist Error: {dist_error:.4f}m", flush=True)
            time.sleep(0.5)

    print(f"Sampling Finished. Found {len(valid_candidates)} reachable base configurations!")
    print("Test ready.")
    p.disconnect()

if __name__ == "__main__":
    main()
