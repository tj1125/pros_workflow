from pathlib import Path
import math
import numpy as np
import pybullet as p
from src.pybullet_ompl import load_planning_config, _find_controllable_joints, _set_joint_positions_direct
from src.pybullet_smoke import _load_python_dependencies, _degrees_to_radians, _load_arm_config
from scripts.run_base_pose_sampling import _load_grasp_debug_data

def main():
    config_path = Path("configs/base_pose_sampling.yaml")
    planner_config_path = Path("configs/pybullet_ompl.yaml")
    npz_path = Path("outputs/pybullet_ompl_snapshot.npz") # Let's assume this or prompt user
    
    # Actually, we can load from base_pose_sampling parsing
    from scripts.run_base_pose_sampling import load_config
    cfg = load_config(config_path)
    
    target_pb, target_rot_pb, voxels_pb = _load_grasp_debug_data(
        Path(cfg["grasp_debug_npz_path"]),
        Path(cfg["planner_config_path"])
    )
    
    planning_config = load_planning_config(Path(cfg["planner_config_path"]))
    arm_config = _load_arm_config()
    _, p, pybullet_data = _load_python_dependencies()
    
    client_id = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.loadURDF("plane.urdf")
    
    base_orientation_rad = [math.radians(v) for v in planning_config.base_orientation_euler_deg]
    base_orientation_xyzw = p.getQuaternionFromEuler(base_orientation_rad)
    robot_id = p.loadURDF(
        planning_config.urdf_path,
        useFixedBase=True,
        basePosition=[0.0, 0.0, planning_config.initial_height],
        baseOrientation=base_orientation_xyzw,
    )
    
    expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])
    controllable_joint_ids, _ = _find_controllable_joints(p, robot_id, expected_joint_count)
    joint_reset_rad = _degrees_to_radians(planning_config.joint_reset_deg)
    _set_joint_positions_direct(p, robot_id, controllable_joint_ids, joint_reset_rad)
    
    from src.pybullet_ompl import _add_debug_axes
    import scipy.spatial.transform as st
    
    # Draw Voxels
    print(f"Drawing {len(voxels_pb)} voxels...")
    voxel_size = 0.05
    half_extents = [voxel_size / 2.0] * 3
    col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.8, 0.2, 0.2, 0.8])
    for v in voxels_pb:
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=v.tolist()
        )
        
    # Draw Target Pose
    target_pos = target_pb.tolist()
    print(f"Drawing Target Pose at: {target_pos}")
    # Draw a little golden ball at the target
    targ_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1.0, 0.8, 0.0, 1.0])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=targ_vis, basePosition=target_pos)
    
    # Draw Target orientation axes (Target XYZ axes)
    target_quat_pb = st.Rotation.from_matrix(target_rot_pb).as_quat()
    _add_debug_axes(p, target_pos, orientation_xyzw=target_quat_pb.tolist(), label="TARGET")
    
    # Calculate Inverse Kinematics (IK)
    print("--- Calculating IK ---")
    ee_link_idx = planning_config.ee_link_index
    # target_quat_pb is already calculated above
    
    # We do a mini-loop over IK to allow PyBullet to settle into a deep minimum
    for _ in range(5):
        ik_joint_poses = p.calculateInverseKinematics(
            robot_id,
            ee_link_idx,
            target_pos,
            target_quat_pb.tolist(),
            maxNumIterations=10000,
            residualThreshold=1e-5
        )
        _set_joint_positions_direct(p, robot_id, controllable_joint_ids, ik_joint_poses[:len(controllable_joint_ids)])
    p.performCollisionDetection()
    
    # Measure error
    final_state = p.getLinkState(robot_id, ee_link_idx, computeForwardKinematics=True)
    final_pos = np.array(final_state[4])
    dist_error = np.linalg.norm(np.array(target_pos) - final_pos)
    print(f"[IK Result] Reached Pos: {final_pos.tolist()}")
    print(f"[IK Result] Position Error: {dist_error*1000:.2f} mm")
    if dist_error < 0.01:
        print("[IK Result] SUCCESS! The robot reached the target position!")
    else:
        print("[IK Result] FAILED. The position is unreachable or orientation conflict prevented it.")
    
    import time
    try:
        import rospy
        from geometry_msgs.msg import PoseWithCovarianceStamped
        has_rospy = True
        rospy.init_node('pybullet_amcl_viewer', anonymous=True, disable_signals=True)
    except ImportError:
        has_rospy = False
        print("[Warning] 'rospy' not found. Cannot subscribe to /amcl_pose.")

    # ros map -> pybullet: x -> -y, y -> x, z -> z
    R_ros2pb = np.array([
        [ 0.0, -1.0, 0.0],
        [ 1.0,  0.0, 0.0],
        [ 0.0,  0.0, 1.0]
    ], dtype=np.float64)

    def amcl_callback(msg):
        # Extract ROS Position
        pos_ros = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        
        # Apply Base Offset: amcl_pose (ros map轉pybullet) + (+0, +0.1288, +0.071)
        # Note: the user formula specified offset is added AFTER ros2pb rotation
        pos_pb = R_ros2pb @ pos_ros
        base_pos_pb = pos_pb + np.array([0.0, 0.1288, 0.071])

        # Extract ROS Orientation
        q = msg.pose.pose.orientation
        rot_ros = st.Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        
        # Convert to PyBullet Orientation
        rot_pb = R_ros2pb @ rot_ros
        quat_pb = st.Rotation.from_matrix(rot_pb).as_quat()

        # Update PyBullet Simulation Base
        p.resetBasePositionAndOrientation(robot_id, base_pos_pb.tolist(), quat_pb.tolist())

    if has_rospy:
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, amcl_callback)
        print("Successfully subscribed to /amcl_pose. Waiting for messages to move base_link!")

    print("Test ready. Close GUI to exit.")
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1. / 240.)

if __name__ == "__main__":
    main()
