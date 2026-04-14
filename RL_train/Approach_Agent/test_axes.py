import math
from pathlib import Path
from src.pybullet_ompl import load_planning_config, _find_controllable_joints, _degrees_to_radians, _set_joint_positions_direct
from src.pybullet_ompl import _load_python_dependencies

np_mod, p, pybullet_data = _load_python_dependencies()
config_path = Path("configs/pybullet_ompl.yaml")
planning_config = load_planning_config(config_path, require_obstacles=False)

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.loadURDF("plane.urdf")
base_orientation_rad = [math.radians(v) for v in planning_config.base_orientation_euler_deg]
base_orientation_xyzw = p.getQuaternionFromEuler(base_orientation_rad)
robot_id = p.loadURDF(
    planning_config.urdf_path,
    useFixedBase=True,
    basePosition=[0.0, 0.0, planning_config.initial_height],
    baseOrientation=base_orientation_xyzw,
)

# Move to reset pose
controllable_joint_ids, _ = _find_controllable_joints(p, robot_id, 5)
joint_reset_rad = _degrees_to_radians(planning_config.joint_reset_deg)
_set_joint_positions_direct(p, robot_id, controllable_joint_ids, joint_reset_rad)
p.performCollisionDetection()

# EE state
ee_state = p.getLinkState(robot_id, planning_config.ee_link_index, computeForwardKinematics=True)
ee_pos, ee_quat = ee_state[0], ee_state[1]
import numpy as np
R_ee = np.array(p.getMatrixFromQuaternion(ee_quat)).reshape(3,3)
print("--- PyBullet EE at reset pose ---")
print("Position:", ee_pos)
print("Rotation Matrix (World):")
print(R_ee)

# Camera state
def find_link(name):
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        if info[12].decode('utf-8') == name:
            return i
    return -1

cam_idx = find_link("camera_1")
if cam_idx >= 0:
    cam_state = p.getLinkState(robot_id, cam_idx, computeForwardKinematics=True)
    cam_pos, cam_quat = cam_state[0], cam_state[1]
    R_cam = np.array(p.getMatrixFromQuaternion(cam_quat)).reshape(3,3)
    print("\n--- Camera at reset pose ---")
    print("Position:", cam_pos)
    print("Rotation Matrix (World):")
    print(R_cam)
else:
    print("camera_1 link not found!")
