import logging
import math
import time
from pathlib import Path

import numpy as np

from agents.car_approach import sample_logic
from agents.car_approach.base_sampler import (
    _load_python_dependencies,
    _load_arm_config,
    _get_reset_camera_transform_in_base_link_frame,
    _amcl_pose_to_pb_world_pose,
    _amcl_snapshot_to_ros_map_pose,
    _capture_live_scene_voxels,
    _attempt_ik_at_base_pose,
    _load_target_object_pointcloud_camera,
)
from agents.car_approach.scripts.run_base_pose_sampling import load_config
from agents.car_approach.src.pybullet_ompl import _find_controllable_joints, load_planning_config
from agents.arm_approach.move_arm import move_arm_for_solution

logger = logging.getLogger(__name__)
CAR_APPROACH_DIR = Path(__file__).resolve().parents[1] / "car_approach"


def _yaw_quat_xyzw(yaw_rad: float) -> np.ndarray:
    half_yaw = 0.5 * float(yaw_rad)
    return np.asarray([0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)], dtype=np.float64)


def run_arm_approach_sync(payload: dict[str, object], context_id: str = "") -> dict[str, object]:
    """
    Evaluates the IK from the *current* amcl pose and moves the arm.
    """
    started_at = time.time()
    
    # Load configs
    base_config_path = CAR_APPROACH_DIR / "configs" / "base_pose_sampling.yaml"
    camera_config_path = CAR_APPROACH_DIR / "configs" / "camera_car_voxel_ompl.yaml"
    cfg = load_config(base_config_path)
    planning_config = load_planning_config(Path(cfg["planner_config_path"]))
    arm_config = _load_arm_config()
    _, p_mod, pybullet_data = _load_python_dependencies()
    
    (
        camera_in_base_link_rotation,
        camera_in_base_link_position,
        base_link_z_pb,
    ) = _get_reset_camera_transform_in_base_link_frame(
        Path(cfg["planner_config_path"]),
        planning_config,
    )
    
    target_object_points_camera = _load_target_object_pointcloud_camera(
        cfg.get("grasp_debug_npz_path")
    )

    # Re-capture the live scene
    live_scene = _capture_live_scene_voxels(
        camera_config_path.resolve(),
        camera_in_base_link_rotation,
        camera_in_base_link_position,
        base_link_z_pb,
        target_object_points_camera=target_object_points_camera,
    )
    
    camera_to_pb_rotation = live_scene.camera_to_pb_rotation
    camera_position_pb = live_scene.camera_position_pb

    # Load grasp records
    visualization_records, grasp_candidates = sample_logic.load_grasp_visualization_records_from_payload(
        payload.get("grasp_result", {}),
        camera_to_pb_rotation=camera_to_pb_rotation,
        camera_position_pb=camera_position_pb,
        source_label="payload.grasp_result",
    )
    
    # Get current AMCL pose
    current_amcl_pose = _amcl_snapshot_to_ros_map_pose(live_scene.amcl_pose)
    if current_amcl_pose is None:
        raise RuntimeError("No /amcl_pose received. Cannot determine arm approach pose.")
    
    current_amcl_pb_pose = _amcl_pose_to_pb_world_pose(current_amcl_pose, z_pb=base_link_z_pb)
    if current_amcl_pb_pose is None:
        raise RuntimeError("Could not transform /amcl_pose to PyBullet world pose.")
    current_amcl_pb_xyz, current_pb_yaw = current_amcl_pb_pose

    # Setup PyBullet for IK evaluation
    client_id = p_mod.connect(p_mod.DIRECT)
    if client_id < 0:
        raise RuntimeError("PyBullet DIRECT unavailable.")

    try:
        p_mod.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_mod.resetSimulation()
        p_mod.setGravity(0.0, 0.0, -9.8)
        p_mod.loadURDF("plane.urdf")

        # Load Obstacles
        voxel_size = float(live_scene.voxel_size_m)
        half_extents = [voxel_size / 2.0] * 3
        col_shape = p_mod.createCollisionShape(p_mod.GEOM_BOX, halfExtents=half_extents)
        obstacle_body_ids = []
        for voxel_center in np.asarray(live_scene.voxel_centers_pb, dtype=np.float64).reshape(-1, 3):
            obstacle_body_ids.append(
                p_mod.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=col_shape,
                    basePosition=voxel_center.astype(float).tolist(),
                )
            )

        # Load Robot
        base_orientation_rad = [math.radians(v) for v in planning_config.base_orientation_euler_deg]
        robot_id = p_mod.loadURDF(
            planning_config.urdf_path,
            useFixedBase=True,
            basePosition=[0.0, 0.0, float(planning_config.initial_height)],
            baseOrientation=p_mod.getQuaternionFromEuler(base_orientation_rad),
        )
        expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])
        controllable_joint_ids, controllable_joint_names = _find_controllable_joints(
            p_mod,
            robot_id,
            expected_joint_count,
        )
        if len(controllable_joint_ids) != expected_joint_count:
            raise RuntimeError(
                "Controllable joint count mismatch: "
                f"expected={expected_joint_count} actual={len(controllable_joint_ids)} "
                f"names={controllable_joint_names}"
            )

        # Rank records
        ranked_records = sample_logic.rank_grasp_records_by_reset_ee_pose(
            visualization_records,
            reset_ee_position_xyz=current_amcl_pb_xyz,
            reset_ee_orientation_xyzw=_yaw_quat_xyzw(current_pb_yaw),
        )

        selected_solution = None
        for record in ranked_records:
            ik_attempt = _attempt_ik_at_base_pose(
                p_mod=p_mod,
                robot_id=robot_id,
                controllable_joint_ids=controllable_joint_ids,
                planning_config=planning_config,
                target_pb=np.asarray(record["target_pb"], dtype=np.float64),
                target_quat_pb=np.asarray(record["target_quat_pb"], dtype=np.float64),
                base_link_xy=(float(current_amcl_pb_xyz[0]), float(current_amcl_pb_xyz[1])),
                base_link_yaw_rad=float(current_pb_yaw),
                obstacle_body_ids=obstacle_body_ids,
                enable_ompl_path_check=False,
            )
            
            feasible = sample_logic._ik_attempt_is_feasible(
                ik_attempt,
                position_tolerance_m=float(cfg.get("position_tolerance_m", planning_config.position_tolerance_m)),
                orientation_tolerance_deg=float(cfg.get("orientation_tolerance_deg", 12.0)),
            )
            
            if feasible:
                selected_solution = ik_attempt
                record["selected_as_best"] = True
                break

    finally:
        try:
            p_mod.disconnect(client_id)
        except Exception:
            pass

    if selected_solution is None:
        logger.error("No feasible IK solution found for arm approach from current pose.")
        return {
            "success": False,
            "status_code": "ARM_APPROACH_NO_IK",
            "phase": "ik_evaluation",
            "message": "No feasible IK solution for the arm at the current base pose.",
            "next_agent": None,
        }

    # Extract joint rads and trigger move_arm
    joint_rads = selected_solution["ik_joint_solution_rad"]
    logger.info(f"Found feasible IK. Moving arm to joints: {joint_rads}")
    
    # move_arm_for_solution handles calling publisher and returns publish metadata.
    arm_result = move_arm_for_solution(
        selected_solution,
        planning_config=planning_config,
        planner_config_path=Path(cfg["planner_config_path"]),
    )
    arm_success = bool(arm_result.get("success", False))
    
    return {
        "success": arm_success,
        "status_code": "ARM_APPROACH_SUCCESS" if arm_success else "ARM_APPROACH_EXEC_FAILED",
        "phase": "arm_motion",
        "ik_solution_rad": joint_rads,
        "arm_result": arm_result,
        "message": "Arm approach finished.",
        "next_agent": None,
        "exec_latency": time.time() - started_at,
    }
