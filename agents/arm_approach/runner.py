import logging
import math
import os
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
from agents.car_approach.src.pybullet_ompl import (
    _find_controllable_joints,
    _set_joint_positions_direct,
    load_planning_config,
)
from agents.arm_approach.move_arm import move_arm_for_solution

logger = logging.getLogger(__name__)
CAR_APPROACH_DIR = Path(__file__).resolve().parents[1] / "car_approach"
ARM_APPROACH_POSITION_TOLERANCE_M = 0.05
ARM_APPROACH_RPY_TOLERANCE_DEG = 30.0
ARM_APPROACH_GRIPPER_OPEN_JOINT_INDEX = 4
ARM_APPROACH_GRIPPER_OPEN_DEG = 80.0
CURRENT_BASE_LINK_LOCAL_XY = (0.0, 0.0)
CURRENT_BASE_LINK_LOCAL_YAW_RAD = 0.0


def _wrap_angle_rad(angle_rad: float) -> float:
    return float((float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi)


def _axis_yaw_xy(axis_xyz: np.ndarray, fallback_yaw_rad: float = 0.0) -> float:
    axis = np.asarray(axis_xyz, dtype=np.float64).reshape(3)
    axis_xy = axis[:2]
    axis_norm = float(np.linalg.norm(axis_xy))
    if axis_norm <= 1e-6:
        return float(fallback_yaw_rad)
    return float(math.atan2(float(axis_xy[1]), float(axis_xy[0])))


def _quat_xyzw_yaw(p_mod, quat_xyzw: np.ndarray, fallback_yaw_rad: float = 0.0) -> float:
    try:
        euler_xyz = p_mod.getEulerFromQuaternion(np.asarray(quat_xyzw, dtype=np.float64).reshape(4).tolist())
    except Exception:
        return float(fallback_yaw_rad)
    return float(euler_xyz[2])


def _quat_xyzw_rpy_rad(p_mod, quat_xyzw: np.ndarray) -> np.ndarray:
    return np.asarray(
        p_mod.getEulerFromQuaternion(np.asarray(quat_xyzw, dtype=np.float64).reshape(4).tolist()),
        dtype=np.float64,
    )


def _annotate_rpy_error(
    ik_attempt: dict[str, object],
    *,
    p_mod,
    target_quat_pb: np.ndarray,
) -> dict[str, object]:
    final_quat = np.asarray(ik_attempt["final_ee_orientation_xyzw"], dtype=np.float64).reshape(4)
    target_quat = np.asarray(target_quat_pb, dtype=np.float64).reshape(4)
    try:
        final_rpy = _quat_xyzw_rpy_rad(p_mod, final_quat)
        target_rpy = _quat_xyzw_rpy_rad(p_mod, target_quat)
        error_rad = np.asarray(
            [_wrap_angle_rad(final_value - target_value) for final_value, target_value in zip(final_rpy, target_rpy)],
            dtype=np.float64,
        )
        error_deg = np.degrees(error_rad)
        ik_attempt["ee_rpy_error_deg"] = {
            "roll": float(error_deg[0]),
            "pitch": float(error_deg[1]),
            "yaw": float(error_deg[2]),
        }
        ik_attempt["ee_rpy_error_abs_deg"] = {
            "roll": float(abs(error_deg[0])),
            "pitch": float(abs(error_deg[1])),
            "yaw": float(abs(error_deg[2])),
        }
        ik_attempt["ee_rpy_error_abs_max_deg"] = float(np.max(np.abs(error_deg)))
    except Exception as exc:
        ik_attempt["ee_rpy_error_deg"] = None
        ik_attempt["ee_rpy_error_abs_deg"] = None
        ik_attempt["ee_rpy_error_abs_max_deg"] = None
        ik_attempt["ee_rpy_error_error"] = str(exc)
    return ik_attempt


def _arm_approach_ik_attempt_is_feasible(
    ik_attempt: dict[str, object],
    *,
    position_tolerance_m: float,
    rpy_tolerance_deg: float,
) -> bool:
    rpy_error_abs_max = ik_attempt.get("ee_rpy_error_abs_max_deg")
    return (
        ik_attempt.get("ik_joint_solution_rad") is not None
        and bool(ik_attempt.get("collision_free", False))
        and float(ik_attempt.get("ee_position_error_m", float("inf"))) <= float(position_tolerance_m)
        and rpy_error_abs_max is not None
        and float(rpy_error_abs_max) <= float(rpy_tolerance_deg)
    )


def _joint_reset_rad_from_planning_config(planning_config) -> list[float]:
    return [math.radians(float(value)) for value in planning_config.joint_reset_deg]


def _with_open_gripper_before_motion(
    solution: dict[str, object],
    planning_config,
) -> tuple[dict[str, object], list[float]]:
    gripper_index = int(ARM_APPROACH_GRIPPER_OPEN_JOINT_INDEX)
    gripper_open_rad = math.radians(float(ARM_APPROACH_GRIPPER_OPEN_DEG))
    goal_joint_positions = [float(value) for value in solution["ik_joint_solution_rad"]]
    start_joint_positions = _joint_reset_rad_from_planning_config(planning_config)

    if 0 <= gripper_index < len(goal_joint_positions):
        goal_joint_positions[gripper_index] = gripper_open_rad
    if 0 <= gripper_index < len(start_joint_positions):
        start_joint_positions[gripper_index] = gripper_open_rad

    adjusted_solution = dict(solution)
    adjusted_solution["ik_joint_solution_rad"] = goal_joint_positions
    adjusted_solution["ik_joint_solution_deg"] = [math.degrees(value) for value in goal_joint_positions]
    adjusted_solution["preopened_gripper_joint_index"] = gripper_index
    adjusted_solution["preopened_gripper_target_deg"] = float(ARM_APPROACH_GRIPPER_OPEN_DEG)
    adjusted_solution["preopened_gripper_target_rad"] = gripper_open_rad
    return adjusted_solution, start_joint_positions


def _current_reset_ee_pose(
    *,
    p_mod,
    robot_id: int,
    controllable_joint_ids: list[int],
    planning_config,
    base_link_xy: tuple[float, float],
    base_link_yaw_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    base_xyz = [
        float(base_link_xy[0]),
        float(base_link_xy[1]),
        float(planning_config.initial_height),
    ]
    base_quat = p_mod.getQuaternionFromEuler([0.0, 0.0, float(base_link_yaw_rad)])
    joint_reset_rad = [math.radians(float(value)) for value in planning_config.joint_reset_deg]

    p_mod.resetBasePositionAndOrientation(robot_id, base_xyz, base_quat)
    _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, joint_reset_rad)
    p_mod.performCollisionDetection()
    ee_state = p_mod.getLinkState(
        robot_id,
        planning_config.ee_link_index,
        computeForwardKinematics=True,
    )
    return (
        np.asarray(ee_state[4], dtype=np.float64),
        np.asarray(ee_state[5], dtype=np.float64),
    )


def _rank_grasp_records_by_current_ee_distance(
    visualization_records: list[dict[str, object]],
    *,
    current_ee_position_xyz: np.ndarray,
    current_ee_orientation_xyzw: np.ndarray,
    p_mod,
) -> list[dict[str, object]]:
    current_ee_position = np.asarray(current_ee_position_xyz, dtype=np.float64).reshape(3)
    current_ee_yaw = _quat_xyzw_yaw(p_mod, current_ee_orientation_xyzw)
    ranked_records = [dict(record) for record in visualization_records]
    for record in ranked_records:
        target_pb = np.asarray(record["target_pb"], dtype=np.float64).reshape(3)
        target_rot_pb = np.asarray(record["target_rot_pb"], dtype=np.float64).reshape(3, 3)
        target_yaw = _axis_yaw_xy(target_rot_pb[:, 0], fallback_yaw_rad=current_ee_yaw)
        yaw_error_rad = abs(_wrap_angle_rad(target_yaw - current_ee_yaw))
        distance_m = float(np.linalg.norm(target_pb - current_ee_position))
        record["current_ee_distance_to_target_grasp_m"] = distance_m
        record["current_ee_yaw_error_to_target_grasp_rad"] = float(yaw_error_rad)
        record["current_ee_yaw_error_to_target_grasp_deg"] = float(math.degrees(yaw_error_rad))

    ranked_records.sort(
        key=lambda record: (
            float(record["current_ee_distance_to_target_grasp_m"]),
            float(record["current_ee_yaw_error_to_target_grasp_rad"]),
            int(record.get("rank", 0)),
        )
    )
    for order_index, record in enumerate(ranked_records, start=1):
        record["target_sample_order"] = int(order_index)
    return ranked_records


def _annotate_ik_attempt(
    ik_attempt: dict[str, object],
    record: dict[str, object],
    *,
    target_pb: np.ndarray,
    target_quat_pb: np.ndarray,
) -> dict[str, object]:
    ik_attempt["target_pb"] = np.asarray(target_pb, dtype=np.float64).astype(float).tolist()
    ik_attempt["target_quat_pb"] = np.asarray(target_quat_pb, dtype=np.float64).astype(float).tolist()
    ik_attempt["grasp_rank"] = int(record.get("rank", 0))
    ik_attempt["target_sample_order"] = int(record.get("target_sample_order", 0))
    ik_attempt["current_ee_distance_to_target_grasp_m"] = float(
        record.get("current_ee_distance_to_target_grasp_m", float("inf"))
    )
    ik_attempt["current_ee_yaw_error_to_target_grasp_deg"] = float(
        record.get("current_ee_yaw_error_to_target_grasp_deg", float("inf"))
    )
    return ik_attempt


def _best_effort_ik_sort_key(ik_attempt: dict[str, object]) -> tuple[float, float, float, int]:
    orientation_error = ik_attempt.get("ee_rpy_error_abs_max_deg")
    if orientation_error is None:
        orientation_error = ik_attempt.get("ee_orientation_error_deg")
    return (
        float(ik_attempt.get("ee_position_error_m", float("inf"))),
        float("inf") if orientation_error is None else float(orientation_error),
        float(ik_attempt.get("joint_reset_delta_norm_l2", float("inf"))),
        int(ik_attempt.get("target_sample_order", 0)),
    )


def _ik_attempt_can_be_best_effort(ik_attempt: dict[str, object]) -> bool:
    return (
        ik_attempt.get("ik_joint_solution_rad") is not None
        and bool(ik_attempt.get("collision_free", False))
    )


def _normalized_grasp_payload(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    nested_result = payload.get("result")
    if isinstance(nested_result, dict):
        return nested_result
    raw_result = payload.get("raw_result")
    if isinstance(raw_result, dict):
        return raw_result
    return payload


def _grasp_payload_has_pose(payload: object) -> bool:
    raw_payload = _normalized_grasp_payload(payload)
    valid_grasps = raw_payload.get("valid_grasp_poses_camera")
    if isinstance(valid_grasps, list) and any(isinstance(item, dict) for item in valid_grasps):
        return True
    best_grasp = raw_payload.get("best_grasp_pose_camera")
    return isinstance(best_grasp, dict) and bool(best_grasp)


def _select_grasp_payload(payload: dict[str, object]) -> dict[str, object] | None:
    for key in ("grasp_result_payload", "grasp_result"):
        if key in payload:
            candidate = payload.get(key)
            if _grasp_payload_has_pose(candidate):
                return _normalized_grasp_payload(candidate)
            return None
    latest_grasp = payload.get("latest_grasp_result")
    if _grasp_payload_has_pose(latest_grasp):
        return _normalized_grasp_payload(latest_grasp)
    if _grasp_payload_has_pose(payload):
        return _normalized_grasp_payload(payload)
    return None


def run_arm_approach_sync(payload: dict[str, object], context_id: str = "") -> dict[str, object]:
    """
    Evaluates the IK from the *current* amcl pose and moves the arm.
    """
    started_at = time.time()
    grasp_payload = _select_grasp_payload(payload)
    if grasp_payload is None:
        return {
            "success": False,
            "status_code": "ARM_APPROACH_NO_GRASP",
            "phase": "grasp_payload",
            "message": "No GraspGen grasp pose data was provided to arm_approach.",
            "graspgen_result_available": False,
            "initialpose_published": False,
            "next_agent": None,
            "exec_latency": time.time() - started_at,
        }
    
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
    try:
        visualization_records, _grasp_candidates = sample_logic.load_grasp_visualization_records_from_payload(
            grasp_payload,
            camera_to_pb_rotation=camera_to_pb_rotation,
            camera_position_pb=camera_position_pb,
            source_label="arm_approach.grasp_payload",
        )
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "success": False,
            "status_code": "ARM_APPROACH_NO_GRASP",
            "phase": "grasp_payload",
            "message": f"Invalid GraspGen grasp pose data for arm_approach: {exc}",
            "graspgen_result_available": False,
            "initialpose_published": False,
            "next_agent": None,
            "exec_latency": time.time() - started_at,
        }
    
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

        current_ee_position_xyz, current_ee_orientation_xyzw = _current_reset_ee_pose(
            p_mod=p_mod,
            robot_id=robot_id,
            controllable_joint_ids=controllable_joint_ids,
            planning_config=planning_config,
            base_link_xy=CURRENT_BASE_LINK_LOCAL_XY,
            base_link_yaw_rad=CURRENT_BASE_LINK_LOCAL_YAW_RAD,
        )

        # Rank target grasp poses by the current gripper EE position.
        ranked_records = _rank_grasp_records_by_current_ee_distance(
            visualization_records,
            current_ee_position_xyz=current_ee_position_xyz,
            current_ee_orientation_xyzw=current_ee_orientation_xyzw,
            p_mod=p_mod,
        )

        selected_solution = None
        selected_record = None
        best_effort_solution = None
        best_effort_record = None
        attempted_count = 0
        position_tolerance_m = float(
            os.getenv(
                "APPROACH_AGENT_ARM_POSITION_TOLERANCE_M",
                str(max(float(cfg.get("position_tolerance_m", planning_config.position_tolerance_m)), ARM_APPROACH_POSITION_TOLERANCE_M)),
            )
        )
        rpy_tolerance_deg = float(
            os.getenv("APPROACH_AGENT_ARM_RPY_TOLERANCE_DEG", str(ARM_APPROACH_RPY_TOLERANCE_DEG))
        )
        for record in ranked_records:
            target_pb = np.asarray(record["target_pb"], dtype=np.float64)
            target_quat_pb = np.asarray(record["target_quat_pb"], dtype=np.float64)
            ik_attempt = _attempt_ik_at_base_pose(
                p_mod=p_mod,
                robot_id=robot_id,
                controllable_joint_ids=controllable_joint_ids,
                planning_config=planning_config,
                target_pb=target_pb,
                target_quat_pb=target_quat_pb,
                base_link_xy=CURRENT_BASE_LINK_LOCAL_XY,
                base_link_yaw_rad=CURRENT_BASE_LINK_LOCAL_YAW_RAD,
                obstacle_body_ids=obstacle_body_ids,
                enable_ompl_path_check=False,
            )
            attempted_count += 1
            _annotate_ik_attempt(
                ik_attempt,
                record,
                target_pb=target_pb,
                target_quat_pb=target_quat_pb,
            )
            _annotate_rpy_error(
                ik_attempt,
                p_mod=p_mod,
                target_quat_pb=target_quat_pb,
            )
            ik_attempt["current_amcl_pb_xyz"] = current_amcl_pb_xyz.astype(float).tolist()
            ik_attempt["current_amcl_pb_yaw_rad"] = float(current_pb_yaw)
            
            feasible = _arm_approach_ik_attempt_is_feasible(
                ik_attempt,
                position_tolerance_m=position_tolerance_m,
                rpy_tolerance_deg=rpy_tolerance_deg,
            )
            ik_attempt["ik_reachable"] = bool(feasible)
            ik_attempt["arm_approach_position_tolerance_m"] = float(position_tolerance_m)
            ik_attempt["arm_approach_rpy_tolerance_deg"] = float(rpy_tolerance_deg)
            if _ik_attempt_can_be_best_effort(ik_attempt) and (
                best_effort_solution is None
                or _best_effort_ik_sort_key(ik_attempt) < _best_effort_ik_sort_key(best_effort_solution)
            ):
                best_effort_solution = ik_attempt
                best_effort_record = record
            
            if feasible:
                selected_solution = ik_attempt
                selected_record = record
                record["selected_as_best"] = True
                ik_attempt["selected_as_best"] = True
                break

    finally:
        try:
            p_mod.disconnect(client_id)
        except Exception:
            pass

    if selected_solution is None:
        if best_effort_solution is not None:
            best_effort_solution["selected_as_best_effort"] = True
            if best_effort_record is not None:
                best_effort_record["selected_as_best_effort"] = True
            logger.warning(
                "No feasible IK solution found; returning best collision-free IK solution "
                "with ee_position_error_m=%s.",
                best_effort_solution.get("ee_position_error_m"),
            )
            return {
                "success": False,
                "status_code": "ARM_APPROACH_BEST_EFFORT_IK",
                "phase": "ik_evaluation",
                "message": (
                    "No IK solution satisfied the pose tolerances. Returning the closest "
                    "collision-free IK result without moving the arm."
                ),
                "selected_solution": best_effort_solution,
                "best_effort_solution": best_effort_solution,
                "ik_solution_rad": best_effort_solution["ik_joint_solution_rad"],
                "attempted_grasp_count": attempted_count,
                "ranked_grasp_count": len(ranked_records),
                "initialpose_published": False,
                "next_agent": None,
                "exec_latency": time.time() - started_at,
            }
        logger.error("No feasible IK solution found for arm approach from current pose.")
        return {
            "success": False,
            "status_code": "ARM_APPROACH_NO_IK",
            "phase": "ik_evaluation",
            "message": "No feasible or collision-free best-effort IK solution for the arm at the current base pose.",
            "attempted_grasp_count": attempted_count,
            "ranked_grasp_count": len(ranked_records),
            "initialpose_published": False,
            "next_agent": None,
            "exec_latency": time.time() - started_at,
        }

    selected_solution, start_joint_positions_rad = _with_open_gripper_before_motion(
        selected_solution,
        planning_config,
    )
    # Extract joint rads and trigger move_arm
    joint_rads = selected_solution["ik_joint_solution_rad"]
    logger.info(f"Found feasible IK. Moving arm to joints: {joint_rads}")
    
    # move_arm_for_solution handles calling publisher and returns publish metadata.
    arm_result = move_arm_for_solution(
        selected_solution,
        planning_config=planning_config,
        planner_config_path=Path(cfg["planner_config_path"]),
        start_joint_positions_rad=start_joint_positions_rad,
    )
    arm_success = bool(arm_result.get("success", False))
    
    return {
        "success": arm_success,
        "status_code": "ARM_APPROACH_SUCCESS" if arm_success else "ARM_APPROACH_EXEC_FAILED",
        "phase": "arm_motion",
        "ik_solution_rad": joint_rads,
        "selected_solution": selected_solution,
        "selected_grasp_rank": None if selected_record is None else int(selected_record.get("rank", 0)),
        "selected_target_sample_order": int(selected_solution.get("target_sample_order", 0)),
        "attempted_grasp_count": attempted_count,
        "ranked_grasp_count": len(ranked_records),
        "arm_result": arm_result,
        "message": "Arm approach finished.",
        "initialpose_published": False,
        "next_agent": None,
        "exec_latency": time.time() - started_at,
    }
