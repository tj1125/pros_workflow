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
    _mirror_camera_points_x,
    _robot_collides_with_obstacles,
    _transform_camera_points_to_local_pb,
)
from agents.car_approach.scripts.run_base_pose_sampling import load_config
from agents.car_approach.src.pybullet_ompl import (
    _find_controllable_joints,
    _set_joint_positions_direct,
    load_planning_config,
)
from agents.car_approach.src.pybullet_smoke import _render_debug_ppm
from agents.arm_approach.move_arm import (
    build_linear_joint_trajectory,
    config_from_environment,
    move_arm_for_solution,
)

logger = logging.getLogger(__name__)
CAR_APPROACH_DIR = Path(__file__).resolve().parents[1] / "car_approach"
ARM_APPROACH_POSITION_TOLERANCE_M = 0.05
ARM_APPROACH_RPY_TOLERANCE_DEG = 30.0
ARM_APPROACH_ROLL_TOLERANCE_DEG = 10.0
ARM_APPROACH_MAX_TARGET_GRASP_POSES = 10
ARM_APPROACH_WRIST_JOINT_INDEX = 3
ARM_APPROACH_TARGET_GRASP_YAW_REFERENCE_PERIOD_DEG = 180.0
ARM_APPROACH_TARGET_GRASP_YAW_TO_WRIST_SIGN = 1.0
ARM_APPROACH_WRIST_YAW_MIN_DEG = 60.0
ARM_APPROACH_WRIST_YAW_MAX_DEG = 120.0
ARM_APPROACH_GRIPPER_OPEN_JOINT_INDEX = 4
ARM_APPROACH_GRIPPER_OPEN_DEG = 80.0
ARM_APPROACH_START_BASE_JOINT_INDEX = 0
CURRENT_BASE_LINK_LOCAL_XY = (0.0, 0.0)
CURRENT_BASE_LINK_LOCAL_YAW_RAD = 0.0
ARM_APPROACH_DEBUG_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
ARM_APPROACH_SIDE_VIEW_YAW_DEG = 90.0
ARM_APPROACH_SIDE_VIEW_PITCH_DEG = 0.0
ARM_APPROACH_SIDE_VIEW_DISTANCE_M = 0.55
ARM_APPROACH_TOPDOWN_VIEW_YAW_DEG = 90.0
ARM_APPROACH_TOPDOWN_VIEW_PITCH_DEG = -89.0
ARM_APPROACH_TOPDOWN_VIEW_DISTANCE_M = 0.70
ARM_APPROACH_PRE_GRIPPER_FORWARD_ENABLED = True
ARM_APPROACH_PRE_GRIPPER_UP_DISTANCE_M = 0.10
ARM_APPROACH_PRE_GRIPPER_FORWARD_DISTANCE_M = 0.10
ARM_APPROACH_PRE_GRIPPER_FORWARD_Z_OFFSET_M = 0.0


def _env_flag(name: str, default: bool) -> bool:
    default_value = "1" if default else "0"
    return os.getenv(name, default_value).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }


def _wrap_angle_rad(angle_rad: float) -> float:
    return float((float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi)


def _nearest_half_turn_reference_and_delta_deg(angle_rad: float) -> tuple[float, float]:
    angle_deg = float(math.degrees(float(angle_rad)) % 360.0)
    reference_candidates_deg = (0.0, 180.0, 360.0)

    def _sort_key(reference_deg: float) -> tuple[float, int, float]:
        delta_deg = angle_deg - float(reference_deg)
        # For exact half-way cases, prefer the positive delta because wrist 90 + delta
        # stays inside the configured wrist range more often than 90 - 90.
        return (abs(delta_deg), 1 if delta_deg < 0.0 else 0, float(reference_deg))

    reference_deg = min(reference_candidates_deg, key=_sort_key)
    return float(reference_deg), float(angle_deg - reference_deg)


def _nearest_half_turn_reference_and_delta_rad(angle_rad: float) -> tuple[float, float]:
    reference_deg, delta_deg = _nearest_half_turn_reference_and_delta_deg(angle_rad)
    return math.radians(reference_deg), math.radians(delta_deg)


def _axis_yaw_xy(axis_xyz: np.ndarray, fallback_yaw_rad: float = 0.0) -> float:
    axis = np.asarray(axis_xyz, dtype=np.float64).reshape(3)
    axis_xy = axis[:2]
    axis_norm = float(np.linalg.norm(axis_xy))
    if axis_norm <= 1e-6:
        return float(fallback_yaw_rad)
    return float(math.atan2(float(axis_xy[1]), float(axis_xy[0])))


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
        ik_attempt["ee_roll_pitch_error_abs_max_deg"] = float(np.max(np.abs(error_deg[:2])))
    except Exception as exc:
        ik_attempt["ee_rpy_error_deg"] = None
        ik_attempt["ee_rpy_error_abs_deg"] = None
        ik_attempt["ee_rpy_error_abs_max_deg"] = None
        ik_attempt["ee_roll_pitch_error_abs_max_deg"] = None
        ik_attempt["ee_rpy_error_error"] = str(exc)
    return ik_attempt


def _rpy_axis_error_abs_deg(ik_attempt: dict[str, object], axis: str) -> float | None:
    rpy_error_abs = ik_attempt.get("ee_rpy_error_abs_deg")
    if not isinstance(rpy_error_abs, dict):
        return None

    axis_error = rpy_error_abs.get(axis)
    if axis_error is None:
        return None
    return float(axis_error)


def _roll_pitch_error_abs_max_deg(ik_attempt: dict[str, object]) -> float | None:
    roll_pitch_error = ik_attempt.get("ee_roll_pitch_error_abs_max_deg")
    if roll_pitch_error is not None:
        return float(roll_pitch_error)

    roll_error = _rpy_axis_error_abs_deg(ik_attempt, "roll")
    pitch_error = _rpy_axis_error_abs_deg(ik_attempt, "pitch")
    if roll_error is None or pitch_error is None:
        return None
    return float(max(float(roll_error), float(pitch_error)))


def _arm_approach_ik_attempt_is_feasible(
    ik_attempt: dict[str, object],
    *,
    position_tolerance_m: float,
    roll_tolerance_deg: float,
    pitch_tolerance_deg: float,
) -> bool:
    roll_error_abs = _rpy_axis_error_abs_deg(ik_attempt, "roll")
    pitch_error_abs = _rpy_axis_error_abs_deg(ik_attempt, "pitch")
    return (
        ik_attempt.get("ik_joint_solution_rad") is not None
        and bool(ik_attempt.get("collision_free", False))
        and float(ik_attempt.get("ee_position_error_m", float("inf"))) <= float(position_tolerance_m)
        and roll_error_abs is not None
        and pitch_error_abs is not None
        and float(roll_error_abs) <= float(roll_tolerance_deg)
        and float(pitch_error_abs) <= float(pitch_tolerance_deg)
    )


def _joint_reset_rad_from_planning_config(planning_config) -> list[float]:
    return [math.radians(float(value)) for value in planning_config.joint_reset_deg]


def _finite_float_or_none(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _dict_get_first(mapping: dict[str, object], keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in mapping:
            return mapping.get(key)
    return None


def _arm_start_base_joint_candidates(payload: dict[str, object]) -> list[tuple[str, dict[str, object]]]:
    candidates: list[tuple[str, dict[str, object]]] = []

    direct_rad = _dict_get_first(
        payload,
        (
            "arm_approach_start_base_joint_rad",
            "arm_start_base_joint_rad",
            "start_base_joint_rad",
        ),
    )
    direct_deg = _dict_get_first(
        payload,
        (
            "arm_approach_start_base_joint_deg",
            "arm_start_base_joint_deg",
            "start_base_joint_deg",
        ),
    )
    if direct_rad is not None or direct_deg is not None:
        direct_candidate = {
            "joint_index": _dict_get_first(
                payload,
                (
                    "arm_approach_start_base_joint_index",
                    "arm_start_base_joint_index",
                    "start_base_joint_index",
                ),
            ),
            "command_joint_position_rad": direct_rad,
            "command_joint_position_deg": direct_deg,
        }
        candidates.append(("payload.arm_approach_start_base_joint", direct_candidate))

    for key in (
        "car_approach_arm_base_alignment_result",
        "last_arm_base_alignment_result",
        "arm_base_alignment_result",
    ):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            candidates.append((f"payload.{key}", candidate))

    for parent_key in (
        "latest_approach_result",
        "car_approach_result",
        "previous_car_approach_result",
    ):
        parent = payload.get(parent_key)
        if not isinstance(parent, dict):
            continue
        alignment = parent.get("arm_base_alignment_result")
        if isinstance(alignment, dict):
            candidates.append((f"payload.{parent_key}.arm_base_alignment_result", alignment))
        parent_rad = parent.get("arm_approach_start_base_joint_rad")
        parent_deg = parent.get("arm_approach_start_base_joint_deg")
        if parent_rad is not None or parent_deg is not None:
            candidates.append(
                (
                    f"payload.{parent_key}.arm_approach_start_base_joint",
                    {
                        "joint_index": parent.get("arm_approach_start_base_joint_index"),
                        "command_joint_position_rad": parent_rad,
                        "command_joint_position_deg": parent_deg,
                    },
                )
            )

    return candidates


def _arm_start_base_joint_override_from_payload(
    payload: dict[str, object],
    *,
    planning_config,
    arm_config: dict[str, object] | None,
) -> dict[str, object] | None:
    default_joint_index = int(
        os.getenv(
            "APPROACH_AGENT_ARM_START_BASE_JOINT_INDEX",
            str(ARM_APPROACH_START_BASE_JOINT_INDEX),
        )
    )
    rad_keys = (
        "command_joint_position_rad",
        "start_base_joint_position_rad",
        "target_joint_position_rad",
        "position_rad",
        "position",
        "value_rad",
    )
    deg_keys = (
        "command_joint_position_deg",
        "start_base_joint_position_deg",
        "target_joint_position_deg",
        "position_deg",
        "angle_deg",
        "value_deg",
    )

    for source_label, candidate in _arm_start_base_joint_candidates(payload):
        raw_joint_index = _dict_get_first(candidate, ("joint_index", "index"))
        try:
            joint_index = default_joint_index if raw_joint_index is None else int(raw_joint_index)
        except (TypeError, ValueError):
            continue

        raw_rad = _finite_float_or_none(_dict_get_first(candidate, rad_keys))
        raw_deg = None if raw_rad is not None else _finite_float_or_none(_dict_get_first(candidate, deg_keys))
        if raw_rad is None and raw_deg is None:
            continue

        raw_value_source = "rad" if raw_rad is not None else "deg"
        target_rad = float(raw_rad) if raw_rad is not None else math.radians(float(raw_deg))
        target_deg = math.degrees(target_rad)
        lower_deg, upper_deg = _joint_limit_deg_from_configs(
            joint_index=joint_index,
            planning_config=planning_config,
            arm_config=arm_config,
        )
        command_deg = _clamp_deg(target_deg, lower_deg, upper_deg)
        command_rad = math.radians(command_deg)
        return {
            "arm_approach_start_base_joint_override_available": True,
            "arm_approach_start_base_joint_override_source": source_label,
            "arm_approach_start_base_joint_override_value_source": raw_value_source,
            "arm_approach_start_base_joint_index": int(joint_index),
            "arm_approach_start_base_joint_raw_rad": float(target_rad),
            "arm_approach_start_base_joint_raw_deg": float(target_deg),
            "arm_approach_start_base_joint_rad": float(command_rad),
            "arm_approach_start_base_joint_deg": float(command_deg),
            "arm_approach_start_base_joint_limit_min_deg": lower_deg,
            "arm_approach_start_base_joint_limit_max_deg": upper_deg,
            "arm_approach_start_base_joint_clamped": not math.isclose(
                float(command_deg),
                float(target_deg),
                rel_tol=0.0,
                abs_tol=1e-9,
            ),
        }

    return None


def _apply_arm_start_base_joint_override(
    start_joint_positions_rad: list[float],
    start_base_joint_override: dict[str, object] | None,
) -> tuple[list[float], dict[str, object]]:
    start_joint_positions = [float(value) for value in start_joint_positions_rad]
    if start_base_joint_override is None:
        return (
            start_joint_positions,
            {
                "arm_approach_start_base_joint_override_applied": False,
                "arm_approach_start_joint_positions_source": "planning_reset",
            },
        )

    joint_index = int(start_base_joint_override["arm_approach_start_base_joint_index"])
    metadata = dict(start_base_joint_override)
    if not (0 <= joint_index < len(start_joint_positions)):
        metadata.update(
            {
                "arm_approach_start_base_joint_override_applied": False,
                "arm_approach_start_base_joint_override_error": (
                    f"joint index {joint_index} is outside start vector length {len(start_joint_positions)}"
                ),
                "arm_approach_start_joint_positions_source": "planning_reset",
            }
        )
        return start_joint_positions, metadata

    start_joint_positions[joint_index] = float(start_base_joint_override["arm_approach_start_base_joint_rad"])
    metadata.update(
        {
            "arm_approach_start_base_joint_override_applied": True,
            "arm_approach_start_joint_positions_source": "car_approach_base_alignment_plus_planning_reset",
            "arm_approach_start_joint_positions_rad": start_joint_positions,
            "arm_approach_start_joint_positions_deg": [math.degrees(value) for value in start_joint_positions],
        }
    )
    return start_joint_positions, metadata


def _joint_limit_deg_from_configs(
    *,
    joint_index: int,
    planning_config,
    arm_config: dict[str, object] | None,
) -> tuple[float | None, float | None]:
    lower_deg: float | None = None
    upper_deg: float | None = None

    try:
        planning_lower, planning_upper = planning_config.joint_bounds_deg[int(joint_index)]
        lower_deg = float(planning_lower)
        upper_deg = float(planning_upper)
    except (AttributeError, IndexError, TypeError, ValueError):
        pass

    try:
        joint_config = arm_config["joints"][int(joint_index)] if arm_config is not None else None
        if isinstance(joint_config, dict):
            arm_lower = joint_config.get("min_angle")
            arm_upper = joint_config.get("max_angle")
            if arm_lower is not None:
                lower_deg = float(arm_lower) if lower_deg is None else max(lower_deg, float(arm_lower))
            if arm_upper is not None:
                upper_deg = float(arm_upper) if upper_deg is None else min(upper_deg, float(arm_upper))
    except (KeyError, TypeError, ValueError):
        pass

    return lower_deg, upper_deg


def _clamp_deg(value_deg: float, lower_deg: float | None, upper_deg: float | None) -> float:
    clamped = float(value_deg)
    if lower_deg is not None:
        clamped = max(clamped, float(lower_deg))
    if upper_deg is not None:
        clamped = min(clamped, float(upper_deg))
    return clamped


def _with_wrist_from_target_grasp_yaw(
    solution: dict[str, object],
    record: dict[str, object],
    planning_config,
    arm_config: dict[str, object] | None,
) -> dict[str, object]:
    wrist_index = int(ARM_APPROACH_WRIST_JOINT_INDEX)
    goal_joint_positions = [float(value) for value in solution["ik_joint_solution_rad"]]
    if not (0 <= wrist_index < len(goal_joint_positions)):
        adjusted = dict(solution)
        adjusted["target_grasp_wrist_yaw_applied"] = False
        adjusted["target_grasp_wrist_yaw_error"] = (
            f"wrist joint index {wrist_index} is outside joint vector length {len(goal_joint_positions)}"
        )
        return adjusted

    try:
        wrist_reset_deg = float(planning_config.joint_reset_deg[wrist_index])
    except (AttributeError, IndexError, TypeError, ValueError):
        wrist_reset_deg = 90.0

    target_rot_pb = np.asarray(record["target_rot_pb"], dtype=np.float64).reshape(3, 3)
    target_yaw_rad = _axis_yaw_xy(target_rot_pb[:, 0], fallback_yaw_rad=0.0)
    target_yaw_normalized_deg = float(math.degrees(target_yaw_rad) % 360.0)
    yaw_reference_rad, yaw_delta_from_reference_rad = _nearest_half_turn_reference_and_delta_rad(target_yaw_rad)
    yaw_delta_from_reference_deg = math.degrees(yaw_delta_from_reference_rad)
    wrist_unclamped_deg = wrist_reset_deg + (
        float(ARM_APPROACH_TARGET_GRASP_YAW_TO_WRIST_SIGN) * yaw_delta_from_reference_deg
    )
    lower_deg, upper_deg = _joint_limit_deg_from_configs(
        joint_index=wrist_index,
        planning_config=planning_config,
        arm_config=arm_config,
    )
    wrist_yaw_min_deg = float(os.getenv("APPROACH_AGENT_ARM_WRIST_YAW_MIN_DEG", str(ARM_APPROACH_WRIST_YAW_MIN_DEG)))
    wrist_yaw_max_deg = float(os.getenv("APPROACH_AGENT_ARM_WRIST_YAW_MAX_DEG", str(ARM_APPROACH_WRIST_YAW_MAX_DEG)))
    if wrist_yaw_min_deg > wrist_yaw_max_deg:
        wrist_yaw_min_deg, wrist_yaw_max_deg = wrist_yaw_max_deg, wrist_yaw_min_deg

    command_lower_deg = wrist_yaw_min_deg if lower_deg is None else max(float(lower_deg), wrist_yaw_min_deg)
    command_upper_deg = wrist_yaw_max_deg if upper_deg is None else min(float(upper_deg), wrist_yaw_max_deg)
    wrist_target_deg = _clamp_deg(wrist_unclamped_deg, command_lower_deg, command_upper_deg)

    goal_joint_positions[wrist_index] = math.radians(wrist_target_deg)

    adjusted = dict(solution)
    adjusted["ik_joint_solution_rad"] = goal_joint_positions
    adjusted["ik_joint_solution_deg"] = [math.degrees(value) for value in goal_joint_positions]
    adjusted["target_grasp_wrist_yaw_applied"] = True
    adjusted["target_grasp_wrist_joint_index"] = wrist_index
    adjusted["target_grasp_yaw_reference_period_deg"] = float(ARM_APPROACH_TARGET_GRASP_YAW_REFERENCE_PERIOD_DEG)
    adjusted["target_grasp_yaw_nearest_reference_rad"] = float(yaw_reference_rad)
    adjusted["target_grasp_yaw_nearest_reference_deg"] = float(math.degrees(yaw_reference_rad) % 360.0)
    adjusted["target_grasp_yaw_rad"] = float(target_yaw_rad)
    adjusted["target_grasp_yaw_deg"] = float(math.degrees(target_yaw_rad))
    adjusted["target_grasp_yaw_normalized_deg"] = target_yaw_normalized_deg
    adjusted["target_grasp_yaw_delta_from_nearest_reference_rad"] = float(yaw_delta_from_reference_rad)
    adjusted["target_grasp_yaw_delta_from_nearest_reference_deg"] = float(yaw_delta_from_reference_deg)
    adjusted["target_grasp_wrist_reset_deg"] = float(wrist_reset_deg)
    adjusted["target_grasp_wrist_sign"] = float(ARM_APPROACH_TARGET_GRASP_YAW_TO_WRIST_SIGN)
    adjusted["target_grasp_wrist_unclamped_deg"] = float(wrist_unclamped_deg)
    adjusted["target_grasp_wrist_target_deg"] = float(wrist_target_deg)
    adjusted["target_grasp_wrist_target_rad"] = float(math.radians(wrist_target_deg))
    adjusted["target_grasp_wrist_limit_min_deg"] = lower_deg
    adjusted["target_grasp_wrist_limit_max_deg"] = upper_deg
    adjusted["target_grasp_wrist_yaw_min_deg"] = float(wrist_yaw_min_deg)
    adjusted["target_grasp_wrist_yaw_max_deg"] = float(wrist_yaw_max_deg)
    adjusted["target_grasp_wrist_command_min_deg"] = float(command_lower_deg)
    adjusted["target_grasp_wrist_command_max_deg"] = float(command_upper_deg)
    adjusted["target_grasp_wrist_clamped"] = not math.isclose(
        float(wrist_target_deg),
        float(wrist_unclamped_deg),
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    return adjusted


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


def _with_pre_gripper_forward_ee(
    solution: dict[str, object],
    *,
    p_mod,
    robot_id: int,
    controllable_joint_ids: list[int],
    planning_config,
    arm_config: dict[str, object],
    base_link_xy: tuple[float, float],
    base_link_yaw_rad: float,
    up_distance_m: float,
    distance_m: float,
    z_offset_m: float,
) -> dict[str, object]:
    adjusted_solution = dict(solution)
    adjusted_solution["pre_gripper_forward_enabled"] = True
    adjusted_solution["pre_gripper_up_distance_m"] = float(up_distance_m)
    adjusted_solution["pre_gripper_up_distance_cm"] = float(up_distance_m) * 100.0
    adjusted_solution["pre_gripper_vertical_direction"] = "up"
    adjusted_solution["pre_gripper_forward_distance_m"] = float(distance_m)
    adjusted_solution["pre_gripper_forward_distance_cm"] = float(distance_m) * 100.0
    adjusted_solution["pre_gripper_forward_z_offset_m"] = float(z_offset_m)
    adjusted_solution["pre_gripper_forward_sequence"] = ["up", "forward"]

    goal_joint_positions = [float(value) for value in solution["ik_joint_solution_rad"]]
    if len(goal_joint_positions) != len(controllable_joint_ids):
        adjusted_solution["pre_gripper_forward_ik_success"] = False
        adjusted_solution["pre_gripper_forward_error"] = (
            "joint vector length mismatch: "
            f"{len(goal_joint_positions)} != {len(controllable_joint_ids)}"
        )
        return adjusted_solution

    base_xyz = [
        float(base_link_xy[0]),
        float(base_link_xy[1]),
        float(planning_config.initial_height),
    ]
    base_quat = p_mod.getQuaternionFromEuler([0.0, 0.0, float(base_link_yaw_rad)])
    p_mod.resetBasePositionAndOrientation(robot_id, base_xyz, base_quat)
    _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, goal_joint_positions)
    p_mod.performCollisionDetection()

    ee_state = p_mod.getLinkState(
        robot_id,
        int(planning_config.ee_link_index),
        computeForwardKinematics=True,
    )
    ee_position = np.asarray(ee_state[4], dtype=np.float64).reshape(3)
    ee_orientation = np.asarray(ee_state[5], dtype=np.float64).reshape(4)
    ee_rotation = np.asarray(p_mod.getMatrixFromQuaternion(ee_orientation.tolist()), dtype=np.float64).reshape(3, 3)
    forward_axis = ee_rotation[:, 0]

    def _solve_target_position(
        target_position: np.ndarray,
        stage_name: str,
    ) -> tuple[list[float] | None, list[float] | None, np.ndarray | None, float | None, str | None]:
        try:
            raw_solution = p_mod.calculateInverseKinematics(
                robot_id,
                int(planning_config.ee_link_index),
                targetPosition=target_position.astype(float).tolist(),
                targetOrientation=ee_orientation.astype(float).tolist(),
            )
        except Exception as exc:
            return None, None, None, None, f"{stage_name} IK failed: {exc}"

        joint_positions = [float(value) for value in raw_solution[: len(controllable_joint_ids)]]
        if len(joint_positions) != len(controllable_joint_ids):
            return (
                None,
                None,
                None,
                None,
                f"{stage_name} IK returned unexpected joint count: "
                f"{len(joint_positions)} != {len(controllable_joint_ids)}",
            )

        joint_deg = [math.degrees(value) for value in joint_positions]
        for joint_index, joint_value_deg in enumerate(joint_deg):
            lower_deg, upper_deg = _joint_limit_deg_from_configs(
                joint_index=joint_index,
                planning_config=planning_config,
                arm_config=arm_config,
            )
            joint_deg[joint_index] = _clamp_deg(joint_value_deg, lower_deg, upper_deg)
        joint_positions = [math.radians(value) for value in joint_deg]

        gripper_index = int(ARM_APPROACH_GRIPPER_OPEN_JOINT_INDEX)
        if 0 <= gripper_index < len(joint_positions):
            joint_positions[gripper_index] = goal_joint_positions[gripper_index]
            joint_deg[gripper_index] = math.degrees(goal_joint_positions[gripper_index])

        p_mod.resetBasePositionAndOrientation(robot_id, base_xyz, base_quat)
        _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, joint_positions)
        p_mod.performCollisionDetection()
        stage_ee_state = p_mod.getLinkState(
            robot_id,
            int(planning_config.ee_link_index),
            computeForwardKinematics=True,
        )
        achieved_position = np.asarray(stage_ee_state[4], dtype=np.float64).reshape(3)
        target_error = float(np.linalg.norm(achieved_position - target_position))
        return joint_positions, joint_deg, achieved_position, target_error, None

    up_target_position = ee_position.copy()
    up_target_position[2] += float(up_distance_m)
    up_target_position[2] += float(z_offset_m)
    up_joint_positions, up_joint_deg, up_ee_position, up_target_error, up_error = _solve_target_position(
        up_target_position,
        "up",
    )
    if up_error is not None or up_joint_positions is None or up_joint_deg is None or up_ee_position is None:
        adjusted_solution["pre_gripper_forward_ik_success"] = False
        adjusted_solution["pre_gripper_forward_error"] = up_error or "up IK failed"
        adjusted_solution["pre_gripper_forward_failed_stage"] = "up"
        return adjusted_solution

    forward_target_position = up_target_position + (forward_axis * float(distance_m))
    (
        forward_joint_positions,
        forward_joint_deg,
        forward_ee_position,
        target_error,
        forward_error,
    ) = _solve_target_position(forward_target_position, "forward")
    if (
        forward_error is not None
        or forward_joint_positions is None
        or forward_joint_deg is None
        or forward_ee_position is None
    ):
        adjusted_solution["pre_gripper_forward_ik_success"] = False
        adjusted_solution["pre_gripper_forward_error"] = forward_error or "forward IK failed"
        adjusted_solution["pre_gripper_forward_failed_stage"] = "forward"
        return adjusted_solution

    achieved_delta = forward_ee_position - ee_position

    adjusted_solution["pre_gripper_forward_ik_success"] = True
    adjusted_solution["pre_gripper_forward_start_ee_position_xyz"] = ee_position.astype(float).tolist()
    adjusted_solution["pre_gripper_forward_axis_xyz"] = forward_axis.astype(float).tolist()
    adjusted_solution["pre_gripper_up_target_position_xyz"] = up_target_position.astype(float).tolist()
    adjusted_solution["pre_gripper_up_achieved_ee_position_xyz"] = up_ee_position.astype(float).tolist()
    adjusted_solution["pre_gripper_up_achieved_delta_xyz"] = (up_ee_position - ee_position).astype(float).tolist()
    adjusted_solution["pre_gripper_up_target_error_m"] = float(up_target_error)
    adjusted_solution["pre_gripper_forward_target_position_xyz"] = forward_target_position.astype(float).tolist()
    adjusted_solution["pre_gripper_forward_achieved_ee_position_xyz"] = forward_ee_position.astype(float).tolist()
    adjusted_solution["pre_gripper_forward_achieved_delta_xyz"] = achieved_delta.astype(float).tolist()
    adjusted_solution["pre_gripper_forward_achieved_distance_m"] = float(np.linalg.norm(achieved_delta))
    adjusted_solution["pre_gripper_forward_target_error_m"] = target_error
    adjusted_solution["post_arrival_up_joint_positions_rad"] = up_joint_positions
    adjusted_solution["post_arrival_up_joint_positions_deg"] = up_joint_deg
    adjusted_solution["post_arrival_forward_joint_positions_rad"] = forward_joint_positions
    adjusted_solution["post_arrival_forward_joint_positions_deg"] = forward_joint_deg
    adjusted_solution["post_arrival_joint_positions_sequence_rad"] = [
        up_joint_positions,
        forward_joint_positions,
    ]
    adjusted_solution["post_arrival_joint_positions_sequence_deg"] = [
        up_joint_deg,
        forward_joint_deg,
    ]
    adjusted_solution["post_arrival_joint_positions_sequence_labels"] = ["up", "forward"]
    return adjusted_solution


def _prefix_path_check_result(prefix: str, result: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}{key}": value for key, value in result.items()}


def _check_interpolated_joint_path_collision(
    *,
    p_mod,
    robot_id: int,
    controllable_joint_ids: list[int],
    planning_config,
    start_joint_positions_rad: list[float],
    goal_joint_positions_rad: list[float],
    interpolation_steps: int,
    include_start: bool,
    obstacle_body_ids: list[int],
    base_link_xy: tuple[float, float],
    base_link_yaw_rad: float,
) -> dict[str, object]:
    waypoints = build_linear_joint_trajectory(
        start_joint_positions_rad,
        goal_joint_positions_rad,
        interpolation_steps=interpolation_steps,
        include_start=include_start,
    )
    result: dict[str, object] = {
        "interpolated_path_checked": True,
        "interpolated_path_collision_free": True,
        "interpolated_path_waypoint_count": int(len(waypoints)),
        "interpolated_path_interpolation_steps": int(interpolation_steps),
        "interpolated_path_include_start": bool(include_start),
        "interpolated_path_first_collision_index": None,
        "interpolated_path_first_collision_joint_rad": None,
        "interpolated_path_first_collision_joint_deg": None,
    }
    if not obstacle_body_ids:
        result["interpolated_path_collision_check_note"] = "skipped_no_obstacles"
        return result

    base_xyz = [
        float(base_link_xy[0]),
        float(base_link_xy[1]),
        float(planning_config.initial_height),
    ]
    base_quat = p_mod.getQuaternionFromEuler([0.0, 0.0, float(base_link_yaw_rad)])
    for waypoint_index, waypoint_rad in enumerate(waypoints):
        p_mod.resetBasePositionAndOrientation(robot_id, base_xyz, base_quat)
        _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, waypoint_rad)
        p_mod.performCollisionDetection()
        if _robot_collides_with_obstacles(p_mod, robot_id, obstacle_body_ids):
            result["interpolated_path_collision_free"] = False
            result["interpolated_path_first_collision_index"] = int(waypoint_index)
            result["interpolated_path_first_collision_joint_rad"] = [float(value) for value in waypoint_rad]
            result["interpolated_path_first_collision_joint_deg"] = [
                float(math.degrees(value)) for value in waypoint_rad
            ]
            return result

    return result


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
    target_object_center_pb_xyz: np.ndarray | None = None,
) -> list[dict[str, object]]:
    current_ee_position = np.asarray(current_ee_position_xyz, dtype=np.float64).reshape(3)
    target_object_center_pb = (
        None
        if target_object_center_pb_xyz is None
        else np.asarray(target_object_center_pb_xyz, dtype=np.float64).reshape(3)
    )
    ranked_records = [dict(record) for record in visualization_records]
    for record in ranked_records:
        target_pb = np.asarray(record["target_pb"], dtype=np.float64).reshape(3)
        target_rot_pb = np.asarray(record["target_rot_pb"], dtype=np.float64).reshape(3, 3)
        target_yaw = _axis_yaw_xy(target_rot_pb[:, 0], fallback_yaw_rad=0.0)
        target_yaw_normalized_deg = float(math.degrees(target_yaw) % 360.0)
        yaw_reference_rad, yaw_delta_from_reference_rad = _nearest_half_turn_reference_and_delta_rad(target_yaw)
        yaw_error_rad = abs(yaw_delta_from_reference_rad)
        distance_m = float(np.linalg.norm(target_pb - current_ee_position))
        center_distance_m = (
            float("inf")
            if target_object_center_pb is None
            else float(np.linalg.norm(target_pb - target_object_center_pb))
        )
        record["current_ee_distance_to_target_grasp_m"] = distance_m
        record["target_grasp_distance_to_object_center_m"] = center_distance_m
        record["target_object_center_pb_xyz"] = (
            None
            if target_object_center_pb is None
            else target_object_center_pb.astype(float).tolist()
        )
        record["current_ee_yaw_error_to_target_grasp_rad"] = float(yaw_error_rad)
        record["current_ee_yaw_error_to_target_grasp_deg"] = float(math.degrees(yaw_error_rad))
        record["target_grasp_yaw_reference_period_deg"] = float(ARM_APPROACH_TARGET_GRASP_YAW_REFERENCE_PERIOD_DEG)
        record["target_grasp_yaw_nearest_reference_rad"] = float(yaw_reference_rad)
        record["target_grasp_yaw_nearest_reference_deg"] = float(math.degrees(yaw_reference_rad) % 360.0)
        record["target_grasp_yaw_rad"] = float(target_yaw)
        record["target_grasp_yaw_deg"] = float(math.degrees(target_yaw))
        record["target_grasp_yaw_normalized_deg"] = target_yaw_normalized_deg
        record["target_grasp_yaw_delta_from_nearest_reference_rad"] = float(yaw_delta_from_reference_rad)
        record["target_grasp_yaw_delta_from_nearest_reference_deg"] = float(math.degrees(yaw_delta_from_reference_rad))

    ranked_records.sort(
        key=lambda record: (
            float(record["target_grasp_distance_to_object_center_m"]),
            float(record["current_ee_distance_to_target_grasp_m"]),
            int(record.get("rank", 0)),
        )
    )
    for order_index, record in enumerate(ranked_records, start=1):
        record["target_sample_order"] = int(order_index)
    return ranked_records


def _target_object_center_pb_from_camera_points(
    target_object_points_camera: np.ndarray | None,
    *,
    camera_to_pb_rotation: np.ndarray,
    camera_position_pb: np.ndarray,
    mirror_camera_x: bool,
) -> np.ndarray | None:
    if target_object_points_camera is None:
        return None
    points_camera = np.asarray(target_object_points_camera, dtype=np.float64).reshape(-1, 3)
    points_camera = points_camera[np.all(np.isfinite(points_camera), axis=1)]
    if len(points_camera) == 0:
        return None
    if mirror_camera_x:
        points_camera = _mirror_camera_points_x(points_camera)
    center_camera = np.mean(points_camera, axis=0).reshape(1, 3)
    center_pb = _transform_camera_points_to_local_pb(
        center_camera,
        camera_to_pb_rotation=camera_to_pb_rotation,
        camera_position_pb=camera_position_pb,
    )
    return np.asarray(center_pb, dtype=np.float64).reshape(3)


def _resolve_debug_output_dir() -> Path:
    raw_override = os.getenv("ARM_APPROACH_DEBUG_OUTPUT_DIR", "").strip()
    candidates = [
        Path(raw_override).expanduser() if raw_override else ARM_APPROACH_DEBUG_OUTPUT_DIR,
        Path("/tmp/arm_approach_outputs"),
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except Exception:
            continue
        if os.access(candidate, os.W_OK):
            return candidate
    return ARM_APPROACH_DEBUG_OUTPUT_DIR


def _derive_topdown_output_path(side_output_path: Path) -> Path:
    side_name = side_output_path.name
    if side_name.startswith("arm_target_side_"):
        return side_output_path.with_name(side_name.replace("arm_target_side_", "arm_target_topdown_", 1))
    return side_output_path.with_name(f"{side_output_path.stem}_topdown{side_output_path.suffix}")


def _capture_target_arrival_side_view(
    *,
    p_mod,
    pybullet_data,
    planning_config,
    arm_config,
    live_scene,
    selected_solution: dict[str, object],
    output_path: Path | None = None,
) -> dict[str, object]:
    client_id = None
    try:
        client_id = p_mod.connect(p_mod.DIRECT)
        if client_id < 0:
            raise RuntimeError("PyBullet DIRECT unavailable.")

        p_mod.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_mod.resetSimulation()
        p_mod.setGravity(0.0, 0.0, -9.8)
        p_mod.loadURDF("plane.urdf")

        voxel_size = float(live_scene.voxel_size_m)
        half_extents = [voxel_size / 2.0] * 3
        col_shape = p_mod.createCollisionShape(p_mod.GEOM_BOX, halfExtents=half_extents)
        vis_shape = p_mod.createVisualShape(
            p_mod.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.8, 0.2, 0.2, 0.8],
        )
        for voxel_center in np.asarray(live_scene.voxel_centers_pb, dtype=np.float64).reshape(-1, 3):
            p_mod.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=voxel_center.astype(float).tolist(),
            )

        base_orientation_rad = [math.radians(v) for v in planning_config.base_orientation_euler_deg]
        robot_id = p_mod.loadURDF(
            planning_config.urdf_path,
            useFixedBase=True,
            basePosition=[0.0, 0.0, float(planning_config.initial_height)],
            baseOrientation=p_mod.getQuaternionFromEuler(base_orientation_rad),
        )
        expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])
        controllable_joint_ids, _ = _find_controllable_joints(p_mod, robot_id, expected_joint_count)
        joint_solution_rad = [float(value) for value in selected_solution["ik_joint_solution_rad"]]
        _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, joint_solution_rad)
        p_mod.performCollisionDetection()

        ee_state = p_mod.getLinkState(
            robot_id,
            int(planning_config.ee_link_index),
            computeForwardKinematics=True,
        )
        ee_position = np.asarray(ee_state[4], dtype=np.float64)
        target_pb = np.asarray(selected_solution.get("target_pb", ee_position), dtype=np.float64).reshape(3)

        target_visual_shape = p_mod.createVisualShape(
            p_mod.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[1.0, 0.85, 0.1, 1.0],
        )
        p_mod.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=target_visual_shape,
            basePosition=target_pb.astype(float).tolist(),
        )

        resolved_output = output_path
        if resolved_output is None:
            debug_output_dir = _resolve_debug_output_dir()
            resolved_output = debug_output_dir / f"arm_target_side_{int(time.time() * 1000)}.ppm"
        topdown_output = _derive_topdown_output_path(resolved_output)

        camera_target = [
            float(target_pb[0]),
            float(target_pb[1]),
            float(max(target_pb[2], ee_position[2])),
        ]
        scene_points = [target_pb.reshape(1, 3), ee_position.reshape(1, 3)]
        voxel_points = np.asarray(live_scene.voxel_centers_pb, dtype=np.float64).reshape(-1, 3)
        if len(voxel_points) > 0:
            finite_voxels = voxel_points[np.all(np.isfinite(voxel_points), axis=1)]
            if len(finite_voxels) > 0:
                scene_points.append(finite_voxels)
        scene_xyz = np.vstack(scene_points)
        scene_xy_extent_m = float(np.max(np.ptp(scene_xyz[:, :2], axis=0))) if len(scene_xyz) > 0 else 0.0
        side_distance_m = max(
            float(os.getenv("ARM_APPROACH_SIDE_VIEW_DISTANCE_M", str(ARM_APPROACH_SIDE_VIEW_DISTANCE_M))),
            (scene_xy_extent_m * 1.25) + 0.20,
        )
        topdown_distance_m = max(
            float(os.getenv("ARM_APPROACH_TOPDOWN_VIEW_DISTANCE_M", str(ARM_APPROACH_TOPDOWN_VIEW_DISTANCE_M))),
            (scene_xy_extent_m * 1.50) + 0.18,
        )
        _render_debug_ppm(
            p_mod,
            np,
            resolved_output,
            width=int(getattr(planning_config, "debug_render_width", 960)),
            height=int(getattr(planning_config, "debug_render_height", 720)),
            camera_target_position=camera_target,
            camera_distance=side_distance_m,
            camera_yaw_deg=float(os.getenv("ARM_APPROACH_SIDE_VIEW_YAW_DEG", str(ARM_APPROACH_SIDE_VIEW_YAW_DEG))),
            camera_pitch_deg=float(os.getenv("ARM_APPROACH_SIDE_VIEW_PITCH_DEG", str(ARM_APPROACH_SIDE_VIEW_PITCH_DEG))),
        )
        _render_debug_ppm(
            p_mod,
            np,
            topdown_output,
            width=int(getattr(planning_config, "debug_render_width", 960)),
            height=int(getattr(planning_config, "debug_render_height", 720)),
            camera_target_position=target_pb.astype(float).tolist(),
            camera_distance=topdown_distance_m,
            camera_yaw_deg=float(
                os.getenv("ARM_APPROACH_TOPDOWN_VIEW_YAW_DEG", str(ARM_APPROACH_TOPDOWN_VIEW_YAW_DEG))
            ),
            camera_pitch_deg=float(
                os.getenv("ARM_APPROACH_TOPDOWN_VIEW_PITCH_DEG", str(ARM_APPROACH_TOPDOWN_VIEW_PITCH_DEG))
            ),
        )
        return {
            "side_view_saved": True,
            "side_view_path": str(resolved_output),
            "topdown_view_saved": True,
            "topdown_view_path": str(topdown_output),
        }
    except Exception as exc:
        return {
            "side_view_saved": False,
            "topdown_view_saved": False,
            "side_view_error": str(exc),
        }
    finally:
        if client_id is not None:
            try:
                p_mod.disconnect(client_id)
            except Exception:
                pass


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
    ik_attempt["target_grasp_distance_to_object_center_m"] = float(
        record.get("target_grasp_distance_to_object_center_m", float("inf"))
    )
    ik_attempt["target_object_center_pb_xyz"] = record.get("target_object_center_pb_xyz")
    ik_attempt["current_ee_yaw_error_to_target_grasp_deg"] = float(
        record.get("current_ee_yaw_error_to_target_grasp_deg", float("inf"))
    )
    return ik_attempt


def _best_effort_ik_sort_key(ik_attempt: dict[str, object]) -> tuple[float, float, float, int]:
    orientation_error = _roll_pitch_error_abs_max_deg(ik_attempt)
    return (
        float(ik_attempt.get("ee_position_error_m", float("inf"))),
        float("inf") if orientation_error is None else float(orientation_error),
        float(ik_attempt.get("joint_reset_delta_norm_l2", float("inf"))),
        int(ik_attempt.get("target_sample_order", 0)),
    )


def _feasible_ik_sort_key(ik_attempt: dict[str, object]) -> tuple[float, float, float, int]:
    orientation_error = _roll_pitch_error_abs_max_deg(ik_attempt)
    return (
        float("inf") if orientation_error is None else float(orientation_error),
        float(ik_attempt.get("ee_position_error_m", float("inf"))),
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
    start_base_joint_override = _arm_start_base_joint_override_from_payload(
        payload,
        planning_config=planning_config,
        arm_config=arm_config,
    )
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
        mirror_camera_x=True,
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
            mirror_camera_x=True,
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

        current_ee_position_xyz, _current_ee_orientation_xyzw = _current_reset_ee_pose(
            p_mod=p_mod,
            robot_id=robot_id,
            controllable_joint_ids=controllable_joint_ids,
            planning_config=planning_config,
            base_link_xy=CURRENT_BASE_LINK_LOCAL_XY,
            base_link_yaw_rad=CURRENT_BASE_LINK_LOCAL_YAW_RAD,
        )

        target_object_center_pb = _target_object_center_pb_from_camera_points(
            target_object_points_camera,
            camera_to_pb_rotation=camera_to_pb_rotation,
            camera_position_pb=camera_position_pb,
            mirror_camera_x=True,
        )

        # Rank target grasp poses by object-center distance, then current gripper EE distance.
        ranked_records = _rank_grasp_records_by_current_ee_distance(
            visualization_records,
            current_ee_position_xyz=current_ee_position_xyz,
            target_object_center_pb_xyz=target_object_center_pb,
        )

        selected_solution = None
        selected_record = None
        best_effort_solution = None
        best_effort_record = None
        feasible_solutions: list[tuple[dict[str, object], dict[str, object], list[float]]] = []
        attempted_count = 0
        path_checked_count = 0
        path_collision_rejected_count = 0
        pre_gripper_forward_rejected_count = 0
        position_tolerance_m = float(
            os.getenv(
                "APPROACH_AGENT_ARM_POSITION_TOLERANCE_M",
                str(max(float(cfg.get("position_tolerance_m", planning_config.position_tolerance_m)), ARM_APPROACH_POSITION_TOLERANCE_M)),
            )
        )
        rpy_tolerance_deg = float(
            os.getenv("APPROACH_AGENT_ARM_RPY_TOLERANCE_DEG", str(ARM_APPROACH_RPY_TOLERANCE_DEG))
        )
        roll_tolerance_deg = float(
            os.getenv("APPROACH_AGENT_ARM_ROLL_TOLERANCE_DEG", str(ARM_APPROACH_ROLL_TOLERANCE_DEG))
        )
        pitch_tolerance_deg = float(
            os.getenv("APPROACH_AGENT_ARM_PITCH_TOLERANCE_DEG", str(rpy_tolerance_deg))
        )
        max_target_grasp_poses = max(
            1,
            int(os.getenv("APPROACH_AGENT_ARM_MAX_TARGET_GRASP_POSES", str(ARM_APPROACH_MAX_TARGET_GRASP_POSES))),
        )
        arm_move_config = config_from_environment(planning_config=planning_config)
        pre_gripper_forward_enabled = bool(arm_move_config.forward_before_gripper_close) and _env_flag(
            "APPROACH_AGENT_ARM_PRE_GRIPPER_FORWARD_ENABLED",
            ARM_APPROACH_PRE_GRIPPER_FORWARD_ENABLED,
        )
        pre_gripper_up_distance_m = max(
            0.0,
            float(
                os.getenv(
                    "APPROACH_AGENT_ARM_PRE_GRIPPER_UP_DISTANCE_M",
                    str(ARM_APPROACH_PRE_GRIPPER_UP_DISTANCE_M),
                )
            ),
        )
        pre_gripper_forward_distance_m = float(
            os.getenv(
                "APPROACH_AGENT_ARM_PRE_GRIPPER_FORWARD_DISTANCE_M",
                str(ARM_APPROACH_PRE_GRIPPER_FORWARD_DISTANCE_M),
            )
        )
        pre_gripper_forward_z_offset_m = float(
            os.getenv(
                "APPROACH_AGENT_ARM_PRE_GRIPPER_FORWARD_Z_OFFSET_M",
                str(ARM_APPROACH_PRE_GRIPPER_FORWARD_Z_OFFSET_M),
            )
        )
        candidate_records = ranked_records[:max_target_grasp_poses]
        for record in candidate_records:
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
            ik_attempt = _with_wrist_from_target_grasp_yaw(
                ik_attempt,
                record,
                planning_config,
                arm_config,
            )
            ik_attempt["current_amcl_pb_xyz"] = current_amcl_pb_xyz.astype(float).tolist()
            ik_attempt["current_amcl_pb_yaw_rad"] = float(current_pb_yaw)
            
            feasible = _arm_approach_ik_attempt_is_feasible(
                ik_attempt,
                position_tolerance_m=position_tolerance_m,
                roll_tolerance_deg=roll_tolerance_deg,
                pitch_tolerance_deg=pitch_tolerance_deg,
            )
            ik_attempt["ik_reachable"] = bool(feasible)
            ik_attempt["arm_approach_position_tolerance_m"] = float(position_tolerance_m)
            ik_attempt["arm_approach_rpy_tolerance_deg"] = float(rpy_tolerance_deg)
            ik_attempt["arm_approach_roll_tolerance_deg"] = float(roll_tolerance_deg)
            ik_attempt["arm_approach_pitch_tolerance_deg"] = float(pitch_tolerance_deg)
            ik_attempt["arm_approach_roll_pitch_tolerance_deg"] = float(rpy_tolerance_deg)
            ik_attempt["arm_approach_roll_pitch_tolerance_by_axis_deg"] = {
                "roll": float(roll_tolerance_deg),
                "pitch": float(pitch_tolerance_deg),
            }
            ik_attempt["arm_approach_orientation_tolerance_axes"] = ["roll", "pitch"]
            ik_attempt["arm_approach_yaw_tolerance_applied"] = False
            ik_attempt["interpolated_path_checked"] = False
            ik_attempt["interpolated_path_collision_free"] = False
            ik_attempt["arm_motion_feasible"] = False
            if _ik_attempt_can_be_best_effort(ik_attempt) and (
                best_effort_solution is None
                or _best_effort_ik_sort_key(ik_attempt) < _best_effort_ik_sort_key(best_effort_solution)
            ):
                best_effort_solution = ik_attempt
                best_effort_record = record
            
            if feasible:
                motion_solution, start_joint_positions_rad = _with_open_gripper_before_motion(
                    ik_attempt,
                    planning_config,
                )
                start_joint_positions_rad, start_base_metadata = _apply_arm_start_base_joint_override(
                    start_joint_positions_rad,
                    start_base_joint_override,
                )
                ik_attempt.update(start_base_metadata)
                motion_solution.update(start_base_metadata)
                motion_solution["pre_gripper_forward_enabled"] = bool(pre_gripper_forward_enabled)
                motion_solution["pre_gripper_up_distance_m"] = float(pre_gripper_up_distance_m)
                motion_solution["pre_gripper_up_distance_cm"] = float(pre_gripper_up_distance_m) * 100.0
                motion_solution["pre_gripper_vertical_direction"] = "up"
                motion_solution["pre_gripper_forward_distance_m"] = float(pre_gripper_forward_distance_m)
                motion_solution["pre_gripper_forward_distance_cm"] = float(pre_gripper_forward_distance_m) * 100.0
                motion_solution["pre_gripper_forward_z_offset_m"] = float(pre_gripper_forward_z_offset_m)
                if pre_gripper_forward_enabled:
                    motion_solution = _with_pre_gripper_forward_ee(
                        motion_solution,
                        p_mod=p_mod,
                        robot_id=robot_id,
                        controllable_joint_ids=controllable_joint_ids,
                        planning_config=planning_config,
                        arm_config=arm_config,
                        base_link_xy=CURRENT_BASE_LINK_LOCAL_XY,
                        base_link_yaw_rad=CURRENT_BASE_LINK_LOCAL_YAW_RAD,
                        up_distance_m=pre_gripper_up_distance_m,
                        distance_m=pre_gripper_forward_distance_m,
                        z_offset_m=pre_gripper_forward_z_offset_m,
                    )
                path_check = _check_interpolated_joint_path_collision(
                    p_mod=p_mod,
                    robot_id=robot_id,
                    controllable_joint_ids=controllable_joint_ids,
                    planning_config=planning_config,
                    start_joint_positions_rad=start_joint_positions_rad,
                    goal_joint_positions_rad=[float(value) for value in motion_solution["ik_joint_solution_rad"]],
                    interpolation_steps=arm_move_config.interpolation_steps,
                    include_start=arm_move_config.publish_start,
                    obstacle_body_ids=obstacle_body_ids,
                    base_link_xy=CURRENT_BASE_LINK_LOCAL_XY,
                    base_link_yaw_rad=CURRENT_BASE_LINK_LOCAL_YAW_RAD,
                )
                path_checked_count += 1
                ik_attempt.update(path_check)
                motion_solution.update(path_check)
                path_collision_free = bool(path_check["interpolated_path_collision_free"])
                ik_attempt["arm_motion_feasible"] = path_collision_free
                motion_solution["arm_motion_feasible"] = path_collision_free
                if path_collision_free and pre_gripper_forward_enabled:
                    if bool(motion_solution.get("pre_gripper_forward_ik_success", False)):
                        raw_post_arrival_sequence = motion_solution.get("post_arrival_joint_positions_sequence_rad")
                        if isinstance(raw_post_arrival_sequence, list) and raw_post_arrival_sequence:
                            post_arrival_sequence = [
                                [float(value) for value in point]
                                for point in raw_post_arrival_sequence
                            ]
                        else:
                            post_arrival_sequence = [
                                [
                                    float(value)
                                    for value in motion_solution["post_arrival_forward_joint_positions_rad"]
                                ]
                            ]
                        post_arrival_labels = motion_solution.get(
                            "post_arrival_joint_positions_sequence_labels",
                            [],
                        )
                        segment_start = [float(value) for value in motion_solution["ik_joint_solution_rad"]]
                        last_forward_path_check: dict[str, object] | None = None
                        path_collision_free = True
                        for segment_index, segment_goal in enumerate(post_arrival_sequence):
                            forward_path_check = _check_interpolated_joint_path_collision(
                                p_mod=p_mod,
                                robot_id=robot_id,
                                controllable_joint_ids=controllable_joint_ids,
                                planning_config=planning_config,
                                start_joint_positions_rad=segment_start,
                                goal_joint_positions_rad=[float(value) for value in segment_goal],
                                interpolation_steps=arm_move_config.interpolation_steps,
                                include_start=False,
                                obstacle_body_ids=obstacle_body_ids,
                                base_link_xy=CURRENT_BASE_LINK_LOCAL_XY,
                                base_link_yaw_rad=CURRENT_BASE_LINK_LOCAL_YAW_RAD,
                            )
                            path_checked_count += 1
                            last_forward_path_check = forward_path_check
                            step_prefix = f"pre_gripper_forward_step_{segment_index + 1}_"
                            prefixed_step_path_check = _prefix_path_check_result(
                                step_prefix,
                                forward_path_check,
                            )
                            ik_attempt.update(prefixed_step_path_check)
                            motion_solution.update(prefixed_step_path_check)
                            if not bool(forward_path_check["interpolated_path_collision_free"]):
                                path_collision_free = False
                                ik_attempt["pre_gripper_forward_failed_step_index"] = int(segment_index + 1)
                                motion_solution["pre_gripper_forward_failed_step_index"] = int(segment_index + 1)
                                if isinstance(post_arrival_labels, list) and segment_index < len(post_arrival_labels):
                                    ik_attempt["pre_gripper_forward_failed_step_label"] = str(
                                        post_arrival_labels[segment_index]
                                    )
                                    motion_solution["pre_gripper_forward_failed_step_label"] = str(
                                        post_arrival_labels[segment_index]
                                    )
                                break
                            segment_start = [float(value) for value in segment_goal]

                        if last_forward_path_check is not None:
                            prefixed_forward_path_check = _prefix_path_check_result(
                                "pre_gripper_forward_",
                                last_forward_path_check,
                            )
                            ik_attempt.update(prefixed_forward_path_check)
                            motion_solution.update(prefixed_forward_path_check)
                        ik_attempt["pre_gripper_forward_interpolated_path_step_count"] = int(
                            len(post_arrival_sequence)
                        )
                        motion_solution["pre_gripper_forward_interpolated_path_step_count"] = int(
                            len(post_arrival_sequence)
                        )
                        ik_attempt["arm_motion_feasible"] = path_collision_free
                        motion_solution["arm_motion_feasible"] = path_collision_free
                        if not path_collision_free:
                            pre_gripper_forward_rejected_count += 1
                    else:
                        path_collision_free = False
                        ik_attempt["arm_motion_feasible"] = False
                        motion_solution["arm_motion_feasible"] = False
                        ik_attempt["pre_gripper_forward_ik_success"] = False
                        ik_attempt["pre_gripper_forward_error"] = motion_solution.get(
                            "pre_gripper_forward_error",
                            "pre-gripper up/forward IK failed",
                        )
                        pre_gripper_forward_rejected_count += 1
                if path_collision_free:
                    feasible_solutions.append((motion_solution, record, start_joint_positions_rad))
                else:
                    path_collision_rejected_count += 1

        if feasible_solutions:
            selected_solution, selected_record, selected_start_joint_positions_rad = min(
                feasible_solutions,
                key=lambda item: _feasible_ik_sort_key(item[0]),
            )
            selected_record["selected_as_best"] = True
            selected_solution["selected_as_best"] = True
            selected_solution["selected_feasible_count"] = int(len(feasible_solutions))
            selected_solution["selected_by"] = "min_ee_roll_pitch_error_abs_max_deg"
            selected_solution["interpolated_path_checked_count"] = int(path_checked_count)
            selected_solution["interpolated_path_collision_rejected_count"] = int(path_collision_rejected_count)
            selected_solution["pre_gripper_forward_rejected_count"] = int(pre_gripper_forward_rejected_count)
        else:
            selected_start_joint_positions_rad = None

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
            best_effort_message = (
                "No IK solution satisfied the pose tolerances. Returning the closest "
                "collision-free IK result without moving the arm."
            )
            if path_collision_rejected_count > 0:
                best_effort_message = (
                    "All pose-feasible IK solutions collided along the interpolated "
                    "move_arm waypoints. Returning the closest collision-free IK "
                    "result without moving the arm."
                )
            logger.warning(
                "No feasible IK solution found; returning best collision-free IK solution "
                "with ee_position_error_m=%s.",
                best_effort_solution.get("ee_position_error_m"),
            )
            return {
                "success": False,
                "status_code": "ARM_APPROACH_BEST_EFFORT_IK",
                "phase": "ik_evaluation",
                "message": best_effort_message,
                "selected_solution": best_effort_solution,
                "best_effort_solution": best_effort_solution,
                "ik_solution_rad": best_effort_solution["ik_joint_solution_rad"],
                "attempted_grasp_count": attempted_count,
                "ranked_grasp_count": len(ranked_records),
                "evaluated_grasp_count": attempted_count,
                "max_target_grasp_poses": max_target_grasp_poses,
                "interpolated_path_checked_count": path_checked_count,
                "interpolated_path_collision_rejected_count": path_collision_rejected_count,
                "pre_gripper_forward_rejected_count": pre_gripper_forward_rejected_count,
                "pointcloud_camera_x_mirrored": bool(live_scene.pointcloud_camera_x_mirrored),
                "arm_approach_start_base_joint_override": start_base_joint_override or {},
                "initialpose_published": False,
                "next_agent": None,
                "exec_latency": time.time() - started_at,
            }
        logger.error("No feasible IK solution found for arm approach from current pose.")
        no_ik_message = "No feasible or collision-free best-effort IK solution for the arm at the current base pose."
        if path_collision_rejected_count > 0:
            no_ik_message = "All pose-feasible IK solutions collided along the interpolated move_arm waypoints."
        if pre_gripper_forward_rejected_count > 0:
            no_ik_message = "All pose-feasible IK solutions failed the pre-gripper EE up/forward sequence or its path check."
        return {
            "success": False,
            "status_code": "ARM_APPROACH_NO_IK",
            "phase": "ik_evaluation",
            "message": no_ik_message,
            "attempted_grasp_count": attempted_count,
            "ranked_grasp_count": len(ranked_records),
            "evaluated_grasp_count": attempted_count,
            "max_target_grasp_poses": max_target_grasp_poses,
            "interpolated_path_checked_count": path_checked_count,
            "interpolated_path_collision_rejected_count": path_collision_rejected_count,
            "pre_gripper_forward_rejected_count": pre_gripper_forward_rejected_count,
            "pointcloud_camera_x_mirrored": bool(live_scene.pointcloud_camera_x_mirrored),
            "arm_approach_start_base_joint_override": start_base_joint_override or {},
            "initialpose_published": False,
            "next_agent": None,
            "exec_latency": time.time() - started_at,
        }

    if selected_start_joint_positions_rad is None:
        selected_solution, selected_start_joint_positions_rad = _with_open_gripper_before_motion(
            selected_solution,
            planning_config,
        )
        selected_start_joint_positions_rad, start_base_metadata = _apply_arm_start_base_joint_override(
            selected_start_joint_positions_rad,
            start_base_joint_override,
        )
        selected_solution.update(start_base_metadata)
    # Extract joint rads and trigger move_arm
    joint_rads = selected_solution["ik_joint_solution_rad"]
    logger.info(f"Found feasible IK. Moving arm to joints: {joint_rads}")
    
    # move_arm_for_solution handles calling publisher and returns publish metadata.
    arm_result = move_arm_for_solution(
        selected_solution,
        planning_config=planning_config,
        config=arm_move_config,
        planner_config_path=Path(cfg["planner_config_path"]),
        start_joint_positions_rad=selected_start_joint_positions_rad,
    )
    side_view_result = _capture_target_arrival_side_view(
        p_mod=p_mod,
        pybullet_data=pybullet_data,
        planning_config=planning_config,
        arm_config=arm_config,
        live_scene=live_scene,
        selected_solution=selected_solution,
    )
    arm_result.update(side_view_result)
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
        "evaluated_grasp_count": attempted_count,
        "max_target_grasp_poses": max_target_grasp_poses,
        "selected_feasible_count": int(selected_solution.get("selected_feasible_count", 0)),
        "interpolated_path_checked_count": path_checked_count,
        "interpolated_path_collision_rejected_count": path_collision_rejected_count,
        "pre_gripper_forward_rejected_count": pre_gripper_forward_rejected_count,
        "pointcloud_camera_x_mirrored": bool(live_scene.pointcloud_camera_x_mirrored),
        "arm_approach_start_base_joint_override": start_base_joint_override or {},
        "arm_result": arm_result,
        "message": "Arm approach finished.",
        "initialpose_published": False,
        "next_agent": None,
        "exec_latency": time.time() - started_at,
    }
