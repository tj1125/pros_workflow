"""Finish the grasp directly from car_approach after the base arrives."""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from agents.arm_approach.move_arm import (
    config_from_environment as arm_move_config_from_environment,
    move_arm_for_solution,
)

from .src.pybullet_ompl import (
    _find_controllable_joints,
    _set_joint_positions_direct,
)


CAR_ARM_FINISH_ENABLED = True
CAR_ARM_FINISH_NODE_NAME = "approach_agent_car_arm_finish"
CAR_ARM_FINISH_GRIPPER_JOINT_INDEX = 4
CAR_ARM_FINISH_GRIPPER_OPEN_DEG = 70.0
CAR_ARM_FINISH_GRIPPER_CLOSE_DEG = 10.0
CAR_ARM_FINISH_WRIST_JOINT_INDEX = 3
CAR_ARM_FINISH_TARGET_GRASP_YAW_REFERENCE_PERIOD_DEG = 180.0
CAR_ARM_FINISH_TARGET_GRASP_YAW_TO_WRIST_SIGN = 1.0
CAR_ARM_FINISH_WRIST_YAW_MIN_DEG = 60.0
CAR_ARM_FINISH_WRIST_YAW_MAX_DEG = 120.0
CAR_ARM_FINISH_PRE_CLOSE_EE_OFFSET_ENABLED = True
CAR_ARM_FINISH_PRE_CLOSE_FORWARD_DISTANCE_M = 0.07
CAR_ARM_FINISH_PRE_CLOSE_DOWN_DISTANCE_M = 0


def env_flag(name: str, default: bool) -> bool:
    default_value = "1" if default else "0"
    return os.getenv(name, default_value).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }


def car_arm_finish_enabled() -> bool:
    return env_flag("APPROACH_AGENT_CAR_FINISH_ARM_ON_ARRIVAL", CAR_ARM_FINISH_ENABLED)


def run_car_arm_finish_sequence(
    solution: dict[str, object],
    *,
    visualization_records: Sequence[dict[str, object]],
    planning_config: Any,
    arm_config: dict[str, object] | None,
    arm_base_target: dict[str, object] | None = None,
    planner_config_path: Path | None = None,
    p_mod: Any | None = None,
    pybullet_data: Any | None = None,
) -> dict[str, object]:
    """Open the gripper, move the arm to the target pose, then close the gripper."""

    if not car_arm_finish_enabled():
        return {
            "success": False,
            "skipped": True,
            "phase": "disabled",
            "message": "car_approach direct arm finish is disabled.",
        }

    try:
        target_record = _selected_visualization_record(
            solution,
            visualization_records,
        )
        start_joint_positions_rad = _joint_reset_rad(planning_config)
        motion_solution, metadata = _build_motion_solution(
            solution,
            target_record=target_record,
            planning_config=planning_config,
            arm_config=arm_config,
            arm_base_target=arm_base_target,
        )
        pre_close_result = _with_pre_close_ee_down_then_forward(
            motion_solution,
            p_mod=p_mod,
            pybullet_data=pybullet_data,
            planning_config=planning_config,
            arm_config=arm_config,
            enabled=env_flag(
                "APPROACH_AGENT_CAR_PRE_CLOSE_EE_OFFSET_ENABLED",
                CAR_ARM_FINISH_PRE_CLOSE_EE_OFFSET_ENABLED,
            ),
        )
        motion_solution = pre_close_result["solution"]
        metadata.update(pre_close_result["metadata"])
        gripper_index = int(metadata["preopened_gripper_joint_index"])
        gripper_open_rad = float(metadata["preopened_gripper_target_rad"])
        if 0 <= gripper_index < len(start_joint_positions_rad):
            start_joint_positions_rad[gripper_index] = gripper_open_rad
        locked_arm_base_joint_index = _optional_int(metadata.get("locked_arm_base_joint_index"))
        locked_arm_base_joint_rad = _optional_float(metadata.get("locked_arm_base_joint_rad"))
        if (
            locked_arm_base_joint_index is not None
            and locked_arm_base_joint_rad is not None
            and 0 <= locked_arm_base_joint_index < len(start_joint_positions_rad)
        ):
            start_joint_positions_rad[locked_arm_base_joint_index] = locked_arm_base_joint_rad

        base_config = arm_move_config_from_environment(planning_config=planning_config)
        move_config = replace(
            base_config,
            node_name=CAR_ARM_FINISH_NODE_NAME,
            publish_start=True,
            close_gripper_on_arrival=True,
            gripper_joint_index=gripper_index,
            gripper_close_rad=math.radians(
                float(
                    os.getenv(
                        "APPROACH_AGENT_CAR_GRIPPER_CLOSE_DEG",
                        str(CAR_ARM_FINISH_GRIPPER_CLOSE_DEG),
                    )
                )
            ),
            forward_before_gripper_close=bool(metadata.get("pre_close_ee_offset_enabled", False)),
            return_to_start_after_gripper_close=False,
        )
        publish_result = move_arm_for_solution(
            motion_solution,
            planning_config=planning_config,
            config=move_config,
            planner_config_path=planner_config_path,
            start_joint_positions_rad=start_joint_positions_rad,
        )
    except Exception as exc:
        return {
            "success": False,
            "skipped": False,
            "phase": "publish_failed",
            "message": str(exc),
        }

    result: dict[str, object] = {
        **metadata,
        **publish_result,
        "success": bool(publish_result.get("success", False)),
        "skipped": False,
        "phase": "published" if bool(publish_result.get("success", False)) else "publish_failed",
        "source": "approach_agent_car_approach_direct_arm_finish",
        "arm_motion_skipped": False,
        "target_pose_source": "selected car_approach IK solution",
        "execution_sequence": [
            "open_gripper",
            "lock_arm_base_joint",
            "move_arm_joints_to_target_pose",
            "move_ee_down",
            "move_ee_forward",
            "close_gripper",
        ],
    }
    if bool(result["success"]):
        print(
            "[base_approach] car_approach direct arm finish completed: "
            f"open_finger={float(metadata['preopened_gripper_target_deg']):.2f}deg "
            f"close_finger={float(result.get('gripper_close_deg', CAR_ARM_FINISH_GRIPPER_CLOSE_DEG)):.2f}deg "
            f"wrist={float(metadata.get('target_grasp_wrist_target_deg', float('nan'))):.2f}deg",
            flush=True,
        )
    return result


def _with_pre_close_ee_down_then_forward(
    solution: dict[str, object],
    *,
    p_mod: Any | None,
    pybullet_data: Any | None,
    planning_config: Any,
    arm_config: dict[str, object] | None,
    enabled: bool,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "pre_close_ee_offset_enabled": bool(enabled),
        "pre_close_ee_offset_sequence": ["down", "forward"],
        "pre_close_ee_forward_distance_m": float(
            os.getenv(
                "APPROACH_AGENT_CAR_PRE_CLOSE_FORWARD_DISTANCE_M",
                str(CAR_ARM_FINISH_PRE_CLOSE_FORWARD_DISTANCE_M),
            )
        ),
        "pre_close_ee_down_distance_m": float(
            os.getenv(
                "APPROACH_AGENT_CAR_PRE_CLOSE_DOWN_DISTANCE_M",
                os.getenv(
                    "APPROACH_AGENT_CAR_PRE_CLOSE_UP_DISTANCE_M",
                    str(CAR_ARM_FINISH_PRE_CLOSE_DOWN_DISTANCE_M),
                ),
            )
        ),
    }
    if not enabled:
        metadata["pre_close_ee_offset_skipped"] = True
        return {"solution": solution, "metadata": metadata}
    if p_mod is None or pybullet_data is None:
        raise ValueError("PyBullet modules are required for pre-close EE down/forward IK.")
    if arm_config is None:
        raise ValueError("arm_config is required for pre-close EE down/forward IK.")

    goal_joint_positions = _goal_joint_rad_from_solution(solution)
    forward_distance_m = float(metadata["pre_close_ee_forward_distance_m"])
    down_distance_m = float(metadata["pre_close_ee_down_distance_m"])
    gripper_index = int(solution.get("preopened_gripper_joint_index", CAR_ARM_FINISH_GRIPPER_JOINT_INDEX))
    locked_arm_base_joint_index = _optional_int(solution.get("locked_arm_base_joint_index"))
    locked_arm_base_joint_rad = _optional_float(solution.get("locked_arm_base_joint_rad"))
    if (
        locked_arm_base_joint_index is not None
        and locked_arm_base_joint_rad is not None
        and 0 <= locked_arm_base_joint_index < len(goal_joint_positions)
    ):
        goal_joint_positions[locked_arm_base_joint_index] = locked_arm_base_joint_rad
        metadata["pre_close_ee_arm_base_joint_locked"] = True
        metadata["pre_close_ee_locked_arm_base_joint_index"] = int(locked_arm_base_joint_index)
        metadata["pre_close_ee_locked_arm_base_joint_rad"] = float(locked_arm_base_joint_rad)
        metadata["pre_close_ee_locked_arm_base_joint_deg"] = math.degrees(float(locked_arm_base_joint_rad))
    else:
        metadata["pre_close_ee_arm_base_joint_locked"] = False

    client_id = p_mod.connect(p_mod.DIRECT)
    if client_id < 0:
        raise RuntimeError("PyBullet DIRECT unavailable for pre-close EE down/forward IK.")

    try:
        p_mod.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_mod.resetSimulation()
        p_mod.setGravity(0.0, 0.0, -9.8)
        p_mod.loadURDF("plane.urdf")

        base_orientation_rad = [math.radians(float(value)) for value in planning_config.base_orientation_euler_deg]
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
                "Controllable joint count mismatch during pre-close EE IK: "
                f"expected={expected_joint_count} actual={len(controllable_joint_ids)} "
                f"names={controllable_joint_names}"
            )
        if len(goal_joint_positions) != len(controllable_joint_ids):
            raise ValueError(
                "Goal joint vector length does not match controllable joints during pre-close EE IK: "
                f"{len(goal_joint_positions)} != {len(controllable_joint_ids)}"
            )

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

        down_target_position = ee_position + np.asarray([0.0, 0.0, -down_distance_m], dtype=np.float64)
        down_joint_positions, down_ee_position, down_error_m = _solve_pre_close_target_position(
            p_mod,
            robot_id=robot_id,
            controllable_joint_ids=controllable_joint_ids,
            planning_config=planning_config,
            arm_config=arm_config,
            target_position=down_target_position,
            target_orientation=ee_orientation,
            seed_joint_positions=goal_joint_positions,
            gripper_index=gripper_index,
            gripper_position_rad=goal_joint_positions[gripper_index] if 0 <= gripper_index < len(goal_joint_positions) else None,
            locked_joint_index=locked_arm_base_joint_index,
            locked_joint_position_rad=locked_arm_base_joint_rad,
            stage_name="down",
        )

        forward_target_position = down_target_position + (forward_axis * forward_distance_m)
        forward_joint_positions, forward_ee_position, forward_error_m = _solve_pre_close_target_position(
            p_mod,
            robot_id=robot_id,
            controllable_joint_ids=controllable_joint_ids,
            planning_config=planning_config,
            arm_config=arm_config,
            target_position=forward_target_position,
            target_orientation=ee_orientation,
            seed_joint_positions=down_joint_positions,
            gripper_index=gripper_index,
            gripper_position_rad=goal_joint_positions[gripper_index] if 0 <= gripper_index < len(goal_joint_positions) else None,
            locked_joint_index=locked_arm_base_joint_index,
            locked_joint_position_rad=locked_arm_base_joint_rad,
            stage_name="forward",
        )
    finally:
        try:
            p_mod.disconnect(client_id)
        except Exception:
            pass

    adjusted_solution = dict(solution)
    adjusted_solution["post_arrival_joint_positions_sequence_rad"] = [
        [float(value) for value in down_joint_positions],
        [float(value) for value in forward_joint_positions],
    ]
    adjusted_solution["post_arrival_joint_positions_sequence_deg"] = [
        [math.degrees(float(value)) for value in down_joint_positions],
        [math.degrees(float(value)) for value in forward_joint_positions],
    ]
    adjusted_solution["post_arrival_joint_positions_sequence_labels"] = ["down", "forward"]

    metadata.update(
        {
            "pre_close_ee_offset_skipped": False,
            "pre_close_ee_offset_ik_success": True,
            "pre_close_ee_start_position_xyz": ee_position.astype(float).tolist(),
            "pre_close_ee_forward_axis_xyz": forward_axis.astype(float).tolist(),
            "pre_close_ee_down_target_position_xyz": down_target_position.astype(float).tolist(),
            "pre_close_ee_down_achieved_position_xyz": down_ee_position.astype(float).tolist(),
            "pre_close_ee_down_target_error_m": float(down_error_m),
            "pre_close_ee_forward_target_position_xyz": forward_target_position.astype(float).tolist(),
            "pre_close_ee_forward_achieved_position_xyz": forward_ee_position.astype(float).tolist(),
            "pre_close_ee_forward_target_error_m": float(forward_error_m),
            "pre_close_ee_joint_positions_sequence_rad": adjusted_solution["post_arrival_joint_positions_sequence_rad"],
            "pre_close_ee_joint_positions_sequence_deg": adjusted_solution["post_arrival_joint_positions_sequence_deg"],
            "pre_close_ee_joint_positions_sequence_labels": ["down", "forward"],
        }
    )
    return {"solution": adjusted_solution, "metadata": metadata}


def _solve_pre_close_target_position(
    p_mod: Any,
    *,
    robot_id: int,
    controllable_joint_ids: list[int],
    planning_config: Any,
    arm_config: dict[str, object],
    target_position: np.ndarray,
    target_orientation: np.ndarray,
    seed_joint_positions: Sequence[float],
    gripper_index: int,
    gripper_position_rad: float | None,
    locked_joint_index: int | None,
    locked_joint_position_rad: float | None,
    stage_name: str,
) -> tuple[list[float], np.ndarray, float]:
    seed = [float(value) for value in seed_joint_positions]
    if (
        locked_joint_index is not None
        and locked_joint_position_rad is not None
        and 0 <= int(locked_joint_index) < len(seed)
    ):
        seed[int(locked_joint_index)] = float(locked_joint_position_rad)
    _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, seed)
    raw_solution = p_mod.calculateInverseKinematics(
        robot_id,
        int(planning_config.ee_link_index),
        targetPosition=np.asarray(target_position, dtype=np.float64).reshape(3).astype(float).tolist(),
        targetOrientation=np.asarray(target_orientation, dtype=np.float64).reshape(4).astype(float).tolist(),
    )
    joint_positions = [float(value) for value in raw_solution[: len(controllable_joint_ids)]]
    if len(joint_positions) != len(controllable_joint_ids):
        raise RuntimeError(
            f"{stage_name} IK returned unexpected joint count: "
            f"{len(joint_positions)} != {len(controllable_joint_ids)}"
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
    if (
        locked_joint_index is not None
        and locked_joint_position_rad is not None
        and 0 <= int(locked_joint_index) < len(joint_positions)
    ):
        joint_positions[int(locked_joint_index)] = float(locked_joint_position_rad)
    if gripper_position_rad is not None and 0 <= int(gripper_index) < len(joint_positions):
        joint_positions[int(gripper_index)] = float(gripper_position_rad)

    _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, joint_positions)
    p_mod.performCollisionDetection()
    ee_state = p_mod.getLinkState(
        robot_id,
        int(planning_config.ee_link_index),
        computeForwardKinematics=True,
    )
    achieved_position = np.asarray(ee_state[4], dtype=np.float64).reshape(3)
    target_error_m = float(np.linalg.norm(achieved_position - np.asarray(target_position, dtype=np.float64).reshape(3)))
    return joint_positions, achieved_position, target_error_m


def _joint_reset_rad(planning_config: Any) -> list[float]:
    joint_reset_deg = getattr(planning_config, "joint_reset_deg", None)
    if joint_reset_deg is None and isinstance(planning_config, dict):
        joint_reset_deg = planning_config.get("joint_reset_deg")
    if joint_reset_deg is None:
        raise ValueError("planning_config.joint_reset_deg is required.")
    return [math.radians(float(value)) for value in joint_reset_deg]


def _goal_joint_rad_from_solution(solution: dict[str, object]) -> list[float]:
    raw_rad = solution.get("ik_joint_solution_rad")
    if isinstance(raw_rad, list):
        return _float_sequence(raw_rad, label="solution.ik_joint_solution_rad")
    raw_deg = solution.get("ik_joint_solution_deg")
    if isinstance(raw_deg, list):
        return [math.radians(value) for value in _float_sequence(raw_deg, label="solution.ik_joint_solution_deg")]
    raise KeyError("solution needs ik_joint_solution_rad or ik_joint_solution_deg.")


def _float_sequence(values: Sequence[Any], *, label: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{label} must not be empty.")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain only finite values.")
    return result


def _build_motion_solution(
    solution: dict[str, object],
    *,
    target_record: dict[str, object] | None,
    planning_config: Any,
    arm_config: dict[str, object] | None,
    arm_base_target: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    goal_joint_positions = _goal_joint_rad_from_solution(solution)
    metadata: dict[str, object] = {
        "raw_goal_joint_positions_rad": [float(value) for value in goal_joint_positions],
        "raw_goal_joint_positions_deg": [math.degrees(float(value)) for value in goal_joint_positions],
    }

    if isinstance(arm_base_target, dict) and arm_base_target.get("command_joint_position_rad") is not None:
        joint_index = int(arm_base_target.get("joint_index", 0))
        if 0 <= joint_index < len(goal_joint_positions):
            locked_joint_rad = float(arm_base_target["command_joint_position_rad"])
            goal_joint_positions[joint_index] = locked_joint_rad
            metadata["arm_base_target_applied"] = True
            metadata["arm_base_target"] = dict(arm_base_target)
            metadata["locked_arm_base_joint"] = True
            metadata["locked_arm_base_joint_index"] = int(joint_index)
            metadata["locked_arm_base_joint_rad"] = float(locked_joint_rad)
            metadata["locked_arm_base_joint_deg"] = math.degrees(float(locked_joint_rad))
            metadata["locked_arm_base_joint_source"] = "arm_base_target.command_joint_position_rad"
        else:
            metadata["arm_base_target_applied"] = False
            metadata["arm_base_target_error"] = (
                f"joint index {joint_index} is outside goal vector length {len(goal_joint_positions)}"
            )
    else:
        metadata["arm_base_target_applied"] = False
        if isinstance(arm_base_target, dict):
            metadata["arm_base_target"] = dict(arm_base_target)

    if not bool(metadata.get("locked_arm_base_joint", False)):
        fallback_joint_index = int(os.getenv("APPROACH_AGENT_CAR_ARM_BASE_JOINT_INDEX", "0"))
        if 0 <= fallback_joint_index < len(goal_joint_positions):
            locked_joint_rad = float(goal_joint_positions[fallback_joint_index])
            metadata["locked_arm_base_joint"] = True
            metadata["locked_arm_base_joint_index"] = int(fallback_joint_index)
            metadata["locked_arm_base_joint_rad"] = float(locked_joint_rad)
            metadata["locked_arm_base_joint_deg"] = math.degrees(float(locked_joint_rad))
            metadata["locked_arm_base_joint_source"] = "selected_solution.ik_joint_solution_rad"
        else:
            raise ValueError(
                f"Cannot lock arm base joint {fallback_joint_index}; "
                f"goal vector length is {len(goal_joint_positions)}."
            )

    goal_joint_positions, wrist_metadata = _apply_wrist_from_target_grasp_yaw(
        goal_joint_positions,
        solution=solution,
        target_record=target_record,
        planning_config=planning_config,
        arm_config=arm_config,
    )
    metadata.update(wrist_metadata)

    gripper_index = int(
        os.getenv(
            "APPROACH_AGENT_CAR_GRIPPER_JOINT_INDEX",
            str(CAR_ARM_FINISH_GRIPPER_JOINT_INDEX),
        )
    )
    gripper_open_deg = float(
        os.getenv(
            "APPROACH_AGENT_CAR_GRIPPER_OPEN_DEG",
            str(CAR_ARM_FINISH_GRIPPER_OPEN_DEG),
        )
    )
    gripper_open_rad = math.radians(gripper_open_deg)
    gripper_lower_deg, gripper_upper_deg = _joint_limit_deg_from_configs(
        joint_index=gripper_index,
        planning_config=planning_config,
        arm_config=arm_config,
    )
    if 0 <= gripper_index < len(goal_joint_positions):
        goal_joint_positions[gripper_index] = gripper_open_rad
        metadata["preopened_gripper_applied"] = True
        metadata["preopened_gripper_limit_min_deg"] = gripper_lower_deg
        metadata["preopened_gripper_limit_max_deg"] = gripper_upper_deg
        if gripper_upper_deg is not None and gripper_open_deg > float(gripper_upper_deg):
            metadata["preopened_gripper_limit_warning"] = (
                f"requested finger open angle {gripper_open_deg:.2f}deg exceeds "
                f"configured upper limit {float(gripper_upper_deg):.2f}deg; "
                "tools/car_control may clamp the command."
            )
    else:
        metadata["preopened_gripper_applied"] = False
        metadata["preopened_gripper_error"] = (
            f"gripper joint index {gripper_index} is outside goal vector length {len(goal_joint_positions)}"
        )

    motion_solution = dict(solution)
    motion_solution["ik_joint_solution_rad"] = [float(value) for value in goal_joint_positions]
    motion_solution["ik_joint_solution_deg"] = [math.degrees(float(value)) for value in goal_joint_positions]
    motion_solution["car_approach_direct_arm_finish"] = True
    motion_solution["locked_arm_base_joint"] = bool(metadata.get("locked_arm_base_joint", False))
    motion_solution["locked_arm_base_joint_index"] = metadata.get("locked_arm_base_joint_index")
    motion_solution["locked_arm_base_joint_rad"] = metadata.get("locked_arm_base_joint_rad")
    motion_solution["locked_arm_base_joint_deg"] = metadata.get("locked_arm_base_joint_deg")
    motion_solution["locked_arm_base_joint_source"] = metadata.get("locked_arm_base_joint_source")
    motion_solution["preopened_gripper_joint_index"] = gripper_index
    motion_solution["preopened_gripper_target_deg"] = gripper_open_deg
    motion_solution["preopened_gripper_target_rad"] = gripper_open_rad

    metadata.update(
        {
            "target_joint_positions_rad": [float(value) for value in goal_joint_positions],
            "target_joint_positions_deg": [math.degrees(float(value)) for value in goal_joint_positions],
            "preopened_gripper_joint_index": gripper_index,
            "preopened_gripper_target_deg": gripper_open_deg,
            "preopened_gripper_target_rad": gripper_open_rad,
        }
    )
    return motion_solution, metadata


def _selected_visualization_record(
    solution: dict[str, object],
    visualization_records: Sequence[dict[str, object]],
) -> dict[str, object] | None:
    target_order = _optional_int(solution.get("target_sample_order"))
    if target_order is not None:
        for record in visualization_records:
            if _optional_int(record.get("target_sample_order")) == target_order:
                return dict(record)

    selected_rank = _optional_int(solution.get("grasp_rank"))
    if selected_rank is not None:
        for record in visualization_records:
            if _optional_int(record.get("rank")) == selected_rank:
                return dict(record)

    for record in visualization_records:
        if bool(record.get("selected_as_best", False)):
            return dict(record)
    return None


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _apply_wrist_from_target_grasp_yaw(
    goal_joint_positions: list[float],
    *,
    solution: dict[str, object],
    target_record: dict[str, object] | None,
    planning_config: Any,
    arm_config: dict[str, object] | None,
) -> tuple[list[float], dict[str, object]]:
    wrist_index = int(CAR_ARM_FINISH_WRIST_JOINT_INDEX)
    adjusted_positions = [float(value) for value in goal_joint_positions]
    if not (0 <= wrist_index < len(adjusted_positions)):
        return adjusted_positions, {
            "target_grasp_wrist_yaw_applied": False,
            "target_grasp_wrist_yaw_error": (
                f"wrist joint index {wrist_index} is outside joint vector length {len(adjusted_positions)}"
            ),
        }

    try:
        wrist_reset_deg = float(planning_config.joint_reset_deg[wrist_index])
    except (AttributeError, IndexError, TypeError, ValueError):
        wrist_reset_deg = 90.0

    target_yaw_rad, target_yaw_source = _target_grasp_yaw_rad(solution, target_record)
    if target_yaw_rad is None:
        return adjusted_positions, {
            "target_grasp_wrist_yaw_applied": False,
            "target_grasp_wrist_yaw_error": "target grasp yaw is unavailable.",
        }

    target_yaw_normalized_deg = float(math.degrees(target_yaw_rad) % 360.0)
    yaw_reference_rad, yaw_delta_from_reference_rad = _nearest_half_turn_reference_and_delta_rad(target_yaw_rad)
    yaw_delta_from_reference_deg = math.degrees(yaw_delta_from_reference_rad)
    wrist_unclamped_deg = wrist_reset_deg + (
        float(CAR_ARM_FINISH_TARGET_GRASP_YAW_TO_WRIST_SIGN) * yaw_delta_from_reference_deg
    )
    lower_deg, upper_deg = _joint_limit_deg_from_configs(
        joint_index=wrist_index,
        planning_config=planning_config,
        arm_config=arm_config,
    )
    wrist_yaw_min_deg = float(
        os.getenv(
            "APPROACH_AGENT_ARM_WRIST_YAW_MIN_DEG",
            str(CAR_ARM_FINISH_WRIST_YAW_MIN_DEG),
        )
    )
    wrist_yaw_max_deg = float(
        os.getenv(
            "APPROACH_AGENT_ARM_WRIST_YAW_MAX_DEG",
            str(CAR_ARM_FINISH_WRIST_YAW_MAX_DEG),
        )
    )
    if wrist_yaw_min_deg > wrist_yaw_max_deg:
        wrist_yaw_min_deg, wrist_yaw_max_deg = wrist_yaw_max_deg, wrist_yaw_min_deg

    command_lower_deg = wrist_yaw_min_deg if lower_deg is None else max(float(lower_deg), wrist_yaw_min_deg)
    command_upper_deg = wrist_yaw_max_deg if upper_deg is None else min(float(upper_deg), wrist_yaw_max_deg)
    wrist_target_deg = _clamp_deg(wrist_unclamped_deg, command_lower_deg, command_upper_deg)
    adjusted_positions[wrist_index] = math.radians(wrist_target_deg)

    return adjusted_positions, {
        "target_grasp_wrist_yaw_applied": True,
        "target_grasp_wrist_yaw_source": target_yaw_source,
        "target_grasp_wrist_joint_index": wrist_index,
        "target_grasp_yaw_reference_period_deg": float(CAR_ARM_FINISH_TARGET_GRASP_YAW_REFERENCE_PERIOD_DEG),
        "target_grasp_yaw_nearest_reference_rad": float(yaw_reference_rad),
        "target_grasp_yaw_nearest_reference_deg": float(math.degrees(yaw_reference_rad) % 360.0),
        "target_grasp_yaw_rad": float(target_yaw_rad),
        "target_grasp_yaw_deg": float(math.degrees(target_yaw_rad)),
        "target_grasp_yaw_normalized_deg": target_yaw_normalized_deg,
        "target_grasp_yaw_delta_from_nearest_reference_rad": float(yaw_delta_from_reference_rad),
        "target_grasp_yaw_delta_from_nearest_reference_deg": float(yaw_delta_from_reference_deg),
        "target_grasp_wrist_reset_deg": float(wrist_reset_deg),
        "target_grasp_wrist_sign": float(CAR_ARM_FINISH_TARGET_GRASP_YAW_TO_WRIST_SIGN),
        "target_grasp_wrist_unclamped_deg": float(wrist_unclamped_deg),
        "target_grasp_wrist_target_deg": float(wrist_target_deg),
        "target_grasp_wrist_target_rad": float(math.radians(wrist_target_deg)),
        "target_grasp_wrist_limit_min_deg": lower_deg,
        "target_grasp_wrist_limit_max_deg": upper_deg,
        "target_grasp_wrist_yaw_min_deg": float(wrist_yaw_min_deg),
        "target_grasp_wrist_yaw_max_deg": float(wrist_yaw_max_deg),
        "target_grasp_wrist_command_min_deg": float(command_lower_deg),
        "target_grasp_wrist_command_max_deg": float(command_upper_deg),
        "target_grasp_wrist_clamped": not math.isclose(
            float(wrist_target_deg),
            float(wrist_unclamped_deg),
            rel_tol=0.0,
            abs_tol=1e-9,
        ),
    }


def _target_grasp_yaw_rad(
    solution: dict[str, object],
    target_record: dict[str, object] | None,
) -> tuple[float | None, str]:
    if isinstance(target_record, dict):
        raw_rot = target_record.get("target_rot_pb")
        if raw_rot is not None:
            try:
                target_rot_pb = np.asarray(raw_rot, dtype=np.float64).reshape(3, 3)
                return _axis_yaw_xy(target_rot_pb[:, 0], fallback_yaw_rad=0.0), "visualization_record.target_rot_pb[:,0]"
            except Exception:
                pass

    for key in (
        "target_grasp_pose_yaw_rad",
        "target_yaw_rad",
        "desired_pb_base_link_yaw_rad",
    ):
        raw_yaw = solution.get(key)
        if raw_yaw is not None:
            try:
                return _wrap_angle_rad(float(raw_yaw)), f"solution.{key}"
            except (TypeError, ValueError):
                pass

    raw_direction = solution.get("backoff_direction_pb_xy")
    if isinstance(raw_direction, list) and len(raw_direction) >= 2:
        try:
            return _axis_yaw_xy(
                np.asarray([float(raw_direction[0]), float(raw_direction[1]), 0.0], dtype=np.float64),
                fallback_yaw_rad=0.0,
            ), "solution.backoff_direction_pb_xy"
        except (TypeError, ValueError):
            pass

    return None, ""


def _axis_yaw_xy(axis_xyz: np.ndarray, fallback_yaw_rad: float = 0.0) -> float:
    axis = np.asarray(axis_xyz, dtype=np.float64).reshape(3)
    axis_xy = axis[:2]
    axis_norm = float(np.linalg.norm(axis_xy))
    if axis_norm <= 1e-6:
        return float(fallback_yaw_rad)
    return float(math.atan2(float(axis_xy[1]), float(axis_xy[0])))


def _nearest_half_turn_reference_and_delta_deg(angle_rad: float) -> tuple[float, float]:
    angle_deg = float(math.degrees(float(angle_rad)) % 360.0)
    reference_candidates_deg = (0.0, 180.0, 360.0)

    def _sort_key(reference_deg: float) -> tuple[float, int, float]:
        delta_deg = angle_deg - float(reference_deg)
        return (abs(delta_deg), 1 if delta_deg < 0.0 else 0, float(reference_deg))

    reference_deg = min(reference_candidates_deg, key=_sort_key)
    return float(reference_deg), float(angle_deg - reference_deg)


def _nearest_half_turn_reference_and_delta_rad(angle_rad: float) -> tuple[float, float]:
    reference_deg, delta_deg = _nearest_half_turn_reference_and_delta_deg(angle_rad)
    return math.radians(reference_deg), math.radians(delta_deg)


def _wrap_angle_rad(angle_rad: float) -> float:
    return float((float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi)


def _joint_limit_deg_from_configs(
    *,
    joint_index: int,
    planning_config: Any,
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
