"""PyBullet IK helpers for arrival-time arm motion."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .debug_log import debug_stage
from .joint_sequence import _round_float_sequence
from .src.geometry import coordinate_transforms as coord


def recompute_arrival_ik(target_base_xyz: Sequence[Any], *, config: Any) -> dict[str, object]:
    import pybullet as p
    import pybullet_data

    target = _float_xyz(target_base_xyz, label="target_base_xyz")
    debug_stage("move_arm", "arrival IK：啟動 PyBullet DIRECT", target_base_xyz=target)
    client_id = p.connect(p.DIRECT)
    if client_id < 0:
        raise RuntimeError("PyBullet DIRECT connection failed for arrival IK.")
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.8)
        p.loadURDF("plane.urdf")
        for search_path in _urdf_search_paths(config.urdf_path):
            p.setAdditionalSearchPath(str(search_path))
        robot_id = p.loadURDF(
            str(config.urdf_path),
            useFixedBase=True,
            basePosition=[0.0, 0.0, float(config.base_height_m)],
            baseOrientation=p.getQuaternionFromEuler(coord.deg_sequence_to_rad(config.base_orientation_euler_deg)),
        )
        joint_ids = _controllable_joint_ids(p, robot_id, config.controllable_joints)
        debug_stage("move_arm", "arrival IK：URDF 已載入，準備 calculateInverseKinematics", joint_count=len(joint_ids))
        if len(joint_ids) != int(config.controllable_joints):
            raise RuntimeError(f"Expected {config.controllable_joints} controllable joints, found {len(joint_ids)}.")

        reset_rad = coord.deg_sequence_to_rad(config.joint_reset_deg[: len(joint_ids)])
        _set_joint_positions(p, robot_id, joint_ids, reset_rad)
        ik = p.calculateInverseKinematics(
            robot_id,
            int(config.ee_link_index),
            targetPosition=target,
            maxNumIterations=int(config.ik_max_iterations),
            residualThreshold=float(config.ik_residual_threshold),
        )
        joint_solution = _clip_to_bounds([float(v) for v in ik[: len(joint_ids)]], config.joint_bounds_deg)
        _set_joint_positions(p, robot_id, joint_ids, joint_solution)
        p.performCollisionDetection()

        ee_state = p.getLinkState(robot_id, int(config.ee_link_index), computeForwardKinematics=True)
        final_ee_xyz = [float(v) for v in ee_state[4]]
        error_xyz = [float(t - a) for t, a in zip(target, final_ee_xyz)]
        position_error_m = float(np.linalg.norm(np.asarray(error_xyz, dtype=np.float64)))
        joint_solution_deg = coord.rad_sequence_to_deg(joint_solution)
        debug_stage(
            "move_arm",
            "arrival IK：計算完成",
            feasible=position_error_m <= float(config.position_tolerance_m),
            position_error=position_error_m,
            tolerance=config.position_tolerance_m,
        )
        debug_stage(
            "move_arm",
            "arrival IK：解出的手臂角度與 FK 末端位置",
            target_base_xyz=_round_float_sequence(target),
            joint_solution_deg=_round_float_sequence(joint_solution_deg),
            final_ee_xyz=_round_float_sequence(final_ee_xyz),
            error_xyz=_round_float_sequence(error_xyz),
            position_only=True,
        )
        return {
            "ik_feasible": bool(position_error_m <= float(config.position_tolerance_m)),
            "target_base_xyz": target,
            "joint_solution_rad": joint_solution,
            "joint_solution_deg": joint_solution_deg,
            "final_ee_position_xyz": final_ee_xyz,
            "position_error_m": position_error_m,
            "error_xyz_m": error_xyz,
            "position_tolerance_m": float(config.position_tolerance_m),
            "position_only": True,
        }
    finally:
        p.disconnect(client_id)


def _controllable_joint_ids(p: Any, robot_id: int, expected_count: int) -> list[int]:
    joint_ids: list[int] = []
    for index in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, index)
        joint_type = info[2]
        joint_name = info[1].decode("utf-8")
        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC) and joint_name != "Revolute 6":
            joint_ids.append(index)
    return joint_ids[: int(expected_count)]


def _set_joint_positions(p: Any, robot_id: int, joint_ids: list[int], joint_positions: Sequence[Any]) -> None:
    for joint_id, value in zip(joint_ids, joint_positions):
        p.resetJointState(robot_id, joint_id, targetValue=float(value), targetVelocity=0.0)


def _clip_to_bounds(values: Sequence[Any], bounds_deg: tuple[tuple[float, float], ...]) -> list[float]:
    result: list[float] = []
    for value, (lower_deg, upper_deg) in zip(values, bounds_deg):
        lower = coord.deg_to_rad(lower_deg)
        upper = coord.deg_to_rad(upper_deg)
        result.append(min(max(float(value), lower), upper))
    return result


def _urdf_search_paths(urdf_path: Path) -> list[Path]:
    return [path for path in (urdf_path.parent, urdf_path.parent.parent) if path.exists()]


def _float_xyz(values: Any, *, label: str) -> list[float]:
    try:
        result = [float(value) for value in values]
    except TypeError as exc:
        raise ValueError(f"{label} must be a numeric [x, y, z] sequence.") from exc
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must contain three finite values.")
    return result
