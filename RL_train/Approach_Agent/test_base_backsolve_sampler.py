import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.spatial.transform as st

import test_base_sampler as sampler


PITCH_SWEEP_DEG = (0.0, -15.0, 15.0, -30.0, 30.0, -45.0, 45.0)


@dataclass(frozen=True)
class MapContact:
    side: str
    ros_map_x: float
    ros_map_y: float
    cell_x: int
    cell_y: int
    local_pb_x: float
    local_pb_y: float


@dataclass(frozen=True)
class FootprintCheck:
    clear: bool
    blocked_count: int
    side_counts: dict[str, int]
    contacts: tuple[MapContact, ...]
    correction_ros_xy: np.ndarray


@dataclass(frozen=True)
class ResetRankedTarget:
    grasp_candidate: sampler.GraspPoseCandidate
    target_pb: np.ndarray
    target_rot_pb: np.ndarray
    target_quat_pb: np.ndarray
    reset_ee_distance_m: float
    reset_ee_error_xyz: np.ndarray
    reset_pitch_error_deg: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live test for target-pose driven base_link backsolve: use the reset arm pose "
            "to infer a fixed-z base_link, repair map-footprint collisions, then run IK + GUI."
        )
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base_pose_sampling.yaml"),
        help="Base sampler YAML. Used for planner path, map path, and tolerances.",
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=Path("configs/camera_car_voxel_ompl.yaml"),
        help="Camera_Car RGBD capture YAML.",
    )
    parser.add_argument(
        "--grasp-json",
        type=Path,
        default=None,
        help="Grasp-agent raw result JSON. Defaults to GRASP_RESULT_JSON, preferred path, then latest session.",
    )
    parser.add_argument(
        "--backsolve-attempts",
        type=int,
        default=int(os.getenv("BASE_BACKSOLVE_ATTEMPTS", "30")),
        help="Deterministic fixed-z base_link backsolve attempts per grasp pose.",
    )
    parser.add_argument(
        "--position-tolerance-m",
        type=float,
        default=float(os.getenv("BASE_BACKSOLVE_POSITION_TOL_M", "0.02")),
        help="Acceptable EE-target position error after IK.",
    )
    parser.add_argument(
        "--pitch-tolerance-deg",
        type=float,
        default=float(os.getenv("BASE_BACKSOLVE_PITCH_TOL_DEG", "45.0")),
        help="Allowed gripper EE local pitch delta relative to target pose.",
    )
    parser.add_argument(
        "--allow-missing-amcl",
        action="store_true",
        help="Continue without /amcl_pose. ROS map footprint repair is skipped without AMCL.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip PyBullet GUI replay.",
    )
    return parser.parse_args()


def _reset_ee_transform_in_base_link(
    p_mod,
    robot_id: int,
    controllable_joint_ids: list[int],
    planning_config,
) -> tuple[np.ndarray, np.ndarray]:
    base_xyz = np.asarray([0.0, 0.0, float(planning_config.initial_height)], dtype=np.float64)
    base_quat = p_mod.getQuaternionFromEuler([0.0, 0.0, 0.0])
    p_mod.resetBasePositionAndOrientation(robot_id, base_xyz.astype(float).tolist(), base_quat)
    sampler._set_joint_positions_direct(
        p_mod,
        robot_id,
        controllable_joint_ids,
        sampler._degrees_to_radians(planning_config.joint_reset_deg),
    )
    p_mod.performCollisionDetection()
    ee_state = p_mod.getLinkState(
        robot_id,
        planning_config.ee_link_index,
        computeForwardKinematics=True,
    )
    ee_xyz = np.asarray(ee_state[4], dtype=np.float64)
    ee_rot = st.Rotation.from_quat(np.asarray(ee_state[5], dtype=np.float64)).as_matrix()
    return ee_xyz - base_xyz, ee_rot


def _best_planar_axis_index(source_rotation: np.ndarray, target_rotation: np.ndarray) -> int:
    source_rotation = np.asarray(source_rotation, dtype=np.float64).reshape(3, 3)
    target_rotation = np.asarray(target_rotation, dtype=np.float64).reshape(3, 3)
    scores = [
        float(np.linalg.norm(source_rotation[:2, index])) + float(np.linalg.norm(target_rotation[:2, index]))
        for index in range(3)
    ]
    return int(np.argmax(scores))


def _axis_yaw_xy(axis_xyz: np.ndarray, fallback_yaw_rad: float = 0.0) -> float:
    axis = np.asarray(axis_xyz, dtype=np.float64).reshape(3)
    norm_xy = float(np.linalg.norm(axis[:2]))
    if norm_xy <= 1e-6:
        return float(fallback_yaw_rad)
    return float(math.atan2(float(axis[1]), float(axis[0])))


def _relative_ee_pitch_error_deg(
    *,
    ee_quat_xyzw: np.ndarray,
    target_rot_pb: np.ndarray,
) -> float:
    ee_rot = st.Rotation.from_quat(np.asarray(ee_quat_xyzw, dtype=np.float64).reshape(4)).as_matrix()
    relative_rot = np.asarray(target_rot_pb, dtype=np.float64).reshape(3, 3).T @ ee_rot
    return float(st.Rotation.from_matrix(relative_rot).as_euler("xyz", degrees=True)[1])


def _rotation_matrix_to_quat(rotation_matrix: np.ndarray) -> np.ndarray:
    return st.Rotation.from_matrix(np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)).as_quat()


def _prepare_reset_distance_ranked_targets(
    *,
    grasp_candidates: list[sampler.GraspPoseCandidate],
    camera_to_pb_rotation: np.ndarray,
    camera_position_pb: np.ndarray,
    planning_config,
    reset_ee_base_xyz: np.ndarray,
    reset_ee_base_rot: np.ndarray,
) -> list[ResetRankedTarget]:
    reset_base_xyz = np.asarray([0.0, 0.0, float(planning_config.initial_height)], dtype=np.float64)
    reset_ee_world_xyz = reset_base_xyz + np.asarray(reset_ee_base_xyz, dtype=np.float64).reshape(3)
    targets: list[ResetRankedTarget] = []
    for grasp_candidate in grasp_candidates:
        target_pb, target_rot_pb, target_quat_pb = sampler._transform_grasp_pose_camera_to_pybullet(
            grasp_candidate,
            camera_to_pb_rotation=camera_to_pb_rotation,
            camera_position_pb=camera_position_pb,
        )
        error_xyz = np.asarray(target_pb, dtype=np.float64).reshape(3) - reset_ee_world_xyz
        reset_pitch_error = _relative_ee_pitch_error_deg(
            ee_quat_xyzw=_rotation_matrix_to_quat(reset_ee_base_rot),
            target_rot_pb=target_rot_pb,
        )
        targets.append(
            ResetRankedTarget(
                grasp_candidate=grasp_candidate,
                target_pb=np.asarray(target_pb, dtype=np.float64),
                target_rot_pb=np.asarray(target_rot_pb, dtype=np.float64),
                target_quat_pb=np.asarray(target_quat_pb, dtype=np.float64),
                reset_ee_distance_m=float(np.linalg.norm(error_xyz)),
                reset_ee_error_xyz=error_xyz.astype(np.float64),
                reset_pitch_error_deg=float(reset_pitch_error),
            )
        )

    targets.sort(
        key=lambda target: (
            float(target.reset_ee_distance_m),
            abs(float(target.reset_pitch_error_deg)),
            int(target.grasp_candidate.rank),
        )
    )
    print("[test_base_backsolve] reset-pose target ranking (nearest first):", flush=True)
    for order_index, target in enumerate(targets, start=1):
        err = target.reset_ee_error_xyz
        print(
            f"  order={order_index:02d} grasp_rank={target.grasp_candidate.rank:02d} "
            f"reset_ee_dist={target.reset_ee_distance_m:.4f}m "
            f"reset_error_xyz=[{err[0]:+.4f},{err[1]:+.4f},{err[2]:+.4f}]m "
            f"reset_pitch_delta={target.reset_pitch_error_deg:+.2f}deg",
            flush=True,
        )
    return targets


def _initial_base_from_reset_ee(
    *,
    target_pb: np.ndarray,
    target_rot_pb: np.ndarray,
    reset_ee_base_xyz: np.ndarray,
    reset_ee_base_rot: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    axis_index = _best_planar_axis_index(reset_ee_base_rot, target_rot_pb)
    reset_axis_yaw = _axis_yaw_xy(reset_ee_base_rot[:, axis_index])
    target_axis_yaw = _axis_yaw_xy(target_rot_pb[:, axis_index], fallback_yaw_rad=reset_axis_yaw)
    base_yaw = sampler._wrap_angle_rad(target_axis_yaw - reset_axis_yaw)
    rot_xy = sampler._yaw_rotation_matrix(base_yaw)[:2, :2]
    base_xy = np.asarray(target_pb, dtype=np.float64).reshape(3)[:2] - rot_xy @ reset_ee_base_xyz[:2]
    return base_xy.astype(np.float64), float(base_yaw), axis_index


def _amcl_pose_from_local_base_pose(
    *,
    current_amcl_pose: sampler.RosMapPose2D | None,
    local_pb_xy: np.ndarray,
    local_pb_yaw_rad: float,
) -> tuple[sampler.RosMapPose2D | None, sampler.RosMapPose2D]:
    base_link_pose = sampler._local_pb_base_pose_to_ros_map_pose(
        (float(local_pb_xy[0]), float(local_pb_xy[1])),
        float(local_pb_yaw_rad),
        current_amcl_pose,
    )
    if current_amcl_pose is None:
        return None, base_link_pose

    base_link_pb_pose = sampler._base_link_pose_to_pb_world_pose(base_link_pose, base_link_z_pb=0.0)
    assert base_link_pb_pose is not None
    base_link_pb_xyz, base_link_pb_yaw = base_link_pb_pose
    offset_pb_xy = (
        sampler._yaw_rotation_matrix(base_link_pb_yaw)[:2, :2]
        @ sampler.BASE_LINK_FROM_AMCL_PB_XY
    )
    amcl_pb_xy = base_link_pb_xyz[:2] - offset_pb_xy
    amcl_ros_x, amcl_ros_y = sampler._pb_world_xy_to_ros_map_xy(
        (float(amcl_pb_xy[0]), float(amcl_pb_xy[1]))
    )
    return (
        sampler.RosMapPose2D(
            x=float(amcl_ros_x),
            y=float(amcl_ros_y),
            yaw_rad=float(base_link_pose.yaw_rad),
        ),
        base_link_pose,
    )


def _side_from_footprint_point(
    local_xy: np.ndarray,
    *,
    half_x: float,
    half_y: float,
) -> str:
    x, y = float(local_xy[0]), float(local_xy[1])
    norm_x = abs(x) / max(float(half_x), 1e-6)
    norm_y = abs(y) / max(float(half_y), 1e-6)
    if norm_x >= norm_y:
        return "front(+pb_x)" if x >= 0.0 else "rear(-pb_x)"
    return "left(+pb_y)" if y >= 0.0 else "right(-pb_y)"


def _check_rectangular_footprint_with_contacts(
    footprint_map: sampler.BaseFootprintMap,
    amcl_pose: sampler.RosMapPose2D | None,
) -> FootprintCheck:
    if amcl_pose is None:
        return FootprintCheck(
            clear=True,
            blocked_count=0,
            side_counts={},
            contacts=(),
            correction_ros_xy=np.zeros(2, dtype=np.float64),
        )

    amcl_pb_pose = sampler._amcl_pose_to_pb_world_pose(amcl_pose, z_pb=0.0)
    assert amcl_pb_pose is not None
    amcl_pb_xyz, amcl_pb_yaw = amcl_pb_pose

    footprint_points = np.asarray(footprint_map.footprint_points_pb_xy, dtype=np.float64).reshape(-1, 2)
    rot_xy = sampler._yaw_rotation_matrix(amcl_pb_yaw)[:2, :2]
    world_pb_xy = amcl_pb_xyz[:2].reshape(1, 2) + footprint_points @ rot_xy.T
    ros_x = world_pb_xy[:, 1]
    ros_y = -world_pb_xy[:, 0]

    origin_x, origin_y = footprint_map.origin_xy
    resolution = float(footprint_map.resolution_m)
    cell_x = np.rint((ros_x - origin_x) / resolution).astype(np.int32)
    cell_y = np.rint((ros_y - origin_y) / resolution).astype(np.int32)
    free_mask = np.asarray(
        [
            (int(x_key), int(y_key)) in footprint_map.free_cell_keys
            for x_key, y_key in zip(cell_x, cell_y)
        ],
        dtype=bool,
    )
    blocked_indices = np.nonzero(~free_mask)[0]
    if len(blocked_indices) == 0:
        return FootprintCheck(
            clear=True,
            blocked_count=0,
            side_counts={},
            contacts=(),
            correction_ros_xy=np.zeros(2, dtype=np.float64),
        )

    half_x = float(footprint_map.length_x_m) * 0.5
    half_y = float(footprint_map.length_y_m) * 0.5
    side_counts: dict[str, int] = {}
    contacts: list[MapContact] = []
    for index in blocked_indices:
        side = _side_from_footprint_point(footprint_points[index], half_x=half_x, half_y=half_y)
        side_counts[side] = side_counts.get(side, 0) + 1
        if len(contacts) < 8:
            contacts.append(
                MapContact(
                    side=side,
                    ros_map_x=float(ros_x[index]),
                    ros_map_y=float(ros_y[index]),
                    cell_x=int(cell_x[index]),
                    cell_y=int(cell_y[index]),
                    local_pb_x=float(footprint_points[index, 0]),
                    local_pb_y=float(footprint_points[index, 1]),
                )
            )

    blocked_local_mean = np.mean(footprint_points[blocked_indices], axis=0)
    if float(np.linalg.norm(blocked_local_mean)) > 1e-6:
        local_away = -blocked_local_mean / float(np.linalg.norm(blocked_local_mean))
        pb_away = rot_xy @ local_away
        correction_ros = np.asarray([pb_away[1], -pb_away[0]], dtype=np.float64)
    else:
        blocked_ros_mean = np.asarray(
            [float(np.mean(ros_x[blocked_indices])), float(np.mean(ros_y[blocked_indices]))],
            dtype=np.float64,
        )
        correction_ros = np.asarray([amcl_pose.x, amcl_pose.y], dtype=np.float64) - blocked_ros_mean

    correction_norm = float(np.linalg.norm(correction_ros))
    if correction_norm <= 1e-6:
        correction_ros = np.asarray([math.cos(amcl_pose.yaw_rad + math.pi), math.sin(amcl_pose.yaw_rad + math.pi)])
    else:
        correction_ros = correction_ros / correction_norm

    return FootprintCheck(
        clear=False,
        blocked_count=int(len(blocked_indices)),
        side_counts=side_counts,
        contacts=tuple(contacts),
        correction_ros_xy=correction_ros.astype(np.float64),
    )


def _format_side_counts(side_counts: dict[str, int]) -> str:
    if not side_counts:
        return "none"
    return ", ".join(f"{side}:{count}" for side, count in sorted(side_counts.items()))


def _format_contacts(contacts: tuple[MapContact, ...]) -> str:
    if not contacts:
        return "[]"
    return "[" + "; ".join(
        (
            f"{contact.side}@ros({contact.ros_map_x:.3f},{contact.ros_map_y:.3f})"
            f"/cell({contact.cell_x},{contact.cell_y})"
        )
        for contact in contacts[:4]
    ) + "]"


def _map_repair_step_m(footprint_map: sampler.BaseFootprintMap, attempt_index: int) -> float:
    base_step = max(float(footprint_map.resolution_m) * 2.0, 0.05)
    return float(base_step * (1.0 + 0.25 * max(int(attempt_index), 0)))


def _update_local_base_from_amcl_repair(
    *,
    repaired_amcl_pose: sampler.RosMapPose2D,
    current_amcl_pose: sampler.RosMapPose2D | None,
) -> tuple[np.ndarray, float]:
    local_xy, local_yaw = sampler._ros_map_amcl_pose_to_local_pb_base_pose(
        repaired_amcl_pose,
        current_amcl_pose,
    )
    return np.asarray(local_xy, dtype=np.float64), float(local_yaw)


def _ik_error_repair(
    *,
    local_xy: np.ndarray,
    local_yaw: float,
    ik_attempt: dict[str, object],
    target_rot_pb: np.ndarray,
) -> tuple[np.ndarray, float]:
    error_xyz = np.asarray(ik_attempt["ik_error_xyz"], dtype=np.float64).reshape(3)
    delta_xy = np.clip(error_xyz[:2] * 0.45, -0.08, 0.08)
    repaired_xy = np.asarray(local_xy, dtype=np.float64).reshape(2) + delta_xy

    ee_quat = np.asarray(ik_attempt["final_ee_orientation_xyzw"], dtype=np.float64).reshape(4)
    ee_rot = st.Rotation.from_quat(ee_quat).as_matrix()
    axis_index = _best_planar_axis_index(ee_rot, target_rot_pb)
    ee_yaw = _axis_yaw_xy(ee_rot[:, axis_index], fallback_yaw_rad=0.0)
    target_yaw = _axis_yaw_xy(target_rot_pb[:, axis_index], fallback_yaw_rad=ee_yaw)
    yaw_error = sampler._wrap_angle_rad(target_yaw - ee_yaw)
    repaired_yaw = sampler._wrap_angle_rad(
        float(local_yaw) + float(np.clip(yaw_error * 0.35, -math.radians(5.0), math.radians(5.0)))
    )
    return repaired_xy.astype(np.float64), float(repaired_yaw)


def _pitch_variant_quat(
    target_rot_pb: np.ndarray,
    pitch_offset_deg: float,
) -> np.ndarray:
    variant_rot = (
        np.asarray(target_rot_pb, dtype=np.float64).reshape(3, 3)
        @ st.Rotation.from_euler("y", float(pitch_offset_deg), degrees=True).as_matrix()
    )
    return _rotation_matrix_to_quat(variant_rot)


def _attempt_pitch_sweep_ik_at_base_pose(
    *,
    p_mod,
    robot_id: int,
    controllable_joint_ids: list[int],
    planning_config,
    target_pb: np.ndarray,
    target_rot_pb: np.ndarray,
    base_link_xy: tuple[float, float],
    base_link_yaw_rad: float,
    obstacle_body_ids: list[int],
    pitch_tolerance_deg: float,
) -> dict[str, object]:
    best_attempt: dict[str, object] | None = None
    best_key: tuple[float, float, float, float] | None = None
    for pitch_offset_deg in PITCH_SWEEP_DEG:
        if abs(float(pitch_offset_deg)) > float(pitch_tolerance_deg) + 1e-6:
            continue
        ik_attempt = sampler._attempt_ik_at_base_pose(
            p_mod=p_mod,
            robot_id=robot_id,
            controllable_joint_ids=controllable_joint_ids,
            planning_config=planning_config,
            target_pb=target_pb,
            target_quat_pb=_pitch_variant_quat(target_rot_pb, float(pitch_offset_deg)),
            base_link_xy=base_link_xy,
            base_link_yaw_rad=base_link_yaw_rad,
            obstacle_body_ids=obstacle_body_ids,
        )
        pitch_error_deg = _relative_ee_pitch_error_deg(
            ee_quat_xyzw=np.asarray(ik_attempt["final_ee_orientation_xyzw"], dtype=np.float64),
            target_rot_pb=target_rot_pb,
        )
        ik_attempt["ee_pitch_error_deg"] = float(pitch_error_deg)
        ik_attempt["target_pitch_offset_deg"] = float(pitch_offset_deg)
        key = (
            float(ik_attempt["ee_position_error_m"]),
            abs(float(pitch_error_deg)),
            0.0 if bool(ik_attempt["collision_free"]) else 1.0,
            float(ik_attempt["joint_reset_delta_norm_l2"] or float("inf")),
        )
        if best_key is None or key < best_key:
            best_key = key
            best_attempt = ik_attempt

    if best_attempt is None:
        raise RuntimeError("Pitch sweep produced no IK attempts.")
    return best_attempt


def _backsolve_solution_sort_key(solution: dict[str, object]) -> tuple[float, float, float, int]:
    return (
        float(solution.get("ee_position_error_m", float("inf"))),
        abs(float(solution.get("ee_pitch_error_deg", float("inf")))),
        float(solution.get("joint_reset_delta_norm_l2", float("inf"))),
        int(solution.get("sample_index", 0)),
    )


def _solution_record_for_attempt(
    *,
    ik_attempt: dict[str, object],
    attempt_index: int,
    local_xy: np.ndarray,
    local_yaw: float,
    ros_map_base_link_pose: sampler.RosMapPose2D,
    ros_map_amcl_pose: sampler.RosMapPose2D | None,
    target_pb: np.ndarray,
    feasible: bool,
    attempt_count: int,
) -> dict[str, object] | None:
    sample_candidate = {
        "sample_index": int(attempt_index),
        "sample_source": "reset_pose_backsolve",
        "map_cell_index": -1,
        "local_pb_xy": (float(local_xy[0]), float(local_xy[1])),
        "local_pb_yaw_rad": float(local_yaw),
        "ros_map_pose": ros_map_base_link_pose,
        "ros_map_amcl_pose": ros_map_amcl_pose,
        "distance_to_target_m": float(
            np.linalg.norm(np.asarray(target_pb, dtype=np.float64).reshape(3)[:2] - local_xy.reshape(2))
        ),
        "approach_error_deg": 0.0,
    }
    sample_stats = {"region_cell_count": int(attempt_count)}
    return sampler._make_sampled_ik_solution_record(
        ik_attempt=ik_attempt,
        sample_candidate=sample_candidate,
        sample_stats=sample_stats,
        reference_base_yaw_pb=0.0,
        feasible=feasible,
    )


def _backsolve_base_link_for_target(
    *,
    p_mod,
    robot_id: int,
    controllable_joint_ids: list[int],
    planning_config,
    obstacle_body_ids: list[int],
    footprint_map: sampler.BaseFootprintMap,
    current_amcl_pose: sampler.RosMapPose2D | None,
    target_pb: np.ndarray,
    target_rot_pb: np.ndarray,
    reset_ee_base_xyz: np.ndarray,
    reset_ee_base_rot: np.ndarray,
    grasp_rank: int,
    attempt_count: int,
    position_tolerance_m: float,
    pitch_tolerance_deg: float,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    local_xy, local_yaw, axis_index = _initial_base_from_reset_ee(
        target_pb=target_pb,
        target_rot_pb=target_rot_pb,
        reset_ee_base_xyz=reset_ee_base_xyz,
        reset_ee_base_rot=reset_ee_base_rot,
    )
    print(
        f"[test_base_backsolve] rank={int(grasp_rank):02d} reset-pose backsolve: "
        f"axis={axis_index} initial_base_pb=({local_xy[0]:.4f},{local_xy[1]:.4f}) "
        f"yaw={math.degrees(local_yaw):.2f}deg attempts={int(attempt_count)}",
        flush=True,
    )

    feasible_solutions: list[dict[str, object]] = []
    closest_solution: dict[str, object] | None = None
    max_attempts = max(1, int(attempt_count))

    for attempt_index in range(max_attempts):
        ros_map_amcl_pose, ros_map_base_link_pose = _amcl_pose_from_local_base_pose(
            current_amcl_pose=current_amcl_pose,
            local_pb_xy=local_xy,
            local_pb_yaw_rad=local_yaw,
        )
        footprint_check = _check_rectangular_footprint_with_contacts(footprint_map, ros_map_amcl_pose)
        if not footprint_check.clear:
            step_m = _map_repair_step_m(footprint_map, attempt_index)
            assert ros_map_amcl_pose is not None
            repaired_amcl_pose = sampler.RosMapPose2D(
                x=float(ros_map_amcl_pose.x + footprint_check.correction_ros_xy[0] * step_m),
                y=float(ros_map_amcl_pose.y + footprint_check.correction_ros_xy[1] * step_m),
                yaw_rad=float(ros_map_amcl_pose.yaw_rad),
            )
            print(
                f"[test_base_backsolve] rank={int(grasp_rank):02d} attempt={attempt_index + 1:02d} "
                f"MAP_BLOCKED blocked_points={footprint_check.blocked_count} "
                f"sides={_format_side_counts(footprint_check.side_counts)} "
                f"contacts={_format_contacts(footprint_check.contacts)} "
                f"repair_ros_delta=({footprint_check.correction_ros_xy[0] * step_m:+.4f},"
                f"{footprint_check.correction_ros_xy[1] * step_m:+.4f})m",
                flush=True,
            )
            local_xy, local_yaw = _update_local_base_from_amcl_repair(
                repaired_amcl_pose=repaired_amcl_pose,
                current_amcl_pose=current_amcl_pose,
            )
            continue

        ik_attempt = _attempt_pitch_sweep_ik_at_base_pose(
            p_mod=p_mod,
            robot_id=robot_id,
            controllable_joint_ids=controllable_joint_ids,
            planning_config=planning_config,
            target_pb=target_pb,
            target_rot_pb=target_rot_pb,
            base_link_xy=(float(local_xy[0]), float(local_xy[1])),
            base_link_yaw_rad=float(local_yaw),
            obstacle_body_ids=obstacle_body_ids,
            pitch_tolerance_deg=float(pitch_tolerance_deg),
        )
        pitch_error = float(ik_attempt["ee_pitch_error_deg"])
        pitch_ok = abs(pitch_error) <= float(pitch_tolerance_deg)
        feasible = (
            float(ik_attempt["ee_position_error_m"]) <= float(position_tolerance_m)
            and pitch_ok
            and bool(ik_attempt["collision_free"])
            and ik_attempt["ik_joint_solution_rad"] is not None
        )
        solution_record = _solution_record_for_attempt(
            ik_attempt=ik_attempt,
            attempt_index=attempt_index,
            local_xy=local_xy,
            local_yaw=local_yaw,
            ros_map_base_link_pose=ros_map_base_link_pose,
            ros_map_amcl_pose=ros_map_amcl_pose,
            target_pb=target_pb,
            feasible=feasible,
            attempt_count=max_attempts,
        )
        if solution_record is not None:
            solution_record["backsolve_attempt_count"] = int(max_attempts)
            solution_record["ee_pitch_error_deg"] = float(pitch_error)
            solution_record["target_pitch_offset_deg"] = float(ik_attempt["target_pitch_offset_deg"])
            if (
                closest_solution is None
                or _backsolve_solution_sort_key(solution_record)
                < _backsolve_solution_sort_key(closest_solution)
            ):
                closest_solution = solution_record
            if feasible:
                feasible_solutions.append(solution_record)

        print(
            f"[test_base_backsolve] rank={int(grasp_rank):02d} attempt={attempt_index + 1:02d} "
            f"MAP_CLEAR ik_dist={float(ik_attempt['ee_position_error_m']):.4f}m "
            f"pitch={pitch_error:+.2f}deg "
            f"pitch_target_offset={float(ik_attempt['target_pitch_offset_deg']):+.1f}deg "
            f"collision_free={int(bool(ik_attempt['collision_free']))} "
            f"feasible={int(feasible)}",
            flush=True,
        )
        if feasible:
            break

        local_xy, local_yaw = _ik_error_repair(
            local_xy=local_xy,
            local_yaw=local_yaw,
            ik_attempt=ik_attempt,
            target_rot_pb=target_rot_pb,
        )

    feasible_solutions.sort(key=_backsolve_solution_sort_key)
    if feasible_solutions:
        feasible_solutions[0]["selected_as_best"] = True
    if closest_solution is not None:
        closest_solution["selected_as_closest"] = True
    return feasible_solutions, closest_solution


def main() -> None:
    args = _parse_args()
    cfg = sampler.load_config(args.base_config)
    planning_config = sampler.load_planning_config(Path(cfg["planner_config_path"]))
    arm_config = sampler._load_arm_config()
    _, p_mod, pybullet_data = sampler._load_python_dependencies()
    position_tolerance_m = float(args.position_tolerance_m)
    pitch_tolerance_deg = float(args.pitch_tolerance_deg)
    footprint_map = sampler._build_base_footprint_map(Path(cfg["map_yaml_path"]))
    (
        camera_in_base_link_rotation,
        camera_in_base_link_position,
        base_link_z_pb,
    ) = sampler._get_reset_camera_transform_in_base_link_frame(
        Path(cfg["planner_config_path"]),
        planning_config,
    )
    grasp_candidates, _, grasp_result_json_path = sampler._load_grasp_candidates_from_result_json(args.grasp_json)

    live_scene = sampler._capture_live_scene_voxels(
        args.camera_config.resolve(),
        camera_in_base_link_rotation,
        camera_in_base_link_position,
        base_link_z_pb,
    )
    voxels_pb = live_scene.voxel_centers_pb
    camera_to_pb_rotation = live_scene.camera_to_pb_rotation
    camera_position_pb = live_scene.camera_position_pb
    current_amcl_pose = sampler._amcl_snapshot_to_ros_map_pose(live_scene.amcl_pose)
    if current_amcl_pose is None and not args.allow_missing_amcl:
        raise RuntimeError(
            "No /amcl_pose was received during Camera_Car capture. "
            "This backsolve test needs AMCL for ROS map footprint repair."
        )

    print(
        f"[test_base_backsolve] grasp_json={grasp_result_json_path} | "
        f"grasps={len(grasp_candidates)} | live RGBD voxels={len(voxels_pb)} | "
        f"depth_points={live_scene.valid_depth_point_count} | "
        f"amcl={'yes' if current_amcl_pose is not None else 'no'} | "
        f"attempts_per_pose={int(args.backsolve_attempts)} | "
        f"position_tol={position_tolerance_m:.3f}m | "
        f"pitch_tol={pitch_tolerance_deg:.1f}deg",
        flush=True,
    )
    sampler._print_ros_map_pose("current /amcl_pose vehicle center", current_amcl_pose)

    client_id = p_mod.connect(p_mod.DIRECT)
    visualization_records: list[dict[str, object]] = []
    first_feasible_result: dict[str, object] | None = None
    try:
        p_mod.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_mod.resetSimulation()
        p_mod.setGravity(0.0, 0.0, -9.8)
        p_mod.loadURDF("plane.urdf")

        voxel_size = float(cfg.get("voxel_size_m", 0.05))
        half_extents = [voxel_size / 2.0] * 3
        col_shape = p_mod.createCollisionShape(p_mod.GEOM_BOX, halfExtents=half_extents)
        vis_shape = p_mod.createVisualShape(
            p_mod.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.8, 0.2, 0.2, 0.8],
        )
        obstacle_body_ids: list[int] = []
        for voxel_center in voxels_pb:
            obstacle_body_ids.append(
                p_mod.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col_shape,
                    baseVisualShapeIndex=vis_shape,
                    basePosition=voxel_center.astype(float).tolist(),
                )
            )

        robot_id = p_mod.loadURDF(
            planning_config.urdf_path,
            useFixedBase=True,
            basePosition=[0.0, 0.0, planning_config.initial_height],
            baseOrientation=p_mod.getQuaternionFromEuler([0.0, 0.0, 0.0]),
        )
        expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])
        controllable_joint_ids, _ = sampler._find_controllable_joints(p_mod, robot_id, expected_joint_count)
        reset_ee_base_xyz, reset_ee_base_rot = _reset_ee_transform_in_base_link(
            p_mod,
            robot_id,
            controllable_joint_ids,
            planning_config,
        )
        print(
            "[test_base_backsolve] reset EE in base_link: "
            f"xyz=({reset_ee_base_xyz[0]:.4f},{reset_ee_base_xyz[1]:.4f},{reset_ee_base_xyz[2]:.4f})m "
            f"fixed_base_z={float(planning_config.initial_height):.4f}m "
            f"reset_ee_world_z={float(planning_config.initial_height + reset_ee_base_xyz[2]):.4f}m",
            flush=True,
        )

        prepared_targets = _prepare_reset_distance_ranked_targets(
            grasp_candidates=grasp_candidates,
            camera_to_pb_rotation=camera_to_pb_rotation,
            camera_position_pb=camera_position_pb,
            planning_config=planning_config,
            reset_ee_base_xyz=reset_ee_base_xyz,
            reset_ee_base_rot=reset_ee_base_rot,
        )

        for prepared_target in prepared_targets:
            grasp_candidate = prepared_target.grasp_candidate
            feasible_solutions, closest_solution = _backsolve_base_link_for_target(
                p_mod=p_mod,
                robot_id=robot_id,
                controllable_joint_ids=controllable_joint_ids,
                obstacle_body_ids=obstacle_body_ids,
                planning_config=planning_config,
                footprint_map=footprint_map,
                current_amcl_pose=current_amcl_pose,
                target_pb=prepared_target.target_pb,
                target_rot_pb=prepared_target.target_rot_pb,
                reset_ee_base_xyz=reset_ee_base_xyz,
                reset_ee_base_rot=reset_ee_base_rot,
                grasp_rank=grasp_candidate.rank,
                attempt_count=int(args.backsolve_attempts),
                position_tolerance_m=position_tolerance_m,
                pitch_tolerance_deg=pitch_tolerance_deg,
            )
            visualization_records.append(
                {
                    "rank": grasp_candidate.rank,
                    "grasp_confidence": grasp_candidate.grasp_confidence,
                    "direct_ik_error_m": float(prepared_target.reset_ee_distance_m),
                    "direct_orientation_error_deg": float(abs(prepared_target.reset_pitch_error_deg)),
                    "target_pb": prepared_target.target_pb.astype(float).tolist(),
                    "target_quat_pb": prepared_target.target_quat_pb.astype(float).tolist(),
                    "feasible_solutions": feasible_solutions,
                    "closest_solution": closest_solution,
                }
            )
            sampler._print_closest_ik_solution_banner(
                rank=grasp_candidate.rank,
                target_pb=prepared_target.target_pb,
                solution=closest_solution,
            )

            if feasible_solutions:
                best_solution = feasible_solutions[0]
                sampler._print_selected_base_link_ros_map_banner(
                    rank=grasp_candidate.rank,
                    amcl_pose=best_solution.get("ros_map_amcl_pose"),
                    base_link_pose=best_solution.get("ros_map_base_link_pose"),
                )
                first_feasible_result = {
                    "rank": grasp_candidate.rank,
                    "feasible_ik_count": len(feasible_solutions),
                    "ee_error_m": float(best_solution["ee_position_error_m"]),
                }
                break
    finally:
        p_mod.disconnect(client_id)

    if first_feasible_result is None:
        print("[test_base_backsolve] No feasible backsolved IK; GUI will show closest attempt.", flush=True)
    else:
        print(
            f"[test_base_backsolve] Visualizing feasible backsolve rank={first_feasible_result['rank']:02d} "
            f"(feasible_ik={first_feasible_result['feasible_ik_count']}, "
            f"ee_error={first_feasible_result['ee_error_m']:.4f}m).",
            flush=True,
        )

    if not args.no_gui:
        sampler._visualize_feasible_ik_results_in_gui(
            p_mod=p_mod,
            pybullet_data=pybullet_data,
            planning_config=planning_config,
            arm_config=arm_config,
            voxels_pb=voxels_pb,
            visualization_records=visualization_records,
        )


if __name__ == "__main__":
    main()
