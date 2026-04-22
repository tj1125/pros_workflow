from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.spatial.transform as st


PLANNING_GRASP_TO_EE = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class GraspPoseCandidate:
    index: int
    rank: int
    grasp_confidence: float
    grasp_distance_to_gripper_midpoint_m: float
    grasp_distance_to_camera_m: float
    position_camera_xyz: np.ndarray
    rotation_camera: np.ndarray


@dataclass(frozen=True)
class SimpleBaseSampleResult:
    visualization_records: list[dict[str, object]]
    selected_record: dict[str, object] | None
    selected_solution: dict[str, object] | None
    feasible: bool
    attempted_count: int
    map_feasible_count: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _preferred_grasp_result_json() -> Path:
    return (
        _repo_root()
        / "logs"
        / "sessions"
        / "fbbc240a8ec8476ea09b15d0d3339ac9"
        / "artifacts"
        / "raw_results"
        / "step_007_grasp_agent.json"
    )


def _find_latest_grasp_result_json() -> Path:
    candidates = sorted(
        (_repo_root() / "logs" / "sessions").glob("*/artifacts/raw_results/step_*_grasp_agent.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("No step_*_grasp_agent.json was found under VLM_RL/logs/sessions.")
    return candidates[-1]


def resolve_grasp_result_json_path(grasp_result_json_path: Path | None = None) -> Path:
    env_override = os.getenv("GRASP_RESULT_JSON", "").strip()
    if grasp_result_json_path is not None:
        return grasp_result_json_path.expanduser().resolve()
    if env_override:
        return Path(env_override).expanduser().resolve()
    preferred = _preferred_grasp_result_json()
    if preferred.exists():
        return preferred.resolve()
    return _find_latest_grasp_result_json()


def load_grasp_candidates_from_result_json(
    grasp_result_json_path: Path | None = None,
) -> tuple[list[GraspPoseCandidate], np.ndarray, Path]:
    json_path = resolve_grasp_result_json_path(grasp_result_json_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    raw_result = payload.get("raw_result", payload)
    gripper_midpoint_camera_xyz = np.asarray(
        raw_result.get("gripper_midpoint_camera_xyz", [0.0, -0.04, 0.11]),
        dtype=np.float64,
    )
    if gripper_midpoint_camera_xyz.shape != (3,):
        raise ValueError(
            "gripper_midpoint_camera_xyz must have shape (3,), "
            f"got {gripper_midpoint_camera_xyz.shape}."
        )

    raw_candidates = raw_result.get("valid_grasp_poses_camera")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        best_grasp = raw_result.get("best_grasp_pose_camera")
        if not isinstance(best_grasp, dict):
            raise KeyError(f"No valid_grasp_poses_camera or best_grasp_pose_camera found in {json_path}")
        raw_candidates = [best_grasp]

    grasp_candidates: list[GraspPoseCandidate] = []
    for index, candidate in enumerate(raw_candidates):
        matrix_4x4 = candidate.get("matrix_4x4")
        if matrix_4x4 is not None:
            grasp_matrix_camera = np.asarray(matrix_4x4, dtype=np.float64)
            if grasp_matrix_camera.shape != (4, 4):
                raise ValueError(
                    f"Candidate matrix_4x4 must have shape (4, 4), got {grasp_matrix_camera.shape}."
                )
            position_camera_xyz = np.asarray(grasp_matrix_camera[:3, 3], dtype=np.float64)
            rotation_camera = np.asarray(grasp_matrix_camera[:3, :3], dtype=np.float64)
        else:
            position_camera_xyz = np.asarray(candidate["position"], dtype=np.float64)
            rotation_camera = np.asarray(candidate["rotation_matrix"], dtype=np.float64)

        if position_camera_xyz.shape != (3,):
            raise ValueError(
                f"Candidate position must have shape (3,), got {position_camera_xyz.shape}."
            )
        if rotation_camera.shape != (3, 3):
            raise ValueError(
                f"Candidate rotation_matrix must have shape (3, 3), got {rotation_camera.shape}."
            )

        grasp_candidates.append(
            GraspPoseCandidate(
                index=index,
                rank=int(candidate.get("rank", index + 1)),
                grasp_confidence=float(candidate.get("grasp_confidence", float("nan"))),
                grasp_distance_to_gripper_midpoint_m=float(
                    candidate.get("grasp_distance_to_gripper_midpoint_m", float("nan"))
                ),
                grasp_distance_to_camera_m=float(candidate.get("grasp_distance_to_camera_m", float("nan"))),
                position_camera_xyz=position_camera_xyz,
                rotation_camera=rotation_camera,
            )
        )

    grasp_candidates.sort(key=lambda item: item.rank)
    return grasp_candidates, gripper_midpoint_camera_xyz, json_path


def transform_camera_points_to_local_pb(
    points_camera_xyz: np.ndarray,
    *,
    camera_to_pb_rotation: np.ndarray,
    camera_position_pb: np.ndarray,
) -> np.ndarray:
    points_camera_xyz = np.asarray(points_camera_xyz, dtype=np.float64).reshape(-1, 3)
    if len(points_camera_xyz) == 0:
        return points_camera_xyz
    return (points_camera_xyz @ np.asarray(camera_to_pb_rotation, dtype=np.float64).reshape(3, 3).T) + np.asarray(
        camera_position_pb,
        dtype=np.float64,
    ).reshape(1, 3)


def transform_grasp_pose_camera_to_pybullet(
    grasp_candidate: GraspPoseCandidate,
    *,
    camera_to_pb_rotation: np.ndarray,
    camera_position_pb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_to_pb_rotation = np.asarray(camera_to_pb_rotation, dtype=np.float64).reshape(3, 3)
    target_pb = transform_camera_points_to_local_pb(
        grasp_candidate.position_camera_xyz.reshape(1, 3),
        camera_to_pb_rotation=camera_to_pb_rotation,
        camera_position_pb=camera_position_pb,
    ).reshape(3)
    target_rot_pb = (camera_to_pb_rotation @ grasp_candidate.rotation_camera) @ PLANNING_GRASP_TO_EE
    target_quat_pb = st.Rotation.from_matrix(target_rot_pb).as_quat()
    return (
        np.asarray(target_pb, dtype=np.float64),
        np.asarray(target_rot_pb, dtype=np.float64),
        np.asarray(target_quat_pb, dtype=np.float64),
    )


def load_grasp_visualization_records(
    grasp_result_json_path: Path | None,
    *,
    camera_to_pb_rotation: np.ndarray,
    camera_position_pb: np.ndarray,
) -> tuple[list[dict[str, object]], list[GraspPoseCandidate], Path]:
    grasp_candidates, _, json_path = load_grasp_candidates_from_result_json(grasp_result_json_path)
    visualization_records: list[dict[str, object]] = []
    for grasp_candidate in grasp_candidates:
        target_pb, target_rot_pb, target_quat_pb = transform_grasp_pose_camera_to_pybullet(
            grasp_candidate,
            camera_to_pb_rotation=camera_to_pb_rotation,
            camera_position_pb=camera_position_pb,
        )
        visualization_records.append(
            {
                "rank": int(grasp_candidate.rank),
                "grasp_confidence": float(grasp_candidate.grasp_confidence),
                "grasp_distance_to_gripper_midpoint_m": float(
                    grasp_candidate.grasp_distance_to_gripper_midpoint_m
                ),
                "grasp_distance_to_camera_m": float(grasp_candidate.grasp_distance_to_camera_m),
                "position_camera_xyz": grasp_candidate.position_camera_xyz.astype(float).tolist(),
                "rotation_camera": grasp_candidate.rotation_camera.astype(float).tolist(),
                "target_pb": target_pb.astype(float).tolist(),
                "target_rot_pb": target_rot_pb.astype(float).tolist(),
                "target_quat_pb": target_quat_pb.astype(float).tolist(),
                "feasible_solutions": [],
                "closest_solution": None,
            }
        )

    return visualization_records, grasp_candidates, json_path


def _wrap_angle_rad(angle_rad: float) -> float:
    return float((float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi)


def _abs_angle_delta_rad(angle_a_rad: float, angle_b_rad: float) -> float:
    return abs(_wrap_angle_rad(float(angle_a_rad) - float(angle_b_rad)))


def _axis_yaw_xy(axis_xyz: np.ndarray, fallback_yaw_rad: float = 0.0) -> float:
    axis = np.asarray(axis_xyz, dtype=np.float64).reshape(3)
    axis_xy = axis[:2]
    axis_norm = float(np.linalg.norm(axis_xy))
    if axis_norm <= 1e-6:
        return float(fallback_yaw_rad)
    return float(math.atan2(float(axis_xy[1]), float(axis_xy[0])))


def _quat_xyzw_yaw(quat_xyzw: np.ndarray, fallback_yaw_rad: float = 0.0) -> float:
    rot = st.Rotation.from_quat(np.asarray(quat_xyzw, dtype=np.float64).reshape(4)).as_matrix()
    return _axis_yaw_xy(rot[:, 0], fallback_yaw_rad=fallback_yaw_rad)


def _ordinal_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    sortable = np.where(np.isfinite(values), values, np.inf)
    order = np.argsort(sortable, kind="stable")
    ranks = np.empty(len(sortable), dtype=np.int32)
    ranks[order] = np.arange(1, len(sortable) + 1, dtype=np.int32)
    return ranks


def rank_grasp_records_by_reset_ee_pose(
    visualization_records: list[dict[str, object]],
    *,
    reset_ee_position_xyz: np.ndarray,
    reset_ee_orientation_xyzw: np.ndarray,
) -> list[dict[str, object]]:
    reset_ee_position = np.asarray(reset_ee_position_xyz, dtype=np.float64).reshape(3)
    reset_ee_yaw = _quat_xyzw_yaw(reset_ee_orientation_xyzw)
    ranked_records = [dict(record) for record in visualization_records]
    if not ranked_records:
        return ranked_records

    distances = []
    yaw_errors = []
    for record in ranked_records:
        target_pb = np.asarray(record["target_pb"], dtype=np.float64).reshape(3)
        target_rot_pb = np.asarray(record["target_rot_pb"], dtype=np.float64).reshape(3, 3)
        target_yaw = _axis_yaw_xy(target_rot_pb[:, 0], fallback_yaw_rad=reset_ee_yaw)
        distance_m = float(np.linalg.norm(target_pb - reset_ee_position))
        yaw_error_rad = _abs_angle_delta_rad(target_yaw, reset_ee_yaw)
        distances.append(distance_m)
        yaw_errors.append(yaw_error_rad)
        record["reset_ee_distance_m"] = distance_m
        record["reset_ee_yaw_error_rad"] = yaw_error_rad
        record["reset_ee_yaw_error_deg"] = float(math.degrees(yaw_error_rad))
        record["target_yaw_rad"] = float(target_yaw)
        record["target_yaw_deg"] = float(math.degrees(target_yaw))

    distance_ranks = _ordinal_ranks(np.asarray(distances, dtype=np.float64))
    yaw_ranks = _ordinal_ranks(np.asarray(yaw_errors, dtype=np.float64))
    for record, distance_rank, yaw_rank in zip(ranked_records, distance_ranks, yaw_ranks):
        average_rank = 0.5 * (float(distance_rank) + float(yaw_rank))
        record["reset_ee_distance_rank"] = int(distance_rank)
        record["reset_ee_yaw_rank"] = int(yaw_rank)
        record["target_average_rank"] = float(average_rank)

    ranked_records.sort(
        key=lambda record: (
            float(record["target_average_rank"]),
            float(record["reset_ee_distance_m"]),
            float(record["reset_ee_yaw_error_rad"]),
            int(record["rank"]),
        )
    )
    for order_index, record in enumerate(ranked_records, start=1):
        record["target_sample_order"] = int(order_index)
    return ranked_records


def _backoff_distances(
    *,
    min_backoff_m: float,
    max_backoff_m: float,
    step_m: float,
) -> list[float]:
    min_backoff = max(0.0, float(min_backoff_m))
    max_backoff = max(min_backoff, float(max_backoff_m))
    step = max(float(step_m), 1e-3)
    distances: list[float] = []
    distance = min_backoff
    while distance <= max_backoff + 1e-9:
        distances.append(float(distance))
        distance += step
    return distances


def _ik_error_sort_key(solution: dict[str, object]) -> tuple[float, float, float, int]:
    orientation_error = solution.get("ee_orientation_error_deg")
    return (
        float(solution.get("ee_position_error_m", float("inf"))),
        float("inf") if orientation_error is None else float(orientation_error),
        float(solution.get("joint_reset_delta_norm_l2", float("inf"))),
        int(solution.get("sample_index", 0)),
    )


def _frontmost_display_sort_key(solution: dict[str, object]) -> tuple[float, int, float, float, float, int]:
    orientation_error = solution.get("ee_orientation_error_deg")
    return (
        float(solution.get("backoff_distance_m", solution.get("ros_map_sample_distance_to_target_m", float("inf")))),
        0 if bool(solution.get("map_footprint_clear", False)) else 1,
        float(solution.get("ee_position_error_m", float("inf"))),
        float("inf") if orientation_error is None else float(orientation_error),
        float(solution.get("joint_reset_delta_norm_l2", float("inf"))),
        int(solution.get("sample_index", 0)),
    )


def _clamp_abs(value: float, max_abs: float) -> float:
    limit = max(0.0, float(max_abs))
    return float(max(-limit, min(limit, float(value))))


def _ik_attempt_is_feasible(
    ik_attempt: dict[str, object],
    *,
    position_tolerance_m: float,
    orientation_tolerance_deg: float,
) -> bool:
    orientation_error = ik_attempt.get("ee_orientation_error_deg")
    return (
        ik_attempt.get("ik_joint_solution_rad") is not None
        and bool(ik_attempt.get("collision_free", False))
        and float(ik_attempt.get("ee_position_error_m", float("inf"))) <= float(position_tolerance_m)
        and orientation_error is not None
        and float(orientation_error) <= float(orientation_tolerance_deg)
    )


def sample_base_pose_for_best_grasp(
    visualization_records: list[dict[str, object]],
    *,
    reset_ee_position_xyz: np.ndarray,
    reset_ee_orientation_xyzw: np.ndarray,
    pose_from_local_base_fn: Callable[[np.ndarray, float], tuple[object | None, object]],
    footprint_is_clear_fn: Callable[[object], bool],
    attempt_ik_fn: Callable[[np.ndarray, float, dict[str, object]], dict[str, object]],
    make_solution_record_fn: Callable[[dict[str, object], dict[str, object], dict[str, object], bool], dict[str, object] | None],
    position_tolerance_m: float,
    orientation_tolerance_deg: float,
    current_amcl_yaw_rad: float | None = None,
    max_amcl_yaw_delta_deg: float | None = None,
    min_backoff_m: float | None = None,
    max_backoff_m: float | None = None,
    step_m: float | None = None,
) -> SimpleBaseSampleResult:
    ranked_records = rank_grasp_records_by_reset_ee_pose(
        visualization_records,
        reset_ee_position_xyz=reset_ee_position_xyz,
        reset_ee_orientation_xyzw=reset_ee_orientation_xyzw,
    )
    if not ranked_records:
        return SimpleBaseSampleResult(
            visualization_records=[],
            selected_record=None,
            selected_solution=None,
            feasible=False,
            attempted_count=0,
            map_feasible_count=0,
        )

    min_backoff = float(os.getenv("BASE_SIMPLE_SAMPLE_MIN_BACKOFF_M", "0.10")) if min_backoff_m is None else float(min_backoff_m)
    max_backoff = float(os.getenv("BASE_SIMPLE_SAMPLE_MAX_BACKOFF_M", "0.45")) if max_backoff_m is None else float(max_backoff_m)
    step = float(os.getenv("BASE_SIMPLE_SAMPLE_STEP_M", "0.02")) if step_m is None else float(step_m)
    max_yaw_delta_deg = (
        float(os.getenv("BASE_SIMPLE_SAMPLE_MAX_AMCL_YAW_DELTA_DEG", "60.0"))
        if max_amcl_yaw_delta_deg is None
        else float(max_amcl_yaw_delta_deg)
    )
    max_yaw_delta_rad = math.radians(max(0.0, max_yaw_delta_deg))
    distances = _backoff_distances(
        min_backoff_m=min_backoff,
        max_backoff_m=max_backoff,
        step_m=step,
    )

    selected_record = ranked_records[0]
    target_pb = np.asarray(selected_record["target_pb"], dtype=np.float64).reshape(3)
    target_rot_pb = np.asarray(selected_record["target_rot_pb"], dtype=np.float64).reshape(3, 3)
    approach_xy = np.asarray(target_rot_pb[:2, 0], dtype=np.float64)
    approach_norm = float(np.linalg.norm(approach_xy))
    if approach_norm <= 1e-6:
        approach_xy = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        approach_xy = approach_xy / approach_norm
    desired_base_yaw_rad = _wrap_angle_rad(float(math.atan2(float(approach_xy[1]), float(approach_xy[0]))))
    base_yaw_rad = (
        _clamp_abs(desired_base_yaw_rad, max_yaw_delta_rad)
        if current_amcl_yaw_rad is not None
        else desired_base_yaw_rad
    )

    selected_record["sample_source"] = "simple_backoff_from_grasp"
    selected_record["sample_backoff_min_m"] = float(min_backoff)
    selected_record["sample_backoff_max_m"] = float(max_backoff)
    selected_record["sample_backoff_step_m"] = float(step)
    selected_record["sample_max_amcl_yaw_delta_deg"] = float(max_yaw_delta_deg)
    selected_record["sample_approach_dir_pb_xy"] = approach_xy.astype(float).tolist()
    selected_record["sample_desired_base_yaw_rad"] = float(desired_base_yaw_rad)
    selected_record["sample_desired_base_yaw_deg"] = float(math.degrees(desired_base_yaw_rad))
    selected_record["sample_used_base_yaw_rad"] = float(base_yaw_rad)
    selected_record["sample_used_base_yaw_deg"] = float(math.degrees(base_yaw_rad))
    selected_record["sample_base_yaw_clamped"] = bool(abs(base_yaw_rad - desired_base_yaw_rad) > 1e-9)
    selected_record["feasible_solutions"] = []
    selected_record["closest_solution"] = None

    closest_solution: dict[str, object] | None = None
    feasible_solution: dict[str, object] | None = None
    attempted_count = 0
    map_feasible_count = 0
    yaw_rejected_count = 0
    footprint_blocked_count = 0
    ik_reachable_count = 0
    sample_stats = {"region_cell_count": int(len(distances))}

    for sample_index, backoff_distance_m in enumerate(distances):
        local_xy = target_pb[:2] - approach_xy * float(backoff_distance_m)
        ros_map_amcl_pose, ros_map_base_link_pose = pose_from_local_base_fn(local_xy, base_yaw_rad)
        yaw_delta_rad: float | None = None
        yaw_within_limit = True
        if current_amcl_yaw_rad is not None and ros_map_amcl_pose is not None:
            yaw_delta_rad = _abs_angle_delta_rad(
                float(getattr(ros_map_amcl_pose, "yaw_rad")),
                float(current_amcl_yaw_rad),
            )
            yaw_within_limit = yaw_delta_rad <= max_yaw_delta_rad
        footprint_clear = ros_map_amcl_pose is not None and footprint_is_clear_fn(ros_map_amcl_pose)
        map_candidate_feasible = bool(yaw_within_limit and footprint_clear)
        if not yaw_within_limit:
            yaw_rejected_count += 1
        if yaw_within_limit and not footprint_clear:
            footprint_blocked_count += 1
        if map_candidate_feasible:
            map_feasible_count += 1

        sample_candidate = {
            "sample_index": int(sample_index),
            "sample_source": "simple_backoff_from_grasp",
            "map_cell_index": -1,
            "local_pb_xy": (float(local_xy[0]), float(local_xy[1])),
            "local_pb_yaw_rad": float(base_yaw_rad),
            "desired_local_pb_yaw_rad": float(desired_base_yaw_rad),
            "base_yaw_clamped": bool(abs(base_yaw_rad - desired_base_yaw_rad) > 1e-9),
            "ros_map_pose": ros_map_base_link_pose,
            "ros_map_amcl_pose": ros_map_amcl_pose,
            "distance_to_target_m": float(backoff_distance_m),
            "approach_error_deg": 0.0,
        }
        ik_attempt = attempt_ik_fn(local_xy, base_yaw_rad, selected_record)
        attempted_count += 1
        ik_feasible = _ik_attempt_is_feasible(
            ik_attempt,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_deg=orientation_tolerance_deg,
        )
        if ik_feasible:
            ik_reachable_count += 1
        feasible = bool(map_candidate_feasible and ik_feasible)
        solution_record = make_solution_record_fn(
            ik_attempt,
            sample_candidate,
            sample_stats,
            feasible,
        )
        if solution_record is None:
            continue
        solution_record["backoff_distance_m"] = float(backoff_distance_m)
        solution_record["target_average_rank"] = float(selected_record["target_average_rank"])
        solution_record["target_sample_order"] = int(selected_record["target_sample_order"])
        solution_record["reset_ee_distance_rank"] = int(selected_record["reset_ee_distance_rank"])
        solution_record["reset_ee_yaw_rank"] = int(selected_record["reset_ee_yaw_rank"])
        solution_record["map_footprint_clear"] = bool(footprint_clear)
        solution_record["amcl_yaw_within_limit"] = bool(yaw_within_limit)
        solution_record["ik_reachable"] = bool(ik_feasible)
        solution_record["desired_pb_base_link_yaw_rad"] = float(desired_base_yaw_rad)
        solution_record["desired_pb_base_link_yaw_deg"] = float(math.degrees(desired_base_yaw_rad))
        solution_record["base_yaw_clamped"] = bool(abs(base_yaw_rad - desired_base_yaw_rad) > 1e-9)
        if current_amcl_yaw_rad is not None:
            if yaw_delta_rad is not None:
                solution_record["ros_map_amcl_yaw_delta_from_current_rad"] = float(yaw_delta_rad)
                solution_record["ros_map_amcl_yaw_delta_from_current_deg"] = float(math.degrees(yaw_delta_rad))
            else:
                solution_record["ros_map_amcl_yaw_delta_from_current_rad"] = None
                solution_record["ros_map_amcl_yaw_delta_from_current_deg"] = None
        solution_record["ros_map_amcl_yaw_max_delta_from_current_deg"] = float(max_yaw_delta_deg)

        if map_candidate_feasible and (
            closest_solution is None
            or _frontmost_display_sort_key(solution_record) < _frontmost_display_sort_key(closest_solution)
        ):
            closest_solution = solution_record
        if feasible:
            feasible_solution = solution_record
            break

    if feasible_solution is not None:
        feasible_solution["selected_as_best"] = True
        feasible_solution["selected_as_closest"] = True
        selected_record["feasible_solutions"] = [feasible_solution]
        selected_record["closest_solution"] = feasible_solution
    elif closest_solution is not None:
        closest_solution["selected_as_closest"] = True
        selected_record["closest_solution"] = closest_solution

    selected_solution = feasible_solution if feasible_solution is not None else closest_solution
    selected_record["sample_attempted_count"] = int(attempted_count)
    selected_record["sample_map_feasible_count"] = int(map_feasible_count)
    selected_record["sample_yaw_rejected_count"] = int(yaw_rejected_count)
    selected_record["sample_footprint_blocked_count"] = int(footprint_blocked_count)
    selected_record["sample_ik_reachable_count"] = int(ik_reachable_count)
    selected_record["sample_success"] = bool(feasible_solution is not None)
    return SimpleBaseSampleResult(
        visualization_records=ranked_records,
        selected_record=selected_record,
        selected_solution=selected_solution,
        feasible=feasible_solution is not None,
        attempted_count=attempted_count,
        map_feasible_count=map_feasible_count,
    )
