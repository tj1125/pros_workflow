from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from src.geometry.transforms import quaternion_xyzw_to_rotation_matrix
from src.pybullet_smoke import (
    APPROACH_AGENT_ROOT,
    DEFAULT_ARM_CONFIG_PATH,
    _add_target_marker,
    _degrees_to_radians,
    _derive_side_output_path,
    _derive_topdown_output_path,
    _find_controllable_joints,
    _hold_gui_open,
    _load_arm_config,
    _load_python_dependencies,
    _load_yaml,
    _read_joint_positions,
    _render_debug_ppm,
    _resolve_input_path,
    _resolve_output_path,
)


@dataclass
class BoxObstacleSpec:
    size: tuple[float, float, float]
    position: tuple[float, float, float]
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class GoalCandidate:
    progress_ratio: float
    joint_positions_rad: tuple[float, ...]
    ee_position_xyz: tuple[float, float, float]
    ee_orientation_xyzw: tuple[float, float, float, float]
    ee_error_m: float
    ee_orientation_error_deg: float | None
    approach_axis_offset_m: float | None
    lateral_offset_m: float | None
    collision_free: bool
    closest_obstacle_distance_m: float | None


@dataclass
class OmplPlanningConfig:
    urdf_path: str
    initial_height: float
    base_orientation_euler_deg: tuple[float, float, float]
    joint_reset_deg: tuple[float, ...]
    test_target_position_base: tuple[float, float, float]
    ee_link_index: int
    position_tolerance_m: float
    planning_timeout_sec: float
    joint_bounds_deg: tuple[tuple[float, float], ...]
    allow_approximate_goal: bool = True
    approximate_goal_candidate_count: int = 21
    planning_range_rad: float = 0.35
    path_interpolation_states: int = 80
    gui: bool = False
    hold_seconds: float = 0.0
    collision_query_distance_m: float = 0.25
    collision_threshold_m: float = 0.0
    debug_render_width: int = 960
    debug_render_height: int = 720
    debug_render_camera_distance: float = 0.8
    debug_render_yaw_deg: float = 45.0
    debug_render_pitch_deg: float = -30.0
    debug_render_output_path: str | None = None
    animation_frame_sleep_sec: float = 0.03
    animation_output_dir: str | None = None
    obstacles: tuple[BoxObstacleSpec, ...] = ()


@dataclass
class OmplPlanningResult:
    pybullet_import_ok: bool = False
    ompl_import_ok: bool = False
    urdf_load_ok: bool = False
    joint_count: int = 0
    expected_joint_count: int | None = None
    reset_pose_applied: bool = False
    ik_success: bool = False
    start_state_collision_free: bool = False
    goal_state_collision_free: bool = False
    path_found: bool = False
    planned_path_collision_free: bool = False
    planning_test_passed: bool = False
    ee_position_error_m: float | None = None
    total_runtime_sec: float | None = None
    scene_build_time_sec: float | None = None
    ik_eval_time_sec: float | None = None
    planning_time_sec: float | None = None
    path_compute_time_sec: float | None = None
    path_compute_time_ms: float | None = None
    path_validation_time_sec: float | None = None
    animation_export_time_sec: float | None = None
    final_render_time_sec: float | None = None
    path_length_joint_space: float | None = None
    path_state_count: int = 0
    planned_path_joint_states_deg: list[list[float]] | None = None
    obstacle_count: int = 0
    min_distance_along_path_m: float | None = None
    closest_start_distance_m: float | None = None
    closest_goal_distance_m: float | None = None
    first_collision_state_index: int | None = None
    planning_attempt_count: int = 0
    planner_name: str = "RRTConnect"
    urdf_path: str | None = None
    arm_config_path: str | None = None
    controllable_joint_names: list[str] | None = None
    target_position_base: list[float] | None = None
    target_orientation_quaternion_xyzw: list[float] | None = None
    requested_goal_joint_positions_deg: list[float] | None = None
    ee_position_before_ik: list[float] | None = None
    ee_orientation_before_ik_quaternion_xyzw: list[float] | None = None
    ee_position_after_ik: list[float] | None = None
    ee_orientation_after_ik_quaternion_xyzw: list[float] | None = None
    start_joint_positions_deg: list[float] | None = None
    goal_joint_positions_deg: list[float] | None = None
    goal_progress_ratio: float | None = None
    ee_orientation_error_deg: float | None = None
    approach_axis_offset_m: float | None = None
    lateral_offset_m: float | None = None
    approximate_goal_used: bool = False
    approximate_goal_reason: str | None = None
    joint_limit_source: str | None = None
    gui_enabled: bool = False
    debug_render_output_path: str | None = None
    debug_render_topdown_output_path: str | None = None
    debug_render_side_output_path: str | None = None
    animation_output_dir: str | None = None
    animation_frame_count: int = 0
    failure_bucket: str | None = None
    error: str | None = None


def _vector3(value: Sequence[Any], field_name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 values.")
    return (float(value[0]), float(value[1]), float(value[2]))


def _vector4(value: Sequence[Any], field_name: str) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError(f"{field_name} must contain exactly 4 values.")
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _rotation_angle_deg_between_matrices(
    rotation_matrix_a: Any,
    rotation_matrix_b: Any,
) -> float:
    relative_rotation = rotation_matrix_a.T @ rotation_matrix_b
    trace_value = float(relative_rotation[0, 0] + relative_rotation[1, 1] + relative_rotation[2, 2])
    cosine = max(-1.0, min(1.0, 0.5 * (trace_value - 1.0)))
    return math.degrees(math.acos(cosine))


def _compute_pose_alignment_metrics(
    *,
    ee_position_xyz: Sequence[float],
    ee_orientation_xyzw: Sequence[float],
    target_position_xyz: Sequence[float],
    target_orientation_xyzw: Sequence[float] | None,
) -> tuple[float | None, float | None, float | None]:
    if target_orientation_xyzw is None:
        return None, None, None

    target_rotation = quaternion_xyzw_to_rotation_matrix(
        _vector4(target_orientation_xyzw, "target_orientation_xyzw")
    ).astype(float)
    ee_rotation = quaternion_xyzw_to_rotation_matrix(
        _vector4(ee_orientation_xyzw, "ee_orientation_xyzw")
    ).astype(float)
    target_position = [float(v) for v in target_position_xyz]
    ee_position = [float(v) for v in ee_position_xyz]
    delta_position = [
        float(ee - target)
        for ee, target in zip(ee_position, target_position)
    ]

    # Treat the target pose local +X axis as the grasp approach direction.
    approach_axis = target_rotation[:, 0]
    axis_norm = float(math.sqrt(float(approach_axis @ approach_axis)))
    if axis_norm <= 1e-8:
        return _rotation_angle_deg_between_matrices(target_rotation, ee_rotation), None, None
    approach_axis = approach_axis / axis_norm
    approach_axis_offset = float(
        (delta_position[0] * approach_axis[0])
        + (delta_position[1] * approach_axis[1])
        + (delta_position[2] * approach_axis[2])
    )
    lateral_vector = [
        delta_position[index] - approach_axis_offset * float(approach_axis[index])
        for index in range(3)
    ]
    lateral_offset = float(math.sqrt(sum(value * value for value in lateral_vector)))
    orientation_error_deg = _rotation_angle_deg_between_matrices(target_rotation, ee_rotation)
    return orientation_error_deg, approach_axis_offset, lateral_offset


def _joint_bounds_deg(value: Sequence[Any], field_name: str) -> tuple[tuple[float, float], ...]:
    bounds: list[tuple[float, float]] = []
    for index, pair in enumerate(value):
        if len(pair) != 2:
            raise ValueError(f"{field_name}[{index}] must contain exactly 2 values.")
        lower = float(pair[0])
        upper = float(pair[1])
        if lower > upper:
            raise ValueError(f"{field_name}[{index}] lower bound cannot exceed upper bound.")
        bounds.append((lower, upper))
    return tuple(bounds)


def _load_ompl_dependencies() -> tuple[Any, Any]:
    try:
        from ompl import base as ob
        from ompl import geometric as og
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Missing OMPL Python bindings. Please run inside the project Docker image "
            "with OMPL installed before using the planning stage."
        ) from exc
    return ob, og


def _load_obstacles(payload: dict[str, Any]) -> tuple[BoxObstacleSpec, ...]:
    raw_obstacles = payload.get("obstacles", [])
    obstacles: list[BoxObstacleSpec] = []
    for index, raw_spec in enumerate(raw_obstacles):
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Obstacle #{index} must be a mapping.")
        obstacles.append(
            BoxObstacleSpec(
                size=_vector3(raw_spec["size"], f"obstacles[{index}].size"),
                position=_vector3(raw_spec["position"], f"obstacles[{index}].position"),
                rgba=_vector4(
                    raw_spec.get("rgba", [0.85, 0.2, 0.2, 0.65]),
                    f"obstacles[{index}].rgba",
                ),
            )
        )
    return tuple(obstacles)


def load_planning_config(config_path: Path) -> OmplPlanningConfig:
    payload = _load_yaml(config_path)
    debug_render_output = payload.get("debug_render_output_path")
    animation_output_dir = payload.get("animation_output_dir")
    obstacles = _load_obstacles(payload)
    if not obstacles:
        raise ValueError("At least one manual box obstacle is required for OMPL planning.")

    return OmplPlanningConfig(
        urdf_path=str(_resolve_input_path(str(payload["urdf_path"]), config_path)),
        initial_height=float(payload["initial_height"]),
        base_orientation_euler_deg=_vector3(payload["base_orientation_euler_deg"], "base_orientation_euler_deg"),
        joint_reset_deg=tuple(float(v) for v in payload["joint_reset_deg"]),
        test_target_position_base=_vector3(payload["test_target_position_base"], "test_target_position_base"),
        ee_link_index=int(payload["ee_link_index"]),
        position_tolerance_m=float(payload["position_tolerance_m"]),
        planning_timeout_sec=float(payload["planning_timeout_sec"]),
        joint_bounds_deg=_joint_bounds_deg(payload["joint_bounds_deg"], "joint_bounds_deg"),
        allow_approximate_goal=bool(payload.get("allow_approximate_goal", True)),
        approximate_goal_candidate_count=int(payload.get("approximate_goal_candidate_count", 21)),
        planning_range_rad=float(payload.get("planning_range_rad", 0.35)),
        path_interpolation_states=int(payload.get("path_interpolation_states", 80)),
        gui=bool(payload.get("gui", False)),
        hold_seconds=float(payload.get("hold_seconds", 0.0)),
        collision_query_distance_m=float(payload.get("collision_query_distance_m", 0.25)),
        collision_threshold_m=float(payload.get("collision_threshold_m", 0.0)),
        debug_render_width=int(payload.get("debug_render_width", 960)),
        debug_render_height=int(payload.get("debug_render_height", 720)),
        debug_render_camera_distance=float(payload.get("debug_render_camera_distance", 0.8)),
        debug_render_yaw_deg=float(payload.get("debug_render_yaw_deg", 45.0)),
        debug_render_pitch_deg=float(payload.get("debug_render_pitch_deg", -30.0)),
        debug_render_output_path=(str(_resolve_output_path(str(debug_render_output), config_path)) if debug_render_output else None),
        animation_frame_sleep_sec=float(payload.get("animation_frame_sleep_sec", 0.03)),
        animation_output_dir=(
            str(_resolve_output_path(str(animation_output_dir), config_path))
            if animation_output_dir
            else None
        ),
        obstacles=obstacles,
    )


def _set_joint_positions_direct(
    p: Any,
    robot_id: int,
    joint_ids: Sequence[int],
    joint_positions_rad: Sequence[float],
) -> None:
    for joint_id, joint_position in zip(joint_ids, joint_positions_rad):
        p.resetJointState(robot_id, joint_id, targetValue=float(joint_position), targetVelocity=0.0)


def _query_robot_obstacle_distances(
    p: Any,
    robot_id: int,
    obstacle_ids: Sequence[int],
    *,
    query_distance: float,
    collision_threshold: float,
) -> tuple[float | None, bool]:
    min_distance: float | None = None
    in_collision = False
    for obstacle_id in obstacle_ids:
        closest_points = p.getClosestPoints(robot_id, obstacle_id, distance=float(query_distance))
        if not closest_points:
            continue
        for point in closest_points:
            distance = float(point[8])
            if min_distance is None or distance < min_distance:
                min_distance = distance
            if distance <= float(collision_threshold):
                in_collision = True
                return min_distance, True
    return min_distance, in_collision


def _create_box_obstacles(p: Any, obstacles: Sequence[BoxObstacleSpec]) -> list[int]:
    obstacle_ids: list[int] = []
    for obstacle in obstacles:
        half_extents = [size * 0.5 for size in obstacle.size]
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=list(obstacle.rgba),
        )
        obstacle_ids.append(
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=list(obstacle.position),
            )
        )
    return obstacle_ids


def _add_anchor_marker(
    p: Any,
    anchor_position: Sequence[float],
    *,
    rgba: Sequence[float] = (0.15, 0.55, 0.95, 0.95),
) -> None:
    marker_shape = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.018,
        rgbaColor=list(rgba),
    )
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=marker_shape,
        basePosition=list(anchor_position),
    )


def _add_debug_axes(
    p: Any,
    origin_xyz: Sequence[float],
    *,
    axis_length: float = 0.12,
    axis_width: float = 2.0,
    life_time_sec: float = 0.0,
    label: str | None = None,
) -> None:
    origin = [float(v) for v in origin_xyz]
    axis_vectors = (
        ([axis_length, 0.0, 0.0], [1.0, 0.15, 0.15]),
        ([0.0, axis_length, 0.0], [0.15, 1.0, 0.15]),
        ([0.0, 0.0, axis_length], [0.15, 0.35, 1.0]),
    )
    for delta_xyz, color_rgb in axis_vectors:
        end_point = [
            origin[0] + float(delta_xyz[0]),
            origin[1] + float(delta_xyz[1]),
            origin[2] + float(delta_xyz[2]),
        ]
        p.addUserDebugLine(
            origin,
            end_point,
            lineColorRGB=color_rgb,
            lineWidth=float(axis_width),
            lifeTime=float(life_time_sec),
        )
    if label:
        p.addUserDebugText(
            str(label),
            [
                origin[0],
                origin[1],
                origin[2] + float(axis_length) * 1.1,
            ],
            textColorRGB=[1.0, 1.0, 1.0],
            textSize=1.2,
            lifeTime=float(life_time_sec),
        )


def get_reset_end_effector_position(config_path: Path) -> tuple[float, float, float]:
    planning_config = load_planning_config(config_path)
    arm_config = _load_arm_config()
    _, p, pybullet_data = _load_python_dependencies()
    expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])
    client_id: int | None = None

    try:
        client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.8)
        p.loadURDF("plane.urdf")

        base_orientation_rad = [math.radians(v) for v in planning_config.base_orientation_euler_deg]
        base_orientation_xyzw = p.getQuaternionFromEuler(base_orientation_rad)
        robot_id = p.loadURDF(
            planning_config.urdf_path,
            useFixedBase=True,
            basePosition=[0.0, 0.0, planning_config.initial_height],
            baseOrientation=base_orientation_xyzw,
        )
        controllable_joint_ids, _ = _find_controllable_joints(p, robot_id, expected_joint_count)
        joint_reset_rad = _degrees_to_radians(planning_config.joint_reset_deg)
        _set_joint_positions_direct(p, robot_id, controllable_joint_ids, joint_reset_rad)
        p.performCollisionDetection()

        if planning_config.ee_link_index >= p.getNumJoints(robot_id):
            raise ValueError(f"ee_link_index {planning_config.ee_link_index} is out of range.")
        ee_position = p.getLinkState(robot_id, planning_config.ee_link_index, computeForwardKinematics=True)[0]
        return (float(ee_position[0]), float(ee_position[1]), float(ee_position[2]))
    finally:
        if client_id is not None:
            try:
                p.disconnect(client_id)
            except Exception:
                pass


def _joint_limits_from_planning_config(
    planning_config: OmplPlanningConfig,
    expected_joint_count: int,
) -> tuple[list[float], list[float]]:
    if len(planning_config.joint_bounds_deg) != expected_joint_count:
        raise ValueError(
            "joint_bounds_deg length does not match controllable joint count: "
            f"{len(planning_config.joint_bounds_deg)} vs {expected_joint_count}"
        )
    lower_bounds = [math.radians(lower) for lower, _ in planning_config.joint_bounds_deg]
    upper_bounds = [math.radians(upper) for _, upper in planning_config.joint_bounds_deg]
    return lower_bounds, upper_bounds


def _is_within_bounds(values: Sequence[float], lower_bounds: Sequence[float], upper_bounds: Sequence[float]) -> bool:
    return all(lower <= value <= upper for value, lower, upper in zip(values, lower_bounds, upper_bounds))


def _extract_state_values(state: Any, dimension: int) -> list[float]:
    return [float(state[index]) for index in range(dimension)]


def _evaluate_joint_state(
    p: Any,
    robot_id: int,
    obstacle_ids: Sequence[int],
    joint_ids: Sequence[int],
    joint_positions: Sequence[float],
    *,
    query_distance: float,
    collision_threshold: float,
) -> tuple[float | None, bool]:
    _set_joint_positions_direct(p, robot_id, joint_ids, joint_positions)
    p.performCollisionDetection()
    return _query_robot_obstacle_distances(
        p,
        robot_id,
        obstacle_ids,
        query_distance=query_distance,
        collision_threshold=collision_threshold,
    )


def _extract_solution_states(path: Any, dimension: int) -> list[list[float]]:
    return [_extract_state_values(path.getState(index), dimension) for index in range(path.getStateCount())]


def _clip_joint_positions(
    joint_positions: Sequence[float],
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
) -> list[float]:
    return [
        min(max(float(value), float(lower)), float(upper))
        for value, lower, upper in zip(joint_positions, lower_bounds, upper_bounds)
    ]


def _interpolate_joint_positions(
    start_joint_positions: Sequence[float],
    goal_joint_positions: Sequence[float],
    progress_ratio: float,
) -> list[float]:
    alpha = float(progress_ratio)
    return [
        float(start + alpha * (goal - start))
        for start, goal in zip(start_joint_positions, goal_joint_positions)
    ]


def _evaluate_goal_candidate(
    p: Any,
    robot_id: int,
    obstacle_ids: Sequence[int],
    joint_ids: Sequence[int],
    joint_positions_rad: Sequence[float],
    *,
    ee_link_index: int,
    requested_target_position_base: Sequence[float],
    requested_target_orientation_xyzw: Sequence[float] | None,
    progress_ratio: float,
    query_distance: float,
    collision_threshold: float,
) -> GoalCandidate:
    closest_distance, in_collision = _evaluate_joint_state(
        p,
        robot_id,
        obstacle_ids,
        joint_ids,
        joint_positions_rad,
        query_distance=query_distance,
        collision_threshold=collision_threshold,
    )
    ee_link_state = p.getLinkState(robot_id, ee_link_index, computeForwardKinematics=True)
    ee_position = ee_link_state[0]
    ee_orientation = ee_link_state[1]
    ee_position_xyz = (float(ee_position[0]), float(ee_position[1]), float(ee_position[2]))
    ee_orientation_xyzw = (
        float(ee_orientation[0]),
        float(ee_orientation[1]),
        float(ee_orientation[2]),
        float(ee_orientation[3]),
    )
    ee_error_m = float(math.dist(ee_position_xyz, tuple(float(v) for v in requested_target_position_base)))
    ee_orientation_error_deg, approach_axis_offset_m, lateral_offset_m = _compute_pose_alignment_metrics(
        ee_position_xyz=ee_position_xyz,
        ee_orientation_xyzw=ee_orientation_xyzw,
        target_position_xyz=requested_target_position_base,
        target_orientation_xyzw=requested_target_orientation_xyzw,
    )
    return GoalCandidate(
        progress_ratio=float(progress_ratio),
        joint_positions_rad=tuple(float(v) for v in joint_positions_rad),
        ee_position_xyz=ee_position_xyz,
        ee_orientation_xyzw=ee_orientation_xyzw,
        ee_error_m=ee_error_m,
        ee_orientation_error_deg=ee_orientation_error_deg,
        approach_axis_offset_m=approach_axis_offset_m,
        lateral_offset_m=lateral_offset_m,
        collision_free=not in_collision,
        closest_obstacle_distance_m=closest_distance,
    )


def _record_goal_candidate(result: OmplPlanningResult, candidate: GoalCandidate) -> None:
    result.goal_joint_positions_deg = [math.degrees(value) for value in candidate.joint_positions_rad]
    result.ee_position_after_ik = [float(v) for v in candidate.ee_position_xyz]
    result.ee_orientation_after_ik_quaternion_xyzw = [float(v) for v in candidate.ee_orientation_xyzw]
    result.ee_position_error_m = float(candidate.ee_error_m)
    result.ee_orientation_error_deg = candidate.ee_orientation_error_deg
    result.approach_axis_offset_m = candidate.approach_axis_offset_m
    result.lateral_offset_m = candidate.lateral_offset_m
    result.closest_goal_distance_m = candidate.closest_obstacle_distance_m
    result.goal_state_collision_free = bool(candidate.collision_free)
    result.goal_progress_ratio = float(candidate.progress_ratio)


def _validate_path_states(
    p: Any,
    robot_id: int,
    obstacle_ids: Sequence[int],
    joint_ids: Sequence[int],
    path_states: Sequence[Sequence[float]],
    *,
    query_distance: float,
    collision_threshold: float,
) -> tuple[float | None, int | None, bool]:
    min_distance_along_path: float | None = None
    first_collision_state_index: int | None = None
    for index, state in enumerate(path_states):
        min_distance, in_collision = _evaluate_joint_state(
            p,
            robot_id,
            obstacle_ids,
            joint_ids,
            state,
            query_distance=query_distance,
            collision_threshold=collision_threshold,
        )
        if min_distance is not None:
            if min_distance_along_path is None or min_distance < min_distance_along_path:
                min_distance_along_path = min_distance
        if in_collision:
            first_collision_state_index = index
            break
    return min_distance_along_path, first_collision_state_index, first_collision_state_index is None


def _animate_planned_path(
    p: Any,
    np: Any,
    robot_id: int,
    joint_ids: Sequence[int],
    path_states: Sequence[Sequence[float]],
    *,
    gui_enabled: bool,
    frame_sleep_sec: float,
    animation_output_dir: Path | None,
    render_width: int,
    render_height: int,
    render_camera_target_position: Sequence[float],
    render_camera_distance: float,
    render_camera_yaw_deg: float,
    render_camera_pitch_deg: float,
) -> int:
    animation_frame_count = 0
    if animation_output_dir is not None:
        animation_output_dir.mkdir(parents=True, exist_ok=True)

    for index, state in enumerate(path_states):
        _set_joint_positions_direct(p, robot_id, joint_ids, state)
        p.performCollisionDetection()
        if animation_output_dir is not None:
            _render_debug_ppm(
                p,
                np,
                animation_output_dir / f"frame_{index:04d}.ppm",
                width=render_width,
                height=render_height,
                camera_target_position=render_camera_target_position,
                camera_distance=render_camera_distance,
                camera_yaw_deg=render_camera_yaw_deg,
                camera_pitch_deg=render_camera_pitch_deg,
            )
            animation_frame_count += 1
        if gui_enabled:
            time.sleep(max(float(frame_sleep_sec), 0.0))
    return animation_frame_count


def run_ompl_planning_test(
    config_path: Path,
    *,
    gui_override: bool = False,
    hold_seconds_override: float | None = None,
    save_debug_ppm_override: str | None = None,
    save_animation_dir_override: str | None = None,
    obstacle_specs_override: Sequence[BoxObstacleSpec] | None = None,
    target_position_override: Sequence[float] | None = None,
    target_orientation_override_xyzw: Sequence[float] | None = None,
    render_camera_target_position_override: Sequence[float] | None = None,
    render_camera_distance_override: float | None = None,
    anchor_marker_position_override: Sequence[float] | None = None,
) -> OmplPlanningResult:
    result = OmplPlanningResult(arm_config_path=str(DEFAULT_ARM_CONFIG_PATH))
    planning_config: OmplPlanningConfig | None = None
    gui_enabled = False
    hold_seconds = 0.0
    time_step = 1.0 / 240.0
    total_runtime_start_time = time.perf_counter()

    try:
        planning_config = load_planning_config(config_path)
        if gui_override:
            planning_config.gui = True
        if hold_seconds_override is not None:
            planning_config.hold_seconds = float(hold_seconds_override)
        if save_debug_ppm_override is not None:
            planning_config.debug_render_output_path = str(Path(save_debug_ppm_override).resolve())
        if save_animation_dir_override is not None:
            planning_config.animation_output_dir = str(Path(save_animation_dir_override).resolve())
        if planning_config.gui:
            planning_config.debug_render_output_path = None
            planning_config.animation_output_dir = None
        arm_config = _load_arm_config()
    except Exception as exc:
        result.failure_bucket = "config"
        result.error = str(exc)
        return result

    requested_target_position = (
        _vector3(target_position_override, "target_position_override")
        if target_position_override is not None
        else planning_config.test_target_position_base
    )
    requested_target_orientation_xyzw = (
        _vector4(target_orientation_override_xyzw, "target_orientation_override_xyzw")
        if target_orientation_override_xyzw is not None
        else None
    )
    result.urdf_path = planning_config.urdf_path
    result.target_position_base = list(requested_target_position)
    if requested_target_orientation_xyzw is not None:
        result.target_orientation_quaternion_xyzw = [float(v) for v in requested_target_orientation_xyzw]
    result.gui_enabled = bool(planning_config.gui)
    result.debug_render_output_path = planning_config.debug_render_output_path
    result.animation_output_dir = planning_config.animation_output_dir
    active_obstacles = tuple(obstacle_specs_override) if obstacle_specs_override is not None else planning_config.obstacles
    result.obstacle_count = len(active_obstacles)
    render_camera_target_position = (
        [float(v) for v in render_camera_target_position_override]
        if render_camera_target_position_override is not None
        else list(requested_target_position)
    )
    render_camera_distance = (
        float(render_camera_distance_override)
        if render_camera_distance_override is not None
        else float(planning_config.debug_render_camera_distance)
    )
    if planning_config.debug_render_output_path:
        result.debug_render_topdown_output_path = str(
            _derive_topdown_output_path(Path(planning_config.debug_render_output_path))
        )
        result.debug_render_side_output_path = str(
            _derive_side_output_path(Path(planning_config.debug_render_output_path))
        )

    try:
        np, p, pybullet_data = _load_python_dependencies()
    except Exception as exc:
        result.failure_bucket = "environment"
        result.error = str(exc)
        return result

    result.pybullet_import_ok = True

    expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])
    result.expected_joint_count = expected_joint_count

    client_id: int | None = None
    try:
        scene_build_start_time = time.perf_counter()
        connection_mode = p.GUI if planning_config.gui else p.DIRECT
        client_id = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        time_step = float(arm_config["pybullet"]["time_step"])
        gui_enabled = bool(planning_config.gui)
        hold_seconds = float(planning_config.hold_seconds)
        p.setGravity(0.0, 0.0, -9.8)
        p.setTimeStep(time_step)
        p.setPhysicsEngineParameter(
            fixedTimeStep=time_step,
            numSolverIterations=100,
            numSubSteps=10,
        )
        p.setRealTimeSimulation(0)
        if gui_enabled:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.loadURDF("plane.urdf")

        base_orientation_rad = [math.radians(v) for v in planning_config.base_orientation_euler_deg]
        base_orientation_xyzw = p.getQuaternionFromEuler(base_orientation_rad)
        robot_id = p.loadURDF(
            planning_config.urdf_path,
            useFixedBase=True,
            basePosition=[0.0, 0.0, planning_config.initial_height],
            baseOrientation=base_orientation_xyzw,
        )
        if gui_enabled:
            _add_debug_axes(
                p,
                [0.0, 0.0, 0.0],
                axis_length=0.18,
                axis_width=2.4,
                label="world",
            )
            _add_debug_axes(
                p,
                [0.0, 0.0, planning_config.initial_height],
                axis_length=0.14,
                axis_width=2.2,
                label="base",
            )
        result.urdf_load_ok = True

        controllable_joint_ids, controllable_joint_names = _find_controllable_joints(
            p,
            robot_id,
            expected_joint_count,
        )
        result.joint_count = len(controllable_joint_ids)
        result.controllable_joint_names = controllable_joint_names
        if result.joint_count != expected_joint_count:
            result.failure_bucket = "model"
            result.error = (
                f"Expected {expected_joint_count} controllable joints, "
                f"found {result.joint_count}."
            )
            return result

        joint_reset_rad = _degrees_to_radians(planning_config.joint_reset_deg)
        if len(joint_reset_rad) != len(controllable_joint_ids):
            result.failure_bucket = "config"
            result.error = (
                "joint_reset_deg length does not match controllable joint count: "
                f"{len(joint_reset_rad)} vs {len(controllable_joint_ids)}"
            )
            return result

        _set_joint_positions_direct(p, robot_id, controllable_joint_ids, joint_reset_rad)
        p.performCollisionDetection()
        actual_reset_positions = _read_joint_positions(p, robot_id, controllable_joint_ids)
        start_joint_positions = list(actual_reset_positions)
        result.start_joint_positions_deg = [math.degrees(value) for value in start_joint_positions]
        result.reset_pose_applied = all(
            abs(expected - actual) <= 1e-6
            for expected, actual in zip(joint_reset_rad, actual_reset_positions)
        )
        if not result.reset_pose_applied:
            result.failure_bucket = "model"
            result.error = "Failed to apply the configured reset pose accurately."
            return result

        if planning_config.ee_link_index >= p.getNumJoints(robot_id):
            result.failure_bucket = "config"
            result.error = f"ee_link_index {planning_config.ee_link_index} is out of range."
            return result

        ee_before_state = p.getLinkState(robot_id, planning_config.ee_link_index, computeForwardKinematics=True)
        ee_before = ee_before_state[0]
        result.ee_position_before_ik = [float(v) for v in ee_before]
        result.ee_orientation_before_ik_quaternion_xyzw = [float(v) for v in ee_before_state[1]]

        obstacle_ids = _create_box_obstacles(p, active_obstacles)
        _add_target_marker(p, requested_target_position)
        if anchor_marker_position_override is not None:
            _add_anchor_marker(p, anchor_marker_position_override)
            if gui_enabled:
                _add_debug_axes(
                    p,
                    anchor_marker_position_override,
                    axis_length=0.12,
                    axis_width=2.0,
                    label="ee_anchor",
                )
        if gui_enabled:
            p.resetDebugVisualizerCamera(
                cameraDistance=render_camera_distance,
                cameraYaw=float(planning_config.debug_render_yaw_deg),
                cameraPitch=float(planning_config.debug_render_pitch_deg),
                cameraTargetPosition=render_camera_target_position,
            )
        result.scene_build_time_sec = float(time.perf_counter() - scene_build_start_time)

        closest_start_distance, start_in_collision = _query_robot_obstacle_distances(
            p,
            robot_id,
            obstacle_ids,
            query_distance=float(planning_config.collision_query_distance_m),
            collision_threshold=float(planning_config.collision_threshold_m),
        )
        result.closest_start_distance_m = closest_start_distance
        result.start_state_collision_free = not start_in_collision
        if start_in_collision:
            result.failure_bucket = "geometry"
            result.error = "The reset pose is already in collision with a manual obstacle."
            return result

        lower_bounds, upper_bounds = _joint_limits_from_planning_config(planning_config, expected_joint_count)
        result.joint_limit_source = "planning_config"
        if not _is_within_bounds(start_joint_positions, lower_bounds, upper_bounds):
            result.failure_bucket = "config"
            result.error = "Reset pose is outside joint_bounds_deg in pybullet_ompl.yaml."
            return result

        ik_eval_start_time = time.perf_counter()
        ik_kwargs: dict[str, Any] = {
            "targetPosition": list(requested_target_position),
        }
        if requested_target_orientation_xyzw is not None:
            ik_kwargs["targetOrientation"] = list(requested_target_orientation_xyzw)
        ik_solution = p.calculateInverseKinematics(
            robot_id,
            planning_config.ee_link_index,
            **ik_kwargs,
        )
        if len(ik_solution) < len(controllable_joint_ids):
            result.failure_bucket = "geometry"
            result.error = "IK returned fewer joint values than required."
            return result

        requested_goal_joint_positions = [float(v) for v in ik_solution[: len(controllable_joint_ids)]]
        result.requested_goal_joint_positions_deg = [
            math.degrees(value) for value in requested_goal_joint_positions
        ]
        clipped_goal_joint_positions = _clip_joint_positions(
            requested_goal_joint_positions,
            lower_bounds,
            upper_bounds,
        )

        candidate_count = max(int(planning_config.approximate_goal_candidate_count), 2)
        progress_values = (
            [float(v) for v in np.linspace(1.0, 0.0, candidate_count)]
            if planning_config.allow_approximate_goal
            else [1.0]
        )

        goal_candidates: list[GoalCandidate] = []
        seen_joint_states: set[tuple[float, ...]] = set()
        for progress_ratio in progress_values:
            candidate_joint_positions = _interpolate_joint_positions(
                start_joint_positions,
                clipped_goal_joint_positions,
                progress_ratio,
            )
            state_key = tuple(round(value, 8) for value in candidate_joint_positions)
            if state_key in seen_joint_states:
                continue
            seen_joint_states.add(state_key)
            goal_candidates.append(
                _evaluate_goal_candidate(
                    p,
                    robot_id,
                    obstacle_ids,
                    controllable_joint_ids,
                    candidate_joint_positions,
                    ee_link_index=planning_config.ee_link_index,
                    requested_target_position_base=requested_target_position,
                    requested_target_orientation_xyzw=requested_target_orientation_xyzw,
                    progress_ratio=progress_ratio,
                    query_distance=float(planning_config.collision_query_distance_m),
                    collision_threshold=float(planning_config.collision_threshold_m),
                )
            )

        result.ik_eval_time_sec = float(time.perf_counter() - ik_eval_start_time)
        if not goal_candidates:
            result.failure_bucket = "geometry"
            result.error = "Failed to build any IK goal candidates."
            return result

        exact_candidate = goal_candidates[0]
        raw_goal_within_bounds = _is_within_bounds(
            requested_goal_joint_positions,
            lower_bounds,
            upper_bounds,
        )
        exact_candidate_is_precise = (
            exact_candidate.collision_free
            and exact_candidate.ee_error_m <= planning_config.position_tolerance_m
            and exact_candidate.progress_ratio >= 1.0 - 1e-6
        )
        result.ik_success = exact_candidate_is_precise

        collision_free_candidates = [candidate for candidate in goal_candidates if candidate.collision_free]
        if not collision_free_candidates:
            _record_goal_candidate(result, exact_candidate)
            result.failure_bucket = "geometry"
            if not raw_goal_within_bounds:
                result.error = (
                    "IK goal exceeded joint_bounds_deg, and no collision-free approximate goal "
                    "candidate could be found."
                )
            else:
                result.error = "No collision-free IK goal candidate could be found."
            return result

        planning_goal_candidates = sorted(
            collision_free_candidates,
            key=lambda candidate: candidate.progress_ratio,
            reverse=True,
        )
        _record_goal_candidate(result, planning_goal_candidates[0])
        if not exact_candidate_is_precise:
            result.approximate_goal_used = True
            if not raw_goal_within_bounds:
                result.approximate_goal_reason = (
                    "Raw IK goal exceeded configured joint bounds, so planning falls back to the "
                    "closest in-bounds candidate."
                )
            elif not exact_candidate.collision_free:
                result.approximate_goal_reason = (
                    "Exact IK goal collided with obstacles, so planning falls back to the nearest "
                    "collision-free candidate."
                )
            else:
                result.approximate_goal_reason = (
                    f"Exact IK goal missed the {planning_config.position_tolerance_m:.3f} m tolerance, "
                    "so planning falls back to the nearest reachable candidate."
                )

        try:
            ob, og = _load_ompl_dependencies()
        except Exception as exc:
            result.failure_bucket = "environment"
            result.error = str(exc)
            return result

        result.ompl_import_ok = True
        _set_joint_positions_direct(p, robot_id, controllable_joint_ids, start_joint_positions)
        p.performCollisionDetection()

        dimension = len(controllable_joint_ids)
        space = ob.RealVectorStateSpace(dimension)
        bounds = ob.RealVectorBounds(dimension)
        for index in range(dimension):
            bounds.setLow(index, lower_bounds[index])
            bounds.setHigh(index, upper_bounds[index])
        space.setBounds(bounds)

        si = ob.SpaceInformation(space)

        def is_state_valid(state: Any) -> bool:
            joint_values = _extract_state_values(state, dimension)
            if not _is_within_bounds(joint_values, lower_bounds, upper_bounds):
                return False
            _, in_collision = _evaluate_joint_state(
                p,
                robot_id,
                obstacle_ids,
                controllable_joint_ids,
                joint_values,
                query_distance=float(planning_config.collision_query_distance_m),
                collision_threshold=float(planning_config.collision_threshold_m),
            )
            return not in_collision

        si.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))
        si.setup()

        start = ob.State(space)
        for index, value in enumerate(start_joint_positions):
            start[index] = float(value)
        result.planning_time_sec = 0.0
        result.path_compute_time_sec = 0.0
        result.path_compute_time_ms = 0.0
        result.path_validation_time_sec = 0.0

        planning_success_candidate: GoalCandidate | None = None
        planning_failure_message = (
            f"OMPL {result.planner_name} failed to find a path to any reachable goal candidate within "
            f"{planning_config.planning_timeout_sec:.3f} seconds."
        )

        for candidate in planning_goal_candidates:
            _set_joint_positions_direct(p, robot_id, controllable_joint_ids, start_joint_positions)
            p.performCollisionDetection()

            goal = ob.State(space)
            for index, value in enumerate(candidate.joint_positions_rad):
                goal[index] = float(value)

            pdef = ob.ProblemDefinition(si)
            pdef.setStartAndGoalStates(start, goal)
            planner = og.RRTConnect(si)
            if hasattr(planner, "setRange") and planning_config.planning_range_rad > 0.0:
                planner.setRange(float(planning_config.planning_range_rad))
            planner.setProblemDefinition(pdef)
            planner.setup()

            result.planning_attempt_count += 1
            planning_start_time = time.perf_counter()
            solved = planner.solve(float(planning_config.planning_timeout_sec))
            elapsed_sec = float(time.perf_counter() - planning_start_time)
            result.planning_time_sec += elapsed_sec
            result.path_compute_time_sec += elapsed_sec
            result.path_compute_time_ms = result.path_compute_time_sec * 1000.0
            if not solved:
                continue

            path = pdef.getSolutionPath()
            target_state_count = max(int(planning_config.path_interpolation_states), int(path.getStateCount()))
            if target_state_count > path.getStateCount():
                path.interpolate(target_state_count)
            path_states = _extract_solution_states(path, dimension)

            path_validation_start_time = time.perf_counter()
            min_distance_along_path, first_collision_state_index, path_collision_free = _validate_path_states(
                p,
                robot_id,
                obstacle_ids,
                controllable_joint_ids,
                path_states,
                query_distance=float(planning_config.collision_query_distance_m),
                collision_threshold=float(planning_config.collision_threshold_m),
            )
            result.path_validation_time_sec += float(time.perf_counter() - path_validation_start_time)
            if not path_collision_free:
                planning_failure_message = (
                    f"OMPL reached a candidate goal at progress {candidate.progress_ratio:.3f}, but the "
                    f"sampled path collided at state {first_collision_state_index}."
                )
                continue

            planning_success_candidate = candidate
            result.path_found = True
            result.path_length_joint_space = float(path.length())
            result.path_state_count = len(path_states)
            result.planned_path_joint_states_deg = [
                [math.degrees(value) for value in state]
                for state in path_states
            ]
            result.min_distance_along_path_m = min_distance_along_path
            result.first_collision_state_index = first_collision_state_index
            result.planned_path_collision_free = True
            _record_goal_candidate(result, candidate)
            if not exact_candidate_is_precise or candidate.progress_ratio < 1.0 - 1e-6:
                result.approximate_goal_used = True
                if result.approximate_goal_reason is None:
                    result.approximate_goal_reason = (
                        "Planning succeeded by backing off from the requested target to the closest "
                        "reachable goal candidate."
                    )
            else:
                result.approximate_goal_used = False
                result.approximate_goal_reason = None

            animation_export_start_time = time.perf_counter()
            result.animation_frame_count = _animate_planned_path(
                p,
                np,
                robot_id,
                controllable_joint_ids,
                path_states,
                gui_enabled=gui_enabled,
                frame_sleep_sec=float(planning_config.animation_frame_sleep_sec),
                animation_output_dir=(
                    Path(planning_config.animation_output_dir)
                    if planning_config.animation_output_dir
                    else None
                ),
                render_width=int(planning_config.debug_render_width),
                render_height=int(planning_config.debug_render_height),
                render_camera_target_position=render_camera_target_position,
                render_camera_distance=render_camera_distance,
                render_camera_yaw_deg=float(planning_config.debug_render_yaw_deg),
                render_camera_pitch_deg=float(planning_config.debug_render_pitch_deg),
            )
            result.animation_export_time_sec = float(time.perf_counter() - animation_export_start_time)
            break

        if planning_success_candidate is None:
            result.failure_bucket = "planning"
            result.error = planning_failure_message
            return result

        result.planning_test_passed = (
            result.pybullet_import_ok
            and result.ompl_import_ok
            and result.urdf_load_ok
            and result.reset_pose_applied
            and result.start_state_collision_free
            and result.path_found
            and result.planned_path_collision_free
            and (result.ik_success or result.approximate_goal_used)
        )
        if not result.planning_test_passed and result.failure_bucket is None:
            result.failure_bucket = "unknown"
            result.error = "Planning test failed without a more specific failure bucket."
        return result
    except Exception as exc:  # pragma: no cover - defensive reporting
        if result.failure_bucket is None:
            result.failure_bucket = "runtime"
        result.error = str(exc)
        return result
    finally:
        if client_id is not None:
            try:
                final_render_start_time = time.perf_counter()
                if planning_config is not None and planning_config.debug_render_output_path:
                    debug_output_path = Path(planning_config.debug_render_output_path)
                    _render_debug_ppm(
                        p,
                        np,
                        debug_output_path,
                        width=int(planning_config.debug_render_width),
                        height=int(planning_config.debug_render_height),
                        camera_target_position=render_camera_target_position,
                        camera_distance=render_camera_distance,
                        camera_yaw_deg=float(planning_config.debug_render_yaw_deg),
                        camera_pitch_deg=float(planning_config.debug_render_pitch_deg),
                    )
                    _render_debug_ppm(
                        p,
                        np,
                        _derive_topdown_output_path(debug_output_path),
                        width=int(planning_config.debug_render_width),
                        height=int(planning_config.debug_render_height),
                        camera_target_position=render_camera_target_position,
                        camera_distance=max(0.55, render_camera_distance * 0.85),
                        camera_yaw_deg=0.0,
                        camera_pitch_deg=-89.0,
                    )
                    _render_debug_ppm(
                        p,
                        np,
                        _derive_side_output_path(debug_output_path),
                        width=int(planning_config.debug_render_width),
                        height=int(planning_config.debug_render_height),
                        camera_target_position=render_camera_target_position,
                        camera_distance=max(0.55, render_camera_distance * 0.9),
                        camera_yaw_deg=90.0,
                        camera_pitch_deg=-12.0,
                    )
                result.final_render_time_sec = float(time.perf_counter() - final_render_start_time)
                _hold_gui_open(p, gui_enabled, hold_seconds, time_step)
                p.disconnect(client_id)
            except Exception:
                pass
        result.total_runtime_sec = float(time.perf_counter() - total_runtime_start_time)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a PyBullet + OMPL planning test with manual box obstacles for Approach_Agent."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=APPROACH_AGENT_ROOT / "configs" / "pybullet_ompl.yaml",
        help="Path to the OMPL planning YAML config.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open PyBullet in GUI mode for manual visualization.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=None,
        help="Keep the GUI window open for a few seconds before exit.",
    )
    parser.add_argument(
        "--save-debug-ppm",
        type=Path,
        default=None,
        help="Save a headless TinyRenderer debug image to a .ppm file.",
    )
    parser.add_argument(
        "--save-animation-dir",
        type=Path,
        default=None,
        help="Save a sequence of headless TinyRenderer path frames to a directory.",
    )
    args = parser.parse_args(argv)

    result = run_ompl_planning_test(
        args.config.resolve(),
        gui_override=bool(args.gui),
        hold_seconds_override=args.hold_seconds,
        save_debug_ppm_override=(str(args.save_debug_ppm.resolve()) if args.save_debug_ppm else None),
        save_animation_dir_override=(str(args.save_animation_dir.resolve()) if args.save_animation_dir else None),
    )
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0 if result.planning_test_passed else 1
