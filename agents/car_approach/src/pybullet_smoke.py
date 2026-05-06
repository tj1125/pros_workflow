from __future__ import annotations

import argparse
import ast
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


APPROACH_AGENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARM_CONFIG_PATH = REPO_ROOT / "tools" / "car_control" / "src" / "arm_control_pkg" / "config" / "arm_config.yaml"
MIMIC_JOINT_NAMES = {"Revolute 6"}


@dataclass
class SmokeTestConfig:
    urdf_path: str
    initial_height: float
    base_orientation_euler_deg: tuple[float, float, float]
    joint_reset_deg: tuple[float, ...]
    test_target_position_base: tuple[float, float, float]
    ee_link_index: int
    obstacle_box_size: tuple[float, float, float]
    obstacle_box_position: tuple[float, float, float]
    position_tolerance_m: float
    gui: bool = False
    hold_seconds: float = 0.0
    collision_query_distance_m: float = 0.25
    path_collision_threshold_m: float = 0.0
    stop_on_path_collision: bool = True
    debug_render_width: int = 960
    debug_render_height: int = 720
    debug_render_camera_distance: float = 0.8
    debug_render_yaw_deg: float = 45.0
    debug_render_pitch_deg: float = -30.0
    debug_render_output_path: str | None = None
    animation_steps: int = 48
    animation_frame_sleep_sec: float = 0.03
    animation_output_dir: str | None = None


@dataclass
class SmokeTestResult:
    pybullet_import_ok: bool = False
    urdf_load_ok: bool = False
    joint_count: int = 0
    ik_success: bool = False
    ee_position_error_m: float | None = None
    collision_api_ok: bool = False
    path_collision_free: bool = False
    goal_state_collision_free: bool = False
    trajectory_reached_goal: bool = False
    smoke_test_passed: bool = False
    expected_joint_count: int | None = None
    reset_pose_applied: bool = False
    ik_solution_length: int = 0
    urdf_path: str | None = None
    arm_config_path: str | None = None
    controllable_joint_names: list[str] | None = None
    ee_position_before_ik: list[float] | None = None
    ee_position_after_ik: list[float] | None = None
    target_position_base: list[float] | None = None
    closest_obstacle_distance_m: float | None = None
    goal_state_obstacle_distance_m: float | None = None
    min_distance_along_path_m: float | None = None
    first_collision_frame_index: int | None = None
    last_safe_frame_index: int | None = None
    trajectory_stop_reason: str | None = None
    gui_enabled: bool = False
    debug_render_output_path: str | None = None
    debug_render_topdown_output_path: str | None = None
    debug_render_side_output_path: str | None = None
    animation_output_dir: str | None = None
    animation_frame_count: int = 0
    failure_bucket: str | None = None
    error: str | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:  # pragma: no cover - environment dependent
        payload = _load_simple_yaml(path)
    else:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in YAML config: {path}")
    return payload


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line in fallback parser: {raw_line}")
        key, value = line.split(":", 1)
        payload[key.strip()] = _parse_simple_yaml_value(value.strip())
    return payload


def _parse_simple_yaml_value(raw_value: str) -> Any:
    if raw_value == "":
        return None
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(raw_value)
    except Exception:
        pass
    try:
        if "." in raw_value or "e" in lowered:
            return float(raw_value)
        return int(raw_value)
    except ValueError:
        return raw_value


def _resolve_input_path(raw_path: str, config_path: Path) -> Path:
    candidate = Path(raw_path)
    search_roots = [config_path.parent, APPROACH_AGENT_ROOT, REPO_ROOT]

    if candidate.is_absolute():
        return candidate

    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved

    return (config_path.parent / candidate).resolve()


def _resolve_output_path(raw_path: str, config_path: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (APPROACH_AGENT_ROOT / candidate).resolve()


def _vector3(value: Sequence[Any], field_name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 values.")
    return (float(value[0]), float(value[1]), float(value[2]))


def _load_arm_config() -> dict[str, Any]:
    payload = _load_yaml(DEFAULT_ARM_CONFIG_PATH)
    if "pybullet" not in payload or "joints_reset" not in payload:
        raise ValueError(f"Unexpected arm config format: {DEFAULT_ARM_CONFIG_PATH}")
    return payload


def load_smoke_config(config_path: Path) -> SmokeTestConfig:
    payload = _load_yaml(config_path)
    debug_render_output = payload.get("debug_render_output_path")
    animation_output_dir = payload.get("animation_output_dir")
    return SmokeTestConfig(
        urdf_path=str(_resolve_input_path(str(payload["urdf_path"]), config_path)),
        initial_height=float(payload["initial_height"]),
        base_orientation_euler_deg=_vector3(payload["base_orientation_euler_deg"], "base_orientation_euler_deg"),
        joint_reset_deg=tuple(float(v) for v in payload["joint_reset_deg"]),
        test_target_position_base=_vector3(payload["test_target_position_base"], "test_target_position_base"),
        ee_link_index=int(payload["ee_link_index"]),
        obstacle_box_size=_vector3(payload["obstacle_box_size"], "obstacle_box_size"),
        obstacle_box_position=_vector3(payload["obstacle_box_position"], "obstacle_box_position"),
        position_tolerance_m=float(payload["position_tolerance_m"]),
        gui=bool(payload.get("gui", False)),
        hold_seconds=float(payload.get("hold_seconds", 0.0)),
        collision_query_distance_m=float(payload.get("collision_query_distance_m", 0.25)),
        path_collision_threshold_m=float(payload.get("path_collision_threshold_m", 0.0)),
        stop_on_path_collision=bool(payload.get("stop_on_path_collision", True)),
        debug_render_width=int(payload.get("debug_render_width", 960)),
        debug_render_height=int(payload.get("debug_render_height", 720)),
        debug_render_camera_distance=float(payload.get("debug_render_camera_distance", 0.8)),
        debug_render_yaw_deg=float(payload.get("debug_render_yaw_deg", 45.0)),
        debug_render_pitch_deg=float(payload.get("debug_render_pitch_deg", -30.0)),
        debug_render_output_path=(
            str(_resolve_output_path(str(debug_render_output), config_path))
            if debug_render_output
            else None
        ),
        animation_steps=int(payload.get("animation_steps", 48)),
        animation_frame_sleep_sec=float(payload.get("animation_frame_sleep_sec", 0.03)),
        animation_output_dir=(
            str(_resolve_output_path(str(animation_output_dir), config_path))
            if animation_output_dir
            else None
        ),
    )


def _degrees_to_radians(values_deg: Sequence[float]) -> list[float]:
    return [math.radians(float(value)) for value in values_deg]


def _load_python_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import pybullet as p
        import pybullet_data
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Missing runtime dependency. Please run inside the project Docker image "
            "or install numpy + pybullet first."
        ) from exc
    return np, p, pybullet_data


def _find_controllable_joints(p: Any, robot_id: int, expected_joint_count: int) -> tuple[list[int], list[str]]:
    joint_ids: list[int] = []
    joint_names: list[str] = []
    num_joints = p.getNumJoints(robot_id)
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_index)
        joint_name = joint_info[1].decode("utf-8")
        joint_type = joint_info[2]
        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC) and joint_name not in MIMIC_JOINT_NAMES:
            joint_ids.append(joint_index)
            joint_names.append(joint_name)

    if len(joint_ids) > expected_joint_count:
        joint_ids = joint_ids[:expected_joint_count]
        joint_names = joint_names[:expected_joint_count]

    return joint_ids, joint_names


def _reset_joint_positions(p: Any, robot_id: int, joint_ids: Sequence[int], joint_positions_rad: Sequence[float]) -> None:
    for joint_id, joint_position in zip(joint_ids, joint_positions_rad):
        p.resetJointState(robot_id, joint_id, targetValue=float(joint_position), targetVelocity=0.0)
    p.stepSimulation()


def _read_joint_positions(p: Any, robot_id: int, joint_ids: Sequence[int]) -> list[float]:
    states = p.getJointStates(robot_id, list(joint_ids))
    return [float(state[0]) for state in states]


def _add_target_marker(p: Any, target_position: Sequence[float]) -> None:
    marker_color = [0.1, 0.9, 0.2, 0.95]

    # A small box exactly at the target makes the desired point easier to spot
    # than a sphere in the off-screen TinyRenderer view.
    target_box_half_extents = [0.02, 0.02, 0.02]
    target_box_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=target_box_half_extents,
        rgbaColor=marker_color,
    )
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=target_box_shape,
        basePosition=list(target_position),
    )

    # Add a short vertical pillar above the target so it stays visible even if
    # the arm partially occludes the box from the camera view.
    pillar_height = 0.12
    pillar_radius = 0.008
    pillar_shape = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=pillar_radius,
        length=pillar_height,
        rgbaColor=marker_color,
    )
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=pillar_shape,
        basePosition=[
            float(target_position[0]),
            float(target_position[1]),
            float(target_position[2]) + pillar_height * 0.5 + 0.01,
        ],
        baseOrientation=p.getQuaternionFromEuler([math.pi * 0.5, 0.0, 0.0]),
    )


def _hold_gui_open(p: Any, gui_enabled: bool, hold_seconds: float, time_step: float) -> None:
    if not gui_enabled or hold_seconds <= 0.0:
        return
    deadline = time.time() + hold_seconds
    sleep_step = min(max(time_step, 1e-3), 0.02)
    while time.time() < deadline:
        p.stepSimulation()
        time.sleep(sleep_step)


def _write_ppm_image(path: Path, width: int, height: int, rgb_data: Any, np: Any) -> None:
    rgba = np.asarray(rgb_data, dtype=np.uint8)
    if rgba.ndim == 1:
        rgba = rgba.reshape(height, width, 4)
    elif rgba.ndim != 3:
        raise ValueError(f"Unexpected RGB buffer shape: {rgba.shape}")

    rgb = rgba[:, :, :3]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        handle.write(header)
        handle.write(rgb.tobytes())


def _render_debug_ppm(
    p: Any,
    np: Any,
    output_path: Path,
    *,
    width: int,
    height: int,
    camera_target_position: Sequence[float],
    camera_distance: float,
    camera_yaw_deg: float,
    camera_pitch_deg: float,
) -> None:
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=list(camera_target_position),
        distance=float(camera_distance),
        yaw=float(camera_yaw_deg),
        pitch=float(camera_pitch_deg),
        roll=0.0,
        upAxisIndex=2,
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=float(width) / float(height),
        nearVal=0.02,
        farVal=3.0,
    )
    _, _, rgb_data, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
    )
    _write_ppm_image(output_path, width, height, rgb_data, np)


def _derive_topdown_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_topdown{output_path.suffix}")


def _derive_side_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_side{output_path.suffix}")


def _render_animation_frame(
    p: Any,
    np: Any,
    output_path: Path,
    *,
    width: int,
    height: int,
    camera_target_position: Sequence[float],
    camera_distance: float,
    camera_yaw_deg: float,
    camera_pitch_deg: float,
) -> None:
    _render_debug_ppm(
        p,
        np,
        output_path,
        width=width,
        height=height,
        camera_target_position=camera_target_position,
        camera_distance=camera_distance,
        camera_yaw_deg=camera_yaw_deg,
        camera_pitch_deg=camera_pitch_deg,
    )


def _query_robot_obstacle_distance(
    p: Any,
    robot_id: int,
    obstacle_id: int,
    *,
    query_distance: float,
    collision_threshold: float,
) -> tuple[float | None, bool]:
    closest_points = p.getClosestPoints(
        robot_id,
        obstacle_id,
        distance=float(query_distance),
    )
    try:
        point_count = len(closest_points)
    except TypeError:
        raise RuntimeError(
            "PyBullet collision query returned an unsupported type: "
            f"{type(closest_points).__name__}"
        )

    if point_count == 0:
        return None, False

    distances = [float(point[8]) for point in closest_points]
    min_distance = min(distances)
    in_collision = any(distance <= float(collision_threshold) for distance in distances)
    return min_distance, in_collision


def _animate_joint_motion(
    p: Any,
    np: Any,
    robot_id: int,
    obstacle_id: int,
    joint_ids: Sequence[int],
    start_joint_positions: Sequence[float],
    goal_joint_positions: Sequence[float],
    *,
    steps: int,
    gui_enabled: bool,
    frame_sleep_sec: float,
    query_distance: float,
    collision_threshold: float,
    stop_on_collision: bool,
    animation_output_dir: Path | None,
    render_width: int,
    render_height: int,
    render_camera_target_position: Sequence[float],
    render_camera_distance: float,
    render_camera_yaw_deg: float,
    render_camera_pitch_deg: float,
) -> dict[str, Any]:
    step_count = max(int(steps), 1)
    animation_frame_count = 0
    output_dir = animation_output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    start = np.asarray(start_joint_positions, dtype=float)
    goal = np.asarray(goal_joint_positions, dtype=float)
    min_distance_along_path: float | None = None
    first_collision_frame_index: int | None = None
    last_safe_frame_index = -1
    trajectory_reached_goal = False
    trajectory_stop_reason = "goal_reached"

    for frame_index in range(step_count + 1):
        alpha = float(frame_index) / float(step_count)
        interp = ((1.0 - alpha) * start) + (alpha * goal)
        _reset_joint_positions(p, robot_id, joint_ids, interp.tolist())
        p.performCollisionDetection()
        min_distance, in_collision = _query_robot_obstacle_distance(
            p,
            robot_id,
            obstacle_id,
            query_distance=query_distance,
            collision_threshold=collision_threshold,
        )

        if min_distance is not None:
            if min_distance_along_path is None or min_distance < min_distance_along_path:
                min_distance_along_path = min_distance

        if output_dir is not None:
            frame_path = output_dir / f"frame_{frame_index:04d}.ppm"
            _render_animation_frame(
                p,
                np,
                frame_path,
                width=render_width,
                height=render_height,
                camera_target_position=render_camera_target_position,
                camera_distance=render_camera_distance,
                camera_yaw_deg=render_camera_yaw_deg,
                camera_pitch_deg=render_camera_pitch_deg,
            )
            animation_frame_count += 1

        if in_collision:
            first_collision_frame_index = frame_index
            trajectory_stop_reason = "collision"
            if stop_on_collision:
                break
        else:
            last_safe_frame_index = frame_index

        if gui_enabled:
            time.sleep(max(float(frame_sleep_sec), 0.0))

    if first_collision_frame_index is None:
        trajectory_reached_goal = True
        last_safe_frame_index = step_count

    return {
        "animation_frame_count": animation_frame_count,
        "min_distance_along_path_m": min_distance_along_path,
        "first_collision_frame_index": first_collision_frame_index,
        "last_safe_frame_index": None if last_safe_frame_index < 0 else last_safe_frame_index,
        "trajectory_reached_goal": trajectory_reached_goal,
        "trajectory_stop_reason": trajectory_stop_reason,
        "path_collision_free": first_collision_frame_index is None,
    }


def _evaluate_goal_state(
    p: Any,
    np: Any,
    robot_id: int,
    obstacle_id: int,
    joint_ids: Sequence[int],
    target_joint_positions: Sequence[float],
    *,
    ee_link_index: int,
    target_position_base: Sequence[float],
    query_distance: float,
    collision_threshold: float,
) -> dict[str, Any]:
    _reset_joint_positions(p, robot_id, joint_ids, target_joint_positions)
    p.performCollisionDetection()
    ee_after = p.getLinkState(robot_id, ee_link_index, computeForwardKinematics=True)[0]
    target_position = np.array(target_position_base, dtype=float)
    ee_after_position = np.array(ee_after, dtype=float)
    position_error = float(np.linalg.norm(ee_after_position - target_position))
    goal_distance, goal_in_collision = _query_robot_obstacle_distance(
        p,
        robot_id,
        obstacle_id,
        query_distance=query_distance,
        collision_threshold=collision_threshold,
    )
    return {
        "ee_position_after_ik": [float(v) for v in ee_after],
        "ee_position_error_m": position_error,
        "goal_state_obstacle_distance_m": goal_distance,
        "goal_state_collision_free": not goal_in_collision,
    }


def run_smoke_test(
    config_path: Path,
    *,
    gui_override: bool = False,
    hold_seconds_override: float | None = None,
    save_debug_ppm_override: str | None = None,
    save_animation_dir_override: str | None = None,
) -> SmokeTestResult:
    result = SmokeTestResult(arm_config_path=str(DEFAULT_ARM_CONFIG_PATH))
    smoke_config: SmokeTestConfig | None = None
    gui_enabled = False
    hold_seconds = 0.0
    time_step = 1.0 / 240.0

    try:
        smoke_config = load_smoke_config(config_path)
        if gui_override:
            smoke_config.gui = True
        if hold_seconds_override is not None:
            smoke_config.hold_seconds = float(hold_seconds_override)
        if save_debug_ppm_override is not None:
            smoke_config.debug_render_output_path = str(Path(save_debug_ppm_override).resolve())
        if save_animation_dir_override is not None:
            smoke_config.animation_output_dir = str(Path(save_animation_dir_override).resolve())
        if smoke_config.gui:
            smoke_config.debug_render_output_path = None
            smoke_config.animation_output_dir = None
        arm_config = _load_arm_config()
    except Exception as exc:
        result.failure_bucket = "config"
        result.error = str(exc)
        return result

    result.urdf_path = smoke_config.urdf_path
    result.target_position_base = list(smoke_config.test_target_position_base)
    result.gui_enabled = bool(smoke_config.gui)
    result.debug_render_output_path = smoke_config.debug_render_output_path
    result.animation_output_dir = smoke_config.animation_output_dir
    if smoke_config.debug_render_output_path:
        result.debug_render_topdown_output_path = str(
            _derive_topdown_output_path(Path(smoke_config.debug_render_output_path))
        )
        result.debug_render_side_output_path = str(
            _derive_side_output_path(Path(smoke_config.debug_render_output_path))
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
        connection_mode = p.GUI if smoke_config.gui else p.DIRECT
        client_id = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        time_step = float(arm_config["pybullet"]["time_step"])
        gui_enabled = bool(smoke_config.gui)
        hold_seconds = float(smoke_config.hold_seconds)
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

        base_orientation_rad = [math.radians(v) for v in smoke_config.base_orientation_euler_deg]
        base_orientation_xyzw = p.getQuaternionFromEuler(base_orientation_rad)
        robot_id = p.loadURDF(
            smoke_config.urdf_path,
            useFixedBase=True,
            basePosition=[0.0, 0.0, smoke_config.initial_height],
            baseOrientation=base_orientation_xyzw,
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

        joint_reset_rad = _degrees_to_radians(smoke_config.joint_reset_deg)
        if len(joint_reset_rad) != len(controllable_joint_ids):
            result.failure_bucket = "config"
            result.error = (
                "joint_reset_deg length does not match controllable joint count: "
                f"{len(joint_reset_rad)} vs {len(controllable_joint_ids)}"
            )
            return result

        _reset_joint_positions(p, robot_id, controllable_joint_ids, joint_reset_rad)
        actual_reset_positions = _read_joint_positions(p, robot_id, controllable_joint_ids)
        result.reset_pose_applied = all(
            abs(expected - actual) <= 1e-6
            for expected, actual in zip(joint_reset_rad, actual_reset_positions)
        )
        if not result.reset_pose_applied:
            result.failure_bucket = "model"
            result.error = "Failed to apply the configured reset pose accurately."
            return result

        if smoke_config.ee_link_index >= p.getNumJoints(robot_id):
            result.failure_bucket = "config"
            result.error = f"ee_link_index {smoke_config.ee_link_index} is out of range."
            return result

        ee_before = p.getLinkState(robot_id, smoke_config.ee_link_index, computeForwardKinematics=True)[0]
        result.ee_position_before_ik = [float(v) for v in ee_before]

        obstacle_half_extents = [size * 0.5 for size in smoke_config.obstacle_box_size]
        obstacle_collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=obstacle_half_extents,
        )
        obstacle_visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=obstacle_half_extents,
            rgbaColor=[0.85, 0.2, 0.2, 0.65],
        )
        obstacle_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=obstacle_collision_shape,
            baseVisualShapeIndex=obstacle_visual_shape,
            basePosition=list(smoke_config.obstacle_box_position),
        )
        _add_target_marker(p, smoke_config.test_target_position_base)
        if gui_enabled:
            p.resetDebugVisualizerCamera(
                cameraDistance=float(smoke_config.debug_render_camera_distance),
                cameraYaw=float(smoke_config.debug_render_yaw_deg),
                cameraPitch=float(smoke_config.debug_render_pitch_deg),
                cameraTargetPosition=list(smoke_config.test_target_position_base),
            )

        try:
            closest_distance, initial_in_collision = _query_robot_obstacle_distance(
                p,
                robot_id,
                obstacle_id,
                query_distance=float(smoke_config.collision_query_distance_m),
                collision_threshold=float(smoke_config.path_collision_threshold_m),
            )
        except RuntimeError as exc:
            result.collision_api_ok = False
            result.failure_bucket = "runtime"
            result.error = str(exc)
            return result
        result.collision_api_ok = True
        result.closest_obstacle_distance_m = closest_distance
        if initial_in_collision:
            result.failure_bucket = "geometry"
            result.error = "The reset pose is already in collision with the obstacle."
            result.path_collision_free = False
            result.trajectory_stop_reason = "start_in_collision"
            return result

        ik_solution = p.calculateInverseKinematics(
            robot_id,
            smoke_config.ee_link_index,
            targetPosition=list(smoke_config.test_target_position_base),
        )
        result.ik_solution_length = len(ik_solution)

        if len(ik_solution) < len(controllable_joint_ids):
            result.failure_bucket = "geometry"
            result.error = "IK returned fewer joint values than required."
            return result

        target_joint_positions = [float(v) for v in ik_solution[: len(controllable_joint_ids)]]
        start_joint_positions = _read_joint_positions(p, robot_id, controllable_joint_ids)
        goal_eval = _evaluate_goal_state(
            p,
            np,
            robot_id,
            obstacle_id,
            controllable_joint_ids,
            target_joint_positions,
            ee_link_index=smoke_config.ee_link_index,
            target_position_base=smoke_config.test_target_position_base,
            query_distance=float(smoke_config.collision_query_distance_m),
            collision_threshold=float(smoke_config.path_collision_threshold_m),
        )
        result.ee_position_after_ik = goal_eval["ee_position_after_ik"]
        result.ee_position_error_m = goal_eval["ee_position_error_m"]
        result.goal_state_obstacle_distance_m = goal_eval["goal_state_obstacle_distance_m"]
        result.goal_state_collision_free = goal_eval["goal_state_collision_free"]
        result.ik_success = (
            result.ee_position_error_m is not None
            and result.ee_position_error_m <= smoke_config.position_tolerance_m
            and result.goal_state_collision_free
        )

        _reset_joint_positions(p, robot_id, controllable_joint_ids, start_joint_positions)
        start_joint_positions = _read_joint_positions(p, robot_id, controllable_joint_ids)
        animation_result = _animate_joint_motion(
            p,
            np,
            robot_id,
            obstacle_id,
            controllable_joint_ids,
            start_joint_positions,
            target_joint_positions,
            steps=int(smoke_config.animation_steps),
            gui_enabled=gui_enabled,
            frame_sleep_sec=float(smoke_config.animation_frame_sleep_sec),
            query_distance=float(smoke_config.collision_query_distance_m),
            collision_threshold=float(smoke_config.path_collision_threshold_m),
            stop_on_collision=bool(smoke_config.stop_on_path_collision),
            animation_output_dir=(
                Path(smoke_config.animation_output_dir)
                if smoke_config.animation_output_dir
                else None
            ),
            render_width=int(smoke_config.debug_render_width),
            render_height=int(smoke_config.debug_render_height),
            render_camera_target_position=smoke_config.test_target_position_base,
            render_camera_distance=float(smoke_config.debug_render_camera_distance),
            render_camera_yaw_deg=float(smoke_config.debug_render_yaw_deg),
            render_camera_pitch_deg=float(smoke_config.debug_render_pitch_deg),
        )
        result.animation_frame_count = int(animation_result["animation_frame_count"])
        result.min_distance_along_path_m = animation_result["min_distance_along_path_m"]
        result.first_collision_frame_index = animation_result["first_collision_frame_index"]
        result.last_safe_frame_index = animation_result["last_safe_frame_index"]
        result.trajectory_reached_goal = bool(animation_result["trajectory_reached_goal"])
        result.trajectory_stop_reason = animation_result["trajectory_stop_reason"]
        result.path_collision_free = bool(animation_result["path_collision_free"])

        if not result.ik_success:
            result.failure_bucket = "geometry"
            if result.ee_position_error_m is None:
                result.error = "IK goal-state error could not be evaluated."
            elif not result.goal_state_collision_free:
                result.error = "IK goal state is in collision with the obstacle."
            else:
                result.error = (
                    f"End-effector error {result.ee_position_error_m:.6f} m exceeded tolerance "
                    f"{smoke_config.position_tolerance_m:.6f} m."
                )
            return result

        if not result.path_collision_free:
            result.failure_bucket = "geometry"
            result.error = (
                "Straight-line joint interpolation collided with the obstacle "
                f"at frame {result.first_collision_frame_index}."
            )
            return result

        result.smoke_test_passed = (
            result.pybullet_import_ok
            and result.urdf_load_ok
            and result.reset_pose_applied
            and result.collision_api_ok
            and result.ik_success
            and result.path_collision_free
            and result.trajectory_reached_goal
        )
        if not result.smoke_test_passed and result.failure_bucket is None:
            result.failure_bucket = "unknown"
            result.error = "Smoke test failed without a more specific failure bucket."
        return result
    except Exception as exc:  # pragma: no cover - defensive reporting
        if result.failure_bucket is None:
            result.failure_bucket = "runtime"
        result.error = str(exc)
        return result
    finally:
        if client_id is not None:
            try:
                if smoke_config is not None and smoke_config.debug_render_output_path:
                    debug_output_path = Path(smoke_config.debug_render_output_path)
                    _render_debug_ppm(
                        p,
                        np,
                        debug_output_path,
                        width=int(smoke_config.debug_render_width),
                        height=int(smoke_config.debug_render_height),
                        camera_target_position=smoke_config.test_target_position_base,
                        camera_distance=float(smoke_config.debug_render_camera_distance),
                        camera_yaw_deg=float(smoke_config.debug_render_yaw_deg),
                        camera_pitch_deg=float(smoke_config.debug_render_pitch_deg),
                    )
                    _render_debug_ppm(
                        p,
                        np,
                        _derive_topdown_output_path(debug_output_path),
                        width=int(smoke_config.debug_render_width),
                        height=int(smoke_config.debug_render_height),
                        camera_target_position=smoke_config.test_target_position_base,
                        camera_distance=max(0.55, float(smoke_config.debug_render_camera_distance) * 0.85),
                        camera_yaw_deg=0.0,
                        camera_pitch_deg=-89.0,
                    )
                    _render_debug_ppm(
                        p,
                        np,
                        _derive_side_output_path(debug_output_path),
                        width=int(smoke_config.debug_render_width),
                        height=int(smoke_config.debug_render_height),
                        camera_target_position=smoke_config.test_target_position_base,
                        camera_distance=max(0.55, float(smoke_config.debug_render_camera_distance) * 0.9),
                        camera_yaw_deg=90.0,
                        camera_pitch_deg=-12.0,
                    )
                _hold_gui_open(p, gui_enabled, hold_seconds, time_step)
                p.disconnect(client_id)
            except Exception:
                pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a minimal PyBullet smoke test for Approach_Agent.")
    parser.add_argument(
        "--config",
        type=Path,
        default=APPROACH_AGENT_ROOT / "configs" / "pybullet_smoke.yaml",
        help="Path to the smoke test YAML config.",
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
        help="Save a sequence of headless TinyRenderer animation frames to a directory.",
    )
    args = parser.parse_args(argv)

    result = run_smoke_test(
        args.config.resolve(),
        gui_override=bool(args.gui),
        hold_seconds_override=args.hold_seconds,
        save_debug_ppm_override=(str(args.save_debug_ppm.resolve()) if args.save_debug_ppm else None),
        save_animation_dir_override=(str(args.save_animation_dir.resolve()) if args.save_animation_dir else None),
    )
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0 if result.smoke_test_passed else 1
