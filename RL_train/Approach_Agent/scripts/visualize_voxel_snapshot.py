from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.pybullet_ompl import load_planning_config
from src.pybullet_smoke import (
    APPROACH_AGENT_ROOT,
    _degrees_to_radians,
    _find_controllable_joints,
    _load_arm_config,
    _load_python_dependencies,
    _reset_joint_positions,
)


def _load_snapshot(npz_path: Path, np: Any) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def _as_xyz_points(name: str, value: Any, np: Any) -> Any:
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must be an Nx3 array, got shape {points.shape}.")
    return points


def _as_xyz_vector(name: str, value: Any, np: Any) -> tuple[float, float, float]:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 values, got shape {vector.shape}.")
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _as_rotation_matrix(name: str, value: Any, np: Any) -> Any:
    rotation = np.asarray(value, dtype=np.float32)
    if rotation.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {rotation.shape}.")
    return rotation


def _estimate_scene_view(positions_xyz: Any, np: Any) -> tuple[list[float], float]:
    if len(positions_xyz) == 0:
        return [0.0, 0.0, 0.3], 1.0

    min_xyz = positions_xyz.min(axis=0)
    max_xyz = positions_xyz.max(axis=0)
    center = ((min_xyz + max_xyz) * 0.5).astype(np.float32)
    extent = np.maximum(max_xyz - min_xyz, 0.05)
    distance = float(np.linalg.norm(extent) * 0.9)
    return [float(center[0]), float(center[1]), float(center[2])], max(0.9, distance)


def _downsample_points(points_xyz: Any, *, max_points: int, np: Any) -> Any:
    if max_points <= 0:
        return np.asarray(points_xyz, dtype=np.float32)
    if len(points_xyz) <= max_points:
        return np.asarray(points_xyz, dtype=np.float32)
    step = max(1, math.ceil(len(points_xyz) / max_points))
    return np.asarray(points_xyz[::step], dtype=np.float32)


def _add_marker(
    p: Any,
    position_xyz: tuple[float, float, float],
    *,
    rgba: tuple[float, float, float, float],
    label: str | None = None,
) -> None:
    sphere_shape = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=0.02,
        rgbaColor=list(rgba),
    )
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=sphere_shape,
        basePosition=list(position_xyz),
    )
    if label:
        p.addUserDebugText(
            label,
            [float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2]) + 0.05],
            textColorRGB=list(rgba[:3]),
            textSize=1.3,
        )


def _add_frame(
    p: Any,
    origin_xyz: tuple[float, float, float],
    *,
    rotation_matrix: Any | None = None,
    axis_length: float = 0.10,
    axis_width: float = 2.0,
    label: str | None = None,
) -> None:
    origin = [float(v) for v in origin_xyz]
    rotation = np.eye(3, dtype=np.float32) if rotation_matrix is None else np.asarray(rotation_matrix, dtype=np.float32)
    axis_colors = (
        [1.0, 0.15, 0.15],
        [0.15, 1.0, 0.15],
        [0.15, 0.35, 1.0],
    )
    for axis_index, color in enumerate(axis_colors):
        axis_direction = rotation[:, axis_index]
        end_point = [
            origin[0] + float(axis_direction[0]) * float(axis_length),
            origin[1] + float(axis_direction[1]) * float(axis_length),
            origin[2] + float(axis_direction[2]) * float(axis_length),
        ]
        p.addUserDebugLine(
            origin,
            end_point,
            lineColorRGB=color,
            lineWidth=float(axis_width),
            lifeTime=0.0,
        )
    if label:
        p.addUserDebugText(
            label,
            [origin[0], origin[1], origin[2] + float(axis_length) * 1.1],
            textColorRGB=[1.0, 1.0, 1.0],
            textSize=1.2,
        )


def _set_joint_positions_direct(
    p: Any,
    robot_id: int,
    joint_ids: list[int],
    joint_positions_rad: list[float],
) -> None:
    for joint_id, joint_position in zip(joint_ids, joint_positions_rad):
        p.resetJointState(robot_id, joint_id, targetValue=float(joint_position), targetVelocity=0.0)


def _create_robot_reset_pose(
    p: Any,
    pybullet_data: Any,
    *,
    planner_config_path: Path,
) -> tuple[int, list[int], tuple[float, float, float], int]:
    planning_config = load_planning_config(planner_config_path)
    arm_config = _load_arm_config()
    expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    base_orientation_rad = [math.radians(v) for v in planning_config.base_orientation_euler_deg]
    base_orientation_xyzw = p.getQuaternionFromEuler(base_orientation_rad)
    robot_id = p.loadURDF(
        planning_config.urdf_path,
        useFixedBase=True,
        basePosition=[0.0, 0.0, planning_config.initial_height],
        baseOrientation=base_orientation_xyzw,
    )

    joint_ids, _ = _find_controllable_joints(p, robot_id, expected_joint_count)
    _reset_joint_positions(
        p,
        robot_id,
        joint_ids,
        _degrees_to_radians(planning_config.joint_reset_deg),
    )
    p.performCollisionDetection()

    ee_link_index = int(planning_config.ee_link_index)
    if ee_link_index >= p.getNumJoints(robot_id):
        raise ValueError(f"ee_link_index {ee_link_index} is out of range for the loaded robot.")

    ee_position = p.getLinkState(robot_id, ee_link_index, computeForwardKinematics=True)[0]
    return robot_id, joint_ids, (float(ee_position[0]), float(ee_position[1]), float(ee_position[2])), ee_link_index


def _load_report_json(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    return json.loads(report_path.read_text(encoding="utf-8"))


def _extract_planned_path_joint_states_deg(report_payload: dict[str, Any]) -> list[list[float]] | None:
    planning_result = report_payload.get("planning_result")
    if not isinstance(planning_result, dict):
        return None
    raw_states = planning_result.get("planned_path_joint_states_deg")
    if raw_states is None:
        return None
    return [[float(v) for v in state] for state in raw_states]


def _extract_best_base_pose(report_payload: dict[str, Any]) -> tuple[tuple[float, float, float] | None, float | None]:
    for container in (report_payload, report_payload.get("planning_result") or {}):
        if not isinstance(container, dict):
            continue
        position = container.get("best_base_pose_pybullet_xyz")
        yaw_deg = container.get("best_base_pose_yaw_deg")
        if position is None or yaw_deg is None:
            continue
        if len(position) != 3:
            raise ValueError("best_base_pose_pybullet_xyz must contain exactly 3 values.")
        return (
            (float(position[0]), float(position[1]), float(position[2])),
            float(yaw_deg),
        )
    return None, None


def _yaw_rotation_matrix(yaw_deg: float, *, np: Any) -> Any:
    yaw_rad = math.radians(float(yaw_deg))
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    return np.asarray(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _animate_joint_path(
    p: Any,
    robot_id: int,
    joint_ids: list[int],
    joint_states_deg: list[list[float]],
    *,
    frame_sleep_sec: float,
) -> None:
    for joint_state_deg in joint_states_deg:
        _set_joint_positions_direct(
            p,
            robot_id,
            joint_ids,
            _degrees_to_radians(joint_state_deg),
        )
        p.performCollisionDetection()
        p.stepSimulation()
        time.sleep(max(float(frame_sleep_sec), 1e-3))


def _add_voxel_boxes(
    p: Any,
    voxel_centers_xyz: Any,
    *,
    voxel_size_m: float,
    rgba: tuple[float, float, float, float],
) -> int:
    if len(voxel_centers_xyz) == 0:
        return 0

    half_extents = [float(voxel_size_m) * 0.5] * 3
    visual_shape = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=list(rgba),
    )
    for center in voxel_centers_xyz:
        p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=visual_shape,
            basePosition=[float(center[0]), float(center[1]), float(center[2])],
        )
    return int(len(voxel_centers_xyz))


def _add_debug_points(
    p: Any,
    points_xyz: Any,
    *,
    point_size: int,
    rgb: tuple[float, float, float],
) -> int:
    if len(points_xyz) == 0:
        return 0
    colors = [list(rgb)] * len(points_xyz)
    p.addUserDebugPoints(
        pointPositions=points_xyz.tolist(),
        pointColorsRGB=colors,
        pointSize=int(point_size),
        lifeTime=0.0,
    )
    return int(len(points_xyz))


def _spin_gui(p: Any, *, hold_seconds: float, time_step: float) -> None:
    if hold_seconds > 0.0:
        deadline = time.time() + hold_seconds
        while p.isConnected() and time.time() < deadline:
            p.stepSimulation()
            time.sleep(min(max(time_step, 1e-3), 0.02))
        return

    try:
        while p.isConnected():
            p.stepSimulation()
            time.sleep(min(max(time_step, 1e-3), 0.02))
    except KeyboardInterrupt:
        return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visualize a camera_car_voxel_ompl voxel_snapshot.npz in PyBullet GUI."
    )
    parser.add_argument(
        "npz_path",
        type=Path,
        help="Path to voxel_snapshot.npz.",
    )
    parser.add_argument(
        "--planner-config",
        type=Path,
        default=APPROACH_AGENT_ROOT / "configs" / "pybullet_ompl.yaml",
        help="Planner config used to load the robot reset pose.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional camera_car_voxel_ompl_report.json for replaying the successful path and future base-pose overlays.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="Voxel edge length in meters. Defaults to 0.05.",
    )
    parser.add_argument(
        "--max-debug-points",
        type=int,
        default=0,
        help="Maximum number of point-cloud samples to draw as debug points. Use 0 to disable point downsampling and draw the full saved point cloud.",
    )
    parser.add_argument(
        "--point-size",
        type=int,
        default=3,
        help="PyBullet debug point size for sampled points.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.0,
        help="Keep the GUI open for a fixed duration. Use 0 to keep it open until Ctrl+C or window close.",
    )
    parser.add_argument(
        "--hide-points",
        action="store_true",
        help="Only draw voxel boxes and markers.",
    )
    parser.add_argument(
        "--play-path",
        action="store_true",
        help="Replay the successful OMPL path stored in --report-json before entering the final GUI hold loop.",
    )
    parser.add_argument(
        "--path-frame-sleep",
        type=float,
        default=0.03,
        help="Seconds to sleep between replayed path states when --play-path is enabled.",
    )
    parser.add_argument(
        "--freeze-render-on-load",
        action="store_true",
        help="Temporarily freeze GUI rendering while building the scene. This avoids progressive draw-in, but can show a black window for large voxel scenes.",
    )
    args = parser.parse_args(argv)

    np, p, pybullet_data = _load_python_dependencies()
    snapshot = _load_snapshot(args.npz_path.resolve(), np)

    voxel_centers = None
    if "voxel_centers_pybullet" in snapshot:
        voxel_centers = _as_xyz_points("voxel_centers_pybullet", snapshot["voxel_centers_pybullet"], np)
    elif "voxel_centers_camera" in snapshot:
        voxel_centers = _as_xyz_points("voxel_centers_camera", snapshot["voxel_centers_camera"], np)
    else:
        raise ValueError("Snapshot does not contain voxel_centers_pybullet or voxel_centers_camera.")

    if "points_pybullet" in snapshot:
        points_xyz = _as_xyz_points("points_pybullet", snapshot["points_pybullet"], np)
    elif "points_camera_selected" in snapshot:
        points_xyz = _as_xyz_points("points_camera_selected", snapshot["points_camera_selected"], np)
    elif "points_camera" in snapshot:
        points_xyz = _as_xyz_points("points_camera", snapshot["points_camera"], np)
    else:
        points_xyz = np.empty((0, 3), dtype=np.float32)

    sampled_points = _downsample_points(points_xyz, max_points=max(int(args.max_debug_points), 0), np=np)

    ee_anchor = (
        _as_xyz_vector("ee_anchor_pybullet_xyz", snapshot["ee_anchor_pybullet_xyz"], np)
        if "ee_anchor_pybullet_xyz" in snapshot
        else None
    )
    target_position = (
        _as_xyz_vector("target_position_pybullet_xyz", snapshot["target_position_pybullet_xyz"], np)
        if "target_position_pybullet_xyz" in snapshot
        else None
    )
    target_rotation = (
        _as_rotation_matrix("target_rotation_pybullet_matrix", snapshot["target_rotation_pybullet_matrix"], np)
        if "target_rotation_pybullet_matrix" in snapshot
        else None
    )
    gripper_midpoint_camera = (
        _as_xyz_vector("gripper_midpoint_camera_xyz", snapshot["gripper_midpoint_camera_xyz"], np)
        if "gripper_midpoint_camera_xyz" in snapshot
        else None
    )
    report_payload = (
        _load_report_json(args.report_json.resolve())
        if args.report_json is not None
        else None
    )
    planned_path_joint_states_deg = (
        _extract_planned_path_joint_states_deg(report_payload)
        if report_payload is not None
        else None
    )
    best_base_position, best_base_yaw_deg = (
        _extract_best_base_pose(report_payload)
        if report_payload is not None
        else (None, None)
    )

    render_positions = voxel_centers if len(voxel_centers) else points_xyz
    camera_target_position, camera_distance = _estimate_scene_view(render_positions, np)

    client_id = p.connect(p.GUI)
    time_step = 1.0 / 240.0
    try:
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.8)
        p.setTimeStep(time_step)
        p.setPhysicsEngineParameter(fixedTimeStep=time_step, numSolverIterations=100, numSubSteps=10)
        p.setRealTimeSimulation(0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        if args.freeze_render_on_load:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        robot_id, joint_ids, reset_ee_position, ee_link_index = _create_robot_reset_pose(
            p,
            pybullet_data,
            planner_config_path=args.planner_config.resolve(),
        )

        voxel_count = _add_voxel_boxes(
            p,
            voxel_centers,
            voxel_size_m=float(args.voxel_size),
            rgba=(0.88, 0.18, 0.18, 0.40),
        )
        drawn_point_count = 0
        if not args.hide_points:
            drawn_point_count = _add_debug_points(
                p,
                sampled_points,
                point_size=int(args.point_size),
                rgb=(0.78, 0.78, 0.78),
            )

        _add_marker(p, reset_ee_position, rgba=(0.15, 0.55, 0.95, 0.95), label="reset_ee")
        _add_frame(p, reset_ee_position, axis_length=0.09, axis_width=2.0, label="reset_ee_frame")
        if ee_anchor is not None:
            _add_marker(p, ee_anchor, rgba=(0.1, 0.9, 0.2, 0.95), label="snapshot_anchor")
            _add_frame(p, ee_anchor, axis_length=0.09, axis_width=2.0, label="snapshot_anchor_frame")
        if target_position is not None:
            _add_marker(p, target_position, rgba=(1.0, 0.85, 0.1, 0.95), label="target_pose")
            _add_frame(
                p,
                target_position,
                rotation_matrix=target_rotation,
                axis_length=0.11,
                axis_width=2.3,
                label="target_pose_frame",
            )
        if best_base_position is not None and best_base_yaw_deg is not None:
            _add_frame(
                p,
                best_base_position,
                rotation_matrix=_yaw_rotation_matrix(best_base_yaw_deg, np=np),
                axis_length=0.14,
                axis_width=2.5,
                label="best_base_pose",
            )

        if gripper_midpoint_camera is not None:
            p.addUserDebugText(
                f"gripper_midpoint_camera={list(round(v, 4) for v in gripper_midpoint_camera)}",
                [camera_target_position[0], camera_target_position[1], camera_target_position[2] + 0.15],
                textColorRGB=[0.2, 0.7, 1.0],
                textSize=1.3,
            )

        summary = {
            "npz_path": str(args.npz_path.resolve()),
            "planner_config_path": str(args.planner_config.resolve()),
            "voxel_count": int(voxel_count),
            "point_count_total": int(len(points_xyz)),
            "point_count_drawn": int(drawn_point_count),
            "ee_link_index": int(ee_link_index),
            "reset_ee_position": [float(v) for v in reset_ee_position],
            "snapshot_anchor": (list(ee_anchor) if ee_anchor is not None else None),
            "target_position_pybullet_xyz": (list(target_position) if target_position is not None else None),
            "target_rotation_pybullet_matrix_present": bool(target_rotation is not None),
            "gripper_midpoint_camera_xyz": (
                list(gripper_midpoint_camera) if gripper_midpoint_camera is not None else None
            ),
            "planned_path_state_count": (
                len(planned_path_joint_states_deg) if planned_path_joint_states_deg is not None else 0
            ),
            "best_base_pose_pybullet_xyz": (list(best_base_position) if best_base_position is not None else None),
            "best_base_pose_yaw_deg": best_base_yaw_deg,
            "camera_target_position": [float(v) for v in camera_target_position],
            "camera_distance": float(camera_distance),
            "snapshot_keys": sorted(snapshot.keys()),
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if drawn_point_count < len(points_xyz):
            print(
                f"Point cloud was downsampled for GUI display: "
                f"{drawn_point_count}/{len(points_xyz)} points drawn."
            )

        p.resetDebugVisualizerCamera(
            cameraDistance=float(camera_distance),
            cameraYaw=45.0,
            cameraPitch=-28.0,
            cameraTargetPosition=camera_target_position,
        )
        if args.freeze_render_on_load:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        if args.play_path and planned_path_joint_states_deg:
            _animate_joint_path(
                p,
                robot_id,
                joint_ids,
                planned_path_joint_states_deg,
                frame_sleep_sec=float(args.path_frame_sleep),
            )

        _spin_gui(
            p,
            hold_seconds=float(args.hold_seconds),
            time_step=time_step,
        )
    finally:
        try:
            if p.isConnected():
                p.disconnect(client_id)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
