from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.pybullet_smoke import _load_python_dependencies


def _latest_coordinate_debug_npz() -> Path:
    candidates = sorted(
        Path("outputs/test_base_sampler_all_grasps").glob("*/coordinate_debug.npz"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            "No coordinate_debug.npz found under outputs/test_base_sampler_all_grasps/*/."
        )
    return candidates[-1]


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def _as_points(payload: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(payload.get(key, np.empty((0, 3))), dtype=np.float64)
    if value.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return value.reshape(-1, 3)


def _as_vector(payload: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(payload.get(key, np.empty((0,))), dtype=np.float64).reshape(-1)
    return value.astype(np.float64)


def _as_scalar(payload: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = np.asarray(payload.get(key, np.asarray([default])), dtype=np.float64).reshape(-1)
    if len(value) == 0:
        return float(default)
    return float(value[0])


def _downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    step = max(1, math.ceil(len(points) / max_points))
    return points[::step]


def _format_xyz(position: np.ndarray) -> str:
    values = np.asarray(position, dtype=np.float64).reshape(-1)
    if len(values) < 3:
        return "unavailable"
    return f"({values[0]:.4f}, {values[1]:.4f}, {values[2]:.4f})"


def _estimate_view(points: np.ndarray) -> tuple[list[float], float]:
    if len(points) == 0:
        return [0.0, 0.0, 0.3], 1.0
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    center = (min_xyz + max_xyz) * 0.5
    extent = np.maximum(max_xyz - min_xyz, 0.1)
    return center.astype(float).tolist(), max(0.9, float(np.linalg.norm(extent) * 0.85))


def _add_text(
    p_mod: Any,
    text: str,
    position: np.ndarray | list[float],
    *,
    color: tuple[float, float, float] = (1.0, 1.0, 1.0),
    size: float = 1.0,
) -> None:
    pos = np.asarray(position, dtype=np.float64).reshape(3)
    p_mod.addUserDebugText(
        text,
        pos.astype(float).tolist(),
        textColorRGB=list(color),
        textSize=float(size),
    )


def _add_sphere(
    p_mod: Any,
    position: np.ndarray,
    *,
    radius: float,
    rgba: tuple[float, float, float, float],
    label: str | None = None,
) -> None:
    pos = np.asarray(position, dtype=np.float64).reshape(-1)
    if len(pos) < 3:
        return
    visual_shape = p_mod.createVisualShape(
        p_mod.GEOM_SPHERE,
        radius=float(radius),
        rgbaColor=list(rgba),
    )
    p_mod.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual_shape,
        basePosition=pos[:3].astype(float).tolist(),
    )
    if label:
        _add_text(
            p_mod,
            label,
            [float(pos[0]), float(pos[1]), float(pos[2]) + radius * 2.0],
            color=tuple(rgba[:3]),
            size=1.0,
        )


def _add_frame(
    p_mod: Any,
    origin: np.ndarray,
    rotation_matrix: np.ndarray,
    *,
    axis_length: float,
    axis_width: float,
    label: str,
) -> None:
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    colors = (
        [1.0, 0.1, 0.1],
        [0.1, 1.0, 0.1],
        [0.1, 0.35, 1.0],
    )
    for axis_idx, color in enumerate(colors):
        direction = rotation_matrix[:, axis_idx]
        end = origin + direction * float(axis_length)
        p_mod.addUserDebugLine(
            origin.astype(float).tolist(),
            end.astype(float).tolist(),
            lineColorRGB=color,
            lineWidth=float(axis_width),
            lifeTime=0.0,
        )
    _add_text(
        p_mod,
        label,
        [float(origin[0]), float(origin[1]), float(origin[2]) + axis_length * 1.15],
        color=(1.0, 1.0, 1.0),
        size=1.0,
    )


def _yaw_rotation_matrix(yaw_rad: float) -> np.ndarray:
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    return np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _draw_voxels(
    p_mod: Any,
    voxels: np.ndarray,
    *,
    voxel_size: float,
    max_voxels: int,
) -> None:
    voxels = _downsample(voxels, max_voxels)
    half_extents = [float(voxel_size) * 0.5] * 3
    collision_shape = p_mod.createCollisionShape(p_mod.GEOM_BOX, halfExtents=half_extents)
    visual_shape = p_mod.createVisualShape(
        p_mod.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=[0.85, 0.15, 0.15, 0.75],
    )
    for voxel in voxels:
        p_mod.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=np.asarray(voxel, dtype=float).tolist(),
        )


def _draw_debug_points(
    p_mod: Any,
    points: np.ndarray,
    *,
    max_points: int,
    color: tuple[float, float, float],
    point_size: float,
) -> None:
    shown = _downsample(points, max_points)
    if len(shown) == 0:
        return
    colors = [list(color)] * len(shown)
    p_mod.addUserDebugPoints(
        shown.astype(float).tolist(),
        colors,
        pointSize=float(point_size),
    )


def _print_summary(payload: dict[str, Any], npz_path: Path) -> None:
    print(f"[coordinate_debug_viewer] npz={npz_path}", flush=True)
    for key in (
        "live_points_camera_xyz",
        "live_points_pb_world_xyz",
        "voxel_centers_pb_world_xyz",
        "target_positions_pb_world_xyz",
    ):
        value = np.asarray(payload.get(key, []))
        print(f"[coordinate_debug_viewer] {key}: shape={value.shape}", flush=True)

    selected_rank = int(_as_scalar(payload, "selected_rank", -1))
    print(f"[coordinate_debug_viewer] selected_rank={selected_rank}", flush=True)
    print(
        "[coordinate_debug_viewer] selected_target_pb_world_xyz="
        f"{_format_xyz(_as_vector(payload, 'selected_target_pb_world_xyz'))}",
        flush=True,
    )
    base = _as_vector(payload, "selected_pb_base_link_xyz")
    base_yaw = _as_scalar(payload, "selected_pb_base_link_yaw_rad")
    print(
        "[coordinate_debug_viewer] selected_pb_base_link="
        f"{_format_xyz(base)} yaw={base_yaw:.6f}rad ({math.degrees(base_yaw):.2f}deg)",
        flush=True,
    )
    amcl = _as_vector(payload, "selected_pb_amcl_link_xyz")
    amcl_yaw = _as_scalar(payload, "selected_pb_amcl_link_yaw_rad")
    print(
        "[coordinate_debug_viewer] selected_pb_amcl_link="
        f"{_format_xyz(amcl)} yaw={amcl_yaw:.6f}rad ({math.degrees(amcl_yaw):.2f}deg)",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize test_base_sampler coordinate_debug.npz in PyBullet."
    )
    parser.add_argument(
        "npz",
        nargs="?",
        type=Path,
        default=None,
        help="Path to coordinate_debug.npz. Defaults to latest output.",
    )
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=0,
        help="Maximum voxels to draw. 0 means draw all.",
    )
    parser.add_argument(
        "--show-points",
        action="store_true",
        help="Also draw transformed live_points_pb_world_xyz as cyan debug points.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=8000,
        help="Maximum live transformed points to draw when --show-points is set.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="Voxel cube size in meters.",
    )
    parser.add_argument(
        "--hold-sec",
        type=float,
        default=0.0,
        help="Seconds to keep the GUI open. 0 means until the window closes.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    npz_path = args.npz.resolve() if args.npz is not None else _latest_coordinate_debug_npz().resolve()
    payload = _load_npz(npz_path)
    _print_summary(payload, npz_path)

    voxels = _as_points(payload, "voxel_centers_pb_world_xyz")
    points_pb = _as_points(payload, "live_points_pb_world_xyz")
    targets_pb = _as_points(payload, "target_positions_pb_world_xyz")
    target_rotations = np.asarray(
        payload.get("target_rotations_pb_world_matrix", np.empty((0, 3, 3))),
        dtype=np.float64,
    ).reshape(-1, 3, 3)

    selected_rank = int(_as_scalar(payload, "selected_rank", -1))
    ranks = np.asarray(payload.get("grasp_ranks", np.empty((0,))), dtype=np.int32).reshape(-1)
    selected_target = _as_vector(payload, "selected_target_pb_world_xyz")
    selected_base = _as_vector(payload, "selected_pb_base_link_xyz")
    selected_base_yaw = _as_scalar(payload, "selected_pb_base_link_yaw_rad")
    selected_amcl = _as_vector(payload, "selected_pb_amcl_link_xyz")
    selected_amcl_yaw = _as_scalar(payload, "selected_pb_amcl_link_yaw_rad")

    _, p_mod, pybullet_data = _load_python_dependencies()
    client_id = p_mod.connect(p_mod.GUI)
    if client_id < 0:
        raise RuntimeError("PyBullet GUI unavailable.")

    try:
        p_mod.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_mod.resetSimulation()
        p_mod.configureDebugVisualizer(p_mod.COV_ENABLE_GUI, 0)
        p_mod.setGravity(0.0, 0.0, -9.8)
        p_mod.loadURDF("plane.urdf")

        _add_frame(
            p_mod,
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64),
            axis_length=0.25,
            axis_width=2.0,
            label="PB WORLD",
        )

        camera_position = _as_vector(payload, "camera_position_pb_world_xyz")
        camera_rotation = np.asarray(
            payload.get("camera_to_pb_rotation_matrix", np.eye(3)),
            dtype=np.float64,
        ).reshape(3, 3)
        if len(camera_position) >= 3:
            _add_sphere(
                p_mod,
                camera_position[:3],
                radius=0.025,
                rgba=(0.2, 0.7, 1.0, 1.0),
                label=f"camera PB {_format_xyz(camera_position)}",
            )
            _add_frame(
                p_mod,
                camera_position[:3],
                camera_rotation,
                axis_length=0.16,
                axis_width=2.0,
                label="CAMERA FRAME IN PB",
            )

        print(f"[coordinate_debug_viewer] Drawing {len(voxels)} voxel centers.", flush=True)
        _draw_voxels(
            p_mod,
            voxels,
            voxel_size=float(args.voxel_size),
            max_voxels=int(args.max_voxels),
        )
        if args.show_points:
            print(f"[coordinate_debug_viewer] Drawing {len(points_pb)} transformed live points.", flush=True)
            _draw_debug_points(
                p_mod,
                points_pb,
                max_points=int(args.max_points),
                color=(0.1, 0.8, 1.0),
                point_size=2.0,
            )

        for target_index, target in enumerate(targets_pb):
            rank = int(ranks[target_index]) if target_index < len(ranks) else target_index + 1
            is_selected = selected_rank == rank
            _add_sphere(
                p_mod,
                target,
                radius=0.04 if is_selected else 0.024,
                rgba=(1.0, 0.8, 0.0, 1.0 if is_selected else 0.55),
                label=f"TARGET G{rank} PB {_format_xyz(target)}",
            )
            if target_index < len(target_rotations):
                _add_frame(
                    p_mod,
                    target,
                    target_rotations[target_index],
                    axis_length=0.12 if is_selected else 0.08,
                    axis_width=2.0 if is_selected else 1.3,
                    label=f"G{rank}",
                )

        if len(selected_target) >= 3:
            _add_text(
                p_mod,
                f"SELECTED target PB xyz={_format_xyz(selected_target)}",
                [selected_target[0], selected_target[1], selected_target[2] + 0.18],
                color=(1.0, 0.9, 0.2),
                size=1.0,
            )

        if len(selected_base) >= 3:
            _add_sphere(
                p_mod,
                selected_base[:3],
                radius=0.035,
                rgba=(0.1, 0.45, 1.0, 1.0),
                label=(
                    f"PB base {_format_xyz(selected_base)} "
                    f"yaw={math.degrees(selected_base_yaw):.2f}deg"
                ),
            )
            _add_frame(
                p_mod,
                selected_base[:3],
                _yaw_rotation_matrix(selected_base_yaw),
                axis_length=0.20,
                axis_width=2.2,
                label="SELECTED BASE",
            )

        if len(selected_amcl) >= 3:
            _add_sphere(
                p_mod,
                selected_amcl[:3],
                radius=0.025,
                rgba=(0.2, 1.0, 0.45, 1.0),
                label=(
                    f"PB amcl {_format_xyz(selected_amcl)} "
                    f"yaw={math.degrees(selected_amcl_yaw):.2f}deg"
                ),
            )
            _add_frame(
                p_mod,
                selected_amcl[:3],
                _yaw_rotation_matrix(selected_amcl_yaw),
                axis_length=0.14,
                axis_width=1.7,
                label="AMCL LINK",
            )

        view_points = voxels
        if len(selected_target) >= 3:
            view_points = np.vstack([view_points, selected_target[:3].reshape(1, 3)])
        if len(selected_base) >= 3:
            view_points = np.vstack([view_points, selected_base[:3].reshape(1, 3)])
        camera_target, camera_distance = _estimate_view(view_points)
        p_mod.resetDebugVisualizerCamera(
            cameraDistance=float(camera_distance),
            cameraYaw=45.0,
            cameraPitch=-25.0,
            cameraTargetPosition=camera_target,
        )

        print("[coordinate_debug_viewer] GUI ready. Close the window to exit.", flush=True)
        deadline = time.time() + float(args.hold_sec) if args.hold_sec > 0.0 else None
        while p_mod.isConnected():
            p_mod.stepSimulation()
            if deadline is not None and time.time() >= deadline:
                break
            time.sleep(1.0 / 240.0)
    finally:
        if p_mod.isConnected():
            p_mod.disconnect(client_id)


if __name__ == "__main__":
    main()
