#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np


NPZ_PATH = Path(__file__).resolve().with_name("latest_grasp_debug.npz")
MESH_CAT_ZMQ_URL = "tcp://127.0.0.1:6000"

DISPLAY_FRAME = "camera_raw"  # camera_raw, camera_y_up, or image_aligned
MAX_SCENE_POINTS = 120000
TOPK_GRASPS = 20
SCENE_POINT_SIZE = 0.003
OBJECT_POINT_SIZE = 0.006

# Matches tool/graspgen_runtime/config/grippers/robotiq_2f_140.yaml.
GRIPPER_WIDTH_M = 0.13603458
GRIPPER_DEPTH_M = 0.2500


def rgb_to_hex(color: tuple[int, int, int]) -> int:
    r, g, b = color
    return (int(r) << 16) + (int(g) << 8) + int(b)


def scalar_text(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data.files:
        return default
    value = data[key]
    if getattr(value, "shape", None) == ():
        return str(value.item())
    return str(value)


def array_or(
    data: np.lib.npyio.NpzFile,
    key: str,
    fallback: np.ndarray,
    *,
    dtype: type = float,
) -> np.ndarray:
    if key in data.files:
        return np.asarray(data[key], dtype=dtype)
    return np.asarray(fallback, dtype=dtype)


def maybe_subsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(points), size=max_points, replace=False)
    return points[keep]


def display_transform(mode: str) -> np.ndarray:
    if mode == "camera_raw":
        return np.eye(4, dtype=float)
    if mode == "camera_y_up":
        return np.diag([1.0, -1.0, 1.0, 1.0]).astype(float)
    if mode == "image_aligned":
        return np.diag([-1.0, -1.0, 1.0, 1.0]).astype(float)
    raise ValueError(f"Unsupported DISPLAY_FRAME={mode!r}")


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(points) == 0:
        return points
    points_h = np.column_stack([points, np.ones(len(points), dtype=float)])
    return (np.asarray(transform, dtype=float) @ points_h.T).T[:, :3]


def transform_pose(pose: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(transform, dtype=float) @ np.asarray(pose, dtype=float)


def make_translation(translation_xyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = np.asarray(translation_xyz, dtype=float).reshape(3)
    return transform


def local_grasps_to_camera(grasps_local: np.ndarray, object_center_camera: np.ndarray) -> np.ndarray:
    grasps_local = np.asarray(grasps_local, dtype=float)
    if grasps_local.size == 0:
        return np.zeros((0, 4, 4), dtype=float)
    if grasps_local.ndim == 2:
        grasps_local = grasps_local[None, ...]
    grasps_camera = np.array(grasps_local, copy=True)
    grasps_camera[:, :3, 3] += np.asarray(object_center_camera, dtype=float).reshape(1, 3)
    return grasps_camera


def score_color(score: float, min_score: float, max_score: float) -> tuple[int, int, int]:
    if max_score <= min_score:
        t = 1.0
    else:
        t = float((score - min_score) / (max_score - min_score))
    t = max(0.0, min(1.0, t))
    return (int(255 * (1.0 - t)), int(80 + 175 * t), 30)


def robotiq_gripper_polyline() -> np.ndarray:
    w = GRIPPER_WIDTH_M
    d = GRIPPER_DEPTH_M
    right_front = np.array([w / 2.0, 0.0, d / 2.0], dtype=np.float32)
    left_front = np.array([-w / 2.0, 0.0, d / 2.0], dtype=np.float32)
    right_tip = np.array([w / 2.0, 0.0, d], dtype=np.float32)
    left_tip = np.array([-w / 2.0, 0.0, d], dtype=np.float32)
    mid = (right_front + left_front) / 2.0
    origin = np.zeros(3, dtype=np.float32)
    return np.stack(
        [right_tip, right_front, mid, origin, mid, left_front, left_tip],
        axis=0,
    )


def set_point_cloud(
    vis,
    name: str,
    points: np.ndarray,
    color: tuple[int, int, int] | np.ndarray,
    *,
    size: float,
) -> None:
    import meshcat.geometry as g

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return
    colors = np.asarray(color, dtype=np.float32)
    if colors.ndim == 1:
        colors = np.tile(colors.reshape(1, 3), (len(points), 1))
    if len(colors) != len(points):
        colors = np.tile(colors[:1].reshape(1, 3), (len(points), 1))
    vis[name].set_object(
        g.PointCloud(
            position=points.T,
            color=(colors / 255.0).T,
            size=size,
        )
    )


def set_line(
    vis,
    name: str,
    points: np.ndarray,
    color: tuple[int, int, int],
    *,
    linewidth: float = 2.0,
) -> None:
    import meshcat.geometry as g

    vis[name].set_object(
        g.Line(
            g.PointsGeometry(np.asarray(points, dtype=np.float32).T),
            g.MeshBasicMaterial(color=rgb_to_hex(color), linewidth=linewidth),
        )
    )


def set_frame(vis, name: str, transform: np.ndarray | None = None, length: float = 0.10) -> None:
    axes = (
        ("x", np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0]], dtype=np.float32), (255, 0, 0)),
        ("y", np.array([[0.0, 0.0, 0.0], [0.0, length, 0.0]], dtype=np.float32), (0, 200, 0)),
        ("z", np.array([[0.0, 0.0, 0.0], [0.0, 0.0, length]], dtype=np.float32), (0, 80, 255)),
    )
    for axis, points, color in axes:
        set_line(vis, f"{name}/{axis}", points, color, linewidth=4.0)
    if transform is not None:
        vis[name].set_transform(np.asarray(transform, dtype=float))


def set_sphere(
    vis,
    name: str,
    translation: np.ndarray,
    color: tuple[int, int, int],
    radius: float = 0.018,
) -> None:
    import meshcat.geometry as g

    vis[name].set_object(
        g.Sphere(radius),
        g.MeshLambertMaterial(color=rgb_to_hex(color)),
    )
    vis[name].set_transform(make_translation(translation))


def set_grasp(vis, name: str, grasp: np.ndarray, color: tuple[int, int, int], linewidth: float = 2.0) -> None:
    set_line(vis, name, robotiq_gripper_polyline(), color, linewidth=linewidth)
    vis[name].set_transform(np.asarray(grasp, dtype=float))


def choose_grasps(data: np.lib.npyio.NpzFile, object_center_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if "valid_grasps_camera" in data.files and "valid_grasp_confidences" in data.files:
        return (
            np.asarray(data["valid_grasps_camera"], dtype=float),
            np.asarray(data["valid_grasp_confidences"], dtype=float),
            "valid_grasps_camera",
        )
    if "collision_free_grasps_local" in data.files and "collision_free_scores" in data.files:
        return (
            local_grasps_to_camera(np.asarray(data["collision_free_grasps_local"], dtype=float), object_center_camera),
            np.asarray(data["collision_free_scores"], dtype=float),
            "collision_free_grasps_local",
        )
    if "all_grasps_local" in data.files and "all_scores" in data.files:
        return (
            local_grasps_to_camera(np.asarray(data["all_grasps_local"], dtype=float), object_center_camera),
            np.asarray(data["all_scores"], dtype=float),
            "all_grasps_local",
        )
    return np.zeros((0, 4, 4), dtype=float), np.zeros((0,), dtype=float), "<none>"


def best_grasp_camera(data: np.lib.npyio.NpzFile, object_center_camera: np.ndarray, grasps: np.ndarray, scores: np.ndarray) -> np.ndarray | None:
    if "best_grasp_camera" in data.files:
        return np.asarray(data["best_grasp_camera"], dtype=float)
    if "best_grasp_local" in data.files:
        return local_grasps_to_camera(np.asarray(data["best_grasp_local"], dtype=float), object_center_camera)[0]
    if len(grasps) > 0:
        return np.asarray(grasps[int(np.argmax(scores)) if len(scores) else 0], dtype=float)
    return None


def visualize() -> None:
    try:
        import meshcat
    except ImportError as exc:
        raise RuntimeError(
            "Python package 'meshcat' is not installed in this environment. "
            "Install it in the same environment used to run this script, then start meshcat-server."
        ) from exc

    if not NPZ_PATH.exists():
        raise FileNotFoundError(NPZ_PATH)

    data = np.load(NPZ_PATH, allow_pickle=True)
    object_center_camera = array_or(data, "object_reference_center_camera", np.zeros(3), dtype=float)
    gripper_midpoint_camera = array_or(data, "gripper_midpoint_camera_xyz", np.zeros(3), dtype=float)
    object_pc_camera = array_or(data, "object_pc_camera", np.zeros((0, 3)), dtype=float)
    scene_pc_camera = array_or(data, "scene_pc_camera", np.zeros((0, 3)), dtype=float)
    grasps_camera, scores, grasp_source = choose_grasps(data, object_center_camera)
    best_camera = best_grasp_camera(data, object_center_camera, grasps_camera, scores)

    order = np.argsort(scores)[::-1][: min(TOPK_GRASPS, len(scores))]
    top_grasps = grasps_camera[order] if len(order) else np.zeros((0, 4, 4), dtype=float)
    top_scores = scores[order] if len(order) else np.zeros((0,), dtype=float)

    display_t = display_transform(DISPLAY_FRAME)
    object_pc_display = transform_points(object_pc_camera, display_t)
    scene_pc_display = maybe_subsample(transform_points(scene_pc_camera, display_t), MAX_SCENE_POINTS)
    object_center_display = transform_points(object_center_camera.reshape(1, 3), display_t)[0]
    gripper_midpoint_display = transform_points(gripper_midpoint_camera.reshape(1, 3), display_t)[0]
    best_display = transform_pose(best_camera, display_t) if best_camera is not None else None
    top_display = np.asarray([transform_pose(grasp, display_t) for grasp in top_grasps], dtype=float)

    vis = meshcat.Visualizer(zmq_url=MESH_CAT_ZMQ_URL)
    vis.delete()

    set_point_cloud(vis, "point_cloud/scene_camera", scene_pc_display, (150, 150, 150), size=SCENE_POINT_SIZE)
    set_point_cloud(vis, "point_cloud/object_camera", object_pc_display, (0, 220, 80), size=OBJECT_POINT_SIZE)
    set_frame(vis, "frames/camera", transform=display_t, length=0.12)
    set_frame(vis, "frames/object_center", transform=make_translation(object_center_display), length=0.09)
    set_frame(vis, "frames/gripper_midpoint", transform=make_translation(gripper_midpoint_display), length=0.08)
    set_sphere(vis, "markers/object_center", object_center_display, (255, 255, 255), radius=0.018)
    set_sphere(vis, "markers/gripper_midpoint", gripper_midpoint_display, (40, 170, 255), radius=0.020)

    if best_display is not None:
        set_frame(vis, "frames/best_grasp", transform=best_display, length=0.11)
        set_grasp(vis, "grasps/best", best_display, (255, 210, 40), linewidth=3.0)

    if len(top_display) > 0:
        min_score = float(np.min(top_scores))
        max_score = float(np.max(top_scores))
        for rank, (grasp, score) in enumerate(zip(top_display, top_scores)):
            color = score_color(float(score), min_score, max_score)
            set_grasp(vis, f"grasps/top_{rank:02d}_score_{score:.3f}", grasp, color, linewidth=1.4)

    object_id = scalar_text(data, "object_id", "unknown")
    camera_name = scalar_text(data, "camera_name", "unknown")
    best_pos = best_camera[:3, 3].round(4).tolist() if best_camera is not None else None

    print(f"Loaded: {NPZ_PATH}")
    print(f"MeshCat URL: {vis.url()}")
    print(f"object_id={object_id}, camera_name={camera_name}, display_frame={DISPLAY_FRAME}")
    print(f"object_points={len(object_pc_camera)}, scene_points={len(scene_pc_camera)}")
    print(f"object_center_camera={object_center_camera.round(4).tolist()}")
    print(f"gripper_midpoint_camera={gripper_midpoint_camera.round(4).tolist()}")
    print(f"grasp_source={grasp_source}, showing_top={len(top_display)} of {len(grasps_camera)}")
    print(f"best_grasp_camera_position={best_pos}")


if __name__ == "__main__":
    try:
        visualize()
    except Exception as exc:
        print(f"Failed to visualize {NPZ_PATH}: {exc}")
        print("Make sure meshcat-server is running, for example: meshcat-server")
        raise
