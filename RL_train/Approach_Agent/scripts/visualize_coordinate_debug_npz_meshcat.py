from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


VOXEL_COLOR = [235, 60, 60]
POINT_COLOR = [40, 210, 255]
TARGET_COLOR = [255, 215, 40]
BASE_COLOR = [40, 120, 255]
AMCL_COLOR = [60, 255, 120]
CAMERA_COLOR = [80, 190, 255]


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


def _import_meshcat_helpers(zmq_url: str):
    try:
        import meshcat
        import meshcat.geometry as g
        import meshcat.transformations as mtf
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            "Failed to import MeshCat visualization dependencies. "
            f"Missing module: {missing}. Install it first, for example: `pip install meshcat`."
        ) from exc

    def rgb2hex(rgb: tuple[int, int, int] | list[int]) -> int:
        return int("0x%02x%02x%02x" % tuple(int(v) for v in rgb), 16)

    def create_visualizer(clear: bool = True):
        vis = meshcat.Visualizer(zmq_url=zmq_url)
        if clear:
            vis.delete()
        return vis

    def make_frame(
        vis: Any,
        name: str,
        *,
        h: float = 0.15,
        radius: float = 0.006,
        opacity: float = 1.0,
        transform: np.ndarray | None = None,
    ) -> None:
        vis[name]["x"].set_object(
            g.Cylinder(height=h, radius=radius),
            g.MeshLambertMaterial(color=0xFF0000, reflectivity=0.8, opacity=opacity),
        )
        rotate_x = mtf.rotation_matrix(np.pi / 2.0, [0, 0, 1])
        rotate_x[0, 3] = h / 2.0
        vis[name]["x"].set_transform(rotate_x)

        vis[name]["y"].set_object(
            g.Cylinder(height=h, radius=radius),
            g.MeshLambertMaterial(color=0x00FF00, reflectivity=0.8, opacity=opacity),
        )
        rotate_y = mtf.rotation_matrix(np.pi / 2.0, [0, 1, 0])
        rotate_y[1, 3] = h / 2.0
        vis[name]["y"].set_transform(rotate_y)

        vis[name]["z"].set_object(
            g.Cylinder(height=h, radius=radius),
            g.MeshLambertMaterial(color=0x0000FF, reflectivity=0.8, opacity=opacity),
        )
        rotate_z = mtf.rotation_matrix(np.pi / 2.0, [1, 0, 0])
        rotate_z[2, 3] = h / 2.0
        vis[name]["z"].set_transform(rotate_z)

        if transform is not None:
            vis[name].set_transform(np.asarray(transform, dtype=float))

    def visualize_pointcloud(
        vis: Any,
        name: str,
        points_xyz: np.ndarray,
        color: list[int],
        *,
        size: float,
    ) -> None:
        points_xyz = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
        if len(points_xyz) == 0:
            return
        color_array = np.ones_like(points_xyz, dtype=np.float32) * (
            np.asarray(color, dtype=np.float32).reshape(1, 3) / 255.0
        )
        vis[name].set_object(
            g.PointCloud(position=points_xyz.T, color=color_array.T, size=float(size))
        )

    def visualize_sphere(
        vis: Any,
        name: str,
        position_xyz: np.ndarray,
        color: list[int],
        *,
        radius: float,
        opacity: float = 1.0,
    ) -> None:
        position_xyz = np.asarray(position_xyz, dtype=float).reshape(-1)
        if len(position_xyz) < 3:
            return
        vis[name].set_object(
            g.Sphere(radius=float(radius)),
            g.MeshLambertMaterial(
                color=rgb2hex(color),
                reflectivity=0.6,
                opacity=float(opacity),
            ),
        )
        vis[name].set_transform(make_translation_transform(position_xyz[:3]))

    def visualize_box(
        vis: Any,
        name: str,
        center_xyz: np.ndarray,
        dims_xyz: np.ndarray,
        color: list[int],
        *,
        wireframe: bool = True,
    ) -> None:
        vis[name].set_object(
            g.Box(np.asarray(dims_xyz, dtype=float)),
            g.MeshBasicMaterial(wireframe=wireframe, color=rgb2hex(color)),
        )
        vis[name].set_transform(make_translation_transform(center_xyz))

    return create_visualizer, make_frame, visualize_pointcloud, visualize_sphere, visualize_box


def _load_npz(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(str(path), allow_pickle=True) as payload:
        return {key: payload[key] for key in payload.files}


def _points(payload: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(payload.get(key, np.empty((0, 3))), dtype=float)
    if value.size == 0:
        return np.empty((0, 3), dtype=float)
    return value.reshape(-1, 3)


def _vector(payload: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(payload.get(key, np.empty((0,))), dtype=float).reshape(-1)


def _scalar(payload: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = np.asarray(payload.get(key, np.asarray([default])), dtype=float).reshape(-1)
    if len(value) == 0:
        return float(default)
    return float(value[0])


def _maybe_subsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(points), size=max_points, replace=False)
    return points[np.sort(indices)]


def _format_xyz(value: np.ndarray) -> str:
    value = np.asarray(value, dtype=float).reshape(-1)
    if len(value) < 3:
        return "unavailable"
    return f"[{value[0]:.4f}, {value[1]:.4f}, {value[2]:.4f}]"


def make_translation_transform(translation_xyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = np.asarray(translation_xyz, dtype=float).reshape(3)
    return transform


def make_rigid_transform(rotation_matrix: np.ndarray, translation_xyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation_matrix, dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation_xyz, dtype=float).reshape(3)
    return transform


def make_yaw_transform(yaw_rad: float, translation_xyz: np.ndarray) -> np.ndarray:
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    rotation = np.asarray(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return make_rigid_transform(rotation, translation_xyz)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize test_base_sampler coordinate_debug.npz in MeshCat."
    )
    parser.add_argument(
        "npz",
        nargs="?",
        type=Path,
        default=None,
        help="Path to coordinate_debug.npz. Defaults to the latest test_base_sampler output.",
    )
    parser.add_argument(
        "--zmq-url",
        default="tcp://127.0.0.1:6000",
        help="MeshCat ZMQ URL. Defaults to tcp://127.0.0.1:6000.",
    )
    parser.add_argument(
        "--show-live-points",
        action="store_true",
        help="Also draw live_points_pb_world_xyz as cyan points.",
    )
    parser.add_argument(
        "--max-live-points",
        type=int,
        default=120000,
        help="Maximum live PB world points to draw when --show-live-points is enabled.",
    )
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=120000,
        help="Maximum voxel centers to draw as a red point cloud. 0 means all.",
    )
    parser.add_argument(
        "--voxel-point-size",
        type=float,
        default=0.012,
        help="MeshCat point size for voxel centers.",
    )
    parser.add_argument(
        "--live-point-size",
        type=float,
        default=0.004,
        help="MeshCat point size for live PB world points.",
    )
    parser.add_argument(
        "--show-voxel-boxes",
        action="store_true",
        help="Also draw a capped number of voxel centers as wireframe boxes.",
    )
    parser.add_argument(
        "--max-voxel-boxes",
        type=int,
        default=250,
        help="Maximum wireframe voxel boxes to draw.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.05,
        help="Voxel edge length in meters.",
    )
    parser.add_argument(
        "--no-keep-alive",
        dest="keep_alive",
        action="store_false",
        help="Exit after publishing the MeshCat scene.",
    )
    parser.set_defaults(keep_alive=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    npz_path = args.npz.expanduser().resolve() if args.npz is not None else _latest_coordinate_debug_npz().resolve()
    payload = _load_npz(npz_path)

    create_visualizer, make_frame, visualize_pointcloud, visualize_sphere, visualize_box = _import_meshcat_helpers(
        args.zmq_url
    )

    voxels = _maybe_subsample(_points(payload, "voxel_centers_pb_world_xyz"), args.max_voxels, seed=1)
    live_points = _maybe_subsample(_points(payload, "live_points_pb_world_xyz"), args.max_live_points, seed=2)
    targets = _points(payload, "target_positions_pb_world_xyz")
    target_rotations = np.asarray(
        payload.get("target_rotations_pb_world_matrix", np.empty((0, 3, 3))),
        dtype=float,
    ).reshape(-1, 3, 3)
    ranks = np.asarray(payload.get("grasp_ranks", np.empty((0,))), dtype=np.int32).reshape(-1)

    selected_rank = int(_scalar(payload, "selected_rank", -1))
    selected_target = _vector(payload, "selected_target_pb_world_xyz")
    selected_base = _vector(payload, "selected_pb_base_link_xyz")
    selected_base_yaw = _scalar(payload, "selected_pb_base_link_yaw_rad")
    selected_amcl = _vector(payload, "selected_pb_amcl_link_xyz")
    selected_amcl_yaw = _scalar(payload, "selected_pb_amcl_link_yaw_rad")
    camera_position = _vector(payload, "camera_position_pb_world_xyz")
    camera_rotation = np.asarray(
        payload.get("camera_to_pb_rotation_matrix", np.eye(3, dtype=float)),
        dtype=float,
    ).reshape(3, 3)

    print(f"Loaded coordinate debug NPZ: {npz_path}")
    print(f"  voxel_centers_pb_world_xyz shown: {len(voxels)}")
    print(f"  live_points_pb_world_xyz shown: {len(live_points) if args.show_live_points else 0}")
    print(f"  target count: {len(targets)}")
    print(f"  selected_rank: {selected_rank}")
    print(f"  selected_target_pb_world_xyz: {_format_xyz(selected_target)}")
    print(f"  selected_pb_base_link_xyz: {_format_xyz(selected_base)} yaw={selected_base_yaw:.6f}rad")
    print(f"  selected_pb_amcl_link_xyz: {_format_xyz(selected_amcl)} yaw={selected_amcl_yaw:.6f}rad")
    print(f"Expect a running MeshCat server on {args.zmq_url}")

    vis = create_visualizer(clear=True)
    print(f"MeshCat URL: {vis.url()}")
    try:
        vis.open()
    except Exception:
        pass

    make_frame(vis, "frames/pb_world", h=0.22, radius=0.004, opacity=0.9, transform=np.eye(4, dtype=float))

    if len(camera_position) >= 3:
        visualize_sphere(
            vis,
            "markers/camera_pb_world",
            camera_position[:3],
            CAMERA_COLOR,
            radius=0.025,
            opacity=1.0,
        )
        make_frame(
            vis,
            "frames/camera_pb_world",
            h=0.15,
            radius=0.004,
            opacity=0.95,
            transform=make_rigid_transform(camera_rotation, camera_position[:3]),
        )

    if len(voxels) > 0:
        visualize_pointcloud(
            vis,
            "scene/voxel_centers_pb_world",
            voxels,
            VOXEL_COLOR,
            size=float(args.voxel_point_size),
        )

    if args.show_live_points and len(live_points) > 0:
        visualize_pointcloud(
            vis,
            "scene/live_points_pb_world",
            live_points,
            POINT_COLOR,
            size=float(args.live_point_size),
        )

    for target_index, target in enumerate(targets):
        rank = int(ranks[target_index]) if target_index < len(ranks) else target_index + 1
        selected = rank == selected_rank
        target_name = f"targets/G{rank:02d}"
        visualize_sphere(
            vis,
            f"{target_name}/marker",
            target,
            TARGET_COLOR,
            radius=0.035 if selected else 0.022,
            opacity=1.0 if selected else 0.45,
        )
        if target_index < len(target_rotations):
            make_frame(
                vis,
                f"{target_name}/frame",
                h=0.13 if selected else 0.09,
                radius=0.004 if selected else 0.003,
                opacity=1.0 if selected else 0.55,
                transform=make_rigid_transform(target_rotations[target_index], target),
            )

    if len(selected_target) >= 3:
        visualize_sphere(
            vis,
            "selected/target_pb_world",
            selected_target[:3],
            TARGET_COLOR,
            radius=0.045,
            opacity=1.0,
        )

    if len(selected_base) >= 3:
        visualize_sphere(
            vis,
            "selected/base_link_pb_world",
            selected_base[:3],
            BASE_COLOR,
            radius=0.04,
            opacity=1.0,
        )
        make_frame(
            vis,
            "selected/base_link_frame",
            h=0.20,
            radius=0.005,
            opacity=1.0,
            transform=make_yaw_transform(selected_base_yaw, selected_base[:3]),
        )

    if len(selected_amcl) >= 3:
        visualize_sphere(
            vis,
            "selected/amcl_link_pb_world",
            selected_amcl[:3],
            AMCL_COLOR,
            radius=0.03,
            opacity=1.0,
        )
        make_frame(
            vis,
            "selected/amcl_link_frame",
            h=0.15,
            radius=0.004,
            opacity=1.0,
            transform=make_yaw_transform(selected_amcl_yaw, selected_amcl[:3]),
        )

    if args.show_voxel_boxes and len(voxels) > 0:
        voxel_boxes = voxels[: max(int(args.max_voxel_boxes), 0)]
        dims = np.asarray([args.voxel_size, args.voxel_size, args.voxel_size], dtype=float)
        for index, center in enumerate(voxel_boxes):
            visualize_box(
                vis,
                f"voxels/boxes/{index:04d}",
                center,
                dims,
                VOXEL_COLOR,
                wireframe=True,
            )
        print(f"  wireframe voxel boxes shown: {len(voxel_boxes)}/{len(voxels)}")

    print("Visualization uploaded to MeshCat.")
    print("  scene/voxel_centers_pb_world : red voxel-center point cloud")
    if args.show_live_points:
        print("  scene/live_points_pb_world   : cyan transformed live depth points")
    print("  targets/G*/frame             : target grasp frames")
    print("  selected/base_link_frame     : selected PB base_link frame")
    print("  selected/amcl_link_frame     : selected PB amcl_link frame")
    print("  frames/camera_pb_world       : camera frame in PB world")

    if not args.keep_alive:
        return

    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
