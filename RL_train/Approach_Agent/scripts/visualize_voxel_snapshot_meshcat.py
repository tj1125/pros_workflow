from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np


SCENE_COLOR = [160, 160, 160]
VOXEL_COLOR = [235, 60, 60]
ANCHOR_COLOR = [40, 170, 255]
TARGET_COLOR = [255, 215, 40]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a camera_car_voxel_ompl voxel_snapshot.npz in MeshCat."
    )
    parser.add_argument(
        "voxel_snapshot",
        type=Path,
        help="Path to voxel_snapshot.npz.",
    )
    parser.add_argument(
        "--max-scene-points",
        type=int,
        default=120000,
        help="Maximum number of scene points to display.",
    )
    parser.add_argument(
        "--scene-point-size",
        type=float,
        default=0.003,
        help="MeshCat point size for the scene cloud.",
    )
    parser.add_argument(
        "--show-scene-points",
        action="store_true",
        help="Overlay the raw scene point cloud in addition to the generated voxel result.",
    )
    parser.add_argument(
        "--voxel-point-size",
        type=float,
        default=0.012,
        help="MeshCat point size for voxel centers.",
    )
    parser.add_argument(
        "--show-voxel-boxes",
        action="store_true",
        help="Also render voxel centers as wireframe boxes in MeshCat.",
    )
    parser.add_argument(
        "--max-voxel-boxes",
        type=int,
        default=250,
        help="Maximum number of voxel boxes to draw when --show-voxel-boxes is set.",
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


def import_meshcat_helpers():
    try:
        import meshcat
        import meshcat.geometry as g
        import meshcat.transformations as mtf
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            "Failed to import MeshCat visualization dependencies. "
            f"Missing module: {missing}. "
            "Install the dependency first, for example: `pip install meshcat`."
        ) from exc

    def rgb2hex(rgb: tuple[int, int, int]) -> int:
        return int("0x%02x%02x%02x" % rgb, 16)

    def create_visualizer(clear: bool = True):
        vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
        if clear:
            vis.delete()
        return vis

    def make_frame(
        vis: Any,
        name: str,
        h: float = 0.15,
        radius: float = 0.01,
        o: float = 1.0,
        T: np.ndarray | None = None,
    ) -> None:
        vis[name]["x"].set_object(
            g.Cylinder(height=h, radius=radius),
            g.MeshLambertMaterial(color=0xFF0000, reflectivity=0.8, opacity=o),
        )
        rotate_x = mtf.rotation_matrix(np.pi / 2.0, [0, 0, 1])
        rotate_x[0, 3] = h / 2
        vis[name]["x"].set_transform(rotate_x)

        vis[name]["y"].set_object(
            g.Cylinder(height=h, radius=radius),
            g.MeshLambertMaterial(color=0x00FF00, reflectivity=0.8, opacity=o),
        )
        rotate_y = mtf.rotation_matrix(np.pi / 2.0, [0, 1, 0])
        rotate_y[1, 3] = h / 2
        vis[name]["y"].set_transform(rotate_y)

        vis[name]["z"].set_object(
            g.Cylinder(height=h, radius=radius),
            g.MeshLambertMaterial(color=0x0000FF, reflectivity=0.8, opacity=o),
        )
        rotate_z = mtf.rotation_matrix(np.pi / 2.0, [1, 0, 0])
        rotate_z[2, 3] = h / 2
        vis[name]["z"].set_transform(rotate_z)

        if T is not None:
            vis[name].set_transform(np.asarray(T, dtype=float))

    def visualize_bbox(
        vis: Any,
        name: str,
        dims: np.ndarray,
        T: np.ndarray | None = None,
        color: list[int] | None = None,
    ) -> None:
        color = color or [255, 0, 0]
        material = g.MeshBasicMaterial(wireframe=True, color=rgb2hex(tuple(color)))
        bbox = g.Box(np.asarray(dims, dtype=float))
        vis[name].set_object(bbox, material)
        if T is not None:
            vis[name].set_transform(np.asarray(T, dtype=float))

    def visualize_pointcloud(
        vis: Any,
        name: str,
        pc: np.ndarray,
        color: np.ndarray | list[int] | None = None,
        transform: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        pc = np.asarray(pc, dtype=np.float32)
        if pc.ndim == 3:
            pc = pc.reshape(-1, pc.shape[-1])
        if len(pc) == 0:
            return

        if color is None:
            color_array = np.ones_like(pc, dtype=np.float32)
        else:
            color_array = np.asarray(color, dtype=np.float32)
            if color_array.ndim == 1:
                color_array = np.ones_like(pc, dtype=np.float32) * color_array.reshape(1, 3)
            elif color_array.ndim == 3:
                color_array = color_array.reshape(-1, color_array.shape[-1])
            color_array = color_array / 255.0

        vis[name].set_object(
            g.PointCloud(position=pc.T, color=color_array.T, **kwargs)
        )
        if transform is not None:
            vis[name].set_transform(np.asarray(transform, dtype=float))

    return create_visualizer, make_frame, visualize_bbox, visualize_pointcloud


def load_npz(path: Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(str(path), allow_pickle=True)


def get_points(
    data: np.lib.npyio.NpzFile,
    *keys: str,
) -> np.ndarray:
    for key in keys:
        if key in data.files:
            points = np.asarray(data[key], dtype=float)
            if points.ndim == 2 and points.shape[1] == 3:
                return points
    raise KeyError(f"None of the expected point-cloud keys were found: {keys}")


def maybe_get_xyz(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data.files:
        return None
    value = np.asarray(data[key], dtype=float).reshape(-1)
    if value.shape != (3,):
        raise ValueError(f"{key} must contain exactly 3 values, got shape {value.shape}.")
    return value


def maybe_subsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def make_translation_transform(translation_xyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = np.asarray(translation_xyz, dtype=float)
    return transform


def make_rigid_transform(rotation_matrix: np.ndarray, translation_xyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation_matrix, dtype=float)
    transform[:3, 3] = np.asarray(translation_xyz, dtype=float)
    return transform


def main() -> None:
    args = parse_args()
    voxel_snapshot = args.voxel_snapshot.expanduser().resolve()

    create_visualizer, make_frame, visualize_bbox, visualize_pointcloud = import_meshcat_helpers()

    data = load_npz(voxel_snapshot)
    scene_points = get_points(
        data,
        "points_pybullet",
        "points_camera_selected_pybullet_basis",
        "points_camera_selected",
        "points_camera",
    )
    voxel_centers = get_points(data, "voxel_centers_pybullet", "voxel_centers_camera")
    ee_anchor = maybe_get_xyz(data, "ee_anchor_pybullet_xyz")
    target_position = maybe_get_xyz(data, "target_position_pybullet_xyz")
    gripper_midpoint_camera = maybe_get_xyz(data, "gripper_midpoint_camera_xyz")
    gripper_midpoint_pybullet = maybe_get_xyz(data, "gripper_midpoint_pybullet_xyz")
    scene_translation = maybe_get_xyz(data, "scene_translation_xyz")
    camera_to_pybullet_rotation = None
    if "camera_to_pybullet_rotation_matrix" in data.files:
        camera_to_pybullet_rotation = np.asarray(data["camera_to_pybullet_rotation_matrix"], dtype=float)
        if camera_to_pybullet_rotation.shape != (3, 3):
            raise ValueError(
                "camera_to_pybullet_rotation_matrix must have shape (3, 3), "
                f"got {camera_to_pybullet_rotation.shape}."
            )
    target_rotation_pybullet = None
    if "target_rotation_pybullet_matrix" in data.files:
        target_rotation_pybullet = np.asarray(data["target_rotation_pybullet_matrix"], dtype=float)
        if target_rotation_pybullet.shape != (3, 3):
            raise ValueError(
                "target_rotation_pybullet_matrix must have shape (3, 3), "
                f"got {target_rotation_pybullet.shape}."
            )

    scene_points_vis = maybe_subsample(scene_points, args.max_scene_points, seed=0)

    print(f"Loaded voxel snapshot: {voxel_snapshot}")
    print(f"  scene points: {len(scene_points)}")
    print(f"  scene points shown: {len(scene_points_vis) if args.show_scene_points else 0}")
    print(f"  voxel centers: {len(voxel_centers)}")
    if ee_anchor is not None:
        print(f"  ee_anchor_pybullet_xyz: {ee_anchor.round(4).tolist()}")
    if target_position is not None:
        print(f"  target_position_pybullet_xyz: {target_position.round(4).tolist()}")
    if gripper_midpoint_camera is not None:
        print(f"  gripper_midpoint_camera_xyz: {gripper_midpoint_camera.round(4).tolist()}")
    if gripper_midpoint_pybullet is not None:
        print(f"  gripper_midpoint_pybullet_xyz: {gripper_midpoint_pybullet.round(4).tolist()}")
    if scene_translation is not None:
        print(f"  scene_translation_xyz: {scene_translation.round(4).tolist()}")
    print("Expect a running MeshCat server on tcp://127.0.0.1:6000")

    vis = create_visualizer(clear=True)
    print(f"MeshCat URL: {vis.url()}")

    if args.show_scene_points and len(scene_points_vis) > 0:
        visualize_pointcloud(
            vis,
            "scene/points",
            scene_points_vis,
            SCENE_COLOR,
            size=args.scene_point_size,
        )

    if len(voxel_centers) > 0:
        visualize_pointcloud(
            vis,
            "scene/voxels",
            voxel_centers,
            VOXEL_COLOR,
            size=args.voxel_point_size,
        )

    make_frame(vis, "frames/world", h=0.12, radius=0.003, o=0.7, T=np.eye(4, dtype=float))
    if scene_translation is not None and camera_to_pybullet_rotation is not None:
        make_frame(
            vis,
            "frames/camera",
            h=0.11,
            radius=0.003,
            o=0.9,
            T=make_rigid_transform(camera_to_pybullet_rotation, scene_translation),
        )
    if ee_anchor is not None:
        make_frame(
            vis,
            "frames/ee_anchor",
            h=0.10,
            radius=0.003,
            o=0.9,
            T=make_translation_transform(ee_anchor),
        )
        visualize_pointcloud(
            vis,
            "markers/ee_anchor",
            ee_anchor.reshape(1, 3),
            ANCHOR_COLOR,
            size=max(args.voxel_point_size * 1.8, 0.02),
        )
    if target_position is not None:
        target_transform = make_translation_transform(target_position)
        if target_rotation_pybullet is not None:
            target_transform = make_rigid_transform(target_rotation_pybullet, target_position)
        make_frame(
            vis,
            "frames/target_pose",
            h=0.11,
            radius=0.003,
            o=0.95,
            T=target_transform,
        )
        visualize_pointcloud(
            vis,
            "markers/target_pose",
            target_position.reshape(1, 3),
            TARGET_COLOR,
            size=max(args.voxel_point_size * 1.8, 0.02),
        )

    if args.show_voxel_boxes and len(voxel_centers) > 0:
        max_voxel_boxes = max(int(args.max_voxel_boxes), 0)
        voxel_boxes = voxel_centers[:max_voxel_boxes]
        for index, center in enumerate(voxel_boxes):
            visualize_bbox(
                vis,
                f"voxels/boxes/{index:04d}",
                np.array([args.voxel_size, args.voxel_size, args.voxel_size], dtype=float),
                T=make_translation_transform(center),
                color=VOXEL_COLOR,
            )
        if len(voxel_centers) > max_voxel_boxes:
            print(
                f"  wireframe voxel boxes shown: {max_voxel_boxes}/{len(voxel_centers)} "
                "(capped by --max-voxel-boxes)"
            )
        else:
            print(f"  wireframe voxel boxes shown: {len(voxel_centers)}")

    print("Visualization uploaded to MeshCat.")
    print("  scene/voxels      : red voxel-center cloud")
    print("  frames/world      : world frame")
    if scene_translation is not None:
        print("  frames/camera     : camera frame mapped into the PyBullet scene")
    if args.show_scene_points:
        print("  scene/points      : gray scene cloud")
    if ee_anchor is not None:
        print("  frames/ee_anchor  : EE anchor frame")
        print("  markers/ee_anchor : blue EE anchor point")
    if target_position is not None:
        print("  frames/target_pose: target grasp pose frame")
        print("  markers/target_pose: yellow target grasp point")
    if args.show_voxel_boxes:
        print("  voxels/boxes/*    : optional wireframe voxel boxes")

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
