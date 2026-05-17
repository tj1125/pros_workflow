#!/usr/bin/env python3
"""
Visualize grasp_agent/data/debug_outputs/latest_grasp_debug.npz in MeshCat.

This script is tailored for the NPZ bundle written by:
3090server/VLM_RL/grasp_agent/pipeline/pipeline.py

Expected workflow:
1. Start a MeshCat server in another terminal: `meshcat-server`
2. Run this script.
3. Open the printed MeshCat URL in a browser.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_ROOT = REPO_ROOT / "3090server" / "VLM_RL"
DEFAULT_DEBUG_NPZ = (
    SERVER_ROOT / "grasp_agent" / "data" / "debug_outputs" / "latest_grasp_debug.npz"
)
DEFAULT_GRIPPER_CONFIG = (
    SERVER_ROOT / "models" / "graspgen_checkpoints" / "graspgen_robotiq_2f_140.yml"
)
VENDOR_ROOT_CANDIDATES = [
    SERVER_ROOT / "get_item_info_agent" / "vendor" / "graspgen_runtime",
    SERVER_ROOT / "get_item_info_agent_no_sam3d" / "vendor" / "graspgen_runtime",
]

SCENE_COLOR = [150, 150, 150]
OBJECT_COLOR = [0, 220, 80]
BEST_GRASP_COLOR = [255, 210, 40]
COLLIDING_COLOR = [220, 60, 60]
GRIPPER_MIDPOINT_COLOR = [40, 170, 255]
MAX_VIS_PITCH_DEG = 15.0
DISABLE_VISIBILITY_FILTERS_FOR = {"latest_grasp_visualization.npz"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize latest_grasp_debug.npz in MeshCat."
    )
    parser.add_argument(
        "debug_npz",
        nargs="?",
        default=str(DEFAULT_DEBUG_NPZ),
        help="Path to latest_grasp_debug.npz",
    )
    parser.add_argument(
        "--gripper-config",
        default=str(DEFAULT_GRIPPER_CONFIG),
        help="Path to GraspGen gripper config yaml.",
    )
    parser.add_argument(
        "--max-scene-points",
        type=int,
        default=120000,
        help="Maximum number of scene points to display.",
    )
    parser.add_argument(
        "--max-colliding",
        type=int,
        default=20,
        help="Maximum number of colliding grasps to display when --show-colliding is set.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Show only the top-k highest-scoring collision-free grasps.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Only show collision-free grasps whose score is >= this threshold.",
    )
    parser.add_argument(
        "--scene-point-size",
        type=float,
        default=0.003,
        help="MeshCat point size for the scene cloud.",
    )
    parser.add_argument(
        "--object-point-size",
        type=float,
        default=0.006,
        help="MeshCat point size for the object cloud.",
    )
    parser.add_argument(
        "--display-frame",
        choices=("image_aligned", "camera_raw", "camera_y_up"),
        default="camera_raw",
        help=(
            "How to display Camera_Car coordinates in MeshCat. "
            "'camera_raw' keeps the OpenCV camera frame (x right, y down, z forward). "
            "'camera_y_up' flips only y for a conventional up-axis display. "
            "'image_aligned' flips x and y so the MeshCat front view matches the RGB image more intuitively."
        ),
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Only show the best grasp and hide the rest.",
    )
    parser.add_argument(
        "--show-colliding",
        action="store_true",
        help="Also show rejected grasps from all_grasps_local.",
    )
    parser.add_argument(
        "--no-keep-alive",
        dest="keep_alive",
        action="store_false",
        help="Exit after publishing the MeshCat scene.",
    )
    parser.set_defaults(keep_alive=True)
    return parser.parse_args()


def resolve_vendor_root() -> Path:
    for candidate in VENDOR_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate graspgen_runtime under 3090server/VLM_RL. "
        f"Tried: {VENDOR_ROOT_CANDIDATES}"
    )


def add_vendor_paths(vendor_root: Path) -> None:
    for extra in (vendor_root, vendor_root / "pointnet2_ops"):
        if extra.exists():
            extra_str = str(extra)
            if extra_str not in sys.path:
                sys.path.insert(0, extra_str)


def import_meshcat_helpers():
    try:
        from grasp_gen.utils.meshcat_utils import (  # type: ignore
            create_visualizer,
            get_color_from_score,
            make_frame,
            visualize_grasp,
            visualize_pointcloud,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise SystemExit(
            "Failed to import MeshCat visualization dependencies. "
            f"Missing module: {missing}. "
            "Install the dependency first, for example: `pip install meshcat`."
        ) from exc

    return (
        create_visualizer,
        get_color_from_score,
        make_frame,
        visualize_grasp,
        visualize_pointcloud,
    )


def load_gripper_name(gripper_config: Path) -> str:
    if not gripper_config.exists():
        return "robotiq_2f_140"
    payload = yaml.safe_load(gripper_config.read_text(encoding="utf-8")) or {}
    data = payload.get("data", {})
    gripper_name = data.get("gripper_name")
    return str(gripper_name) if gripper_name else "robotiq_2f_140"


def load_npz(path: Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(str(path), allow_pickle=True)


def is_centered_visualization_npz(data: np.lib.npyio.NpzFile) -> bool:
    return "pc_object" in data.files and "all_grasps" in data.files


def scalar_value(data: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in data.files:
        return default
    value = data[key]
    if getattr(value, "shape", None) == ():
        return str(value.item())
    return str(value)


def get_array(
    data: np.lib.npyio.NpzFile,
    key: str,
    *,
    fallback: np.ndarray | None = None,
    dtype: type | None = float,
) -> np.ndarray:
    if key in data.files:
        value = np.asarray(data[key], dtype=dtype)
        return value
    if fallback is None:
        raise KeyError(f"Required key not found in NPZ: {key}")
    return np.asarray(fallback, dtype=dtype)


def maybe_subsample(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def local_grasps_to_camera(
    grasps_local: np.ndarray,
    object_center_camera: np.ndarray,
) -> np.ndarray:
    grasps_local = np.asarray(grasps_local, dtype=float)
    if grasps_local.size == 0:
        return np.zeros((0, 4, 4), dtype=float)
    if grasps_local.ndim == 2:
        grasps_local = grasps_local[None, ...]
    grasps_camera = np.array(grasps_local, copy=True)
    grasps_camera[:, :3, 3] += np.asarray(object_center_camera, dtype=float)[None, :]
    return grasps_camera


def make_translation_transform(translation_xyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = np.asarray(translation_xyz, dtype=float)
    return transform


def display_transform(mode: str) -> np.ndarray:
    if mode == "camera_raw":
        return np.eye(4, dtype=float)
    if mode == "camera_y_up":
        return np.diag([1.0, -1.0, 1.0, 1.0]).astype(float)
    if mode == "image_aligned":
        return np.diag([-1.0, -1.0, 1.0, 1.0]).astype(float)
    raise ValueError(f"Unsupported display mode: {mode}")


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return points.reshape(-1, 3)
    points_h = np.concatenate(
        [points, np.ones((len(points), 1), dtype=float)],
        axis=1,
    )
    transformed = (transform @ points_h.T).T
    return transformed[:, :3]


def transform_pose(pose: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(transform, dtype=float) @ np.asarray(pose, dtype=float)


def grasp_pitch_degrees(grasps: np.ndarray) -> np.ndarray:
    grasps = np.asarray(grasps, dtype=float)
    if grasps.size == 0:
        return np.zeros((0,), dtype=float)
    if grasps.ndim == 2:
        grasps = grasps[None, ...]

    rotation = grasps[:, :3, :3]
    sy = np.sqrt(rotation[:, 0, 0] ** 2 + rotation[:, 1, 0] ** 2)
    return np.degrees(np.arctan2(-rotation[:, 2, 0], sy))


def filter_grasps_by_pitch(
    grasps: np.ndarray,
    *,
    max_abs_pitch_deg: float = MAX_VIS_PITCH_DEG,
    scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    grasps = np.asarray(grasps, dtype=float)
    if grasps.size == 0:
        empty_grasps = np.zeros((0, 4, 4), dtype=float)
        empty_pitch = np.zeros((0,), dtype=float)
        if scores is None:
            return empty_grasps, None, empty_pitch
        return empty_grasps, np.zeros((0,), dtype=float), empty_pitch

    pitch_deg = grasp_pitch_degrees(grasps)
    keep = np.abs(pitch_deg) <= float(max_abs_pitch_deg)
    filtered_grasps = grasps[keep]
    filtered_pitch = pitch_deg[keep]
    if scores is None:
        return filtered_grasps, None, filtered_pitch

    scores = np.asarray(scores, dtype=float)
    return filtered_grasps, scores[keep], filtered_pitch


def filter_grasps_to_object_front(
    grasps_camera: np.ndarray,
    object_center_camera: np.ndarray,
    *,
    scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Keep only grasps whose origin is in front of the object center in camera Z."""
    grasps_camera = np.asarray(grasps_camera, dtype=float)
    if grasps_camera.size == 0:
        empty = np.zeros((0, 4, 4), dtype=float)
        if scores is None:
            return empty, None
        return empty, np.zeros((0,), dtype=float)

    if grasps_camera.ndim == 2:
        grasps_camera = grasps_camera[None, ...]
    center_z = float(np.asarray(object_center_camera, dtype=float)[2])
    keep = grasps_camera[:, 2, 3] <= center_z
    filtered_grasps = grasps_camera[keep]
    if scores is None:
        return filtered_grasps, None

    scores = np.asarray(scores, dtype=float)
    return filtered_grasps, scores[keep]


def select_topk_grasps(
    grasps: np.ndarray,
    scores: np.ndarray,
    topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    grasps = np.asarray(grasps, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(grasps) == 0 or len(scores) == 0 or topk <= 0:
        return np.zeros((0, 4, 4), dtype=float), np.zeros((0,), dtype=float)
    order = np.argsort(-scores)
    keep = order[: min(topk, len(order))]
    return grasps[keep], scores[keep]


def build_free_grasps_camera(
    data: np.lib.npyio.NpzFile,
    object_center_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if "collision_free_grasps" in data.files:
        grasps = get_array(data, "collision_free_grasps", fallback=np.zeros((0, 4, 4)))
        scores = get_array(data, "collision_free_scores", fallback=np.zeros((0,)))
        return np.asarray(grasps, dtype=float), np.asarray(scores, dtype=float)

    if "collision_free_grasps_local" in data.files:
        grasps_local = get_array(
            data,
            "collision_free_grasps_local",
            fallback=np.zeros((0, 4, 4)),
        )
        scores = get_array(data, "collision_free_scores", fallback=np.zeros((0,)))
        return local_grasps_to_camera(grasps_local, object_center_camera), scores

    all_grasps_local = get_array(data, "all_grasps_local", fallback=np.zeros((0, 4, 4)))
    all_scores = get_array(data, "all_scores", fallback=np.zeros((0,)))
    mask = get_array(
        data,
        "collision_free_mask",
        fallback=np.ones(len(all_grasps_local), dtype=bool),
        dtype=bool,
    )
    return local_grasps_to_camera(all_grasps_local[mask], object_center_camera), all_scores[mask]


def build_colliding_grasps_camera(
    data: np.lib.npyio.NpzFile,
    object_center_camera: np.ndarray,
) -> np.ndarray:
    if "all_grasps" in data.files:
        all_grasps = get_array(data, "all_grasps", fallback=np.zeros((0, 4, 4)))
        mask = get_array(
            data,
            "collision_free_mask",
            fallback=np.ones(len(all_grasps), dtype=bool),
            dtype=bool,
        )
        return np.asarray(all_grasps[~mask], dtype=float)

    if "all_grasps_local" not in data.files:
        return np.zeros((0, 4, 4), dtype=float)
    all_grasps_local = get_array(data, "all_grasps_local", fallback=np.zeros((0, 4, 4)))
    mask = get_array(
        data,
        "collision_free_mask",
        fallback=np.ones(len(all_grasps_local), dtype=bool),
        dtype=bool,
    )
    return local_grasps_to_camera(all_grasps_local[~mask], object_center_camera)


def should_apply_visibility_filters(debug_npz: Path) -> bool:
    return debug_npz.name not in DISABLE_VISIBILITY_FILTERS_FOR


def main() -> None:
    args = parse_args()
    debug_npz = Path(args.debug_npz).expanduser().resolve()
    gripper_config = Path(args.gripper_config).expanduser().resolve()

    vendor_root = resolve_vendor_root()
    add_vendor_paths(vendor_root)
    (
        create_visualizer,
        get_color_from_score,
        make_frame,
        visualize_grasp,
        visualize_pointcloud,
    ) = import_meshcat_helpers()

    data = load_npz(debug_npz)
    apply_visibility_filters = should_apply_visibility_filters(debug_npz)
    centered_visualization_npz = is_centered_visualization_npz(data)

    object_id = scalar_value(
        data,
        "object_id",
        default=scalar_value(data, "target_label", default="unknown"),
    )
    camera_name = scalar_value(
        data,
        "camera_name",
        default=scalar_value(data, "primary_camera_id", default="unknown"),
    )
    object_center_camera = (
        np.zeros(3, dtype=float)
        if centered_visualization_npz
        else get_array(data, "object_reference_center_camera", fallback=np.zeros(3))
    )
    gripper_midpoint_camera = get_array(
        data,
        "gripper_midpoint_camera_xyz",
        fallback=(
            np.zeros(3, dtype=float)
            if centered_visualization_npz
            else np.array([0.0, -0.04, 0.11], dtype=float)
        ),
    )
    object_pc_camera = get_array(
        data,
        "object_pc_camera" if not centered_visualization_npz else "pc_object",
        fallback=(
            get_array(data, "object_pc_local") + object_center_camera[None, :]
            if not centered_visualization_npz
            else np.zeros((0, 3), dtype=float)
        ),
    )
    scene_pc_camera = get_array(
        data,
        "scene_pc_camera" if not centered_visualization_npz else "pc_scene",
        fallback=np.zeros((0, 3)),
    )
    all_scores = get_array(data, "all_scores", fallback=np.zeros((0,)))
    free_grasps_camera, free_scores = build_free_grasps_camera(data, object_center_camera)
    colliding_grasps_camera = build_colliding_grasps_camera(data, object_center_camera)
    if "best_grasp_camera" in data.files:
        best_grasp_camera = get_array(data, "best_grasp_camera")
    elif "best_grasp_local" in data.files:
        best_grasp_camera = local_grasps_to_camera(
            get_array(data, "best_grasp_local", fallback=np.eye(4)),
            object_center_camera,
        )[0]
    elif len(free_grasps_camera) > 0:
        best_idx = int(np.argmax(free_scores)) if len(free_scores) > 0 else 0
        best_grasp_camera = np.asarray(free_grasps_camera[best_idx], dtype=float)
    elif len(colliding_grasps_camera) > 0:
        best_grasp_camera = np.asarray(colliding_grasps_camera[0], dtype=float)
    else:
        best_grasp_camera = np.eye(4, dtype=float)

    best_grasp_pitch_deg = grasp_pitch_degrees(best_grasp_camera)[0]
    if apply_visibility_filters:
        free_grasps_camera, free_scores = filter_grasps_to_object_front(
            free_grasps_camera,
            object_center_camera,
            scores=free_scores,
        )
        colliding_grasps_camera, _ = filter_grasps_to_object_front(
            colliding_grasps_camera,
            object_center_camera,
        )
        free_grasps_camera, free_scores, _ = filter_grasps_by_pitch(
            free_grasps_camera,
            scores=free_scores,
        )
        colliding_grasps_camera, _, _ = filter_grasps_by_pitch(
            colliding_grasps_camera,
        )

    if args.score_threshold is not None and len(free_scores) > 0:
        keep = free_scores >= args.score_threshold
        free_grasps_camera = free_grasps_camera[keep]
        free_scores = free_scores[keep]

    free_grasps_camera, free_scores = select_topk_grasps(
        free_grasps_camera,
        free_scores,
        args.topk,
    )

    if apply_visibility_filters:
        free_pitch_deg = grasp_pitch_degrees(free_grasps_camera)
        best_grasp_filtered_out = (
            abs(best_grasp_pitch_deg) > MAX_VIS_PITCH_DEG
            or float(best_grasp_camera[2, 3]) > float(object_center_camera[2])
        )
        best_grasp_visualized = False
        if best_grasp_filtered_out:
            if len(free_grasps_camera) > 0:
                best_grasp_camera = np.asarray(free_grasps_camera[0], dtype=float)
                best_grasp_pitch_deg = float(free_pitch_deg[0])
                best_grasp_visualized = True
            else:
                best_grasp_camera = None
        else:
            best_grasp_visualized = True
    else:
        best_grasp_visualized = best_grasp_camera is not None

    display_T = display_transform(args.display_frame)
    object_pc_display = transform_points(object_pc_camera, display_T)
    scene_pc_display = transform_points(scene_pc_camera, display_T)
    object_center_display = transform_points(object_center_camera.reshape(1, 3), display_T)[0]
    gripper_midpoint_display = transform_points(gripper_midpoint_camera.reshape(1, 3), display_T)[0]
    best_grasp_display = (
        transform_pose(best_grasp_camera, display_T)
        if best_grasp_camera is not None
        else None
    )
    free_grasps_display = np.asarray(
        [transform_pose(grasp, display_T) for grasp in free_grasps_camera],
        dtype=float,
    ) if len(free_grasps_camera) else np.zeros((0, 4, 4), dtype=float)
    colliding_grasps_display = np.asarray(
        [transform_pose(grasp, display_T) for grasp in colliding_grasps_camera],
        dtype=float,
    ) if len(colliding_grasps_camera) else np.zeros((0, 4, 4), dtype=float)

    scene_display = maybe_subsample(scene_pc_display, args.max_scene_points, seed=0)
    gripper_name = load_gripper_name(gripper_config)

    print(f"Loaded debug NPZ: {debug_npz}")
    print(f"  object_id: {object_id}")
    print(f"  camera_name: {camera_name}")
    print(f"  object points: {len(object_pc_camera)}")
    print(f"  scene points: {len(scene_pc_camera)}")
    print(f"  all grasps: {len(all_scores)}")
    print(f"  centered visualization npz: {centered_visualization_npz}")
    print(f"  gripper midpoint (camera): {gripper_midpoint_camera.round(4).tolist()}")
    if apply_visibility_filters:
        print(f"  pitch filter: +/-{MAX_VIS_PITCH_DEG:.1f} deg")
        print("  position filter: grasp origin must satisfy grasp_z <= object_center_z")
    else:
        print("  pitch filter: disabled")
        print("  position filter: disabled")
    print(f"  visualized top-k collision-free grasps: {len(free_scores)} (topk={args.topk})")
    print(f"  visualized colliding grasps: {len(colliding_grasps_display)}")
    print(f"  best grasp pitch (camera): {best_grasp_pitch_deg:.4f}")
    if best_grasp_camera is not None:
        print(f"  best grasp position (camera): {best_grasp_camera[:3, 3].round(4).tolist()}")
    else:
        print("  best grasp position (camera): hidden by visibility filters")
    print(f"  display_frame: {args.display_frame}")
    print(f"  gripper: {gripper_name}")
    print("Expect a running MeshCat server on tcp://127.0.0.1:6000")

    vis = create_visualizer(clear=True)
    print(f"MeshCat URL: {vis.url()}")

    if len(scene_display) > 0:
        visualize_pointcloud(
            vis,
            "scene_pc",
            scene_display,
            SCENE_COLOR,
            size=args.scene_point_size,
        )
    if len(object_pc_camera) > 0:
        visualize_pointcloud(
            vis,
            "object_pc",
            object_pc_display,
            OBJECT_COLOR,
            size=args.object_point_size,
        )

    make_frame(vis, "frames/camera", h=0.10, radius=0.003, o=0.7, T=display_T)
    make_frame(
        vis,
        "frames/object_center",
        h=0.08,
        radius=0.003,
        o=0.8,
        T=make_translation_transform(object_center_display),
    )
    make_frame(
        vis,
        "frames/gripper_midpoint",
        h=0.08,
        radius=0.003,
        o=0.9,
        T=make_translation_transform(gripper_midpoint_display),
    )
    visualize_pointcloud(
        vis,
        "markers/gripper_midpoint",
        gripper_midpoint_display.reshape(1, 3),
        GRIPPER_MIDPOINT_COLOR,
        size=max(args.object_point_size * 3.0, 0.02),
    )
    if best_grasp_display is not None:
        make_frame(
            vis,
            "frames/best_grasp",
            h=0.10,
            radius=0.003,
            o=0.9,
            T=best_grasp_display,
        )

        visualize_grasp(
            vis,
            "grasps/best",
            best_grasp_display,
            color=BEST_GRASP_COLOR,
            gripper_name=gripper_name,
            linewidth=2.5,
        )

    if not args.best_only and len(free_grasps_display) > 0:
        free_colors = np.asarray(
            get_color_from_score(free_scores, use_255_scale=True),
            dtype=np.uint8,
        )
        for idx, (grasp_camera, color) in enumerate(zip(free_grasps_display, free_colors)):
            visualize_grasp(
                vis,
                f"grasps/collision_free/{idx:03d}",
                grasp_camera,
                color=color.tolist(),
                gripper_name=gripper_name,
                linewidth=1.1,
            )

    if args.show_colliding and len(colliding_grasps_display) > 0:
        for idx, grasp_camera in enumerate(colliding_grasps_display[: args.max_colliding]):
            visualize_grasp(
                vis,
                f"grasps/colliding/{idx:03d}",
                grasp_camera,
                color=COLLIDING_COLOR,
                gripper_name=gripper_name,
                linewidth=0.5,
            )

    print("Visualization uploaded to MeshCat.")
    print("  scene_pc          : gray background cloud")
    print("  object_pc         : green object cloud")
    print("  frames/*          : camera/object/gripper midpoint/best grasp frames")
    print("  markers/gripper_midpoint : blue point for the gripper midpoint")
    if apply_visibility_filters:
        print(f"  note              : only grasps with |pitch| <= {MAX_VIS_PITCH_DEG:.1f} deg are visualized")
        print("  note              : grasps behind the object center in camera Z are hidden")
    else:
        print("  note              : pitch filtering disabled for this NPZ")
        print("  note              : front/back grasp hiding disabled for this NPZ")
    if centered_visualization_npz:
        print("  note              : pc_object / pc_scene / grasps are visualized in the centered frame stored in the NPZ")
    if best_grasp_visualized:
        print("  grasps/best       : highlighted best grasp")
    else:
        print("  grasps/best       : hidden because no grasp remained after visibility filters")
    print("  note              : image_aligned mode is for easier visual comparison with the RGB image, not raw OpenCV camera coordinates")
    if not args.best_only:
        print("  grasps/collision_free/* : top-k score-colored collision-free grasps")
    if args.show_colliding:
        print("  grasps/colliding/*      : rejected grasps in red")

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
