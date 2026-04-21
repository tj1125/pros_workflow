import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.spatial.transform as st

from scripts.run_base_pose_sampling import load_config
from src.camera_car_voxel_ompl import load_camera_car_voxel_ompl_config
from src.geometry.depth_backprojection import decode_depth_png_bytes
from src.geometry.map_loader import load_free_cells_unity_xz, load_map_meta
from src.geometry.voxelization import voxelize_points
from src.io.camera_capture import AmclPoseSnapshot, capture_rgbd_snapshot
from src.io.intrinsics import load_camera_intrinsics
from src.pybullet_ompl import (
    _add_debug_axes,
    _compute_pose_alignment_metrics,
    _find_controllable_joints,
    _set_joint_positions_direct,
    get_reset_camera_transform,
    load_planning_config,
)
from src.pybullet_smoke import (
    _degrees_to_radians,
    _load_arm_config,
    _load_python_dependencies,
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
class PreparedGraspTarget:
    grasp_candidate: GraspPoseCandidate
    target_pb: np.ndarray
    target_rot_pb: np.ndarray
    target_quat_pb: np.ndarray
    direct_ik_error_m: float
    direct_ik_error_xyz: np.ndarray
    direct_orientation_error_deg: float | None


@dataclass(frozen=True)
class LiveSceneCapture:
    voxel_centers_pb: np.ndarray
    camera_to_pb_rotation: np.ndarray
    camera_position_pb: np.ndarray
    amcl_pose: AmclPoseSnapshot | None
    valid_depth_point_count: int


@dataclass(frozen=True)
class RosMapPose2D:
    x: float
    y: float
    yaw_rad: float


@dataclass(frozen=True)
class BaseFootprintMap:
    free_cell_keys: frozenset[tuple[int, int]]
    free_cell_centers_xy: np.ndarray
    footprint_points_pb_xy: np.ndarray
    origin_xy: tuple[float, float]
    resolution_m: float
    length_x_m: float
    length_y_m: float


LIVE_VOXEL_MIN_DEPTH_M = 0.19
LIVE_VOXEL_MAX_DEPTH_M = 0.8
LIVE_VOXEL_SCENE_POINT_STRIDE = 4
LIVE_VOXEL_SCENE_DOWNSAMPLE_M = 0.01
BASE_LINK_YAW_MAX_DELTA_FROM_CURRENT_DEG = 15.0
BASE_FOOTPRINT_LENGTH_X_M = 0.33
BASE_FOOTPRINT_LENGTH_Y_M = 0.42
BASE_LINK_FROM_AMCL_PB_XY = np.asarray([0.0, 0.1288], dtype=np.float64)
BASE_SAMPLE_MIN_DISTANCE_M = 0.20
BASE_SAMPLE_MAX_DISTANCE_M = 0.35
BASE_SAMPLE_APPROACH_HALF_ANGLE_DEG = 20.0


PLANNING_CAMERA_TO_PB_LOCAL = np.asarray(
    [
        [0.0, 0.0, 1.0],   # pb_x =  cam_z
        [-1.0, 0.0, 0.0],  # pb_y = -cam_x
        [0.0, -1.0, 0.0],  # pb_z = -cam_y
    ],
    dtype=np.float64,
)


PLANNING_GRASP_TO_EE = np.asarray(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


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


def _resolve_grasp_result_json_path(grasp_result_json_path: Path | None = None) -> Path:
    env_override = os.getenv("GRASP_RESULT_JSON", "").strip()
    if grasp_result_json_path is not None:
        return grasp_result_json_path.expanduser().resolve()
    if env_override:
        return Path(env_override).expanduser().resolve()
    preferred = _preferred_grasp_result_json()
    if preferred.exists():
        return preferred.resolve()
    return _find_latest_grasp_result_json()


def _wrap_angle_rad(angle_rad: float) -> float:
    return float((float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi)


def _abs_angle_delta_rad(angle_a_rad: float, angle_b_rad: float) -> float:
    return abs(_wrap_angle_rad(float(angle_a_rad) - float(angle_b_rad)))


def _rectangular_footprint_points_pb_xy(
    *,
    length_x_m: float,
    length_y_m: float,
    resolution_m: float,
) -> np.ndarray:
    step = max(float(resolution_m) * 0.5, 0.01)
    half_x = float(length_x_m) * 0.5
    half_y = float(length_y_m) * 0.5
    x_values = np.arange(-half_x, half_x + step * 0.5, step, dtype=np.float64)
    y_values = np.arange(-half_y, half_y + step * 0.5, step, dtype=np.float64)
    corners = np.asarray(
        [
            [-half_x, -half_y],
            [-half_x, half_y],
            [half_x, -half_y],
            [half_x, half_y],
        ],
        dtype=np.float64,
    )
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="xy")
    grid_points = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])
    return np.unique(np.vstack([grid_points, corners]), axis=0)


def _build_base_footprint_map(map_yaml_path: Path) -> BaseFootprintMap:
    map_meta = load_map_meta(map_yaml_path)
    free_cells_map_xy = np.asarray(load_free_cells_unity_xz(map_meta), dtype=np.float64).reshape(-1, 2)
    if len(free_cells_map_xy) == 0:
        raise RuntimeError(f"Map has no free cells: {map_yaml_path}")

    resolution = float(map_meta.resolution_m)
    if resolution <= 0.0:
        raise ValueError(f"Map resolution must be positive, got {resolution}.")

    origin_x, origin_y = (float(v) for v in map_meta.origin_xy)
    free_cell_keys = frozenset(
        (
            int(round((float(cell[0]) - origin_x) / resolution)),
            int(round((float(cell[1]) - origin_y) / resolution)),
        )
        for cell in free_cells_map_xy
    )
    footprint_points = _rectangular_footprint_points_pb_xy(
        length_x_m=BASE_FOOTPRINT_LENGTH_X_M,
        length_y_m=BASE_FOOTPRINT_LENGTH_Y_M,
        resolution_m=resolution,
    )

    return BaseFootprintMap(
        free_cell_keys=free_cell_keys,
        free_cell_centers_xy=free_cells_map_xy,
        footprint_points_pb_xy=footprint_points,
        origin_xy=(origin_x, origin_y),
        resolution_m=resolution,
        length_x_m=float(BASE_FOOTPRINT_LENGTH_X_M),
        length_y_m=float(BASE_FOOTPRINT_LENGTH_Y_M),
    )


def _yaw_from_quaternion_xyzw(quaternion_xyzw: tuple[float, float, float, float]) -> float:
    x, y, z, w = (float(v) for v in quaternion_xyzw)
    # Standard ROS ENU planar yaw from an xyzw quaternion.
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return _wrap_angle_rad(math.atan2(siny_cosp, cosy_cosp))


def _stamp_to_seconds(stamp) -> float | None:
    if stamp is None:
        return None
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except Exception:
        return None


def _wait_for_amcl_pose(
    amcl_topic: str,
    *,
    timeout_sec: float,
) -> AmclPoseSnapshot | None:
    try:
        import rclpy
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from rclpy.node import Node
    except ImportError:
        return None

    class _AmclWaitNode(Node):
        def __init__(self) -> None:
            super().__init__(f"approach_agent_amcl_wait_{int(time.time())}")
            self.latest_msg: PoseWithCovarianceStamped | None = None
            self.create_subscription(PoseWithCovarianceStamped, amcl_topic, self._on_amcl, 10)

        def _on_amcl(self, msg: PoseWithCovarianceStamped) -> None:
            self.latest_msg = msg

    rclpy.init(args=None)
    node = _AmclWaitNode()
    try:
        deadline = time.monotonic() + float(timeout_sec)
        while node.latest_msg is None and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.latest_msg is None:
            return None
        position = node.latest_msg.pose.pose.position
        orientation = node.latest_msg.pose.pose.orientation
        return AmclPoseSnapshot(
            stamp_sec=_stamp_to_seconds(node.latest_msg.header.stamp),
            position_xyz=(float(position.x), float(position.y), float(position.z)),
            orientation_xyzw=(
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            ),
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _amcl_snapshot_to_ros_map_pose(amcl_pose: AmclPoseSnapshot | None) -> RosMapPose2D | None:
    if amcl_pose is None:
        return None
    return RosMapPose2D(
        x=float(amcl_pose.position_xyz[0]),
        y=float(amcl_pose.position_xyz[1]),
        yaw_rad=_yaw_from_quaternion_xyzw(amcl_pose.orientation_xyzw),
    )


def _make_transform(rotation_matrix: np.ndarray, translation_xyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation_xyz, dtype=np.float64).reshape(3)
    return transform


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -(transform[:3, :3].T @ transform[:3, 3])
    return inverse


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


def _unit_xy_or_default(vector_xyz: np.ndarray, default_xy: tuple[float, float] = (1.0, 0.0)) -> np.ndarray:
    vector_xyz = np.asarray(vector_xyz, dtype=np.float64).reshape(3)
    vector_xy = np.asarray([vector_xyz[0], vector_xyz[1]], dtype=np.float64)
    vector_norm = float(np.linalg.norm(vector_xy))
    if vector_norm <= 1e-4:
        return np.asarray(default_xy, dtype=np.float64)
    return vector_xy / vector_norm


def _get_reset_camera_transform_in_base_link_frame(
    planner_config_path: Path,
    planning_config,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return visual camera frame pose relative to reset base_link, from FK."""
    camera_position_pb, camera_quat_pb = get_reset_camera_transform(planner_config_path)
    camera_position_pb = np.asarray(camera_position_pb, dtype=np.float64).reshape(3)
    camera_rotation_link_pb = st.Rotation.from_quat(camera_quat_pb).as_matrix()
    camera_rotation_visual_pb = camera_rotation_link_pb @ PLANNING_CAMERA_TO_PB_LOCAL
    t_pb_camera_visual_reset = _make_transform(camera_rotation_visual_pb, camera_position_pb)

    base_rotation_reset = st.Rotation.from_euler(
        "xyz",
        [float(v) for v in planning_config.base_orientation_euler_deg],
        degrees=True,
    ).as_matrix()
    t_pb_base_reset = _make_transform(
        base_rotation_reset,
        np.asarray([0.0, 0.0, float(planning_config.initial_height)], dtype=np.float64),
    )
    t_base_link_camera_visual = _invert_transform(t_pb_base_reset) @ t_pb_camera_visual_reset
    return (
        t_base_link_camera_visual[:3, :3].astype(np.float64),
        t_base_link_camera_visual[:3, 3].astype(np.float64),
        float(planning_config.initial_height),
    )


def _base_link_pose_to_pb_world_pose(
    base_link_pose: RosMapPose2D | None,
    *,
    base_link_z_pb: float,
) -> tuple[np.ndarray, float] | None:
    if base_link_pose is None:
        return None
    pb_x, pb_y = _ros_map_xy_to_pb_world_xy((base_link_pose.x, base_link_pose.y))
    pb_yaw = _ros_map_yaw_to_pb_yaw(base_link_pose.yaw_rad)
    return (
        np.asarray([pb_x, pb_y, float(base_link_z_pb)], dtype=np.float64),
        float(pb_yaw),
    )


def _amcl_pose_to_pb_world_pose(
    amcl_pose: RosMapPose2D | None,
    *,
    z_pb: float = 0.0,
) -> tuple[np.ndarray, float] | None:
    return _base_link_pose_to_pb_world_pose(amcl_pose, base_link_z_pb=z_pb)


def _base_link_world_from_amcl_pb_pose(
    amcl_pb_xyz: np.ndarray,
    amcl_pb_yaw: float,
    *,
    z_pb: float,
) -> np.ndarray:
    offset_xy = _yaw_rotation_matrix(amcl_pb_yaw)[:2, :2] @ BASE_LINK_FROM_AMCL_PB_XY
    return np.asarray(
        [
            float(amcl_pb_xyz[0] + offset_xy[0]),
            float(amcl_pb_xyz[1] + offset_xy[1]),
            float(z_pb),
        ],
        dtype=np.float64,
    )


def _camera_local_transform_from_base_link(
    *,
    camera_in_base_link_rotation: np.ndarray,
    camera_in_base_link_position: np.ndarray,
    base_link_z_pb: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Place the reset arm-mounted depth camera in the local PB base_link scene."""
    t_pb_base_link = _make_transform(
        np.eye(3, dtype=np.float64),
        np.asarray([0.0, 0.0, float(base_link_z_pb)], dtype=np.float64),
    )
    t_base_link_camera_visual = _make_transform(
        camera_in_base_link_rotation,
        camera_in_base_link_position,
    )
    t_pb_camera_visual = t_pb_base_link @ t_base_link_camera_visual
    return (
        t_pb_camera_visual[:3, :3].astype(np.float64),
        t_pb_camera_visual[:3, 3].astype(np.float64),
    )


def _transform_camera_points_to_pybullet_camera_local_axes(points_camera_xyz: np.ndarray) -> np.ndarray:
    """Apply only the camera-axis convention: cam x->-PB y, cam y->-PB z, cam z->PB x."""
    points_camera_xyz = np.asarray(points_camera_xyz, dtype=np.float64).reshape(-1, 3)
    if len(points_camera_xyz) == 0:
        return points_camera_xyz
    return points_camera_xyz @ PLANNING_CAMERA_TO_PB_LOCAL.T


def _transform_camera_points_to_local_pb(
    points_camera_xyz: np.ndarray,
    *,
    camera_to_pb_rotation: np.ndarray,
    camera_position_pb: np.ndarray,
) -> np.ndarray:
    points_camera_xyz = np.asarray(points_camera_xyz, dtype=np.float64).reshape(-1, 3)
    if len(points_camera_xyz) == 0:
        return points_camera_xyz
    return (points_camera_xyz @ camera_to_pb_rotation.T) + camera_position_pb.reshape(1, 3)


def _transform_camera_rotation_to_pybullet_ee(
    rotation_camera: np.ndarray,
    *,
    camera_to_pb_rotation: np.ndarray,
) -> np.ndarray:
    """Convert a camera-frame grasp rotation into the local PB dynamic gripper/EE frame."""
    rotation_camera = np.asarray(rotation_camera, dtype=np.float64)
    if rotation_camera.shape != (3, 3):
        raise ValueError(f"rotation_camera must have shape (3, 3), got {rotation_camera.shape}.")
    return (camera_to_pb_rotation @ rotation_camera) @ PLANNING_GRASP_TO_EE


def _print_ros_map_pose(label: str, pose: RosMapPose2D | dict[str, object] | None) -> None:
    if pose is None:
        print(f"[test_base_sampler] {label}: unavailable", flush=True)
        return
    if isinstance(pose, RosMapPose2D):
        x = float(pose.x)
        y = float(pose.y)
        yaw_rad = float(pose.yaw_rad)
    else:
        x = float(pose["x"])
        y = float(pose["y"])
        yaw_rad = float(pose["yaw_rad"])
    print(
        f"[test_base_sampler] {label}: "
        f"ros_map_x={x:.4f}m ros_map_y={y:.4f}m "
        f"yaw={yaw_rad:.6f}rad ({math.degrees(yaw_rad):.2f}deg)",
        flush=True,
    )


def _format_ros_map_pose_lines(pose: RosMapPose2D | dict[str, object] | None) -> str:
    if pose is None:
        return " ros_map    : unavailable\n"
    if isinstance(pose, RosMapPose2D):
        x = float(pose.x)
        y = float(pose.y)
        yaw_rad = float(pose.yaw_rad)
        yaw_deg = math.degrees(yaw_rad)
    else:
        x = float(pose["x"])
        y = float(pose["y"])
        yaw_rad = float(pose["yaw_rad"])
        yaw_deg = float(pose.get("yaw_deg", math.degrees(yaw_rad)))
    return (
        f" ros_map_x  : {x:.4f} m\n"
        f" ros_map_y  : {y:.4f} m\n"
        f" ros_yaw    : {yaw_rad:.6f} rad  ({yaw_deg:.2f} deg)\n"
    )


def _print_selected_base_link_ros_map_banner(
    *,
    rank: int,
    base_link_pose: RosMapPose2D | dict[str, object] | None,
    amcl_pose: RosMapPose2D | dict[str, object] | None,
) -> None:
    print(
        "\n"
        "============================================================\n"
        " SELECTED SAMPLED ROS MAP POSES\n"
        "------------------------------------------------------------\n"
        f" grasp_rank : {int(rank):02d}\n"
        " AMCL / VEHICLE CENTER ROS MAP\n"
        f"{_format_ros_map_pose_lines(amcl_pose)}"
        "------------------------------------------------------------\n"
        " BASE_LINK ROS MAP\n"
        f"{_format_ros_map_pose_lines(base_link_pose)}"
        "============================================================\n",
        flush=True,
    )


def _print_closest_ik_distance_banner(
    *,
    rank: int,
    target_pb: np.ndarray,
    solution: dict[str, object] | None,
) -> None:
    if solution is None:
        print(
            "\n"
            "############################################################\n"
            " CLOSEST IK DISTANCE DETAILS\n"
            "------------------------------------------------------------\n"
            f" grasp_rank : {int(rank):02d}\n"
            " status     : unavailable\n"
            "############################################################\n",
            flush=True,
        )
        return

    target_xyz = np.asarray(target_pb, dtype=np.float64).reshape(3)
    ee_xyz = np.asarray(solution["final_ee_position_xyz"], dtype=np.float64).reshape(3)
    ik_error_xyz = np.asarray(solution["ik_error_xyz_m"], dtype=np.float64).reshape(3)
    orientation_error = solution.get("ee_orientation_error_deg")
    approach_axis_offset = solution.get("approach_axis_offset_m")
    lateral_offset = solution.get("lateral_offset_m")

    print(
        "\n"
        "############################################################\n"
        " CLOSEST IK DISTANCE DETAILS\n"
        "------------------------------------------------------------\n"
        f" grasp_rank : {int(rank):02d}\n"
        f" ik_feasible: {int(bool(solution.get('ik_feasible', False)))}\n"
        "------------------------------------------------------------\n"
        " EE vs TARGET LOCAL PB AFTER IK MOVE\n"
        f" target_xyz : [{target_xyz[0]:.4f}, {target_xyz[1]:.4f}, {target_xyz[2]:.4f}] m\n"
        f" ee_xyz     : [{ee_xyz[0]:.4f}, {ee_xyz[1]:.4f}, {ee_xyz[2]:.4f}] m\n"
        f" error_xyz  : [{ik_error_xyz[0]:+.4f}, {ik_error_xyz[1]:+.4f}, {ik_error_xyz[2]:+.4f}] m\n"
        f" ee_target_dist : {float(solution['ee_position_error_m']):.4f} m\n"
        f" ee_target_ori  : {'nan' if orientation_error is None else f'{float(orientation_error):.2f}'} deg\n"
        f" approach_x : {'nan' if approach_axis_offset is None else f'{float(approach_axis_offset):.4f}'} m\n"
        f" lateral    : {'nan' if lateral_offset is None else f'{float(lateral_offset):.4f}'} m\n"
        "############################################################\n",
        flush=True,
    )


def _print_pb_pose(label: str, xyz: object, yaw_rad: float | None = None) -> None:
    xyz_arr = np.asarray(xyz, dtype=float).reshape(-1)
    if len(xyz_arr) < 3:
        print(f"[test_base_sampler] {label}: unavailable", flush=True)
        return
    yaw_text = ""
    if yaw_rad is not None:
        yaw = float(yaw_rad)
        yaw_text = f" yaw={yaw:.6f}rad ({math.degrees(yaw):.2f}deg)"
    print(
        f"[test_base_sampler] {label}: "
        f"pb_x={float(xyz_arr[0]):.4f}m "
        f"pb_y={float(xyz_arr[1]):.4f}m "
        f"pb_z={float(xyz_arr[2]):.4f}m"
        f"{yaw_text}",
        flush=True,
    )


def _print_camera_to_pb_axis_mapping() -> None:
    basis_camera = np.eye(3, dtype=np.float64)
    basis_pb_local = _transform_camera_points_to_pybullet_camera_local_axes(basis_camera)
    print(
        "[test_base_sampler] camera point -> PB local axis mapping: "
        f"cam +X -> {basis_pb_local[0].astype(float).tolist()}, "
        f"cam +Y -> {basis_pb_local[1].astype(float).tolist()}, "
        f"cam +Z -> {basis_pb_local[2].astype(float).tolist()} "
        "(rule: pb=[cam_z, -cam_x, -cam_y])",
        flush=True,
    )


def _load_grasp_candidates_from_result_json(
    grasp_result_json_path: Path | None = None,
) -> tuple[list[GraspPoseCandidate], np.ndarray, Path]:
    json_path = _resolve_grasp_result_json_path(grasp_result_json_path)
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


def _voxel_downsample_points(points_xyz: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if len(points_xyz) == 0 or voxel_size_m <= 0.0:
        return np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    buckets = np.floor(np.asarray(points_xyz, dtype=np.float32) / float(voxel_size_m)).astype(np.int32)
    _, keep_indices = np.unique(buckets, axis=0, return_index=True)
    return np.asarray(points_xyz[np.sort(keep_indices)], dtype=np.float32)


def _backproject_live_depth_to_camera_points(
    depth_m: np.ndarray,
    intrinsic_matrix: np.ndarray,
    *,
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """Mirror grasp_agent.pipeline._depth_to_point_cloud for scene_pc_camera."""
    depth_m = np.asarray(depth_m, dtype=np.float32)
    intrinsic_matrix = np.asarray(intrinsic_matrix, dtype=np.float64).reshape(3, 3)
    fx = float(intrinsic_matrix[0, 0])
    fy = float(intrinsic_matrix[1, 1])
    cx = float(intrinsic_matrix[0, 2])
    cy = float(intrinsic_matrix[1, 2])

    valid = (
        np.isfinite(depth_m)
        & (depth_m > float(min_depth_m))
        & (depth_m < float(max_depth_m))
    )
    v_idx, u_idx = np.nonzero(valid)
    if len(u_idx) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    if stride > 1:
        keep = np.arange(0, len(u_idx), int(stride))
        u_idx = u_idx[keep]
        v_idx = v_idx[keep]

    z = depth_m[v_idx, u_idx].astype(np.float32)
    x = (u_idx.astype(np.float32) - cx) * z / fx
    y = (v_idx.astype(np.float32) - cy) * z / fy
    return np.column_stack([x, y, z]).astype(np.float32)


def _capture_live_scene_voxels(
    camera_config_path: Path,
    camera_in_base_link_rotation: np.ndarray,
    camera_in_base_link_position: np.ndarray,
    base_link_z_pb: float,
) -> LiveSceneCapture:
    camera_cfg = load_camera_car_voxel_ompl_config(camera_config_path)
    snapshot = capture_rgbd_snapshot(
        camera_cfg.camera_name,
        timeout_sec=camera_cfg.capture_timeout_sec,
        amcl_topic=camera_cfg.amcl_topic,
    )
    amcl_pose = snapshot.amcl_pose
    if amcl_pose is None:
        amcl_pose = _wait_for_amcl_pose(
            camera_cfg.amcl_topic,
            timeout_sec=camera_cfg.capture_timeout_sec,
        )
    camera_to_pb_rotation, camera_position_pb = _camera_local_transform_from_base_link(
        camera_in_base_link_rotation=camera_in_base_link_rotation,
        camera_in_base_link_position=camera_in_base_link_position,
        base_link_z_pb=base_link_z_pb,
    )
    intrinsics = load_camera_intrinsics(Path(camera_cfg.intrinsics_path))
    depth_metric_m = decode_depth_png_bytes(snapshot.depth_bytes)
    points_camera = _backproject_live_depth_to_camera_points(
        depth_metric_m,
        intrinsics.k,
        min_depth_m=LIVE_VOXEL_MIN_DEPTH_M,
        max_depth_m=LIVE_VOXEL_MAX_DEPTH_M,
        stride=LIVE_VOXEL_SCENE_POINT_STRIDE,
    )
    points_camera = _voxel_downsample_points(points_camera, voxel_size_m=LIVE_VOXEL_SCENE_DOWNSAMPLE_M)
    if len(points_camera) == 0:
        raise RuntimeError("Live Camera_Car RGBD produced zero valid camera-frame points.")

    points_pybullet = _transform_camera_points_to_local_pb(
        points_camera,
        camera_to_pb_rotation=camera_to_pb_rotation,
        camera_position_pb=camera_position_pb,
    )

    voxel_centers_pb = voxelize_points(
        points_pybullet,
        voxel_size_m=camera_cfg.voxel_size_m,
        max_voxels=camera_cfg.max_voxel_obstacles,
    )
    if len(voxel_centers_pb) == 0:
        raise RuntimeError("Live Camera_Car RGBD produced zero occupied voxels.")
    return LiveSceneCapture(
        voxel_centers_pb=np.asarray(voxel_centers_pb, dtype=np.float64),
        camera_to_pb_rotation=np.asarray(camera_to_pb_rotation, dtype=np.float64),
        camera_position_pb=np.asarray(camera_position_pb, dtype=np.float64),
        amcl_pose=amcl_pose,
        valid_depth_point_count=int(len(points_camera)),
    )


def _transform_grasp_pose_camera_to_pybullet(
    grasp_candidate: GraspPoseCandidate,
    *,
    camera_to_pb_rotation: np.ndarray,
    camera_position_pb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_pb = _transform_camera_points_to_local_pb(
        grasp_candidate.position_camera_xyz.reshape(1, 3),
        camera_to_pb_rotation=camera_to_pb_rotation,
        camera_position_pb=camera_position_pb,
    ).reshape(3)
    target_rot_pb = _transform_camera_rotation_to_pybullet_ee(
        grasp_candidate.rotation_camera,
        camera_to_pb_rotation=camera_to_pb_rotation,
    )
    target_quat_pb = st.Rotation.from_matrix(target_rot_pb).as_quat()
    return (
        np.asarray(target_pb, dtype=np.float64),
        np.asarray(target_rot_pb, dtype=np.float64),
        np.asarray(target_quat_pb, dtype=np.float64),
    )


def _pb_world_xy_to_ros_map_xy(pb_xy: tuple[float, float]) -> tuple[float, float]:
    pb_x, pb_y = float(pb_xy[0]), float(pb_xy[1])
    return (pb_y, -pb_x)


def _ros_map_xy_to_pb_world_xy(ros_map_xy: tuple[float, float]) -> tuple[float, float]:
    ros_x, ros_y = float(ros_map_xy[0]), float(ros_map_xy[1])
    return (-ros_y, ros_x)


def _pb_yaw_to_ros_map_yaw(pb_yaw_rad: float) -> float:
    return _wrap_angle_rad(float(pb_yaw_rad) - math.pi * 0.5)


def _ros_map_yaw_to_pb_yaw(ros_yaw_rad: float) -> float:
    return _wrap_angle_rad(float(ros_yaw_rad) + math.pi * 0.5)


def _pb_world_pose_to_ros_map_pose(
    pb_xy: tuple[float, float],
    pb_yaw_rad: float,
) -> RosMapPose2D:
    ros_x, ros_y = _pb_world_xy_to_ros_map_xy(pb_xy)
    return RosMapPose2D(
        x=ros_x,
        y=ros_y,
        yaw_rad=_pb_yaw_to_ros_map_yaw(pb_yaw_rad),
    )


def _local_pb_base_pose_to_ros_map_pose(
    local_pb_xy: tuple[float, float],
    local_pb_yaw_rad: float,
    current_amcl_pose: RosMapPose2D | None,
) -> RosMapPose2D:
    if current_amcl_pose is None:
        return _pb_world_pose_to_ros_map_pose(local_pb_xy, local_pb_yaw_rad)

    current_amcl_pb_pose = _amcl_pose_to_pb_world_pose(current_amcl_pose, z_pb=0.0)
    assert current_amcl_pb_pose is not None
    current_amcl_pb_xyz, current_pb_yaw = current_amcl_pb_pose
    current_base_link_pb_xyz = _base_link_world_from_amcl_pb_pose(
        current_amcl_pb_xyz,
        current_pb_yaw,
        z_pb=0.0,
    )
    current_rot = _yaw_rotation_matrix(current_pb_yaw)
    local_delta = np.asarray([float(local_pb_xy[0]), float(local_pb_xy[1]), 0.0], dtype=np.float64)
    candidate_pb_xy = current_base_link_pb_xyz[:2] + (current_rot @ local_delta)[:2]
    candidate_pb_yaw = _wrap_angle_rad(float(current_pb_yaw) + float(local_pb_yaw_rad))
    return _pb_world_pose_to_ros_map_pose(
        (float(candidate_pb_xy[0]), float(candidate_pb_xy[1])),
        candidate_pb_yaw,
    )


def _local_pb_xy_to_ros_map_xy(
    local_pb_xy: tuple[float, float],
    current_amcl_pose: RosMapPose2D | None,
) -> tuple[float, float]:
    if current_amcl_pose is None:
        return _pb_world_xy_to_ros_map_xy(local_pb_xy)

    current_amcl_pb_pose = _amcl_pose_to_pb_world_pose(current_amcl_pose, z_pb=0.0)
    assert current_amcl_pb_pose is not None
    current_amcl_pb_xyz, current_pb_yaw = current_amcl_pb_pose
    current_base_link_pb_xyz = _base_link_world_from_amcl_pb_pose(
        current_amcl_pb_xyz,
        current_pb_yaw,
        z_pb=0.0,
    )
    current_rot_xy = _yaw_rotation_matrix(current_pb_yaw)[:2, :2]
    local_delta = np.asarray(local_pb_xy, dtype=np.float64).reshape(2)
    candidate_pb_xy = current_base_link_pb_xyz[:2] + current_rot_xy @ local_delta
    return _pb_world_xy_to_ros_map_xy((float(candidate_pb_xy[0]), float(candidate_pb_xy[1])))


def _ros_map_amcl_pose_to_local_pb_base_pose(
    amcl_pose: RosMapPose2D,
    current_amcl_pose: RosMapPose2D | None,
) -> tuple[tuple[float, float], float]:
    candidate_amcl_pb_pose = _amcl_pose_to_pb_world_pose(amcl_pose, z_pb=0.0)
    assert candidate_amcl_pb_pose is not None
    candidate_amcl_pb_xyz, candidate_pb_yaw = candidate_amcl_pb_pose
    candidate_base_link_pb_xyz = _base_link_world_from_amcl_pb_pose(
        candidate_amcl_pb_xyz,
        candidate_pb_yaw,
        z_pb=0.0,
    )

    if current_amcl_pose is None:
        return (
            (float(candidate_base_link_pb_xyz[0]), float(candidate_base_link_pb_xyz[1])),
            float(candidate_pb_yaw),
        )

    current_amcl_pb_pose = _amcl_pose_to_pb_world_pose(current_amcl_pose, z_pb=0.0)
    assert current_amcl_pb_pose is not None
    current_amcl_pb_xyz, current_pb_yaw = current_amcl_pb_pose
    current_base_link_pb_xyz = _base_link_world_from_amcl_pb_pose(
        current_amcl_pb_xyz,
        current_pb_yaw,
        z_pb=0.0,
    )
    current_rot_xy = _yaw_rotation_matrix(current_pb_yaw)[:2, :2]
    local_pb_xy = (candidate_base_link_pb_xyz[:2] - current_base_link_pb_xyz[:2]) @ current_rot_xy
    local_pb_yaw = _wrap_angle_rad(candidate_pb_yaw - current_pb_yaw)
    return ((float(local_pb_xy[0]), float(local_pb_xy[1])), float(local_pb_yaw))


def _local_pb_direction_to_ros_map_xy(
    direction_local_pb_xyz: np.ndarray,
    current_amcl_pose: RosMapPose2D | None,
) -> np.ndarray:
    direction_local_xy = _unit_xy_or_default(direction_local_pb_xyz)
    if current_amcl_pose is None:
        direction_pb_world_xy = direction_local_xy
    else:
        current_pb_pose = _amcl_pose_to_pb_world_pose(current_amcl_pose, z_pb=0.0)
        assert current_pb_pose is not None
        _, current_pb_yaw = current_pb_pose
        direction_pb_world_xy = _yaw_rotation_matrix(current_pb_yaw)[:2, :2] @ direction_local_xy

    direction_ros_xy = np.asarray(
        [direction_pb_world_xy[1], -direction_pb_world_xy[0]],
        dtype=np.float64,
    )
    direction_norm = float(np.linalg.norm(direction_ros_xy))
    if direction_norm <= 1e-6:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    return direction_ros_xy / direction_norm


def _rectangular_footprint_is_clear_on_map(
    footprint_map: BaseFootprintMap,
    amcl_pose: RosMapPose2D,
) -> bool:
    amcl_pb_pose = _amcl_pose_to_pb_world_pose(amcl_pose, z_pb=0.0)
    assert amcl_pb_pose is not None
    amcl_pb_xyz, amcl_pb_yaw = amcl_pb_pose
    footprint_points_pb = np.asarray(footprint_map.footprint_points_pb_xy, dtype=np.float64).reshape(-1, 2)
    rot_xy = _yaw_rotation_matrix(amcl_pb_yaw)[:2, :2]
    world_pb_xy = amcl_pb_xyz[:2].reshape(1, 2) + footprint_points_pb @ rot_xy.T

    ros_x = world_pb_xy[:, 1]
    ros_y = -world_pb_xy[:, 0]
    origin_x, origin_y = footprint_map.origin_xy
    resolution = float(footprint_map.resolution_m)
    keys = zip(
        np.rint((ros_x - origin_x) / resolution).astype(np.int32),
        np.rint((ros_y - origin_y) / resolution).astype(np.int32),
    )
    return all((int(key_x), int(key_y)) in footprint_map.free_cell_keys for key_x, key_y in keys)


def _ros_map_pose_to_dict(pose: RosMapPose2D) -> dict[str, float]:
    return {
        "x": float(pose.x),
        "y": float(pose.y),
        "yaw_rad": float(pose.yaw_rad),
        "yaw_deg": float(math.degrees(pose.yaw_rad)),
    }


def _robot_collides_with_obstacles(
    p_mod,
    robot_id: int,
    obstacle_body_ids: list[int],
) -> bool:
    for obstacle_body_id in obstacle_body_ids:
        if p_mod.getClosestPoints(robot_id, obstacle_body_id, distance=0.0):
            return True
    return False


def _joint_limit_metrics(
    joint_solution_rad: np.ndarray,
    planning_config,
) -> tuple[float, float, float]:
    lower_bounds_rad = np.radians([float(lower) for lower, _ in planning_config.joint_bounds_deg]).astype(np.float64)
    upper_bounds_rad = np.radians([float(upper) for _, upper in planning_config.joint_bounds_deg]).astype(np.float64)
    joint_span_rad = np.maximum(upper_bounds_rad - lower_bounds_rad, 1e-6)
    margin_to_lower = joint_solution_rad - lower_bounds_rad
    margin_to_upper = upper_bounds_rad - joint_solution_rad
    margin_ratio = np.minimum(margin_to_lower, margin_to_upper) / joint_span_rad
    reset_joint_rad = np.radians(np.asarray(planning_config.joint_reset_deg, dtype=np.float64))
    reset_delta_norm = (joint_solution_rad - reset_joint_rad) / joint_span_rad
    return (
        float(np.min(margin_ratio)),
        float(np.mean(margin_ratio)),
        float(np.linalg.norm(reset_delta_norm)),
    )


def _feasible_ik_solution_sort_key(solution: dict[str, object]) -> tuple[float, float, float, float, float, int]:
    return (
        -float(solution.get("joint_limit_margin_min_ratio", -1.0)),
        -float(solution.get("joint_limit_margin_mean_ratio", -1.0)),
        float(solution.get("joint_reset_delta_norm_l2", float("inf"))),
        float(solution.get("ee_orientation_error_deg", float("inf"))),
        float(solution.get("ee_position_error_m", float("inf"))),
        int(solution.get("sample_index", 0)),
    )


def _closest_ik_solution_sort_key(solution: dict[str, object]) -> tuple[float, float, float, int]:
    orientation_error = solution.get("ee_orientation_error_deg")
    return (
        float(solution.get("ee_position_error_m", float("inf"))),
        float("inf") if orientation_error is None else float(orientation_error),
        float(solution.get("joint_reset_delta_norm_l2", float("inf"))),
        int(solution.get("sample_index", 0)),
    )


def _attempt_ik_at_base_pose(
    *,
    p_mod,
    robot_id: int,
    controllable_joint_ids: list[int],
    planning_config,
    target_pb: np.ndarray,
    target_quat_pb: np.ndarray,
    base_link_xy: tuple[float, float],
    base_link_yaw_rad: float,
    obstacle_body_ids: list[int] | None = None,
    solve_attempts: int = 3,
) -> dict[str, object]:
    base_xyz = [
        float(base_link_xy[0]),
        float(base_link_xy[1]),
        float(planning_config.initial_height),
    ]
    base_quat = p_mod.getQuaternionFromEuler([0.0, 0.0, float(base_link_yaw_rad)])
    joint_reset_rad = _degrees_to_radians(planning_config.joint_reset_deg)

    p_mod.resetBasePositionAndOrientation(robot_id, base_xyz, base_quat)
    _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, joint_reset_rad)

    ik_joint_poses = None
    for _ in range(max(int(solve_attempts), 1)):
        ik_joint_poses = p_mod.calculateInverseKinematics(
            robot_id,
            planning_config.ee_link_index,
            target_pb.astype(float).tolist(),
            target_quat_pb.astype(float).tolist(),
            maxNumIterations=5000,
            residualThreshold=1e-4,
        )
        _set_joint_positions_direct(
            p_mod,
            robot_id,
            controllable_joint_ids,
            ik_joint_poses[: len(controllable_joint_ids)],
        )

    p_mod.performCollisionDetection()
    final_state = p_mod.getLinkState(
        robot_id,
        planning_config.ee_link_index,
        computeForwardKinematics=True,
    )
    final_pos = np.asarray(final_state[4], dtype=np.float64)
    final_quat_xyzw = np.asarray(final_state[5], dtype=np.float64)
    error_xyz = np.asarray(target_pb - final_pos, dtype=np.float64)
    dist_error = float(np.linalg.norm(error_xyz))
    orientation_error_deg, approach_axis_offset_m, lateral_offset_m = _compute_pose_alignment_metrics(
        ee_position_xyz=final_pos.astype(float).tolist(),
        ee_orientation_xyzw=final_quat_xyzw.astype(float).tolist(),
        target_position_xyz=target_pb.astype(float).tolist(),
        target_orientation_xyzw=target_quat_pb.astype(float).tolist(),
    )
    collision_free = True
    if obstacle_body_ids is not None:
        collision_free = not _robot_collides_with_obstacles(p_mod, robot_id, obstacle_body_ids)

    joint_solution_rad = None
    joint_limit_margin_min_ratio = None
    joint_limit_margin_mean_ratio = None
    joint_reset_delta_norm_l2 = None
    if ik_joint_poses is not None:
        joint_solution_rad = np.asarray(ik_joint_poses[: len(controllable_joint_ids)], dtype=np.float64)
        (
            joint_limit_margin_min_ratio,
            joint_limit_margin_mean_ratio,
            joint_reset_delta_norm_l2,
        ) = _joint_limit_metrics(joint_solution_rad, planning_config)

    return {
        "pb_base_link_xyz": base_xyz,
        "pb_base_link_yaw_rad": float(base_link_yaw_rad),
        "final_ee_position_xyz": final_pos,
        "final_ee_orientation_xyzw": final_quat_xyzw,
        "ik_error_xyz": error_xyz,
        "ee_position_error_m": dist_error,
        "ee_orientation_error_deg": orientation_error_deg,
        "approach_axis_offset_m": approach_axis_offset_m,
        "lateral_offset_m": lateral_offset_m,
        "collision_free": bool(collision_free),
        "ik_joint_solution_rad": joint_solution_rad,
        "joint_limit_margin_min_ratio": joint_limit_margin_min_ratio,
        "joint_limit_margin_mean_ratio": joint_limit_margin_mean_ratio,
        "joint_reset_delta_norm_l2": joint_reset_delta_norm_l2,
    }


def _prepare_ranked_grasp_targets(
    *,
    p_mod,
    robot_id: int,
    controllable_joint_ids: list[int],
    planning_config,
    grasp_candidates: list[GraspPoseCandidate],
    camera_to_pb_rotation: np.ndarray,
    camera_position_pb: np.ndarray,
) -> list[PreparedGraspTarget]:
    prepared_targets: list[PreparedGraspTarget] = []
    for grasp_candidate in grasp_candidates:
        target_pb, target_rot_pb, target_quat_pb = _transform_grasp_pose_camera_to_pybullet(
            grasp_candidate,
            camera_to_pb_rotation=camera_to_pb_rotation,
            camera_position_pb=camera_position_pb,
        )
        direct_ik_attempt = _attempt_ik_at_base_pose(
            p_mod=p_mod,
            robot_id=robot_id,
            controllable_joint_ids=controllable_joint_ids,
            planning_config=planning_config,
            target_pb=target_pb,
            target_quat_pb=target_quat_pb,
            base_link_xy=(0.0, 0.0),
            base_link_yaw_rad=0.0,
            obstacle_body_ids=None,
        )
        prepared_targets.append(
            PreparedGraspTarget(
                grasp_candidate=grasp_candidate,
                target_pb=target_pb,
                target_rot_pb=target_rot_pb,
                target_quat_pb=target_quat_pb,
                direct_ik_error_m=float(direct_ik_attempt["ee_position_error_m"]),
                direct_ik_error_xyz=np.asarray(direct_ik_attempt["ik_error_xyz"], dtype=np.float64),
                direct_orientation_error_deg=(
                    None
                    if direct_ik_attempt["ee_orientation_error_deg"] is None
                    else float(direct_ik_attempt["ee_orientation_error_deg"])
                ),
            )
        )

    prepared_targets.sort(
        key=lambda item: (
            float(item.direct_ik_error_m),
            float(item.direct_orientation_error_deg if item.direct_orientation_error_deg is not None else float("inf")),
            int(item.grasp_candidate.rank),
        )
    )
    return prepared_targets


def _sample_base_link_poses_from_aligned_ros_map(
    *,
    footprint_map: BaseFootprintMap,
    current_amcl_pose: RosMapPose2D | None,
    target_pb: np.ndarray,
    target_rot_pb: np.ndarray,
    rng: np.random.Generator,
    max_samples: int,
    max_yaw_delta_rad: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    stats: dict[str, object] = {
        "target_ros_map_xy": None,
        "safe_cell_count": int(len(footprint_map.free_cell_centers_xy)),
        "region_cell_count": 0,
        "sampled_count": 0,
        "footprint_rejected_count": 0,
    }
    if current_amcl_pose is None:
        return [], stats

    free_cells_xy = np.asarray(footprint_map.free_cell_centers_xy, dtype=np.float64).reshape(-1, 2)
    if len(free_cells_xy) == 0:
        return [], stats

    max_samples = max(int(max_samples), 0)
    if max_samples <= 0:
        return [], stats

    target_ros_xy = np.asarray(
        _local_pb_xy_to_ros_map_xy(
            (float(target_pb[0]), float(target_pb[1])),
            current_amcl_pose,
        ),
        dtype=np.float64,
    )
    stats["target_ros_map_xy"] = (float(target_ros_xy[0]), float(target_ros_xy[1]))

    yaw_delta_by_cell = rng.uniform(
        -float(max_yaw_delta_rad),
        float(max_yaw_delta_rad),
        size=len(free_cells_xy),
    ).astype(np.float64)
    ros_map_yaw_by_cell = np.asarray(
        [
            _wrap_angle_rad(float(current_amcl_pose.yaw_rad) + float(yaw_delta))
            for yaw_delta in yaw_delta_by_cell
        ],
        dtype=np.float64,
    )
    pb_yaw_by_cell = ros_map_yaw_by_cell + math.pi * 0.5
    offset_pb_x = float(BASE_LINK_FROM_AMCL_PB_XY[0])
    offset_pb_y = float(BASE_LINK_FROM_AMCL_PB_XY[1])
    offset_world_pb_x = np.cos(pb_yaw_by_cell) * offset_pb_x - np.sin(pb_yaw_by_cell) * offset_pb_y
    offset_world_pb_y = np.sin(pb_yaw_by_cell) * offset_pb_x + np.cos(pb_yaw_by_cell) * offset_pb_y
    candidate_base_link_xy = free_cells_xy + np.column_stack([offset_world_pb_y, -offset_world_pb_x])

    base_to_target_xy = target_ros_xy.reshape(1, 2) - candidate_base_link_xy
    distances = np.linalg.norm(base_to_target_xy, axis=1)
    distance_mask = (
        (distances >= BASE_SAMPLE_MIN_DISTANCE_M)
        & (distances <= BASE_SAMPLE_MAX_DISTANCE_M)
    )

    direction_mask = np.zeros(len(free_cells_xy), dtype=bool)
    valid_distance = distances > 1e-6
    direction_rows = distance_mask & valid_distance
    if np.any(direction_rows):
        base_to_target_dir = base_to_target_xy[direction_rows] / distances[direction_rows].reshape(-1, 1)
        target_approach_dir = _local_pb_direction_to_ros_map_xy(
            target_rot_pb[:, 0],
            current_amcl_pose,
        )
        dot = np.clip(base_to_target_dir @ target_approach_dir.reshape(2), -1.0, 1.0)
        approach_error_deg = np.full(len(free_cells_xy), np.inf, dtype=np.float64)
        approach_error_deg[direction_rows] = np.degrees(np.arccos(dot))
        direction_mask = approach_error_deg <= BASE_SAMPLE_APPROACH_HALF_ANGLE_DEG
    else:
        approach_error_deg = np.full(len(free_cells_xy), np.inf, dtype=np.float64)

    region_mask = distance_mask & direction_mask
    region_indices = np.nonzero(region_mask)[0]
    stats["region_cell_count"] = int(len(region_indices))
    if len(region_indices) == 0:
        return [], stats

    candidates: list[dict[str, object]] = []
    shuffled_region_indices = np.asarray(region_indices, dtype=np.int64)
    rng.shuffle(shuffled_region_indices)
    footprint_rejected_count = 0
    for cell_index in shuffled_region_indices:
        if len(candidates) >= max_samples:
            break
        ros_map_yaw = float(ros_map_yaw_by_cell[cell_index])
        ros_map_amcl_pose = RosMapPose2D(
            x=float(free_cells_xy[cell_index, 0]),
            y=float(free_cells_xy[cell_index, 1]),
            yaw_rad=ros_map_yaw,
        )
        if not _rectangular_footprint_is_clear_on_map(footprint_map, ros_map_amcl_pose):
            footprint_rejected_count += 1
            continue

        local_pb_xy, local_pb_yaw = _ros_map_amcl_pose_to_local_pb_base_pose(
            ros_map_amcl_pose,
            current_amcl_pose,
        )
        ros_map_base_link_pose = _local_pb_base_pose_to_ros_map_pose(
            local_pb_xy,
            local_pb_yaw,
            current_amcl_pose,
        )
        candidates.append(
            {
                "sample_index": int(len(candidates)),
                "map_cell_index": int(cell_index),
                "local_pb_xy": (float(local_pb_xy[0]), float(local_pb_xy[1])),
                "local_pb_yaw_rad": float(local_pb_yaw),
                "ros_map_pose": ros_map_base_link_pose,
                "ros_map_amcl_pose": ros_map_amcl_pose,
                "distance_to_target_m": float(distances[cell_index]),
                "approach_error_deg": float(approach_error_deg[cell_index]),
            }
        )

    stats["sampled_count"] = int(len(candidates))
    stats["footprint_rejected_count"] = int(footprint_rejected_count)
    return candidates, stats


def _make_sampled_ik_solution_record(
    *,
    ik_attempt: dict[str, object],
    sample_candidate: dict[str, object],
    sample_stats: dict[str, object],
    reference_base_yaw_pb: float,
    feasible: bool,
) -> dict[str, object] | None:
    joint_solution_rad = ik_attempt["ik_joint_solution_rad"]
    if joint_solution_rad is None:
        return None

    joint_solution_rad = np.asarray(joint_solution_rad, dtype=np.float64)
    joint_solution_deg = np.degrees(joint_solution_rad)
    base_xyz = [float(v) for v in ik_attempt["pb_base_link_xyz"]]
    base_yaw = float(ik_attempt["pb_base_link_yaw_rad"])
    base_yaw_delta_from_reference = _abs_angle_delta_rad(base_yaw, reference_base_yaw_pb)
    error_xyz = np.asarray(ik_attempt["ik_error_xyz"], dtype=np.float64)
    orientation_error_deg = ik_attempt["ee_orientation_error_deg"]
    ros_map_pose = sample_candidate["ros_map_pose"]
    ros_map_amcl_pose = sample_candidate.get("ros_map_amcl_pose")

    return {
        "sample_index": int(sample_candidate["sample_index"]),
        "sample_source": str(sample_candidate.get("sample_source", "ros_map")),
        "ros_map_sample_region_cell_count": int(sample_stats["region_cell_count"]),
        "ros_map_sample_cell_index": int(sample_candidate["map_cell_index"]),
        "ros_map_sample_distance_to_target_m": float(sample_candidate["distance_to_target_m"]),
        "ros_map_sample_approach_error_deg": float(sample_candidate["approach_error_deg"]),
        "pb_base_link_xyz": base_xyz,
        "pb_base_link_yaw_rad": base_yaw,
        "pb_base_link_yaw_deg": float(math.degrees(base_yaw)),
        "pb_base_link_reference_yaw_rad": float(reference_base_yaw_pb),
        "pb_base_link_reference_yaw_deg": float(math.degrees(reference_base_yaw_pb)),
        "pb_base_link_yaw_delta_from_reference_rad": float(base_yaw_delta_from_reference),
        "pb_base_link_yaw_delta_from_reference_deg": float(math.degrees(base_yaw_delta_from_reference)),
        "pb_base_link_yaw_max_delta_from_reference_deg": float(BASE_LINK_YAW_MAX_DELTA_FROM_CURRENT_DEG),
        "ros_map_base_link_pose": _ros_map_pose_to_dict(ros_map_pose),
        "ros_map_amcl_pose": (
            None
            if not isinstance(ros_map_amcl_pose, RosMapPose2D)
            else _ros_map_pose_to_dict(ros_map_amcl_pose)
        ),
        "final_ee_position_xyz": np.asarray(ik_attempt["final_ee_position_xyz"], dtype=np.float64).astype(float).tolist(),
        "final_ee_orientation_xyzw": np.asarray(
            ik_attempt["final_ee_orientation_xyzw"],
            dtype=np.float64,
        ).astype(float).tolist(),
        "ee_position_error_m": float(ik_attempt["ee_position_error_m"]),
        "ee_orientation_error_deg": None if orientation_error_deg is None else float(orientation_error_deg),
        "approach_axis_offset_m": (
            None
            if ik_attempt["approach_axis_offset_m"] is None
            else float(ik_attempt["approach_axis_offset_m"])
        ),
        "lateral_offset_m": (
            None
            if ik_attempt["lateral_offset_m"] is None
            else float(ik_attempt["lateral_offset_m"])
        ),
        "collision_free": bool(ik_attempt["collision_free"]),
        "ik_feasible": bool(feasible),
        "ik_error_xyz_m": error_xyz.astype(float).tolist(),
        "ik_joint_solution_rad": joint_solution_rad.astype(float).tolist(),
        "ik_joint_solution_deg": joint_solution_deg.astype(float).tolist(),
        "joint_limit_margin_min_ratio": float(ik_attempt["joint_limit_margin_min_ratio"]),
        "joint_limit_margin_mean_ratio": float(ik_attempt["joint_limit_margin_mean_ratio"]),
        "joint_reset_delta_norm_l2": float(ik_attempt["joint_reset_delta_norm_l2"]),
        "refinement_applied": False,
    }


def _sample_feasible_ik_solutions(
    *,
    p_mod,
    robot_id: int,
    controllable_joint_ids: list[int],
    obstacle_body_ids: list[int],
    planning_config,
    target_pb: np.ndarray,
    target_rot_pb: np.ndarray,
    target_quat_pb: np.ndarray,
    rng_seed: int,
    position_tolerance_m: float,
    orientation_tolerance_deg: float,
    current_amcl_pose: RosMapPose2D | None = None,
    footprint_map: BaseFootprintMap | None = None,
    grasp_rank: int | None = None,
    num_samples: int = 10,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    rng = np.random.default_rng(int(rng_seed))
    valid_candidates: list[dict[str, object]] = []
    closest_candidate: dict[str, object] | None = None
    reference_base_yaw_pb = 0.0
    max_base_yaw_delta_rad = math.radians(BASE_LINK_YAW_MAX_DELTA_FROM_CURRENT_DEG)
    if footprint_map is None:
        sample_candidates: list[dict[str, object]] = []
        sample_stats: dict[str, object] = {
            "target_ros_map_xy": None,
            "safe_cell_count": 0,
            "region_cell_count": 0,
            "sampled_count": 0,
            "footprint_rejected_count": 0,
        }
    else:
        sample_candidates, sample_stats = _sample_base_link_poses_from_aligned_ros_map(
            footprint_map=footprint_map,
            current_amcl_pose=current_amcl_pose,
            target_pb=target_pb,
            target_rot_pb=target_rot_pb,
            rng=rng,
            max_samples=int(num_samples),
            max_yaw_delta_rad=max_base_yaw_delta_rad,
        )

    for sample_candidate in sample_candidates:
        local_pb_xy = sample_candidate["local_pb_xy"]
        local_pb_yaw = float(sample_candidate["local_pb_yaw_rad"])
        ik_attempt = _attempt_ik_at_base_pose(
            p_mod=p_mod,
            robot_id=robot_id,
            controllable_joint_ids=controllable_joint_ids,
            planning_config=planning_config,
            target_pb=target_pb,
            target_quat_pb=target_quat_pb,
            base_link_xy=local_pb_xy,
            base_link_yaw_rad=local_pb_yaw,
            obstacle_body_ids=obstacle_body_ids,
        )

        dist_error = float(ik_attempt["ee_position_error_m"])
        collision_free = bool(ik_attempt["collision_free"])
        joint_solution_rad = ik_attempt["ik_joint_solution_rad"]

        orientation_error_deg = ik_attempt["ee_orientation_error_deg"]
        orientation_ok = (
            orientation_error_deg is not None
            and float(orientation_error_deg) <= float(orientation_tolerance_deg)
        )
        feasible = (
            dist_error <= float(position_tolerance_m)
            and orientation_ok
            and collision_free
            and joint_solution_rad is not None
        )
        solution_record = _make_sampled_ik_solution_record(
            ik_attempt=ik_attempt,
            sample_candidate=sample_candidate,
            sample_stats=sample_stats,
            reference_base_yaw_pb=reference_base_yaw_pb,
            feasible=feasible,
        )
        if solution_record is None:
            continue

        if (
            closest_candidate is None
            or _closest_ik_solution_sort_key(solution_record)
            < _closest_ik_solution_sort_key(closest_candidate)
        ):
            closest_candidate = solution_record

        if feasible:
            valid_candidates.append(solution_record)

    if closest_candidate is None:
        direct_ik_attempt = _attempt_ik_at_base_pose(
            p_mod=p_mod,
            robot_id=robot_id,
            controllable_joint_ids=controllable_joint_ids,
            planning_config=planning_config,
            target_pb=target_pb,
            target_quat_pb=target_quat_pb,
            base_link_xy=(0.0, 0.0),
            base_link_yaw_rad=0.0,
            obstacle_body_ids=obstacle_body_ids,
        )
        direct_orientation_error = direct_ik_attempt["ee_orientation_error_deg"]
        direct_feasible = (
            float(direct_ik_attempt["ee_position_error_m"]) <= float(position_tolerance_m)
            and direct_orientation_error is not None
            and float(direct_orientation_error) <= float(orientation_tolerance_deg)
            and bool(direct_ik_attempt["collision_free"])
            and direct_ik_attempt["ik_joint_solution_rad"] is not None
        )
        direct_sample_candidate = {
            "sample_index": -1,
            "sample_source": "direct_current_base_fallback",
            "map_cell_index": -1,
            "local_pb_xy": (0.0, 0.0),
            "local_pb_yaw_rad": 0.0,
            "ros_map_pose": _local_pb_base_pose_to_ros_map_pose(
                (0.0, 0.0),
                0.0,
                current_amcl_pose,
            ),
            "ros_map_amcl_pose": current_amcl_pose,
            "distance_to_target_m": float(np.linalg.norm(np.asarray(target_pb, dtype=np.float64)[:2])),
            "approach_error_deg": float("inf"),
        }
        fallback_stats = {
            "region_cell_count": int(sample_stats["region_cell_count"]),
        }
        direct_solution_record = _make_sampled_ik_solution_record(
            ik_attempt=direct_ik_attempt,
            sample_candidate=direct_sample_candidate,
            sample_stats=fallback_stats,
            reference_base_yaw_pb=reference_base_yaw_pb,
            feasible=direct_feasible,
        )
        if direct_solution_record is not None:
            closest_candidate = direct_solution_record
            if direct_feasible:
                valid_candidates.append(direct_solution_record)

    valid_candidates.sort(key=_feasible_ik_solution_sort_key)
    if valid_candidates:
        valid_candidates[0]["selected_as_best"] = True
    if closest_candidate is not None:
        closest_candidate["selected_as_closest"] = True
    return valid_candidates, closest_candidate


def _select_closest_visualization_solution(
    visualization_records: list[dict[str, object]],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    closest_record: dict[str, object] | None = None
    closest_solution: dict[str, object] | None = None
    for record in visualization_records:
        solution = record.get("closest_solution")
        if not isinstance(solution, dict):
            continue
        if (
            closest_solution is None
            or _closest_ik_solution_sort_key(solution) < _closest_ik_solution_sort_key(closest_solution)
        ):
            closest_record = record
            closest_solution = solution
    return closest_record, closest_solution


def _rank_rgba(rank: int) -> tuple[float, float, float, float]:
    palette = (
        (0.95, 0.42, 0.24, 0.95),
        (0.20, 0.72, 0.40, 0.95),
        (0.20, 0.55, 0.95, 0.95),
        (0.95, 0.82, 0.22, 0.95),
        (0.70, 0.45, 0.95, 0.95),
        (0.15, 0.80, 0.82, 0.95),
    )
    return palette[(max(int(rank), 1) - 1) % len(palette)]


def _add_base_marker(
    p_mod,
    position_xyz: list[float],
    rgba: tuple[float, float, float, float],
    *,
    radius: float,
) -> None:
    visual_shape = p_mod.createVisualShape(
        p_mod.GEOM_SPHERE,
        radius=float(radius),
        rgbaColor=list(rgba),
    )
    p_mod.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual_shape,
        basePosition=[float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2])],
    )


def _animate_gui_ik_solution(
    p_mod,
    *,
    robot_id: int,
    controllable_joint_ids: list[int],
    planning_config,
    base_xyz: list[float],
    base_yaw_rad: float,
    joint_solution_rad: list[float],
) -> None:
    step_count = max(1, int(os.getenv("BASE_SAMPLER_GUI_ANIMATION_STEPS", "80")))
    frame_sleep_sec = max(0.0, float(os.getenv("BASE_SAMPLER_GUI_FRAME_SLEEP_SEC", "0.025")))
    joint_reset_rad = np.asarray(_degrees_to_radians(planning_config.joint_reset_deg), dtype=np.float64)
    joint_goal_rad = np.asarray(joint_solution_rad, dtype=np.float64)
    if len(joint_reset_rad) != len(joint_goal_rad):
        _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, joint_goal_rad)
        p_mod.stepSimulation()
        return

    p_mod.resetBasePositionAndOrientation(
        robot_id,
        base_xyz,
        p_mod.getQuaternionFromEuler([0.0, 0.0, float(base_yaw_rad)]),
    )
    for frame_index in range(step_count + 1):
        alpha = float(frame_index) / float(step_count)
        smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        joint_state = joint_reset_rad + (joint_goal_rad - joint_reset_rad) * smooth_alpha
        _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, joint_state)
        p_mod.stepSimulation()
        if frame_sleep_sec > 0.0:
            time.sleep(frame_sleep_sec)


def _spin_gui(p_mod, *, hold_seconds: float, time_step: float) -> None:
    if hold_seconds > 0.0:
        deadline = time.time() + float(hold_seconds)
        while p_mod.isConnected() and time.time() < deadline:
            p_mod.stepSimulation()
            time.sleep(min(max(float(time_step), 1e-3), 0.02))
        return

    try:
        while p_mod.isConnected():
            p_mod.stepSimulation()
            time.sleep(min(max(float(time_step), 1e-3), 0.02))
    except KeyboardInterrupt:
        return


def _compute_gui_camera_view(
    *,
    planning_config,
    visualization_records: list[dict[str, object]],
    best_view_solution: dict[str, object] | None,
) -> tuple[list[float], float]:
    points: list[np.ndarray] = [
        np.asarray([0.0, 0.0, float(planning_config.initial_height)], dtype=np.float64)
    ]
    for record in visualization_records:
        points.append(np.asarray(record["target_pb"], dtype=np.float64).reshape(3))
    if best_view_solution is not None:
        points.append(np.asarray(best_view_solution["pb_base_link_xyz"], dtype=np.float64).reshape(3))

    finite_points = [point for point in points if np.all(np.isfinite(point))]
    if not finite_points:
        return [0.0, 0.0, float(planning_config.initial_height)], 1.2

    point_arr = np.vstack(finite_points)
    lower = np.min(point_arr, axis=0)
    upper = np.max(point_arr, axis=0)
    center = 0.5 * (lower + upper)
    extent = float(np.max(upper - lower))
    camera_distance = max(1.2, extent * 1.8 + 0.6)
    return center.astype(float).tolist(), camera_distance


def _visualize_feasible_ik_results_in_gui(
    *,
    p_mod,
    pybullet_data,
    planning_config,
    arm_config,
    voxels_pb: np.ndarray,
    visualization_records: list[dict[str, object]],
) -> None:
    expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])
    hold_seconds = float(os.getenv("BASE_SAMPLER_GUI_HOLD_SEC", "0.0"))
    max_base_markers_per_grasp = int(os.getenv("BASE_SAMPLER_GUI_MAX_MARKERS_PER_GRASP", "40"))
    client_id: int | None = None

    try:
        client_id = p_mod.connect(p_mod.GUI)
        if client_id < 0:
            print("[test_base_sampler] PyBullet GUI unavailable; skipping visualization.", flush=True)
            return

        p_mod.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_mod.resetSimulation()
        p_mod.setGravity(0.0, 0.0, -9.8)
        p_mod.configureDebugVisualizer(p_mod.COV_ENABLE_GUI, 0)
        p_mod.loadURDF("plane.urdf")

        print(f"[test_base_sampler] GUI drawing {len(voxels_pb)} voxels...", flush=True)
        voxel_size = 0.05
        half_extents = [voxel_size / 2.0] * 3
        voxel_collision_shape = p_mod.createCollisionShape(
            p_mod.GEOM_BOX,
            halfExtents=half_extents,
        )
        voxel_visual_shape = p_mod.createVisualShape(
            p_mod.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.8, 0.2, 0.2, 0.8],
        )
        for voxel_center in voxels_pb:
            p_mod.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=voxel_collision_shape,
                baseVisualShapeIndex=voxel_visual_shape,
                basePosition=voxel_center.astype(float).tolist(),
            )

        base_orientation_rad = [math.radians(v) for v in planning_config.base_orientation_euler_deg]
        base_orientation_xyzw = p_mod.getQuaternionFromEuler(base_orientation_rad)
        robot_id = p_mod.loadURDF(
            planning_config.urdf_path,
            useFixedBase=True,
            basePosition=[0.0, 0.0, planning_config.initial_height],
            baseOrientation=base_orientation_xyzw,
        )
        print(
            "[test_base_sampler] GUI robot loaded at reset base="
            f"{[0.0, 0.0, float(planning_config.initial_height)]}",
            flush=True,
        )
        controllable_joint_ids, _ = _find_controllable_joints(p_mod, robot_id, expected_joint_count)
        joint_reset_rad = _degrees_to_radians(planning_config.joint_reset_deg)
        _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, joint_reset_rad)

        best_view_record: dict[str, object] | None = None
        best_view_solution: dict[str, object] | None = None
        total_feasible_count = 0
        camera_target = [0.0, 0.0, planning_config.initial_height]
        if visualization_records:
            camera_target = list(np.asarray(visualization_records[0]["target_pb"], dtype=float))

        target_visual_shape = p_mod.createVisualShape(
            p_mod.GEOM_SPHERE,
            radius=0.03,
            rgbaColor=[1.0, 0.8, 0.0, 1.0],
        )

        for record in visualization_records:
            rank = int(record["rank"])
            target_pb = np.asarray(record["target_pb"], dtype=float)
            target_quat_pb = np.asarray(record["target_quat_pb"], dtype=float)
            feasible_solutions = list(record["feasible_solutions"])
            closest_solution = record.get("closest_solution")
            total_feasible_count += len(feasible_solutions)
            rank_rgba = _rank_rgba(rank)

            print(
                f"[test_base_sampler] GUI drawing target rank={rank:02d} at "
                f"{target_pb.astype(float).tolist()}",
                flush=True,
            )
            p_mod.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=target_visual_shape,
                basePosition=target_pb.astype(float).tolist(),
            )
            _add_debug_axes(
                p_mod,
                target_pb.astype(float).tolist(),
                orientation_xyzw=target_quat_pb.astype(float).tolist(),
                axis_length=0.12,
                axis_width=1.8,
                label=f"TARGET G{rank} ({len(feasible_solutions)})",
            )
            p_mod.addUserDebugText(
                (
                    f"PB target xyz=({target_pb[0]:.3f}, "
                    f"{target_pb[1]:.3f}, {target_pb[2]:.3f})"
                ),
                textPosition=[float(target_pb[0]), float(target_pb[1]), float(target_pb[2] + 0.12)],
                textColorRGB=[1.0, 0.9, 0.2],
                textSize=0.9,
            )

            shown_solutions = sorted(
                feasible_solutions,
                key=_feasible_ik_solution_sort_key,
            )[: max(1, max_base_markers_per_grasp)]
            for marker_index, solution in enumerate(shown_solutions):
                _add_base_marker(
                    p_mod,
                    list(solution["pb_base_link_xyz"]),
                    rank_rgba,
                    radius=0.018 if marker_index == 0 else 0.012,
                )

            if isinstance(closest_solution, dict):
                _add_base_marker(
                    p_mod,
                    list(closest_solution["pb_base_link_xyz"]),
                    (1.0, 1.0, 1.0, 0.95),
                    radius=0.024,
                )
                best_solution_is_better = (
                    best_view_solution is None
                    or _closest_ik_solution_sort_key(closest_solution)
                    < _closest_ik_solution_sort_key(best_view_solution)
                )
                if best_solution_is_better:
                    best_view_record = record
                    best_view_solution = closest_solution

        if best_view_record is not None and best_view_solution is not None:
            rank = int(best_view_record["rank"])
            target_pb = np.asarray(best_view_record["target_pb"], dtype=float)
            base_xyz = [float(v) for v in best_view_solution["pb_base_link_xyz"]]
            base_yaw_rad = float(best_view_solution["pb_base_link_yaw_rad"])
            joint_solution_rad = [float(v) for v in best_view_solution["ik_joint_solution_rad"]]
            _animate_gui_ik_solution(
                p_mod,
                robot_id=robot_id,
                controllable_joint_ids=controllable_joint_ids,
                planning_config=planning_config,
                base_xyz=base_xyz,
                base_yaw_rad=base_yaw_rad,
                joint_solution_rad=joint_solution_rad,
            )
            p_mod.resetBasePositionAndOrientation(
                robot_id,
                base_xyz,
                p_mod.getQuaternionFromEuler([0.0, 0.0, base_yaw_rad]),
            )
            _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, joint_solution_rad)
            p_mod.performCollisionDetection()
            _add_debug_axes(
                p_mod,
                base_xyz,
                orientation_xyzw=p_mod.getQuaternionFromEuler([0.0, 0.0, base_yaw_rad]),
                axis_length=0.16,
                axis_width=1.8,
                label=f"BASE G{rank}",
            )
            p_mod.addUserDebugText(
                (
                    f"PB base xyz=({base_xyz[0]:.3f}, {base_xyz[1]:.3f}, "
                    f"{base_xyz[2]:.3f}) yaw={math.degrees(base_yaw_rad):.2f}deg"
                ),
                textPosition=[base_xyz[0], base_xyz[1], base_xyz[2] + 0.24],
                textColorRGB=[0.4, 0.9, 1.0],
                textSize=0.95,
            )
            camera_target = target_pb.astype(float).tolist()
            closest_state = "feasible" if bool(best_view_solution.get("ik_feasible", False)) else "closest"
            orientation_error = best_view_solution.get("ee_orientation_error_deg")
            orientation_error_text = "nan" if orientation_error is None else f"{float(orientation_error):.2f}"
            p_mod.addUserDebugText(
                (
                    f"rank={rank}  show={closest_state}  feasible={len(best_view_record['feasible_solutions'])}  "
                    f"ee_err={float(best_view_solution['ee_position_error_m']):.4f}m  "
                    f"ori_err={orientation_error_text}deg"
                ),
                textPosition=[target_pb[0], target_pb[1], target_pb[2] + 0.24],
                textColorRGB=[1.0, 1.0, 1.0],
                textSize=1.1,
            )
            print(
                f"[test_base_sampler] GUI open. Animating closest sampled IK rank={rank:02d} "
                f"({closest_state}) at base={base_xyz}, pb_gui_yaw={math.degrees(base_yaw_rad):.2f}deg, "
                f"ee_err={float(best_view_solution['ee_position_error_m']):.4f}m.",
                flush=True,
            )
        else:
            reset_base_xyz = [0.0, 0.0, float(planning_config.initial_height)]
            _add_debug_axes(
                p_mod,
                reset_base_xyz,
                orientation_xyzw=base_orientation_xyzw,
                axis_length=0.18,
                axis_width=1.8,
                label="RESET BASE",
            )
            p_mod.addUserDebugText(
                (
                    f"PB reset base xyz=({reset_base_xyz[0]:.3f}, "
                    f"{reset_base_xyz[1]:.3f}, {reset_base_xyz[2]:.3f})"
                ),
                textPosition=[reset_base_xyz[0], reset_base_xyz[1], reset_base_xyz[2] + 0.24],
                textColorRGB=[0.4, 0.9, 1.0],
                textSize=0.95,
            )
            print(
                "[test_base_sampler] GUI open. No sampled IK attempt was available; "
                "showing reset robot, scene, and targets.",
                flush=True,
            )

        camera_target, camera_distance = _compute_gui_camera_view(
            planning_config=planning_config,
            visualization_records=visualization_records,
            best_view_solution=best_view_solution,
        )
        p_mod.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=45.0,
            cameraPitch=-25.0,
            cameraTargetPosition=camera_target,
        )
        p_mod.addUserDebugText(
            (
                f"grasp candidates={len(visualization_records)}  "
                f"total feasible IK={total_feasible_count}"
            ),
            textPosition=[camera_target[0], camera_target[1], camera_target[2] + 0.35],
            textColorRGB=[1.0, 1.0, 0.2],
            textSize=1.1,
        )

        _spin_gui(p_mod, hold_seconds=hold_seconds, time_step=1.0 / 240.0)
    except Exception as exc:
        print(f"[test_base_sampler] GUI visualization failed: {exc}", flush=True)
    finally:
        if client_id is not None:
            try:
                p_mod.disconnect(client_id)
            except Exception:
                pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live integration harness for Camera_Car RGBD + grasp-agent JSON + "
            "/amcl_pose + PyBullet IK base-pose sampling + GUI replay."
        )
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base_pose_sampling.yaml"),
        help="Base sampler YAML. Used for planner path and tolerances.",
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
        help=(
            "Grasp-agent raw result JSON. Defaults to GRASP_RESULT_JSON, then the "
            "preferred step_007 path, then latest logs/sessions/*/step_*_grasp_agent.json."
        ),
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=int(os.getenv("BASE_SAMPLER_NUM_SAMPLES", "10")),
        help="Random PB base samples per grasp target.",
    )
    parser.add_argument(
        "--allow-missing-amcl",
        action="store_true",
        help="Continue even if no /amcl_pose was received during camera capture.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip PyBullet GUI replay.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    base_config_path = args.base_config
    camera_config_path = args.camera_config

    cfg = load_config(base_config_path)
    planning_config = load_planning_config(Path(cfg["planner_config_path"]))
    arm_config = _load_arm_config()
    _, p_mod, pybullet_data = _load_python_dependencies()
    position_tolerance_m = float(cfg.get("position_tolerance_m", planning_config.position_tolerance_m))
    orientation_tolerance_deg = float(cfg.get("orientation_tolerance_deg", 12.0))
    footprint_map = _build_base_footprint_map(Path(cfg["map_yaml_path"]))
    (
        camera_in_base_link_rotation,
        camera_in_base_link_position,
        base_link_z_pb,
    ) = _get_reset_camera_transform_in_base_link_frame(
        Path(cfg["planner_config_path"]),
        planning_config,
    )
    grasp_candidates, _, grasp_result_json_path = _load_grasp_candidates_from_result_json(args.grasp_json)

    live_scene = _capture_live_scene_voxels(
        camera_config_path.resolve(),
        camera_in_base_link_rotation,
        camera_in_base_link_position,
        base_link_z_pb,
    )
    voxels_pb = live_scene.voxel_centers_pb
    camera_to_pb_rotation = live_scene.camera_to_pb_rotation
    camera_position_pb = live_scene.camera_position_pb
    current_amcl_pose = _amcl_snapshot_to_ros_map_pose(live_scene.amcl_pose)
    if current_amcl_pose is None and not args.allow_missing_amcl:
        raise RuntimeError(
            "No /amcl_pose was received during Camera_Car capture. "
            "Use --allow-missing-amcl only if you want a PB-local GUI replay."
        )
    print(
        f"[test_base_sampler] grasp_json={grasp_result_json_path} | "
        f"grasps={len(grasp_candidates)} | live RGBD voxels={len(voxels_pb)} | "
        f"depth_points={live_scene.valid_depth_point_count} | "
        f"amcl={'yes' if current_amcl_pose is not None else 'no'}",
        flush=True,
    )
    print(
        "[test_base_sampler] aligned ROS map sampling: "
        f"rect_footprint_pb_x={footprint_map.length_x_m:.3f}m "
        f"rect_footprint_pb_y={footprint_map.length_y_m:.3f}m "
        f"base_link_from_amcl_pb_xy=[{BASE_LINK_FROM_AMCL_PB_XY[0]:.4f}, "
        f"{BASE_LINK_FROM_AMCL_PB_XY[1]:.4f}]m "
        f"map_resolution={footprint_map.resolution_m:.3f}m "
        f"free_cells={len(footprint_map.free_cell_keys)} "
        f"footprint_check_points={len(footprint_map.footprint_points_pb_xy)} "
        f"distance=[{BASE_SAMPLE_MIN_DISTANCE_M:.2f},{BASE_SAMPLE_MAX_DISTANCE_M:.2f}]m "
        f"approach_half_angle={BASE_SAMPLE_APPROACH_HALF_ANGLE_DEG:.1f}deg "
        f"amcl_align={'yes' if current_amcl_pose is not None else 'no'}",
        flush=True,
    )
    _print_ros_map_pose("current /amcl_pose vehicle center", current_amcl_pose)
    current_base_link_ros_pose = (
        None
        if current_amcl_pose is None
        else _local_pb_base_pose_to_ros_map_pose((0.0, 0.0), 0.0, current_amcl_pose)
    )
    _print_ros_map_pose("current ROS map base_link derived from /amcl_pose + offset", current_base_link_ros_pose)
    current_amcl_pb_pose = _amcl_pose_to_pb_world_pose(
        current_amcl_pose,
        z_pb=0.0,
    )
    if current_amcl_pb_pose is None:
        _print_pb_pose("current /amcl_pose vehicle center converted to PB map frame", [])
        _print_pb_pose("current PB base_link derived from /amcl_pose + offset", [])
    else:
        current_amcl_pb_xyz, current_amcl_pb_yaw = current_amcl_pb_pose
        _print_pb_pose(
            "current /amcl_pose vehicle center converted to PB map frame",
            current_amcl_pb_xyz,
            current_amcl_pb_yaw,
        )
        current_base_link_pb_xyz = _base_link_world_from_amcl_pb_pose(
            current_amcl_pb_xyz,
            current_amcl_pb_yaw,
            z_pb=base_link_z_pb,
        )
        _print_pb_pose(
            "current PB base_link derived from /amcl_pose + offset",
            current_base_link_pb_xyz,
            current_amcl_pb_yaw,
        )
    _print_camera_to_pb_axis_mapping()
    _print_pb_pose("reset depth camera local PB pose from default arm posture", camera_position_pb)

    client_id = p_mod.connect(p_mod.DIRECT)
    visualization_records: list[dict[str, object]] = []
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
                    basePosition=voxel_center.tolist(),
                )
            )

        robot_id = p_mod.loadURDF(
            planning_config.urdf_path,
            useFixedBase=True,
            basePosition=[0.0, 0.0, planning_config.initial_height],
            baseOrientation=p_mod.getQuaternionFromEuler([0.0, 0.0, 0.0]),
        )
        expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])
        controllable_joint_ids, _ = _find_controllable_joints(p_mod, robot_id, expected_joint_count)
        prepared_targets = _prepare_ranked_grasp_targets(
            p_mod=p_mod,
            robot_id=robot_id,
            controllable_joint_ids=controllable_joint_ids,
            planning_config=planning_config,
            grasp_candidates=grasp_candidates,
            camera_to_pb_rotation=camera_to_pb_rotation,
            camera_position_pb=camera_position_pb,
        )

        for prepared_target in prepared_targets:
            grasp_candidate = prepared_target.grasp_candidate
            target_pb = prepared_target.target_pb
            target_rot_pb = prepared_target.target_rot_pb
            target_quat_pb = prepared_target.target_quat_pb
            feasible_solutions, closest_solution = _sample_feasible_ik_solutions(
                p_mod=p_mod,
                robot_id=robot_id,
                controllable_joint_ids=controllable_joint_ids,
                obstacle_body_ids=obstacle_body_ids,
                planning_config=planning_config,
                target_pb=target_pb,
                target_rot_pb=target_rot_pb,
                target_quat_pb=target_quat_pb,
                rng_seed=42 + grasp_candidate.rank,
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_deg=orientation_tolerance_deg,
                current_amcl_pose=current_amcl_pose,
                footprint_map=footprint_map,
                grasp_rank=grasp_candidate.rank,
                num_samples=int(args.num_samples),
            )

            visualization_records.append(
                {
                    "rank": grasp_candidate.rank,
                    "grasp_confidence": grasp_candidate.grasp_confidence,
                    "direct_ik_error_m": float(prepared_target.direct_ik_error_m),
                    "direct_orientation_error_deg": prepared_target.direct_orientation_error_deg,
                    "target_pb": target_pb.astype(float).tolist(),
                    "target_quat_pb": target_quat_pb.astype(float).tolist(),
                    "feasible_solutions": feasible_solutions,
                    "closest_solution": closest_solution,
                }
            )

            if feasible_solutions:
                break
    finally:
        p_mod.disconnect(client_id)

    closest_record, closest_solution = _select_closest_visualization_solution(visualization_records)
    if closest_record is None:
        _print_closest_ik_distance_banner(
            rank=0,
            target_pb=np.zeros(3, dtype=np.float64),
            solution=None,
        )
    else:
        _print_closest_ik_distance_banner(
            rank=int(closest_record["rank"]),
            target_pb=np.asarray(closest_record["target_pb"], dtype=np.float64),
            solution=closest_solution,
        )

    if not args.no_gui:
        _visualize_feasible_ik_results_in_gui(
            p_mod=p_mod,
            pybullet_data=pybullet_data,
            planning_config=planning_config,
            arm_config=arm_config,
            voxels_pb=voxels_pb,
            visualization_records=visualization_records,
        )


if __name__ == "__main__":
    main()
