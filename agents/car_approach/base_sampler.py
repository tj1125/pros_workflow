import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.spatial.transform as st

from . import move_car, sample_logic
from .scripts.run_base_pose_sampling import load_config
from .src.camera_car_voxel_ompl import load_camera_car_voxel_ompl_config
from .src.geometry.depth_backprojection import decode_depth_png_bytes
from .src.geometry.map_loader import load_free_cells_unity_xz, load_map_meta
from .src.geometry.voxelization import voxelize_points
from .src.io.camera_capture import AmclPoseSnapshot, capture_rgbd_snapshot
from .src.io.intrinsics import load_camera_intrinsics
from .src.pybullet_ompl import (
    _add_debug_axes,
    _compute_pose_alignment_metrics,
    _find_controllable_joints,
    _load_ompl_dependencies,
    _set_joint_positions_direct,
    get_reset_camera_transform,
    load_planning_config,
)
from .src.pybullet_smoke import (
    _degrees_to_radians,
    _load_arm_config,
    _load_python_dependencies,
)


GraspPoseCandidate = sample_logic.GraspPoseCandidate


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
    voxel_size_m: float
    camera_to_pb_rotation: np.ndarray
    camera_position_pb: np.ndarray
    amcl_pose: AmclPoseSnapshot | None
    captured_amcl_pose: AmclPoseSnapshot | None
    valid_depth_point_count: int
    obstacle_depth_point_count: int
    target_object_point_count: int
    target_excluded_depth_point_count: int


@dataclass(frozen=True)
class RosMapPose2D:
    x: float
    y: float
    yaw_rad: float


@dataclass(frozen=True)
class MapFreeSpace:
    free_cell_keys: frozenset[tuple[int, int]]
    origin_xy: tuple[float, float]
    resolution_m: float
    vehicle_footprint_points_pb_xy: np.ndarray


@dataclass(frozen=True)
class ApproachAgentRunConfig:
    base_config_path: Path = Path("configs/base_pose_sampling.yaml")
    camera_config_path: Path = Path("configs/camera_car_voxel_ompl.yaml")
    grasp_json_path: Path | None = None
    grasp_result_payload: dict[str, object] | None = None
    allow_missing_amcl: bool = False
    run_rule_navigation: bool = True
    evaluate_current_pose_only: bool = False
    show_gui: bool = False
    write_map_png: bool = True
    map_png_path: Path | None = None


APPROACH_AGENT_DIR = Path(__file__).resolve().parent
LIVE_VOXEL_MIN_DEPTH_M = 0.19
LIVE_VOXEL_MAX_DEPTH_M = 1.0
MAX_GRASP_POSES_TO_EVALUATE = 10
TARGET_OBJECT_POINTCLOUD_KEY = "object_pc_camera"
BASE_LINK_YAW_MAX_DELTA_FROM_CURRENT_DEG = 20.0
BASE_LINK_FROM_AMCL_PB_XY = np.asarray([0.0, 0.1288], dtype=np.float64)
VEHICLE_BASE_LENGTH_X_M = 0.30
VEHICLE_BASE_LENGTH_Y_M = 0.32


PLANNING_CAMERA_TO_PB_LOCAL = np.asarray(
    [
        [0.0, 0.0, 1.0],   # pb_x =  cam_z
        [-1.0, 0.0, 0.0],  # pb_y = -cam_x
        [0.0, -1.0, 0.0],  # pb_z = -cam_y
    ],
    dtype=np.float64,
)


def _resolve_grasp_result_json_path(grasp_result_json_path: Path | None = None) -> Path:
    return sample_logic.resolve_grasp_result_json_path(grasp_result_json_path)


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


def _build_map_free_space(map_yaml_path: Path) -> MapFreeSpace:
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
    vehicle_footprint_points_pb_xy = _rectangular_footprint_points_pb_xy(
        length_x_m=VEHICLE_BASE_LENGTH_X_M,
        length_y_m=VEHICLE_BASE_LENGTH_Y_M,
        resolution_m=resolution,
    )

    return MapFreeSpace(
        free_cell_keys=free_cell_keys,
        origin_xy=(origin_x, origin_y),
        resolution_m=resolution,
        vehicle_footprint_points_pb_xy=vehicle_footprint_points_pb_xy,
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
            covariance=tuple(float(value) for value in node.latest_msg.pose.covariance),
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


def _print_ros_map_pose(label: str, pose: RosMapPose2D | dict[str, object] | None) -> None:
    if pose is None:
        print(f"[base_approach] {label}: unavailable", flush=True)
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
        f"[base_approach] {label}: "
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


def _run_rule_navigation_for_solution(
    solution: dict[str, object],
    *,
    initial_pose: AmclPoseSnapshot | None,
) -> dict[str, object]:
    if not _env_flag("APPROACH_AGENT_RULE_NAV", True):
        print("[base_approach] rule navigation disabled by APPROACH_AGENT_RULE_NAV.", flush=True)
        return {"success": False, "skipped": True, "phase": "disabled_by_env"}

    publish_initial_pose = _env_flag("APPROACH_AGENT_PUBLISH_INITIAL_POSE", True)
    initial_pose_topic = os.getenv("APPROACH_AGENT_INITIAL_POSE_TOPIC", move_car.DEFAULT_INITIAL_POSE_TOPIC).strip()
    initial_pose_topic = initial_pose_topic or move_car.DEFAULT_INITIAL_POSE_TOPIC
    initial_pose_publish_count = int(os.getenv("APPROACH_AGENT_INITIAL_POSE_PUBLISH_COUNT", "5"))
    initial_pose_interval_sec = float(os.getenv("APPROACH_AGENT_INITIAL_POSE_INTERVAL_SEC", "0.1"))
    initial_pose_wait_for_subscribers_sec = float(
        os.getenv("APPROACH_AGENT_INITIAL_POSE_WAIT_FOR_SUBSCRIBERS_SEC", "2.0")
    )
    rule_config = move_car.RuleNavigationConfig(
        amcl_topic=os.getenv("APPROACH_AGENT_AMCL_TOPIC", move_car.DEFAULT_AMCL_TOPIC).strip()
        or move_car.DEFAULT_AMCL_TOPIC,
        initial_pose_topic=initial_pose_topic,
        initial_pose_frame_id=move_car.DEFAULT_FRAME_ID,
        initial_pose_publish_count=initial_pose_publish_count if publish_initial_pose else 0,
        initial_pose_interval_sec=initial_pose_interval_sec,
        initial_pose_wait_for_subscribers_sec=initial_pose_wait_for_subscribers_sec,
        front_wheel_topic=os.getenv(
            "APPROACH_AGENT_FRONT_WHEEL_TOPIC",
            move_car.DEFAULT_FRONT_WHEEL_TOPIC,
        ).strip()
        or move_car.DEFAULT_FRONT_WHEEL_TOPIC,
        rear_wheel_topic=os.getenv(
            "APPROACH_AGENT_REAR_WHEEL_TOPIC",
            move_car.DEFAULT_REAR_WHEEL_TOPIC,
        ).strip()
        or move_car.DEFAULT_REAR_WHEEL_TOPIC,
        xy_tolerance_m=float(os.getenv("APPROACH_AGENT_RULE_XY_TOLERANCE_M", "0.03")),
        face_target_yaw_tolerance_rad=float(os.getenv("APPROACH_AGENT_RULE_FACE_YAW_TOLERANCE_RAD", "0.08")),
        drive_heading_tolerance_rad=float(os.getenv("APPROACH_AGENT_RULE_DRIVE_HEADING_TOLERANCE_RAD", "0.14")),
        final_yaw_tolerance_rad=float(os.getenv("APPROACH_AGENT_RULE_FINAL_YAW_TOLERANCE_RAD", "0.08")),
        slow_approach_distance_m=float(os.getenv("APPROACH_AGENT_RULE_SLOW_DISTANCE_M", "0.12")),
        command_period_sec=float(os.getenv("APPROACH_AGENT_RULE_COMMAND_PERIOD_SEC", "0.1")),
        amcl_wait_timeout_sec=float(os.getenv("APPROACH_AGENT_RULE_AMCL_WAIT_TIMEOUT_SEC", "5.0")),
        amcl_stale_timeout_sec=float(os.getenv("APPROACH_AGENT_RULE_AMCL_STALE_TIMEOUT_SEC", "1.0")),
        max_duration_sec=float(os.getenv("APPROACH_AGENT_RULE_MAX_DURATION_SEC", "120.0")),
        stop_repeat=int(os.getenv("APPROACH_AGENT_RULE_STOP_REPEAT", "5")),
        stop_interval_sec=float(os.getenv("APPROACH_AGENT_RULE_STOP_INTERVAL_SEC", "0.03")),
        log_interval_sec=float(os.getenv("APPROACH_AGENT_RULE_LOG_INTERVAL_SEC", "1.0")),
    )
    if initial_pose is None:
        print(
            "[base_approach] /initialpose source unavailable: no capture-time /amcl_pose.",
            flush=True,
        )
    else:
        initial_yaw = _yaw_from_quaternion_xyzw(initial_pose.orientation_xyzw)
        print(
            "[base_approach] /initialpose source=capture-time /amcl_pose: "
            f"x={initial_pose.position_xyz[0]:.3f} "
            f"y={initial_pose.position_xyz[1]:.3f} "
            f"yaw={initial_yaw:.3f} "
            f"stamp={initial_pose.stamp_sec}",
            flush=True,
        )
    result = move_car.drive_to_pose_by_rule(
        solution,
        initial_pose=initial_pose if publish_initial_pose else None,
        config=rule_config,
        prefer_amcl_pose=True,
    )
    print(
        "[base_approach] rule navigation result = "
        f"{result}.",
        flush=True,
    )
    return result

def _print_closest_ik_solution_banner(
    *,
    rank: int,
    target_pb: np.ndarray,
    solution: dict[str, object] | None,
) -> None:
    if solution is None:
        print(
            "\n"
            "############################################################\n"
            " CLOSEST SAMPLED IK ATTEMPT\n"
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
    base_xyz = np.asarray(solution["pb_base_link_xyz"], dtype=np.float64).reshape(3)
    ros_base_link_pose = solution.get("ros_map_base_link_pose")
    ros_amcl_pose = solution.get("ros_map_amcl_pose")
    orientation_error = solution.get("ee_orientation_error_deg")
    approach_axis_offset = solution.get("approach_axis_offset_m")
    lateral_offset = solution.get("lateral_offset_m")
    backoff_distance = solution.get("backoff_distance_m")
    yaw_delta_deg = solution.get("ros_map_amcl_yaw_delta_from_current_deg")
    yaw_limit_deg = solution.get("ros_map_amcl_yaw_max_delta_from_current_deg")

    print(
        "\n"
        "############################################################\n"
        " CLOSEST SAMPLED IK ATTEMPT\n"
        "------------------------------------------------------------\n"
        f" grasp_rank : {int(rank):02d}\n"
        f" ik_feasible: {int(bool(solution.get('ik_feasible', False)))}\n"
        f" source     : {solution.get('sample_source', 'unknown')}\n"
        f" sample_idx : {int(solution['sample_index'])}\n"
        f" region_cell: {int(solution['ros_map_sample_region_cell_count'])}\n"
        f" backoff    : {'nan' if backoff_distance is None else f'{float(backoff_distance):.4f}'} m\n"
        f" map_clear  : {int(bool(solution.get('map_clear', False)))}\n"
        f" yaw_ok     : {int(bool(solution.get('amcl_yaw_within_limit', False)))}"
        f"  delta={'nan' if yaw_delta_deg is None else f'{float(yaw_delta_deg):.2f}'} deg"
        f"  limit={'nan' if yaw_limit_deg is None else f'{float(yaw_limit_deg):.2f}'} deg\n"
        f" yaw_clamp  : {int(bool(solution.get('base_yaw_clamped', False)))}"
        f"  desired_pb_yaw={float(solution.get('desired_pb_base_link_yaw_deg', solution['pb_base_link_yaw_deg'])):.2f} deg\n"
        f" ik_reach   : {int(bool(solution.get('ik_reachable', False)))}\n"
        "------------------------------------------------------------\n"
        " AMCL / VEHICLE CENTER ROS MAP\n"
        f"{_format_ros_map_pose_lines(ros_amcl_pose)}"
        "------------------------------------------------------------\n"
        " BASE_LINK ROS MAP\n"
        f"{_format_ros_map_pose_lines(ros_base_link_pose)}"
        "------------------------------------------------------------\n"
        " BASE_LINK LOCAL PB\n"
        f" pb_x       : {float(base_xyz[0]):.4f} m\n"
        f" pb_y       : {float(base_xyz[1]):.4f} m\n"
        f" pb_z       : {float(base_xyz[2]):.4f} m\n"
        f" pb_yaw     : {float(solution['pb_base_link_yaw_rad']):.6f} rad  "
        f"({float(solution['pb_base_link_yaw_deg']):.2f} deg)\n"
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
        print(f"[base_approach] {label}: unavailable", flush=True)
        return
    yaw_text = ""
    if yaw_rad is not None:
        yaw = float(yaw_rad)
        yaw_text = f" yaw={yaw:.6f}rad ({math.degrees(yaw):.2f}deg)"
    print(
        f"[base_approach] {label}: "
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
        "[base_approach] camera point -> PB local axis mapping: "
        f"cam +X -> {basis_pb_local[0].astype(float).tolist()}, "
        f"cam +Y -> {basis_pb_local[1].astype(float).tolist()}, "
        f"cam +Z -> {basis_pb_local[2].astype(float).tolist()} "
        "(rule: pb=[cam_z, -cam_x, -cam_y])",
        flush=True,
    )


def _load_grasp_candidates_from_result_json(
    grasp_result_json_path: Path | None = None,
) -> tuple[list[GraspPoseCandidate], np.ndarray, Path]:
    return sample_logic.load_grasp_candidates_from_result_json(grasp_result_json_path)


def _voxel_downsample_points(points_xyz: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if len(points_xyz) == 0 or voxel_size_m <= 0.0:
        return np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    buckets = np.floor(np.asarray(points_xyz, dtype=np.float32) / float(voxel_size_m)).astype(np.int32)
    _, keep_indices = np.unique(buckets, axis=0, return_index=True)
    return np.asarray(points_xyz[np.sort(keep_indices)], dtype=np.float32)


def _repo_relative_existing_path(path: Path) -> Path:
    path = path.expanduser()
    if path.exists():
        return path.resolve()

    marker = "/VLM_RL/"
    path_text = str(path)
    if marker in path_text:
        repo_relative = path_text.split(marker, 1)[1]
        candidate = Path(__file__).resolve().parents[2] / repo_relative
        if candidate.exists():
            return candidate.resolve()
    return path


def _load_target_object_pointcloud_camera(npz_path: Path | None) -> np.ndarray | None:
    if npz_path is None:
        return None

    resolved_path = _repo_relative_existing_path(Path(npz_path))
    if not resolved_path.exists():
        print(
            f"[base_approach] target object point cloud skipped: NPZ not found: {resolved_path}",
            flush=True,
        )
        return None

    try:
        with np.load(resolved_path, allow_pickle=True) as payload:
            if TARGET_OBJECT_POINTCLOUD_KEY not in payload:
                print(
                    f"[base_approach] target object point cloud skipped: "
                    f"'{TARGET_OBJECT_POINTCLOUD_KEY}' missing in {resolved_path}",
                    flush=True,
                )
                return None
            points_camera = np.asarray(payload[TARGET_OBJECT_POINTCLOUD_KEY], dtype=np.float32).reshape(-1, 3)
    except Exception as exc:
        print(
            f"[base_approach] target object point cloud skipped: failed to load {resolved_path}: {exc}",
            flush=True,
        )
        return None

    finite_mask = np.all(np.isfinite(points_camera), axis=1)
    points_camera = points_camera[finite_mask]
    if len(points_camera) == 0:
        print(
            f"[base_approach] target object point cloud skipped: no finite points in {resolved_path}",
            flush=True,
        )
        return None

    print(
        f"[base_approach] loaded target object point cloud: {len(points_camera)} points from {resolved_path}",
        flush=True,
    )
    return np.asarray(points_camera, dtype=np.float32)


def _filter_points_outside_target_voxels(
    points_pb: np.ndarray,
    target_points_pb: np.ndarray | None,
    *,
    voxel_size_m: float,
) -> tuple[np.ndarray, int]:
    points_pb = np.asarray(points_pb, dtype=np.float64).reshape(-1, 3)
    if target_points_pb is None or len(target_points_pb) == 0 or len(points_pb) == 0 or voxel_size_m <= 0.0:
        return points_pb, 0

    target_points_pb = np.asarray(target_points_pb, dtype=np.float64).reshape(-1, 3)
    target_points_pb = target_points_pb[np.all(np.isfinite(target_points_pb), axis=1)]
    if len(target_points_pb) == 0:
        return points_pb, 0

    voxel_size = float(voxel_size_m)
    point_keys = np.floor(points_pb / voxel_size).astype(np.int64)
    target_keys = np.unique(np.floor(target_points_pb / voxel_size).astype(np.int64), axis=0)
    target_key_set = {tuple(key) for key in target_keys.tolist()}
    keep_mask = np.fromiter(
        (tuple(key) not in target_key_set for key in point_keys.tolist()),
        dtype=bool,
        count=len(point_keys),
    )
    removed_count = int(len(points_pb) - int(np.count_nonzero(keep_mask)))
    return points_pb[keep_mask], removed_count


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
    target_object_points_camera: np.ndarray | None = None,
) -> LiveSceneCapture:
    camera_cfg = load_camera_car_voxel_ompl_config(camera_config_path)
    snapshot = capture_rgbd_snapshot(
        camera_cfg.camera_name,
        timeout_sec=camera_cfg.capture_timeout_sec,
        amcl_topic=camera_cfg.amcl_topic,
    )
    captured_amcl_pose = snapshot.amcl_pose
    amcl_pose = captured_amcl_pose
    if amcl_pose is None:
        amcl_pose = _wait_for_amcl_pose(
            camera_cfg.amcl_topic,
            timeout_sec=camera_cfg.capture_timeout_sec,
        )
    if captured_amcl_pose is None:
        print(
            "[base_approach] RGBD capture had no simultaneous /amcl_pose; "
            "/initialpose will be skipped before rule navigation.",
            flush=True,
        )
    else:
        print(
            "[base_approach] recorded capture-time /amcl_pose for /initialpose: "
            f"x={captured_amcl_pose.position_xyz[0]:.3f} "
            f"y={captured_amcl_pose.position_xyz[1]:.3f} "
            f"stamp={captured_amcl_pose.stamp_sec}",
            flush=True,
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
        stride=camera_cfg.pixel_stride,
    )
    points_camera = _voxel_downsample_points(points_camera, voxel_size_m=camera_cfg.voxel_size_m)
    if len(points_camera) == 0:
        raise RuntimeError("Live Camera_Car RGBD produced zero valid camera-frame points.")

    points_pybullet = _transform_camera_points_to_local_pb(
        points_camera,
        camera_to_pb_rotation=camera_to_pb_rotation,
        camera_position_pb=camera_position_pb,
    )
    target_object_point_count = 0
    target_excluded_point_count = 0
    if target_object_points_camera is not None and len(target_object_points_camera) > 0:
        target_object_points_camera = np.asarray(target_object_points_camera, dtype=np.float64).reshape(-1, 3)
        target_object_point_count = int(len(target_object_points_camera))
        target_points_pybullet = _transform_camera_points_to_local_pb(
            target_object_points_camera,
            camera_to_pb_rotation=camera_to_pb_rotation,
            camera_position_pb=camera_position_pb,
        )
        points_pybullet, target_excluded_point_count = _filter_points_outside_target_voxels(
            points_pybullet,
            target_points_pybullet,
            voxel_size_m=camera_cfg.voxel_size_m,
        )
        print(
            f"[base_approach] excluded target object from live obstacle voxels: "
            f"target_points={target_object_point_count} "
            f"removed_depth_points={target_excluded_point_count} "
            f"remaining_obstacle_points={len(points_pybullet)}",
            flush=True,
        )
    voxel_centers_pb = voxelize_points(
        points_pybullet,
        voxel_size_m=camera_cfg.voxel_size_m,
        max_voxels=camera_cfg.max_voxel_obstacles,
    )
    if len(voxel_centers_pb) == 0 and target_excluded_point_count <= 0:
        raise RuntimeError("Live Camera_Car RGBD produced zero occupied voxels.")
    if len(voxel_centers_pb) == 0:
        print(
            "[base_approach] live obstacle voxels are empty after target-object exclusion.",
            flush=True,
        )
    return LiveSceneCapture(
        voxel_centers_pb=np.asarray(voxel_centers_pb, dtype=np.float64),
        voxel_size_m=float(camera_cfg.voxel_size_m),
        camera_to_pb_rotation=np.asarray(camera_to_pb_rotation, dtype=np.float64),
        camera_position_pb=np.asarray(camera_position_pb, dtype=np.float64),
        amcl_pose=amcl_pose,
        captured_amcl_pose=captured_amcl_pose,
        valid_depth_point_count=int(len(points_camera)),
        obstacle_depth_point_count=int(len(points_pybullet)),
        target_object_point_count=int(target_object_point_count),
        target_excluded_depth_point_count=int(target_excluded_point_count),
    )


def _transform_grasp_pose_camera_to_pybullet(
    grasp_candidate: GraspPoseCandidate,
    *,
    camera_to_pb_rotation: np.ndarray,
    camera_position_pb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return sample_logic.transform_grasp_pose_camera_to_pybullet(
        grasp_candidate,
        camera_to_pb_rotation=camera_to_pb_rotation,
        camera_position_pb=camera_position_pb,
    )


def _pb_world_xy_to_ros_map_xy(pb_xy: tuple[float, float]) -> tuple[float, float]:
    pb_x, pb_y = float(pb_xy[0]), float(pb_xy[1])
    return (-pb_y, pb_x)


def _ros_map_xy_to_pb_world_xy(ros_map_xy: tuple[float, float]) -> tuple[float, float]:
    ros_x, ros_y = float(ros_map_xy[0]), float(ros_map_xy[1])
    return (ros_y, -ros_x)


def _pb_yaw_to_ros_map_yaw(pb_yaw_rad: float) -> float:
    return _wrap_angle_rad(float(pb_yaw_rad))


def _ros_map_yaw_to_pb_yaw(ros_yaw_rad: float) -> float:
    return _wrap_angle_rad(float(ros_yaw_rad))


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


def _amcl_pose_from_local_base_pose(
    *,
    current_amcl_pose: RosMapPose2D | None,
    local_pb_xy: np.ndarray,
    local_pb_yaw_rad: float,
) -> tuple[RosMapPose2D | None, RosMapPose2D]:
    base_link_pose = _local_pb_base_pose_to_ros_map_pose(
        (float(local_pb_xy[0]), float(local_pb_xy[1])),
        float(local_pb_yaw_rad),
        current_amcl_pose,
    )
    if current_amcl_pose is None:
        return None, base_link_pose

    base_link_pb_pose = _base_link_pose_to_pb_world_pose(base_link_pose, base_link_z_pb=0.0)
    assert base_link_pb_pose is not None
    base_link_pb_xyz, base_link_pb_yaw = base_link_pb_pose
    offset_pb_xy = _yaw_rotation_matrix(base_link_pb_yaw)[:2, :2] @ BASE_LINK_FROM_AMCL_PB_XY
    amcl_pb_xy = base_link_pb_xyz[:2] - offset_pb_xy
    amcl_ros_x, amcl_ros_y = _pb_world_xy_to_ros_map_xy(
        (float(amcl_pb_xy[0]), float(amcl_pb_xy[1]))
    )
    return (
        RosMapPose2D(
            x=float(amcl_ros_x),
            y=float(amcl_ros_y),
            yaw_rad=float(base_link_pose.yaw_rad),
        ),
        base_link_pose,
    )


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
        [-direction_pb_world_xy[1], direction_pb_world_xy[0]],
        dtype=np.float64,
    )
    direction_norm = float(np.linalg.norm(direction_ros_xy))
    if direction_norm <= 1e-6:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    return direction_ros_xy / direction_norm


def _ros_map_pose_is_clear_on_map(
    map_free_space: MapFreeSpace,
    ros_map_pose: RosMapPose2D,
) -> bool:
    amcl_pb_pose = _amcl_pose_to_pb_world_pose(ros_map_pose, z_pb=0.0)
    if amcl_pb_pose is None:
        return False
    amcl_pb_xyz, amcl_pb_yaw = amcl_pb_pose
    footprint_points_pb = np.asarray(
        map_free_space.vehicle_footprint_points_pb_xy,
        dtype=np.float64,
    ).reshape(-1, 2)
    rot_xy = _yaw_rotation_matrix(amcl_pb_yaw)[:2, :2]
    footprint_world_pb_xy = amcl_pb_xyz[:2].reshape(1, 2) + footprint_points_pb @ rot_xy.T
    footprint_world_ros_xy = np.asarray(
        [_pb_world_xy_to_ros_map_xy((float(pb_x), float(pb_y))) for pb_x, pb_y in footprint_world_pb_xy],
        dtype=np.float64,
    )
    origin_x, origin_y = map_free_space.origin_xy
    resolution = float(map_free_space.resolution_m)
    keys = zip(
        np.rint((footprint_world_ros_xy[:, 0] - origin_x) / resolution).astype(np.int32),
        np.rint((footprint_world_ros_xy[:, 1] - origin_y) / resolution).astype(np.int32),
    )
    return all((int(key_x), int(key_y)) in map_free_space.free_cell_keys for key_x, key_y in keys)


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


def _joint_bounds_rad_from_planning_config(planning_config, joint_count: int) -> tuple[list[float], list[float]]:
    if len(planning_config.joint_bounds_deg) != int(joint_count):
        raise ValueError(
            "joint_bounds_deg length does not match controllable joint count: "
            f"{len(planning_config.joint_bounds_deg)} vs {joint_count}"
        )
    lower_bounds = [math.radians(float(lower)) for lower, _ in planning_config.joint_bounds_deg]
    upper_bounds = [math.radians(float(upper)) for _, upper in planning_config.joint_bounds_deg]
    return lower_bounds, upper_bounds


def _joint_values_within_bounds(
    joint_values: list[float] | tuple[float, ...] | np.ndarray,
    lower_bounds: list[float],
    upper_bounds: list[float],
) -> bool:
    return all(
        float(lower) <= float(value) <= float(upper)
        for value, lower, upper in zip(joint_values, lower_bounds, upper_bounds)
    )


def _ompl_state_values(state: object, dimension: int) -> list[float]:
    return [float(state[index]) for index in range(int(dimension))]


def _ompl_path_states(path: object, dimension: int) -> list[list[float]]:
    return [_ompl_state_values(path.getState(index), dimension) for index in range(path.getStateCount())]


def _closest_robot_obstacle_distance_and_collision(
    p_mod,
    robot_id: int,
    obstacle_body_ids: list[int],
    *,
    query_distance_m: float,
    collision_threshold_m: float,
) -> tuple[float | None, bool]:
    closest_distance: float | None = None
    in_collision = False
    query_distance = max(float(query_distance_m), float(collision_threshold_m), 0.0)
    threshold = float(collision_threshold_m)
    for obstacle_body_id in obstacle_body_ids:
        closest_points = p_mod.getClosestPoints(robot_id, obstacle_body_id, distance=query_distance)
        for point in closest_points:
            distance = float(point[8])
            if closest_distance is None or distance < closest_distance:
                closest_distance = distance
            if distance <= threshold:
                in_collision = True
    return closest_distance, in_collision


def _check_ompl_path_for_ik_solution(
    *,
    p_mod,
    robot_id: int,
    controllable_joint_ids: list[int],
    planning_config,
    base_xyz: list[float],
    base_yaw_rad: float,
    goal_joint_solution_rad: np.ndarray,
    obstacle_body_ids: list[int],
) -> dict[str, object]:
    result: dict[str, object] = {
        "ompl_checked": True,
        "ompl_path_found": False,
        "ompl_path_collision_free": False,
        "ompl_planning_time_sec": 0.0,
        "ompl_path_state_count": 0,
        "ompl_min_distance_along_path_m": None,
        "ompl_first_collision_state_index": None,
        "ompl_error": None,
    }
    if len(obstacle_body_ids) == 0:
        result.update(
            {
                "ompl_path_found": True,
                "ompl_path_collision_free": True,
                "ompl_error": "skipped_no_obstacles",
            }
        )
        return result

    try:
        ob, og = _load_ompl_dependencies()
        dimension = len(controllable_joint_ids)
        start_joint_positions = [float(v) for v in _degrees_to_radians(planning_config.joint_reset_deg)]
        goal_joint_positions = [float(v) for v in np.asarray(goal_joint_solution_rad, dtype=np.float64).reshape(-1)[:dimension]]
        if len(start_joint_positions) != dimension or len(goal_joint_positions) != dimension:
            result["ompl_error"] = (
                "joint vector length mismatch: "
                f"start={len(start_joint_positions)} goal={len(goal_joint_positions)} expected={dimension}"
            )
            return result

        lower_bounds, upper_bounds = _joint_bounds_rad_from_planning_config(planning_config, dimension)
        if not _joint_values_within_bounds(start_joint_positions, lower_bounds, upper_bounds):
            result["ompl_error"] = "reset joint state is outside joint_bounds_deg"
            return result
        if not _joint_values_within_bounds(goal_joint_positions, lower_bounds, upper_bounds):
            result["ompl_error"] = "IK goal joint state is outside joint_bounds_deg"
            return result

        base_quat = p_mod.getQuaternionFromEuler([0.0, 0.0, float(base_yaw_rad)])

        def _set_candidate_state(joint_values: list[float] | tuple[float, ...]) -> tuple[float | None, bool]:
            p_mod.resetBasePositionAndOrientation(robot_id, base_xyz, base_quat)
            _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, joint_values)
            p_mod.performCollisionDetection()
            return _closest_robot_obstacle_distance_and_collision(
                p_mod,
                robot_id,
                obstacle_body_ids,
                query_distance_m=float(planning_config.collision_query_distance_m),
                collision_threshold_m=float(planning_config.collision_threshold_m),
            )

        _, start_in_collision = _set_candidate_state(start_joint_positions)
        if start_in_collision:
            result["ompl_error"] = "reset joint state is in collision"
            return result
        _, goal_in_collision = _set_candidate_state(goal_joint_positions)
        if goal_in_collision:
            result["ompl_error"] = "IK goal joint state is in collision"
            return result

        space = ob.RealVectorStateSpace(dimension)
        bounds = ob.RealVectorBounds(dimension)
        for index in range(dimension):
            bounds.setLow(index, lower_bounds[index])
            bounds.setHigh(index, upper_bounds[index])
        space.setBounds(bounds)

        si = ob.SpaceInformation(space)

        def _is_state_valid(state: object) -> bool:
            joint_values = _ompl_state_values(state, dimension)
            if not _joint_values_within_bounds(joint_values, lower_bounds, upper_bounds):
                return False
            _, in_collision = _set_candidate_state(joint_values)
            return not in_collision

        si.setStateValidityChecker(ob.StateValidityCheckerFn(_is_state_valid))
        si.setup()

        start_state = ob.State(space)
        goal_state = ob.State(space)
        for index, value in enumerate(start_joint_positions):
            start_state[index] = float(value)
        for index, value in enumerate(goal_joint_positions):
            goal_state[index] = float(value)

        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start_state, goal_state)
        planner = og.RRTConnect(si)
        if hasattr(planner, "setRange") and float(planning_config.planning_range_rad) > 0.0:
            planner.setRange(float(planning_config.planning_range_rad))
        planner.setProblemDefinition(pdef)
        planner.setup()

        planning_start_time = time.perf_counter()
        solved = planner.solve(float(planning_config.planning_timeout_sec))
        result["ompl_planning_time_sec"] = float(time.perf_counter() - planning_start_time)
        if not solved:
            result["ompl_error"] = (
                f"OMPL RRTConnect failed within {float(planning_config.planning_timeout_sec):.3f}s"
            )
            return result

        path = pdef.getSolutionPath()
        target_state_count = max(int(planning_config.path_interpolation_states), int(path.getStateCount()))
        if target_state_count > path.getStateCount():
            path.interpolate(target_state_count)
        path_states = _ompl_path_states(path, dimension)

        min_distance_along_path: float | None = None
        first_collision_state_index: int | None = None
        for state_index, joint_values in enumerate(path_states):
            min_distance, in_collision = _set_candidate_state(joint_values)
            if min_distance is not None:
                if min_distance_along_path is None or min_distance < min_distance_along_path:
                    min_distance_along_path = min_distance
            if in_collision:
                first_collision_state_index = state_index
                break

        result["ompl_path_found"] = True
        result["ompl_path_state_count"] = int(len(path_states))
        result["ompl_min_distance_along_path_m"] = min_distance_along_path
        result["ompl_first_collision_state_index"] = first_collision_state_index
        result["ompl_path_collision_free"] = first_collision_state_index is None
        if first_collision_state_index is not None:
            result["ompl_error"] = f"OMPL path collided at interpolated state {first_collision_state_index}"
        return result
    except Exception as exc:
        result["ompl_error"] = str(exc)
        return result


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
    enable_ompl_path_check: bool = True,
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

    ompl_result: dict[str, object] = {
        "ompl_checked": False,
        "ompl_path_found": False,
        "ompl_path_collision_free": False,
        "ompl_planning_time_sec": None,
        "ompl_path_state_count": 0,
        "ompl_min_distance_along_path_m": None,
        "ompl_first_collision_state_index": None,
        "ompl_error": None,
    }
    if enable_ompl_path_check and joint_solution_rad is not None and obstacle_body_ids is not None:
        ompl_result = _check_ompl_path_for_ik_solution(
            p_mod=p_mod,
            robot_id=robot_id,
            controllable_joint_ids=controllable_joint_ids,
            planning_config=planning_config,
            base_xyz=base_xyz,
            base_yaw_rad=float(base_link_yaw_rad),
            goal_joint_solution_rad=joint_solution_rad,
            obstacle_body_ids=obstacle_body_ids,
        )

    result = {
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
    result.update(ompl_result)
    return result


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
        "ompl_checked": bool(ik_attempt.get("ompl_checked", False)),
        "ompl_path_found": bool(ik_attempt.get("ompl_path_found", False)),
        "ompl_path_collision_free": bool(ik_attempt.get("ompl_path_collision_free", False)),
        "ompl_planning_time_sec": ik_attempt.get("ompl_planning_time_sec"),
        "ompl_path_state_count": int(ik_attempt.get("ompl_path_state_count", 0) or 0),
        "ompl_min_distance_along_path_m": ik_attempt.get("ompl_min_distance_along_path_m"),
        "ompl_first_collision_state_index": ik_attempt.get("ompl_first_collision_state_index"),
        "ompl_error": ik_attempt.get("ompl_error"),
        "refinement_applied": False,
    }


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


def _vehicle_amcl_from_base_link_local_pb(
    base_xyz: list[float] | np.ndarray,
    base_yaw_rad: float,
) -> np.ndarray:
    base_arr = np.asarray(base_xyz, dtype=np.float64).reshape(3)
    offset_xy = _yaw_rotation_matrix(float(base_yaw_rad))[:2, :2] @ BASE_LINK_FROM_AMCL_PB_XY
    return np.asarray(
        [
            float(base_arr[0] - offset_xy[0]),
            float(base_arr[1] - offset_xy[1]),
            float(base_arr[2]),
        ],
        dtype=np.float64,
    )


def _add_vehicle_base_visual(
    p_mod,
    *,
    base_xyz: list[float] | np.ndarray,
    base_yaw_rad: float,
) -> None:
    body_height_m = float(os.getenv("BASE_SAMPLER_GUI_VEHICLE_BODY_HEIGHT_M", "0.05"))
    half_extents = [
        VEHICLE_BASE_LENGTH_X_M * 0.5,
        VEHICLE_BASE_LENGTH_Y_M * 0.5,
        max(body_height_m, 1e-3) * 0.5,
    ]
    amcl_xyz = _vehicle_amcl_from_base_link_local_pb(base_xyz, base_yaw_rad)
    visual_shape = p_mod.createVisualShape(
        p_mod.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=[0.05, 0.25, 1.0, 0.42],
    )
    p_mod.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual_shape,
        basePosition=[float(amcl_xyz[0]), float(amcl_xyz[1]), float(half_extents[2])],
        baseOrientation=p_mod.getQuaternionFromEuler([0.0, 0.0, float(base_yaw_rad)]),
    )


def _add_arm_base_link_debug_axes(
    p_mod,
    *,
    base_xyz: list[float] | np.ndarray,
    base_yaw_rad: float | None = None,
    orientation_xyzw: list[float] | tuple[float, float, float, float] | None = None,
    label: str = "arm base_link",
) -> None:
    base_position = np.asarray(base_xyz, dtype=np.float64).reshape(3)
    axis_length = float(os.getenv("BASE_SAMPLER_GUI_BASE_LINK_AXIS_LENGTH_M", "0.18"))
    axis_width = float(os.getenv("BASE_SAMPLER_GUI_BASE_LINK_AXIS_WIDTH", "4.0"))
    if orientation_xyzw is None:
        orientation_xyzw = p_mod.getQuaternionFromEuler([0.0, 0.0, float(base_yaw_rad or 0.0)])
    _add_debug_axes(
        p_mod,
        base_position.astype(float).tolist(),
        orientation_xyzw=orientation_xyzw,
        axis_length=axis_length,
        axis_width=axis_width,
        label=label,
    )


def _draw_map_clear_candidate_points(
    p_mod,
    *,
    visualization_records: list[dict[str, object]],
    z_pb: float,
) -> int:
    radius = float(os.getenv("BASE_SAMPLER_GUI_MAP_CLEAR_POINT_RADIUS_M", "0.005"))
    radius = max(radius, 1e-5)
    visual_shape = p_mod.createVisualShape(
        p_mod.GEOM_SPHERE,
        radius=radius,
        rgbaColor=[0.1, 0.9, 1.0, 0.95],
    )
    drawn_count = 0
    for record in visualization_records:
        candidates = record.get("map_clear_candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            local_xy = candidate.get("local_pb_xy")
            if local_xy is None:
                continue
            local_xy_arr = np.asarray(local_xy, dtype=np.float64).reshape(2)
            p_mod.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual_shape,
                basePosition=[
                    float(local_xy_arr[0]),
                    float(local_xy_arr[1]),
                    float(z_pb),
                ],
            )
            drawn_count += 1
    print(
        f"[base_approach] GUI drew {drawn_count} ROS-map-clear sample points "
        f"(radius={radius:.4f}m).",
        flush=True,
    )
    return drawn_count


def _ros_map_pose_like_to_dict(pose: object) -> dict[str, float] | None:
    if isinstance(pose, RosMapPose2D):
        return _ros_map_pose_to_dict(pose)
    if isinstance(pose, dict):
        try:
            yaw_rad = float(pose["yaw_rad"])
            return {
                "x": float(pose["x"]),
                "y": float(pose["y"]),
                "yaw_rad": yaw_rad,
                "yaw_deg": float(pose.get("yaw_deg", math.degrees(yaw_rad))),
            }
        except Exception:
            return None
    return None


def _gui_solution_from_map_candidate(
    map_candidate: dict[str, object],
    *,
    planning_config,
) -> dict[str, object] | None:
    local_xy = map_candidate.get("local_pb_xy")
    if local_xy is None:
        return None

    local_xy_arr = np.asarray(local_xy, dtype=np.float64).reshape(2)
    base_yaw_rad = float(map_candidate.get("local_pb_yaw_rad", 0.0))
    ros_map_amcl_pose = _ros_map_pose_like_to_dict(map_candidate.get("ros_map_amcl_pose"))
    ros_map_base_link_pose = _ros_map_pose_like_to_dict(map_candidate.get("ros_map_pose"))
    return {
        "sample_index": int(map_candidate.get("sample_index", -1)),
        "sample_source": str(map_candidate.get("sample_source", "map_clear_candidate")),
        "pb_base_link_xyz": [
            float(local_xy_arr[0]),
            float(local_xy_arr[1]),
            float(planning_config.initial_height),
        ],
        "pb_base_link_yaw_rad": float(base_yaw_rad),
        "pb_base_link_yaw_deg": float(math.degrees(base_yaw_rad)),
        "ik_joint_solution_rad": None,
        "map_clear": True,
        "amcl_yaw_within_limit": True,
        "ik_reachable": False,
        "ik_feasible": False,
        "ee_position_error_m": None,
        "ee_orientation_error_deg": None,
        "backoff_distance_m": float(map_candidate.get("distance_to_target_m", float("nan"))),
        "ros_map_amcl_pose": ros_map_amcl_pose,
        "ros_map_base_link_pose": ros_map_base_link_pose,
    }


def _select_gui_display_solution(
    visualization_records: list[dict[str, object]],
    *,
    planning_config,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    best_record: dict[str, object] | None = None
    best_solution: dict[str, object] | None = None

    for record in visualization_records:
        feasible_solutions = record.get("feasible_solutions")
        if not isinstance(feasible_solutions, list):
            continue
        for solution in feasible_solutions:
            if not isinstance(solution, dict):
                continue
            if not (
                bool(solution.get("ik_feasible", False))
                and bool(solution.get("map_clear", False))
                and bool(solution.get("amcl_yaw_within_limit", False))
            ):
                continue
            if best_solution is None or _closest_ik_solution_sort_key(solution) < _closest_ik_solution_sort_key(best_solution):
                best_record = record
                best_solution = solution
    if best_solution is not None:
        return best_record, best_solution

    for record in visualization_records:
        solution = record.get("closest_solution")
        if not isinstance(solution, dict):
            continue
        if best_solution is None or _closest_ik_solution_sort_key(solution) < _closest_ik_solution_sort_key(best_solution):
            best_record = record
            best_solution = solution
    if best_solution is not None:
        return best_record, best_solution

    for record in visualization_records:
        candidates = record.get("map_clear_candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            solution = _gui_solution_from_map_candidate(candidate, planning_config=planning_config)
            if solution is not None:
                return record, solution

    return None, None


def _write_solution_vehicle_base_ros_map_png(
    *,
    map_yaml_path: Path,
    solution: dict[str, object],
    output_path: Path,
) -> bool:
    amcl_pose_dict = _ros_map_pose_like_to_dict(solution.get("ros_map_amcl_pose"))
    if amcl_pose_dict is None:
        print(
            "[base_approach] map_occupied.png skipped: solution has no ROS-map AMCL pose.",
            flush=True,
        )
        return False

    try:
        from PIL import Image, ImageDraw  # type: ignore[import]
    except ImportError:
        print(
            "[base_approach] map_occupied.png skipped: Pillow is not installed.",
            flush=True,
        )
        return False

    map_meta = load_map_meta(map_yaml_path)
    image = Image.open(map_meta.pgm_path).convert("RGB")
    _, height_px = image.size

    amcl_pose = RosMapPose2D(
        x=float(amcl_pose_dict["x"]),
        y=float(amcl_pose_dict["y"]),
        yaw_rad=float(amcl_pose_dict["yaw_rad"]),
    )
    amcl_pb_pose = _amcl_pose_to_pb_world_pose(amcl_pose, z_pb=0.0)
    assert amcl_pb_pose is not None
    amcl_pb_xyz, amcl_pb_yaw = amcl_pb_pose

    half_x = VEHICLE_BASE_LENGTH_X_M * 0.5
    half_y = VEHICLE_BASE_LENGTH_Y_M * 0.5
    corners_local = np.asarray(
        [
            [-half_x, -half_y],
            [half_x, -half_y],
            [half_x, half_y],
            [-half_x, half_y],
        ],
        dtype=np.float64,
    )
    rot_xy = _yaw_rotation_matrix(amcl_pb_yaw)[:2, :2]
    corners_pb_xy = amcl_pb_xyz[:2].reshape(1, 2) + corners_local @ rot_xy.T
    corners_ros_xy = np.column_stack([-corners_pb_xy[:, 1], corners_pb_xy[:, 0]])

    origin_x, origin_y = map_meta.origin_xy
    resolution = float(map_meta.resolution_m)
    polygon_px = [
        (
            float((ros_x - origin_x) / resolution),
            float(height_px - 1 - ((ros_y - origin_y) / resolution)),
        )
        for ros_x, ros_y in corners_ros_xy
    ]

    draw = ImageDraw.Draw(image)
    draw.polygon(polygon_px, fill=(0, 96, 255), outline=(0, 32, 180))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(
        f"[base_approach] wrote ROS-map occupied footprint image: {output_path} "
        f"amcl=({amcl_pose.x:.3f}, {amcl_pose.y:.3f}, {math.degrees(amcl_pose.yaw_rad):.2f}deg)",
        flush=True,
    )
    return True


def _write_best_display_solution_ros_map_png(
    *,
    map_yaml_path: Path,
    planning_config,
    visualization_records: list[dict[str, object]],
    output_path: Path,
) -> bool:
    _, solution = _select_gui_display_solution(
        visualization_records,
        planning_config=planning_config,
    )
    if solution is None:
        print(
            "[base_approach] map_occupied.png skipped: no display solution or map-clear candidate.",
            flush=True,
        )
        return False
    return _write_solution_vehicle_base_ros_map_png(
        map_yaml_path=map_yaml_path,
        solution=solution,
        output_path=output_path,
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
) -> tuple[list[float], float, float, float]:
    points: list[np.ndarray] = [
        np.asarray([0.0, 0.0, float(planning_config.initial_height)], dtype=np.float64)
    ]
    for record in visualization_records:
        points.append(np.asarray(record["target_pb"], dtype=np.float64).reshape(3))
    if best_view_solution is not None:
        points.append(np.asarray(best_view_solution["pb_base_link_xyz"], dtype=np.float64).reshape(3))
        final_ee_position = best_view_solution.get("final_ee_position_xyz")
        if final_ee_position is not None:
            points.append(np.asarray(final_ee_position, dtype=np.float64).reshape(3))

    finite_points = [point for point in points if np.all(np.isfinite(point))]
    if not finite_points:
        return [0.0, 0.0, float(planning_config.initial_height)], 1.2, 90.0, 0.0

    point_arr = np.vstack(finite_points)
    lower = np.min(point_arr, axis=0)
    upper = np.max(point_arr, axis=0)
    center = 0.5 * (lower + upper)
    extent = float(np.max(upper - lower))
    camera_distance = max(1.2, extent * 2.1 + 0.7)
    camera_yaw = 90.0

    yaw_override = os.getenv("BASE_SAMPLER_GUI_CAMERA_YAW_DEG", "").strip()
    if yaw_override:
        camera_yaw = float(yaw_override)
    camera_pitch = float(os.getenv("BASE_SAMPLER_GUI_CAMERA_PITCH_DEG", "0.0"))
    camera_yaw = ((float(camera_yaw) + 180.0) % 360.0) - 180.0
    return center.astype(float).tolist(), camera_distance, camera_yaw, camera_pitch


def _visualize_feasible_ik_results_in_gui(
    *,
    p_mod,
    pybullet_data,
    planning_config,
    arm_config,
    voxels_pb: np.ndarray,
    voxel_size_m: float,
    visualization_records: list[dict[str, object]],
) -> None:
    expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])
    hold_seconds = float(os.getenv("BASE_SAMPLER_GUI_HOLD_SEC", "0.0"))
    client_id: int | None = None

    try:
        client_id = p_mod.connect(p_mod.GUI)
        if client_id < 0:
            print("[base_approach] PyBullet GUI unavailable; skipping visualization.", flush=True)
            return

        p_mod.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_mod.resetSimulation()
        p_mod.setGravity(0.0, 0.0, -9.8)
        p_mod.configureDebugVisualizer(p_mod.COV_ENABLE_GUI, 0)
        p_mod.loadURDF("plane.urdf")

        voxel_size = float(voxel_size_m)
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
        for voxel_center in np.asarray(voxels_pb, dtype=np.float64).reshape(-1, 3):
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
        controllable_joint_ids, _ = _find_controllable_joints(p_mod, robot_id, expected_joint_count)
        joint_reset_rad = _degrees_to_radians(planning_config.joint_reset_deg)
        _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, joint_reset_rad)

        best_view_record: dict[str, object] | None = None
        best_view_solution: dict[str, object] | None = None
        best_animation_base_xyz: list[float] | None = None
        best_animation_base_yaw_rad: float | None = None
        best_animation_joint_solution_rad: list[float] | None = None
        camera_target = [0.0, 0.0, planning_config.initial_height]
        if visualization_records:
            camera_target = list(np.asarray(visualization_records[0]["target_pb"], dtype=float))

        target_visual_shape = p_mod.createVisualShape(
            p_mod.GEOM_SPHERE,
            radius=0.03,
            rgbaColor=[1.0, 0.8, 0.0, 1.0],
        )
        best_view_record, best_view_solution = _select_gui_display_solution(
            visualization_records,
            planning_config=planning_config,
        )

        if best_view_solution is None:
            records_with_closest = sum(
                1 for record in visualization_records if isinstance(record.get("closest_solution"), dict)
            )
            records_with_map_candidate = sum(
                1
                for record in visualization_records
                if isinstance(record.get("map_clear_candidates"), list) and len(record["map_clear_candidates"]) > 0
            )
            print(
                "[base_approach] GUI has no feasible, closest, or map-clear base candidate to display "
                f"(records={len(visualization_records)} "
                f"closest_records={records_with_closest} "
                f"map_candidate_records={records_with_map_candidate}).",
                flush=True,
            )

        if best_view_record is not None and best_view_solution is not None:
            rank = int(best_view_record["rank"])
            target_pb = np.asarray(best_view_record["target_pb"], dtype=float)
            p_mod.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=target_visual_shape,
                basePosition=target_pb.astype(float).tolist(),
            )
            base_xyz = [float(v) for v in best_view_solution["pb_base_link_xyz"]]
            base_yaw_rad = float(best_view_solution["pb_base_link_yaw_rad"])
            raw_joint_solution = best_view_solution.get("ik_joint_solution_rad")
            joint_solution_rad = (
                None
                if raw_joint_solution is None
                else [float(v) for v in raw_joint_solution]
            )
            if joint_solution_rad is not None:
                best_animation_base_xyz = base_xyz
                best_animation_base_yaw_rad = base_yaw_rad
                best_animation_joint_solution_rad = joint_solution_rad
            p_mod.resetBasePositionAndOrientation(
                robot_id,
                base_xyz,
                p_mod.getQuaternionFromEuler([0.0, 0.0, base_yaw_rad]),
            )
            _set_joint_positions_direct(
                p_mod,
                robot_id,
                controllable_joint_ids,
                joint_solution_rad if joint_solution_rad is not None else joint_reset_rad,
            )
            p_mod.performCollisionDetection()
            _add_arm_base_link_debug_axes(
                p_mod,
                base_xyz=base_xyz,
                base_yaw_rad=base_yaw_rad,
            )
            _add_vehicle_base_visual(
                p_mod,
                base_xyz=base_xyz,
                base_yaw_rad=base_yaw_rad,
            )
            camera_target = target_pb.astype(float).tolist()
            map_clear = bool(best_view_solution.get("map_clear", False))
            yaw_ok = bool(best_view_solution.get("amcl_yaw_within_limit", False))
            ik_reachable = bool(best_view_solution.get("ik_reachable", False))
            closest_state = (
                "feasible"
                if bool(best_view_solution.get("ik_feasible", False)) and map_clear and yaw_ok
                else "closest/debug"
            )
            orientation_error = best_view_solution.get("ee_orientation_error_deg")
            orientation_error_text = "nan" if orientation_error is None else f"{float(orientation_error):.2f}"
            ee_position_error = best_view_solution.get("ee_position_error_m")
            ee_position_error_text = (
                "nan"
                if ee_position_error is None
                else f"{float(ee_position_error):.4f}"
            )
            p_mod.addUserDebugText(
                (
                    f"rank={rank}  show={closest_state}  feasible={len(best_view_record['feasible_solutions'])}  "
                    f"map_clear={int(map_clear)}  yaw_ok={int(yaw_ok)}  ik_reach={int(ik_reachable)}  "
                    f"ee_err={ee_position_error_text}m  "
                    f"ori_err={orientation_error_text}deg"
                ),
                textPosition=[target_pb[0], target_pb[1], target_pb[2] + 0.24],
                textColorRGB=[1.0, 1.0, 1.0],
                textSize=1.1,
            )
            print(
                f"[base_approach] GUI showing {closest_state} solution for "
                f"grasp_rank={rank:02d}: "
                f"sample_idx={int(best_view_solution['sample_index'])} "
                f"backoff={float(best_view_solution.get('backoff_distance_m', float('nan'))):.3f}m "
                f"map_clear={int(map_clear)} "
                f"yaw_ok={int(yaw_ok)} "
                f"ik_reach={int(ik_reachable)} "
                f"ee_err={ee_position_error_text}m "
                f"ori_err={orientation_error_text}",
                flush=True,
            )
            if joint_solution_rad is None:
                print(
                    "[base_approach] GUI has no IK joint solution to animate; "
                    "showing the closest ROS-map-clear base candidate only.",
                    flush=True,
                )

        camera_records = [best_view_record] if best_view_record is not None else []
        if camera_records:
            camera_target, camera_distance, camera_yaw, camera_pitch = _compute_gui_camera_view(
                planning_config=planning_config,
                visualization_records=camera_records,
                best_view_solution=best_view_solution,
            )
        else:
            reset_base_xyz = np.asarray(
                [0.0, 0.0, float(planning_config.initial_height)],
                dtype=np.float64,
            )
            _add_arm_base_link_debug_axes(
                p_mod,
                base_xyz=reset_base_xyz,
                orientation_xyzw=base_orientation_xyzw,
                label="reset arm base_link",
            )
            scene_points = [reset_base_xyz.reshape(1, 3)]
            voxel_points = np.asarray(voxels_pb, dtype=np.float64).reshape(-1, 3)
            if len(voxel_points) > 0:
                finite_voxels = voxel_points[np.all(np.isfinite(voxel_points), axis=1)]
                if len(finite_voxels) > 0:
                    scene_points.append(finite_voxels)
            point_arr = np.vstack(scene_points)
            lower = np.min(point_arr, axis=0)
            upper = np.max(point_arr, axis=0)
            camera_target = (0.5 * (lower + upper)).astype(float).tolist()
            scene_extent = float(np.max(upper - lower))
            camera_distance = max(0.9, scene_extent * 1.8 + 0.45)
            camera_yaw = float(os.getenv("BASE_SAMPLER_GUI_CAMERA_YAW_DEG", "90.0"))
            camera_pitch = float(os.getenv("BASE_SAMPLER_GUI_CAMERA_PITCH_DEG", "0.0"))
            camera_yaw = ((camera_yaw + 180.0) % 360.0) - 180.0
        p_mod.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target,
        )
        if (
            best_animation_base_xyz is not None
            and best_animation_base_yaw_rad is not None
            and best_animation_joint_solution_rad is not None
        ):
            _animate_gui_ik_solution(
                p_mod,
                robot_id=robot_id,
                controllable_joint_ids=controllable_joint_ids,
                planning_config=planning_config,
                base_xyz=best_animation_base_xyz,
                base_yaw_rad=best_animation_base_yaw_rad,
                joint_solution_rad=best_animation_joint_solution_rad,
            )
            p_mod.resetBasePositionAndOrientation(
                robot_id,
                best_animation_base_xyz,
                p_mod.getQuaternionFromEuler([0.0, 0.0, best_animation_base_yaw_rad]),
            )
            _set_joint_positions_direct(
                p_mod,
                robot_id,
                controllable_joint_ids,
                best_animation_joint_solution_rad,
            )
            p_mod.performCollisionDetection()

        _spin_gui(p_mod, hold_seconds=hold_seconds, time_step=1.0 / 240.0)
    except Exception as exc:
        print(f"[base_approach] GUI visualization failed: {exc}", flush=True)
    finally:
        if client_id is not None:
            try:
                p_mod.disconnect(client_id)
            except Exception:
                pass


def _run_simple_sample_logic_for_gui(
    *,
    p_mod,
    pybullet_data,
    cfg: dict[str, object],
    planning_config,
    arm_config,
    voxels_pb: np.ndarray,
    voxel_size_m: float,
    visualization_records: list[dict[str, object]],
    map_free_space: MapFreeSpace,
    current_amcl_pose: RosMapPose2D | None,
    evaluate_current_pose_only: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    client_id = p_mod.connect(p_mod.DIRECT)
    if client_id < 0:
        print("[base_approach] PyBullet DIRECT unavailable; GUI will show grasp poses only.", flush=True)
        return visualization_records, None

    try:
        p_mod.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_mod.resetSimulation()
        p_mod.setGravity(0.0, 0.0, -9.8)
        p_mod.loadURDF("plane.urdf")

        voxel_size = float(voxel_size_m)
        half_extents = [voxel_size / 2.0] * 3
        col_shape = p_mod.createCollisionShape(p_mod.GEOM_BOX, halfExtents=half_extents)
        vis_shape = p_mod.createVisualShape(
            p_mod.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.8, 0.2, 0.2, 0.8],
        )
        obstacle_body_ids: list[int] = []
        for voxel_center in np.asarray(voxels_pb, dtype=np.float64).reshape(-1, 3):
            obstacle_body_ids.append(
                p_mod.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=col_shape,
                    baseVisualShapeIndex=vis_shape,
                    basePosition=voxel_center.astype(float).tolist(),
                )
            )

        base_orientation_rad = [math.radians(v) for v in planning_config.base_orientation_euler_deg]
        robot_id = p_mod.loadURDF(
            planning_config.urdf_path,
            useFixedBase=True,
            basePosition=[0.0, 0.0, float(planning_config.initial_height)],
            baseOrientation=p_mod.getQuaternionFromEuler(base_orientation_rad),
        )
        expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])
        controllable_joint_ids, _ = _find_controllable_joints(p_mod, robot_id, expected_joint_count)
        _set_joint_positions_direct(
            p_mod,
            robot_id,
            controllable_joint_ids,
            _degrees_to_radians(planning_config.joint_reset_deg),
        )
        p_mod.performCollisionDetection()
        reset_ee_state = p_mod.getLinkState(
            robot_id,
            planning_config.ee_link_index,
            computeForwardKinematics=True,
        )
        reset_ee_position_xyz = np.asarray(reset_ee_state[4], dtype=np.float64)
        reset_ee_orientation_xyzw = np.asarray(reset_ee_state[5], dtype=np.float64)

        def _pose_from_local_base(local_xy: np.ndarray, local_yaw: float):
            return _amcl_pose_from_local_base_pose(
                current_amcl_pose=current_amcl_pose,
                local_pb_xy=np.asarray(local_xy, dtype=np.float64),
                local_pb_yaw_rad=float(local_yaw),
            )

        def _map_pose_is_clear(ros_map_amcl_pose: object) -> bool:
            if not isinstance(ros_map_amcl_pose, RosMapPose2D):
                return False
            return _ros_map_pose_is_clear_on_map(map_free_space, ros_map_amcl_pose)

        def _attempt_ik(local_xy: np.ndarray, local_yaw: float, record: dict[str, object]) -> dict[str, object]:
            return _attempt_ik_at_base_pose(
                p_mod=p_mod,
                robot_id=robot_id,
                controllable_joint_ids=controllable_joint_ids,
                planning_config=planning_config,
                target_pb=np.asarray(record["target_pb"], dtype=np.float64),
                target_quat_pb=np.asarray(record["target_quat_pb"], dtype=np.float64),
                base_link_xy=(float(local_xy[0]), float(local_xy[1])),
                base_link_yaw_rad=float(local_yaw),
                obstacle_body_ids=obstacle_body_ids,
                enable_ompl_path_check=False,
            )

        def _make_solution_record(
            ik_attempt: dict[str, object],
            sample_candidate: dict[str, object],
            sample_stats: dict[str, object],
            feasible: bool,
        ) -> dict[str, object] | None:
            return _make_sampled_ik_solution_record(
                ik_attempt=ik_attempt,
                sample_candidate=sample_candidate,
                sample_stats=sample_stats,
                reference_base_yaw_pb=0.0,
                feasible=bool(feasible),
            )

        sample_result = sample_logic.sample_base_pose_for_best_grasp(
            visualization_records,
            reset_ee_position_xyz=reset_ee_position_xyz,
            reset_ee_orientation_xyzw=reset_ee_orientation_xyzw,
            pose_from_local_base_fn=_pose_from_local_base,
            map_pose_is_clear_fn=_map_pose_is_clear,
            attempt_ik_fn=_attempt_ik,
            make_solution_record_fn=_make_solution_record,
            position_tolerance_m=float(cfg.get("position_tolerance_m", planning_config.position_tolerance_m)),
            orientation_tolerance_deg=float(cfg.get("orientation_tolerance_deg", 12.0)),
            current_amcl_yaw_rad=None if current_amcl_pose is None else float(current_amcl_pose.yaw_rad),
            min_backoff_m=0.0 if evaluate_current_pose_only else None,
            max_backoff_m=0.0 if evaluate_current_pose_only else None,
            max_amcl_yaw_delta_deg=0.0 if evaluate_current_pose_only else None,
        )

        print("[base_approach] simple sample target ranking:", flush=True)
        for record in sample_result.visualization_records:
            print(
                f"  order={int(record['target_sample_order']):02d} "
                f"grasp_rank={int(record['rank']):02d} "
                f"avg_rank={float(record['target_average_rank']):.2f} "
                f"dist_rank={int(record['reset_ee_distance_rank'])} "
                f"yaw_rank={int(record['reset_ee_yaw_rank'])} "
                f"reset_dist={float(record['reset_ee_distance_m']):.4f}m "
                f"yaw_err={float(record['reset_ee_yaw_error_deg']):.2f}deg",
                flush=True,
            )

        selected_record = sample_result.selected_record
        selected_solution = sample_result.selected_solution
        evaluated_records = [
            record
            for record in sample_result.visualization_records
            if "sample_attempted_count" in record
        ]
        evaluated_grasp_count = len(evaluated_records)
        total_yaw_rejected = sum(int(record.get("sample_yaw_rejected_count", 0)) for record in evaluated_records)
        total_map_blocked = sum(int(record.get("sample_map_blocked_count", 0)) for record in evaluated_records)
        total_ik_reachable = sum(int(record.get("sample_ik_reachable_count", 0)) for record in evaluated_records)
        if selected_record is None:
            print("[base_approach] simple sample: no grasp pose was available.", flush=True)
        elif selected_solution is None:
            closest_solution = selected_record.get("closest_solution")
            print(
                f"[base_approach] simple sample: no ROS-map-clear reachable IK solution after "
                f"trying {evaluated_grasp_count} grasp targets "
                f"(best_display_grasp_rank={int(selected_record['rank']):02d}, "
                f"map_feasible={sample_result.map_feasible_count}, "
                f"yaw_rejected={total_yaw_rejected}, "
                f"map_blocked={total_map_blocked}, "
                f"ik_reachable={total_ik_reachable}, "
                f"attempted={sample_result.attempted_count}).",
                flush=True,
            )
            if isinstance(closest_solution, dict):
                _print_closest_ik_solution_banner(
                    rank=int(selected_record["rank"]),
                    target_pb=np.asarray(selected_record["target_pb"], dtype=np.float64),
                    solution=closest_solution,
                )
        else:
            print(
                f"[base_approach] simple sample selected feasible solution for "
                f"grasp_rank={int(selected_record['rank']):02d} "
                f"after trying {evaluated_grasp_count} grasp targets: "
                f"backoff={float(selected_solution.get('backoff_distance_m', float('nan'))):.3f}m "
                f"map_clear={int(bool(selected_solution.get('map_clear', False)))} "
                f"yaw_ok={int(bool(selected_solution.get('amcl_yaw_within_limit', False)))} "
                f"yaw_clamped={int(bool(selected_solution.get('base_yaw_clamped', False)))} "
                f"ee_err={float(selected_solution['ee_position_error_m']):.4f}m "
                f"ori_err={selected_solution.get('ee_orientation_error_deg')}",
                flush=True,
            )
            _print_closest_ik_solution_banner(
                rank=int(selected_record["rank"]),
                target_pb=np.asarray(selected_record["target_pb"], dtype=np.float64),
                solution=selected_solution,
            )
            if sample_result.feasible:
                _print_selected_base_link_ros_map_banner(
                    rank=int(selected_record["rank"]),
                    amcl_pose=selected_solution.get("ros_map_amcl_pose"),
                    base_link_pose=selected_solution.get("ros_map_base_link_pose"),
                )

        return sample_result.visualization_records, selected_solution if sample_result.feasible else None
    finally:
        try:
            p_mod.disconnect(client_id)
        except Exception:
            pass


def _resolve_approach_local_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (APPROACH_AGENT_DIR / path).resolve()


def _resolve_optional_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def run_approach_agent(run_config: ApproachAgentRunConfig | None = None) -> dict[str, object]:
    run_config = run_config or ApproachAgentRunConfig()
    started_at = time.time()
    base_config_path = _resolve_approach_local_path(run_config.base_config_path)
    camera_config_path = _resolve_approach_local_path(run_config.camera_config_path)
    grasp_json_path = _resolve_optional_path(run_config.grasp_json_path)

    cfg = load_config(base_config_path)
    planning_config = load_planning_config(Path(cfg["planner_config_path"]))
    arm_config = _load_arm_config()
    _, p_mod, pybullet_data = _load_python_dependencies()
    map_free_space = _build_map_free_space(Path(cfg["map_yaml_path"]))
    (
        camera_in_base_link_rotation,
        camera_in_base_link_position,
        base_link_z_pb,
    ) = _get_reset_camera_transform_in_base_link_frame(
        Path(cfg["planner_config_path"]),
        planning_config,
    )
    target_object_points_camera = _load_target_object_pointcloud_camera(
        cfg.get("grasp_debug_npz_path")
    )

    live_scene = _capture_live_scene_voxels(
        camera_config_path.resolve(),
        camera_in_base_link_rotation,
        camera_in_base_link_position,
        base_link_z_pb,
        target_object_points_camera=target_object_points_camera,
    )
    voxels_pb = live_scene.voxel_centers_pb
    camera_to_pb_rotation = live_scene.camera_to_pb_rotation
    camera_position_pb = live_scene.camera_position_pb
    if run_config.grasp_result_payload is not None:
        visualization_records, grasp_candidates = sample_logic.load_grasp_visualization_records_from_payload(
            run_config.grasp_result_payload,
            camera_to_pb_rotation=camera_to_pb_rotation,
            camera_position_pb=camera_position_pb,
            source_label="ApproachAgentRunConfig.grasp_result_payload",
        )
        grasp_source = "payload"
    else:
        visualization_records, grasp_candidates, resolved_grasp_json_path = sample_logic.load_grasp_visualization_records(
            grasp_json_path,
            camera_to_pb_rotation=camera_to_pb_rotation,
            camera_position_pb=camera_position_pb,
        )
        grasp_source = str(resolved_grasp_json_path)
    total_grasp_count = len(grasp_candidates)
    if total_grasp_count > MAX_GRASP_POSES_TO_EVALUATE:
        visualization_records = visualization_records[:MAX_GRASP_POSES_TO_EVALUATE]
        grasp_candidates = grasp_candidates[:MAX_GRASP_POSES_TO_EVALUATE]
        print(
            f"[base_approach] using top {MAX_GRASP_POSES_TO_EVALUATE} grasp poses "
            f"out of {total_grasp_count}.",
            flush=True,
        )
    current_amcl_pose = _amcl_snapshot_to_ros_map_pose(live_scene.amcl_pose)
    if current_amcl_pose is None and not run_config.allow_missing_amcl:
        raise RuntimeError(
            "No /amcl_pose was received during Camera_Car capture. "
            "Use allow_missing_amcl only if you want a PB-local GUI replay."
        )
    print(
        f"[base_approach] grasp_source={grasp_source} | "
        f"grasps={len(grasp_candidates)}/{total_grasp_count} | live RGBD voxels={len(voxels_pb)} | "
        f"voxel_size={live_scene.voxel_size_m:.3f}m | "
        f"depth_points={live_scene.valid_depth_point_count} | "
        f"obstacle_points={live_scene.obstacle_depth_point_count} | "
        f"target_points={live_scene.target_object_point_count} | "
        f"target_removed={live_scene.target_excluded_depth_point_count} | "
        f"amcl={'yes' if current_amcl_pose is not None else 'no'}",
        flush=True,
    )
    print(
        "[base_approach] aligned ROS map context: "
        f"base_link_from_amcl_pb_xy=[{BASE_LINK_FROM_AMCL_PB_XY[0]:.4f}, "
        f"{BASE_LINK_FROM_AMCL_PB_XY[1]:.4f}]m "
        f"map_resolution={map_free_space.resolution_m:.3f}m "
        f"free_cells={len(map_free_space.free_cell_keys)} "
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

    visualization_records, selected_solution = _run_simple_sample_logic_for_gui(
        p_mod=p_mod,
        pybullet_data=pybullet_data,
        cfg=cfg,
        planning_config=planning_config,
        arm_config=arm_config,
        voxels_pb=voxels_pb,
        voxel_size_m=live_scene.voxel_size_m,
        visualization_records=visualization_records,
        map_free_space=map_free_space,
        current_amcl_pose=current_amcl_pose,
        evaluate_current_pose_only=run_config.evaluate_current_pose_only,
    )
    goal_pose_solution = selected_solution
    if goal_pose_solution is None:
        _, goal_pose_solution = _select_gui_display_solution(
            visualization_records,
            planning_config=planning_config,
        )
        if goal_pose_solution is not None:
            print(
                "[base_approach] no feasible solution; driving to best available "
                "closest/debug candidate.",
                flush=True,
            )

    nav_result: dict[str, object] = {
        "success": False,
        "skipped": True,
        "phase": "not_started",
    }
    phase = "sampled"
    message = "Base approach target sampled."
    if goal_pose_solution is None:
        phase = "no_target_pose"
        message = "No target pose solution available; rule navigation was not started."
        print(f"[base_approach] {message}", flush=True)
    elif not run_config.run_rule_navigation:
        phase = "rule_navigation_skipped"
        message = "Rule navigation disabled by caller."
        nav_result = {"success": False, "skipped": True, "phase": "disabled_by_caller"}
        print("[base_approach] rule navigation disabled by caller.", flush=True)
    else:
        nav_result = _run_rule_navigation_for_solution(
            goal_pose_solution,
            initial_pose=live_scene.captured_amcl_pose,
        )
        if bool(nav_result.get("success", False)):
            phase = "done"
            message = "Base approach reached the selected pose; arm motion is deferred."
            print(
                "[base_approach] base approach complete. Arm movement is intentionally "
                "deferred to Arm_Approach_Agent.",
                flush=True,
            )
        else:
            phase = "rule_navigation_failed"
            message = "Rule navigation did not reach the selected pose."

    map_png_path = run_config.map_png_path or (APPROACH_AGENT_DIR / "outputs" / "map_occupied.png")
    if run_config.write_map_png:
        _write_best_display_solution_ros_map_png(
            map_yaml_path=Path(cfg["map_yaml_path"]),
            planning_config=planning_config,
            visualization_records=visualization_records,
            output_path=map_png_path,
        )

    if run_config.show_gui:
        _visualize_feasible_ik_results_in_gui(
            p_mod=p_mod,
            pybullet_data=pybullet_data,
            planning_config=planning_config,
            arm_config=arm_config,
            voxels_pb=voxels_pb,
            voxel_size_m=live_scene.voxel_size_m,
            visualization_records=visualization_records,
        )
    else:
        print(
            "[base_approach] PyBullet GUI skipped. Use --gui or BASE_SAMPLER_SHOW_GUI=1 to enable replay.",
            flush=True,
        )

    success = bool(nav_result.get("success", False)) if run_config.run_rule_navigation else goal_pose_solution is not None
    status_code = "APPROACH_SUCCESS" if success else "APPROACH_FAIL"
    next_agent = "Arm_Approach_Agent" if success else None
    return {
        "success": success,
        "status_code": status_code,
        "phase": phase,
        "message": message,
        "next_agent": next_agent,
        "grasp_source": grasp_source,
        "grasp_count": len(grasp_candidates),
        "total_grasp_count": total_grasp_count,
        "live_scene": {
            "voxel_count": int(len(voxels_pb)),
            "voxel_size_m": float(live_scene.voxel_size_m),
            "valid_depth_point_count": int(live_scene.valid_depth_point_count),
            "obstacle_depth_point_count": int(live_scene.obstacle_depth_point_count),
            "target_object_point_count": int(live_scene.target_object_point_count),
            "target_excluded_depth_point_count": int(live_scene.target_excluded_depth_point_count),
            "amcl_available": current_amcl_pose is not None,
        },
        "selected_solution": goal_pose_solution or {},
        "selected_ik_feasible": bool((goal_pose_solution or {}).get("ik_feasible", False)),
        "nav_result": nav_result,
        "arm_motion_skipped": True,
        "map_png_path": str(map_png_path) if run_config.write_map_png else "",
        "visualization_record_count": len(visualization_records),
        "elapsed_sec": time.time() - started_at,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live sampler for Camera_Car RGBD voxels, grasp poses, /amcl_pose ROS-map "
            "alignment, and rule-based base approach."
        )
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/base_pose_sampling.yaml"),
        help="Base config YAML. Used for planner path, map path, and alignment diagnostics.",
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
        "--allow-missing-amcl",
        action="store_true",
        help="Continue even if no /amcl_pose was received during camera capture.",
    )
    gui_group = parser.add_mutually_exclusive_group()
    gui_group.add_argument(
        "--gui",
        dest="show_gui",
        action="store_true",
        default=None,
        help="Show PyBullet GUI replay.",
    )
    gui_group.add_argument(
        "--no-gui",
        dest="show_gui",
        action="store_false",
        help="Skip PyBullet GUI replay. This is the default.",
    )
    parser.add_argument(
        "--no-publish-goal-pose",
        dest="no_publish_goal_pose",
        action="store_true",
        help="Do not run rule-based car movement after sampling.",
    )
    parser.add_argument(
        "--no-rule-nav",
        dest="no_publish_goal_pose",
        action="store_true",
        help="Do not run rule-based car movement after sampling.",
    )
    return parser.parse_args()


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _should_show_gui(args: argparse.Namespace) -> bool:
    if args.show_gui is not None:
        return bool(args.show_gui)
    return _env_flag("BASE_SAMPLER_SHOW_GUI", False)


def main():
    args = _parse_args()
    show_gui = _should_show_gui(args)
    result = run_approach_agent(
        ApproachAgentRunConfig(
            base_config_path=args.base_config,
            camera_config_path=args.camera_config,
            grasp_json_path=args.grasp_json,
            allow_missing_amcl=args.allow_missing_amcl,
            run_rule_navigation=not args.no_publish_goal_pose,
            show_gui=show_gui,
        )
    )
    print(
        "[base_approach] approach agent result: "
        f"success={bool(result.get('success', False))} "
        f"phase={result.get('phase')} "
        f"message={result.get('message')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
