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


LIVE_VOXEL_MIN_DEPTH_M = 0.19
LIVE_VOXEL_MAX_DEPTH_M = 1.5
LIVE_VOXEL_SCENE_POINT_STRIDE = 4
LIVE_VOXEL_SCENE_DOWNSAMPLE_M = 0.01
BASE_LINK_YAW_MAX_DELTA_FROM_CURRENT_DEG = 30.0


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


def _yaw_is_within_limit(
    yaw_rad: float,
    reference_yaw_rad: float | None,
    max_delta_rad: float,
) -> bool:
    if reference_yaw_rad is None:
        return True
    return _abs_angle_delta_rad(yaw_rad, reference_yaw_rad) <= float(max_delta_rad)


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


def _backproject_depth_to_camera_points_like_grasp_npz(
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
    points_camera = _backproject_depth_to_camera_points_like_grasp_npz(
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
    current_base_link_pose: RosMapPose2D | None,
) -> RosMapPose2D:
    if current_base_link_pose is None:
        return _pb_world_pose_to_ros_map_pose(local_pb_xy, local_pb_yaw_rad)

    current_pb_pose = _base_link_pose_to_pb_world_pose(
        current_base_link_pose,
        base_link_z_pb=0.0,
    )
    assert current_pb_pose is not None
    current_pb_xyz, current_pb_yaw = current_pb_pose
    current_rot = _yaw_rotation_matrix(current_pb_yaw)
    local_delta = np.asarray([float(local_pb_xy[0]), float(local_pb_xy[1]), 0.0], dtype=np.float64)
    candidate_pb_xy = current_pb_xyz[:2] + (current_rot @ local_delta)[:2]
    candidate_pb_yaw = _wrap_angle_rad(float(current_pb_yaw) + float(local_pb_yaw_rad))
    return _pb_world_pose_to_ros_map_pose(
        (float(candidate_pb_xy[0]), float(candidate_pb_xy[1])),
        candidate_pb_yaw,
    )


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
    current_base_link_pose: RosMapPose2D | None = None,
    num_samples: int = 500,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(int(rng_seed))
    valid_candidates: list[dict[str, object]] = []
    target_pos = target_pb.tolist()
    reference_base_yaw_pb: float | None = None
    max_base_yaw_delta_rad = math.radians(BASE_LINK_YAW_MAX_DELTA_FROM_CURRENT_DEG)
    if current_base_link_pose is not None:
        reference_base_yaw_pb = 0.0

    for sample_index in range(int(num_samples)):
        distance = float(rng.uniform(0.20, 0.48))
        angle_offset = float(rng.uniform(math.radians(-10), math.radians(10)))

        # target_rot_pb is the dynamic gripper/EE frame in the local PB scene, not base_link.
        # Its local +X is the gripper forward direction; sample the arm base behind it.
        v_x_xy = _unit_xy_or_default(target_rot_pb[:, 0])
        ray_angle = math.atan2(float(v_x_xy[1]), float(v_x_xy[0]))
        theta = ray_angle + angle_offset

        pb_bx = float(target_pos[0] - distance * math.cos(theta))
        pb_by = float(target_pos[1] - distance * math.sin(theta))
        pb_yaw = float(math.atan2(target_pos[1] - pb_by, target_pos[0] - pb_bx))

        final_pb_bx = pb_bx
        final_pb_by = pb_by
        final_pb_yaw = pb_yaw
        if not _yaw_is_within_limit(final_pb_yaw, reference_base_yaw_pb, max_base_yaw_delta_rad):
            continue

        best_attempt: dict[str, object] | None = None
        current_pb_bx = final_pb_bx
        current_pb_by = final_pb_by
        current_pb_yaw = final_pb_yaw

        # After the first IK solve, nudge the base in the XY direction of the EE error.
        for _ in range(3):
            if not _yaw_is_within_limit(current_pb_yaw, reference_base_yaw_pb, max_base_yaw_delta_rad):
                break
            ik_attempt = _attempt_ik_at_base_pose(
                p_mod=p_mod,
                robot_id=robot_id,
                controllable_joint_ids=controllable_joint_ids,
                planning_config=planning_config,
                target_pb=target_pb,
                target_quat_pb=target_quat_pb,
                base_link_xy=(current_pb_bx, current_pb_by),
                base_link_yaw_rad=current_pb_yaw,
                obstacle_body_ids=obstacle_body_ids,
            )
            is_better_attempt = (
                best_attempt is None
                or float(ik_attempt["ee_position_error_m"]) < float(best_attempt["ee_position_error_m"])
            )
            if is_better_attempt:
                best_attempt = dict(ik_attempt)

            error_xyz = np.asarray(ik_attempt["ik_error_xyz"], dtype=np.float64)
            error_xy = error_xyz[:2]
            if np.linalg.norm(error_xy) < 1e-4:
                break

            current_pb_bx += float(error_xy[0]) * 0.75
            current_pb_by += float(error_xy[1]) * 0.75
            current_pb_yaw = float(math.atan2(target_pos[1] - current_pb_by, target_pos[0] - current_pb_bx))

        if best_attempt is None:
            continue

        dist_error = float(best_attempt["ee_position_error_m"])
        collision_free = bool(best_attempt["collision_free"])
        joint_solution_rad = best_attempt["ik_joint_solution_rad"]

        orientation_error_deg = best_attempt["ee_orientation_error_deg"]
        orientation_ok = (
            orientation_error_deg is not None
            and float(orientation_error_deg) <= float(orientation_tolerance_deg)
        )

        if (
            dist_error <= float(position_tolerance_m)
            and orientation_ok
            and collision_free
            and joint_solution_rad is not None
        ):
            joint_solution_deg = np.degrees(joint_solution_rad)
            best_base_xyz = [float(v) for v in best_attempt["pb_base_link_xyz"]]
            best_base_yaw = float(best_attempt["pb_base_link_yaw_rad"])
            best_base_yaw_delta_from_reference = (
                None
                if reference_base_yaw_pb is None
                else _abs_angle_delta_rad(best_base_yaw, reference_base_yaw_pb)
            )
            best_error_xyz = np.asarray(best_attempt["ik_error_xyz"], dtype=np.float64)
            best_ros_map_pose = _local_pb_base_pose_to_ros_map_pose(
                (best_base_xyz[0], best_base_xyz[1]),
                best_base_yaw,
                current_base_link_pose,
            )
            solution_record: dict[str, object] = {
                "sample_index": sample_index,
                "pb_base_link_xyz": best_base_xyz,
                "pb_base_link_yaw_rad": best_base_yaw,
                "pb_base_link_yaw_deg": float(math.degrees(best_base_yaw)),
                "pb_base_link_reference_yaw_rad": (
                    None if reference_base_yaw_pb is None else float(reference_base_yaw_pb)
                ),
                "pb_base_link_reference_yaw_deg": (
                    None if reference_base_yaw_pb is None else float(math.degrees(reference_base_yaw_pb))
                ),
                "pb_base_link_yaw_delta_from_reference_rad": (
                    None
                    if best_base_yaw_delta_from_reference is None
                    else float(best_base_yaw_delta_from_reference)
                ),
                "pb_base_link_yaw_delta_from_reference_deg": (
                    None
                    if best_base_yaw_delta_from_reference is None
                    else float(math.degrees(best_base_yaw_delta_from_reference))
                ),
                "pb_base_link_yaw_max_delta_from_reference_deg": float(
                    BASE_LINK_YAW_MAX_DELTA_FROM_CURRENT_DEG
                ),
                "ros_map_base_link_pose": _ros_map_pose_to_dict(best_ros_map_pose),
                "ee_position_error_m": dist_error,
                "ee_orientation_error_deg": float(orientation_error_deg),
                "approach_axis_offset_m": (
                    None
                    if best_attempt["approach_axis_offset_m"] is None
                    else float(best_attempt["approach_axis_offset_m"])
                ),
                "lateral_offset_m": (
                    None
                    if best_attempt["lateral_offset_m"] is None
                    else float(best_attempt["lateral_offset_m"])
                ),
                "ik_error_xyz_m": best_error_xyz.astype(float).tolist(),
                "ik_joint_solution_rad": joint_solution_rad.astype(float).tolist(),
                "ik_joint_solution_deg": joint_solution_deg.astype(float).tolist(),
                "joint_limit_margin_min_ratio": float(best_attempt["joint_limit_margin_min_ratio"]),
                "joint_limit_margin_mean_ratio": float(best_attempt["joint_limit_margin_mean_ratio"]),
                "joint_reset_delta_norm_l2": float(best_attempt["joint_reset_delta_norm_l2"]),
                "refinement_applied": True,
            }
            valid_candidates.append(solution_record)

    valid_candidates.sort(key=_feasible_ik_solution_sort_key)
    if valid_candidates:
        valid_candidates[0]["selected_as_best"] = True
    return valid_candidates


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

            if feasible_solutions:
                best_solution = sorted(feasible_solutions, key=_feasible_ik_solution_sort_key)[0]
                best_solution_is_better = (
                    best_view_solution is None
                    or _feasible_ik_solution_sort_key(best_solution)
                    < _feasible_ik_solution_sort_key(best_view_solution)
                )
                if best_solution_is_better:
                    best_view_record = record
                    best_view_solution = best_solution

        if best_view_record is not None and best_view_solution is not None:
            rank = int(best_view_record["rank"])
            target_pb = np.asarray(best_view_record["target_pb"], dtype=float)
            base_xyz = [float(v) for v in best_view_solution["pb_base_link_xyz"]]
            base_yaw_rad = float(best_view_solution["pb_base_link_yaw_rad"])
            joint_solution_rad = [float(v) for v in best_view_solution["ik_joint_solution_rad"]]
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
            p_mod.addUserDebugText(
                (
                    f"rank={rank}  feasible={len(best_view_record['feasible_solutions'])}  "
                    f"ee_err={float(best_view_solution['ee_position_error_m']):.4f}m  "
                    f"ori_err={float(best_view_solution['ee_orientation_error_deg']):.2f}deg"
                ),
                textPosition=[target_pb[0], target_pb[1], target_pb[2] + 0.24],
                textColorRGB=[1.0, 1.0, 1.0],
                textSize=1.1,
            )
            print(
                f"[test_base_sampler] GUI open. Showing best feasible grasp rank={rank:02d} "
                f"at base={base_xyz}, pb_gui_yaw={math.degrees(base_yaw_rad):.2f}deg.",
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
                "[test_base_sampler] GUI open. No feasible IK solution found; "
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
        default=int(os.getenv("BASE_SAMPLER_NUM_SAMPLES", "500")),
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
    current_base_link_pose = _amcl_snapshot_to_ros_map_pose(live_scene.amcl_pose)
    if current_base_link_pose is None and not args.allow_missing_amcl:
        raise RuntimeError(
            "No /amcl_pose was received during Camera_Car capture. "
            "Use --allow-missing-amcl only if you want a PB-local GUI replay."
        )
    print(
        f"[test_base_sampler] grasp_json={grasp_result_json_path} | "
        f"grasps={len(grasp_candidates)} | live RGBD voxels={len(voxels_pb)} | "
        f"depth_points={live_scene.valid_depth_point_count} | "
        f"amcl={'yes' if current_base_link_pose is not None else 'no'}",
        flush=True,
    )
    _print_ros_map_pose("current /amcl_pose base_link", current_base_link_pose)
    current_base_link_pb_pose = _base_link_pose_to_pb_world_pose(
        current_base_link_pose,
        base_link_z_pb=base_link_z_pb,
    )
    if current_base_link_pb_pose is None:
        _print_pb_pose("current /amcl_pose converted to PB map frame", [])
    else:
        current_base_link_pb_xyz, current_base_link_pb_yaw = current_base_link_pb_pose
        _print_pb_pose(
            "current /amcl_pose converted to PB map frame",
            current_base_link_pb_xyz,
            current_base_link_pb_yaw,
        )
    _print_camera_to_pb_axis_mapping()
    _print_pb_pose("reset depth camera local PB pose from default arm posture", camera_position_pb)

    client_id = p_mod.connect(p_mod.DIRECT)
    visualization_records: list[dict[str, object]] = []
    first_feasible_result: dict[str, object] | None = None
    try:
        p_mod.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_mod.resetSimulation()
        p_mod.setGravity(0.0, 0.0, -9.8)
        p_mod.loadURDF("plane.urdf")

        voxel_size = 0.05
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
            feasible_solutions = _sample_feasible_ik_solutions(
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
                current_base_link_pose=current_base_link_pose,
                num_samples=int(args.num_samples),
            )

            direct_orientation_error_deg = prepared_target.direct_orientation_error_deg
            direct_orientation_error_text = (
                float("nan")
                if direct_orientation_error_deg is None
                else float(direct_orientation_error_deg)
            )
            print(
                f"[grasp rank={grasp_candidate.rank:02d}] "
                f"direct_ik={prepared_target.direct_ik_error_m:.4f}m "
                f"direct_ori={direct_orientation_error_text:.2f}deg "
                f"conf={grasp_candidate.grasp_confidence:.4f} "
                f"dist_to_midpoint={grasp_candidate.grasp_distance_to_gripper_midpoint_m:.4f}m "
                f"-> feasible_ik={len(feasible_solutions)}",
                flush=True,
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
                }
            )

            if feasible_solutions:
                best_feasible_solution = feasible_solutions[0]
                _print_pb_pose(
                    f"target local PB pose rank={grasp_candidate.rank:02d}",
                    target_pb,
                )
                _print_pb_pose(
                    f"selected local PB base_link rank={grasp_candidate.rank:02d}",
                    best_feasible_solution["pb_base_link_xyz"],
                    float(best_feasible_solution["pb_base_link_yaw_rad"]),
                )
                _print_ros_map_pose(
                    f"selected feasible ROS map base_link pose rank={grasp_candidate.rank:02d}",
                    best_feasible_solution.get("ros_map_base_link_pose"),
                )
                first_feasible_result = {
                    "rank": grasp_candidate.rank,
                    "feasible_ik_count": len(feasible_solutions),
                    "direct_ik_error_m": float(prepared_target.direct_ik_error_m),
                }
                print(
                    f"[test_base_sampler] First feasible IK found at grasp rank={grasp_candidate.rank:02d}; "
                    "launching GUI next.",
                    flush=True,
                )
                break
    finally:
        p_mod.disconnect(client_id)

    if first_feasible_result is not None:
        print(
            f"[test_base_sampler] Visualizing first feasible grasp rank={first_feasible_result['rank']:02d} "
            f"(feasible_ik={first_feasible_result['feasible_ik_count']}, "
            f"direct_ik={first_feasible_result['direct_ik_error_m']:.4f}m)",
            flush=True,
        )
    else:
        print("[test_base_sampler] No feasible IK found; GUI will show scene and targets.", flush=True)

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
