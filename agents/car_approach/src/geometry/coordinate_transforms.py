"""Coordinate and pose conversion helpers for the car approach pipeline."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def wrap_angle_rad(angle_rad: float) -> float:
    return float((float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi)


def deg_to_rad(value_deg: Any) -> float:
    return math.radians(float(value_deg))


def rad_to_deg(value_rad: Any) -> float:
    return math.degrees(float(value_rad))


def deg_sequence_to_rad(values_deg: Sequence[Any]) -> list[float]:
    return [deg_to_rad(value) for value in values_deg]


def rad_sequence_to_deg(values_rad: Sequence[Any]) -> list[float]:
    return [rad_to_deg(value) for value in values_rad]


def yaw_from_quat_z_w(qz: Any, qw: Any) -> float:
    return wrap_angle_rad(2.0 * math.atan2(float(qz), float(qw)))


def yaw_from_pose_like(pose: Any) -> float:
    if isinstance(pose, dict):
        if "yaw_rad" in pose:
            return wrap_angle_rad(float(pose["yaw_rad"]))
        if "yaw" in pose:
            return wrap_angle_rad(float(pose["yaw"]))
        if "yaw_deg" in pose:
            return wrap_angle_rad(math.radians(float(pose["yaw_deg"])))
        if all(key in pose for key in ("qx", "qy", "qz", "qw")):
            return yaw_from_quat_xyzw([pose["qx"], pose["qy"], pose["qz"], pose["qw"]])
        if "qz" in pose and "qw" in pose:
            return yaw_from_quat_z_w(pose["qz"], pose["qw"])
    if hasattr(pose, "yaw_rad"):
        return wrap_angle_rad(float(getattr(pose, "yaw_rad")))
    if hasattr(pose, "yaw"):
        return wrap_angle_rad(float(getattr(pose, "yaw")))
    raise ValueError("pose needs yaw_rad, yaw, yaw_deg, or qz/qw.")


def yaw_to_planar_quat(yaw_rad: float) -> dict[str, float]:
    yaw = wrap_angle_rad(yaw_rad)
    return {
        "qx": 0.0,
        "qy": 0.0,
        "qz": float(math.sin(yaw / 2.0)),
        "qw": float(math.cos(yaw / 2.0)),
    }


def yaw_from_quat_xyzw(value: Any) -> float:
    x, y, z, w = np.asarray(value, dtype=np.float64).reshape(4)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return wrap_angle_rad(math.atan2(float(siny_cosp), float(cosy_cosp)))


def yaw_rotation_matrix(yaw_rad: float) -> np.ndarray:
    yaw = float(yaw_rad)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return np.asarray(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def pose2d_from_any(pose: Any) -> dict[str, float]:
    if isinstance(pose, dict):
        if "position_xyz" in pose and "orientation_xyzw" in pose:
            position = np.asarray(pose["position_xyz"], dtype=np.float64).reshape(3)
            return {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
                "yaw_rad": yaw_from_quat_xyzw(pose["orientation_xyzw"]),
            }
        return {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "z": float(pose.get("z", 0.0)),
            "yaw_rad": yaw_from_pose_like(pose),
        }
    return {
        "x": float(getattr(pose, "x")),
        "y": float(getattr(pose, "y")),
        "z": float(getattr(pose, "z", 0.0)),
        "yaw_rad": yaw_from_pose_like(pose),
    }


def pose2d_dict(x: Any, y: Any, yaw_rad: Any, *, z: Any = 0.0) -> dict[str, float]:
    yaw = wrap_angle_rad(float(yaw_rad))
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "yaw_rad": yaw,
        "yaw_deg": rad_to_deg(yaw),
    }


def matrix4x4_from_config(value: Any) -> np.ndarray:
    if value is None:
        return np.eye(4, dtype=np.float64)
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"transform matrix must be 4x4, got {matrix.shape}.")
    return matrix


def transform_points_xyz(points_xyz: Any, transform_matrix_4x4: Any) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    transform = matrix4x4_from_config(transform_matrix_4x4)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return (points.astype(np.float64) @ rotation.T + translation.reshape(1, 3)).astype(np.float32)


def transform_position_rotation(
    position_xyz: Any,
    rotation_matrix: Any,
    transform_matrix_4x4: Any,
) -> tuple[np.ndarray, np.ndarray]:
    transform = matrix4x4_from_config(transform_matrix_4x4)
    position = np.asarray(position_xyz, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    transformed_position = transform[:3, :3] @ position + transform[:3, 3]
    transformed_rotation = transform[:3, :3] @ rotation
    return transformed_position.astype(np.float64), transformed_rotation.astype(np.float64)


def transform_planar_pose_xy_yaw(
    x: Any,
    y: Any,
    yaw_rad: Any,
    transform_matrix_4x4: Any,
    *,
    z: Any = 0.0,
) -> dict[str, float]:
    transform = matrix4x4_from_config(transform_matrix_4x4)
    position = transform[:3, :3] @ np.asarray([float(x), float(y), float(z)], dtype=np.float64) + transform[:3, 3]
    direction = transform[:3, :3] @ np.asarray([math.cos(float(yaw_rad)), math.sin(float(yaw_rad)), 0.0], dtype=np.float64)
    return pose2d_dict(position[0], position[1], math.atan2(float(direction[1]), float(direction[0])), z=position[2])


def rotate_xy(value_xy: Sequence[Any], yaw_rad: Any) -> np.ndarray:
    values = np.asarray(value_xy, dtype=np.float64).reshape(2)
    return yaw_rotation_matrix(float(yaw_rad))[:2, :2] @ values


def pybullet_xy_delta_to_ros_map_delta(
    pb_xy_delta: Sequence[Any],
    *,
    reference_ros_yaw_rad: Any = 0.0,
) -> np.ndarray:
    """Convert a PB-scene XY delta into a global ROS-map XY delta."""
    local_ros_delta = pybullet_xy_delta_to_ros_xy_delta(pb_xy_delta)
    return rotate_xy(local_ros_delta, reference_ros_yaw_rad)


def ros_map_delta_to_pybullet_xy_delta(
    ros_map_delta: Sequence[Any],
    *,
    reference_ros_yaw_rad: Any = 0.0,
) -> np.ndarray:
    """Convert a global ROS-map XY delta back into the PB-scene XY frame."""
    local_ros_delta = rotate_xy(ros_map_delta, -float(reference_ros_yaw_rad))
    return ros_xy_delta_to_pybullet_xy_delta(local_ros_delta)


def arm_base_local_pb_offset_to_ros_map_delta(
    local_pb_xy: Sequence[Any],
    pybullet_yaw_rad: Any,
    *,
    reference_ros_yaw_rad: Any = 0.0,
) -> np.ndarray:
    """Convert an arm-base-local PyBullet XY offset into a global ROS-map XY delta."""
    offset_pb = np.asarray(local_pb_xy, dtype=np.float64).reshape(2)
    rotated_offset_pb = yaw_rotation_matrix(float(pybullet_yaw_rad))[:2, :2] @ offset_pb
    return pybullet_xy_delta_to_ros_map_delta(
        rotated_offset_pb,
        reference_ros_yaw_rad=reference_ros_yaw_rad,
    )


def _car_center_offset_ros_for_pybullet_yaw(
    pybullet_yaw_rad: Any,
    car_center_from_arm_base_pb_xy: Sequence[Any],
    *,
    reference_ros_yaw_rad: Any = 0.0,
) -> np.ndarray:
    return arm_base_local_pb_offset_to_ros_map_delta(
        car_center_from_arm_base_pb_xy,
        pybullet_yaw_rad,
        reference_ros_yaw_rad=reference_ros_yaw_rad,
    )


def local_pybullet_base_to_ros_map_poses(
    local_pb_xy: Sequence[Any],
    local_pb_yaw_rad: Any,
    *,
    reference_amcl_pose: Any | None,
    arm_base_link_pb_xyz: Sequence[Any] = (0.0, 0.0, 0.0),
    car_center_from_arm_base_pb_xy: Sequence[Any] = (0.0, -0.1285),
    reference_pb_yaw_rad: Any = 0.0,
) -> tuple[dict[str, float] | None, dict[str, float]]:
    arm_pb_xy = np.asarray(local_pb_xy, dtype=np.float64).reshape(2)
    reference_arm_pb_xy = np.asarray(arm_base_link_pb_xyz, dtype=np.float64).reshape(3)[:2]

    if reference_amcl_pose is None:
        reference_ros_yaw = 0.0
        reference_arm_ros_xy = np.asarray([0.0, 0.0], dtype=np.float64)
    else:
        reference = pose2d_from_any(reference_amcl_pose)
        reference_ros_yaw = float(reference["yaw_rad"])
        reference_amcl_ros_xy = np.asarray([reference["x"], reference["y"]], dtype=np.float64)
        reference_car_center_offset_ros = _car_center_offset_ros_for_pybullet_yaw(
            reference_pb_yaw_rad,
            car_center_from_arm_base_pb_xy,
            reference_ros_yaw_rad=reference_ros_yaw,
        )
        reference_arm_ros_xy = reference_amcl_ros_xy - reference_car_center_offset_ros

    delta_ros = pybullet_xy_delta_to_ros_map_delta(
        arm_pb_xy - reference_arm_pb_xy,
        reference_ros_yaw_rad=reference_ros_yaw,
    )
    arm_ros_xy = reference_arm_ros_xy + delta_ros
    ros_yaw = wrap_angle_rad(float(reference_ros_yaw) + wrap_angle_rad(float(local_pb_yaw_rad) - float(reference_pb_yaw_rad)))
    candidate_car_center_offset_ros = _car_center_offset_ros_for_pybullet_yaw(
        local_pb_yaw_rad,
        car_center_from_arm_base_pb_xy,
        reference_ros_yaw_rad=reference_ros_yaw,
    )

    car_center_ros_xy = arm_ros_xy + candidate_car_center_offset_ros
    arm_pose = pose2d_dict(arm_ros_xy[0], arm_ros_xy[1], ros_yaw)
    car_center_pose = pose2d_dict(car_center_ros_xy[0], car_center_ros_xy[1], ros_yaw)
    return car_center_pose, arm_pose


def ros_map_amcl_pose_to_local_pybullet_base_pose(
    amcl_pose: Any,
    *,
    reference_amcl_pose: Any | None,
    arm_base_link_pb_xyz: Sequence[Any] = (0.0, 0.0, 0.0),
    car_center_from_arm_base_pb_xy: Sequence[Any] = (0.0, -0.1285),
    reference_pb_yaw_rad: Any = 0.0,
) -> tuple[tuple[float, float], float]:
    candidate = pose2d_from_any(amcl_pose)
    candidate_amcl_ros_xy = np.asarray([candidate["x"], candidate["y"]], dtype=np.float64)

    if reference_amcl_pose is None:
        reference_ros_yaw = 0.0
        reference_arm_ros_xy = np.asarray([0.0, 0.0], dtype=np.float64)
    else:
        reference = pose2d_from_any(reference_amcl_pose)
        reference_ros_yaw = float(reference["yaw_rad"])
        reference_amcl_ros_xy = np.asarray([reference["x"], reference["y"]], dtype=np.float64)
        reference_car_center_offset_ros = _car_center_offset_ros_for_pybullet_yaw(
            reference_pb_yaw_rad,
            car_center_from_arm_base_pb_xy,
            reference_ros_yaw_rad=reference_ros_yaw,
        )
        reference_arm_ros_xy = reference_amcl_ros_xy - reference_car_center_offset_ros

    candidate_pb_yaw = wrap_angle_rad(
        float(candidate["yaw_rad"]) - float(reference_ros_yaw) + float(reference_pb_yaw_rad)
    )
    candidate_car_center_offset_ros = _car_center_offset_ros_for_pybullet_yaw(
        candidate_pb_yaw,
        car_center_from_arm_base_pb_xy,
        reference_ros_yaw_rad=reference_ros_yaw,
    )
    candidate_arm_ros_xy = candidate_amcl_ros_xy - candidate_car_center_offset_ros
    delta_ros = candidate_arm_ros_xy - reference_arm_ros_xy

    reference_arm_pb_xy = np.asarray(arm_base_link_pb_xyz, dtype=np.float64).reshape(3)[:2]
    arm_pb_xy = reference_arm_pb_xy + ros_map_delta_to_pybullet_xy_delta(
        delta_ros,
        reference_ros_yaw_rad=reference_ros_yaw,
    )
    return (float(arm_pb_xy[0]), float(arm_pb_xy[1])), float(candidate_pb_yaw)

def planar_pose_error(planned_pose: Any, actual_pose: Any) -> dict[str, float]:
    planned = pose2d_from_any(planned_pose)
    actual = pose2d_from_any(actual_pose)
    dx = float(actual["x"] - planned["x"])
    dy = float(actual["y"] - planned["y"])
    yaw_error = wrap_angle_rad(float(actual["yaw_rad"] - planned["yaw_rad"]))
    return {
        "dx_m": dx,
        "dy_m": dy,
        "xy_norm_m": float(math.hypot(dx, dy)),
        "yaw_error_rad": yaw_error,
        "yaw_error_deg": rad_to_deg(yaw_error),
    }


def footprint_world_xy(
    footprint_points_xy: Any,
    *,
    pose: Any,
) -> np.ndarray:
    pose2d = pose2d_from_any(pose)
    points = np.asarray(footprint_points_xy, dtype=np.float64).reshape(-1, 2)
    rotation = yaw_rotation_matrix(pose2d["yaw_rad"])[:2, :2]
    origin = np.asarray([pose2d["x"], pose2d["y"]], dtype=np.float64).reshape(1, 2)
    return origin + points @ rotation.T


def camera_xyz_to_pybullet_delta(camera_xyz: Any) -> np.ndarray:
    cam = np.asarray(camera_xyz, dtype=np.float64).reshape(3)
    return np.asarray([cam[0], -cam[2], cam[1]], dtype=np.float64)


def camera_points_to_pybullet_world(points_camera_xyz: Any, camera_position_pb_xyz: Any) -> np.ndarray:
    points = np.asarray(points_camera_xyz, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    origin = np.asarray(camera_position_pb_xyz, dtype=np.float64).reshape(3)
    transformed = np.column_stack([points[:, 0], -points[:, 2], points[:, 1]]).astype(np.float64)
    return (transformed + origin.reshape(1, 3)).astype(np.float32)


def camera_grasp_pose_to_pybullet(
    position_camera_xyz: Any,
    rotation_camera_matrix: Any,
    *,
    camera_position_pb_xyz: Any,
) -> tuple[np.ndarray, np.ndarray]:
    position_pb = camera_xyz_to_pybullet_delta(position_camera_xyz) + np.asarray(camera_position_pb_xyz, dtype=np.float64).reshape(3)
    camera_to_pb_axes = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    rotation_camera = np.asarray(rotation_camera_matrix, dtype=np.float64).reshape(3, 3)
    rotation_pb = camera_to_pb_axes @ rotation_camera
    # IK target axes are remapped in-place after the grasp pose is placed in PB:
    # [new +X, new +Y, new +Z] = [old grasp +Z, old grasp -X, old grasp -Y].
    grasp_to_ik_axes = np.asarray(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rotation_pb = rotation_pb @ grasp_to_ik_axes
    return position_pb.astype(np.float64), rotation_pb.astype(np.float64)


def pybullet_xy_delta_to_ros_xy_delta(pb_xy_delta: Any) -> np.ndarray:
    pb = np.asarray(pb_xy_delta, dtype=np.float64).reshape(2)
    return np.asarray([-pb[1], pb[0]], dtype=np.float64)


def ros_xy_delta_to_pybullet_xy_delta(ros_xy_delta: Any) -> np.ndarray:
    ros = np.asarray(ros_xy_delta, dtype=np.float64).reshape(2)
    return np.asarray([ros[1], -ros[0]], dtype=np.float64)



def xy_distance(current_x: float, current_y: float, target_x: float, target_y: float) -> float:
    return math.hypot(float(target_x) - float(current_x), float(target_y) - float(current_y))


def target_heading_error(current_x: float, current_y: float, current_yaw_rad: float, target_x: float, target_y: float) -> float:
    target_heading = math.atan2(float(target_y) - float(current_y), float(target_x) - float(current_x))
    return wrap_angle_rad(target_heading - float(current_yaw_rad))


def target_reverse_heading_error(current_x: float, current_y: float, current_yaw_rad: float, target_x: float, target_y: float) -> float:
    target_heading = math.atan2(float(target_y) - float(current_y), float(target_x) - float(current_x))
    return wrap_angle_rad(target_heading + math.pi - float(current_yaw_rad))


def unit_xy(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    return arr / norm


def grasp_backoff_base_pose(
    position_xyz: Sequence[Any],
    rotation_matrix: Sequence[Any],
    *,
    backoff_distance_m: float,
    yaw_offset_rad: float,
) -> tuple[np.ndarray, float, float]:
    position = np.asarray(position_xyz, dtype=np.float64).reshape(3)
    rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    target_xy = position[:2]
    approach_xy = unit_xy(rotation[:2, 0])
    # PyBullet local -Y is base_footprint +X/front for this robot.
    desired_yaw = wrap_angle_rad(math.atan2(float(approach_xy[0]), -float(approach_xy[1])))
    base_xy = target_xy - approach_xy * float(backoff_distance_m)
    yaw_rad = wrap_angle_rad(desired_yaw + float(yaw_offset_rad))
    return base_xy, desired_yaw, yaw_rad


def matrix4x4_position_rotation(value: Any) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"matrix_4x4 must be 4x4, got {matrix.shape}.")
    return matrix[:3, 3].astype(np.float64), matrix[:3, :3].astype(np.float64)


def quat_xyzw_to_matrix(value: Any) -> np.ndarray:
    x, y, z, w = np.asarray(value, dtype=np.float64).reshape(4)
    norm = math.sqrt(float(x * x + y * y + z * z + w * w))
    if norm <= 1e-12:
        raise ValueError("Quaternion norm must be non-zero.")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_xyzw(matrix: Any) -> list[float]:
    m = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([x, y, z, w], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    return quat.astype(float).tolist()


def quat_angle_error_deg(quat_a: Sequence[Any], quat_b: Sequence[Any]) -> float:
    qa = np.asarray(quat_a, dtype=np.float64).reshape(4)
    qb = np.asarray(quat_b, dtype=np.float64).reshape(4)
    qa /= np.linalg.norm(qa)
    qb /= np.linalg.norm(qb)
    dot = abs(float(np.dot(qa, qb)))
    dot = max(-1.0, min(1.0, dot))
    return float(math.degrees(2.0 * math.acos(dot)))


def world_to_base_local(
    target_world_xyz: Sequence[Any],
    *,
    base_xyz: Sequence[Any],
    base_yaw_rad: float,
    arm_base_height_m: float,
) -> list[float]:
    target = np.asarray(target_world_xyz, dtype=np.float64).reshape(3)
    base = np.asarray(base_xyz, dtype=np.float64).reshape(3)
    delta = target - base
    cos_yaw = math.cos(-float(base_yaw_rad))
    sin_yaw = math.sin(-float(base_yaw_rad))
    return [
        float(cos_yaw * delta[0] - sin_yaw * delta[1]),
        float(sin_yaw * delta[0] + cos_yaw * delta[1]),
        float(arm_base_height_m + delta[2]),
    ]


def joint_angle_error_rad(current_rad: float, target_rad: float) -> float:
    error = abs(float(current_rad) - float(target_rad))
    wrapped = abs((error + math.pi) % (2.0 * math.pi) - math.pi)
    return min(error, wrapped)


def backproject_depth_to_points(
    depth_metric_m: np.ndarray,
    intrinsic_k: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
    pixel_stride: int = 1,
    mirror_x: bool = False,
    exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    depth = np.asarray(depth_metric_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth_metric_m must be a 2D array.")

    stride = max(int(pixel_stride), 1)
    height, width = depth.shape
    ys, xs = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[::stride, ::stride]
    valid_mask = np.isfinite(z) & (z >= float(min_depth_m)) & (z <= float(max_depth_m))
    if exclude_mask is not None:
        mask = np.asarray(exclude_mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError(f"exclude_mask shape {mask.shape} must match depth shape {depth.shape}.")
        valid_mask &= ~mask[::stride, ::stride]
    if not np.any(valid_mask):
        return np.empty((0, 3), dtype=np.float32)

    k = np.asarray(intrinsic_k, dtype=np.float32).reshape(3, 3)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    if abs(fx) <= 1e-9 or abs(fy) <= 1e-9:
        raise ValueError("Camera intrinsics fx/fy must be non-zero.")

    x = (xs.astype(np.float32) - cx) * z / fx
    if mirror_x:
        x *= -1.0
    y = (ys.astype(np.float32) - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    return points[valid_mask].astype(np.float32)


def voxelize_points(
    points_xyz: np.ndarray,
    *,
    voxel_size_m: float,
    max_voxels: int | None = None,
    selection_origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)

    voxel_size = float(voxel_size_m)
    if voxel_size <= 0.0:
        raise ValueError(f"voxel_size_m must be positive, got {voxel_size}.")

    finite_points = points[np.all(np.isfinite(points), axis=1)]
    if len(finite_points) == 0:
        return np.empty((0, 3), dtype=np.float32)

    voxel_indices = np.floor(finite_points / voxel_size).astype(np.int32)
    unique_indices = np.unique(voxel_indices, axis=0)
    voxel_centers = (unique_indices.astype(np.float32) + 0.5) * voxel_size

    if max_voxels is not None and int(max_voxels) > 0 and len(voxel_centers) > int(max_voxels):
        origin = np.asarray(selection_origin_xyz, dtype=np.float32).reshape(1, 3)
        keep = np.argsort(np.linalg.norm(voxel_centers - origin, axis=1), kind="stable")[: int(max_voxels)]
        voxel_centers = voxel_centers[keep]

    order = np.lexsort((voxel_centers[:, 2], voxel_centers[:, 1], voxel_centers[:, 0]))
    return voxel_centers[order].astype(np.float32)
