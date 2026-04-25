import math
import numpy as np

# We assume RosMapPose2D is passed as an object with x, y, yaw_rad attributes
# or as a dict. We handle both implicitly or use duck-typing.

def _format_ros_map_pose_lines(pose) -> str:
    if pose is None:
        return " ros_map    : unavailable\n"
    if hasattr(pose, 'x'):
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

def print_ros_map_pose(label: str, pose) -> None:
    if pose is None:
        print(f"[test_base_sampler] {label}: unavailable", flush=True)
        return
    if hasattr(pose, 'x'):
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

def print_selected_base_link_ros_map_banner(*, rank: int, base_link_pose, amcl_pose) -> None:
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

def print_closest_ik_solution_banner(*, rank: int, target_pb: np.ndarray, solution: dict | None) -> None:
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

    print(
        "\n"
        "############################################################\n"
        " CLOSEST SAMPLED IK ATTEMPT\n"
        "------------------------------------------------------------\n"
        f" grasp_rank : {int(rank):02d}\n"
        f" ik_feasible: {int(bool(solution.get('ik_feasible', False)))}\n"
        f" source     : {solution.get('sample_source', 'unknown')}\n"
        f" sample_idx : {int(solution.get('sample_index', 0))}\n"
        f" region_cell: {int(solution.get('ros_map_sample_region_cell_count', 0))}\n"
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
        f" pb_yaw     : {float(solution.get('pb_base_link_yaw_rad', 0.0)):.6f} rad  "
        f"({float(solution.get('pb_base_link_yaw_deg', 0.0)):.2f} deg)\n"
        "------------------------------------------------------------\n"
        " EE vs TARGET LOCAL PB AFTER IK MOVE\n"
        f" target_xyz : [{target_xyz[0]:.4f}, {target_xyz[1]:.4f}, {target_xyz[2]:.4f}] m\n"
        f" ee_xyz     : [{ee_xyz[0]:.4f}, {ee_xyz[1]:.4f}, {ee_xyz[2]:.4f}] m\n"
        f" error_xyz  : [{ik_error_xyz[0]:+.4f}, {ik_error_xyz[1]:+.4f}, {ik_error_xyz[2]:+.4f}] m\n"
        f" ee_target_dist : {float(solution.get('ee_position_error_m', 0.0)):.4f} m\n"
        f" ee_target_ori  : {'nan' if orientation_error is None else f'{float(orientation_error):.2f}'} deg\n"
        f" approach_x : {'nan' if approach_axis_offset is None else f'{float(approach_axis_offset):.4f}'} m\n"
        f" lateral    : {'nan' if lateral_offset is None else f'{float(lateral_offset):.4f}'} m\n"
        "############################################################\n",
        flush=True,
    )

def print_pb_pose(label: str, xyz: object, yaw_rad: float | None = None) -> None:
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

def print_camera_to_pb_axis_mapping(transform_fn) -> None:
    basis_camera = np.eye(3, dtype=np.float64)
    basis_pb_local = transform_fn(basis_camera)
    print(
        "[test_base_sampler] camera point -> PB local axis mapping: "
        f"cam +X -> {basis_pb_local[0].astype(float).tolist()}, "
        f"cam +Y -> {basis_pb_local[1].astype(float).tolist()}, "
        f"cam +Z -> {basis_pb_local[2].astype(float).tolist()} "
        "(rule: pb=[cam_z, -cam_x, -cam_y])",
        flush=True,
    )
