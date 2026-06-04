"""Direct arm finish sequence after the car reaches the selected base pose."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .debug_log import debug_stage
from .arm_ik import _float_xyz, recompute_arrival_ik
from .src.geometry import coordinate_transforms as coord

from .joint_sequence import (
    DEFAULT_ARM_TOPIC,
    DEFAULT_JOINT_STATE_GROUPS,
    DEFAULT_JOINT_STATE_TOPIC,
    _debug_joint_phase_result,
    _float_sequence,
    _linear_interpolated_joint_positions,
    _map_joint_state_positions,
    _publish_joint_positions_for_duration,
    _publish_until_joint_leaves_position,
    _publish_waypoint_until_reached,
    _rad_sequence_to_deg_or_none,
    _round_float_sequence,
    _wait_for_joint_state_positions,
    _wait_for_publisher_subscription,
)


DEFAULT_CUBE_Z_DISTANCE_TOPIC = "/cube_z_distance"


@dataclass(frozen=True)
class ArmMotionConfig:
    urdf_path: Path
    base_height_m: float
    base_orientation_euler_deg: tuple[float, float, float]
    joint_reset_deg: tuple[float, ...]
    joint_bounds_deg: tuple[tuple[float, float], ...]
    ee_link_index: int
    controllable_joints: int
    position_tolerance_m: float
    ik_max_iterations: int
    ik_residual_threshold: float
    gripper_joint_index: int = 4
    gripper_open_deg: float = 60.0
    gripper_close_deg: float = 10.0
    arm_topic: str = DEFAULT_ARM_TOPIC
    joint_state_topic: str = DEFAULT_JOINT_STATE_TOPIC
    joint_state_groups: tuple[tuple[str, ...], ...] = DEFAULT_JOINT_STATE_GROUPS
    joint_state_wait_sec: float = 5.0
    command_timeout_sec: float = 5.0
    command_tolerance_rad: float = 0.08
    republish_interval_sec: float = 0.02
    waypoint_steps: int = 3
    reset_hold_sec: float = 3.0
    gripper_close_timeout_sec: float = 2.0
    publisher_match_timeout_sec: float = 1.0
    cube_z_distance_topic: str = DEFAULT_CUBE_Z_DISTANCE_TOPIC
    cube_z_distance_final_waypoint_stop_threshold_m: float = 0.025
    cube_z_distance_final_verify_threshold_m: float = 0.04
    cube_z_distance_poll_interval_sec: float = 0.02
    cube_z_distance_verify_timeout_sec: float = 1.0


@dataclass(frozen=True)
class BasePose:
    x: float
    y: float
    z: float
    yaw_rad: float
    source: str


def run_arrival_grasp_sequence(
    solution: dict[str, object],
    *,
    nav_result: dict[str, object],
    config: Any,
) -> dict[str, object]:
    """Recompute IK at the arrived base pose, grasp, and return the arm home."""
    debug_stage("move_arm", "開始手臂 arrival grasp sequence：抵達後重算 IK、開夾爪、移動、夾取、return_reset")
    try:
        target_world_xyz = _target_world_xyz(solution)
        planned_base = _planned_base_pose(solution, config)
        planned_target_base_xyz = coord.world_to_base_local(
            target_world_xyz,
            base_xyz=[planned_base.x, planned_base.y, planned_base.z],
            base_yaw_rad=planned_base.yaw_rad,
            arm_base_height_m=config.base_height_m,
        )
        arrived_base = _arrived_base_pose(solution, nav_result, config)
        arrival_base_error = _arrival_base_error(solution, arrived_base)
        target_base_xyz = coord.world_to_base_local(
            target_world_xyz,
            base_xyz=[arrived_base.x, arrived_base.y, arrived_base.z],
            base_yaw_rad=arrived_base.yaw_rad,
            arm_base_height_m=config.base_height_m,
        )
        target_base_delta_xyz = _sequence_delta(target_base_xyz, planned_target_base_xyz)
        debug_stage(
            "move_arm",
            "arrival pose 對齊資訊：planned/actual AMCL 與 arm base",
            planned_amcl=solution.get("goal_pose"),
            actual_amcl=nav_result.get("final_amcl_pose") if isinstance(nav_result, dict) else None,
            planned_base=_base_pose_dict(planned_base),
            actual_base=_base_pose_dict(arrived_base),
            base_error=arrival_base_error,
        )
        debug_stage(
            "move_arm",
            "arrival grasp target 補償資訊：planned local vs actual local",
            target_world_xyz=_round_float_sequence(target_world_xyz),
            planned_target_base_xyz=_round_float_sequence(planned_target_base_xyz),
            actual_target_base_xyz=_round_float_sequence(target_base_xyz),
            target_base_delta_xyz=_round_float_sequence(target_base_delta_xyz),
            selected_ik_deg=_round_float_sequence(solution.get("ik_joint_solution_deg")),
        )
        debug_stage(
            "move_arm",
            "階段 1：根據實際抵達 AMCL 換回 PyBullet arm base，並把 target 轉到手臂 local frame",
            arrived_base=_base_pose_dict(arrived_base),
            target_base_xyz=target_base_xyz,
            xy_error=arrival_base_error.get("xy_norm_m"),
            yaw_error=arrival_base_error.get("yaw_error_rad"),
        )
        debug_stage("move_arm", "階段 2：重新計算 arrival IK")
        ik_result = recompute_arrival_ik(target_base_xyz, config=config)
        if not bool(ik_result.get("ik_feasible", False)):
            debug_stage(
                "move_arm",
                "階段 2 失敗：arrival IK 沒有達到 tolerance",
                position_error=ik_result.get("position_error_m"),
                tolerance=ik_result.get("position_tolerance_m"),
            )
            return {
                "success": False,
                "phase": "arrival_ik_failed",
                "message": "Arrival-adjusted arm IK did not meet position tolerance.",
                "target_world_xyz": target_world_xyz,
                "target_world_rotation_matrix": solution.get("target_rotation_matrix"),
                "planned_amcl_pose": solution.get("goal_pose"),
                "actual_amcl_pose": nav_result.get("final_amcl_pose") if isinstance(nav_result, dict) else None,
                "planned_base_pose": _base_pose_dict(planned_base),
                "arrived_base_pose": _base_pose_dict(arrived_base),
                "arrival_base_error": arrival_base_error,
                "planned_target_base_xyz": planned_target_base_xyz,
                "actual_target_base_xyz": target_base_xyz,
                "arrival_target_base_delta_xyz": target_base_delta_xyz,
                "selected_ik_joint_solution_deg": solution.get("ik_joint_solution_deg"),
                **ik_result,
            }
        debug_stage(
            "move_arm",
            "階段 2 完成：arrival IK 可行，準備送手臂 joint sequence",
            position_error=ik_result.get("position_error_m"),
        )
        debug_stage("move_arm", "階段 3：執行 open_gripper -> move_to_target -> close_gripper -> return_reset -> cube_z_verify")
        execute_result = execute_grasp_joint_sequence(
            ik_result["joint_solution_rad"],
            config=config,
        )
        debug_stage(
            "move_arm",
            "階段 3 完成：手臂 joint sequence 結束",
            success=execute_result.get("success"),
            phase=execute_result.get("phase"),
        )
        return {
            "success": bool(execute_result.get("success", False)),
            "phase": "done" if execute_result.get("success") else "joint_sequence_failed",
            "message": execute_result.get("message", "Arm sequence finished."),
            "target_world_xyz": target_world_xyz,
            "target_world_rotation_matrix": solution.get("target_rotation_matrix"),
            "planned_amcl_pose": solution.get("goal_pose"),
            "actual_amcl_pose": nav_result.get("final_amcl_pose") if isinstance(nav_result, dict) else None,
            "planned_base_pose": _base_pose_dict(planned_base),
            "arrived_base_pose": _base_pose_dict(arrived_base),
            "arrival_base_error": arrival_base_error,
            "planned_target_base_xyz": planned_target_base_xyz,
            "actual_target_base_xyz": target_base_xyz,
            "arrival_target_base_delta_xyz": target_base_delta_xyz,
            "selected_ik_joint_solution_deg": solution.get("ik_joint_solution_deg"),
            "target_base_xyz": target_base_xyz,
            "arrival_ik": ik_result,
            **execute_result,
        }
    except Exception as exc:
        debug_stage("move_arm", "手臂 arrival grasp sequence 失敗：發生例外", error=str(exc))
        return {
            "success": False,
            "phase": "error",
            "message": str(exc),
        }


def execute_grasp_joint_sequence(goal_joint_rad: Sequence[Any], *, config: ArmMotionConfig) -> dict[str, object]:
    debug_stage("move_arm", "joint sequence：開始連接 ROS arm topic")
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float32, Float64
        from trajectory_msgs.msg import JointTrajectoryPoint
    except ImportError as exc:
        debug_stage("move_arm", "joint sequence 失敗：ROS 套件無法 import", error=str(exc))
        return {"success": False, "phase": "missing_ros", "message": str(exc)}

    goal_positions = _float_sequence(goal_joint_rad, label="goal_joint_rad")
    joint_count = len(goal_positions)
    gripper_index = int(config.gripper_joint_index)
    if not (0 <= gripper_index < joint_count):
        raise ValueError(f"gripper joint index {gripper_index} is outside joint vector length {joint_count}.")
    ignored_joint_state_indices = (gripper_index,)

    fallback_reset_positions = coord.deg_sequence_to_rad(config.joint_reset_deg[:joint_count])
    if len(fallback_reset_positions) != joint_count:
        raise ValueError(f"joint_reset_deg needs {joint_count} values.")
    target_positions = list(goal_positions)
    target_positions[gripper_index] = coord.deg_to_rad(config.gripper_open_deg)
    close_positions = list(target_positions)
    close_positions[gripper_index] = coord.deg_to_rad(config.gripper_close_deg)

    owns_rclpy = False
    node = None
    latest_joint_positions: list[float] | None = None
    latest_raw_joint_positions: list[float] | None = None
    latest_cube_z_distance_m: float | None = None
    latest_cube_z_distance_time_sec: float | None = None

    def get_latest_raw_positions() -> list[float] | None:
        return None if latest_raw_joint_positions is None else list(latest_raw_joint_positions)

    def get_latest_positions() -> list[float] | None:
        return None if latest_joint_positions is None else list(latest_joint_positions)

    def get_latest_cube_z_distance(*, since_time_sec: float | None = None) -> tuple[float, float] | None:
        if latest_cube_z_distance_m is None or latest_cube_z_distance_time_sec is None:
            return None
        if since_time_sec is not None and latest_cube_z_distance_time_sec < float(since_time_sec):
            return None
        return float(latest_cube_z_distance_m), float(latest_cube_z_distance_time_sec)

    def on_joint_state(msg: JointState) -> None:
        nonlocal latest_joint_positions, latest_raw_joint_positions
        mapped = _map_joint_state_positions(
            msg,
            joint_state_groups=config.joint_state_groups,
            joint_count=joint_count,
            ignored_joint_indices=ignored_joint_state_indices,
        )
        if mapped is not None:
            latest_raw_joint_positions = mapped
            latest_joint_positions = list(mapped)

    def on_cube_z_distance(msg: Any) -> None:
        nonlocal latest_cube_z_distance_m, latest_cube_z_distance_time_sec
        try:
            value = float(msg.data)
        except Exception:
            return
        if math.isfinite(value):
            latest_cube_z_distance_m = value
            latest_cube_z_distance_time_sec = time.monotonic()

    phases: list[dict[str, object]] = []
    try:
        if not rclpy.ok():
            rclpy.init(args=None)
            owns_rclpy = True
        node = Node("approach_refactor_arm_finish")
        publisher = node.create_publisher(JointTrajectoryPoint, config.arm_topic, 10)
        node.create_subscription(JointState, config.joint_state_topic, on_joint_state, 10)
        cube_z_msg_type = _cube_z_distance_msg_type(
            rclpy,
            node,
            config.cube_z_distance_topic,
            Float32,
            Float64,
            timeout_sec=config.publisher_match_timeout_sec,
        )
        node.create_subscription(cube_z_msg_type, config.cube_z_distance_topic, on_cube_z_distance, 10)
        publisher_subscription_count_at_start = _wait_for_publisher_subscription(
            rclpy,
            node,
            publisher,
            timeout_sec=config.publisher_match_timeout_sec,
        )
        debug_stage(
            "move_arm",
            "joint sequence：ROS publisher/subscriber 已建立",
            arm_topic=config.arm_topic,
            joint_state_topic=config.joint_state_topic,
            arm_topic_subscription_count=publisher_subscription_count_at_start,
            cube_z_distance_topic=config.cube_z_distance_topic,
            cube_z_distance_msg_type=getattr(cube_z_msg_type, "__name__", str(cube_z_msg_type)),
        )
        time.sleep(0.2)

        debug_stage("move_arm", "joint sequence：等待目前 joint state，僅作 debug；waypoint 起點使用 config reset joints")
        observed_mapped_start_positions = _wait_for_joint_state_positions(
            rclpy,
            node,
            get_latest_positions,
            timeout_sec=config.joint_state_wait_sec,
        )
        observed_raw_start_positions = get_latest_raw_positions()
        start_positions = list(fallback_reset_positions)
        start_source = "config_joint_reset_deg"
        observed_start_source = "joint_states_absolute" if observed_mapped_start_positions is not None else "unavailable"
        if observed_mapped_start_positions is None:
            debug_stage("move_arm", "joint sequence：等不到 joint state，仍使用 config reset 角度當 waypoint 起點")
        else:
            debug_stage(
                "move_arm",
                "joint sequence：已取得目前 joint state，raw 視為 absolute joint angles",
                raw_joint_state_deg=_round_float_sequence(_rad_sequence_to_deg_or_none(observed_raw_start_positions)),
                mapped_joint_state_deg=_round_float_sequence(_rad_sequence_to_deg_or_none(observed_mapped_start_positions)),
            )
        return_reset_positions = list(fallback_reset_positions)
        return_reset_source = "config_joint_reset_deg"
        open_positions = list(start_positions)
        open_positions[gripper_index] = coord.deg_to_rad(config.gripper_open_deg)
        debug_stage(
            "move_arm",
            "joint sequence：角度計畫",
            joint_state_frame="absolute_joint_positions",
            start_deg=_round_float_sequence(coord.rad_sequence_to_deg(start_positions)),
            observed_raw_start_deg=_round_float_sequence(_rad_sequence_to_deg_or_none(observed_raw_start_positions)),
            observed_mapped_start_deg=_round_float_sequence(_rad_sequence_to_deg_or_none(observed_mapped_start_positions)),
            open_deg=_round_float_sequence(coord.rad_sequence_to_deg(open_positions)),
            target_deg=_round_float_sequence(coord.rad_sequence_to_deg(target_positions)),
            close_deg=_round_float_sequence(coord.rad_sequence_to_deg(close_positions)),
            return_reset_deg=_round_float_sequence(coord.rad_sequence_to_deg(return_reset_positions)),
            ignored_joint_indices=list(ignored_joint_state_indices),
            publisher_subscription_count_at_start=publisher_subscription_count_at_start if "publisher_subscription_count_at_start" in locals() else None,
            cube_z_distance_topic=str(config.cube_z_distance_topic),
        )

        debug_stage("move_arm", "joint sequence 階段：open_gripper，等待夾爪到 60 deg")
        open_result = _publish_waypoint_until_reached(
            rclpy,
            node,
            publisher,
            get_latest_positions,
            open_positions,
            phase="open_gripper",
            tolerance_rad=config.command_tolerance_rad,
            timeout_sec=config.command_timeout_sec,
            republish_interval_sec=config.republish_interval_sec,
            ignored_joint_indices=(),
        )
        phases.append(open_result)
        _debug_joint_phase_result("joint sequence waypoint 結果：open_gripper", open_result)

        if bool(phases[-1].get("success", False)):
            waypoints = _linear_interpolated_joint_positions(
                open_positions,
                target_positions,
                steps=config.waypoint_steps,
            )
            debug_stage("move_arm", "joint sequence 階段：move_to_target", waypoint_count=len(waypoints))
            for index, waypoint in enumerate(waypoints, start=1):
                debug_stage(
                    "move_arm",
                    "joint sequence 階段：move_to_target waypoint",
                    index=index,
                    total=len(waypoints),
                    target_deg=_round_float_sequence(coord.rad_sequence_to_deg(waypoint)),
                )
                is_final_waypoint = index == len(waypoints)
                final_waypoint_started_at = time.monotonic() if is_final_waypoint else None
                waypoint_result = _publish_waypoint_until_reached(
                    rclpy,
                    node,
                    publisher,
                    get_latest_positions,
                    waypoint,
                    phase=f"move_to_target_{index:02d}",
                    tolerance_rad=config.command_tolerance_rad,
                    timeout_sec=config.command_timeout_sec,
                    republish_interval_sec=config.republish_interval_sec,
                    ignored_joint_indices=ignored_joint_state_indices,
                    early_stop_reader=get_latest_cube_z_distance if is_final_waypoint else None,
                    early_stop_threshold_m=config.cube_z_distance_final_waypoint_stop_threshold_m if is_final_waypoint else 0.0,
                    early_stop_since_time_sec=final_waypoint_started_at,
                    early_stop_poll_interval_sec=config.cube_z_distance_poll_interval_sec,
                )
                phases.append(waypoint_result)
                _debug_joint_phase_result(
                    "joint sequence waypoint 結果：move_to_target",
                    waypoint_result,
                    waypoint_index=index,
                    waypoint_total=len(waypoints),
                )
                if not bool(phases[-1].get("success", False)):
                    break

        reached_target = bool(phases) and all(
            bool(phase.get("success", False))
            for phase in phases
            if str(phase.get("phase", "")).startswith(("open_gripper", "move_to_target"))
        )
        if reached_target:
            debug_stage("move_arm", "joint sequence 階段：close_gripper")
            close_result = _publish_until_joint_leaves_position(
                rclpy,
                node,
                publisher,
                get_latest_positions,
                close_positions,
                phase="close_gripper",
                joint_index=gripper_index,
                reference_position_rad=coord.deg_to_rad(config.gripper_open_deg),
                leave_tolerance_rad=config.command_tolerance_rad,
                timeout_sec=config.gripper_close_timeout_sec,
                republish_interval_sec=config.republish_interval_sec,
                ignored_joint_indices=ignored_joint_state_indices,
            )
            phases.append(close_result)
            _debug_joint_phase_result("joint sequence command 結果：close_gripper", close_result)
        else:
            debug_stage("move_arm", "joint sequence 階段失敗：move_to_target 沒完成，略過 close_gripper")
            phases.append({"success": False, "skipped": True, "phase": "close_gripper", "message": "target move failed"})

        if any(int(phase.get("published_count", 0) or 0) > 0 for phase in phases):
            debug_stage("move_arm", "joint sequence 階段：return_reset 回到 config reset joints")
            return_result = _publish_joint_positions_for_duration(
                rclpy,
                node,
                publisher,
                get_latest_positions,
                return_reset_positions,
                phase="return_reset",
                duration_sec=config.reset_hold_sec,
                republish_interval_sec=config.republish_interval_sec,
                ignored_joint_indices=ignored_joint_state_indices,
            )
            phases.append(return_result)
            _debug_joint_phase_result("joint sequence command 結果：return_reset", return_result)

            verify_since = time.monotonic()
            cube_verify_result = _wait_for_cube_z_distance_below(
                rclpy,
                node,
                get_latest_cube_z_distance,
                threshold_m=config.cube_z_distance_final_verify_threshold_m,
                timeout_sec=config.cube_z_distance_verify_timeout_sec,
                poll_interval_sec=config.cube_z_distance_poll_interval_sec,
                since_time_sec=verify_since,
            )
            phases.append(cube_verify_result)
            debug_stage(
                "move_arm",
                "joint sequence 結果：reset 後 cube_z_distance 驗證",
                success=cube_verify_result.get("success"),
                distance_m=cube_verify_result.get("distance_m"),
                threshold_m=cube_verify_result.get("threshold_m"),
                timed_out=cube_verify_result.get("timed_out"),
            )
    finally:
        if node is not None:
            node.destroy_node()
        if owns_rclpy and rclpy.ok():
            rclpy.shutdown()

    success = bool(phases) and all(bool(phase.get("success", False)) for phase in phases)
    failed_phase = next((phase for phase in phases if not bool(phase.get("success", False))), None)
    debug_stage("move_arm", "joint sequence 結束", success=success, failed_phase=(failed_phase or {}).get("phase"))
    return {
        "success": success,
        "phase": "done" if success else str((failed_phase or {}).get("phase", "failed")),
        "message": "arm open/move/grasp/return_reset/cube_z_verify sequence finished" if success else str((failed_phase or {}).get("message", "arm sequence failed")),
        "execution_sequence": ["recompute_ik", "open_gripper_confirm_60deg", "move_to_target", "close_gripper_confirm_not_60deg", "return_reset", "verify_cube_z_distance_after_reset"],
        "joint_start_source": start_source if "start_source" in locals() else "unavailable",
        "observed_joint_start_source": observed_start_source if "observed_start_source" in locals() else "unavailable",
        "joint_state_frame": "absolute_joint_positions",
        "observed_raw_start_positions_rad": observed_raw_start_positions if "observed_raw_start_positions" in locals() else None,
        "observed_raw_start_positions_deg": None if "observed_raw_start_positions" not in locals() or observed_raw_start_positions is None else coord.rad_sequence_to_deg(observed_raw_start_positions),
        "observed_mapped_start_positions_rad": observed_mapped_start_positions if "observed_mapped_start_positions" in locals() else None,
        "observed_mapped_start_positions_deg": None if "observed_mapped_start_positions" not in locals() or observed_mapped_start_positions is None else coord.rad_sequence_to_deg(observed_mapped_start_positions),
        "arm_topic": str(config.arm_topic),
        "joint_state_topic": str(config.joint_state_topic),
        "joint_state_groups": [list(group) for group in config.joint_state_groups],
        "phases": phases,
        "target_positions_rad": target_positions,
        "target_positions_deg": coord.rad_sequence_to_deg(target_positions),
        "close_positions_rad": close_positions,
        "close_positions_deg": coord.rad_sequence_to_deg(close_positions),
        "return_reset_positions_rad": return_reset_positions if "return_reset_positions" in locals() else fallback_reset_positions,
        "return_reset_positions_deg": coord.rad_sequence_to_deg(return_reset_positions if "return_reset_positions" in locals() else fallback_reset_positions),
        "return_reset_source": return_reset_source if "return_reset_source" in locals() else "config_joint_reset_deg",
        "fallback_reset_positions_rad": fallback_reset_positions,
        "fallback_reset_positions_deg": coord.rad_sequence_to_deg(fallback_reset_positions),
        "gripper_joint_index": gripper_index,
        "gripper_open_deg": float(config.gripper_open_deg),
        "gripper_close_deg": float(config.gripper_close_deg),
        "joint_state_ignored_indices": list(ignored_joint_state_indices),
        "arm_topic_subscription_count_at_start": publisher_subscription_count_at_start if "publisher_subscription_count_at_start" in locals() else None,
        "cube_z_distance_topic": str(config.cube_z_distance_topic),
        "cube_z_distance_final_waypoint_stop_threshold_m": float(config.cube_z_distance_final_waypoint_stop_threshold_m),
        "cube_z_distance_final_verify_threshold_m": float(config.cube_z_distance_final_verify_threshold_m),
        "latest_cube_z_distance_m": latest_cube_z_distance_m,
    }


def _cube_z_distance_msg_type(
    rclpy_module: Any,
    node: Any,
    topic: str,
    float32_type: Any,
    float64_type: Any,
    *,
    timeout_sec: float,
) -> Any:
    normalized_topic = str(topic).rstrip("/") or str(topic)
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while True:
        try:
            topic_types = node.get_topic_names_and_types()
        except Exception:
            return float32_type
        for topic_name, type_names in topic_types:
            if str(topic_name).rstrip("/") != normalized_topic:
                continue
            if "std_msgs/msg/Float64" in type_names:
                return float64_type
            if "std_msgs/msg/Float32" in type_names:
                return float32_type
        if time.monotonic() >= deadline:
            return float32_type
        rclpy_module.spin_once(node, timeout_sec=0.05)


def _wait_for_cube_z_distance_below(
    rclpy_module: Any,
    node: Any,
    reader: Any,
    *,
    threshold_m: float,
    timeout_sec: float,
    poll_interval_sec: float,
    since_time_sec: float | None,
) -> dict[str, object]:
    threshold = float(threshold_m)
    if threshold <= 0.0:
        return {
            "success": True,
            "skipped": True,
            "phase": "verify_cube_z_distance_after_reset",
            "message": "cube_z_distance final verification disabled",
            "threshold_m": threshold,
        }
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    poll_interval = max(0.001, float(poll_interval_sec))
    last_distance: float | None = None
    last_received_at: float | None = None
    while time.monotonic() < deadline:
        rclpy_module.spin_once(node, timeout_sec=min(0.05, poll_interval))
        reading = reader(since_time_sec=since_time_sec)
        if reading is None:
            continue
        distance, received_at = reading
        last_distance = float(distance)
        last_received_at = float(received_at)
        if last_distance < threshold:
            return {
                "success": True,
                "phase": "verify_cube_z_distance_after_reset",
                "message": "cube_z_distance is below final success threshold after reset",
                "distance_m": last_distance,
                "threshold_m": threshold,
                "received_at_monotonic_sec": last_received_at,
                "timed_out": False,
            }
    return {
        "success": False,
        "phase": "verify_cube_z_distance_after_reset",
        "message": "cube_z_distance did not go below final success threshold after reset",
        "distance_m": last_distance,
        "threshold_m": threshold,
        "received_at_monotonic_sec": last_received_at,
        "timed_out": True,
        "timeout_sec": float(timeout_sec),
    }


def reset_arm_to_config_pose(*, config: ArmMotionConfig, reason: str = "cleanup") -> dict[str, object]:
    debug_stage("move_arm", "cleanup：準備把手臂回到 config reset joints", reason=reason)
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectoryPoint
    except ImportError as exc:
        debug_stage("move_arm", "cleanup reset 失敗：ROS 套件無法 import", error=str(exc))
        return {"success": False, "phase": "missing_ros", "message": str(exc), "reason": reason}

    reset_positions = coord.deg_sequence_to_rad(config.joint_reset_deg[: int(config.controllable_joints)])
    joint_count = len(reset_positions)
    if joint_count != int(config.controllable_joints):
        return {
            "success": False,
            "phase": "invalid_reset_config",
            "message": f"joint_reset_deg needs {config.controllable_joints} values.",
            "reason": reason,
        }
    gripper_index = int(config.gripper_joint_index)
    ignored_joint_state_indices = (gripper_index,) if 0 <= gripper_index < joint_count else ()

    owns_rclpy = False
    node = None
    latest_joint_positions: list[float] | None = None
    latest_raw_joint_positions: list[float] | None = None

    def get_latest_raw_positions() -> list[float] | None:
        return None if latest_raw_joint_positions is None else list(latest_raw_joint_positions)

    def get_latest_positions() -> list[float] | None:
        return None if latest_joint_positions is None else list(latest_joint_positions)

    def on_joint_state(msg: JointState) -> None:
        nonlocal latest_joint_positions, latest_raw_joint_positions
        mapped = _map_joint_state_positions(
            msg,
            joint_state_groups=config.joint_state_groups,
            joint_count=joint_count,
            ignored_joint_indices=ignored_joint_state_indices,
        )
        if mapped is not None:
            latest_raw_joint_positions = mapped
            latest_joint_positions = list(mapped)

    try:
        if not rclpy.ok():
            rclpy.init(args=None)
            owns_rclpy = True
        node = Node("approach_refactor_arm_cleanup_reset")
        publisher = node.create_publisher(JointTrajectoryPoint, config.arm_topic, 10)
        node.create_subscription(JointState, config.joint_state_topic, on_joint_state, 10)
        time.sleep(0.2)
        start_positions = _wait_for_joint_state_positions(
            rclpy,
            node,
            get_latest_positions,
            timeout_sec=config.joint_state_wait_sec,
        )
        raw_start_positions = get_latest_raw_positions()
        result = _publish_joint_positions_for_duration(
            rclpy,
            node,
            publisher,
            get_latest_positions,
            reset_positions,
            phase="cleanup_reset_joints",
            duration_sec=config.reset_hold_sec,
            republish_interval_sec=config.republish_interval_sec,
            ignored_joint_indices=ignored_joint_state_indices,
        )
        result.update(
            {
                "reason": reason,
                "reset_positions_rad": reset_positions,
                "reset_positions_deg": coord.rad_sequence_to_deg(reset_positions),
                "start_positions_rad": start_positions,
                "start_positions_deg": None if start_positions is None else coord.rad_sequence_to_deg(start_positions),
                "raw_start_positions_rad": raw_start_positions,
                "raw_start_positions_deg": None if raw_start_positions is None else coord.rad_sequence_to_deg(raw_start_positions),
                "joint_state_frame": "absolute_joint_positions",
                "joint_state_ignored_indices": list(ignored_joint_state_indices),
            }
        )
        _debug_joint_phase_result("cleanup reset 結果：手臂回到 config reset joints", result)
        return result
    except Exception as exc:
        debug_stage("move_arm", "cleanup reset 失敗：發生例外", error=str(exc), reason=reason)
        return {"success": False, "phase": "cleanup_reset_error", "message": str(exc), "reason": reason}
    finally:
        if node is not None:
            node.destroy_node()
        if owns_rclpy and rclpy.ok():
            rclpy.shutdown()


def _sequence_delta(values: Sequence[Any], reference: Sequence[Any]) -> list[float]:
    current = _float_sequence(values, label="values")
    base = _float_sequence(reference, label="reference")
    if len(current) != len(base):
        raise ValueError("values and reference must have the same length.")
    return [float(value - ref) for value, ref in zip(current, base)]


def _target_world_xyz(solution: dict[str, object]) -> list[float]:
    for key in ("target_xyz", "target_pb", "target_position_pybullet_xyz", "final_ee_position_xyz"):
        if solution.get(key) is not None:
            return _float_xyz(solution[key], label=f"solution.{key}")
    raise KeyError("selected solution needs target_xyz for arrival arm IK.")


def _arrived_base_pose(solution: dict[str, object], nav_result: dict[str, object], config: ArmMotionConfig) -> BasePose:
    final_pose = nav_result.get("final_amcl_pose") if isinstance(nav_result, dict) else None
    reference_amcl_pose = solution.get("reference_amcl_pose") if isinstance(solution, dict) else None
    arm_base_link_pb_xyz = solution.get("arm_base_link_pb_xyz", [-0.001193, -0.001505, 0.039689])
    car_center_from_arm_base_pb_xy = solution.get("car_center_from_arm_base_pb_xy", [0.0, -0.1285])
    if isinstance(final_pose, dict) and final_pose.get("x") is not None and final_pose.get("y") is not None:
        if reference_amcl_pose is not None:
            local_xy, local_yaw = coord.ros_map_amcl_pose_to_local_pybullet_base_pose(
                final_pose,
                reference_amcl_pose=reference_amcl_pose,
                arm_base_link_pb_xyz=arm_base_link_pb_xyz,
                car_center_from_arm_base_pb_xy=car_center_from_arm_base_pb_xy,
                reference_pb_yaw_rad=float(solution.get("reference_pb_yaw_rad", coord.deg_to_rad(config.base_orientation_euler_deg[2]))),
            )
            planned_base = _planned_base_pose(solution, config)
            return BasePose(
                x=float(local_xy[0]),
                y=float(local_xy[1]),
                z=planned_base.z,
                yaw_rad=float(local_yaw),
                source="nav_result.final_amcl_pose_to_local_pybullet_base",
            )
        raise ValueError("final_amcl_pose is available but selected solution has no reference_amcl_pose for frame alignment.")
    return _planned_base_pose(solution, config)


def _planned_base_pose(solution: dict[str, object], config: ArmMotionConfig) -> BasePose:
    base_xyz = _float_xyz(solution["pb_base_link_xyz"], label="solution.pb_base_link_xyz")
    return BasePose(
        x=float(base_xyz[0]),
        y=float(base_xyz[1]),
        z=float(base_xyz[2] if len(base_xyz) > 2 else config.base_height_m),
        yaw_rad=float(solution["pb_base_link_yaw_rad"]),
        source="selected_solution.pb_base_link_pose",
    )


def _arrival_base_error(solution: dict[str, object], arrived_base: BasePose) -> dict[str, object]:
    planned = _planned_base_pose_like(solution)
    actual = {"x": arrived_base.x, "y": arrived_base.y, "z": arrived_base.z, "yaw_rad": arrived_base.yaw_rad}
    error = coord.planar_pose_error(planned, actual)
    return {
        "planned_base_pose": planned,
        "actual_base_pose": actual,
        **error,
    }


def _planned_base_pose_like(solution: dict[str, object]) -> dict[str, float]:
    base_xyz = _float_xyz(solution["pb_base_link_xyz"], label="solution.pb_base_link_xyz")
    return coord.pose2d_dict(
        base_xyz[0],
        base_xyz[1],
        float(solution["pb_base_link_yaw_rad"]),
        z=base_xyz[2],
    )


def _base_pose_dict(base_pose: BasePose) -> dict[str, object]:
    return {
        "x": float(base_pose.x),
        "y": float(base_pose.y),
        "z": float(base_pose.z),
        "yaw_rad": float(base_pose.yaw_rad),
        "yaw_deg": coord.rad_to_deg(base_pose.yaw_rad),
        "source": base_pose.source,
    }
