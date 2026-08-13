"""Minimal rule-based car driver for sampled 2-D goals."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

from .debug_log import debug_stage
from .src.geometry import coordinate_transforms as coord


DEFAULT_AMCL_TOPIC = "/amcl_pose"
DEFAULT_FRONT_WHEEL_TOPIC = "car_C_front_wheel"
DEFAULT_REAR_WHEEL_TOPIC = "car_C_rear_wheel"

RULE_ACTION_MAPPINGS: dict[str, tuple[float, float, float, float]] = {
    "FORWARD_SLOW": (4.0, 4.0, 4.0, 4.0),
    "BACKWARD_SLOW": (-4.0, -4.0, -4.0, -4.0),
    "COUNTERCLOCKWISE_ROTATION_SLOW": (-5.85, 6.5, -7.8, 7.15),
    "CLOCKWISE_ROTATION_SLOW": (6.5, -5.85, 7.15, -7.8),
    "STOP": (0.0, 0.0, 0.0, 0.0),
}


@dataclass(frozen=True)
class GoalPose2D:
    x: float
    y: float
    yaw_rad: float
    z: float = 0.0


@dataclass(frozen=True)
class RuleNavigationConfig:
    amcl_topic: str = DEFAULT_AMCL_TOPIC
    front_wheel_topic: str = DEFAULT_FRONT_WHEEL_TOPIC
    rear_wheel_topic: str = DEFAULT_REAR_WHEEL_TOPIC
    xy_tolerance_m: float = 0.03
    face_target_yaw_tolerance_rad: float = 0.08
    drive_heading_tolerance_rad: float = 0.14
    final_yaw_tolerance_rad: float = 0.08
    command_period_sec: float = 0.1
    amcl_wait_timeout_sec: float = 5.0
    amcl_stale_timeout_sec: float = 1.0
    max_duration_sec: float = 120.0
    stop_repeat: int = 5
    stop_interval_sec: float = 0.03
    log_interval_sec: float = 1.0


def drive_to_pose_by_rule(
    pose: Any,
    *,
    config: RuleNavigationConfig | None = None,
    reverse_drive: bool = False,
) -> dict[str, Any]:
    cfg = config or RuleNavigationConfig()
    target_pose = goal_pose_from_any(pose)
    heading_error_fn = _target_reverse_heading_error if reverse_drive else _target_heading_error
    drive_action = "BACKWARD_SLOW" if reverse_drive else "FORWARD_SLOW"
    mode_label = "reverse return" if reverse_drive else "rule navigation"
    face_phase_name = "face_start_for_backward" if reverse_drive else "face_target"
    drive_phase_name = "backward_to_start" if reverse_drive else "drive_to_target"
    debug_stage(
        "move_car",
        "開始回程倒車到原位" if reverse_drive else "開始導航到 selected base pose",
        target_pose=goal_pose_to_nav_dict(target_pose),
        mode="backward" if reverse_drive else "forward",
    )

    try:
        import rclpy
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from rclpy.node import Node
        from std_msgs.msg import Float32MultiArray
    except ImportError as exc:
        debug_stage("move_car", "導航失敗：ROS 套件無法 import", error=str(exc))
        return {
            "success": False,
            "phase": "missing_ros",
            "target_pose": goal_pose_to_nav_dict(target_pose),
            "message": str(exc),
        }

    owns_rclpy = False
    node = None
    front_wheel_publisher = None
    rear_wheel_publisher = None
    latest_amcl_msg: PoseWithCovarianceStamped | None = None
    latest_amcl_received_at = 0.0

    def _amcl_callback(msg: PoseWithCovarianceStamped) -> None:
        nonlocal latest_amcl_msg, latest_amcl_received_at
        latest_amcl_msg = msg
        latest_amcl_received_at = time.monotonic()

    try:
        if not rclpy.ok():
            rclpy.init(args=None)
            owns_rclpy = True

        node = Node("approach_refactor_rule_navigator")
        front_wheel_publisher = node.create_publisher(Float32MultiArray, cfg.front_wheel_topic, 10)
        rear_wheel_publisher = node.create_publisher(Float32MultiArray, cfg.rear_wheel_topic, 10)
        amcl_subscription = node.create_subscription(PoseWithCovarianceStamped, cfg.amcl_topic, _amcl_callback, 10)
        debug_stage(
            "move_car",
            "ROS topic 已建立，準備等待 AMCL",
            amcl_topic=cfg.amcl_topic,
            front_topic=cfg.front_wheel_topic,
            rear_topic=cfg.rear_wheel_topic,
        )

        def _fresh_pose(timeout_sec: float) -> GoalPose2D | None:
            deadline = time.monotonic() + max(0.0, float(timeout_sec))
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                if latest_amcl_msg is None:
                    continue
                if time.monotonic() - latest_amcl_received_at <= max(0.1, cfg.amcl_stale_timeout_sec):
                    return _pose_from_amcl_msg(latest_amcl_msg)
            return None

        def _run_phase(phase_name: str, choose_action, is_done, deadline: float | None):
            next_log_at = 0.0
            while rclpy.ok():
                if deadline is not None and time.monotonic() >= deadline:
                    return False, None
                current_pose = _fresh_pose(cfg.amcl_wait_timeout_sec)
                if current_pose is None:
                    return False, None
                distance_m = _distance_to_target(current_pose, target_pose)
                heading_error = heading_error_fn(current_pose, target_pose) if distance_m > 1e-6 else 0.0
                final_yaw_error = coord.wrap_angle_rad(target_pose.yaw_rad - current_pose.yaw_rad)
                if time.monotonic() >= next_log_at:
                    debug_stage(
                        "move_car",
                        "導航進度：持續控制輪子並檢查 AMCL 與目標差距",
                        mode=mode_label,
                        phase=phase_name,
                        current=[current_pose.x, current_pose.y, current_pose.yaw_rad],
                        target=[target_pose.x, target_pose.y, target_pose.yaw_rad],
                        distance_m=distance_m,
                        heading_error_rad=heading_error,
                        final_yaw_error_rad=final_yaw_error,
                    )
                    next_log_at = time.monotonic() + max(0.2, cfg.log_interval_sec)
                if is_done(distance_m, heading_error, final_yaw_error):
                    _publish_stop_burst(front_wheel_publisher, rear_wheel_publisher, cfg)
                    return True, current_pose
                _publish_wheel_action(
                    front_wheel_publisher,
                    rear_wheel_publisher,
                    choose_action(distance_m, heading_error, final_yaw_error),
                )
                _spin_until(time.monotonic() + max(0.02, cfg.command_period_sec), node, rclpy)
            return False, None

        target_payload = goal_pose_to_nav_dict(target_pose)
        debug_stage("move_car", "等待最新 AMCL pose 作為導航起點")
        initial_pose = _fresh_pose(cfg.amcl_wait_timeout_sec)
        if initial_pose is None:
            debug_stage("move_car", "導航失敗：等不到 fresh AMCL pose")
            return {
                "success": False,
                "phase": "wait_amcl",
                "target_pose": target_payload,
                "initial_amcl_pose": {},
                "final_amcl_pose": {},
                "message": "No fresh /amcl_pose received.",
            }

        debug_stage("move_car", "取得初始 AMCL pose", initial_amcl=_pose_dict(initial_pose))

        timeout = max(0.0, float(cfg.max_duration_sec))
        deadline = None if timeout <= 0.0 else time.monotonic() + timeout

        debug_stage("move_car", "導航階段：先轉向目標方向", phase=face_phase_name)
        face_done, _ = _run_phase(
            face_phase_name,
            lambda distance, heading_error, final_yaw_error: _rotation_action_for_error(heading_error),
            lambda distance, heading_error, final_yaw_error: (
                distance <= cfg.xy_tolerance_m or abs(heading_error) <= cfg.face_target_yaw_tolerance_rad
            ),
            deadline,
        )
        if not face_done:
            debug_stage("move_car", "導航階段失敗：轉向目標方向失敗", phase=face_phase_name)
            return _navigation_failure(face_phase_name, target_payload, latest_amcl_msg, initial_pose)
        debug_stage("move_car", "導航階段完成：轉向目標方向完成", phase=face_phase_name)

        debug_stage("move_car", "導航階段：開始直線移動到目標 XY", phase=drive_phase_name, action=drive_action)
        drive_done, _ = _run_phase(
            drive_phase_name,
            lambda distance, heading_error, final_yaw_error: (
                _rotation_action_for_error(heading_error)
                if abs(heading_error) > cfg.drive_heading_tolerance_rad
                else drive_action
            ),
            lambda distance, heading_error, final_yaw_error: distance <= cfg.xy_tolerance_m,
            deadline,
        )
        if not drive_done:
            debug_stage("move_car", "導航階段失敗：移動到目標 XY 失敗", phase=drive_phase_name)
            return _navigation_failure(drive_phase_name, target_payload, latest_amcl_msg, initial_pose)
        debug_stage("move_car", "導航階段完成：已到目標 XY tolerance", phase=drive_phase_name)

        debug_stage("move_car", "導航階段：開始對齊最終 yaw")
        align_done, final_pose = _run_phase(
            "align_target_yaw",
            lambda distance, heading_error, final_yaw_error: _rotation_action_for_error(final_yaw_error),
            lambda distance, heading_error, final_yaw_error: abs(final_yaw_error) <= cfg.final_yaw_tolerance_rad,
            deadline,
        )
        if not align_done:
            debug_stage("move_car", "導航階段失敗：最終 yaw 對齊失敗")
            return _navigation_failure("align_target_yaw", target_payload, latest_amcl_msg, initial_pose)
        debug_stage("move_car", "導航階段完成：最終 yaw 對齊完成", final_amcl=_pose_dict(final_pose))

        node.destroy_subscription(amcl_subscription)
        debug_stage("move_car", "move_car 完成", final_amcl=_pose_dict(final_pose))
        return {
            "success": True,
            "phase": "done",
            "drive_mode": "backward" if reverse_drive else "forward",
            "drive_action": drive_action,
            "target_pose": target_payload,
            "initial_amcl_pose": _pose_dict(initial_pose),
            "final_amcl_pose": _pose_dict(final_pose) if final_pose is not None else {},
            "final_distance_m": None if final_pose is None else _distance_to_target(final_pose, target_pose),
            "final_yaw_error_rad": None if final_pose is None else abs(coord.wrap_angle_rad(target_pose.yaw_rad - final_pose.yaw_rad)),
        }
    finally:
        if front_wheel_publisher is not None and rear_wheel_publisher is not None:
            try:
                _publish_stop_burst(front_wheel_publisher, rear_wheel_publisher, cfg)
            except Exception:
                pass
        if node is not None:
            node.destroy_node()
        if owns_rclpy and rclpy.ok():
            rclpy.shutdown()


def goal_pose_from_any(pose: Any) -> GoalPose2D:
    if isinstance(pose, GoalPose2D):
        return pose
    if isinstance(pose, dict):
        if "goal_pose" in pose and isinstance(pose["goal_pose"], dict):
            return goal_pose_from_any(pose["goal_pose"])
        return GoalPose2D(
            x=float(pose["x"]),
            y=float(pose["y"]),
            z=float(pose.get("z", 0.0)),
            yaw_rad=coord.yaw_from_pose_like(pose),
        )
    return GoalPose2D(
        x=float(getattr(pose, "x")),
        y=float(getattr(pose, "y")),
        z=float(getattr(pose, "z", 0.0)),
        yaw_rad=coord.yaw_from_pose_like(pose),
    )


def goal_pose_to_nav_dict(goal_pose: GoalPose2D) -> dict[str, float]:
    yaw = coord.wrap_angle_rad(goal_pose.yaw_rad)
    quat = coord.yaw_to_planar_quat(yaw)
    return {
        "x": float(goal_pose.x),
        "y": float(goal_pose.y),
        "z": float(goal_pose.z),
        **quat,
        "yaw": float(yaw),
        "yaw_rad": float(yaw),
    }


def _pose_from_amcl_msg(amcl_msg: Any) -> GoalPose2D:
    pose = amcl_msg.pose.pose
    return GoalPose2D(
        x=float(pose.position.x),
        y=float(pose.position.y),
        z=float(pose.position.z),
        yaw_rad=coord.yaw_from_quat_z_w(pose.orientation.z, pose.orientation.w),
    )


def _pose_dict(pose: GoalPose2D | None) -> dict[str, float]:
    if pose is None:
        return {}
    return {
        "x": float(pose.x),
        "y": float(pose.y),
        "z": float(pose.z),
        "yaw_rad": float(pose.yaw_rad),
        "yaw_deg": float(coord.rad_to_deg(pose.yaw_rad)),
    }


def _publish_wheel_action(front_wheel_publisher: Any, rear_wheel_publisher: Any, action: str) -> None:
    from std_msgs.msg import Float32MultiArray

    front_left, front_right, rear_left, rear_right = RULE_ACTION_MAPPINGS[action]
    front_msg = Float32MultiArray()
    front_msg.data = [float(front_left), float(front_right)]
    rear_msg = Float32MultiArray()
    rear_msg.data = [float(rear_left), float(rear_right)]
    front_wheel_publisher.publish(front_msg)
    rear_wheel_publisher.publish(rear_msg)


def _publish_stop_burst(front_wheel_publisher: Any, rear_wheel_publisher: Any, cfg: RuleNavigationConfig) -> None:
    for index in range(max(1, int(cfg.stop_repeat))):
        _publish_wheel_action(front_wheel_publisher, rear_wheel_publisher, "STOP")
        if index < max(1, int(cfg.stop_repeat)) - 1:
            time.sleep(max(0.0, float(cfg.stop_interval_sec)))


def _spin_until(deadline: float, node: Any, rclpy: Any) -> None:
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))


def _rotation_action_for_error(yaw_error_rad: float) -> str:
    return "CLOCKWISE_ROTATION_SLOW" if yaw_error_rad < 0.0 else "COUNTERCLOCKWISE_ROTATION_SLOW"


def _distance_to_target(current_pose: GoalPose2D, target_pose: GoalPose2D) -> float:
    return coord.xy_distance(current_pose.x, current_pose.y, target_pose.x, target_pose.y)


def _target_heading_error(current_pose: GoalPose2D, target_pose: GoalPose2D) -> float:
    return coord.target_heading_error(current_pose.x, current_pose.y, current_pose.yaw_rad, target_pose.x, target_pose.y)


def _target_reverse_heading_error(current_pose: GoalPose2D, target_pose: GoalPose2D) -> float:
    return coord.target_reverse_heading_error(current_pose.x, current_pose.y, current_pose.yaw_rad, target_pose.x, target_pose.y)


def _navigation_failure(
    phase: str,
    target_pose: dict[str, float],
    latest_amcl_msg: Any | None,
    initial_pose: GoalPose2D | None,
) -> dict[str, Any]:
    return {
        "success": False,
        "phase": phase,
        "target_pose": target_pose,
        "initial_amcl_pose": _pose_dict(initial_pose),
        "final_amcl_pose": _pose_dict(_pose_from_amcl_msg(latest_amcl_msg)) if latest_amcl_msg is not None else {},
        "message": f"Rule navigation failed during {phase}.",
    }


def drive_back_to_pose_by_rule(pose: Any, *, config: RuleNavigationConfig | None = None) -> dict[str, Any]:
    debug_stage("move_car", "開始 drive_back_to_pose_by_rule：回程會用倒車規則", return_pose=pose)
    return drive_to_pose_by_rule(pose, config=config, reverse_drive=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the car to a sampled 2-D pose.")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--yaw-rad", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = drive_to_pose_by_rule(GoalPose2D(x=args.x, y=args.y, yaw_rad=args.yaw_rad))
    debug_stage("move_car", "CLI 執行結果", result=result)


if __name__ == "__main__":
    main()
