"""Rule-based direct movement runner for Orchestrator.minor_nav_node.

Input:  JSON payload via --payload.
Output: single-line JSON result on stdout.

This runner intentionally uses agents.car_approach.move_car.drive_to_pose_by_rule,
so minor navigation sends wheel commands directly and does not publish /goal_pose
or wait for Nav2 global-plan topics.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", default="{}")
    args = parser.parse_args()

    try:
        payload = json.loads(args.payload or "{}")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
    except Exception as exc:
        print(json.dumps(_fail("payload_error", str(exc)), ensure_ascii=False), flush=True)
        return

    with _redirect_fd_stdout_to_stderr(), contextlib.redirect_stdout(sys.stderr):
        result = _run(payload)
    print(json.dumps(_json_safe(result), ensure_ascii=False), flush=True)


def _run(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        from agents.car_approach import move_car

        goal_pose = payload.get("goal_pose", {})
        if not isinstance(goal_pose, dict) or not goal_pose:
            return _fail("missing_goal_pose", "minor_nav_rule_runner requires goal_pose.")

        rank = int(payload.get("rank", 0) or 0)
        goal_pose_index = int(payload.get("goal_pose_index", 0) or 0)
        goal_pose_source = str(payload.get("goal_pose_source", "minor_nav") or "minor_nav")
        source = str(payload.get("source", "minor_nav") or "minor_nav")
        started_at = time.time()
        events = [
            {
                "event": "rule_navigation_started",
                "rank": rank,
                "goal_pose_index": goal_pose_index,
                "goal_pose_source": goal_pose_source,
                "source": source,
                "direct_control": True,
                "timestamp": started_at,
            }
        ]

        nav_result = move_car.drive_to_pose_by_rule(
            goal_pose,
            initial_pose=None,
            initial_pose_source="minor_nav disabled /initialpose",
            config=_rule_navigation_config(move_car),
            prefer_amcl_pose=True,
            reverse_drive=bool(payload.get("reverse_drive", False)),
        )
        success = bool(nav_result.get("success", False))
        message = str(
            nav_result.get("message")
            or (
                "minor_nav rule navigation reached target."
                if success
                else "minor_nav rule navigation failed."
            )
        )
        events.append(
            {
                "event": "arrived" if success else "navigation_failed",
                "detail": message,
                "rank": rank,
                "goal_pose_index": goal_pose_index,
                "goal_pose_source": goal_pose_source,
                "source": source,
                "direct_control": True,
                "phase": nav_result.get("phase", ""),
                "timestamp": time.time(),
            }
        )
        nav_result["source"] = source
        nav_result["goal_pose_source"] = goal_pose_source
        nav_result["rank"] = rank
        nav_result["goal_pose_index"] = goal_pose_index
        return {
            "success": success,
            "plan_ready": False,
            "message": message,
            "events": events,
            "nav_result": nav_result,
            "elapsed_sec": time.time() - started_at,
        }
    except Exception as exc:
        return _fail("error", str(exc))


def _rule_navigation_config(move_car: Any) -> Any:
    return move_car.RuleNavigationConfig(
        amcl_topic=_env_str("MINOR_NAV_AMCL_TOPIC", "APPROACH_AGENT_AMCL_TOPIC", move_car.DEFAULT_AMCL_TOPIC),
        initial_pose_topic=_env_str(
            "MINOR_NAV_INITIAL_POSE_TOPIC",
            "APPROACH_AGENT_INITIAL_POSE_TOPIC",
            move_car.DEFAULT_INITIAL_POSE_TOPIC,
        ),
        initial_pose_frame_id=move_car.DEFAULT_FRAME_ID,
        initial_pose_publish_count=0,
        initial_pose_interval_sec=_env_float(
            "MINOR_NAV_INITIAL_POSE_INTERVAL_SEC",
            "APPROACH_AGENT_INITIAL_POSE_INTERVAL_SEC",
            0.1,
        ),
        initial_pose_wait_for_subscribers_sec=_env_float(
            "MINOR_NAV_INITIAL_POSE_WAIT_FOR_SUBSCRIBERS_SEC",
            "APPROACH_AGENT_INITIAL_POSE_WAIT_FOR_SUBSCRIBERS_SEC",
            2.0,
        ),
        initial_pose_settle_sec=_env_float(
            "MINOR_NAV_INITIAL_POSE_SETTLE_SEC",
            "APPROACH_AGENT_INITIAL_POSE_SETTLE_SEC",
            2.0,
        ),
        front_wheel_topic=_env_str(
            "MINOR_NAV_FRONT_WHEEL_TOPIC",
            "APPROACH_AGENT_FRONT_WHEEL_TOPIC",
            move_car.DEFAULT_FRONT_WHEEL_TOPIC,
        ),
        rear_wheel_topic=_env_str(
            "MINOR_NAV_REAR_WHEEL_TOPIC",
            "APPROACH_AGENT_REAR_WHEEL_TOPIC",
            move_car.DEFAULT_REAR_WHEEL_TOPIC,
        ),
        xy_tolerance_m=_env_float("MINOR_NAV_RULE_XY_TOLERANCE_M", "APPROACH_AGENT_RULE_XY_TOLERANCE_M", 0.03),
        face_target_yaw_tolerance_rad=_env_float(
            "MINOR_NAV_RULE_FACE_YAW_TOLERANCE_RAD",
            "APPROACH_AGENT_RULE_FACE_YAW_TOLERANCE_RAD",
            0.08,
        ),
        drive_heading_tolerance_rad=_env_float(
            "MINOR_NAV_RULE_DRIVE_HEADING_TOLERANCE_RAD",
            "APPROACH_AGENT_RULE_DRIVE_HEADING_TOLERANCE_RAD",
            0.14,
        ),
        final_yaw_tolerance_rad=_env_float(
            "MINOR_NAV_RULE_FINAL_YAW_TOLERANCE_RAD",
            "APPROACH_AGENT_RULE_FINAL_YAW_TOLERANCE_RAD",
            0.08,
        ),
        slow_approach_distance_m=_env_float(
            "MINOR_NAV_RULE_SLOW_DISTANCE_M",
            "APPROACH_AGENT_RULE_SLOW_DISTANCE_M",
            0.12,
        ),
        command_period_sec=_env_float(
            "MINOR_NAV_RULE_COMMAND_PERIOD_SEC",
            "APPROACH_AGENT_RULE_COMMAND_PERIOD_SEC",
            0.1,
        ),
        amcl_wait_timeout_sec=_env_float(
            "MINOR_NAV_RULE_AMCL_WAIT_TIMEOUT_SEC",
            "APPROACH_AGENT_RULE_AMCL_WAIT_TIMEOUT_SEC",
            5.0,
        ),
        amcl_stale_timeout_sec=_env_float(
            "MINOR_NAV_RULE_AMCL_STALE_TIMEOUT_SEC",
            "APPROACH_AGENT_RULE_AMCL_STALE_TIMEOUT_SEC",
            1.0,
        ),
        max_duration_sec=_env_float(
            "MINOR_NAV_RULE_MAX_DURATION_SEC",
            "APPROACH_AGENT_RULE_MAX_DURATION_SEC",
            120.0,
        ),
        stop_repeat=_env_int("MINOR_NAV_RULE_STOP_REPEAT", "APPROACH_AGENT_RULE_STOP_REPEAT", 5),
        stop_interval_sec=_env_float(
            "MINOR_NAV_RULE_STOP_INTERVAL_SEC",
            "APPROACH_AGENT_RULE_STOP_INTERVAL_SEC",
            0.03,
        ),
        log_interval_sec=_env_float(
            "MINOR_NAV_RULE_LOG_INTERVAL_SEC",
            "APPROACH_AGENT_RULE_LOG_INTERVAL_SEC",
            1.0,
        ),
    )


def _env_str(primary: str, fallback: str, default: str) -> str:
    value = os.getenv(primary)
    if value is None:
        value = os.getenv(fallback)
    value = str(value or "").strip()
    return value or default


def _env_float(primary: str, fallback: str, default: float) -> float:
    return float(_env_str(primary, fallback, str(default)))


def _env_int(primary: str, fallback: str, default: int) -> int:
    return int(float(_env_str(primary, fallback, str(default))))


def _fail(phase: str, message: str) -> dict[str, Any]:
    now = time.time()
    return {
        "success": False,
        "plan_ready": False,
        "message": message,
        "events": [
            {
                "event": "navigation_failed",
                "detail": message,
                "source": "minor_nav",
                "direct_control": True,
                "phase": phase,
                "timestamp": now,
            }
        ],
        "nav_result": {
            "success": False,
            "phase": phase,
            "message": message,
        },
    }


@contextlib.contextmanager
def _redirect_fd_stdout_to_stderr():
    sys.stdout.flush()
    saved_stdout_fd = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return str(value)


if __name__ == "__main__":
    main()
