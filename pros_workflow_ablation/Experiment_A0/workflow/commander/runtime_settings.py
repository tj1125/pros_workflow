from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_WORKFLOW_ROOT = Path(__file__).resolve().parent.parent
_RUNTIME_CONFIG_PATH = _WORKFLOW_ROOT / "config" / "commander_runtime.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Runtime config must be a YAML object: {path}")
    return payload


@lru_cache(maxsize=1)
def _load_runtime_config() -> dict[str, Any]:
    return _load_yaml(_RUNTIME_CONFIG_PATH)


def _section(name: str) -> dict[str, Any]:
    payload = _load_runtime_config().get(name)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Missing commander runtime config section: {name}")
    return payload


def _required(section: dict[str, Any], section_name: str, key: str) -> Any:
    if key not in section or section[key] is None:
        raise RuntimeError(f"Missing commander runtime config value: {section_name}.{key}")
    return section[key]


def _required_float(section: dict[str, Any], section_name: str, key: str) -> float:
    return float(_required(section, section_name, key))


def _required_str(section: dict[str, Any], section_name: str, key: str) -> str:
    return str(_required(section, section_name, key))


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(default if value is None else value)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return str(default if value is None else value)


def _nav_runner_section() -> dict[str, Any]:
    return _section("nav_runner")


def goal_tolerance_m_default() -> float:
    runner = _nav_runner_section()
    return _required_float(runner, "nav_runner", "goal_tolerance_m")


def goal_heading_tolerance_rad_default() -> float:
    runner = _nav_runner_section()
    if runner.get("goal_heading_tolerance_rad") is not None:
        return float(runner["goal_heading_tolerance_rad"])
    if runner.get("goal_heading_tolerance_deg") is not None:
        return math.radians(float(runner["goal_heading_tolerance_deg"]))
    raise RuntimeError("Missing commander runtime config value: nav_runner.goal_heading_tolerance_rad")


def default_initial_pose() -> dict[str, Any]:
    section = _section("initial_pose")
    return {
        "frame_id": _required_str(section, "initial_pose", "frame_id"),
        "x": _required_float(section, "initial_pose", "x"),
        "y": _required_float(section, "initial_pose", "y"),
        "z": _required_float(section, "initial_pose", "z"),
        "qx": _required_float(section, "initial_pose", "qx"),
        "qy": _required_float(section, "initial_pose", "qy"),
        "qz": _required_float(section, "initial_pose", "qz"),
        "qw": _required_float(section, "initial_pose", "qw"),
    }


def nav_runner_payload_defaults() -> dict[str, Any]:
    runner = _nav_runner_section()
    legacy_heading_deg = os.getenv("NAV_GOAL_HEADING_TOLERANCE_DEG")
    heading_default = (
        math.radians(float(legacy_heading_deg))
        if legacy_heading_deg is not None
        else goal_heading_tolerance_rad_default()
    )
    return {
        "plan_timeout_sec": _env_float(
            "NAV_PLAN_TIMEOUT_SEC",
            _required_float(runner, "nav_runner", "plan_timeout_sec"),
        ),
        "arrival_timeout_sec": _env_float(
            "NAV_ARRIVAL_TIMEOUT_SEC",
            _required_float(runner, "nav_runner", "arrival_timeout_sec"),
        ),
        "publish_interval_sec": _env_float(
            "NAV_PUBLISH_INTERVAL_SEC",
            _required_float(runner, "nav_runner", "publish_interval_sec"),
        ),
        "goal_tolerance_m": _env_float("NAV_GOAL_TOLERANCE_M", goal_tolerance_m_default()),
        "goal_heading_tolerance_rad": _env_float(
            "NAV_GOAL_HEADING_TOLERANCE_RAD",
            heading_default,
        ),
        "status_topic": _env_str(
            "NAV_STATUS_TOPIC",
            _required_str(runner, "nav_runner", "status_topic"),
        ),
        "home_status_topic": _env_str(
            "NAV_HOME_STATUS_TOPIC",
            _required_str(runner, "nav_runner", "home_status_topic"),
        ),
        "nav_result_topic": _env_str(
            "NAV_RESULT_TOPIC",
            _required_str(runner, "nav_runner", "nav_result_topic"),
        ),
    }


def ros_map_origin_unity() -> tuple[float, float]:
    transform = _section("map_transform")
    return (
        _required_float(transform, "map_transform", "ros_map_origin_unity_x"),
        _required_float(transform, "map_transform", "ros_map_origin_unity_z"),
    )


def world_position_update_threshold_m() -> float:
    world_position = _section("world_position")
    return _env_float(
        "WORLD_POSITION_UPDATE_THRESHOLD_M",
        _required_float(world_position, "world_position", "update_threshold_m"),
    )


def ros_subprocess_settings() -> dict[str, str]:
    settings = _section("ros_subprocess")
    return {
        "python_bin": _env_str(
            "ROS_PYTHON_BIN",
            _required_str(settings, "ros_subprocess", "python_bin"),
        ),
        "ros_setup_bash": _env_str(
            "ROS_SETUP_BASH",
            _required_str(settings, "ros_subprocess", "ros_setup_bash"),
        ),
        "overlay_setup_bash": _env_str(
            "ROS_OVERLAY_SETUP_BASH",
            _required_str(settings, "ros_subprocess", "overlay_setup_bash"),
        ),
        "pythonpath": _env_str(
            "ROS_PYTHONPATH",
            _required_str(settings, "ros_subprocess", "pythonpath"),
        ),
        "ld_library_path": _env_str(
            "ROS_LD_LIBRARY_PATH",
            _required_str(settings, "ros_subprocess", "ld_library_path"),
        ),
    }
