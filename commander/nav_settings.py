from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import yaml


_MAPPER_PARAMS_PATH = (
    Path(__file__).resolve().parent.parent
    / "tools"
    / "car_control"
    / "src"
    / "nav_goal_bridge_pkg"
    / "config"
    / "mapper_params.yaml"
)
_DEFAULT_GOAL_TOLERANCE_M = 0.10
_DEFAULT_GOAL_HEADING_TOLERANCE_RAD = math.pi


def _nested_get(payload: dict, *keys, default=None):
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


@lru_cache(maxsize=1)
def _load_mapper_params() -> dict:
    try:
        with _MAPPER_PARAMS_PATH.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}


def goal_tolerance_m_default() -> float:
    payload = _load_mapper_params()
    tolerance = _nested_get(
        payload,
        "car_control_node",
        "ros__parameters",
        "approach_stop_xy_tolerance_m",
        default=_DEFAULT_GOAL_TOLERANCE_M,
    )
    return float(tolerance)


def goal_heading_tolerance_rad_default() -> float:
    payload = _load_mapper_params()
    heading_tolerance_rad = _nested_get(
        payload,
        "car_control_node",
        "ros__parameters",
        "align_stop_yaw_tolerance_rad",
        default=None,
    )
    if heading_tolerance_rad is not None:
        return float(heading_tolerance_rad)

    legacy_heading_tolerance_deg = _nested_get(
        payload,
        "car_control_node",
        "ros__parameters",
        "align_stop_yaw_tolerance_deg",
        default=None,
    )
    if legacy_heading_tolerance_deg is not None:
        return math.radians(float(legacy_heading_tolerance_deg))

    return float(
        _nested_get(
            payload,
            "controller_server",
            "ros__parameters",
            "general_goal_checker",
            "yaw_goal_tolerance",
            default=_DEFAULT_GOAL_HEADING_TOLERANCE_RAD,
        )
    )
