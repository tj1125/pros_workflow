"""Configuration loader shared by the live base sampler.

The active entry point is ``test_base_sampler.py``.  This module only keeps the
path normalization and scalar config parsing that the sampler needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..src.pybullet_smoke import _load_yaml, _resolve_input_path


def _resolve(path_str: str, config_path: Path) -> Path:
    return _resolve_input_path(path_str, config_path)


def _vec(raw: Any, name: str, n: int) -> tuple[float, ...]:
    if len(raw) != n:
        raise ValueError(f"{name} must have {n} values, got {len(raw)}.")
    return tuple(float(v) for v in raw)


def _optional_path(raw_path: object, config_path: Path) -> Path | None:
    if raw_path is None:
        return None
    raw_text = str(raw_path).strip()
    if raw_text.lower() in {"", "null", "none"}:
        return None
    return _resolve(raw_text, config_path)


def load_config(config_path: Path) -> dict[str, Any]:
    payload = _load_yaml(config_path)

    return {
        "map_yaml_path": _resolve(str(payload["map_yaml_path"]), config_path),
        "planner_config_path": _resolve(str(payload["planner_config_path"]), config_path),
        "grasp_debug_npz_path": _optional_path(payload.get("grasp_debug_npz_path"), config_path),
        "base_link_from_amcl_pb_xy": _vec(
            payload.get("base_link_from_amcl_pb_xy", [0.0, 0.1288]),
            "base_link_from_amcl_pb_xy",
            2,
        ),
        "position_tolerance_m": float(payload.get("position_tolerance_m", 0.03)),
        "orientation_tolerance_deg": float(payload.get("orientation_tolerance_deg", 12.0)),
    }
