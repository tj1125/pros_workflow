"""Callable runtime entry for the base approach policy."""

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def run_base_approach_sync(
    params: Dict[str, Any],
    *,
    context_id: str = "",
) -> Dict[str, Any]:
    try:
        from . import base_sampler

        grasp_payload = _grasp_payload_from_params(params)
        config = base_sampler.ApproachAgentRunConfig(
            base_config_path=_path_param(
                params,
                "base_config_path",
                "base_config",
                default=Path("configs/base_pose_sampling.yaml"),
            ),
            camera_config_path=_path_param(
                params,
                "camera_config_path",
                "camera_config",
                default=Path("configs/camera_car_voxel_ompl.yaml"),
            ),
            grasp_json_path=_optional_path_param(params, "grasp_json_path", "grasp_json"),
            grasp_result_payload=grasp_payload,
            initial_pose=_dict_param(params, "initial_pose"),
            initial_pose_source=_str_param(
                params,
                "initial_pose_source",
                default="",
            ),
            allow_missing_amcl=_bool_param(params, "allow_missing_amcl", False),
            run_rule_navigation=_run_rule_navigation_from_params(params),
            show_gui=_bool_param(params, "show_gui", False),
            write_map_png=_bool_param(params, "write_map_png", True),
            map_png_path=_optional_path_param(params, "map_png_path"),
        )
        result = base_sampler.run_approach_agent(config)
        return {
            "result": result,
            "success": bool(result.get("success", False)),
        }
    except Exception as exc:
        logger.exception("Base approach runtime failed")
        return {
            "result": {
                "success": False,
                "status_code": "APPROACH_FAIL",
                "phase": "error",
                "error": str(exc),
                "context_id": context_id,
                "next_agent": None,
            },
            "success": False,
        }


def _bool_param(params: Dict[str, Any], name: str, default: bool) -> bool:
    raw_value = params.get(name)
    if raw_value is None:
        return bool(default)
    if isinstance(raw_value, bool):
        return raw_value
    return str(raw_value).strip().lower() not in {"0", "false", "no", "off", ""}


def _path_param(
    params: Dict[str, Any],
    *names: str,
    default: Path,
) -> Path:
    for name in names:
        raw_value = params.get(name)
        if raw_value:
            return Path(str(raw_value)).expanduser()
    return default


def _optional_path_param(params: Dict[str, Any], *names: str) -> Path | None:
    for name in names:
        raw_value = params.get(name)
        if raw_value:
            return Path(str(raw_value)).expanduser()
    return None


def _dict_param(params: Dict[str, Any], *names: str) -> dict[str, object] | None:
    for name in names:
        raw_value = params.get(name)
        if isinstance(raw_value, dict) and raw_value:
            return raw_value
    return None


def _str_param(params: Dict[str, Any], name: str, *, default: str = "") -> str:
    raw_value = params.get(name)
    if raw_value is None:
        return default
    return str(raw_value)


def _run_rule_navigation_from_params(params: Dict[str, Any]) -> bool:
    run_rule_navigation = _bool_param(
        params,
        "run_rule_navigation",
        _bool_param(params, "rule_navigation", True),
    )
    if "no_publish_goal_pose" in params:
        run_rule_navigation = not _bool_param(params, "no_publish_goal_pose", False)
    if "no_rule_nav" in params:
        run_rule_navigation = not _bool_param(params, "no_rule_nav", False)
    return run_rule_navigation


def _grasp_payload_from_params(params: Dict[str, Any]) -> dict[str, object] | None:
    for key in ("grasp_result_payload", "grasp_result", "latest_grasp_result"):
        payload = params.get(key)
        if isinstance(payload, dict):
            nested_result = payload.get("result")
            if isinstance(nested_result, dict):
                return nested_result
            return payload
    return None
