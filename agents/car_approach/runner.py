"""Callable runtime entry for the refactored approach pipeline."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_TARGET_MASK_KEYS = (
    "target_mask",
    "target_object_mask",
    "object_mask",
    "object_segmentation_mask",
    "segmentation_mask",
    "mask",
)
_TARGET_MASK_BASE64_KEYS = (
    "target_mask_base64",
    "target_mask_png_base64",
    "target_object_mask_base64",
    "target_object_mask_png_base64",
    "object_mask_base64",
    "object_mask_png_base64",
    "segmentation_mask_base64",
    "segmentation_mask_png_base64",
    "mask_base64",
    "mask_png_base64",
)


def run_base_approach_sync(
    params: Dict[str, Any],
    *,
    context_id: str = "",
) -> Dict[str, Any]:
    try:
        from . import pipeline
        from .debug_log import debug_stage

        debug_stage(
            "runner",
            "收到 car_approach payload，準備進入 pipeline",
            context_id=context_id,
            has_grasp_payload=any(isinstance(params.get(key), dict) for key in ("grasp_result_payload", "grasp_result")),
            has_pointcloud=any(key in params for key in ("pointcloud_xyz", "pointcloud", "scene_pointcloud_xyz")),
            has_depth=any(params.get(key) for key in ("depth_png_bytes", "depth_image_bytes", "depth_png_base64", "depth_image_base64")),
            has_target_mask=any(key in params for key in (*_TARGET_MASK_KEYS, *_TARGET_MASK_BASE64_KEYS)),
        )
        result = pipeline.run_approach_pipeline(
            pipeline.ApproachPipelineRunConfig(
                config_path=_path_param(
                    params,
                    "config_path",
                    "car_approach_config_path",
                    "car_approach_config",
                    "base_config_path",
                    "base_config",
                    default=Path("configs/car_approach.yaml"),
                ),
                grasp_json_path=_optional_path_param(params, "grasp_json_path", "grasp_json"),
                grasp_result_payload=_grasp_payload_from_params(params),
                pointcloud_xyz=_pointcloud_from_params(params),
                depth_png_bytes=_depth_png_bytes_from_params(params),
                target_mask=_target_mask_from_params(params),
            )
        )
        debug_stage("runner", "pipeline 執行結束", success=bool(result.get("success", False)), phase=result.get("phase"))
        return {"result": result, "success": bool(result.get("success", False))}
    except Exception as exc:
        try:
            from .debug_log import debug_stage

            debug_stage("runner", "car_approach runtime 失敗：發生例外", error=str(exc))
        except Exception:
            pass
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



def _path_param(params: Dict[str, Any], *names: str, default: Path) -> Path:
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



def _grasp_payload_from_params(params: Dict[str, Any]) -> dict[str, object] | None:
    for key in ("grasp_result_payload", "grasp_result"):
        payload = params.get(key)
        if isinstance(payload, dict):
            nested_result = payload.get("result")
            grasp_payload = nested_result if isinstance(nested_result, dict) else payload
            artifact_payload = _grasp_payload_from_raw_result_ref(grasp_payload)
            return artifact_payload or grasp_payload
    return None


def _grasp_payload_from_raw_result_ref(payload: dict[str, object]) -> dict[str, object] | None:
    raw_ref = payload.get("raw_result_ref")
    if not isinstance(raw_ref, dict):
        return None
    raw_path = raw_ref.get("path")
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    candidates = [path] if path.is_absolute() else [Path("logs") / path, path]
    for candidate in candidates:
        try:
            loaded = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def _pointcloud_from_params(params: Dict[str, Any]) -> object | None:
    for key in ("pointcloud_xyz", "pointcloud", "scene_pointcloud_xyz"):
        if key in params:
            return params[key]
    return None


def _depth_png_bytes_from_params(params: Dict[str, Any]) -> bytes | None:
    for key in ("depth_png_bytes", "depth_image_bytes"):
        payload = params.get(key)
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return bytes(payload)
    for key in ("depth_png_base64", "depth_image_base64"):
        payload = params.get(key)
        if isinstance(payload, str) and payload.strip():
            return base64.b64decode(payload)
    return None


def _target_mask_from_params(params: Dict[str, Any]) -> object | None:
    for key in _TARGET_MASK_KEYS:
        if key in params:
            return params[key]
    for key in _TARGET_MASK_BASE64_KEYS:
        payload = params.get(key)
        if isinstance(payload, str) and payload.strip():
            return base64.b64decode(payload)
    return None
