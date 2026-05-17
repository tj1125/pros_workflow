"""A2A Executor for GetItemInfoAgentNoSAM3D using DataPart/FilePart contracts."""

from __future__ import annotations

import base64
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import UnsupportedOperationError

from a2a_utils.response import build_error, build_success
from get_item_info_agent_no_sam3d.pipeline.constants import DEFAULT_SCENE_CONFIG
from tool.runtime.memory import release_cuda_memory

logger = logging.getLogger(__name__)
_ACCEPTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


class GetItemInfoNoSam3dExecutor(AgentExecutor):
    """A2A executor: multi-view RGB FileParts + world_position DataPart -> goal poses."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            parts = context.message.parts
            if len(parts) < 2:
                await event_queue.enqueue_event(build_error("Expected one DataPart metadata plus at least one image FilePart.", context))
                return

            body = _extract_metadata(parts[0])
            yolo_class = str(body.get("yolo_class", "") or "").strip()
            scene_config_path = Path(body.get("scene_config", str(DEFAULT_SCENE_CONFIG)))
            selected_camera = str(body.get("selected_camera", "") or "").strip()
            world_position_data = body.get("world_position_data")
            debug_save = bool(body.get("debug_save", False))

            if not yolo_class:
                await event_queue.enqueue_event(build_error("Missing 'yolo_class' in request metadata.", context))
                return
            if world_position_data is None:
                await event_queue.enqueue_event(build_error("Missing 'world_position_data' in request metadata.", context))
                return

            camera_names = [str(name).strip() for name in (body.get("camera_names") or []) if str(name).strip()]
            image_parts = parts[1:]
            if not camera_names:
                camera_names = [_part_camera_name(part, idx) for idx, part in enumerate(image_parts, start=1)]
            if len(camera_names) != len(image_parts):
                await event_queue.enqueue_event(build_error("camera_names length must match uploaded image FileParts.", context))
                return

            camera_images = {
                camera_name: _extract_file_bytes(part, accepted_mime_types=_ACCEPTED_IMAGE_TYPES)
                for camera_name, part in zip(camera_names, image_parts)
            }
            logger.info(
                "GetItemInfoNoSAM3D: class=%s, config=%s, selected_camera=%s, uploaded=%s, context_id=%s",
                yolo_class,
                scene_config_path,
                selected_camera or "N/A",
                camera_names,
                getattr(context.message, "context_id", ""),
            )

            with tempfile.TemporaryDirectory(prefix="get_item_info_no_sam3d_") as tmp_dir:
                tmp = Path(tmp_dir)
                image_paths_by_camera: dict[str, Path] = {}
                for camera_name, image_bytes in camera_images.items():
                    image_path = tmp / f"{_safe_camera_name(camera_name)}.png"
                    image_path.write_bytes(image_bytes)
                    image_paths_by_camera[camera_name] = image_path

                from get_item_info_agent_no_sam3d.pipeline.pipeline import run_pipeline

                result = run_pipeline(
                    scene_config=scene_config_path,
                    yolo_class_name=yolo_class,
                    goal_output=tmp / "goal_pose.json",
                    debug_save=debug_save,
                    image_paths_by_camera=image_paths_by_camera,
                    primary_camera_id=selected_camera or None,
                    world_position_data=world_position_data,
                )

            await event_queue.enqueue_event(build_success(result, context, name="item_info"))
        except Exception as exc:
            logger.error("GetItemInfoNoSAM3D execution error: %s", exc, exc_info=True)
            await event_queue.enqueue_event(build_error(str(exc), context))
        finally:
            try:
                release_cuda_memory()
            except Exception as cleanup_exc:
                logger.warning("GetItemInfoNoSAM3D cleanup failed: %s", cleanup_exc)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError()


def _extract_metadata(part: Any) -> dict[str, Any]:
    root = getattr(part, "root", part)
    kind = getattr(root, "kind", "")
    if kind == "data" or hasattr(root, "data"):
        data = getattr(root, "data", None)
        if isinstance(data, dict):
            return data
    if kind == "text" or hasattr(root, "text"):
        parsed = json.loads(str(getattr(root, "text", "") or "{}"))
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("First request part must be a DataPart or JSON TextPart metadata object.")


def _extract_file_bytes(part: Any, *, accepted_mime_types: set[str]) -> bytes:
    root = getattr(part, "root", part)
    if getattr(root, "kind", "") != "file" and not hasattr(root, "file"):
        raise ValueError("Image payloads must be A2A FilePart values, not text/base64 fields.")
    file_value = getattr(root, "file", None)
    mime_type = str(getattr(file_value, "mime_type", "") or "")
    if mime_type and mime_type not in accepted_mime_types:
        raise ValueError(f"Unsupported image media type: {mime_type}")
    raw = getattr(file_value, "bytes", "") if file_value is not None else ""
    if not raw:
        raise ValueError("FilePart must contain inline bytes for this service.")
    return base64.b64decode(raw)


def _part_camera_name(part: Any, index: int) -> str:
    root = getattr(part, "root", part)
    metadata = getattr(root, "metadata", None) or {}
    camera_name = str(metadata.get("camera_name", "") or "").strip()
    return camera_name or f"camera_{index}"


def _safe_camera_name(camera_name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in camera_name)
