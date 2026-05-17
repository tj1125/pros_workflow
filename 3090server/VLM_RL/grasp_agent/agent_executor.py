from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import UnsupportedOperationError

from a2a_utils.response import build_error, build_success
from grasp_agent.pipeline.constants import DEFAULT_CONFIG

logger = logging.getLogger(__name__)
_ACCEPTED_RGB_TYPES = {"image/jpeg", "image/png", "image/webp"}
_ACCEPTED_DEPTH_TYPES = {"image/png", "application/octet-stream"}


class GraspAgentExecutor(AgentExecutor):
    """A2A executor: Camera_Car RGBD FileParts + object DataPart -> grasp pose JSON."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            parts = context.message.parts
            if len(parts) < 3:
                await event_queue.enqueue_event(build_error("Expected DataPart metadata plus RGB and depth FileParts.", context))
                return

            body = _extract_metadata(parts[0])
            object_id = str(body.get("object_id", "") or "").strip()
            camera_name = str(body.get("camera_name", "Camera_Car") or "Camera_Car").strip()
            config_path = Path(body.get("config_path", str(DEFAULT_CONFIG)))
            if not object_id:
                await event_queue.enqueue_event(build_error("Missing 'object_id' in request metadata.", context))
                return

            rgb_bytes = None
            depth_bytes = None
            for part in parts[1:]:
                semantic = _semantic(part)
                if semantic == "rgb":
                    rgb_bytes = _extract_file_bytes(part, accepted_mime_types=_ACCEPTED_RGB_TYPES)
                elif semantic == "depth":
                    depth_bytes = _extract_file_bytes(part, accepted_mime_types=_ACCEPTED_DEPTH_TYPES)
            if rgb_bytes is None or depth_bytes is None:
                await event_queue.enqueue_event(build_error("RGB and depth FileParts with semantic metadata are required.", context))
                return

            from grasp_agent.pipeline.pipeline import run_pipeline

            result = run_pipeline(
                config_path=config_path,
                object_id=object_id,
                camera_name=camera_name,
                rgb_bytes=rgb_bytes,
                depth_bytes=depth_bytes,
            )
            logger.info(
                "GraspAgent complete. object_id=%s, grasp_confidence=%.4f, num_valid_grasps=%d",
                object_id,
                float(result["grasp_confidence"]),
                int(result.get("num_valid_grasps", 0)),
            )
            await event_queue.enqueue_event(build_success(result, context, name="grasp_result"))
        except Exception as exc:
            logger.error("GraspAgent execution error: %s", exc, exc_info=True)
            await event_queue.enqueue_event(build_error(str(exc), context))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError("Cancellation not supported")


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


def _semantic(part: Any) -> str:
    root = getattr(part, "root", part)
    metadata = getattr(root, "metadata", None) or {}
    return str(metadata.get("semantic", "") or "").strip().lower()


def _extract_file_bytes(part: Any, *, accepted_mime_types: set[str]) -> bytes:
    root = getattr(part, "root", part)
    if getattr(root, "kind", "") != "file" and not hasattr(root, "file"):
        raise ValueError("RGBD payloads must be A2A FilePart values, not text/base64 fields.")
    file_value = getattr(root, "file", None)
    mime_type = str(getattr(file_value, "mime_type", "") or "")
    if mime_type and mime_type not in accepted_mime_types:
        raise ValueError(f"Unsupported media type: {mime_type}")
    raw = getattr(file_value, "bytes", "") if file_value is not None else ""
    if not raw:
        raise ValueError("FilePart must contain inline bytes for this service.")
    return base64.b64decode(raw)
