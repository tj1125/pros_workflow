"""
agent_executor.py — A2A Executor for GetItemInfoAgent.

Accepts a multi-part A2A message:
  - parts[0]: JSON text   → { "yolo_class": "<class_name>", "scene_config": "<path>" }
  - parts[1]: image bytes → camera A image (PNG/JPG)
  - parts[2]: image bytes → camera B image (PNG/JPG)

Returns the goal_pose.json content as a JSON text response.
"""

from __future__ import annotations

import base64
import json
import logging
import tempfile
from pathlib import Path

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import UnsupportedOperationError

from a2a_utils.response import build_error, build_success
from pipeline.constants import DEFAULT_SCENE_CONFIG

logger = logging.getLogger(__name__)


class GetItemInfoExecutor(AgentExecutor):
    """Full-pipeline A2A executor: stereo images → goal_pose JSON."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            parts = context.message.parts
            if len(parts) < 3:
                await event_queue.enqueue_event(
                    build_error("Expected 3 message parts: JSON config, image_a bytes, image_b bytes.")
                )
                return

            # --- Part 0: JSON config -----------------------------------------------
            body = json.loads(parts[0].root.text)
            yolo_class: str = body.get("yolo_class", "")
            scene_config_path = Path(body.get("scene_config", str(DEFAULT_SCENE_CONFIG)))

            if not yolo_class:
                await event_queue.enqueue_event(build_error("Missing 'yolo_class' in request body."))
                return

            logger.info("GetItemInfo: class=%s, config=%s", yolo_class, scene_config_path)

            # --- Parts 1 & 2: image bytes ------------------------------------------
            image_a_bytes = _extract_bytes(parts[1])
            image_b_bytes = _extract_bytes(parts[2])

            with tempfile.TemporaryDirectory(prefix="get_item_info_") as tmp_dir:
                tmp = Path(tmp_dir)

                image_a_path = tmp / "image_a.png"
                image_b_path = tmp / "image_b.png"
                image_a_path.write_bytes(image_a_bytes)
                image_b_path.write_bytes(image_b_bytes)

                goal_output = tmp / "goal_pose.json"

                # Late import keeps heavy model loading out of server startup.
                from pipeline.pipeline import run_pipeline

                result = run_pipeline(
                    scene_config=scene_config_path,
                    yolo_class_name=yolo_class,
                    image_a=image_a_path,
                    image_b=image_b_path,
                    goal_output=goal_output,
                    debug_save=False,
                )

            logger.info("GetItemInfo: pipeline complete. center_world=%s", result.get("center_world"))
            await event_queue.enqueue_event(build_success(result))

        except Exception as exc:
            logger.error("GetItemInfo execution error: %s", exc, exc_info=True)
            await event_queue.enqueue_event(build_error(str(exc)))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract_bytes(part) -> bytes:
    """Extract raw bytes from an A2A message part (inline data or base64 text)."""
    root = part.root
    # InlineDataPart (binary)
    if hasattr(root, "inline_data") and root.inline_data is not None:
        data = root.inline_data.data
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        # Some SDK versions wrap it as base64 string
        return base64.b64decode(data)
    # TextPart fallback: assume base64-encoded image
    if hasattr(root, "text") and root.text:
        return base64.b64decode(root.text)
    raise ValueError(f"Cannot extract bytes from A2A part: {root!r}")
