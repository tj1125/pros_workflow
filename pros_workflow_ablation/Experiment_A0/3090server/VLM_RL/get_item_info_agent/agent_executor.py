"""
agent_executor.py — A2A Executor for GetItemInfoAgent.

Accepts a multi-part A2A message:
  - parts[0]: JSON text   → {
        "yolo_class": "<class_name>",
        "scene_config": "<path>",
        "selected_camera": "Camera_Room1_1",
        "camera_names": ["Camera_Room1_1", "Camera_Room1_2", "Camera_Room1_3"]
    }
  - parts[1...]: image bytes/base64 text in the same order as camera_names

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
from get_item_info_agent.pipeline.constants import DEFAULT_SCENE_CONFIG
from tool.runtime.memory import release_cuda_memory

logger = logging.getLogger(__name__)


class GetItemInfoExecutor(AgentExecutor):
    """Full-pipeline A2A executor: multi-view RGB images → goal_pose JSON."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            parts = context.message.parts
            if len(parts) < 3:
                await event_queue.enqueue_event(
                    build_error("Expected JSON config plus at least 2 camera image parts.")
                )
                return

            # --- Part 0: JSON config -----------------------------------------------
            body = json.loads(parts[0].root.text)
            yolo_class: str = body.get("yolo_class", "")
            scene_config_path = Path(body.get("scene_config", str(DEFAULT_SCENE_CONFIG)))
            selected_camera = str(body.get("selected_camera", "")).strip()
            camera_names = body.get("camera_names") or []
            camera_names = [str(camera_name).strip() for camera_name in camera_names if str(camera_name).strip()]

            if not yolo_class:
                await event_queue.enqueue_event(build_error("Missing 'yolo_class' in request body."))
                return

            if not camera_names:
                camera_names = [f"camera_{idx}" for idx in range(1, len(parts))]

            if len(camera_names) != len(parts) - 1:
                await event_queue.enqueue_event(
                    build_error("camera_names length must match the number of uploaded images.")
                )
                return

            logger.info(
                "GetItemInfo: class=%s, config=%s, selected_camera=%s, uploaded=%s",
                yolo_class,
                scene_config_path,
                selected_camera or "N/A",
                camera_names,
            )

            # --- Parts 1...N: image bytes ------------------------------------------
            camera_images = {
                camera_name: _extract_bytes(parts[idx])
                for idx, camera_name in enumerate(camera_names, start=1)
            }

            with tempfile.TemporaryDirectory(prefix="get_item_info_") as tmp_dir:
                tmp = Path(tmp_dir)
                image_paths_by_camera: dict[str, Path] = {}
                for camera_name, image_bytes in camera_images.items():
                    image_path = tmp / f"{_safe_camera_name(camera_name)}.png"
                    image_path.write_bytes(image_bytes)
                    image_paths_by_camera[camera_name] = image_path

                goal_output = tmp / "goal_pose.json"

                # Late import keeps heavy model loading out of server startup.
                from get_item_info_agent.pipeline.pipeline import run_pipeline

                result = run_pipeline(
                    scene_config=scene_config_path,
                    yolo_class_name=yolo_class,
                    image_a=None,
                    image_b=None,
                    goal_output=goal_output,
                    debug_save=False,
                    image_paths_by_camera=image_paths_by_camera,
                    primary_camera_id=selected_camera or None,
                )

            logger.info("GetItemInfo: pipeline complete. center_world=%s", result.get("center_world"))
            await event_queue.enqueue_event(build_success(result))

        except Exception as exc:
            logger.error("GetItemInfo execution error: %s", exc, exc_info=True)
            await event_queue.enqueue_event(build_error(str(exc)))
        finally:
            try:
                release_cuda_memory()
            except Exception as cleanup_exc:
                logger.warning("GetItemInfo cleanup failed: %s", cleanup_exc)

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


def _safe_camera_name(camera_name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in camera_name)
