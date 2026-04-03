"""
agent_executor.py — A2A Executor for GetItemInfoAgentNoSam3D.

Expected request:
  - parts[0]: JSON text
      {
        "yolo_class": "doll",
        "scene_config": "...optional...",
        "selected_camera": "camera_room1_12",
        "camera_names": ["camera_room1_12", "camera_room1_13", ...],
        "world_position_data": {
          "data": "{\"0\": [...], \"1\": [...], ...}"
        }
      }
  - parts[1...]: image bytes/base64 in the same order as camera_names
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
from get_item_info_agent_no_sam3d.pipeline.constants import DEFAULT_SCENE_CONFIG
from tool.runtime.memory import release_cuda_memory

logger = logging.getLogger(__name__)


class GetItemInfoNoSam3dExecutor(AgentExecutor):
    """A2A executor: multi-view RGB images + /world_position_data -> goal_pose JSON."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            parts = context.message.parts
            if len(parts) < 2:
                await event_queue.enqueue_event(
                    build_error("Expected JSON config plus at least 1 camera image part.")
                )
                return

            body = json.loads(parts[0].root.text)
            yolo_class: str = body.get("yolo_class", "")
            scene_config_path = Path(body.get("scene_config", str(DEFAULT_SCENE_CONFIG)))
            selected_camera = str(body.get("selected_camera", "")).strip()
            world_position_data = body.get("world_position_data")
            debug_save = bool(body.get("debug_save", False))

            if not yolo_class:
                await event_queue.enqueue_event(build_error("Missing 'yolo_class' in request body."))
                return
            if world_position_data is None:
                await event_queue.enqueue_event(build_error("Missing 'world_position_data' in request body."))
                return

            camera_names = body.get("camera_names") or []
            camera_names = [str(camera_name).strip() for camera_name in camera_names if str(camera_name).strip()]
            if not camera_names:
                camera_names = [f"camera_{idx}" for idx in range(1, len(parts))]

            if len(camera_names) != len(parts) - 1:
                await event_queue.enqueue_event(
                    build_error("camera_names length must match the number of uploaded images.")
                )
                return

            logger.info(
                "GetItemInfoNoSam3D: class=%s, config=%s, selected_camera=%s, uploaded=%s",
                yolo_class,
                scene_config_path,
                selected_camera or "N/A",
                camera_names,
            )

            camera_images = {
                camera_name: _extract_bytes(parts[idx])
                for idx, camera_name in enumerate(camera_names, start=1)
            }

            with tempfile.TemporaryDirectory(prefix="get_item_info_no_sam3d_") as tmp_dir:
                tmp = Path(tmp_dir)
                image_paths_by_camera: dict[str, Path] = {}
                for camera_name, image_bytes in camera_images.items():
                    image_path = tmp / f"{_safe_camera_name(camera_name)}.png"
                    image_path.write_bytes(image_bytes)
                    image_paths_by_camera[camera_name] = image_path

                goal_output = tmp / "goal_pose.json"

                from get_item_info_agent_no_sam3d.pipeline.pipeline import run_pipeline

                result = run_pipeline(
                    scene_config=scene_config_path,
                    yolo_class_name=yolo_class,
                    goal_output=goal_output,
                    debug_save=debug_save,
                    image_paths_by_camera=image_paths_by_camera,
                    primary_camera_id=selected_camera or None,
                    world_position_data=world_position_data,
                )

            logger.info(
                "GetItemInfoNoSam3D: pipeline complete. center_world=%s",
                result.get("center_world"),
            )
            await event_queue.enqueue_event(build_success(result))

        except Exception as exc:
            logger.error("GetItemInfoNoSam3D execution error: %s", exc, exc_info=True)
            await event_queue.enqueue_event(build_error(str(exc)))
        finally:
            try:
                release_cuda_memory()
            except Exception as cleanup_exc:
                logger.warning("GetItemInfoNoSam3D cleanup failed: %s", cleanup_exc)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError()


def _extract_bytes(part) -> bytes:
    root = part.root
    if hasattr(root, "inline_data") and root.inline_data is not None:
        data = root.inline_data.data
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        return base64.b64decode(data)
    if hasattr(root, "text") and root.text:
        return base64.b64decode(root.text)
    raise ValueError(f"Cannot extract bytes from A2A part: {root!r}")


def _safe_camera_name(camera_name: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_")) else "_" for ch in camera_name)

