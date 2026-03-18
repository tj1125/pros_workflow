from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import UnsupportedOperationError

from a2a_utils.response import build_error, build_success
from grasp_agent.pipeline.constants import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class GraspAgentExecutor(AgentExecutor):
    """A2A executor: Camera_Car RGBD + object_id -> best grasp pose JSON."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            parts = context.message.parts
            if not parts:
                await event_queue.enqueue_event(build_error("Empty request"))
                return

            body = json.loads(parts[0].root.text)
            object_id = str(body.get("object_id", "")).strip()
            camera_name = str(body.get("camera_name", "Camera_Car")).strip() or "Camera_Car"
            rgb_base64 = body.get("rgb_base64", "")
            depth_base64 = body.get("depth_base64", "")
            config_path = Path(body.get("config_path", str(DEFAULT_CONFIG)))

            if not object_id:
                await event_queue.enqueue_event(build_error("Missing 'object_id' in request body."))
                return
            if not rgb_base64 or not depth_base64:
                await event_queue.enqueue_event(build_error("Both rgb_base64 and depth_base64 are required."))
                return

            from grasp_agent.pipeline.pipeline import run_pipeline

            result = run_pipeline(
                config_path=config_path,
                object_id=object_id,
                camera_name=camera_name,
                rgb_bytes=base64.b64decode(rgb_base64),
                depth_bytes=base64.b64decode(depth_base64),
            )
            logger.info(
                "GraspAgent complete. object_id=%s, grasp_confidence=%.4f",
                object_id,
                float(result["grasp_confidence"]),
            )
            await event_queue.enqueue_event(build_success(result))

        except Exception as exc:
            logger.error("GraspAgent execution error: %s", exc, exc_info=True)
            await event_queue.enqueue_event(build_error(str(exc)))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError("Cancellation not supported")
