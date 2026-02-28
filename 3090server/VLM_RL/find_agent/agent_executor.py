"""
agent_executor.py — FindAgent's A2A Executor implementation
"""

import json
import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import UnsupportedOperationError

# Local imports
from a2a_utils.response import build_error_artifact, build_success_artifact
from .yolo_service import YoloService

logger = logging.getLogger(__name__)


class FindAgentExecutor(AgentExecutor):
    """
    AgentExecutor for YOLO-based object detection.
    
    Parses A2A requests, delegates to YoloService, and returns A2A artifacts.
    """

    def __init__(self):
        self.yolo_service = YoloService("yolo11n.pt")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            parts = context.message.parts
            if not parts:
                await event_queue.enqueue_event(**build_error_artifact("Empty request"))
                return

            # Commander's find_agent.py sends JSON in parts[0].text
            body = json.loads(parts[0].root.text)
            camera_images: dict = body.get("camera_images", {})
            task_desc: str = body.get("task_description", "")

            if not camera_images:
                await event_queue.enqueue_event(**build_error_artifact("No camera images provided"))
                return

            logger.info(f"Find request received for {len(camera_images)} images. Task: {task_desc}")

            # Run detection
            yolo_detections = self.yolo_service.detect_and_annotate(camera_images)

            # Return as standard A2A artifact
            logger.info(f"Returning {len(yolo_detections)} detections.")
            await event_queue.enqueue_event(**build_success_artifact({"yolo_detections": yolo_detections}))

        except Exception as e:
            logger.error(f"FindAgent execution error: {e}", exc_info=True)
            await event_queue.enqueue_event(**build_error_artifact(str(e)))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError("Cancellation not supported")
