"""
agent_executor.py

Accepts camera + bbox and returns full 3D coordinates.
Currently returns dummy data.
"""

import json
import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import UnsupportedOperationError

from a2a_utils.response import build_error, build_success

logger = logging.getLogger(__name__)

class GetItemInfoExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            parts = context.message.parts
            if not parts:
                await event_queue.enqueue_event(build_error("Empty request"))
                return

            body = json.loads(parts[0].root.text)
            camera = body.get("camera", "Unknown")
            bbox = body.get("bbox", [])
            label = body.get("label", "Unknown")
            det_id = body.get("detection_id", 0)

            logger.info(f"Retrieving 3D info for item '{label}' (ID:{det_id}) from {camera}")
            
            # --- 3D Estimation Logic goes here (Point Cloud / Depth Camera projection) ---
            # Dummy logic for now
            pos_3d = [0.8, -0.2, 0.4] 
            size_est = [0.1, 0.1, 0.1]

            result_data = {
                "label": label,
                "detection_id": det_id,
                "camera": camera,
                "bbox": bbox,
                "position_3d": pos_3d,
                "size_estimate": size_est,
            }

            logger.info("Return 3D estimation result.")
            await event_queue.enqueue_event(build_success(result_data))

        except Exception as e:
            logger.error(f"GetItemInfo execution error: {e}", exc_info=True)
            await event_queue.enqueue_event(build_error(str(e)))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError()
