"""
view_agent/agent_executor.py — A2A Executor for ViewAgent

Accepts a single-part A2A message:
  parts[0]: JSON text with the following schema:
    {
      "rgb_frames_b64"   : List[str],   # base64-encoded RGB images (≤3)
      "depth_frames_b64" : List[str],   # base64-encoded depth images (≤3)
      "joint_state"      : Dict[str, float],  # joint_name -> angle (rad)
      "history_action"   : List[float]  # last 6-DOF delta executed
    }

Returns a JSON text message:
    {
      "delta_joints" : List[float],  # 6-DOF joint displacement increments
      "confidence"   : float         # actor confidence ∈ [0, 1]
    }
"""

from __future__ import annotations

import base64
import json
import logging
from io import BytesIO
from typing import List

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import UnsupportedOperationError

from a2a_utils.response import build_error, build_success
from view_agent.policy_service import PolicyService

logger = logging.getLogger(__name__)


class ViewAgentExecutor(AgentExecutor):
    """
    AgentExecutor for the SAC-based View Agent.

    Parses A2A requests, decodes base64 images into PIL.Image objects,
    delegates to PolicyService for inference, and returns A2A artifacts.
    """

    def __init__(self):
        # PolicyService is lazy-loaded on first infer() call
        self._policy = PolicyService()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            parts = context.message.parts
            if not parts:
                await event_queue.enqueue_event(build_error("Empty request"))
                return

            # -- Parse JSON body -------------------------------------------------
            body = json.loads(parts[0].root.text)
            rgb_b64: List[str] = body.get("rgb_frames_b64", [])
            depth_b64: List[str] = body.get("depth_frames_b64", [])
            joint_state: dict = body.get("joint_state", {})
            history_action: list = body.get("history_action", [0.0] * 6)

            if not rgb_b64:
                await event_queue.enqueue_event(
                    build_error("No RGB frames provided in rgb_frames_b64")
                )
                return

            logger.info(
                f"ViewAgent: received {len(rgb_b64)} RGB frame(s), "
                f"{len(depth_b64)} depth frame(s)."
            )

            # -- Decode images ---------------------------------------------------
            rgb_images = [_b64_to_pil(b) for b in rgb_b64]
            depth_images = [_b64_to_pil(b) for b in depth_b64]

            # -- Build observation dict -----------------------------------------
            obs = {
                "rgb_frames": rgb_images,
                "depth_frames": depth_images,
                "joint_state": joint_state,
                "history_action": history_action,
            }

            # -- Policy inference -----------------------------------------------
            result = self._policy.infer(obs)
            logger.info(
                f"ViewAgent: delta_joints={result['delta_joints']}, "
                f"confidence={result['confidence']:.3f}"
            )
            await event_queue.enqueue_event(build_success(result))

        except Exception as e:
            logger.error(f"ViewAgent execution error: {e}", exc_info=True)
            await event_queue.enqueue_event(build_error(str(e)))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise UnsupportedOperationError("Cancellation not supported")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _b64_to_pil(b64_string: str):
    """Decode a base64-encoded image string to a PIL.Image."""
    from PIL import Image
    raw = base64.b64decode(b64_string)
    return Image.open(BytesIO(raw)).convert("RGB")
