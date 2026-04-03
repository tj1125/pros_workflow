"""
agents/view_agent.py — View Agent Node (Active Viewpoint Adjustment)

Role in the system:
  - LangGraph Node: triggered when mild occlusion is detected between
    the gripper and the target object (resolvable via arm micro-adjustment).
  - A2A Client: sends current RGB frames, depth frames, joint state and
    action history to the View Agent inference server (RTX 3090, Port 9003)
    for SAC policy inference.

Server (3090) responsibilities:
  - Extract CLIP + DINOv2 features from 3 stacked RGB frames
  - Run SAC Actor to produce 6-DOF joint delta action
  - Return: delta_joints (6-DOF), confidence score

Commander responsibilities:
  - Capture latest 3 RGB + depth frames from camera
  - Send joint state and recent action history
  - Execute returned delta_joints via arm controller
"""

import asyncio
import base64
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

logger = logging.getLogger(__name__)


class ViewAgent:
    """
    View Agent Node: adjusts the robot arm/chassis pose to improve visual
    observation, resolving mild occlusion between gripper and target object.

    Execution flow:
        1. Receive params (rgb_frames_b64, depth_frames_b64, joint_state,
           history_action) from CommanderState
        2. Send A2A HTTPS request to View SAC server (RTX 3090, Port 9003)
        3. Parse returned 6-DOF joint delta action
        4. Return delta_joints for arm controller execution

    Mock mode: returns a zero delta action without any inference.
    """

    AGENT_NAME = "View Agent"
    # Number of temporal frames expected by the SAC policy (frame stack)
    FRAME_STACK = 3

    def __init__(self, http_client: Optional[httpx.AsyncClient] = None):
        self._http_client = http_client or httpx.AsyncClient(timeout=30.0)
        self._inf_url = os.getenv("INF_VIEW_URL", "")
        self._use_mock = (
            os.getenv("MOCK_MODE", "true").lower() == "true" or not self._inf_url
        )

    async def execute(
        self, params: Dict[str, Any], context_id: str = ""
    ) -> Dict[str, Any]:
        """
        Execute the View Agent.

        Expected keys in params:
            rgb_frames_b64   : List[str]  — base64-encoded RGB images (≤FRAME_STACK)
            depth_frames_b64 : List[str]  — base64-encoded depth images (≤FRAME_STACK)
            joint_state      : Dict       — current joint angles (rad) keyed by joint name
            history_action   : List[float]— last executed 6-DOF delta (for temporal context)

        Returns:
            {
              "result": {
                "delta_joints"  : [6 floats],   # 6-DOF joint displacement increments
                "confidence"    : float,         # SAC policy confidence ∈ [0, 1]
              },
              "success": bool
            }
        """
        if self._use_mock:
            return await self._mock_execute(params)
        return await self._a2a_execute(params, context_id)

    # ------------------------------------------------------------------
    # Mock execution
    # ------------------------------------------------------------------
    async def _mock_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return a dummy zero-delta action for testing without a 3090 server."""
        await asyncio.sleep(0.3)
        logger.info(f"[{self.AGENT_NAME}] Mock: returning zero delta_joints.")
        return {
            "result": {
                "delta_joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "confidence": 0.0,
            },
            "success": True,
        }

    # ------------------------------------------------------------------
    # Real A2A execution
    # ------------------------------------------------------------------
    async def _a2a_execute(
        self, params: Dict[str, Any], context_id: str
    ) -> Dict[str, Any]:
        """
        Real mode:
          1. Package rgb_frames, depth_frames, joint_state, history_action as JSON.
          2. Send to View A2A Server on RTX 3090 (Port 9003).
          3. Parse returned delta_joints.
        """
        try:
            resolver = A2ACardResolver(
                httpx_client=self._http_client,
                base_url=self._inf_url,
            )
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=self._http_client, agent_card=agent_card)

            # Payload: see 3090server/VLM_RL/view_agent/agent_executor.py for schema
            request_body = {
                "rgb_frames_b64": params.get("rgb_frames_b64", []),
                "depth_frames_b64": params.get("depth_frames_b64", []),
                "joint_state": params.get("joint_state", {}),
                "history_action": params.get("history_action", [0.0] * 6),
            }

            payload = {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": json.dumps(request_body)}],
                    "message_id": uuid.uuid4().hex,
                    "context_id": context_id,
                }
            }
            request = SendMessageRequest(
                id=str(uuid.uuid4()),
                params=MessageSendParams(**payload),
            )

            logger.info(
                f"[{self.AGENT_NAME}] Sending A2A request to {self._inf_url} "
                f"({len(request_body['rgb_frames_b64'])} RGB frames)"
            )
            response = await client.send_message(request)
            result = self._parse_response(response)
            return {"result": result, "success": True}

        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] A2A call failed: {e}")
            return {
                "result": {
                    "delta_joints": [0.0] * 6,
                    "confidence": 0.0,
                },
                "success": False,
            }

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(response: Any) -> Dict[str, Any]:
        """
        Parse delta_joints from A2A text message response.

        The 3090 server uses new_agent_text_message(), so the result is in
        response.root.result.parts[0].root.text as a JSON string.
        """
        try:
            parts = response.root.result.parts
            if parts:
                data = json.loads(parts[0].root.text)
                return {
                    "delta_joints": data.get("delta_joints", [0.0] * 6),
                    "confidence": float(data.get("confidence", 0.0)),
                }
        except Exception as e:
            logger.error(f"[View Agent] Failed to parse response: {e}")
        return {"delta_joints": [0.0] * 6, "confidence": 0.0}

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def image_path_to_b64(image_path: Path) -> str:
        """Convert an image file on disk to a base64-encoded string."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def build_params_from_camera(
        rgb_paths: List[Path],
        depth_paths: List[Path],
        joint_angles: Dict[str, float],
        history_action: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Convenience builder: convert image paths to base64 and assemble params dict.

        Args:
            rgb_paths      : List of up to FRAME_STACK RGB image paths (newest last)
            depth_paths    : List of up to FRAME_STACK depth image paths (newest last)
            joint_angles   : Dict of joint_name -> angle_rad
            history_action : Last 6-DOF action, defaults to zeros
        """
        rgb_b64 = [ViewAgent.image_path_to_b64(p) for p in rgb_paths[-ViewAgent.FRAME_STACK:]]
        depth_b64 = [ViewAgent.image_path_to_b64(p) for p in depth_paths[-ViewAgent.FRAME_STACK:]]
        return {
            "rgb_frames_b64": rgb_b64,
            "depth_frames_b64": depth_b64,
            "joint_state": joint_angles,
            "history_action": history_action or [0.0] * 6,
        }
