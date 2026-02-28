"""
agents/get_item_info_agent.py — Get Item Info Agent

Role in the system:
  - LangGraph Node: called after human confirms the target detection ID
  - A2A Client: sends camera + bbox to INF_GET_ITEM_INFO_URL on RTX 3090
                for detailed 3D pose, size, and label estimation

Server (3090) responsibilities:
  - Receive camera name and bounding box
  - Run depth estimation / point cloud projection
  - Return full 3D world position, orientation, and bounding box size

Commander stores the result in state["target_object"] for use throughout the session.
"""

import asyncio
import logging
import os
import uuid
from typing import Any, Dict

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

logger = logging.getLogger(__name__)


class GetItemInfoAgent:
    """
    Get Item Info Agent: retrieves full 3D information about the selected target object.

    execute() takes params:
      {
        "camera": "Camera_Car",
        "bbox": [x1, y1, x2, y2],
        "label": "cup",
        "detection_id": 1
      }

    Returns:
      {
        "result": {
          "label": "cup",
          "detection_id": 1,
          "camera": "Camera_Car",
          "bbox": [...],
          "position_3d": [x, y, z],     # world coordinates in meters
          "size_estimate": [w, h, d],   # approximate size in meters
        },
        "success": bool
      }
    """

    AGENT_NAME = "GetItemInfo Agent"

    def __init__(self, http_client: httpx.AsyncClient = None):
        self._http_client = http_client or httpx.AsyncClient(timeout=60.0)
        self._inf_url = os.getenv("INF_GET_ITEM_INFO_URL", "")
        self._use_mock = os.getenv("MOCK_MODE", "true").lower() == "true" or not self._inf_url

    async def execute(self, params: Dict[str, Any], context_id: str = "") -> Dict[str, Any]:
        if self._use_mock:
            return await self._mock_execute(params)
        return await self._a2a_execute(params, context_id)

    async def _mock_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock: return dummy 3D info for testing without GPU server."""
        await asyncio.sleep(0.5)
        label = params.get("label", "unknown")
        det_id = params.get("detection_id", 0)
        camera = params.get("camera", "Camera_Car")
        bbox = params.get("bbox", [0, 0, 0, 0])

        result = {
            "label": label,
            "detection_id": det_id,
            "camera": camera,
            "bbox": bbox,
            "position_3d": [1.0, 0.0, 0.5],    # dummy world position (x, y, z)
            "size_estimate": [0.08, 0.12, 0.08], # dummy size (w, h, d) in meters
        }
        logger.info(f"[{self.AGENT_NAME}] Mock: retrieved info for '{label}' (ID={det_id}).")
        return {"result": result, "success": True}

    async def _a2a_execute(self, params: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Real: re-capture image and send camera+bbox to RTX 3090 for 3D estimation."""
        from commander.camera import get_camera_image_base64

        camera = params.get("camera", "Camera_Car")
        b64 = await get_camera_image_base64(camera, timeout_sec=15.0)
        if not b64:
            logger.error(f"[{self.AGENT_NAME}] Failed to capture image from {camera}.")
            return {"result": {}, "success": False}

        try:
            import json
            resolver = A2ACardResolver(httpx_client=self._http_client, base_url=self._inf_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=self._http_client, agent_card=agent_card)

            payload = {
                "message": {
                    "role": "user",
                    "parts": [{
                        "kind": "text",
                        "text": json.dumps({
                            "camera": camera,
                            "bbox": params.get("bbox"),
                            "image_base64": b64,
                            "detection_id": params.get("detection_id"),
                        })
                    }],
                    "message_id": uuid.uuid4().hex,
                    "context_id": context_id,
                }
            }
            request = SendMessageRequest(
                id=str(uuid.uuid4()),
                params=MessageSendParams(**payload),
            )

            logger.info(f"[{self.AGENT_NAME}] Sending A2A request to {self._inf_url}")
            response = await client.send_message(request)
            return {"result": self._parse_response(response), "success": True}

        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] A2A call failed: {e}")
            return {"result": {}, "success": False}

    @staticmethod
    def _parse_response(response: Any) -> Dict[str, Any]:
        """Parse 3D info from A2A artifact response."""
        import json
        try:
            result = response.root.result
            if hasattr(result, "artifacts") and result.artifacts:
                parts = result.artifacts[0].parts
                if parts:
                    return json.loads(parts[0].root.text)
        except Exception:
            pass
        return {}
