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
        "yolo_class": "doll",  # or "apple", "wine"
      }

    Returns:
      {
        "result": {
          "center_world": [x, y, z],
          "group_ranking": [...],
          "goal_pose_path": "..."
        },
        "success": bool
      }
    """

    AGENT_NAME = "GetItemInfo Agent"

    def __init__(self, http_client: httpx.AsyncClient = None):
        self._http_client = http_client or httpx.AsyncClient(timeout=120.0)
        self._inf_url = os.getenv("INF_GET_ITEM_INFO_URL", "")
        self._use_mock = os.getenv("MOCK_MODE", "true").lower() == "true" or not self._inf_url

    async def execute(self, params: Dict[str, Any], context_id: str = "") -> Dict[str, Any]:
        if self._use_mock:
            return await self._mock_execute(params)
        return await self._a2a_execute(params, context_id)

    async def _mock_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock: return dummy 3D info for testing without GPU server."""
        await asyncio.sleep(0.5)
        yolo_class = params.get("yolo_class", "unknown")

        result = {
            "center_world": [1.0, 0.0, 0.5],    # dummy world position (x, y, z)
            "group_ranking": [],
            "goal_pose_path": "/tmp/mock_goal_pose.json",
        }
        logger.info(f"[{self.AGENT_NAME}] Mock: retrieved info for '{yolo_class}'.")
        return {"result": result, "success": True}

    async def _a2a_execute(self, params: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """Real: use passed stereo pair and send to RTX 3090 for full 3D perception pipeline."""
        import json

        yolo_class = params.get("yolo_class", "unknown")
        cam_a_b64 = params.get("cam_a_b64", "")
        cam_b_b64 = params.get("cam_b_b64", "")

        if not cam_a_b64 or not cam_b_b64:
            logger.error(f"[{self.AGENT_NAME}] Left or right stereo image is missing.")
            return {"result": {}, "success": False}

        try:
            resolver = A2ACardResolver(httpx_client=self._http_client, base_url=self._inf_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=self._http_client, agent_card=agent_card)

            payload = {
                "message": {
                    "role": "user",
                    "parts": [
                        { # Part 0: JSON setup
                            "kind": "text",
                            "text": json.dumps({
                                "yolo_class": yolo_class,
                            })
                        },
                        { # Part 1: Image A
                            "kind": "text", 
                            "text": cam_a_b64
                        },
                        { # Part 2: Image B
                            "kind": "text",
                            "text": cam_b_b64
                        }
                    ],
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
        import logging
        log = logging.getLogger(__name__)
        
        try:
            return json.loads(response.root.result.parts[0].root.text)
        except Exception as e:
            log.error(f"[_parse_response] Failed to parse A2A response: {e}", exc_info=True)
            return {}
