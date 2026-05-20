"""
agents/get_item_info_agent.py — Get Item Info Agent

Role in the system:
  - LangGraph Node: called after human confirms the target detection ID
  - A2A Client: sends all available fixed-room RGB images to
                INF_GET_ITEM_INFO_URL on RTX 3090 for detailed 3D pose,
                size, and label estimation

Server (3090) responsibilities:
  - Receive the preferred camera plus all uploaded RGB images
  - Choose a usable multi-view / stereo subset from the upload
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
        "yolo_class": "doll",  # one of: apple, box, coffee, cup, doll, gaobear, hpb, xbox
        "selected_camera": "Camera_Room1_12",
        "camera_names": ["Camera_Room1_12", "Camera_Room1_13", "Camera_Room1_14", "Camera_Room1_15"],
        "camera_images": {"Camera_Room1_12": "...", ...},
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
        """Real: send all available room-camera RGB images to RTX 3090."""
        import json

        yolo_class = params.get("yolo_class", "unknown")
        selected_camera = str(params.get("selected_camera", "")).strip()
        camera_images = params.get("camera_images", {}) or {}
        ordered_camera_names = params.get("camera_names") or list(camera_images.keys())
        ordered_camera_names = [str(camera_name).strip() for camera_name in ordered_camera_names if str(camera_name).strip()]

        if len(ordered_camera_names) < 2:
            logger.error(f"[{self.AGENT_NAME}] At least two camera images are required.")
            return {"result": {}, "success": False}

        missing_cameras = [camera_name for camera_name in ordered_camera_names if not camera_images.get(camera_name)]
        if missing_cameras:
            logger.error(f"[{self.AGENT_NAME}] Missing camera images for: {missing_cameras}")
            return {"result": {}, "success": False}

        try:
            resolver = A2ACardResolver(httpx_client=self._http_client, base_url=self._inf_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=self._http_client, agent_card=agent_card)

            payload = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": json.dumps({
                                "yolo_class": yolo_class,
                                "selected_camera": selected_camera,
                                "camera_names": ordered_camera_names,
                            })
                        }
                    ] + [
                        {
                            "kind": "text",
                            "text": camera_images[camera_name],
                        }
                        for camera_name in ordered_camera_names
                    ],
                    "message_id": uuid.uuid4().hex,
                    "context_id": context_id,
                }
            }
            request = SendMessageRequest(
                id=str(uuid.uuid4()),
                params=MessageSendParams(**payload),
            )

            logger.info(
                f"[{self.AGENT_NAME}] Sending {len(ordered_camera_names)} camera image(s) "
                f"to {self._inf_url} (selected={selected_camera or 'N/A'})"
            )
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
