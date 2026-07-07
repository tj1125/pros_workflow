"""A2A client for RGBD grasp generation."""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any, Dict

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

from .a2a_adapter import data_part, extract_result_payload, file_part, require_agent_card_modes

logger = logging.getLogger(__name__)


class GraspAgent:
    """Generate feasible 6-DoF grasp poses via a remote A2A service."""

    AGENT_NAME = "GraspGen Agent"

    def __init__(self, http_client: httpx.AsyncClient = None):
        self._http_client = http_client or httpx.AsyncClient(timeout=30.0)
        self._inf_url = os.getenv("INF_GRASP_URL", "").strip()

    async def execute(self, params: Dict[str, Any], context_id: str = "") -> Dict[str, Any]:
        return await self._a2a_execute(params, context_id)

    async def _a2a_execute(self, params: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        object_id = str(params.get("object_id", "") or "").strip()
        rgb_base64 = params.get("rgb_base64", "")
        depth_base64 = params.get("depth_base64", "")
        camera_name = str(params.get("camera_name", "Camera_Car") or "Camera_Car")
        if not object_id or not rgb_base64 or not depth_base64:
            missing = [
                key
                for key, value in (("object_id", object_id), ("rgb", rgb_base64), ("depth", depth_base64))
                if not value
            ]
            return {"result": {"error": f"missing required params {missing}"}, "success": False, "a2a_task_id": ""}

        try:
            resolver = A2ACardResolver(httpx_client=self._http_client, base_url=self._inf_url)
            agent_card = await resolver.get_agent_card()
            require_agent_card_modes(agent_card, input_modes={"data", "file"}, output_modes={"data"}, skill_ids={"generate_grasp_pose"})
            client = A2AClient(httpx_client=self._http_client, agent_card=agent_card)

            metadata = {"object_id": object_id, "camera_name": camera_name}
            for key in ("target_center_world", "target_instance_key", "amcl_pose", "ros_map_origin_unity"):
                value = params.get(key)
                if value not in (None, "", [], {}):
                    metadata[key] = value
            parts = [
                data_part(metadata),
                file_part(name=f"{camera_name}.rgb.jpg", data_base64=rgb_base64, mime_type="image/jpeg", metadata={"camera_name": camera_name, "semantic": "rgb"}),
                file_part(name=f"{camera_name}.depth.png", data_base64=depth_base64, mime_type="image/png", metadata={"camera_name": camera_name, "semantic": "depth"}),
            ]
            payload = {
                "message": {
                    "role": "user",
                    "parts": parts,
                    "message_id": uuid.uuid4().hex,
                    "context_id": context_id,
                }
            }
            request = SendMessageRequest(id=str(uuid.uuid4()), params=MessageSendParams(**payload))
            logger.info("[%s] Sending RGBD FileParts for object_id=%s to %s", self.AGENT_NAME, object_id, self._inf_url)
            response = await client.send_message(request)
            result_payload, a2a_task_id = extract_result_payload(response)
            return {"result": result_payload, "success": "error" not in result_payload, "a2a_task_id": a2a_task_id}
        except Exception as exc:
            logger.error("[%s] A2A call failed: %s", self.AGENT_NAME, exc, exc_info=True)
            return {"result": {"error": str(exc)}, "success": False, "a2a_task_id": ""}
