"""A2A client for RGBD grasp generation."""

from __future__ import annotations

import asyncio
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

    def __init__(self, http_client: httpx.AsyncClient = None, use_mock: bool | None = None):
        self._http_client = http_client or httpx.AsyncClient(timeout=30.0)
        self._inf_url = os.getenv("INF_GRASP_URL", "").strip()
        self._use_mock = bool(use_mock) if use_mock is not None else (os.getenv("MOCK_MODE", "true").lower() == "true" or not self._inf_url)

    async def execute(self, params: Dict[str, Any], context_id: str = "") -> Dict[str, Any]:
        if self._use_mock:
            return await self._mock_execute(params)
        return await self._a2a_execute(params, context_id)

    async def _mock_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.8)
        obj_id = params.get("object_id", "unknown")
        mock_result = {
            "object_id": obj_id,
            "camera_name": params.get("camera_name", "Camera_Car"),
            "detection_confidence": 0.95,
            "grasp_confidence": 0.91,
            "num_candidate_grasps": 1,
            "num_valid_grasps": 1,
            "best_grasp_pose_camera": {
                "frame": "camera",
                "position": [0.30, 0.10, 0.50],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "valid_grasp_poses_camera": [
                {
                    "rank": 1,
                    "frame": "camera",
                    "position": [0.30, 0.10, 0.50],
                    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "grasp_confidence": 0.91,
                    "grasp_distance_to_gripper_midpoint_m": 0.08,
                    "grasp_distance_to_camera_m": 0.59,
                }
            ],
        }
        return {"result": mock_result, "success": True, "a2a_task_id": ""}

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

            parts = [
                data_part({"object_id": object_id, "camera_name": camera_name}),
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
