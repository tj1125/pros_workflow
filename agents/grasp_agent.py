"""
grasp_agent.py — GraspGen Agent Node

Role in the system:
  - LangGraph Node: called by the Orchestrator via Conditional Edge
  - A2A Client: sends inference request to the Inference GraspGen server on RTX 3090

Returns 6-DoF grasp pose data via the A2A artifact response.
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

from .schemas import AgentResponseFormat

logger = logging.getLogger(__name__)


class GraspAgent:
    """
    GraspGen Agent Node: generates 6-DoF grasp poses via GPU inference.

    Execution flow:
        1. Receive module_params (object_id, Camera_Car RGBD image, scene context)
           from CommanderState
        2. Send A2A HTTPS request to Inference GraspGen server (RTX 3090)
        3. Parse returned grasp pose and update CommanderState
    """

    AGENT_NAME = "GraspGen Agent"

    def __init__(self, http_client: httpx.AsyncClient = None):
        self._http_client = http_client or httpx.AsyncClient(timeout=30.0)
        self._inf_url = os.getenv("INF_GRASP_URL", "")
        self._use_mock = os.getenv("MOCK_MODE", "true").lower() == "true" or not self._inf_url

    async def execute(
        self, params: Dict[str, Any], context_id: str = ""
    ) -> Dict[str, Any]:
        if self._use_mock:
            return await self._mock_execute(params)
        return await self._a2a_execute(params, context_id)

    async def _mock_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.8)
        obj_id = params.get("object_id", "unknown")
        logger.info(
            f"[{self.AGENT_NAME}] Mock: generating grasp for {obj_id} "
            f"(rgb={bool(params.get('rgb_base64'))}, depth={bool(params.get('depth_base64'))})"
        )
        mock_result = {
            "object_id": obj_id,
            "camera_name": params.get("camera_name", "Camera_Car"),
            "detection_confidence": 0.95,
            "grasp_confidence": 0.91,
            "num_candidate_grasps": 1,
            "best_grasp_pose_camera": {
                "frame": "camera",
                "position": [0.30, 0.10, 0.50],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
        return {
            "result": mock_result,
            "success": True,
        }

    async def _a2a_execute(
        self, params: Dict[str, Any], context_id: str
    ) -> Dict[str, Any]:
        object_id = params.get("object_id", "unknown")
        rgb_base64 = params.get("rgb_base64", "")
        depth_base64 = params.get("depth_base64", "")
        camera_name = params.get("camera_name", "Camera_Car")

        if not object_id or not rgb_base64 or not depth_base64:
            missing = [
                key
                for key, value in (
                    ("object_id", object_id),
                    ("rgb_base64", rgb_base64),
                    ("depth_base64", depth_base64),
                )
                if not value
            ]
            logger.error(f"[{self.AGENT_NAME}] Missing required grasp params: {missing}")
            return {
                "result": {"error": f"missing required params {missing}"},
                "success": False,
            }

        try:
            resolver = A2ACardResolver(
                httpx_client=self._http_client,
                base_url=self._inf_url,
            )
            agent_card = await resolver.get_agent_card()
            client = A2AClient(
                httpx_client=self._http_client,
                agent_card=agent_card,
            )

            request_body = {
                "object_id": object_id,
                "camera_name": camera_name,
                "rgb_base64": rgb_base64,
                "depth_base64": depth_base64,
            }
            payload = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": json.dumps(request_body),
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

            logger.info(
                f"[{self.AGENT_NAME}] Sending Camera_Car RGBD grasp request "
                f"for object_id={object_id} to {self._inf_url}"
            )
            response = await client.send_message(request)
            result_payload = self._parse_response(response)
            success = "error" not in result_payload
            return {"result": result_payload, "success": success}

        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] A2A call failed: {e}")
            return {"result": {"error": str(e)}, "success": False}

    @staticmethod
    def _parse_response(response: Any) -> Dict[str, Any]:
        try:
            parts = response.root.result.parts
            if parts:
                return json.loads(parts[0].root.text)
        except Exception as exc:
            logger.error(f"[{GraspAgent.AGENT_NAME}] Failed to parse response: {exc}")
        return {"error": "No grasp result data"}
