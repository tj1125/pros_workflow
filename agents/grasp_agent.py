"""
grasp_agent.py — GraspGen Agent Node

Role in the system:
  - LangGraph Node: called by the Orchestrator via Conditional Edge
  - A2A Client: sends inference request to the Inference GraspGen server on RTX 3090

Returns 6-DoF grasp pose data via the A2A artifact response.
"""

import asyncio
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
        1. Receive module_params (object_id, scene context) from CommanderState
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
        logger.info(f"[{self.AGENT_NAME}] Mock: generating grasp for {obj_id}")
        return {
            "result": (
                f"[MockGrasp] 6-DoF pose for '{obj_id}': "
                "position=[0.3, 0.1, 0.5], quaternion=[0, 0, 0, 1]"
            ),
            "success": True,
        }

    async def _a2a_execute(
        self, params: Dict[str, Any], context_id: str
    ) -> Dict[str, Any]:
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

            payload = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": f"Generate grasp pose with params: {params}",
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
            result_text = self._parse_response(response)
            return {"result": result_text, "success": True}

        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] A2A call failed: {e}")
            return {"result": f"Error: {e}", "success": False}

    @staticmethod
    def _parse_response(response: Any) -> str:
        try:
            result = response.root.result
            if hasattr(result, "artifacts") and result.artifacts:
                parts = result.artifacts[0].parts
                if parts:
                    return parts[0].root.text
        except Exception:
            pass
        return "No grasp result data"
