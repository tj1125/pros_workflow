"""
nav_agent.py — Nav Agent Node

Role in the system:
  - LangGraph Node: called by the Orchestrator via Conditional Edge
  - A2A Client: sends inference request to the Inference NAV server on RTX 3090

When MOCK_MODE is enabled or INF_NAV_URL is not set, uses local mock logic.
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


class NavAgent:
    """
    Nav Agent Node: controls robot base movement via navigation inference.

    Execution flow:
        1. Receive module_params from CommanderState
        2. Send A2A HTTPS request to Inference NAV server (RTX 3090)
        3. Parse inference result and return updated state dict
    """

    AGENT_NAME = "Nav Agent"

    def __init__(self, http_client: httpx.AsyncClient = None):
        self._http_client = http_client or httpx.AsyncClient(timeout=30.0)
        self._inf_url = os.getenv("INF_NAV_URL", "")
        self._use_mock = os.getenv("MOCK_MODE", "true").lower() == "true" or not self._inf_url

    async def execute(
        self, params: Dict[str, Any], context_id: str = ""
    ) -> Dict[str, Any]:
        """
        Execute navigation inference.
        Returns a dict with 'result' (str) and 'success' (bool).
        """
        if self._use_mock:
            return await self._mock_execute(params)
        return await self._a2a_execute(params, context_id)

    async def _mock_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Local mock: simulate navigation command."""
        await asyncio.sleep(0.5)
        target = params.get("target_point", [0.0, 0.0, 0.0])
        logger.info(f"[{self.AGENT_NAME}] Mock: navigating to {target}")
        return {
            "result": f"[MockNAV] Robot navigated to position {target}",
            "success": True,
        }

    async def _a2a_execute(
        self, params: Dict[str, Any], context_id: str
    ) -> Dict[str, Any]:
        """Send A2A request to Inference NAV server on RTX 3090."""
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
                            "text": f"Navigate with params: {params}",
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

            # Parse artifact from A2A response
            result_text = self._parse_response(response)
            logger.info(f"[{self.AGENT_NAME}] A2A response: {result_text}")
            return {"result": result_text, "success": True}

        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] A2A call failed: {e}")
            return {"result": f"Error: {e}", "success": False}

    @staticmethod
    def _parse_response(response: Any) -> str:
        """Extract text from the first artifact part of the A2A response."""
        try:
            result = response.root.result
            if hasattr(result, "artifacts") and result.artifacts:
                parts = result.artifacts[0].parts
                if parts:
                    return parts[0].root.text
        except Exception:
            pass
        return "No result data"
