"""
agents/find_agent.py — Find Agent Node

Role in the system:
  - LangGraph Node: called by the Orchestrator once at the start of each task
  - A2A Client: sends camera images to the Inference Find server on RTX 3090
                for object detection and 3D localization

Returns a list of candidate objects found in the scene.
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List

import httpx
import yaml
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

logger = logging.getLogger(__name__)

# Path to config directory relative to this file's package root
_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_cameras() -> List[Dict[str, str]]:
    """Load camera list from config/cameras.yaml."""
    path = _CONFIG_DIR / "cameras.yaml"
    with open(path) as f:
        return yaml.safe_load(f).get("cameras", [])


def _load_objects() -> List[Dict[str, str]]:
    """Load graspable object list from config/objects.yaml."""
    path = _CONFIG_DIR / "objects.yaml"
    with open(path) as f:
        return yaml.safe_load(f).get("graspable_objects", [])


class FindAgent:
    """
    Find Agent Node: locates candidate target objects using multi-camera images.

    Execution flow:
        1. Read all camera names from config/cameras.yaml
        2. Capture images from each camera via ROS (camera.py)
        3. Send images + task description to A2A Find Inference Server (RTX 3090: 8005)
        4. Parse and return detected candidate objects with 3D positions

    Mock mode returns all objects from config/objects.yaml with dummy positions.
    """

    AGENT_NAME = "Find Agent"

    def __init__(self, http_client: httpx.AsyncClient = None):
        self._http_client = http_client or httpx.AsyncClient(timeout=60.0)
        self._inf_url = os.getenv("INF_FIND_URL", "")
        self._use_mock = os.getenv("MOCK_MODE", "true").lower() == "true" or not self._inf_url

    async def execute(
        self, params: Dict[str, Any], context_id: str = ""
    ) -> Dict[str, Any]:
        if self._use_mock:
            return await self._mock_execute(params)
        return await self._a2a_execute(params, context_id)

    async def _mock_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return all known objects with dummy 3D positions for local testing."""
        await asyncio.sleep(0.5)
        objects = _load_objects()
        candidates = [
            {
                "id": obj["id"],
                "label": obj["label"],
                "position_3d": [0.0, 0.0, 0.0],  # placeholder
                "camera": "Camera_Car",
            }
            for obj in objects
        ]
        logger.info(f"[{self.AGENT_NAME}] Mock: found {len(candidates)} candidate objects.")
        return {"result": candidates, "success": True}

    async def _a2a_execute(
        self, params: Dict[str, Any], context_id: str
    ) -> Dict[str, Any]:
        """
        Capture images from all cameras, then send to Find inference server via A2A.
        """
        from commander.camera import get_camera_image_base64

        task_desc = params.get("task_description", "")
        cameras = _load_cameras()

        # Collect images from each camera
        image_payloads = []
        for cam in cameras:
            b64 = await get_camera_image_base64(cam["name"], timeout_sec=15.0)
            if b64:
                image_payloads.append({"camera": cam["name"], "image_base64": b64})
                logger.info(f"[{self.AGENT_NAME}] Image captured from {cam['name']}.")
            else:
                logger.warning(f"[{self.AGENT_NAME}] Failed to capture from {cam['name']}, skipping.")

        if not image_payloads:
            return {"result": [], "success": False}

        try:
            resolver = A2ACardResolver(
                httpx_client=self._http_client, base_url=self._inf_url
            )
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=self._http_client, agent_card=agent_card)

            payload = {
                "message": {
                    "role": "user",
                    "parts": [
                        {
                            "kind": "text",
                            "text": (
                                f"Task: {task_desc}\n"
                                f"Find all graspable objects and return their 3D positions.\n"
                                f"Camera images are provided as base64."
                            ),
                        }
                    ]
                    + [
                        {"kind": "text", "text": f"camera:{p['camera']}", "metadata": {"image_base64": p["image_base64"]}}
                        for p in image_payloads
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
            result = self._parse_response(response)
            return {"result": result, "success": True}

        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] A2A call failed: {e}")
            return {"result": [], "success": False}

    @staticmethod
    def _parse_response(response: Any) -> List[Dict[str, Any]]:
        """Extract candidate object list from A2A response."""
        try:
            result = response.root.result
            if hasattr(result, "artifacts") and result.artifacts:
                parts = result.artifacts[0].parts
                if parts:
                    import json
                    return json.loads(parts[0].root.text)
        except Exception:
            pass
        return []
