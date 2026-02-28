"""
agents/find_agent.py — Find Agent Node (YOLO-based Object Detection)

Role in the system:
  - LangGraph Node: called once after input_node
  - A2A Client: sends multi-camera images to INF_FIND_URL on RTX 3090
                for YOLO detection + bbox annotation

Server (3090) responsibilities:
  - Run YOLO on each image
  - Draw globally-numbered bounding boxes (1, 2, 3...) on each image
  - Return: annotated images (base64) + detection metadata per number

Commander responsibilities:
  - Save annotated images to logs/find_candidates/
  - Show file path to user for review
  - Wait for user to pick a number or type "no"
"""

import asyncio
import json
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

_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_cameras() -> List[Dict[str, str]]:
    """Load camera list from config/cameras.yaml."""
    with open(_CONFIG_DIR / "cameras.yaml") as f:
        return yaml.safe_load(f).get("cameras", [])


def _load_objects() -> List[Dict[str, str]]:
    """Load graspable object list from config/objects.yaml."""
    with open(_CONFIG_DIR / "objects.yaml") as f:
        return yaml.safe_load(f).get("graspable_objects", [])


class FindAgent:
    """
    Find Agent: sends all camera images to YOLO inference server on RTX 3090.

    A2A Server (3090) does:
      1. YOLO detection on each image
      2. Draw globally-numbered bounding boxes (1, 2, 3...) on each image
      3. Return annotated images (base64) + detection metadata

    Mock returns fake detections from config/objects.yaml with dummy images.

    execute() returns:
      {
        "result": {
          "yolo_detections": {
            1: {"camera": "Camera_Car", "bbox": [x1,y1,x2,y2],
                "label": "cup", "conf": 0.92,
                "annotated_image_base64": "..."},
            2: ...
          }
        },
        "success": bool
      }
    """

    AGENT_NAME = "Find Agent"

    def __init__(self, http_client: httpx.AsyncClient = None):
        self._http_client = http_client or httpx.AsyncClient(timeout=60.0)
        self._inf_url = os.getenv("INF_FIND_URL", "")
        self._use_mock = os.getenv("MOCK_MODE", "true").lower() == "true" or not self._inf_url

    async def execute(self, params: Dict[str, Any], context_id: str = "") -> Dict[str, Any]:
        if self._use_mock:
            return await self._mock_execute(params)
        return await self._a2a_execute(params, context_id)

    async def _mock_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mock: generate fake YOLO detections from config/objects.yaml.
        Returns a globally-numbered detection dict with dummy metadata (no real images).
        """
        await asyncio.sleep(0.5)
        objects = _load_objects()
        cameras = _load_cameras()
        cam_name = cameras[0]["name"] if cameras else "Camera_Car"

        yolo_detections: Dict[int, Dict[str, Any]] = {}
        for i, obj in enumerate(objects, start=1):
            yolo_detections[i] = {
                "camera": cam_name,
                "bbox": [100, 100, 300, 300],       # dummy bbox
                "label": obj["label"],
                "conf": 0.9,
                "annotated_image_base64": None,     # no real image in mock
            }

        logger.info(f"[{self.AGENT_NAME}] Mock: {len(yolo_detections)} detection(s) generated.")
        return {"result": {"yolo_detections": yolo_detections}, "success": True}

    async def _a2a_execute(self, params: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        """
        Real mode:
          1. Capture images from all cameras via camera.py (Python 3.10 ROS subprocess).
          2. Bundle as {camera_name: base64_string} and send to Find A2A Server on 3090.
          3. Server runs YOLO, annotates images, returns yolo_detections dict.
        """
        from commander.camera import get_camera_image_base64

        cameras = _load_cameras()

        # -- Step 1: Capture images from all cameras -------------------------
        camera_images: Dict[str, str] = {}   # {camera_name: base64}
        for cam in cameras:
            logger.info(f"[{self.AGENT_NAME}] Capturing image from {cam['name']} ...")
            b64 = await get_camera_image_base64(cam["name"], timeout_sec=15.0)
            if b64:
                camera_images[cam["name"]] = b64
                logger.info(f"[{self.AGENT_NAME}] ✅ {cam['name']} captured ({len(b64)//1024} KB).")
            else:
                logger.warning(f"[{self.AGENT_NAME}] ⚠ {cam['name']} returned no image, skipping.")

        if not camera_images:
            logger.error(f"[{self.AGENT_NAME}] No images captured from any camera.")
            return {"result": {"yolo_detections": {}}, "success": False}

        # -- Step 2: Send to 3090 Find A2A Server ----------------------------
        try:
            resolver = A2ACardResolver(httpx_client=self._http_client, base_url=self._inf_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=self._http_client, agent_card=agent_card)

            # Payload: see 3090server/VLM_RL/find_agent/agent_executor.py for schema
            request_body = {
                "camera_images": camera_images,
                "task_description": params.get("task_description", ""),
                "target_object": params.get("target_object", {}),
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
                f"[{self.AGENT_NAME}] Sending {len(camera_images)} image(s) to {self._inf_url}"
            )
            response = await client.send_message(request)
            return {"result": self._parse_response(response), "success": True}

        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] A2A call failed: {e}")
            return {"result": {"yolo_detections": {}}, "success": False}

    @staticmethod
    def _parse_response(response: Any) -> Dict[str, Any]:
        """Parse YOLO detections from A2A artifact response."""
        try:
            result = response.root.result
            if hasattr(result, "artifacts") and result.artifacts:
                parts = result.artifacts[0].parts
                if parts:
                    return json.loads(parts[0].root.text)
        except Exception:
            pass
        return {"yolo_detections": {}}
