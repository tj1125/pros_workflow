"""
agents/get_item_info_agent_no_sam3d.py — Local A2A client for the no_sam3d item-info service.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

logger = logging.getLogger(__name__)


class GetItemInfoNoSam3DAgent:
    AGENT_NAME = "GetItemInfoNoSam3D Agent"

    def __init__(self, http_client: httpx.AsyncClient = None):
        self._http_client = http_client or httpx.AsyncClient(timeout=120.0)
        preferred_url = os.getenv("INF_GET_ITEM_INFO_NO_SAM3D_URL", "").strip()
        fallback_url = os.getenv("INF_GET_ITEM_INFO_URL", "").strip()
        self._inf_url = preferred_url or fallback_url
        if preferred_url:
            logger.info("[%s] Using INF_GET_ITEM_INFO_NO_SAM3D_URL=%s", self.AGENT_NAME, preferred_url)
        elif fallback_url:
            logger.info("[%s] Falling back to legacy INF_GET_ITEM_INFO_URL=%s", self.AGENT_NAME, fallback_url)
        self._use_mock = os.getenv("MOCK_MODE", "true").lower() == "true" or not self._inf_url

    async def execute(self, params: Dict[str, Any], context_id: str = "") -> Dict[str, Any]:
        if self._use_mock:
            return await self._mock_execute(params)
        return await self._a2a_execute(params, context_id)

    async def _mock_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.2)
        result = {
            "center_world": params.get("center_world", [1.0, 0.0, 0.5]),
            "center_world_coordinate_frame": "unity_world",
            "group_ranking": [],
            "goal_pose_path": "/tmp/mock_goal_pose.json",
            "primary_camera_id": params.get("selected_camera", ""),
            "target_instance_key": params.get("target_instance_key", ""),
            "target_topic_key": params.get("target_topic_key", ""),
            "target_object": {
                "label": params.get("target_label") or params.get("target_item_id", "unknown"),
                "id": params.get("target_instance_id", -1),
                "instance_id": params.get("target_instance_id", -1),
                "instance_key": params.get("target_instance_key", ""),
                "topic_key": params.get("target_topic_key", ""),
            },
            "objects": [],
            "num_matched_objects": 1,
        }
        return {"result": result, "success": True}

    async def _a2a_execute(self, params: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        camera_images = params.get("camera_images", {}) or {}
        camera_names = [
            str(camera_name).strip()
            for camera_name in (params.get("camera_names") or list(camera_images.keys()))
            if str(camera_name).strip()
        ]
        if not camera_names:
            logger.error("[%s] No camera images provided.", self.AGENT_NAME)
            return {"result": {}, "success": False}

        missing_cameras = [camera_name for camera_name in camera_names if not camera_images.get(camera_name)]
        if missing_cameras:
            logger.error("[%s] Missing camera images for: %s", self.AGENT_NAME, missing_cameras)
            return {"result": {}, "success": False}

        world_position_data = params.get("world_position_data")
        if world_position_data is None:
            logger.error("[%s] world_position_data is required.", self.AGENT_NAME)
            return {"result": {}, "success": False}

        try:
            resolver = A2ACardResolver(httpx_client=self._http_client, base_url=self._inf_url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=self._http_client, agent_card=agent_card)

            request_body = {
                "yolo_class": params.get("target_item_id") or params.get("target_label") or "",
                "target_item_id": params.get("target_item_id"),
                "target_instance_id": params.get("target_instance_id"),
                "target_instance_key": params.get("target_instance_key"),
                "target_topic_key": params.get("target_topic_key"),
                "target_label": params.get("target_label") or params.get("target_item_id") or "",
                "selected_camera": params.get("selected_camera", ""),
                "camera_names": camera_names,
                "center_world": params.get("center_world"),
                "bboxes_by_camera": params.get("bboxes_by_camera", {}),
                "world_position_data": world_position_data,
            }

            payload = {
                "message": {
                    "role": "user",
                    "parts": [
                        {"kind": "text", "text": json.dumps(request_body)}
                    ] + [
                        {"kind": "text", "text": camera_images[camera_name]}
                        for camera_name in camera_names
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
                "[%s] Sending %d camera image(s) to %s (selected=%s target=%s)",
                self.AGENT_NAME,
                len(camera_names),
                self._inf_url,
                params.get("selected_camera", "") or "N/A",
                params.get("target_instance_key", "") or params.get("target_item_id", ""),
            )
            response = await client.send_message(request)
            return {"result": self._parse_response(response), "success": True}
        except Exception as exc:
            logger.error("[%s] A2A call failed: %s", self.AGENT_NAME, exc, exc_info=True)
            return {"result": {}, "success": False}

    @staticmethod
    def _parse_response(response: Any) -> Dict[str, Any]:
        try:
            return json.loads(response.root.result.parts[0].root.text)
        except Exception as exc:
            logger.error("[GetItemInfoNoSam3D Agent] Failed to parse response: %s", exc, exc_info=True)
            return {}
