"""A2A client for the no-SAM3D item-info service."""

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


class GetItemInfoNoSam3DAgent:
    AGENT_NAME = "GetItemInfoNoSam3D Agent"

    def __init__(self, http_client: httpx.AsyncClient = None, use_mock: bool | None = None):
        self._http_client = http_client or httpx.AsyncClient(timeout=120.0)
        self._inf_url = os.getenv("INF_GET_ITEM_INFO_NO_SAM3D_URL", "").strip()
        self._use_mock = bool(use_mock) if use_mock is not None else (os.getenv("MOCK_MODE", "true").lower() == "true" or not self._inf_url)

    async def execute(self, params: Dict[str, Any], context_id: str = "") -> Dict[str, Any]:
        if self._use_mock:
            return await self._mock_execute(params)
        return await self._a2a_execute(params, context_id)

    async def _mock_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.2)
        result = {
            "center_world": params.get("center_world", [1.0, 0.0, 0.5]),
            "center_world_coordinate_frame": "unity_world",
            "group_ranking": [
                {
                    "rank": 1,
                    "orientation_group": "mock_front",
                    "best_goal_pose_ros_map": [1.0, 0.0],
                    "best_confidence": 0.8,
                    "map_feasible": True,
                    "selection_mode": "mock",
                }
            ],
            "goal_pose_path": "/tmp/mock_goal_pose.json",
            "primary_camera_id": params.get("selected_camera", ""),
            "target_instance_key": params.get("target_instance_key", ""),
            "target_topic_key": params.get("target_topic_key", ""),
            "objects": [],
            "num_matched_objects": 1,
        }
        return {"result": result, "success": True, "a2a_task_id": ""}

    async def _a2a_execute(self, params: Dict[str, Any], context_id: str) -> Dict[str, Any]:
        camera_images = params.get("camera_images", {}) or {}
        camera_names = [
            str(camera_name).strip()
            for camera_name in (params.get("camera_names") or list(camera_images.keys()))
            if str(camera_name).strip()
        ]
        if not camera_names:
            return {"result": {"error": "no camera images provided"}, "success": False}

        missing_cameras = [camera_name for camera_name in camera_names if not camera_images.get(camera_name)]
        if missing_cameras:
            return {"result": {"error": f"missing camera images for {missing_cameras}"}, "success": False}

        world_position_data = params.get("world_position_data")
        if world_position_data is None:
            return {"result": {"error": "world_position_data is required"}, "success": False}

        try:
            resolver = A2ACardResolver(httpx_client=self._http_client, base_url=self._inf_url)
            agent_card = await resolver.get_agent_card()
            require_agent_card_modes(agent_card, input_modes={"data", "file"}, output_modes={"data"}, skill_ids={"get_item_info_no_sam3d"})
            client = A2AClient(httpx_client=self._http_client, agent_card=agent_card)

            metadata = {
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
            parts = [data_part(metadata)]
            parts.extend(
                file_part(
                    name=f"{camera_name}.jpg",
                    data_base64=camera_images[camera_name],
                    mime_type="image/jpeg",
                    metadata={"camera_name": camera_name},
                )
                for camera_name in camera_names
            )
            payload = {
                "message": {
                    "role": "user",
                    "parts": parts,
                    "message_id": uuid.uuid4().hex,
                    "context_id": context_id,
                }
            }
            request = SendMessageRequest(id=str(uuid.uuid4()), params=MessageSendParams(**payload))

            logger.info(
                "[%s] Sending %d FilePart image(s) to %s (selected=%s target=%s)",
                self.AGENT_NAME,
                len(camera_names),
                self._inf_url,
                params.get("selected_camera", "") or "N/A",
                params.get("target_instance_key", "") or params.get("target_item_id", ""),
            )
            response = await client.send_message(request)
            result_payload, a2a_task_id = extract_result_payload(response)
            return {"result": result_payload, "success": "error" not in result_payload, "a2a_task_id": a2a_task_id}
        except Exception as exc:
            logger.error("[%s] A2A call failed: %s", self.AGENT_NAME, exc, exc_info=True)
            return {"result": {"error": str(exc)}, "success": False, "a2a_task_id": ""}
