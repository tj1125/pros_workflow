import asyncio
import json
import logging
import math
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Literal

import httpx
import yaml
from langgraph.graph import END, StateGraph

from .brain import Brain
from .logger import TraceLogger
from .nav_settings import (
    goal_heading_tolerance_deg_default,
    goal_tolerance_m_default,
)
from .state import CommanderState

logger = logging.getLogger(__name__)
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_graspable_objects() -> list[dict[str, Any]]:
    with (_CONFIG_DIR / "objects.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle).get("graspable_objects", [])


class Orchestrator:
    """
    Builds and runs the LangGraph decision graph.

    Graph topology:
        observe_node → reason_node
            input_node → find_node → get_item_info_no_sam3d_node → nav_move_node
            nav_move_node → observe_node → reason_node
            reason_node --[nav_agent]-------> nav_node ┐
            reason_node --[grasp_agent]-----> grasp_node ├─→ update_memory_node → observe_node
            reason_node --[approach_agent]--> approach_node ┘
            reason_node --[view_agent]------> view_node ┘
            reason_node --[DONE]-----------> END
    """

    def __init__(self, trace_logger: TraceLogger, use_mock: bool = True):
        self.logger = trace_logger
        self.brain = Brain(use_mock=use_mock)
        self.http_client = httpx.AsyncClient(timeout=120.0)
        self.use_mock = use_mock
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        workflow = StateGraph(CommanderState)

        # Core nodes
        workflow.add_node("input_node", self._input_node)
        workflow.add_node("find_node", self._find_node)
        workflow.add_node("observe_node", self._observe_node)
        workflow.add_node("reason_node", self._reason_node)
        workflow.add_node("update_memory_node", self._update_memory_node)

        # Individual Agent nodes (A2A Clients)
        workflow.add_node("nav_node", self._nav_node)
        workflow.add_node("nav_move_node", self._nav_move_node)
        workflow.add_node("grasp_node", self._grasp_node)
        workflow.add_node("approach_node", self._approach_node)
        workflow.add_node("view_node", self._view_node)
        workflow.add_node("get_item_info_no_sam3d_node", self._get_item_info_no_sam3d_node)

        # Entry point
        workflow.set_entry_point("input_node")

        # Fixed edges
        workflow.add_edge("input_node", "find_node")
        workflow.add_edge("get_item_info_no_sam3d_node", "nav_move_node")
        workflow.add_edge("observe_node", "reason_node")
        workflow.add_edge("nav_node", "nav_move_node")
        workflow.add_edge("grasp_node", "update_memory_node")
        workflow.add_edge("approach_node", "update_memory_node")
        workflow.add_edge("view_node", "update_memory_node")
        workflow.add_edge("update_memory_node", "observe_node")

        # find_node → get_item_info_no_sam3d_node or END
        workflow.add_conditional_edges(
            "find_node",
            self._route_find,
            {"get_item_info_no_sam3d_node": "get_item_info_no_sam3d_node", "end": END},
        )

        # reason_node → agent node or END
        workflow.add_conditional_edges(
            "reason_node",
            self._route_decision,
            {
                "nav_node":      "nav_node",
                "grasp_node":    "grasp_node",
                "approach_node": "approach_node",
                "view_node":     "view_node",
                "end":           END,
            },
        )
        workflow.add_conditional_edges(
            "nav_move_node",
            self._route_nav_move,
            {
                "observe_node": "observe_node",
                "update_memory_node": "update_memory_node",
            },
        )

        return workflow.compile()

    # ------------------------------------------------------------------
    # Node: human input (runs ONCE at graph start)
    # ------------------------------------------------------------------

    async def _input_node(self, state: CommanderState) -> Dict[str, Any]:
        """
        Display available objects from config/objects.yaml and wait for user selection.
        If task_description is already set (e.g. injected by tests), skip stdin.
        """
        # Allow tests or programmatic callers to pre-fill task_description
        existing = state.get("task_description", "").strip()
        if existing:
            logger.info(f"[input_node] Task pre-filled: {existing}")
            return {"task_description": existing, "current_status": "INPUT_RECEIVED"}

        objects = _load_graspable_objects()

        print("\n" + "=" * 60)
        print("  VLM-RL 多代理人抓取系統")
        print("=" * 60)
        print("請選擇你要抓取的目標物：")
        for i, obj in enumerate(objects, 1):
            print(f"  {i}. {obj.get('label', obj.get('id', '未知'))}")
        print("=" * 60)

        loop = asyncio.get_event_loop()
        while True:
            choice_str = await loop.run_in_executor(
                None,
                lambda: input(f"\n請輸入目標物編號 (1-{len(objects)})：\n> "),
            )

            try:
                idx = int(choice_str.strip()) - 1
                if 0 <= idx < len(objects):
                    selected = objects[idx]
                    break
                else:
                    print("❌ 錯誤：輸入數字不在範圍內，請重新輸入。")
            except ValueError:
                print("❌ 錯誤：格式不正確，請輸入有效數字。")

        task_desc = f"抓取{selected.get('label', selected.get('id', '目標物'))}"

        task_desc = task_desc.strip() or "請抓取桌上的目標物件"
        logger.info(f"[input_node] Task received: {task_desc}")
        print(f"\n✅ 任務已確認：{task_desc}")
        print("-" * 60)

        return {
            "task_description": task_desc,
            "target_object": {
                "id": selected.get("id"),
                "label": selected.get("label")
            },
            "current_status": "INPUT_RECEIVED",
        }

    # ------------------------------------------------------------------
    # Node: find (runs once — world_position_data + human confirmation)
    # ------------------------------------------------------------------

    async def _capture_room_camera_images(
        self,
        camera_names: list[str],
        timeout_sec: float = 10.0,
    ) -> Dict[str, str]:
        from .camera_groups import room_camera_topic
        from .room_topics import get_compressed_image_topic_base64

        ordered_unique: list[str] = []
        seen: set[str] = set()
        for camera_name in camera_names:
            camera_text = str(camera_name).strip()
            if not camera_text or camera_text in seen:
                continue
            if not room_camera_topic(camera_text):
                continue
            seen.add(camera_text)
            ordered_unique.append(camera_text)

        if not ordered_unique:
            return {}

        image_results = await asyncio.gather(
            *(
                get_compressed_image_topic_base64(
                    room_camera_topic(camera_name),
                    timeout_sec=timeout_sec,
                )
                for camera_name in ordered_unique
            )
        )
        return {
            camera_name: image_b64
            for camera_name, image_b64 in zip(ordered_unique, image_results)
            if image_b64
        }

    @staticmethod
    def _pick_primary_room_camera(
        candidate: Dict[str, Any],
        camera_images: Dict[str, str],
    ) -> tuple[str, list[float]]:
        from .world_position import bbox_area

        best_camera = ""
        best_bbox: list[float] = []
        best_area = -1.0
        bboxes_by_camera = candidate.get("bboxes_by_camera", {}) or {}
        for camera_name in candidate.get("camsrc", []) or []:
            if camera_name not in camera_images:
                continue
            bbox = bboxes_by_camera.get(camera_name)
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            area = bbox_area(bbox)
            if area > best_area:
                best_area = area
                best_camera = camera_name
                best_bbox = bbox
        return best_camera, best_bbox

    async def _find_node(self, state: CommanderState) -> Dict[str, Any]:
        """
        1. Read /world_position_data once.
        2. Snapshot room-camera RGB topics and save per-instance preview crops.
        3. Ask the user to choose the desired instance and store a fully-prepared
           selected_target for get_item_info_no_sam3d_node.
        """
        target_obj = state.get("target_object", {})
        target_item_id = str(target_obj.get("id") or target_obj.get("label") or "").strip()
        target_label = str(target_obj.get("label") or target_item_id or "目標物").strip()

        from .room_topics import get_topic_string_message, save_preview_bbox_annotated
        from .world_position import normalized_item_id, parse_world_position_payload

        requested_item_id = normalized_item_id(target_item_id or target_label)
        world_position_payload: Dict[str, Any]
        raw_candidates: list[Dict[str, Any]]

        if self.use_mock:
            mock_candidate = {
                "item_id": requested_item_id or "target",
                "instance_id": 1,
                "instance_key": f"{requested_item_id or 'target'}_1",
                "topic_key": "mock",
                "center_world": [1.2, 0.4, 2.8],
                "camsrc": [],
                "bboxes_by_camera": {},
            }
            world_position_payload = {"data": json.dumps({"mock": []})}
            raw_candidates = [mock_candidate]
        else:
            world_position_raw = await get_topic_string_message("/world_position_data", timeout_sec=5.0)
            if not world_position_raw:
                logger.error("[find_node] Failed to read /world_position_data.")
                print("❌ 失敗：無法收到 /world_position_data。")
                return {
                    "yolo_detections": {},
                    "selected_detection_id": 0,
                    "selected_target": {},
                    "find_complete": True,
                    "current_status": "TARGET_NOT_FOUND",
                }
            world_position_payload = {"data": world_position_raw}
            try:
                raw_candidates = parse_world_position_payload(world_position_payload)
            except Exception as exc:
                logger.error("[find_node] Failed to parse /world_position_data: %s", exc, exc_info=True)
                print("❌ 失敗：/world_position_data 格式無法解析。")
                return {
                    "yolo_detections": {},
                    "selected_detection_id": 0,
                    "selected_target": {},
                    "find_complete": True,
                    "current_status": "TARGET_NOT_FOUND",
                }

        matching_candidates = [
            candidate
            for candidate in raw_candidates
            if candidate.get("item_id") == requested_item_id
        ]

        unique_camera_names: list[str] = []
        seen_cameras: set[str] = set()
        for candidate in matching_candidates:
            for camera_name in candidate.get("camsrc", []) or []:
                if camera_name in seen_cameras:
                    continue
                seen_cameras.add(camera_name)
                unique_camera_names.append(camera_name)
        camera_images = (
            {}
            if self.use_mock
            else await self._capture_room_camera_images(unique_camera_names, timeout_sec=10.0)
        )

        yolo_detections: Dict[int, Dict[str, Any]] = {}
        for display_id, candidate in enumerate(matching_candidates, start=1):
            primary_camera, primary_bbox = self._pick_primary_room_camera(candidate, camera_images)
            available_camera_names = [
                camera_name
                for camera_name in candidate.get("camsrc", []) or []
                if camera_images.get(camera_name)
            ]
            preview_path = ""
            if primary_camera and primary_bbox and camera_images.get(primary_camera):
                preview_file = Path("logs/find_candidates") / f"{candidate['instance_key']}.jpg"
                if save_preview_bbox_annotated(camera_images[primary_camera], primary_bbox, preview_file):
                    preview_path = str(preview_file)

            yolo_detections[display_id] = {
                "label": target_label,
                "item_id": candidate["item_id"],
                "instance_id": int(candidate["instance_id"]),
                "instance_key": candidate["instance_key"],
                "topic_key": candidate.get("topic_key", ""),
                "center_world": list(candidate.get("center_world", [])),
                "camsrc": list(candidate.get("camsrc", [])),
                "bboxes_by_camera": dict(candidate.get("bboxes_by_camera", {})),
                "primary_camera": primary_camera,
                "camera_names": available_camera_names,
                "preview_path": preview_path,
                "source": "mock_world_position" if self.use_mock else "world_position_data",
            }

        print("\n" + "=" * 55)
        print("  🔍 world_position_data 候選結果：")
        print("=" * 55)
        if not yolo_detections:
            print(f"  ⚠ 找不到類別 '{target_label}' 的任何 instance。")
        else:
            for det_id, det in yolo_detections.items():
                print(
                    f"  [{det_id}] {det.get('instance_key','?')}  "
                    f"center_world={det.get('center_world', [])}  "
                    f"camsrc={det.get('camsrc', [])}"
                )
                if det.get("preview_path"):
                    print(f"      預覽圖: {det['preview_path']}")
        print("=" * 55)

        loop = asyncio.get_event_loop()

        if not yolo_detections:
            logger.info("[find_node] No matching instance found for %s.", requested_item_id)
            return {
                "yolo_detections": yolo_detections,
                "selected_detection_id": 0,
                "selected_target": {},
                "find_complete": True,
                "current_status": "TARGET_NOT_FOUND",
            }

        while True:
            choice_str = await loop.run_in_executor(
                None,
                lambda: input(
                    f"\n請輸入目標物編號 (1-{len(yolo_detections)}) 或輸入 no 表示未找到：\n> "
                ),
            )

            choice = choice_str.strip().lower()
            if choice == "no":
                logger.info("[find_node] User indicated no valid instance found.")
                return {
                    "yolo_detections": yolo_detections,
                    "selected_detection_id": 0,
                    "selected_target": {},
                    "find_complete": True,
                    "current_status": "TARGET_NOT_FOUND",
                }

            try:
                selected_id = int(choice)
                if selected_id in yolo_detections:
                    det = yolo_detections[selected_id]
                    break
                else:
                    print("❌ 錯誤：輸入號碼不在選項內，請重新輸入。")
            except ValueError:
                print("❌ 錯誤：格式不正確，請輸入數字或 'no'。")

        selected_target = {
            **det,
            "selected_camera": det.get("primary_camera", ""),
            "camera_images": {
                camera_name: camera_images[camera_name]
                for camera_name in det.get("camera_names", [])
                if camera_name in camera_images
            },
            "world_position_data": world_position_payload,
        }

        logger.info("[find_node] User selected instance %s.", det.get("instance_key", "unknown"))
        print(f"\n✅ 選定目標：[{selected_id}] {det.get('instance_key', '未知目標')}")
        print("-" * 55)

        return {
            "yolo_detections": yolo_detections,
            "selected_detection_id": selected_id,
            "selected_target": selected_target,
            "find_complete": True,
            "current_status": "TARGET_SELECTED_FROM_WORLD_POSITION",
        }

    # ------------------------------------------------------------------
    # Conditional edge: after find_node
    # ------------------------------------------------------------------

    def _route_find(self, state: CommanderState) -> str:
        """Route to get_item_info_no_sam3d_node if target found, else END."""
        if state.get("selected_detection_id", 0) == 0:
            logger.info("[route_find] No target → ending graph.")
            return "end"
        return "get_item_info_no_sam3d_node"

    # ------------------------------------------------------------------
    # Node: get_item_info_no_sam3d (runs once — retrieve full 3D info)
    # ------------------------------------------------------------------

    async def _get_item_info_no_sam3d_node(self, state: CommanderState) -> Dict[str, Any]:
        """
        Takes the fully-prepared selected_target from find_node, refreshes the
        exact instance from /world_position_data, and delegates multi-view
        geometry estimation to GetItemInfoNoSam3DAgent.
        """
        from agents.get_item_info_agent_no_sam3d import GetItemInfoNoSam3DAgent
        from .room_topics import get_topic_string_message, save_preview_bbox_annotated
        from .world_position import find_instance

        target_obj = state.get("target_object", {}) or {}
        selected_target = dict(state.get("selected_target") or {})
        if not selected_target:
            logger.error("[get_item_info_no_sam3d_node] selected_target is missing.")
            print("❌ 失敗：find_node 沒有提供 selected_target。")
            return {"current_status": "ITEM_INFO_NO_SAM3D_FAILED"}

        if self.use_mock:
            world_position_payload = selected_target.get("world_position_data") or {"data": json.dumps({"mock": []})}
            refreshed_target = {
                **selected_target,
                "center_world": selected_target.get("center_world", [1.2, 0.4, 2.8]),
            }
            camera_images = dict(selected_target.get("camera_images", {}) or {})
        else:
            world_position_raw = await get_topic_string_message("/world_position_data", timeout_sec=5.0)
            if not world_position_raw:
                logger.error("[get_item_info_no_sam3d_node] Failed to refresh /world_position_data.")
                print("❌ 失敗：進入 no_sam3d 前無法刷新 /world_position_data。")
                return {"current_status": "ITEM_INFO_NO_SAM3D_FAILED"}
            world_position_payload = {"data": world_position_raw}
            try:
                refreshed_target = find_instance(
                    world_position_payload,
                    selected_target.get("item_id", ""),
                    int(selected_target.get("instance_id", -1)),
                )
            except Exception as exc:
                logger.error("[get_item_info_no_sam3d_node] Failed to parse refreshed /world_position_data: %s", exc, exc_info=True)
                print("❌ 失敗：刷新後的 /world_position_data 格式無法解析。")
                return {"current_status": "ITEM_INFO_NO_SAM3D_FAILED"}
            if not refreshed_target:
                logger.error(
                    "[get_item_info_no_sam3d_node] Instance disappeared. item=%s instance=%s",
                    selected_target.get("item_id", ""),
                    selected_target.get("instance_id", -1),
                )
                print("❌ 失敗：重新訂閱後找不到剛才選定的 instance。")
                return {"current_status": "ITEM_INFO_NO_SAM3D_FAILED"}
            camera_images = await self._capture_room_camera_images(
                refreshed_target.get("camsrc", []) or [],
                timeout_sec=10.0,
            )

        primary_camera, primary_bbox = self._pick_primary_room_camera(refreshed_target, camera_images)
        available_cameras = [
            camera_name
            for camera_name in refreshed_target.get("camsrc", []) or []
            if camera_images.get(camera_name)
        ]
        if not self.use_mock and not available_cameras:
            logger.error("[get_item_info_no_sam3d_node] No usable room-camera images for %s.", refreshed_target.get("instance_key", "unknown"))
            print("❌ 失敗：選定目標沒有任何可用的房間相機影像。")
            return {"current_status": "ITEM_INFO_NO_SAM3D_FAILED"}
        if not primary_camera and available_cameras:
            primary_camera = available_cameras[0]

        preview_path = str(selected_target.get("preview_path", "") or "")
        if primary_camera and primary_bbox and camera_images.get(primary_camera):
            preview_file = Path("logs/find_candidates") / f"{refreshed_target['instance_key']}.jpg"
            if save_preview_bbox_annotated(camera_images[primary_camera], primary_bbox, preview_file):
                preview_path = str(preview_file)

        label = str(
            selected_target.get("label")
            or target_obj.get("label")
            or refreshed_target.get("item_id")
            or "目標物"
        ).strip()
        params = {
            "target_item_id": refreshed_target.get("item_id"),
            "target_instance_id": int(refreshed_target.get("instance_id", -1)),
            "target_instance_key": refreshed_target.get("instance_key"),
            "target_topic_key": refreshed_target.get("topic_key"),
            "target_label": label,
            "selected_camera": primary_camera,
            "camera_names": available_cameras,
            "camera_images": {
                camera_name: camera_images[camera_name]
                for camera_name in available_cameras
                if camera_name in camera_images
            },
            "center_world": list(refreshed_target.get("center_world", [])),
            "bboxes_by_camera": dict(refreshed_target.get("bboxes_by_camera", {})),
            "world_position_data": world_position_payload,
        }
        if self.use_mock:
            agent_result = {
                "center_world": list(refreshed_target.get("center_world", [])),
                "center_world_coordinate_frame": "unity_world",
                "group_ranking": [],
                "goal_pose_path": "/tmp/mock_goal_pose.json",
                "primary_camera_id": primary_camera,
                "target_instance_key": refreshed_target.get("instance_key"),
                "target_topic_key": refreshed_target.get("topic_key"),
                "target_object": {
                    "label": label,
                    "id": int(refreshed_target.get("instance_id", -1)),
                    "instance_id": int(refreshed_target.get("instance_id", -1)),
                    "instance_key": refreshed_target.get("instance_key"),
                    "topic_key": refreshed_target.get("topic_key"),
                },
                "objects": [],
                "num_matched_objects": 1,
            }
        else:
            agent = GetItemInfoNoSam3DAgent(http_client=self.http_client)
            print(
                f"📡 將選定目標 '{refreshed_target.get('instance_key', label)}' 與 "
                f"{available_cameras} 傳送至 GetItemInfoNoSam3DAgent..."
            )
            result = await agent.execute(params, state.get("context_id", ""))
            if not result.get("success", False):
                logger.error("[get_item_info_no_sam3d_node] GetItemInfoNoSam3DAgent execution failed.")
                print("❌ 失敗：GetItemInfoNoSam3DAgent 沒有成功回傳結果。")
                return {"current_status": "ITEM_INFO_NO_SAM3D_FAILED"}
            agent_result = result.get("result", {}) or {}

        refreshed_selected_target = {
            **selected_target,
            **refreshed_target,
            "label": label,
            "selected_camera": primary_camera,
            "primary_camera": primary_camera,
            "camera_names": available_cameras,
            "camera_images": {
                camera_name: camera_images[camera_name]
                for camera_name in available_cameras
                if camera_name in camera_images
            },
            "preview_path": preview_path,
            "world_position_data": world_position_payload,
        }
        target_object = {
            **target_obj,
            **refreshed_selected_target,
            **agent_result,
            "id": refreshed_target.get("item_id", target_obj.get("id")),
            "label": label,
        }

        current_rank = int(state.get("current_goal_rank", 1) or 1)
        if current_rank < 1:
            current_rank = 1
        goal_pose, goal_pose_err = self._goal_pose_for_rank(target_object, current_rank)
        if goal_pose_err:
            logger.warning(
                "[get_item_info_no_sam3d_node] Target '%s' has no goal_pose yet: %s",
                refreshed_target.get("instance_key", label),
                goal_pose_err,
            )
            print(f"\n📦 no_sam3d 目標資訊已取得！ goal_pose = N/A ({goal_pose_err})")
        else:
            logger.info(
                "[get_item_info_no_sam3d_node] Target '%s' goal_pose retrieved: %s",
                refreshed_target.get("instance_key", label),
                goal_pose,
            )
            print(f"\n📦 no_sam3d 目標資訊已取得！ goal_pose = {goal_pose}")

        return {
            "selected_target": refreshed_selected_target,
            "target_object": target_object,
            "current_goal_rank": current_rank,
            "nav_goal_pose": goal_pose if not goal_pose_err else {},
            "nav_move_source": "bootstrap",
            "force_initialpose": True,
            "current_status": "ITEM_INFO_NO_SAM3D_READY",
        }

    # ------------------------------------------------------------------
    # Node: observe
    # ------------------------------------------------------------------


    async def _observe_node(self, state: CommanderState) -> Dict[str, Any]:
        """Fetch current environment observation (mock or Rosbridge)."""
        task_desc = state.get("task_description", "抓取目標物件")

        if self.use_mock:
            obs = {
                "description": (
                    f"任務：{task_desc} "
                    f"| Mock scene at step {state.get('retry_count', 0)}: "
                    "Target object visible on table with partial occlusion."
                ),
                "image_base64": None,
            }
        else:
            from .camera import get_camera_image_base64
            image_b64 = await get_camera_image_base64("Camera_Car", timeout_sec=15.0)
            obs = {
                "description": task_desc,
                "image_base64": image_b64,
            }

        logger.info(f"[observe_node] Observation captured (Has Image: {bool(obs.get('image_base64'))}).")
        return {"current_observation": obs, "current_status": "OBSERVED"}

    # ------------------------------------------------------------------
    # Node: reason
    # ------------------------------------------------------------------

    async def _reason_node(self, state: CommanderState) -> Dict[str, Any]:
        """Call the VLM Brain to produce a BrainDecision."""
        out = await self.brain.reason(state)
        decision = out["prediction"]
        logger.info(
            f"[reason_node] call_module={decision.call_module} "
            f"latency={out['latency']:.2f}s"
        )
        return {
            "reasoning": decision.reasoning,
            "call_module": decision.call_module,
            "module_params": decision.module_params,
            "decision_latency": out["latency"],
            "current_status": "REASONED",
        }

    # ------------------------------------------------------------------
    # Conditional edge
    # ------------------------------------------------------------------

    def _route_decision(
        self, state: CommanderState
    ) -> Literal["nav_node", "grasp_node", "approach_node", "view_node", "end"]:
        module = state.get("call_module", "")
        if module == "DONE" or state.get("task_complete", False):
            logger.info("[route] Task complete — ending graph.")
            return "end"
        mapping = {
            "nav_agent":      "nav_node",
            "grasp_agent":    "grasp_node",
            "approach_agent": "approach_node",
            "view_agent":     "view_node",
        }
        return mapping.get(module, "end")

    def _route_nav_move(self, state: CommanderState) -> Literal["observe_node", "update_memory_node"]:
        source = state.get("nav_move_source", "")
        if source == "bootstrap":
            return "observe_node"
        return "update_memory_node"

    # ------------------------------------------------------------------
    # Agent nodes (each is an A2A Client calling RTX 3090)
    # ------------------------------------------------------------------

    async def _nav_node(self, state: CommanderState) -> Dict[str, Any]:
        """Prepare navigation context and delegate execution to nav_move_node."""
        current_rank = int(state.get("current_goal_rank", 1) or 1)
        if current_rank < 1:
            current_rank = 1
        goal_data, err = self._goal_pose_for_rank(state.get("target_object", {}), current_rank)
        update: Dict[str, Any] = {
            "current_goal_rank": current_rank,
            "nav_move_source": "reason_loop",
            "force_initialpose": bool(state.get("module_params", {}).get("force_initialpose", False)),
            "current_status": "NAV_CONTEXT_READY",
        }
        if err:
            update["agent_result"] = f"[NAV_CONTEXT_ERROR] {err}"
            update["agent_success"] = False
            update["nav_goal_pose"] = {}
        else:
            update["nav_goal_pose"] = goal_data
        return update

    async def _nav_move_node(self, state: CommanderState) -> Dict[str, Any]:
        """
        Blocking navigation executor:
        1. Publish /initialpose and /goal_pose.
        2. Wait for the tools-side navigation stack to expose a global plan.
        3. Observe AMCL until the robot reaches the requested goal pose.
        """
        source = state.get("nav_move_source", "reason_loop")
        target_object = state.get("target_object", {})
        group_ranking = target_object.get("group_ranking", []) or []
        if not group_ranking:
            return {
                "agent_result": "[NAV] group_ranking is empty, cannot navigate.",
                "agent_success": False,
                "nav_plan_ready": False,
                "nav_arrived": False,
                "nav_move_events": [],
                "current_status": "NAV_FAILED",
                "_exec_latency": 0.0,
            }

        start_t = time.time()
        same_rank_retries = max(0, int(os.getenv("NAV_SAME_RANK_RETRIES", "0")))
        max_attempt_per_rank = same_rank_retries + 1
        plan_timeout = float(os.getenv("NAV_PLAN_TIMEOUT_SEC", "8"))
        arrival_timeout = float(os.getenv("NAV_ARRIVAL_TIMEOUT_SEC", "120"))
        publish_interval = float(os.getenv("NAV_PUBLISH_INTERVAL_SEC", "0.1"))
        goal_tolerance_m = float(
            os.getenv("NAV_GOAL_TOLERANCE_M", str(goal_tolerance_m_default()))
        )
        goal_heading_tolerance_deg = float(
            os.getenv(
                "NAV_GOAL_HEADING_TOLERANCE_DEG",
                str(goal_heading_tolerance_deg_default()),
            )
        )

        rank = int(state.get("current_goal_rank", 1) or 1)
        if rank < 1:
            rank = 1

        force_initialpose = bool(state.get("force_initialpose", False))
        all_events = []
        last_error = "unknown navigation error"
        last_attempt = 0

        while rank <= len(group_ranking):
            goal_data, goal_err = self._goal_pose_for_rank(target_object, rank)
            if goal_err:
                all_events.append({
                    "event": "rank_advanced",
                    "rank_from": rank,
                    "rank_to": rank + 1,
                    "detail": goal_err,
                })
                rank += 1
                continue

            for attempt in range(1, max_attempt_per_rank + 1):
                last_attempt = attempt
                publish_initialpose = force_initialpose
                if self.use_mock:
                    await asyncio.sleep(0.2)
                    mock_events = [
                        {"event": "goal_publishing", "rank": rank, "attempt": attempt},
                        {"event": "plan_ready", "rank": rank, "attempt": attempt},
                        {"event": "arrived", "rank": rank, "attempt": attempt},
                    ]
                    return {
                        "current_goal_rank": rank,
                        "nav_attempt": attempt,
                        "nav_goal_pose": goal_data,
                        "nav_plan_ready": True,
                        "nav_arrived": True,
                        "nav_move_events": mock_events,
                        "agent_result": f"[MOCK_NAV] Arrived at rank {rank} (attempt {attempt}).",
                        "agent_success": True,
                        "current_status": "NAV_COMPLETED",
                        "_exec_latency": time.time() - start_t,
                    }

                payload = {
                    "goal_pose": goal_data,
                    "publish_initialpose": publish_initialpose,
                    "initial_pose": self._default_initial_pose(),
                    "plan_timeout_sec": plan_timeout,
                    "arrival_timeout_sec": arrival_timeout,
                    "publish_interval_sec": publish_interval,
                    "goal_tolerance_m": goal_tolerance_m,
                    "goal_heading_tolerance_deg": goal_heading_tolerance_deg,
                    "status_topic": "/nav_move/status",
                    "attempt": attempt,
                    "rank": rank,
                    "source": source,
                }
                result = await self._run_nav_move_runner(payload)
                events = result.get("events", [])
                all_events.extend(events)
                plan_ready = bool(result.get("plan_ready", False))
                success = bool(result.get("success", False))
                if publish_initialpose and plan_ready:
                    force_initialpose = False

                if success:
                    return {
                        "current_goal_rank": rank,
                        "nav_attempt": attempt,
                        "nav_goal_pose": goal_data,
                        "nav_plan_ready": plan_ready,
                        "nav_arrived": True,
                        "nav_move_events": all_events,
                        "agent_result": result.get(
                            "message",
                            f"[NAV] Arrived at rank {rank} (attempt {attempt}).",
                        ),
                        "agent_success": True,
                        "current_status": "NAV_COMPLETED",
                        "_exec_latency": time.time() - start_t,
                    }

                last_error = result.get("message", "navigation attempt failed")
                all_events.append(
                    {
                        "event": "attempt_failed",
                        "rank": rank,
                        "attempt": attempt,
                        "detail": last_error,
                    }
                )

            all_events.append(
                {
                    "event": "rank_advanced",
                    "rank_from": rank,
                    "rank_to": rank + 1,
                    "detail": "exhausted retries on current rank",
                }
            )
            rank += 1

        all_events.append(
            {
                "event": "navigation_failed",
                "detail": last_error,
            }
        )
        logger.error(f"[nav_move_node] Navigation failed completely. Inner error: {last_error}")
        return {
            "current_goal_rank": rank,
            "nav_attempt": last_attempt,
            "nav_goal_pose": {},
            "nav_plan_ready": False,
            "nav_arrived": False,
            "nav_move_events": all_events,
            "agent_result": f"[NAV] Failed after exhausting all ranks: {last_error}",
            "agent_success": False,
            "current_status": "NAV_FAILED",
            "_exec_latency": time.time() - start_t,
        }

    async def _grasp_node(self, state: CommanderState) -> Dict[str, Any]:
        """GraspGen Agent Node: capture Camera_Car RGBD and call INF_GRASP."""
        from agents.grasp_agent import GraspAgent
        from commander.camera import get_camera_rgbd_base64

        target_object = state.get("target_object", {}) or {}
        object_id = target_object.get("id")
        if not object_id:
            logger.error("[grasp_node] target_object.id is missing.")
            return {
                "agent_result": "[GRASP] target_object.id is missing, cannot call grasp agent.",
                "agent_success": False,
                "current_status": "EXECUTED",
                "_exec_latency": 0.0,
            }

        print("\n📷 正在擷取 Camera_Car 的 RGBD 影像...")
        rgbd = await get_camera_rgbd_base64("Camera_Car", timeout_sec=15.0)
        if not rgbd:
            logger.error("[grasp_node] Failed to get Camera_Car RGBD image.")
            return {
                "agent_result": "[GRASP] Failed to capture Camera_Car RGBD image.",
                "agent_success": False,
                "current_status": "EXECUTED",
                "_exec_latency": 0.0,
            }

        params = dict(state.get("module_params", {}) or {})
        params.update(
            {
                "object_id": object_id,
                "camera_name": "Camera_Car",
                "rgb_base64": rgbd.get("rgb_base64"),
                "depth_base64": rgbd.get("depth_base64"),
            }
        )
        logger.info("[grasp_node] Captured Camera_Car RGBD and prepared grasp request for '%s'.", object_id)
        agent = GraspAgent(http_client=self.http_client)
        return await self._run_agent(agent, state, params_override=params)

    async def _approach_node(self, state: CommanderState) -> Dict[str, Any]:
        """Approach Agent Node: guide arm to pre-grasp point (local control, no GPU)."""
        from agents.approach_agent import ApproachAgent
        agent = ApproachAgent()
        return await self._run_agent(agent, state, has_http=False)

    async def _view_node(self, state: CommanderState) -> Dict[str, Any]:
        """View Agent Node: adjust camera/arm posture via INF_VIEW (A2A Server)."""
        from agents.view_agent import ViewAgent
        agent = ViewAgent(http_client=self.http_client)
        return await self._run_agent(agent, state)

    async def _run_agent(
        self,
        agent: Any,
        state: CommanderState,
        has_http: bool = True,
        params_override: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Common runner: call agent.execute() and return state updates."""
        params = params_override if params_override is not None else state.get("module_params", {})
        context_id = state.get("context_id", uuid.uuid4().hex)
        start = time.time()

        result = await agent.execute(params, context_id)
        exec_latency = time.time() - start

        logger.info(
            f"[{agent.AGENT_NAME}] finished in {exec_latency:.2f}s, "
            f"success={result['success']}"
        )
        return {
            "agent_result": result["result"],
            "agent_success": bool(result.get("success", False)),
            "current_status": "EXECUTED",
            "_exec_latency": exec_latency,
        }

    # ------------------------------------------------------------------
    # Node: update memory
    # ------------------------------------------------------------------

    async def _update_memory_node(self, state: CommanderState) -> Dict[str, Any]:
        """Append last action to history (capped at 3) and write trace log."""
        module = state.get("call_module", "")
        reasoning = state.get("reasoning", "")
        result = state.get("agent_result", "")
        decision_latency = state.get("decision_latency", 0.0)
        exec_latency = state.get("_exec_latency", 0.0)
        context_id = state.get("context_id", "")

        mem_entry = {
            "action": module,
            "reasoning": reasoning,
            "result": result,
            "success": bool(state.get("agent_success", bool(result))),
        }

        success_flag = bool(state.get("agent_success", bool(result)))
        self.logger.log_trace(
            agent_called=module,
            reasoning=reasoning,
            decision_latency=decision_latency,
            execution_latency=exec_latency,
            success=success_flag,
            context_id=context_id,
            extra_info={"result": result},
        )

        logger.info("[update_memory_node] History updated and trace logged.")
        return {
            "history_buffer": [mem_entry],
            "retry_count": state.get("retry_count", 0) + 1,
            "current_status": "MEMORY_UPDATED",
        }

    @staticmethod
    def _target_center_world_to_map_xy(target_object: Dict[str, Any]) -> tuple[float | None, float | None]:
        center_world = target_object.get("center_world", [])
        if not isinstance(center_world, list) or len(center_world) < 3:
            return None, None

        target_map_x = 6.0 - float(center_world[2])
        target_map_y = float(center_world[0]) - 3.314
        return target_map_x, target_map_y

    def _goal_pose_for_rank(self, target_object: Dict[str, Any], rank: int) -> tuple[Dict[str, Any], str]:
        group_ranking = target_object.get("group_ranking", []) or []
        rank_idx = rank - 1
        if rank_idx < 0 or rank_idx >= len(group_ranking):
            return {}, f"rank={rank} out of range"

        goal_data = group_ranking[rank_idx] or {}
        goal_pose_ros = goal_data.get("best_goal_pose_ros_map", [])
        if not isinstance(goal_pose_ros, list) or len(goal_pose_ros) < 2:
            return {}, f"rank={rank} missing best_goal_pose_ros_map"

        goal_x = float(goal_pose_ros[0])
        goal_y = float(goal_pose_ros[1])
        yaw = 0.0
        target_map_x, target_map_y = self._target_center_world_to_map_xy(target_object)
        if target_map_x is not None and target_map_y is not None:
            dx = target_map_x - goal_x
            dy = target_map_y - goal_y
            yaw = math.atan2(dy, dx)

        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        goal_pose = {
            "x": goal_x,
            "y": goal_y,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": qz,
            "qw": qw,
            "yaw": yaw,
        }
        if target_map_x is not None and target_map_y is not None:
            goal_pose["face_target_x"] = target_map_x
            goal_pose["face_target_y"] = target_map_y
        return goal_pose, ""

    def _default_initial_pose(self) -> Dict[str, Any]:
        return {
            "x": 3.3596361258505296,
            "y": -3.1483084430012473,
            "z": 0.0,
            "qx": 0.0,
            "qy": 0.0,
            "qz": 0.0,
            "qw": 1.0,
            "covariance": [
                0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.06853892060437211,
            ],
        }

    async def _run_nav_move_runner(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import shlex
        safe_payload = shlex.quote(json.dumps(payload))
        # Unset uv env vars so /usr/bin/python3 (3.10) runs cleanly with ROS 2.
        # Explicitly set PYTHONPATH and LD_LIBRARY_PATH to the ROS humble paths so
        # both the Python modules and their compiled .so extensions are found.
        ros_py = "/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages"
        ros_lib = "/opt/ros/humble/lib"
        cmd = (
            "unset VIRTUAL_ENV PYTHONPATH PYTHONHOME && "
            "source /opt/ros/humble/setup.bash && "
            "source /workspaces/nav_install/setup.bash 2>/dev/null || true && "
            f"export LD_LIBRARY_PATH={ros_lib}:${{LD_LIBRARY_PATH:-}} && "
            f"PYTHONPATH={ros_py} "
            f"/usr/bin/python3 -m commander.nav_move_runner --payload {safe_payload}"
        )

        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            executable='/bin/bash'
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err_msg = stderr.decode().strip() or "nav_move_runner failed"
            logger.error(f"[nav_move_node] runner exit={proc.returncode}: {err_msg}")
            return {
                "success": False,
                "plan_ready": False,
                "message": err_msg,
                "events": [],
            }

        try:
            return json.loads(stdout.decode().strip() or "{}")
        except json.JSONDecodeError:
            text = stdout.decode().strip()
            logger.error(f"[nav_move_node] runner returned non-JSON output: {text}")
            return {
                "success": False,
                "plan_ready": False,
                "message": "Invalid nav_move_runner output",
                "events": [],
            }

    async def aclose(self) -> None:
        """Clean up the shared httpx client."""
        await self.http_client.aclose()
