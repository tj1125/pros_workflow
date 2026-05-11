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
    goal_heading_tolerance_rad_default,
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
            reason_node --[nav_agent]-------> update_item_info_1_node → nav_node
            reason_node --[grasp_agent]-----> update_item_info_2_node → car_grasp_node → car_approach_node → update_memory_node → observe_node
            reason_node --[car_approach_agent]--> update_item_info_2_node ┘
            reason_node --[DONE]-----------> nav_home_node → END

    grasp_node: 擷取 RGBD、呼叫 GraspAgent，取得 6-DoF 抓取位姿並寫入 latest_grasp_result。
    car_approach_node: 讀取 latest_grasp_result，呼叫 CarApproachAgent 移動車體至接近點。
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
        workflow.add_node("update_item_info_1_node", self._update_item_info_1_node)
        workflow.add_node("update_item_info_2_node", self._update_item_info_2_node)
        workflow.add_node("observe_node", self._observe_node)
        workflow.add_node("reason_node", self._reason_node)
        workflow.add_node("update_memory_node", self._update_memory_node)

        # Individual Agent nodes (A2A Clients)
        workflow.add_node("nav_node", self._nav_node)
        workflow.add_node("nav_move_node", self._nav_move_node)
        workflow.add_node("nav_home_node", self._nav_home_node)
        workflow.add_node("car_grasp_node", self._car_grasp_node)
        workflow.add_node("car_approach_node", self._car_approach_node)
        workflow.add_node("get_item_info_no_sam3d_node", self._get_item_info_no_sam3d_node)

        # Entry point
        workflow.set_entry_point("input_node")

        # Fixed edges
        workflow.add_edge("input_node", "find_node")
        workflow.add_edge("get_item_info_no_sam3d_node", "nav_move_node")
        workflow.add_edge("observe_node", "reason_node")
        workflow.add_edge("nav_node", "nav_move_node")
        workflow.add_edge("nav_home_node", END)
        
        # Grasp nodes statically route to their respective approach nodes
        workflow.add_edge("car_grasp_node", "car_approach_node")
        workflow.add_edge("car_approach_node", "update_memory_node")
        
        workflow.add_edge("update_memory_node", "observe_node")

        # find_node → get_item_info_no_sam3d_node or END
        workflow.add_conditional_edges(
            "find_node",
            self._route_find,
            {"get_item_info_no_sam3d_node": "get_item_info_no_sam3d_node", "end": END},
        )

        workflow.add_conditional_edges(
            "update_item_info_1_node",
            self._route_update_item_info_1,
            {
                "get_item_info_no_sam3d_node": "get_item_info_no_sam3d_node",
                "nav_home_node": "nav_home_node",
                "nav_node": "nav_node",
            },
        )
        workflow.add_conditional_edges(
            "update_item_info_2_node",
            self._route_update_item_info_2,
            {
                "get_item_info_no_sam3d_node": "get_item_info_no_sam3d_node",
                "nav_home_node": "nav_home_node",
                "car_grasp_node": "car_grasp_node",
            },
        )

        # reason_node → update_item_info_*_node → agent node or END
        workflow.add_conditional_edges(
            "reason_node",
            self._route_decision,
            {
                "nav_node": "update_item_info_1_node",
                "car_grasp_node": "update_item_info_2_node",
                "end":        "nav_home_node",
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
    def _world_position_camera_names(candidates: list[Dict[str, Any]]) -> list[str]:
        ordered_unique: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            for camera_name in candidate.get("camsrc", []) or []:
                camera_text = str(camera_name).strip()
                if not camera_text or camera_text in seen:
                    continue
                seen.add(camera_text)
                ordered_unique.append(camera_text)
        return ordered_unique

    @staticmethod
    def _world_position_db_from_candidates(
        world_position_payload: Dict[str, Any],
        candidates: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        instances: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            instance_key = str(candidate.get("instance_key", "")).strip()
            if not instance_key:
                continue
            try:
                instance_id = int(candidate.get("instance_id", -1))
            except (TypeError, ValueError):
                instance_id = -1
            instances[instance_key] = {
                "item_id": candidate.get("item_id", ""),
                "instance_id": instance_id,
                "instance_key": instance_key,
                "topic_key": candidate.get("topic_key", ""),
                "center_world": list(candidate.get("center_world", [])),
                "camsrc": list(candidate.get("camsrc", [])),
                "bboxes_by_camera": dict(candidate.get("bboxes_by_camera", {})),
            }
        return {
            "payload": world_position_payload,
            "instances": instances,
            "updated_at": time.time(),
        }

    @staticmethod
    def _selected_target_instance_key(selected_target: Dict[str, Any]) -> str:
        return str(selected_target.get("instance_key", "")).strip()

    @staticmethod
    def _find_selected_world_position_candidate(
        candidates: list[Dict[str, Any]],
        selected_target: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        selected_key = Orchestrator._selected_target_instance_key(selected_target)
        selected_item_id = str(selected_target.get("item_id") or selected_target.get("id") or "").strip()
        try:
            selected_instance_id = int(selected_target.get("instance_id", -1))
        except (TypeError, ValueError):
            selected_instance_id = -1

        for candidate in candidates:
            if selected_key and candidate.get("instance_key") == selected_key:
                return candidate
            if (
                selected_item_id
                and candidate.get("item_id") == selected_item_id
                and int(candidate.get("instance_id", -1)) == selected_instance_id
            ):
                return candidate
        return None

    @staticmethod
    def _center_world_distance_m(old_center: Any, new_center: Any) -> float:
        try:
            old_xyz = [float(value) for value in list(old_center)[:3]]
            new_xyz = [float(value) for value in list(new_center)[:3]]
        except Exception:
            return math.inf
        if len(old_xyz) != 3 or len(new_xyz) != 3:
            return math.inf
        return math.dist(old_xyz, new_xyz)

    @staticmethod
    def _target_object_with_invalidated_item_info(
        target_object: Dict[str, Any],
        refreshed_selected_target: Dict[str, Any],
    ) -> Dict[str, Any]:
        updated = {
            **target_object,
            **refreshed_selected_target,
            "id": refreshed_selected_target.get("item_id", target_object.get("id")),
            "label": refreshed_selected_target.get("label", target_object.get("label", "")),
        }
        for key in (
            "group_ranking",
            "goal_pose_path",
            "objects",
            "num_matched_objects",
            "primary_camera_id",
            "center_world_coordinate_frame",
        ):
            updated.pop(key, None)
        return updated

    async def _update_item_info_node(
        self,
        state: CommanderState,
        source_node: str,
    ) -> Dict[str, Any]:
        selected_target = dict(state.get("selected_target") or {})
        base_update: Dict[str, Any] = {
            "world_position_target_changed": False,
            "world_position_update_source_node": source_node,
            "world_position_update_distance_m": 0.0,
            "world_position_update_reason": "unchanged",
        }
        if not selected_target:
            return {
                **base_update,
                "world_position_update_reason": "no_selected_target",
                "current_status": "WORLD_POSITION_UPDATE_SKIPPED",
            }

        if self.use_mock:
            world_position_payload = selected_target.get("world_position_data") or {"data": json.dumps({"mock": []})}
            world_position_db = self._world_position_db_from_candidates(
                world_position_payload,
                [selected_target],
            )
            return {
                **base_update,
                "world_position_db": world_position_db,
                "world_position_db_updated_at": world_position_db["updated_at"],
                "current_status": "WORLD_POSITION_UNCHANGED",
            }

        from .room_topics import get_topic_string_message
        from .world_position import parse_world_position_payload

        world_position_raw = await get_topic_string_message("/world_position_data", timeout_sec=5.0)
        if not world_position_raw:
            logger.error("[%s] Failed to read /world_position_data.", source_node)
            return {
                **base_update,
                "world_position_update_reason": "read_failed",
                "current_status": "WORLD_POSITION_UPDATE_READ_FAILED",
            }

        world_position_payload = {"data": world_position_raw}
        try:
            refreshed_candidates = parse_world_position_payload(world_position_payload)
        except Exception as exc:
            logger.error("[%s] Failed to parse /world_position_data: %s", source_node, exc, exc_info=True)
            return {
                **base_update,
                "world_position_update_reason": "parse_failed",
                "current_status": "WORLD_POSITION_UPDATE_PARSE_FAILED",
            }

        world_position_db = self._world_position_db_from_candidates(
            world_position_payload,
            refreshed_candidates,
        )
        refreshed_target = self._find_selected_world_position_candidate(
            refreshed_candidates,
            selected_target,
        )
        if not refreshed_target:
            logger.warning(
                "[%s] Selected target disappeared from /world_position_data: %s",
                source_node,
                selected_target.get("instance_key", "unknown"),
            )
            print("❌ 物品消失了。準備返回 home。")
            return {
                **base_update,
                "world_position_db": world_position_db,
                "world_position_db_updated_at": world_position_db["updated_at"],
                "world_position_update_reason": "target_missing",
                "agent_result": "物品消失了。",
                "agent_success": False,
                "current_status": "TARGET_LOST_IN_WORLD_POSITION",
            }

        previous_db = state.get("world_position_db", {}) or {}
        previous_instances = previous_db.get("instances", {}) if isinstance(previous_db, dict) else {}
        previous_target = previous_instances.get(refreshed_target.get("instance_key", ""))
        label = str(
            selected_target.get("label")
            or (state.get("target_object", {}) or {}).get("label")
            or refreshed_target.get("item_id")
            or "目標物"
        ).strip()
        refreshed_selected_target = {
            **selected_target,
            **refreshed_target,
            "label": label,
            "world_position_data": world_position_payload,
        }

        if not previous_target:
            return {
                **base_update,
                "selected_target": refreshed_selected_target,
                "world_position_db": world_position_db,
                "world_position_db_updated_at": world_position_db["updated_at"],
                "world_position_update_reason": "db_created",
                "current_status": "WORLD_POSITION_DB_CREATED",
            }

        moved_distance = self._center_world_distance_m(
            previous_target.get("center_world", []),
            refreshed_target.get("center_world", []),
        )
        threshold_m = float(os.getenv("WORLD_POSITION_UPDATE_THRESHOLD_M", "0.05"))
        if moved_distance <= threshold_m:
            return {
                **base_update,
                "world_position_update_distance_m": moved_distance,
                "current_status": "WORLD_POSITION_UNCHANGED",
            }

        target_object = self._target_object_with_invalidated_item_info(
            dict(state.get("target_object") or {}),
            refreshed_selected_target,
        )
        logger.info(
            "[%s] Target moved %.3f m (> %.3f m); routing back to get_item_info_no_sam3d_node.",
            source_node,
            moved_distance,
            threshold_m,
        )
        return {
            **base_update,
            "selected_target": refreshed_selected_target,
            "target_object": target_object,
            "world_position_db": world_position_db,
            "world_position_db_updated_at": world_position_db["updated_at"],
            "world_position_target_changed": True,
            "world_position_update_distance_m": moved_distance,
            "world_position_update_reason": "target_moved",
            "current_goal_rank": 1,
            "nav_goal_pose": {},
            "nav_plan_ready": False,
            "nav_arrived": False,
            "nav_attempt": 0,
            "nav_move_events": [],
            "latest_nav_result": {},
            "latest_grasp_result": {},
            "current_status": "WORLD_POSITION_TARGET_MOVED",
        }

    async def _update_item_info_1_node(self, state: CommanderState) -> Dict[str, Any]:
        return await self._update_item_info_node(state, "update_item_info_1_node")

    async def _update_item_info_2_node(self, state: CommanderState) -> Dict[str, Any]:
        return await self._update_item_info_node(state, "update_item_info_2_node")

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
        world_position_db = self._world_position_db_from_candidates(
            world_position_payload,
            raw_candidates,
        )
        world_position_db_update = {
            "world_position_db": world_position_db,
            "world_position_db_updated_at": world_position_db["updated_at"],
            "world_position_target_changed": False,
            "world_position_update_source_node": "find_node",
            "world_position_update_distance_m": 0.0,
            "world_position_update_reason": "db_created",
        }

        world_camera_names = self._world_position_camera_names(raw_candidates)
        camera_images = (
            {}
            if self.use_mock
            else await self._capture_room_camera_images(world_camera_names, timeout_sec=10.0)
        )
        available_world_camera_names = [
            camera_name
            for camera_name in world_camera_names
            if camera_images.get(camera_name)
        ]

        yolo_detections: Dict[int, Dict[str, Any]] = {}
        for display_id, candidate in enumerate(matching_candidates, start=1):
            primary_camera, primary_bbox = self._pick_primary_room_camera(candidate, camera_images)
            target_camera_names = [
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
                "camera_names": available_world_camera_names,
                "target_camera_names": target_camera_names,
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
                **world_position_db_update,
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
                    **world_position_db_update,
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
            **world_position_db_update,
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

    @staticmethod
    def _world_position_target_missing(state: CommanderState) -> bool:
        return state.get("world_position_update_reason", "") == "target_missing"

    def _route_update_item_info_1(self, state: CommanderState) -> str:
        if self._world_position_target_missing(state):
            return "nav_home_node"
        if state.get("world_position_target_changed", False):
            return "get_item_info_no_sam3d_node"
        return "nav_node"

    def _route_update_item_info_2(self, state: CommanderState) -> str:
        if self._world_position_target_missing(state):
            return "nav_home_node"
        if state.get("world_position_target_changed", False):
            return "get_item_info_no_sam3d_node"
        return "car_grasp_node"

    # ------------------------------------------------------------------
    # Node: get_item_info_no_sam3d (runs once — retrieve full 3D info)
    # ------------------------------------------------------------------

    async def _get_item_info_no_sam3d_node(self, state: CommanderState) -> Dict[str, Any]:
        """
        Takes the fully-prepared selected_target from find_node/update_item_info
        and delegates multi-view geometry estimation to GetItemInfoNoSam3DAgent.
        """
        from agents.get_item_info_agent_no_sam3d import GetItemInfoNoSam3DAgent
        from .room_topics import save_preview_bbox_annotated
        from .world_position import parse_world_position_payload

        target_obj = state.get("target_object", {}) or {}
        selected_target = dict(state.get("selected_target") or {})
        if not selected_target:
            logger.error("[get_item_info_no_sam3d_node] selected_target is missing.")
            print("❌ 失敗：find_node 沒有提供 selected_target。")
            return {"current_status": "ITEM_INFO_NO_SAM3D_FAILED"}

        previous_world_position_db = state.get("world_position_db", {}) or {}
        world_position_payload = (
            selected_target.get("world_position_data")
            or (
                previous_world_position_db.get("payload")
                if isinstance(previous_world_position_db, dict)
                else None
            )
        )
        if not world_position_payload:
            logger.error("[get_item_info_no_sam3d_node] selected_target has no world_position_data.")
            print("❌ 失敗：selected_target 沒有 world_position_data。")
            return {"current_status": "ITEM_INFO_NO_SAM3D_FAILED"}

        if self.use_mock:
            refreshed_target = {
                **selected_target,
                "center_world": selected_target.get("center_world", [1.2, 0.4, 2.8]),
            }
            refreshed_candidates = [refreshed_target]
        else:
            try:
                refreshed_candidates = parse_world_position_payload(world_position_payload)
            except Exception as exc:
                logger.error("[get_item_info_no_sam3d_node] Failed to parse state world_position_data: %s", exc, exc_info=True)
                print("❌ 失敗：state 內的 world_position_data 格式無法解析。")
                return {"current_status": "ITEM_INFO_NO_SAM3D_FAILED"}

            refreshed_target = self._find_selected_world_position_candidate(
                refreshed_candidates,
                selected_target,
            )
            if not refreshed_target:
                logger.error(
                    "[get_item_info_no_sam3d_node] Selected target missing from state world_position_data: %s",
                    selected_target.get("instance_key", "unknown"),
                )
                print("❌ 失敗：state 內找不到剛才選定的 instance。")
                return {"current_status": "ITEM_INFO_NO_SAM3D_FAILED"}

        world_camera_names = self._world_position_camera_names(refreshed_candidates)
        if not world_camera_names:
            world_camera_names = list(selected_target.get("camera_names", []) or [])
        camera_images = (
            dict(selected_target.get("camera_images", {}) or {})
            if self.use_mock
            else await self._capture_room_camera_images(
                world_camera_names,
                timeout_sec=10.0,
            )
        )
        world_position_db = self._world_position_db_from_candidates(
            world_position_payload,
            refreshed_candidates,
        )

        primary_camera, primary_bbox = self._pick_primary_room_camera(refreshed_target, camera_images)
        target_available_cameras = [
            camera_name
            for camera_name in refreshed_target.get("camsrc", []) or []
            if camera_images.get(camera_name)
        ]
        available_cameras = [
            camera_name
            for camera_name in (world_camera_names if not self.use_mock else target_available_cameras)
            if camera_images.get(camera_name)
        ]
        if not self.use_mock and not available_cameras:
            logger.error("[get_item_info_no_sam3d_node] No usable room-camera images for refreshed /world_position_data.")
            print("❌ 失敗：/world_position_data 內的物件沒有任何可用的房間相機影像。")
            return {"current_status": "ITEM_INFO_NO_SAM3D_FAILED"}
        if not self.use_mock and not target_available_cameras:
            logger.error("[get_item_info_no_sam3d_node] No usable target room-camera images for %s.", refreshed_target.get("instance_key", "unknown"))
            print("❌ 失敗：選定目標沒有任何可用的房間相機影像。")
            return {"current_status": "ITEM_INFO_NO_SAM3D_FAILED"}
        if not primary_camera and target_available_cameras:
            primary_camera = target_available_cameras[0]

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
            "target_camera_names": target_available_cameras,
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
            "target_camera_names": target_available_cameras,
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
        item_info_refresh_count = int(state.get("item_info_refresh_count", 0) or 0)
        should_publish_initialpose = item_info_refresh_count == 0
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
            "force_initialpose": should_publish_initialpose,
            "item_info_refresh_count": item_info_refresh_count + 1,
            "world_position_db": world_position_db,
            "world_position_db_updated_at": world_position_db["updated_at"],
            "world_position_target_changed": False,
            "world_position_update_source_node": "get_item_info_no_sam3d_node",
            "world_position_update_distance_m": 0.0,
            "world_position_update_reason": "item_info_refreshed",
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
    ) -> Literal["nav_node", "car_grasp_node", "end"]:
        module = state.get("call_module", "")
        if module == "DONE" or state.get("task_complete", False):
            logger.info("[route] Task complete — navigating home before ending graph.")
            return "end"
        mapping = {
            "nav_agent":          "nav_node",
            "grasp_agent":        "car_grasp_node",
            "approach_agent":     "car_grasp_node",
            "car_approach_agent": "car_grasp_node",
        }
        if module in {"arm_approach_agent", "view_agent"}:
            logger.warning("%s is disabled; ending without executing it.", module)
            return "end"
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
        module_params = state.get("module_params", {}) or {}
        target_object = state.get("target_object", {}) or {}
        current_rank = int(state.get("current_goal_rank", 1) or 1)
        if current_rank < 1:
            current_rank = 1

        requested_rank = None
        for key in ("goal_rank", "target_rank", "current_goal_rank", "rank"):
            if key not in module_params:
                continue
            try:
                requested_rank = int(module_params[key])
            except (TypeError, ValueError):
                continue
            break

        if requested_rank is not None:
            current_rank = max(1, requested_rank)
        else:
            latest_nav = state.get("latest_nav_result", {}) or {}
            previous_nav_rank = int(latest_nav.get("rank", 0) or 0) or current_rank
            previous_nav_arrived = bool(latest_nav.get("arrived", False)) or bool(
                state.get("nav_arrived", False)
            )
            if previous_nav_arrived and previous_nav_rank == current_rank:
                next_rank = current_rank + 1
                logger.info(
                    "[nav_node] nav_agent requested a new observation position; "
                    "advancing goal rank %s -> %s",
                    current_rank,
                    next_rank,
                )
                print(
                    f"\n🔁 nav_agent 切換觀察點：rank {current_rank} -> {next_rank}",
                    flush=True,
                )
                current_rank = next_rank

        goal_data, err = self._goal_pose_for_rank(target_object, current_rank)
        update: Dict[str, Any] = {
            "call_module": "nav_agent",
            "current_goal_rank": current_rank,
            "nav_move_source": "reason_loop",
            "force_initialpose": bool(module_params.get("force_initialpose", False)),
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
        arrival_timeout = float(os.getenv("NAV_ARRIVAL_TIMEOUT_SEC", "180"))
        publish_interval = float(os.getenv("NAV_PUBLISH_INTERVAL_SEC", "0.1"))
        goal_tolerance_m = float(
            os.getenv("NAV_GOAL_TOLERANCE_M", str(goal_tolerance_m_default()))
        )
        legacy_goal_heading_tolerance_deg = os.getenv("NAV_GOAL_HEADING_TOLERANCE_DEG")
        goal_heading_tolerance_rad = float(
            os.getenv(
                "NAV_GOAL_HEADING_TOLERANCE_RAD",
                str(
                    math.radians(float(legacy_goal_heading_tolerance_deg))
                    if legacy_goal_heading_tolerance_deg is not None
                    else goal_heading_tolerance_rad_default()
                ),
            )
        )

        rank = int(state.get("current_goal_rank", 1) or 1)
        if rank < 1:
            rank = 1
        if rank > len(group_ranking):
            last_error = f"rank={rank} out of range; no remaining goal_pose candidates"
            logger.error(f"[nav_move_node] {last_error}")
            return {
                "current_goal_rank": rank,
                "nav_attempt": 0,
                "nav_goal_pose": {},
                "nav_plan_ready": False,
                "nav_arrived": False,
                "nav_move_events": [
                    {
                        "event": "navigation_failed",
                        "detail": last_error,
                        "rank": rank,
                        "attempt": 0,
                        "source": source,
                    }
                ],
                "agent_result": f"[NAV] {last_error}",
                "agent_success": False,
                "current_status": "NAV_FAILED",
                "_exec_latency": time.time() - start_t,
            }

        force_initialpose = bool(state.get("force_initialpose", False))
        all_events = []
        last_error = "unknown navigation error"
        last_attempt = 0

        while rank <= len(group_ranking):
            goal_data, goal_err = self._goal_pose_for_rank(target_object, rank)
            if goal_err:
                next_rank = rank + 1
                print(
                    f"\n🔁 切換 goal_pose：rank {rank} -> rank {next_rank}，原因：{goal_err}",
                    flush=True,
                )
                all_events.append({
                    "event": "rank_advanced",
                    "rank_from": rank,
                    "rank_to": next_rank,
                    "detail": goal_err,
                })
                rank = next_rank
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
                    "goal_heading_tolerance_rad": goal_heading_tolerance_rad,
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

            next_rank = rank + 1
            print(
                f"\n🔁 切換 goal_pose：rank {rank} -> rank {next_rank}，原因：{last_error}",
                flush=True,
            )
            all_events.append(
                {
                    "event": "rank_advanced",
                    "rank_from": rank,
                    "rank_to": next_rank,
                    "detail": "exhausted retries on current rank",
                }
            )
            rank = next_rank

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

    async def _nav_home_node(self, state: CommanderState) -> Dict[str, Any]:
        """
        Final home navigation executor.

        Uses the original hardcoded initial pose as /goal_pose. It does not
        publish /initialpose, so AMCL is not reset; the pose is only a target.
        """
        start_t = time.time()
        goal_pose = dict(self._default_initial_pose())
        plan_timeout = float(
            os.getenv("NAV_HOME_PLAN_TIMEOUT_SEC", os.getenv("NAV_PLAN_TIMEOUT_SEC", "8"))
        )
        arrival_timeout = float(
            os.getenv("NAV_HOME_ARRIVAL_TIMEOUT_SEC", os.getenv("NAV_ARRIVAL_TIMEOUT_SEC", "180"))
        )
        publish_interval = float(
            os.getenv("NAV_HOME_PUBLISH_INTERVAL_SEC", os.getenv("NAV_PUBLISH_INTERVAL_SEC", "0.1"))
        )
        goal_tolerance_m = float(
            os.getenv(
                "NAV_HOME_GOAL_TOLERANCE_M",
                os.getenv("NAV_GOAL_TOLERANCE_M", str(goal_tolerance_m_default())),
            )
        )
        legacy_goal_heading_tolerance_deg = (
            os.getenv("NAV_HOME_GOAL_HEADING_TOLERANCE_DEG")
            or os.getenv("NAV_GOAL_HEADING_TOLERANCE_DEG")
        )
        goal_heading_tolerance_rad = float(
            os.getenv(
                "NAV_HOME_GOAL_HEADING_TOLERANCE_RAD",
                os.getenv(
                    "NAV_GOAL_HEADING_TOLERANCE_RAD",
                    str(
                        math.radians(float(legacy_goal_heading_tolerance_deg))
                        if legacy_goal_heading_tolerance_deg is not None
                        else goal_heading_tolerance_rad_default()
                    ),
                ),
            )
        )

        if self.use_mock:
            await asyncio.sleep(0.2)
            mock_events = [
                {"event": "goal_publishing", "rank": 0, "attempt": 1, "source": "nav_home"},
                {"event": "plan_ready", "rank": 0, "attempt": 1, "source": "nav_home"},
                {"event": "arrived", "rank": 0, "attempt": 1, "source": "nav_home"},
            ]
            return {
                "nav_goal_pose": goal_pose,
                "nav_plan_ready": True,
                "nav_arrived": True,
                "nav_attempt": 1,
                "nav_move_source": "nav_home",
                "nav_move_events": mock_events,
                "agent_result": "[MOCK_NAV_HOME] Arrived at hardcoded initial pose.",
                "agent_success": True,
                "task_complete": True,
                "current_status": "NAV_HOME_COMPLETED",
                "_exec_latency": time.time() - start_t,
            }

        payload = {
            "goal_pose": goal_pose,
            "publish_initialpose": False,
            "initial_pose": self._default_initial_pose(),
            "plan_timeout_sec": plan_timeout,
            "arrival_timeout_sec": arrival_timeout,
            "publish_interval_sec": publish_interval,
            "goal_tolerance_m": goal_tolerance_m,
            "goal_heading_tolerance_rad": goal_heading_tolerance_rad,
            "status_topic": "/nav_home/status",
            "attempt": 1,
            "rank": 0,
            "source": "nav_home",
        }
        result = await self._run_nav_move_runner(payload)
        events = result.get("events", [])
        plan_ready = bool(result.get("plan_ready", False))
        success = bool(result.get("success", False))
        message = result.get(
            "message",
            "home navigation completed" if success else "home navigation failed",
        )

        return {
            "nav_goal_pose": goal_pose,
            "nav_plan_ready": plan_ready,
            "nav_arrived": success,
            "nav_attempt": 1,
            "nav_move_source": "nav_home",
            "nav_move_events": events,
            "agent_result": f"[NAV_HOME] {message}",
            "agent_success": success,
            "task_complete": True,
            "current_status": "NAV_HOME_COMPLETED" if success else "NAV_HOME_FAILED",
            "_exec_latency": time.time() - start_t,
        }

    async def _do_grasp_logic(self, state: CommanderState, context_label: str) -> Dict[str, Any]:
        """GraspGen Agent Logic — outputs information only."""
        from agents.grasp_agent import GraspAgent
        from commander.camera import get_camera_rgbd_base64

        target_object = state.get("target_object", {}) or {}
        object_id = target_object.get("id")
        if not object_id:
            logger.error(f"[{context_label}] target_object.id is missing — passing empty grasp result.")
            return {"latest_grasp_result": {}}

        # --- Step 1: Capture RGBD ---
        print(f"\n📷 [{context_label}] 擷取 Camera_Car RGBD...")
        rgbd = await get_camera_rgbd_base64("Camera_Car", timeout_sec=15.0)
        if not rgbd:
            logger.error(f"[{context_label}] Failed to get Camera_Car RGBD — passing empty grasp result.")
            return {"latest_grasp_result": {}}

        # --- Step 2: Call GraspAgent ---
        params = dict(state.get("module_params", {}) or {})
        params.update(
            {
                "object_id": object_id,
                "camera_name": "Camera_Car",
                "rgb_base64": rgbd.get("rgb_base64"),
                "depth_base64": rgbd.get("depth_base64"),
            }
        )
        logger.info(f"[{context_label}] RGBD captured, calling GraspAgent for '{object_id}'.")

        agent = GraspAgent(http_client=self.http_client)
        result = await agent.execute(params, state.get("context_id", ""))
        success = bool(result.get("success", False))
        payload = result.get("result", {})
        if not isinstance(payload, dict):
            payload = {}
        logger.info(f"[{context_label}] GraspAgent finished, success={success}")

        # --- Step 3: Pack data for approach_node (no agent_result / agent_success here) ---
        return {
            "latest_grasp_result": {
                "object_id": payload.get("object_id") or object_id,
                "camera_name": payload.get("camera_name", "Camera_Car"),
                "grasp_success": success,
                "grasp_confidence": payload.get("grasp_confidence"),
                "num_candidate_grasps": payload.get("num_candidate_grasps"),
                "num_valid_grasps": payload.get("num_valid_grasps"),
                "best_grasp_pose_camera": payload.get("best_grasp_pose_camera", {}),
                "valid_grasp_poses_camera": payload.get("valid_grasp_poses_camera", []),
                "object_reference_center_camera": payload.get("object_reference_center_camera", []),
                # Raw images carried forward
                "rgb_base64": rgbd.get("rgb_base64"),
                "depth_base64": rgbd.get("depth_base64"),
            },
        }

    async def _car_grasp_node(self, state: CommanderState) -> Dict[str, Any]:
        return await self._do_grasp_logic(state, "car_grasp_node")

    @staticmethod
    def _set_default_grasp_result(params: Dict[str, Any], latest_grasp: Dict[str, Any]) -> None:
        if "grasp_result_payload" in params or "grasp_result" in params:
            return
        params["grasp_result"] = latest_grasp

    async def _car_approach_node(self, state: CommanderState) -> Dict[str, Any]:
        """Car Approach Agent Node."""
        from agents.car_approach_agent import CarApproachAgent

        params = dict(state.get("module_params", {}) or {})
        latest_grasp = state.get("latest_grasp_result", {}) or {}
        latest_nav = state.get("latest_nav_result", {}) or {}
        last_car_approach_amcl_pose = state.get("last_car_approach_amcl_pose", {}) or {}
        previous_nav_goal_pose = (
            state.get("nav_goal_pose")
            or latest_nav.get("goal_pose")
            or {}
        )
        if last_car_approach_amcl_pose:
            print("\n📍 [car_approach_node] 偵測到上一次 car_approach 結束的 /amcl_pose，保留為狀態紀錄，不作為 /initialpose publish...")
            logger.info(
                "[car_approach_node] last_car_approach_amcl_pose=%s",
                last_car_approach_amcl_pose,
            )
        elif previous_nav_goal_pose:
            print("\n📍 [car_approach_node] 第一次 car_approach，偵測到上一個 Nav2 /goal_pose，但不會拿來 publish /initialpose...")
            logger.info("[car_approach_node] previous_nav_goal_pose=%s", previous_nav_goal_pose)
        else:
            logger.warning(
                "[car_approach_node] no previous car_approach AMCL pose or Nav2 goal_pose available; "
                "car_approach will use capture-time /amcl_pose only as an internal reference if available."
            )

        # Inject grasp result and keep prior pose snapshots for internal reference only.
        self._set_default_grasp_result(params, latest_grasp)
        if previous_nav_goal_pose:
            params["previous_nav_goal_pose"] = previous_nav_goal_pose
        if last_car_approach_amcl_pose:
            params["last_car_approach_amcl_pose"] = last_car_approach_amcl_pose

        agent = CarApproachAgent()
        result = await self._run_agent(agent, state, has_http=False, params_override=params)
        result["call_module"] = "car_approach_agent"
        return result

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

    @staticmethod
    def _condense_text(value: Any, max_len: int = 240) -> str:
        """Convert arbitrary values to a short, single-line summary."""
        if isinstance(value, str):
            text = value
        else:
            try:
                text = json.dumps(value, ensure_ascii=False, sort_keys=True)
            except TypeError:
                text = str(value)
        compact = " ".join(text.split())
        if len(compact) <= max_len:
            return compact
        return compact[: max_len - 3] + "..."

    def _build_memory_outcome(
        self,
        module: str,
        result: Any,
        success: bool,
        state: CommanderState,
    ) -> tuple[str, Dict[str, Any]]:
        """Build a concise outcome summary and key facts for rolling memory."""
        if module == "nav_agent":
            rank = int(state.get("current_goal_rank", 0) or 0)
            attempt = int(state.get("nav_attempt", 0) or 0)
            arrived = bool(state.get("nav_arrived", False))
            plan_ready = bool(state.get("nav_plan_ready", False))
            summary = self._condense_text(result)
            if not summary:
                if success:
                    summary = f"Navigation reached rank {rank} on attempt {attempt}."
                else:
                    summary = f"Navigation failed at rank {rank} on attempt {attempt}."
            return summary, {
                "rank": rank,
                "attempt": attempt,
                "arrived": arrived,
                "plan_ready": plan_ready,
            }

        if module == "grasp_agent":
            payload = result if isinstance(result, dict) else {}
            object_id = payload.get("object_id") or state.get("target_object", {}).get("id")
            grasp_confidence = payload.get("grasp_confidence")
            pose_ready = bool(payload.get("best_grasp_pose_camera"))
            if success and pose_ready:
                confidence_text = (
                    f"{float(grasp_confidence):.3f}"
                    if grasp_confidence is not None
                    else "n/a"
                )
                summary = (
                    f"Best grasp ready for object_id={object_id} "
                    f"(confidence={confidence_text})."
                )
            else:
                summary = self._condense_text(payload.get("error") or result)
            return summary, {
                "object_id": object_id,
                "grasp_confidence": grasp_confidence,
                "pose_ready": pose_ready,
            }

        if module in {"approach_agent", "car_approach_agent"}:
            payload = result if isinstance(result, dict) else {}
            status_code = str(
                payload.get("status_code")
                or ("APPROACH_SUCCESS" if success else "APPROACH_FAIL")
            )
            phase = payload.get("phase", "")
            next_agent = payload.get("next_agent")
            message = payload.get("message") or payload.get("error") or result
            nav_result = payload.get("nav_result") or {}
            final_amcl_pose = {}
            if isinstance(nav_result, dict):
                final_amcl_pose = nav_result.get("final_amcl_pose") or {}
            final_amcl_pose = payload.get("final_amcl_pose") or final_amcl_pose
            summary = f"{status_code}: {self._condense_text(message)}"
            if success and next_agent:
                summary = f"{summary} next_agent={next_agent}"
            return summary, {
                "success": success,
                "status_code": status_code,
                "phase": phase,
                "next_agent": next_agent,
                "nav_success": bool((payload.get("nav_result") or {}).get("success", False))
                if isinstance(payload.get("nav_result"), dict)
                else False,
                "initial_pose_source": payload.get("initial_pose_source")
                or (nav_result.get("initial_pose_source") if isinstance(nav_result, dict) else ""),
                "final_amcl_pose_recorded": bool(final_amcl_pose),
                "arm_success": bool((payload.get("arm_result") or {}).get("success", False))
                if isinstance(payload.get("arm_result"), dict)
                else False,
            }

        return self._condense_text(result), {}

    def _build_latest_result_update(
        self,
        module: str,
        result: Any,
        success: bool,
        state: CommanderState,
        trace_id: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build dedicated latest-result state for the executed module."""
        if module == "nav_agent":
            nav_events = [
                event
                for event in (state.get("nav_move_events", []) or [])
                if str((event or {}).get("event", "") or "").strip().lower() != "tracking"
            ]
            latest_key = "latest_nav_result"
            latest_value = {
                "trace_id": trace_id,
                "message": self._condense_text(result),
                "goal_pose": state.get("nav_goal_pose", {}),
                "rank": int(state.get("current_goal_rank", 0) or 0),
                "attempt": int(state.get("nav_attempt", 0) or 0),
                "arrived": bool(state.get("nav_arrived", False)),
                "plan_ready": bool(state.get("nav_plan_ready", False)),
                "event_count": len(nav_events),
                "last_event": nav_events[-1] if nav_events else {},
            }
            return {latest_key: latest_value}, {"updated_latest_keys": [latest_key]}

        if module == "grasp_agent":
            payload = result if isinstance(result, dict) else {}
            latest_key = "latest_grasp_result"
            latest_value = {
                "trace_id": trace_id,
                "object_id": payload.get("object_id") or state.get("target_object", {}).get("id"),
                "camera_name": payload.get("camera_name", "Camera_Car"),
                "grasp_confidence": payload.get("grasp_confidence"),
                "num_candidate_grasps": payload.get("num_candidate_grasps"),
                "num_valid_grasps": payload.get("num_valid_grasps"),
                "best_grasp_pose_camera": payload.get("best_grasp_pose_camera", {}),
                "valid_grasp_poses_camera": payload.get("valid_grasp_poses_camera", []),
                "object_reference_center_camera": payload.get(
                    "object_reference_center_camera",
                    [],
                ),
            }
            return {latest_key: latest_value}, {"updated_latest_keys": [latest_key]}

        if module in {"approach_agent", "car_approach_agent"}:
            payload = result if isinstance(result, dict) else {}
            status_code = str(
                payload.get("status_code")
                or ("APPROACH_SUCCESS" if success else "APPROACH_FAIL")
            )
            latest_key = "latest_approach_result"
            nav_result = payload.get("nav_result", {})
            arm_base_alignment_result = payload.get("arm_base_alignment_result", {})
            if not isinstance(arm_base_alignment_result, dict):
                arm_base_alignment_result = {}
            final_amcl_pose = {}
            if isinstance(nav_result, dict):
                final_amcl_pose = nav_result.get("final_amcl_pose") or {}
            final_amcl_pose = payload.get("final_amcl_pose") or final_amcl_pose
            latest_value = {
                "trace_id": trace_id,
                "module": module,
                "summary": self._condense_text(payload.get("message") or payload.get("error") or result),
                "success": success,
                "status_code": status_code,
                "phase": payload.get("phase", ""),
                "next_agent": payload.get("next_agent"),
                "initial_pose_source": payload.get("initial_pose_source")
                or (nav_result.get("initial_pose_source") if isinstance(nav_result, dict) else ""),
                "final_amcl_pose": final_amcl_pose,
                "nav_result": nav_result,
                "arm_result": payload.get("arm_result", {}),
                "arm_base_alignment_result": arm_base_alignment_result,
                "selected_solution": payload.get("selected_solution", {}),
            }
            latest_update = {latest_key: latest_value}
            updated_keys = [latest_key]
            if module == "car_approach_agent" and isinstance(final_amcl_pose, dict) and final_amcl_pose:
                latest_update["last_car_approach_amcl_pose"] = final_amcl_pose
                updated_keys.append("last_car_approach_amcl_pose")
            if module == "car_approach_agent" and any(
                arm_base_alignment_result.get(key) is not None
                for key in (
                    "command_joint_position_rad",
                    "command_joint_position_deg",
                    "target_joint_position_rad",
                    "target_joint_position_deg",
                )
            ):
                latest_update["last_arm_base_alignment_result"] = arm_base_alignment_result
                updated_keys.append("last_arm_base_alignment_result")
            return latest_update, {"updated_latest_keys": updated_keys}

        return {}, {}

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
        trace_id = uuid.uuid4().hex

        success_flag = bool(state.get("agent_success", bool(result)))
        outcome_summary, key_facts = self._build_memory_outcome(
            module,
            result,
            success_flag,
            state,
        )
        memory_reasoning = self._condense_text(reasoning, max_len=320)
        mem_entry = {
            "action": module,
            "reasoning": memory_reasoning,
            "result": outcome_summary,
            "success": success_flag,
            "key_facts": key_facts,
            "trace_id": trace_id,
        }
        latest_update, state_refs = self._build_latest_result_update(
            module,
            result,
            success_flag,
            state,
            trace_id,
        )

        self.logger.log_trace(
            agent_called=module,
            reasoning=reasoning,
            decision_latency=decision_latency,
            execution_latency=exec_latency,
            success=success_flag,
            context_id=context_id,
            trace_id=trace_id,
            memory_entry=mem_entry,
            state_refs=state_refs,
            extra_info={"raw_result": result},
        )

        logger.info("[update_memory_node] History updated and trace logged.")
        return {
            "history_buffer": [mem_entry],
            "retry_count": state.get("retry_count", 0) + 1,
            "current_status": "MEMORY_UPDATED",
            **latest_update,
        }

    @staticmethod
    def _target_center_world_to_map_xy(target_object: Dict[str, Any]) -> tuple[float | None, float | None]:
        center_world = target_object.get("center_world", [])
        if not isinstance(center_world, list) or len(center_world) < 3:
            return None, None

        target_map_x = 6.0 - float(center_world[2])
        target_map_y = float(center_world[0]) - 3.0
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
            "x": 3.4133476128639803,
            "y": -3.040367824880008,
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
            "source /workspaces/install/setup.bash 2>/dev/null || true && "
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
