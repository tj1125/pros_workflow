from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Literal

from ..brain import BrainDecision
from ..camera.registry import configured_room_cameras
from ..contracts import (
    ApproachResult,
    DecisionRecord,
    GraspResult,
    RequestedObject,
    ItemInfoResult,
    NavGoal,
    NavResult,
    NavigationState,
    Observation,
    RoomCameraSnapshot,
    TaskContext,
    SelectedInstance,
    WorldPositionSnapshot,
    dump_model,
)
from ..nav.goal_builder import (
    goal_pose_db_from_item_info,
    goal_pose_for_rank,
    goal_pose_from_ros_map,
    target_center_world_to_map_xy,
)
from ..object_catalog import lexical_related_object_options, load_graspable_objects, object_option, valid_object_index
from ..prompts import RELATED_OBJECT_SELECTION_HUMAN_TEMPLATE, RELATED_OBJECT_SELECTION_SYSTEM_PROMPT
from ..runtime_settings import (
    default_initial_pose,
    nav_runner_payload_defaults,
    ros_map_origin_unity,
    ros_subprocess_settings,
    world_position_update_threshold_m,
)
from ..state import CommanderState
from ..storage.artifact_store import ArtifactStore

logger = logging.getLogger(__name__)


class PickFlowMixin:
    @staticmethod
    def _nav_goal_from_pose(goal: dict[str, Any]) -> NavGoal:
        return NavGoal(**{key: value for key, value in goal.items() if key in NavGoal.model_fields})

    async def _resolve_related_object_options(
        self,
        task_text: str,
        objects: list[dict[str, Any]],
        selected_index: int = 0,
    ) -> list[dict[str, Any]]:
        selected_index = valid_object_index(selected_index, objects)
        if self.use_mock or getattr(self, "_related_object_model", None) is None:
            return lexical_related_object_options(task_text, objects, selected_index=selected_index)

        from langchain_core.messages import HumanMessage, SystemMessage

        listing_lines = []
        for idx, obj in enumerate(objects, 1):
            listing_lines.append(f"{idx}. label={obj.get('label','')}")
        listing = "\n".join(listing_lines)
        system = RELATED_OBJECT_SELECTION_SYSTEM_PROMPT
        human = RELATED_OBJECT_SELECTION_HUMAN_TEMPLATE.format(
            listing=listing,
            task_text=task_text,
        )
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                getattr(self, "_related_object_model").invoke,
                [SystemMessage(content=system), HumanMessage(content=human)],
            )
            raw_indices = list(getattr(result, "related_object_indices", []) or [])
        except Exception as exc:
            logger.warning("[input_node] related-object LLM failed; using lexical fallback: %s", exc)
            return lexical_related_object_options(task_text, objects, selected_index=selected_index)

        ordered_indices: list[int] = []
        seeded_indices = [selected_index] if selected_index else []
        for raw_idx in [*seeded_indices, *raw_indices]:
            idx = valid_object_index(raw_idx, objects)
            if idx and idx not in ordered_indices:
                ordered_indices.append(idx)
        options: list[dict[str, Any]] = []
        for rank, idx in enumerate(ordered_indices):
            reason = "selected_object" if idx == selected_index else "llm_related_config_description"
            score = 100 if idx == selected_index else max(60, 95 - rank)
            options.append(object_option(objects[idx - 1], idx, reason=reason, score=score))
        return options

    async def _input_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        objects = load_graspable_objects()
        existing_request = str(state.get("human_reply", "") or "").strip()
        task_text = existing_request
        if not task_text:
            status = "TASK_OBJECT_NOT_SELECTED"
            return {
                "task_intent": "general_chat",
                "ai_reply": "我還沒有收到要抓取的目標描述，請直接說出目標物名稱。",
                "current_status": status,
                "last_execution": self._execution(state, "input_node", status, started, success=False, message="empty task text"),
            }
        normalized_task = task_text
        task = TaskContext(
            task_id=uuid.uuid4().hex,
            original_user_request=task_text,
            normalized_task=normalized_task,
            task_type="pick_and_place",
            success_criteria=["目標物已被抓取", "手臂與夾爪收尾完成", "任務完成後返回 home"],
            done_policy="When approach_result.success is true and arm_result.success is true, the task can be marked DONE and routed home.",
        )
        status = "INPUT_RECEIVED"
        print(f"\n任務已確認：{normalized_task}")
        candidate_options = await self._resolve_related_object_options(task_text, objects)
        candidate_ids = [str(option.get("id", "")) for option in candidate_options if str(option.get("id", "")).strip()]
        candidate_labels = {str(option.get("id", "")): str(option.get("label", option.get("id", ""))) for option in candidate_options if str(option.get("id", "")).strip()}
        primary_option = candidate_options[0] if candidate_options else {}
        object_id = str(primary_option.get("id", ""))
        label = str(primary_option.get("label", object_id or task_text))
        requested = RequestedObject(
            id=object_id,
            label=label,
            candidate_ids=candidate_ids,
            candidate_labels=candidate_labels,
            candidate_match_notes=candidate_options,
        )
        if candidate_ids:
            related_text = "、".join(f"{candidate_labels[obj_id]}({obj_id})" for obj_id in candidate_ids)
            print(f"LLM 依 objects.yaml 找到相關物品候選：{related_text}")
        else:
            print("LLM 沒有從 objects.yaml 找到相關物品候選。")
        return {
            "task": dump_model(task),
            "requested_object": dump_model(requested),
            "current_status": status,
            "last_execution": self._execution(state, "input_node", status, started, message=normalized_task),
        }

    async def _capture_room_camera_images(self, state: CommanderState, camera_names: list[str], timeout_sec: float = 10.0) -> dict[str, dict[str, Any]]:
        from ..camera.registry import room_camera_topic
        from ..perception.room_topics import get_compressed_image_topic_base64

        snapshots: dict[str, dict[str, Any]] = {}
        seen: set[str] = set()
        ordered = []
        for camera_name in camera_names:
            camera_text = str(camera_name).strip()
            if not camera_text or camera_text in seen:
                continue
            topic = room_camera_topic(camera_text)
            if not topic:
                continue
            seen.add(camera_text)
            ordered.append((camera_text, topic))
        results = await asyncio.gather(
            *(get_compressed_image_topic_base64(topic, timeout_sec=timeout_sec) for _, topic in ordered)
        )
        for (camera_name, topic), encoded in zip(ordered, results):
            if not encoded:
                continue
            image_key = self._remember_transient_base64(state, "room_camera_rgb", encoded)
            snapshots[camera_name] = dump_model(RoomCameraSnapshot(camera_name=camera_name, topic=topic, image_key=image_key))
        return snapshots

    @staticmethod
    def _item_info_room_camera_names() -> list[str]:
        return configured_room_cameras()

    def _pick_primary_room_camera(self, candidate: Dict[str, Any], room_cameras: dict[str, Any]) -> tuple[str, list[float]]:
        from ..perception.world_position import bbox_area

        best_camera = ""
        best_bbox: list[float] = []
        best_area = -1.0
        bboxes = candidate.get("bboxes_by_camera", {}) or {}
        for camera_name in candidate.get("camsrc", []) or []:
            if camera_name not in room_cameras:
                continue
            bbox = bboxes.get(camera_name)
            if isinstance(bbox, list) and len(bbox) == 4:
                area = bbox_area(bbox)
                if area > best_area:
                    best_camera, best_bbox, best_area = camera_name, [float(v) for v in bbox], area
        return best_camera, best_bbox

    def _prepare_candidate_previews(
        self,
        *,
        state: CommanderState,
        matches: list[Dict[str, Any]],
        room_cameras: dict[str, Any],
        store: ArtifactStore,
    ) -> dict[str, dict[str, Any]]:
        if not room_cameras:
            return {}
        from ..perception.room_topics import save_preview_bbox_annotated

        preview_dir = store.artifacts_dir / "find_candidates"
        prepared: dict[str, dict[str, Any]] = {}
        for idx, candidate in enumerate(matches, 1):
            instance_key = str(candidate.get("instance_key", f"candidate_{idx}"))
            primary_camera, primary_bbox = self._pick_primary_room_camera(candidate, room_cameras)
            if not primary_camera or not primary_bbox or primary_camera not in room_cameras:
                continue
            image_key = str((room_cameras.get(primary_camera) or {}).get("image_key", ""))
            if not image_key:
                continue
            try:
                image_b64 = self._load_transient_base64(state, image_key)
            except Exception as exc:
                logger.warning("[find_node] Failed to load transient room camera image for preview: %s", exc)
                continue
            safe_camera = re.sub(r"[^A-Za-z0-9_.-]+", "_", primary_camera)
            safe_instance = re.sub(r"[^A-Za-z0-9_.-]+", "_", instance_key)
            preview_file = preview_dir / f"{idx:02d}__{safe_instance}__{safe_camera}.jpg"
            if not save_preview_bbox_annotated(image_b64, primary_bbox, preview_file):
                continue
            preview_ref = store.register_file(
                "find_candidates",
                preview_file,
                created_by_node="find_node",
                metadata={
                    "instance_key": instance_key,
                    "camera_name": primary_camera,
                    "selection_index": idx,
                },
                mime_type="image/jpeg",
            )
            prepared[instance_key] = {
                "selection_index": idx,
                "primary_camera": primary_camera,
                "primary_bbox": primary_bbox,
                "preview_path": str(store.resolve_path(preview_ref)),
                "preview_ref": preview_ref,
            }
        return prepared

    async def _find_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        requested = state.get("requested_object", {}) or {}
        requested_id = str(requested.get("id") or requested.get("label") or "").strip()
        label = str(requested.get("label") or requested_id or "目標物")
        store = self._artifact_store(state)
        from ..perception.world_position import normalized_item_id, parse_world_position_payload

        candidate_ids = requested.get("candidate_ids") if isinstance(requested.get("candidate_ids"), list) else []
        candidate_labels = requested.get("candidate_labels") if isinstance(requested.get("candidate_labels"), dict) else {}
        wanted = normalized_item_id(requested_id or label)
        wanted_ids = {normalized_item_id(value) for value in candidate_ids if str(value or "").strip()}
        if wanted:
            wanted_ids.add(wanted)
        if not wanted_ids:
            wanted_ids = {wanted or "target"}
        label_lookup = {normalized_item_id(key): str(value) for key, value in candidate_labels.items()}
        if wanted:
            label_lookup.setdefault(wanted, label)
        if self.use_mock:
            raw_payload = {"data": json.dumps({"mock": []})}
            candidates = [{"item_id": wanted or "target", "instance_id": 1, "instance_key": f"{wanted or 'target'}_1", "topic_key": "mock", "center_world": [1.2, 0.4, 2.8], "camsrc": [], "bboxes_by_camera": {}}]
        else:
            from ..perception.room_topics import get_topic_string_message
            raw = await get_topic_string_message("/world_position_data", timeout_sec=5.0)
            if not raw:
                status = "TARGET_NOT_FOUND"
                return {"selected_instance": {}, "current_status": status, "last_execution": self._execution(state, "find_node", status, started, success=False, error="/world_position_data read failed")}
            raw_payload = {"data": raw}
            candidates = parse_world_position_payload(raw_payload)
        room_cameras = {} if self.use_mock else await self._capture_room_camera_images(state, self._item_info_room_camera_names(), timeout_sec=10.0)
        matches = [candidate for candidate in candidates if candidate.get("item_id") in wanted_ids]
        if not matches:
            status = "TARGET_NOT_FOUND"
            updated_at = time.time()
            snapshot_id = store.save_world_snapshot_raw(raw_payload, candidate_count=len(candidates), created_by_node="find_node", update_reason="no_matching_instance", updated_at=updated_at, store_raw_payload=False)
            world = WorldPositionSnapshot(snapshot_id=snapshot_id, candidate_count=len(candidates), updated_at=updated_at, update_source_node="find_node", update_reason="no_matching_instance")
            missing = ", ".join(sorted(wanted_ids))
            return {"selected_instance": {}, "world_position": dump_model(world), "room_cameras": room_cameras, "current_status": status, "last_execution": self._execution(state, "find_node", status, started, success=False, message=f"No instance for {label}; checked [{missing}]")}

        candidate_previews = self._prepare_candidate_previews(
            state=state,
            matches=matches,
            room_cameras=room_cameras,
            store=store,
        )

        print("\nworld_position_data 候選照片：")
        for idx, candidate in enumerate(matches, 1):
            preview = candidate_previews.get(str(candidate.get("instance_key", "")), {})
            preview_note = f" preview={preview.get('preview_path', '')}" if preview else " preview=unavailable"
            item_id = str(candidate.get("item_id", ""))
            relation = "同種物品" if item_id == wanted else "相近不同物品"
            object_label = label_lookup.get(item_id, item_id)
            print(f"  [{idx}] {candidate.get('instance_key')} {relation}={object_label} center_world={candidate.get('center_world', [])} camsrc={candidate.get('camsrc', [])}{preview_note}")
        if self.use_mock and len(matches) == 1:
            selected_idx = 1
        else:
            loop = asyncio.get_event_loop()
            while True:
                choice = await loop.run_in_executor(None, lambda: input(f"\n請輸入候選照片編號 (1-{len(matches)}) 或 no：\n> "))
                if choice.strip().lower() == "no":
                    status = "TARGET_NOT_FOUND"
                    updated_at = time.time()
                    snapshot_id = store.save_world_snapshot_raw(raw_payload, candidate_count=len(candidates), created_by_node="find_node", update_reason="user_rejected", updated_at=updated_at, store_raw_payload=False)
                    world = WorldPositionSnapshot(snapshot_id=snapshot_id, candidate_count=len(candidates), updated_at=updated_at, update_source_node="find_node", update_reason="user_rejected")
                    return {"selected_instance": {}, "world_position": dump_model(world), "room_cameras": room_cameras, "current_status": status, "last_execution": self._execution(state, "find_node", status, started, success=False, message="User selected no target")}
                try:
                    selected_idx = int(choice)
                    if 1 <= selected_idx <= len(matches):
                        break
                except ValueError:
                    pass
                print("格式不正確，請重新輸入。")
        candidate = matches[selected_idx - 1]
        preview = candidate_previews.get(str(candidate.get("instance_key", "")), {})
        primary_camera = str(preview.get("primary_camera", ""))
        primary_bbox = preview.get("primary_bbox", []) or []
        preview_ref = preview.get("preview_ref")
        preview_path = str(preview.get("preview_path", ""))
        if not primary_camera:
            primary_camera, primary_bbox = self._pick_primary_room_camera(candidate, room_cameras)
        if primary_camera and primary_camera in room_cameras:
            room_cameras[primary_camera]["bbox"] = primary_bbox
            if preview_path:
                room_cameras[primary_camera]["preview_path"] = preview_path
            if preview_ref:
                room_cameras[primary_camera]["preview_ref"] = dump_model(preview_ref)
        selected = SelectedInstance(
            item_id=str(candidate.get("item_id", "")),
            instance_id=int(candidate.get("instance_id", -1)),
            instance_key=str(candidate.get("instance_key", "")),
            topic_key=str(candidate.get("topic_key", "")),
            center_world=[float(v) for v in candidate.get("center_world", [])],
            camsrc=[str(v) for v in candidate.get("camsrc", []) or []],
            bboxes_by_camera={str(k): [float(x) for x in v] for k, v in (candidate.get("bboxes_by_camera", {}) or {}).items()},
            primary_camera=primary_camera,
            primary_bbox=primary_bbox,
            preview_ref=preview_ref,
        )
        updated_at = time.time()
        snapshot_id = store.save_world_snapshot_raw(raw_payload, candidate_count=len(candidates), selected_instance_key=selected.instance_key, created_by_node="find_node", update_reason="db_created", updated_at=updated_at)
        world = WorldPositionSnapshot(snapshot_id=snapshot_id, candidate_count=len(candidates), selected_instance_key=selected.instance_key, updated_at=updated_at, update_source_node="find_node", update_reason="db_created")
        requested_updated = dict(requested)
        selected_label = label_lookup.get(selected.item_id, selected.item_id)
        requested_updated.update({"id": selected.item_id, "label": selected_label})
        status = "TARGET_SELECTED_FROM_WORLD_POSITION"
        return {"requested_object": requested_updated, "selected_instance": dump_model(selected), "world_position": dump_model(world), "room_cameras": room_cameras, "current_status": status, "last_execution": self._execution(state, "find_node", status, started, message=selected.instance_key)}

    def _route_find(self, state: CommanderState) -> str:
        return "get_item_info_no_sam3d_node" if state.get("selected_instance") else "end"

    async def _get_item_info_no_sam3d_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        from agents.get_item_info_agent_no_sam3d import GetItemInfoNoSam3DAgent

        store = self._artifact_store(state)
        selected = state.get("selected_instance", {}) or {}
        world = state.get("world_position", {}) or {}
        room_cameras = state.get("room_cameras", {}) or {}
        if not selected or not world.get("snapshot_id"):
            status = "ITEM_INFO_NO_SAM3D_FAILED"
            return {"current_status": status, "last_execution": self._execution(state, "get_item_info_no_sam3d_node", status, started, success=False, error="missing selected instance or world snapshot")}
        camera_names = [] if self.use_mock else self._item_info_room_camera_names()
        if not self.use_mock:
            latest_room_cameras = await self._capture_room_camera_images(state, camera_names, timeout_sec=10.0)
            room_cameras = {**room_cameras, **latest_room_cameras}
            missing_cameras = [name for name in camera_names if name not in room_cameras]
            if missing_cameras:
                status = "ITEM_INFO_NO_SAM3D_FAILED"
                return {
                    "room_cameras": room_cameras,
                    "current_status": status,
                    "last_execution": self._execution(
                        state,
                        "get_item_info_no_sam3d_node",
                        status,
                        started,
                        success=False,
                        error=f"missing required room camera images: {missing_cameras}",
                    ),
                }
        try:
            camera_images = {name: self._load_transient_base64(state, str(room_cameras[name].get("image_key", ""))) for name in camera_names}
        except Exception as exc:
            status = "ITEM_INFO_NO_SAM3D_FAILED"
            return {"room_cameras": room_cameras, "current_status": status, "last_execution": self._execution(state, "get_item_info_no_sam3d_node", status, started, success=False, error=f"missing transient room camera image: {exc}")}
        raw_world = store.load_world_snapshot_raw(str(world["snapshot_id"]))
        params = {
            "target_item_id": selected.get("item_id"),
            "target_instance_id": selected.get("instance_id"),
            "target_instance_key": selected.get("instance_key"),
            "target_topic_key": selected.get("topic_key"),
            "target_label": (state.get("requested_object") or {}).get("label", selected.get("item_id", "")),
            "selected_camera": selected.get("primary_camera", ""),
            "camera_names": camera_names,
            "camera_images": camera_images,
            "center_world": selected.get("center_world", []),
            "bboxes_by_camera": selected.get("bboxes_by_camera", {}),
            "world_position_data": raw_world,
        }
        agent = GetItemInfoNoSam3DAgent(http_client=self.http_client, use_mock=self.use_mock)
        result = await agent.execute(params, state.get("context_id", ""))
        success = bool(result.get("success", False))
        payload = result.get("result", {}) if isinstance(result.get("result", {}), dict) else {}
        if not success or not payload:
            status = "ITEM_INFO_NO_SAM3D_FAILED"
            return {"item_info": dump_model(ItemInfoResult()), "current_status": status, "last_execution": self._execution(state, "get_item_info_no_sam3d_node", status, started, success=False, error="empty item-info result")}
        payload_target = payload.get("target_object", {}) if isinstance(payload.get("target_object", {}), dict) else {}
        payload_instance_key = str(payload.get("target_instance_key") or payload_target.get("instance_key") or "")
        selected_instance_key = str(selected.get("instance_key", "") or "")
        payload_instance_id = payload.get("target_instance_id", payload_target.get("instance_id", payload_target.get("id")))
        selected_instance_id = selected.get("instance_id")
        try:
            payload_instance_id_int = int(payload_instance_id)
            selected_instance_id_int = int(selected_instance_id)
        except (TypeError, ValueError):
            payload_instance_id_int = None
            selected_instance_id_int = None
        if not payload_instance_key and payload_instance_id_int is not None and selected.get("item_id"):
            payload_instance_key = f"{selected.get('item_id')}_{payload_instance_id_int}"
        instance_key_mismatch = bool(payload_instance_key and selected_instance_key and payload_instance_key != selected_instance_key)
        instance_id_mismatch = bool(
            payload_instance_id_int is not None
            and selected_instance_id_int is not None
            and payload_instance_id_int != selected_instance_id_int
        )
        if instance_key_mismatch or instance_id_mismatch:
            status = "ITEM_INFO_NO_SAM3D_FAILED"
            message = f"item-info target mismatch: selected={selected_instance_key or selected_instance_id}, got={payload_instance_key or payload_instance_id}"
            return {
                "item_info": dump_model(ItemInfoResult()),
                "current_status": status,
                "last_execution": self._execution(
                    state,
                    "get_item_info_no_sam3d_node",
                    status,
                    started,
                    success=False,
                    error=message,
                ),
            }

        group_ranking = payload.get("group_ranking", []) or []
        item_info = ItemInfoResult(
            center_world=[float(v) for v in payload.get("center_world", selected.get("center_world", []))],
            center_world_coordinate_frame=str(payload.get("center_world_coordinate_frame", "unity_world")),
            primary_camera_id=str(payload.get("primary_camera_id", selected.get("primary_camera", ""))),
            target_instance_key=str(payload.get("target_instance_key", selected.get("instance_key", ""))),
            target_topic_key=str(payload.get("target_topic_key", selected.get("topic_key", ""))),
            group_ranking=group_ranking,
            goal_pose_path=str(payload.get("goal_pose_path", "")),
            a2a_task_id=str(result.get("a2a_task_id", "")),
        )
        goal_pose_db = self._goal_pose_db_from_item_info(item_info.model_dump(mode="json"), 1)
        nav_goal, err = self._goal_pose_for_rank(item_info.model_dump(mode="json"), 1)
        navigation = NavigationState(
            current_goal_rank=1,
            current_goal_pose_index=0,
            goal_pose_db=goal_pose_db,
            nav_goal=self._nav_goal_from_pose(nav_goal) if not err else None,
            nav_goal_pose_source="rank_best",
            nav_move_source="bootstrap",
            force_initialpose=state.get("navigation", {}).get("result") in ({}, None),
        )
        status = "ITEM_INFO_NO_SAM3D_READY"
        return {"item_info": dump_model(item_info), "navigation": dump_model(navigation), "room_cameras": room_cameras, "current_status": status, "last_execution": self._execution(state, "get_item_info_no_sam3d_node", status, started, success=not bool(err), message="item info ready" if not err else err)}

    def _route_get_item_info_no_sam3d(self, state: CommanderState) -> Literal["nav_move_node", "nav_home_node"]:
        navigation = state.get("navigation", {}) or {}
        return "nav_move_node" if state.get("current_status") == "ITEM_INFO_NO_SAM3D_READY" and navigation.get("nav_goal") else "nav_home_node"

    async def _update_item_info_node(self, state: CommanderState, source_node: str) -> Dict[str, Any]:
        started = time.time()
        selected = state.get("selected_instance", {}) or {}
        if not selected:
            status = "WORLD_POSITION_UPDATE_SKIPPED"
            world = WorldPositionSnapshot(update_source_node=source_node, update_reason="no_selected_instance")
            return {"world_position": dump_model(world), "current_status": status, "last_execution": self._execution(state, source_node, status, started)}
        if self.use_mock:
            status = "WORLD_POSITION_UNCHANGED"
            world = dict(state.get("world_position", {}) or {})
            world["snapshot_id"] = ""
            world.update({"target_changed": False, "update_source_node": source_node, "update_reason": "unchanged", "update_distance_m": 0.0})
            return {"world_position": world, "current_status": status, "last_execution": self._execution(state, source_node, status, started)}
        from ..perception.room_topics import get_topic_string_message
        from ..perception.world_position import parse_world_position_payload
        raw = await get_topic_string_message("/world_position_data", timeout_sec=5.0)
        if not raw:
            status = "WORLD_POSITION_UPDATE_READ_FAILED"
            world = WorldPositionSnapshot(update_source_node=source_node, update_reason="read_failed")
            return {"world_position": dump_model(world), "current_status": status, "last_execution": self._execution(state, source_node, status, started, success=False)}
        store = self._artifact_store(state)
        raw_payload = {"data": raw}
        candidates = parse_world_position_payload(raw_payload)
        refreshed, match_info = self._find_selected_candidate(candidates, selected)
        if not refreshed:
            status = "TARGET_LOST_IN_WORLD_POSITION"
            world = WorldPositionSnapshot(candidate_count=len(candidates), selected_instance_key=selected.get("instance_key", ""), updated_at=time.time(), target_changed=False, update_source_node=source_node, update_reason="target_missing")
            return {"world_position": dump_model(world), "navigation": {"nav_goal": {}, "goal_pose_db": {}, "result": {}}, "grasp_result": {}, "current_status": status, "last_execution": self._execution(state, source_node, status, started, success=False, message="target missing")}
        moved = self._center_world_distance_m(selected.get("center_world", []), refreshed.get("center_world", []))
        threshold = world_position_update_threshold_m()
        target_changed = moved > threshold
        id_reassigned = bool(match_info.get("id_reassigned", False))
        updated_at = time.time()
        update_reason = "target_moved" if target_changed else ("same_target_id_reassigned" if id_reassigned else "unchanged")
        refreshed_key = str(refreshed.get("instance_key", selected.get("instance_key", "")))
        snapshot_id = ""
        if target_changed:
            snapshot_id = store.save_world_snapshot_raw(raw_payload, candidate_count=len(candidates), selected_instance_key=refreshed_key, created_by_node=source_node, target_changed=True, update_reason=update_reason, update_distance_m=moved, updated_at=updated_at)
        world = WorldPositionSnapshot(snapshot_id=snapshot_id, candidate_count=len(candidates), selected_instance_key=refreshed_key, updated_at=updated_at, target_changed=target_changed, update_source_node=source_node, update_distance_m=moved, update_reason=update_reason)
        status = "WORLD_POSITION_TARGET_MOVED" if target_changed else "WORLD_POSITION_UNCHANGED"
        update: dict[str, Any] = {"world_position": dump_model(world), "current_status": status, "last_execution": self._execution(state, source_node, status, started, message=update_reason)}
        if target_changed or id_reassigned:
            selected_updated = dict(selected)
            selected_updated.update({
                "item_id": refreshed.get("item_id", selected.get("item_id", "")),
                "instance_id": int(refreshed.get("instance_id", selected.get("instance_id", -1))),
                "instance_key": refreshed_key,
                "topic_key": refreshed.get("topic_key", selected.get("topic_key", "")),
                "center_world": refreshed.get("center_world", selected.get("center_world", [])),
                "camsrc": refreshed.get("camsrc", selected.get("camsrc", [])),
                "bboxes_by_camera": refreshed.get("bboxes_by_camera", selected.get("bboxes_by_camera", {})),
            })
            update["selected_instance"] = selected_updated
        if id_reassigned and not target_changed:
            item_info_updated = dict(state.get("item_info", {}) or {})
            if item_info_updated:
                item_info_updated["target_instance_key"] = refreshed_key
                item_info_updated["target_topic_key"] = refreshed.get("topic_key", item_info_updated.get("target_topic_key", ""))
                update["item_info"] = item_info_updated
        if target_changed:
            update.update({"item_info": {}, "navigation": {"current_goal_rank": 1, "current_goal_pose_index": 0, "goal_pose_db": {}, "nav_goal": {}, "result": {}}, "grasp_result": {}})
        return update

    async def _update_item_info_1_node(self, state: CommanderState) -> Dict[str, Any]:
        return await self._update_item_info_node(state, "update_item_info_1_node")

    async def _update_item_info_2_node(self, state: CommanderState) -> Dict[str, Any]:
        return await self._update_item_info_node(state, "update_item_info_2_node")

    def _route_update_item_info_1(self, state: CommanderState) -> str:
        world = state.get("world_position", {}) or {}
        if world.get("update_reason") == "target_missing":
            return "nav_home_node"
        return "get_item_info_no_sam3d_node" if world.get("target_changed") else "major_nav_node"

    def _route_update_item_info_2(self, state: CommanderState) -> str:
        world = state.get("world_position", {}) or {}
        if world.get("update_reason") == "target_missing":
            return "nav_home_node"
        return "get_item_info_no_sam3d_node" if world.get("target_changed") else "car_grasp_node"

    @staticmethod
    def _find_selected_candidate(candidates: list[dict[str, Any]], selected: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        key = str(selected.get("instance_key", ""))
        for candidate in candidates:
            if str(candidate.get("instance_key", "")) == key:
                return candidate, {"match_mode": "exact", "id_reassigned": False, "distance_m": 0.0, "instance_id_delta": 0}

        selected_item_id = str(selected.get("item_id", ""))
        try:
            selected_instance_id = int(selected.get("instance_id"))
        except (TypeError, ValueError):
            return None, {"match_mode": "missing"}

        fallback_matches: list[tuple[float, int, dict[str, Any]]] = []
        for candidate in candidates:
            if str(candidate.get("item_id", "")) != selected_item_id:
                continue
            try:
                candidate_instance_id = int(candidate.get("instance_id"))
            except (TypeError, ValueError):
                continue
            instance_delta = abs(candidate_instance_id - selected_instance_id)
            if instance_delta > 1:
                continue
            distance_m = PickFlowMixin._center_world_distance_m(selected.get("center_world", []), candidate.get("center_world", []))
            if distance_m <= 0.03:
                fallback_matches.append((distance_m, instance_delta, candidate))

        if not fallback_matches:
            return None, {"match_mode": "missing"}
        distance_m, instance_delta, candidate = min(fallback_matches, key=lambda item: (item[0], item[1]))
        return candidate, {
            "match_mode": "nearby_same_item_id",
            "id_reassigned": True,
            "distance_m": distance_m,
            "instance_id_delta": instance_delta,
            "previous_instance_key": key,
            "new_instance_key": str(candidate.get("instance_key", "")),
        }

    async def _observe_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        task = state.get("task", {}) or {}
        base_description = task.get("normalized_task") or task.get("original_user_request") or "抓取目標物件"
        description = str(base_description)
        image_key = ""
        image_ref = None
        status = "OBSERVED"
        success = True
        message = "Camera_Car observation captured"
        if not self.use_mock:
            from ..camera import get_camera_image_base64
            image_b64 = await get_camera_image_base64("Camera_Car", timeout_sec=15.0)
            if image_b64:
                image_key = self._remember_transient_base64(state, "camera_car_rgb", image_b64)
            else:
                status = "OBSERVED_WITHOUT_IMAGE"
                success = False
                message = "Camera_Car capture timed out or returned no image"
                description = f"{description}. Camera_Car image unavailable; reason over typed task/navigation/result state only."
        observation = Observation(description=description, image_key=image_key, image_ref=image_ref)
        return {"observation": dump_model(observation), "current_status": status, "last_execution": self._execution(state, "observe_node", status, started, success=success, message=message)}

    async def _reason_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        try:
            out = await self.brain.reason(state, artifact_store=self._artifact_store(state), image_loader=lambda key: self._load_transient_base64(state, key))
        except Exception as exc:
            logger.error("[reason_node] structured Brain decision failed: %s", exc, exc_info=True)
            status = "REASON_FAILED"
            record = DecisionRecord(
                reasoning=f"Brain structured output failed: {exc}",
                call_module="",
                module_params={},
                latency_sec=time.time() - started,
                model=self.brain._model_name(),
            )
            return {
                "decision": dump_model(record),
                "module_params": {},
                "current_status": status,
                "last_execution": self._execution(state, "reason_node", status, started, success=False, error=str(exc)),
            }
        decision = self._apply_decision_safety_guard(state, out["prediction"])
        record = DecisionRecord(reasoning=decision.reasoning, call_module=decision.call_module, module_params=decision.module_params, latency_sec=float(out["latency"]), model=out.get("model", ""))
        status = "REASONED"
        return {"decision": dump_model(record), "module_params": decision.module_params, "current_status": status, "last_execution": self._execution(state, "reason_node", status, started, message=decision.reasoning)}


    def _apply_decision_safety_guard(self, state: CommanderState, decision: BrainDecision) -> BrainDecision:
        if decision.call_module not in {"grasp_agent", "car_approach_agent"}:
            return decision
        should_override, reason = self._should_force_major_nav_after_failed_attempt(state)
        if not should_override:
            return decision
        params = dict(decision.module_params or {})
        params["overridden_from"] = decision.call_module
        params["override_reason"] = reason
        return BrainDecision(
            reasoning=f"Safety override: {reason}",
            call_module="major_nav_node",
            module_params=params,
        )

    def _should_force_major_nav_after_failed_attempt(self, state: CommanderState) -> tuple[bool, str]:
        navigation = state.get("navigation", {}) or {}
        item_info = state.get("item_info", {}) or {}
        current_rank = int(navigation.get("current_goal_rank", 1) or 1)
        _, next_goal_error = self._goal_pose_for_rank(item_info, current_rank + 1)
        if next_goal_error:
            return False, ""

        approach = state.get("approach_result", {}) or {}
        if approach and approach.get("success") is False:
            phase = str(approach.get("phase", "") or approach.get("status_code", "") or "approach_failed")
            return True, f"latest approach attempt failed at rank {current_rank} ({phase}); use next ranked goal pose before repeating grasp"

        grasp = state.get("grasp_result", {}) or {}
        if grasp and grasp.get("success") is False:
            return True, f"latest grasp attempt failed at rank {current_rank}; use next ranked goal pose before repeating grasp"

        return False, ""

    def _route_decision(self, state: CommanderState) -> Literal["major_nav_node", "car_grasp_node", "end"]:
        module = (state.get("decision", {}) or {}).get("call_module", "")
        if module == "DONE" or state.get("task_complete", False):
            return "end"
        if module in {"nav_agent", "major_nav_agent", "major_nav_node"}:
            return "major_nav_node"
        if module in {"grasp_agent", "approach_agent", "car_approach_agent"}:
            return "car_grasp_node"
        return "end"

    async def _major_nav_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        navigation = state.get("navigation", {}) or {}
        current_rank = int(navigation.get("current_goal_rank", 1) or 1)
        next_rank = current_rank + 1
        goal, err = self._goal_pose_for_rank(state.get("item_info", {}) or {}, next_rank)
        if err:
            status = "MAJOR_NAV_EXHAUSTED"
            return {"task_complete": True, "current_status": status, "last_execution": self._execution(state, "major_nav_node", status, started, success=False, message=err)}
        nav_state = NavigationState(**{**navigation, "current_goal_rank": next_rank, "current_goal_pose_index": 0, "nav_goal": self._nav_goal_from_pose(goal), "nav_goal_pose_source": "major_nav", "nav_move_source": "major_nav", "force_initialpose": False})
        status = "MAJOR_NAV_CONTEXT_READY"
        return {"navigation": dump_model(nav_state), "current_status": status, "last_execution": self._execution(state, "major_nav_node", status, started, message=f"rank {next_rank}")}

    def _route_major_nav(self, state: CommanderState) -> Literal["nav_move_node", "nav_home_node"]:
        return "nav_home_node" if state.get("task_complete") or state.get("current_status") == "MAJOR_NAV_EXHAUSTED" else "nav_move_node"

    def _route_nav_move(self, state: CommanderState) -> Literal["observe_node", "update_memory_node"]:
        return "observe_node" if (state.get("navigation", {}) or {}).get("nav_move_source") == "bootstrap" else "update_memory_node"

    def _route_car_approach(self, state: CommanderState) -> Literal["end", "update_memory_node"]:
        if (state.get("approach_result", {}) or {}).get("success") is True:
            return "end"
        return "update_memory_node"

    async def _nav_move_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        navigation = dict(state.get("navigation", {}) or {})
        goal = navigation.get("nav_goal") or {}
        if not goal:
            status = "NAV_FAILED"
            nav_result = NavResult(arrived=False, plan_ready=False, message="missing nav goal")
            navigation["result"] = dump_model(nav_result)
            return {"navigation": navigation, "current_status": status, "last_execution": self._execution(state, "nav_move_node", status, started, success=False, error="missing nav goal")}
        rank = int(navigation.get("current_goal_rank", goal.get("goal_rank", 1)) or 1)
        goal_pose_index = int(navigation.get("current_goal_pose_index", goal.get("goal_pose_index", 0)) or 0)
        if self.use_mock:
            events = [{"event": "goal_publishing", "rank": rank, "goal_pose_index": goal_pose_index}, {"event": "plan_ready", "rank": rank}, {"event": "arrived", "rank": rank}]
            nav_result = NavResult(goal=self._nav_goal_from_pose(goal), arrived=True, plan_ready=True, attempt=1, events=events, message="mock nav arrived")
            navigation["result"] = dump_model(nav_result)
            status = "NAV_COMPLETED"
            return {"navigation": navigation, "current_status": status, "last_execution": self._execution(state, "nav_move_node", status, started, message=nav_result.message)}
        payload = self._nav_runner_payload(state, goal, rank, goal_pose_index)
        result = await self._run_nav_move_runner(payload)
        success = bool(result.get("success", False))
        nav_result = NavResult(goal=self._nav_goal_from_pose(goal), arrived=success, plan_ready=bool(result.get("plan_ready", False)), attempt=int(payload.get("attempt", 1)), events=result.get("events", []) or [], message=str(result.get("message", "")))
        navigation["result"] = dump_model(nav_result)
        status = "NAV_COMPLETED" if success else "NAV_FAILED"
        return {"navigation": navigation, "current_status": status, "last_execution": self._execution(state, "nav_move_node", status, started, success=success, message=nav_result.message)}

    def _nav_runner_payload(self, state: CommanderState, goal: dict[str, Any], rank: int, goal_pose_index: int) -> dict[str, Any]:
        payload = nav_runner_payload_defaults()
        payload.update(
            {
                "goal_pose": goal,
                "publish_initialpose": bool((state.get("navigation", {}) or {}).get("force_initialpose", False)),
                "initial_pose": default_initial_pose(),
                "attempt": 1,
                "rank": rank,
                "goal_pose_index": goal_pose_index,
                "goal_pose_source": goal.get("goal_pose_source", ""),
                "source": (state.get("navigation", {}) or {}).get("nav_move_source", "reason_loop"),
            }
        )
        return payload

    async def _nav_home_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        goal = self._default_initial_pose()
        goal.pop("covariance", None)
        if self.use_mock:
            events = [{"event": "goal_publishing", "source": "nav_home"}, {"event": "arrived", "source": "nav_home"}]
            success = True
            message = "mock home arrived"
        else:
            payload = self._nav_runner_payload({**state, "navigation": {"force_initialpose": False, "nav_move_source": "nav_home"}}, goal, 0, 0)
            payload["status_topic"] = str(payload.pop("home_status_topic"))
            result = await self._run_nav_move_runner(payload)
            events = result.get("events", []) or []
            success = bool(result.get("success", False))
            message = str(result.get("message", "home navigation completed" if success else "home navigation failed"))
        nav_goal = self._nav_goal_from_pose(goal)
        nav_result = NavResult(goal=nav_goal, arrived=success, plan_ready=success, attempt=1, events=events, message=message)
        navigation = {**(state.get("navigation", {}) or {}), "nav_move_source": "nav_home", "nav_goal": dump_model(nav_goal), "result": dump_model(nav_result)}
        status = "NAV_HOME_COMPLETED" if success else "NAV_HOME_FAILED"
        return {"navigation": navigation, "task_complete": True, "current_status": status, "last_execution": self._execution(state, "nav_home_node", status, started, success=success, message=message)}

    async def _car_grasp_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        from agents.grasp_agent import GraspAgent
        from ..camera import get_camera_rgbd_base64

        selected_for_object = state.get("selected_instance", {}) or {}
        object_id = selected_for_object.get("item_id") or (state.get("requested_object", {}) or {}).get("id", "")
        if not object_id:
            status = "GRASP_FAILED"
            return {"grasp_result": dump_model(GraspResult(success=False)), "current_status": status, "last_execution": self._execution(state, "car_grasp_node", status, started, success=False, error="missing object id")}
        selected = state.get("selected_instance", {}) or {}
        item_info = state.get("item_info", {}) or {}
        target_center_world = selected.get("center_world") or item_info.get("center_world") or []
        target_instance_key = str(selected.get("instance_key") or item_info.get("target_instance_key") or "")
        rgbd: dict[str, str] = {}
        amcl_pose: dict[str, Any] | None = None
        if self.use_mock:
            rgbd = {"camera_name": "Camera_Car", "rgb_base64": "", "depth_base64": ""}
        else:
            from ..perception.room_topics import get_amcl_pose

            rgbd, amcl_pose = await asyncio.gather(
                get_camera_rgbd_base64("Camera_Car", timeout_sec=15.0),
                get_amcl_pose(timeout_sec=5.0),
            )
            rgbd = rgbd or {}
            if not rgbd:
                status = "GRASP_FAILED"
                return {"grasp_result": dump_model(GraspResult(object_id=object_id, target_instance_key=target_instance_key, success=False)), "current_status": status, "last_execution": self._execution(state, "car_grasp_node", status, started, success=False, error="missing RGBD")}
        origin_x, origin_z = ros_map_origin_unity()
        params = {
            **(state.get("module_params", {}) or {}),
            "object_id": object_id,
            "camera_name": "Camera_Car",
            "rgb_base64": rgbd.get("rgb_base64", ""),
            "depth_base64": rgbd.get("depth_base64", ""),
            "target_center_world": target_center_world,
            "target_instance_key": target_instance_key,
            "amcl_pose": amcl_pose or {},
            "ros_map_origin_unity": {"x": origin_x, "z": origin_z},
        }
        agent = GraspAgent(http_client=self.http_client, use_mock=self.use_mock)
        result = await agent.execute(params, state.get("context_id", ""))
        success = bool(result.get("success", False))
        payload = result.get("result", {}) if isinstance(result.get("result", {}), dict) else {}
        grasp = GraspResult(
            object_id=payload.get("object_id") or object_id,
            camera_name=payload.get("camera_name", "Camera_Car"),
            success=success,
            target_instance_key=str(payload.get("target_instance_key") or target_instance_key),
            bbox_xyxy=payload.get("bbox_xyxy", []) if isinstance(payload.get("bbox_xyxy", []), list) else [],
            detection_confidence=payload.get("detection_confidence"),
            target_selection=payload.get("target_selection", {}) if isinstance(payload.get("target_selection", {}), dict) else {},
            grasp_confidence=payload.get("grasp_confidence"),
            num_candidate_grasps=payload.get("num_candidate_grasps"),
            num_valid_grasps=payload.get("num_valid_grasps"),
            best_grasp_pose_camera=payload.get("best_grasp_pose_camera", {}),
            valid_grasp_poses_camera=payload.get("valid_grasp_poses_camera", []) if isinstance(payload.get("valid_grasp_poses_camera", []), list) else [],
            object_reference_center_camera=payload.get("object_reference_center_camera", []),
            a2a_task_id=str(result.get("a2a_task_id", "")),
        )
        status = "GRASP_READY" if success else "GRASP_FAILED"
        return {"grasp_result": dump_model(grasp), "current_status": status, "last_execution": self._execution(state, "car_grasp_node", status, started, success=success)}

    async def _car_approach_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        from agents.car_approach_agent import CarApproachAgent

        params = dict(state.get("module_params", {}) or {})
        params.setdefault("grasp_result", state.get("grasp_result", {}) or {})
        agent = CarApproachAgent(use_mock=self.use_mock)
        result = await agent.execute(params, state.get("context_id", ""))
        success = bool(result.get("success", False))
        payload = result.get("result", {}) if isinstance(result.get("result", {}), dict) else {}
        approach = ApproachResult(
            success=success,
            status_code=str(payload.get("status_code", "APPROACH_SUCCESS" if success else "APPROACH_FAIL")),
            phase=str(payload.get("phase", "")),
            message=str(payload.get("message") or payload.get("error") or ""),
            next_agent=payload.get("next_agent"),
            nav_result=payload.get("nav_result", {}) if isinstance(payload.get("nav_result", {}), dict) else {},
            arm_result=payload.get("arm_result", {}) if isinstance(payload.get("arm_result", {}), dict) else {},
            car_return_result=payload.get("car_return_result", {}) if isinstance(payload.get("car_return_result", {}), dict) else {},
            arm_base_alignment_result=payload.get("arm_base_alignment_result", {}) if isinstance(payload.get("arm_base_alignment_result", {}), dict) else {},
            selected_solution=payload.get("selected_solution", {}) if isinstance(payload.get("selected_solution", {}), dict) else {},
            closest_solution=payload.get("closest_solution", {}) if isinstance(payload.get("closest_solution", {}), dict) else {},
            selected_solution_source=str(payload.get("selected_solution_source", "") or ""),
            fallback_to_closest_solution=bool(payload.get("fallback_to_closest_solution", False)),
            sampling_summary=payload.get("sampling_summary", {}) if isinstance(payload.get("sampling_summary", {}), dict) else {},
        )
        status = "APPROACH_COMPLETED" if success else "APPROACH_FAILED"
        return {"approach_result": dump_model(approach), "current_status": status, "last_execution": self._execution(state, "car_approach_node", status, started, success=success, message=approach.message)}

    async def _update_memory_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        execution = state.get("last_execution", {}) or {}
        decision = state.get("decision", {}) or {}
        action = decision.get("call_module") or execution.get("node_name", "")
        summary, facts = self._memory_summary(state)
        entry = {"action": action, "reasoning": decision.get("reasoning", ""), "result": summary, "success": bool(execution.get("success", True)), "key_facts": facts, "trace_id": uuid.uuid4().hex}
        session_summary = self._update_session_summary(state.get("session_summary", ""), entry)
        status = "MEMORY_UPDATED"
        return {"history_buffer": [entry], "session_summary": session_summary, "retry_count": int(state.get("retry_count", 0) or 0) + 1, "current_status": status, "last_execution": self._execution(state, "update_memory_node", status, started)}

    def _memory_summary(self, state: CommanderState) -> tuple[str, dict[str, Any]]:
        execution = state.get("last_execution", {}) or {}
        if state.get("approach_result"):
            payload = state["approach_result"]
            return payload.get("message") or payload.get("status_code", "approach complete"), {"approach_success": payload.get("success"), "arm_success": bool((payload.get("arm_result") or {}).get("success", False))}
        if state.get("grasp_result"):
            payload = state["grasp_result"]
            return "grasp result ready" if payload.get("success") else "grasp failed", {"grasp_confidence": payload.get("grasp_confidence"), "pose_ready": bool(payload.get("best_grasp_pose_camera"))}
        nav = (state.get("navigation", {}) or {}).get("result", {}) or {}
        if nav:
            return nav.get("message", "navigation updated"), {"arrived": nav.get("arrived"), "plan_ready": nav.get("plan_ready")}
        return execution.get("message") or execution.get("status", "node updated"), {}

    @staticmethod
    def _update_session_summary(existing: str, entry: dict[str, Any]) -> str:
        line = f"- {entry.get('action')}: {entry.get('result')} success={entry.get('success')}"
        lines = ([existing] if existing else []) + [line]
        text = "\n".join(lines)
        return text[-4000:]

    @staticmethod
    def _center_world_distance_m(a: list[Any], b: list[Any]) -> float:
        if not isinstance(a, list) or not isinstance(b, list) or len(a) < 3 or len(b) < 3:
            return 0.0
        return math.sqrt(sum((float(a[idx]) - float(b[idx])) ** 2 for idx in range(3)))

    def _goal_pose_db_from_item_info(self, item_info: dict[str, Any], current_rank: int) -> dict[str, Any]:
        return goal_pose_db_from_item_info(item_info, current_rank)

    def _target_center_world_to_map_xy(self, item_info: Dict[str, Any]) -> tuple[float | None, float | None]:
        origin_x, origin_z = ros_map_origin_unity()
        return target_center_world_to_map_xy(
            item_info,
            ros_map_origin_unity_x=origin_x,
            ros_map_origin_unity_z=origin_z,
        )

    def _goal_pose_from_ros_map(self, item_info: Dict[str, Any], goal_pose_ros: Any) -> tuple[Dict[str, Any], str]:
        origin_x, origin_z = ros_map_origin_unity()
        return goal_pose_from_ros_map(
            item_info,
            goal_pose_ros,
            ros_map_origin_unity_x=origin_x,
            ros_map_origin_unity_z=origin_z,
        )

    def _goal_pose_for_rank(self, item_info: Dict[str, Any], rank: int) -> tuple[Dict[str, Any], str]:
        origin_x, origin_z = ros_map_origin_unity()
        return goal_pose_for_rank(
            item_info,
            rank,
            ros_map_origin_unity_x=origin_x,
            ros_map_origin_unity_z=origin_z,
        )

    def _default_initial_pose(self) -> Dict[str, Any]:
        return default_initial_pose()

    async def _run_nav_move_runner(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import shlex
        safe_payload = shlex.quote(json.dumps(payload))
        ros = ros_subprocess_settings()
        ros_setup = shlex.quote(ros["ros_setup_bash"])
        overlay_setup = shlex.quote(ros["overlay_setup_bash"])
        ros_py = shlex.quote(ros["pythonpath"])
        ros_lib = shlex.quote(ros["ld_library_path"])
        python_bin = shlex.quote(ros["python_bin"])
        cmd = (
            "unset VIRTUAL_ENV PYTHONPATH PYTHONHOME && "
            f"source {ros_setup} && "
            f"source {overlay_setup} 2>/dev/null || true && "
            f"export LD_LIBRARY_PATH={ros_lib}:${{LD_LIBRARY_PATH:-}} && "
            f"PYTHONPATH={ros_py} {python_bin} -m commander.nav.move_runner --payload {safe_payload}"
        )
        proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, executable="/bin/bash")
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return {"success": False, "plan_ready": False, "message": stderr.decode().strip() or "nav_move_runner failed", "events": []}
        try:
            return json.loads(stdout.decode().strip() or "{}")
        except json.JSONDecodeError:
            return {"success": False, "plan_ready": False, "message": "Invalid nav_move_runner output", "events": []}

