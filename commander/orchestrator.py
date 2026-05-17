from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Literal

import httpx
import yaml
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from .artifact_store import ArtifactStore, artifact_ref_json
from .brain import Brain
from .contracts import (
    ApproachResult,
    DecisionRecord,
    GraspResult,
    ItemInfoResult,
    NavGoal,
    NavResult,
    NavigationState,
    NodeExecution,
    Observation,
    RequestedObject,
    RoomCameraSnapshot,
    SelectedInstance,
    TaskContext,
    WorldPositionSnapshot,
    dump_model,
)
from .logger import TraceLogger
from .nav_settings import goal_heading_tolerance_rad_default, goal_tolerance_m_default
from .state import CommanderState

logger = logging.getLogger(__name__)
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_GOODBYE_TOKENS = {"bye", "exit", "quit", "q", "再見", "掰掰", "結束"}
_PICK_TASK_PATTERN = re.compile(r"\b(pick up|pick|grab|grasp|get|fetch|take|hold)\b|抓取|拾取|拿|抓|夾|取")


def _load_graspable_objects() -> list[dict[str, Any]]:
    with (_CONFIG_DIR / "objects.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle).get("graspable_objects", [])


class TaskClassification(BaseModel):
    intent: Literal["general_chat", "specific_task"] = Field(
        description="Route to general_chat for casual conversation, specific_task for picking tasks."
    )
    selected_object_index: int = 0
    reasoning: str = ""


class ChatReply(BaseModel):
    reply: str


class Orchestrator:
    """LangGraph orchestrator with typed state slices and explicit artifacts."""

    def __init__(self, trace_logger: TraceLogger, use_mock: bool = True):
        self._init_common(trace_logger=trace_logger, use_mock=use_mock)
        checkpoint_path = self._checkpoint_path()
        self._checkpoint_conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
        self._checkpoint_cm = None
        self._checkpointer = SqliteSaver(self._checkpoint_conn)
        self._checkpointer.setup()
        self.graph = self._build_graph()

    @classmethod
    async def create(cls, trace_logger: TraceLogger, use_mock: bool = True) -> "Orchestrator":
        self = cls.__new__(cls)
        self._init_common(trace_logger=trace_logger, use_mock=use_mock)
        self._checkpoint_conn = None
        self._checkpoint_cm = AsyncSqliteSaver.from_conn_string(str(self._checkpoint_path()))
        self._checkpointer = await self._checkpoint_cm.__aenter__()
        await self._checkpointer.setup()
        self.graph = self._build_graph()
        return self

    def _init_common(self, *, trace_logger: TraceLogger, use_mock: bool) -> None:
        self.logger = trace_logger
        self.brain = Brain(use_mock=use_mock)
        self.http_client = httpx.AsyncClient(timeout=120.0)
        self.use_mock = use_mock
        self._classifier_model = None
        self._chat_model = None
        self._artifact_stores: dict[str, ArtifactStore] = {}
        if not use_mock:
            self._init_ollama_chat_models()

    @staticmethod
    def _checkpoint_path() -> Path:
        checkpoint_path = Path(os.getenv("LANGGRAPH_CHECKPOINT_DB", "logs/langgraph_checkpoints.sqlite"))
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        return checkpoint_path

    def _artifact_store(self, state: CommanderState | dict[str, Any]) -> ArtifactStore:
        context_id = str(state.get("context_id", "") or "default")
        store = self._artifact_stores.get(context_id)
        if store is None:
            store = ArtifactStore(context_id=context_id)
            self._artifact_stores[context_id] = store
        return store

    def _init_ollama_chat_models(self) -> None:
        from langchain_openai import ChatOpenAI

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        classifier_model_name = os.getenv("OLLAMA_CLASSIFIER_MODEL", "gemma3:1b")
        chat_model_name = os.getenv("OLLAMA_CHAT_MODEL", os.getenv("OLLAMA_MODEL", "gemma4:31b"))
        self._classifier_model = ChatOpenAI(
            model=classifier_model_name,
            openai_api_key="ollama",
            openai_api_base=f"{base_url}/v1",
            temperature=0,
        ).with_structured_output(TaskClassification)
        self._chat_model = ChatOpenAI(
            model=chat_model_name,
            openai_api_key="ollama",
            openai_api_base=f"{base_url}/v1",
            temperature=0.7,
        ).with_structured_output(ChatReply)

    def _build_graph(self) -> Any:
        workflow = StateGraph(CommanderState)
        workflow.add_node("greeting_node", self._greeting_node)
        workflow.add_node("human_reply_node", self._human_reply_node)
        workflow.add_node("task_classification_node", self._task_classification_node)
        workflow.add_node("ai_reply_node", self._ai_reply_node)
        workflow.add_node("chat_memory_node", self._chat_memory_node)
        workflow.add_node("goodbye_node", self._goodbye_node)
        workflow.add_node("input_node", self._input_node)
        workflow.add_node("find_node", self._find_node)
        workflow.add_node("get_item_info_no_sam3d_node", self._get_item_info_no_sam3d_node)
        workflow.add_node("update_item_info_1_node", self._update_item_info_1_node)
        workflow.add_node("update_item_info_2_node", self._update_item_info_2_node)
        workflow.add_node("nav_move_node", self._nav_move_node)
        workflow.add_node("nav_home_node", self._nav_home_node)
        workflow.add_node("observe_node", self._observe_node)
        workflow.add_node("reason_node", self._reason_node)
        workflow.add_node("major_nav_node", self._major_nav_node)
        workflow.add_node("car_grasp_node", self._car_grasp_node)
        workflow.add_node("car_approach_node", self._car_approach_node)
        workflow.add_node("update_memory_node", self._update_memory_node)

        workflow.set_entry_point("greeting_node")
        workflow.add_edge("greeting_node", "human_reply_node")
        workflow.add_edge("ai_reply_node", "chat_memory_node")
        workflow.add_edge("chat_memory_node", "human_reply_node")
        workflow.add_edge("goodbye_node", END)
        workflow.add_edge("input_node", "find_node")
        workflow.add_edge("observe_node", "reason_node")
        workflow.add_edge("nav_home_node", "goodbye_node")
        workflow.add_edge("car_grasp_node", "car_approach_node")
        workflow.add_edge("car_approach_node", "update_memory_node")
        workflow.add_edge("update_memory_node", "observe_node")

        workflow.add_conditional_edges(
            "human_reply_node",
            self._route_human_reply,
            {"task_classification_node": "task_classification_node", "goodbye_node": "goodbye_node"},
        )
        workflow.add_conditional_edges(
            "task_classification_node",
            self._route_task_classification,
            {"ai_reply_node": "ai_reply_node", "input_node": "input_node"},
        )
        workflow.add_conditional_edges(
            "find_node",
            self._route_find,
            {"get_item_info_no_sam3d_node": "get_item_info_no_sam3d_node", "end": "goodbye_node"},
        )
        workflow.add_conditional_edges(
            "get_item_info_no_sam3d_node",
            self._route_get_item_info_no_sam3d,
            {"nav_move_node": "nav_move_node", "nav_home_node": "nav_home_node"},
        )
        workflow.add_conditional_edges(
            "reason_node",
            self._route_decision,
            {"major_nav_node": "update_item_info_1_node", "car_grasp_node": "update_item_info_2_node", "end": "nav_home_node"},
        )
        workflow.add_conditional_edges(
            "update_item_info_1_node",
            self._route_update_item_info_1,
            {"get_item_info_no_sam3d_node": "get_item_info_no_sam3d_node", "nav_home_node": "nav_home_node", "major_nav_node": "major_nav_node"},
        )
        workflow.add_conditional_edges(
            "update_item_info_2_node",
            self._route_update_item_info_2,
            {"get_item_info_no_sam3d_node": "get_item_info_no_sam3d_node", "nav_home_node": "nav_home_node", "car_grasp_node": "car_grasp_node"},
        )
        workflow.add_conditional_edges(
            "major_nav_node",
            self._route_major_nav,
            {"nav_move_node": "nav_move_node", "nav_home_node": "nav_home_node"},
        )
        workflow.add_conditional_edges(
            "nav_move_node",
            self._route_nav_move,
            {"observe_node": "observe_node", "update_memory_node": "update_memory_node"},
        )
        return workflow.compile(checkpointer=self._checkpointer)

    def _execution(
        self,
        state: CommanderState,
        node_name: str,
        status: str,
        started_at: float,
        *,
        success: bool = True,
        message: str = "",
        error: str = "",
        route_to: str = "",
    ) -> dict[str, Any]:
        latency = time.time() - started_at
        self.logger.log_trace(
            agent_called=node_name,
            reasoning=message or status,
            decision_latency=latency,
            execution_latency=0.0,
            success=success,
            context_id=state.get("context_id", ""),
            trace_id=uuid.uuid4().hex,
            extra_info={"status": status, "error": error, "route_to": route_to},
        )
        return dump_model(
            NodeExecution(
                node_name=node_name,
                status=status,
                success=success,
                message=message,
                latency_sec=round(latency, 4),
                error=error,
                route_to=route_to,
            )
        )

    async def _greeting_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        if state.get("greeting_sent", False):
            status = "GREETING_SKIPPED"
            return {"current_status": status, "last_execution": self._execution(state, "greeting_node", status, started)}
        print("\n嗨～有什麼需要幫忙的嗎？", flush=True)
        status = "GREETING_SENT"
        return {"greeting_sent": True, "current_status": status, "last_execution": self._execution(state, "greeting_node", status, started)}

    async def _human_reply_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        existing_reply = str(state.get("human_reply", "") or "").strip()
        if existing_reply and not state.get("task"):
            status = "HUMAN_REPLY_PREFILLED"
            return {"current_status": status, "last_execution": self._execution(state, "human_reply_node", status, started)}
        loop = asyncio.get_event_loop()
        try:
            reply = await loop.run_in_executor(None, lambda: input("> "))
        except EOFError:
            reply = "bye"
        status = "HUMAN_REPLY_RECEIVED"
        return {"human_reply": reply.strip(), "current_status": status, "last_execution": self._execution(state, "human_reply_node", status, started)}

    async def _task_classification_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        human_reply = str(state.get("human_reply", "") or "").strip()
        objects = _load_graspable_objects()
        try:
            classification = self._mock_task_classification(human_reply, objects) if self.use_mock else await self._llm_task_classification(human_reply, objects)
        except Exception as exc:
            logger.error("[task_classification_node] classifier failed: %s", exc, exc_info=True)
            classification = self._mock_task_classification(human_reply, objects)
        selected_index = self._valid_object_index(classification.selected_object_index, objects)
        status = "TASK_CLASSIFIED"
        return {
            "task_intent": classification.intent,
            "selected_object_index": selected_index,
            "decision": dump_model(DecisionRecord(reasoning=classification.reasoning, call_module="task_classification", module_params={"selected_object_index": selected_index})),
            "current_status": status,
            "last_execution": self._execution(state, "task_classification_node", status, started, message=classification.reasoning),
        }

    async def _ai_reply_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        human_reply = str(state.get("human_reply", "") or "").strip()
        try:
            history = self._chat_history_from_history_buffer(state.get("history_buffer", []))
            reply = self._mock_ai_reply(human_reply) if self.use_mock else await self._llm_ai_reply(human_reply, history)
        except Exception as exc:
            logger.error("[ai_reply_node] chat failed: %s", exc, exc_info=True)
            reply = f"我有收到：「{human_reply}」。如果需要抓取物件，也可以直接告訴我要抓什麼。"
        print(reply, flush=True)
        status = "AI_REPLY_SENT"
        return {"ai_reply": reply, "current_status": status, "last_execution": self._execution(state, "ai_reply_node", status, started, message=reply)}

    async def _chat_memory_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        entry = {
            "type": "chat_turn",
            "action": "general_chat",
            "human": state.get("human_reply", ""),
            "ai": state.get("ai_reply", ""),
            "success": True,
            "timestamp": time.time(),
            "trace_id": uuid.uuid4().hex,
        }
        status = "CHAT_MEMORY_UPDATED"
        return {"history_buffer": [entry], "current_status": status, "last_execution": self._execution(state, "chat_memory_node", status, started)}

    async def _goodbye_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        message = "對話及任務結束，祝您有美好的一天～"
        print(f"\n{message}", flush=True)
        status = "GOODBYE_SENT"
        return {"current_status": status, "last_execution": self._execution(state, "goodbye_node", status, started, message=message)}

    def _route_human_reply(self, state: CommanderState) -> Literal["task_classification_node", "goodbye_node"]:
        return "goodbye_node" if self._is_goodbye_reply(state.get("human_reply", "")) else "task_classification_node"

    def _route_task_classification(self, state: CommanderState) -> Literal["ai_reply_node", "input_node"]:
        selected = self._valid_object_index(state.get("selected_object_index", 0), _load_graspable_objects())
        return "input_node" if state.get("task_intent") == "specific_task" and selected else "ai_reply_node"

    @staticmethod
    def _is_goodbye_reply(reply: str) -> bool:
        return reply.strip().casefold() in _GOODBYE_TOKENS

    @staticmethod
    def _valid_object_index(value: Any, objects: list[dict[str, Any]]) -> int:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            return 0
        return idx if 1 <= idx <= len(objects) else 0

    @staticmethod
    def _normalize_match_text(value: Any) -> str:
        text = str(value or "").casefold().replace("_", " ").replace("-", " ")
        text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _object_terms(cls, obj: dict[str, Any]) -> list[str]:
        raw_terms: list[Any] = [obj.get("id", ""), obj.get("label", "")]
        aliases = obj.get("aliases", [])
        if isinstance(aliases, str):
            raw_terms.append(aliases)
        elif isinstance(aliases, list):
            raw_terms.extend(aliases)
        terms: list[str] = []
        for term in raw_terms:
            normalized = cls._normalize_match_text(term)
            if normalized and normalized not in terms:
                terms.append(normalized)
        return terms

    @classmethod
    def _infer_object_index(cls, text: str, objects: list[dict[str, Any]]) -> int:
        normalized_text = cls._normalize_match_text(text)
        if not normalized_text:
            return 0
        for idx, obj in enumerate(objects, 1):
            for term in cls._object_terms(obj):
                if term and term in normalized_text:
                    return idx
        return 0

    @classmethod
    def _has_pick_task_intent(cls, text: str) -> bool:
        return bool(_PICK_TASK_PATTERN.search(cls._normalize_match_text(text)))

    @staticmethod
    def _chat_history_from_history_buffer(history_buffer: Any) -> list[dict[str, Any]]:
        if not isinstance(history_buffer, list):
            return []
        return [entry for entry in history_buffer if isinstance(entry, dict) and entry.get("type") == "chat_turn"]

    def _mock_task_classification(self, text: str, objects: list[dict[str, Any]]) -> TaskClassification:
        idx = self._infer_object_index(text, objects)
        has_task_intent = self._has_pick_task_intent(text)
        if not str(text or "").strip():
            return TaskClassification(intent="general_chat", selected_object_index=0, reasoning="Empty message.")
        if idx:
            return TaskClassification(intent="specific_task", selected_object_index=idx, reasoning="Matched object alias.")
        if has_task_intent:
            return TaskClassification(intent="general_chat", selected_object_index=0, reasoning="Detected pick/grasp intent, but no known object alias matched.")
        return TaskClassification(intent="general_chat", selected_object_index=0, reasoning="No robot picking intent or known object alias.")

    async def _llm_task_classification(self, text: str, objects: list[dict[str, Any]]) -> TaskClassification:
        if self._classifier_model is None:
            return self._mock_task_classification(text, objects)
        listing_lines = []
        for idx, obj in enumerate(objects, 1):
            aliases = obj.get("aliases", [])
            alias_text = ", ".join(str(alias) for alias in aliases) if isinstance(aliases, list) else str(aliases or "")
            listing_lines.append(f"{idx}. id={obj.get('id','')} label={obj.get('label','')} aliases=[{alias_text}]")
        listing = "\n".join(listing_lines)
        system = (
            "You route a robot commander conversation. Return intent=specific_task only when the human asks "
            "the robot to pick/grab/get/fetch/take/hold an object, or when they provide an object name as a task reply. "
            "Match the human request against object id, label, and aliases, including cross-language synonyms. "
            "selected_object_index must be the 1-based index from the object list. Return 0 only if no listed object is requested. "
            "Examples: 'The human wants to pick up the brown teddy bear.' matches the brown teddy bear/doll object; "
            "casual chat such as 'hello' is general_chat with selected_object_index=0."
        )
        human = f"Objects:\n{listing}\n\nHuman message:\n{text}"
        messages = [SystemMessage(content=system), HumanMessage(content=human)]
        result = await asyncio.get_event_loop().run_in_executor(None, self._classifier_model.invoke, messages)
        return result if isinstance(result, TaskClassification) else TaskClassification.model_validate(result)

    @staticmethod
    def _mock_ai_reply(text: str) -> str:
        return "可以，我在。你可以直接說要抓哪個物件。" if text else "我在，請告訴我要抓什麼。"

    async def _llm_ai_reply(self, text: str, history: list[dict[str, Any]]) -> str:
        if self._chat_model is None:
            return self._mock_ai_reply(text)
        messages: list[Any] = [SystemMessage(content="Reply briefly in Traditional Chinese. Use recent chat history only as context.")]
        for turn in history[-12:]:
            human = str(turn.get("human", "") or "").strip()
            ai = str(turn.get("ai", "") or "").strip()
            if human:
                messages.append(HumanMessage(content=human))
            if ai:
                messages.append(AIMessage(content=ai))
        messages.append(HumanMessage(content=text))
        result = await asyncio.get_event_loop().run_in_executor(None, self._chat_model.invoke, messages)
        return result.reply if isinstance(result, ChatReply) else str(result)

    async def _input_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        objects = _load_graspable_objects()
        existing_request = str(state.get("human_reply", "") or "").strip()
        selected_index = self._valid_object_index(state.get("selected_object_index", 0), objects)
        if not selected_index:
            status = "TASK_OBJECT_NOT_SELECTED"
            return {
                "task_intent": "general_chat",
                "ai_reply": "我還沒有判斷出要抓哪一個物件，請直接說出目標物名稱。",
                "current_status": status,
                "last_execution": self._execution(state, "input_node", status, started, success=False, message="selected_object_index missing"),
            }
        selected = objects[selected_index - 1]
        label = str(selected.get("label") or selected.get("id") or "目標物")
        object_id = str(selected.get("id") or label)
        normalized_task = f"抓取{label}"
        task_text = existing_request or normalized_task
        task = TaskContext(
            task_id=uuid.uuid4().hex,
            original_user_request=task_text,
            normalized_task=normalized_task,
            task_type="pick_and_place",
            success_criteria=["目標物已被抓取", "手臂與夾爪收尾完成", "任務完成後返回 home"],
            done_policy="When approach_result.success is true and arm_result.success is true, the task can be marked DONE and routed home.",
        )
        requested = RequestedObject(id=object_id, label=label)
        status = "INPUT_RECEIVED"
        print(f"\n任務已確認：{normalized_task}")
        return {
            "task": dump_model(task),
            "requested_object": dump_model(requested),
            "current_status": status,
            "last_execution": self._execution(state, "input_node", status, started, message=normalized_task),
        }

    async def _capture_room_camera_images(self, state: CommanderState, camera_names: list[str], timeout_sec: float = 10.0) -> dict[str, dict[str, Any]]:
        from .camera_groups import room_camera_topic
        from .room_topics import get_compressed_image_topic_base64

        store = self._artifact_store(state)
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
            ref = store.save_base64_image("room_camera_rgb", encoded, created_by_node="capture_room_camera_images", metadata={"camera_name": camera_name, "topic": topic})
            snapshots[camera_name] = dump_model(RoomCameraSnapshot(camera_name=camera_name, topic=topic, rgb_ref=ref))
        return snapshots

    @staticmethod
    def _world_position_camera_names(candidates: list[Dict[str, Any]]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            for camera_name in candidate.get("camsrc", []) or []:
                camera_text = str(camera_name).strip()
                if camera_text and camera_text not in seen:
                    seen.add(camera_text)
                    ordered.append(camera_text)
        return ordered

    def _pick_primary_room_camera(self, candidate: Dict[str, Any], room_cameras: dict[str, Any]) -> tuple[str, list[float]]:
        from .world_position import bbox_area

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
        from .room_topics import save_preview_bbox_annotated

        context_id = str(state.get("context_id", store.context_id) or store.context_id)
        preview_dir = Path("logs") / "sessions" / context_id / "find_candidates"
        prepared: dict[str, dict[str, Any]] = {}
        for idx, candidate in enumerate(matches, 1):
            instance_key = str(candidate.get("instance_key", f"candidate_{idx}"))
            primary_camera, primary_bbox = self._pick_primary_room_camera(candidate, room_cameras)
            if not primary_camera or not primary_bbox or primary_camera not in room_cameras:
                continue
            image_ref = (room_cameras.get(primary_camera) or {}).get("rgb_ref")
            if not image_ref:
                continue
            try:
                image_b64 = store.load_base64(image_ref)
            except Exception as exc:
                logger.warning("[find_node] Failed to load room camera artifact for preview: %s", exc)
                continue
            safe_camera = re.sub(r"[^A-Za-z0-9_.-]+", "_", primary_camera)
            safe_instance = re.sub(r"[^A-Za-z0-9_.-]+", "_", instance_key)
            preview_file = preview_dir / f"{idx:02d}__{safe_instance}__{safe_camera}.jpg"
            if not save_preview_bbox_annotated(image_b64, primary_bbox, preview_file):
                continue
            preview_ref = store.save_file(
                "preview_image",
                preview_file,
                created_by_node="find_node",
                metadata={
                    "selection_index": idx,
                    "instance_key": instance_key,
                    "camera_name": primary_camera,
                },
            )
            prepared[instance_key] = {
                "selection_index": idx,
                "primary_camera": primary_camera,
                "primary_bbox": primary_bbox,
                "preview_ref": preview_ref,
                "preview_path": str(preview_file),
            }
        return prepared

    async def _find_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        requested = state.get("requested_object", {}) or {}
        requested_id = str(requested.get("id") or requested.get("label") or "").strip()
        label = str(requested.get("label") or requested_id or "目標物")
        store = self._artifact_store(state)
        from .world_position import normalized_item_id, parse_world_position_payload

        wanted = normalized_item_id(requested_id or label)
        if self.use_mock:
            raw_payload = {"data": json.dumps({"mock": []})}
            candidates = [{"item_id": wanted or "target", "instance_id": 1, "instance_key": f"{wanted or 'target'}_1", "topic_key": "mock", "center_world": [1.2, 0.4, 2.8], "camsrc": [], "bboxes_by_camera": {}}]
        else:
            from .room_topics import get_topic_string_message
            raw = await get_topic_string_message("/world_position_data", timeout_sec=5.0)
            if not raw:
                status = "TARGET_NOT_FOUND"
                return {"selected_instance": {}, "current_status": status, "last_execution": self._execution(state, "find_node", status, started, success=False, error="/world_position_data read failed")}
            raw_payload = {"data": raw}
            candidates = parse_world_position_payload(raw_payload)
        raw_ref = store.save_json("world_position_raw", raw_payload, created_by_node="find_node")
        room_cameras = {} if self.use_mock else await self._capture_room_camera_images(state, self._world_position_camera_names(candidates), timeout_sec=10.0)
        matches = [candidate for candidate in candidates if candidate.get("item_id") == wanted]
        if not matches:
            status = "TARGET_NOT_FOUND"
            world = WorldPositionSnapshot(raw_payload_ref=raw_ref, candidate_count=len(candidates), updated_at=time.time(), update_source_node="find_node", update_reason="no_matching_instance")
            return {"selected_instance": {}, "world_position": dump_model(world), "room_cameras": room_cameras, "current_status": status, "last_execution": self._execution(state, "find_node", status, started, success=False, message=f"No instance for {label}")}

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
            print(f"  [{idx}] {candidate.get('instance_key')} center_world={candidate.get('center_world', [])} camsrc={candidate.get('camsrc', [])}{preview_note}")
        if self.use_mock and len(matches) == 1:
            selected_idx = 1
        else:
            loop = asyncio.get_event_loop()
            while True:
                choice = await loop.run_in_executor(None, lambda: input(f"\n請輸入候選照片編號 (1-{len(matches)}) 或 no：\n> "))
                if choice.strip().lower() == "no":
                    status = "TARGET_NOT_FOUND"
                    return {"selected_instance": {}, "current_status": status, "last_execution": self._execution(state, "find_node", status, started, success=False, message="User selected no target")}
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
        if not primary_camera:
            primary_camera, primary_bbox = self._pick_primary_room_camera(candidate, room_cameras)
        if primary_camera and primary_camera in room_cameras:
            room_cameras[primary_camera]["bbox"] = primary_bbox
            if preview_ref:
                room_cameras[primary_camera]["preview_ref"] = artifact_ref_json(preview_ref)
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
        world = WorldPositionSnapshot(raw_payload_ref=raw_ref, candidate_count=len(candidates), selected_instance_key=selected.instance_key, updated_at=time.time(), update_source_node="find_node", update_reason="db_created")
        status = "TARGET_SELECTED_FROM_WORLD_POSITION"
        return {"selected_instance": dump_model(selected), "world_position": dump_model(world), "room_cameras": room_cameras, "current_status": status, "last_execution": self._execution(state, "find_node", status, started, message=selected.instance_key)}

    def _route_find(self, state: CommanderState) -> str:
        return "get_item_info_no_sam3d_node" if state.get("selected_instance") else "end"

    async def _get_item_info_no_sam3d_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        from agents.get_item_info_agent_no_sam3d import GetItemInfoNoSam3DAgent

        store = self._artifact_store(state)
        selected = state.get("selected_instance", {}) or {}
        world = state.get("world_position", {}) or {}
        room_cameras = state.get("room_cameras", {}) or {}
        if not selected or not world.get("raw_payload_ref"):
            status = "ITEM_INFO_NO_SAM3D_FAILED"
            return {"current_status": status, "last_execution": self._execution(state, "get_item_info_no_sam3d_node", status, started, success=False, error="missing selected instance or world ref")}
        camera_names = [name for name in selected.get("camsrc", []) if name in room_cameras]
        if not self.use_mock and not camera_names:
            status = "ITEM_INFO_NO_SAM3D_FAILED"
            return {"current_status": status, "last_execution": self._execution(state, "get_item_info_no_sam3d_node", status, started, success=False, error="no target room camera refs")}
        camera_images = {name: store.load_base64(room_cameras[name]["rgb_ref"]) for name in camera_names}
        raw_world = store.load_json(world["raw_payload_ref"])
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
        request_ref = store.save_json("a2a_raw_request", {**params, "camera_images": {k: room_cameras[k]["rgb_ref"] for k in camera_names}}, created_by_node="get_item_info_no_sam3d_node")
        agent = GetItemInfoNoSam3DAgent(http_client=self.http_client, use_mock=self.use_mock)
        result = await agent.execute(params, state.get("context_id", ""))
        success = bool(result.get("success", False))
        payload = result.get("result", {}) if isinstance(result.get("result", {}), dict) else {}
        raw_ref = store.save_json("a2a_raw_result", payload, created_by_node="get_item_info_no_sam3d_node", metadata={"request_artifact_id": request_ref.artifact_id})
        if not success or not payload:
            status = "ITEM_INFO_NO_SAM3D_FAILED"
            return {"item_info": dump_model(ItemInfoResult(raw_result_ref=raw_ref)), "current_status": status, "last_execution": self._execution(state, "get_item_info_no_sam3d_node", status, started, success=False, error="empty item-info result")}
        group_ranking = payload.get("group_ranking", []) or []
        item_info = ItemInfoResult(
            center_world=[float(v) for v in payload.get("center_world", selected.get("center_world", []))],
            center_world_coordinate_frame=str(payload.get("center_world_coordinate_frame", "unity_world")),
            primary_camera_id=str(payload.get("primary_camera_id", selected.get("primary_camera", ""))),
            target_instance_key=str(payload.get("target_instance_key", selected.get("instance_key", ""))),
            target_topic_key=str(payload.get("target_topic_key", selected.get("topic_key", ""))),
            group_ranking=group_ranking,
            goal_pose_path=str(payload.get("goal_pose_path", "")),
            raw_result_ref=raw_ref,
            a2a_task_id=str(result.get("a2a_task_id", "")),
        )
        goal_pose_db = self._goal_pose_db_from_item_info(item_info.model_dump(mode="json"), 1)
        nav_goal, err = self._goal_pose_for_rank(item_info.model_dump(mode="json"), 1)
        navigation = NavigationState(
            current_goal_rank=1,
            current_goal_pose_index=0,
            goal_pose_db=goal_pose_db,
            nav_goal=NavGoal(**nav_goal) if not err else None,
            nav_goal_pose_source="rank_best",
            nav_move_source="bootstrap",
            force_initialpose=state.get("navigation", {}).get("result") in ({}, None),
        )
        status = "ITEM_INFO_NO_SAM3D_READY"
        return {"item_info": dump_model(item_info), "navigation": dump_model(navigation), "current_status": status, "last_execution": self._execution(state, "get_item_info_no_sam3d_node", status, started, success=not bool(err), message="item info ready" if not err else err)}

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
            world.update({"target_changed": False, "update_source_node": source_node, "update_reason": "unchanged", "update_distance_m": 0.0})
            return {"world_position": world, "current_status": status, "last_execution": self._execution(state, source_node, status, started)}
        from .room_topics import get_topic_string_message
        from .world_position import parse_world_position_payload
        raw = await get_topic_string_message("/world_position_data", timeout_sec=5.0)
        if not raw:
            status = "WORLD_POSITION_UPDATE_READ_FAILED"
            world = WorldPositionSnapshot(update_source_node=source_node, update_reason="read_failed")
            return {"world_position": dump_model(world), "current_status": status, "last_execution": self._execution(state, source_node, status, started, success=False)}
        store = self._artifact_store(state)
        raw_payload = {"data": raw}
        raw_ref = store.save_json("world_position_raw", raw_payload, created_by_node=source_node)
        candidates = parse_world_position_payload(raw_payload)
        refreshed = self._find_selected_candidate(candidates, selected)
        if not refreshed:
            status = "TARGET_LOST_IN_WORLD_POSITION"
            world = WorldPositionSnapshot(raw_payload_ref=raw_ref, candidate_count=len(candidates), selected_instance_key=selected.get("instance_key", ""), updated_at=time.time(), target_changed=False, update_source_node=source_node, update_reason="target_missing")
            return {"world_position": dump_model(world), "navigation": {"nav_goal": {}, "goal_pose_db": {}, "result": {}}, "grasp_result": {}, "current_status": status, "last_execution": self._execution(state, source_node, status, started, success=False, message="target missing")}
        moved = self._center_world_distance_m(selected.get("center_world", []), refreshed.get("center_world", []))
        threshold = float(os.getenv("WORLD_POSITION_UPDATE_THRESHOLD_M", "0.3"))
        target_changed = moved > threshold
        world = WorldPositionSnapshot(raw_payload_ref=raw_ref, candidate_count=len(candidates), selected_instance_key=selected.get("instance_key", ""), updated_at=time.time(), target_changed=target_changed, update_source_node=source_node, update_distance_m=moved, update_reason="target_moved" if target_changed else "unchanged")
        update: dict[str, Any] = {"world_position": dump_model(world), "current_status": "WORLD_POSITION_TARGET_MOVED" if target_changed else "WORLD_POSITION_UNCHANGED", "last_execution": self._execution(state, source_node, "WORLD_POSITION_TARGET_MOVED" if target_changed else "WORLD_POSITION_UNCHANGED", started)}
        if target_changed:
            selected_updated = dict(selected)
            selected_updated.update({"center_world": refreshed.get("center_world", selected.get("center_world", [])), "camsrc": refreshed.get("camsrc", selected.get("camsrc", [])), "bboxes_by_camera": refreshed.get("bboxes_by_camera", selected.get("bboxes_by_camera", {}))})
            update.update({"selected_instance": selected_updated, "item_info": {}, "navigation": {"current_goal_rank": 1, "current_goal_pose_index": 0, "goal_pose_db": {}, "nav_goal": {}, "result": {}}, "grasp_result": {}})
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
    def _find_selected_candidate(candidates: list[dict[str, Any]], selected: dict[str, Any]) -> dict[str, Any] | None:
        key = str(selected.get("instance_key", ""))
        for candidate in candidates:
            if str(candidate.get("instance_key", "")) == key:
                return candidate
        return None

    async def _observe_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        task = state.get("task", {}) or {}
        description = task.get("normalized_task") or task.get("original_user_request") or "抓取目標物件"
        image_ref = None
        if not self.use_mock:
            from .camera import get_camera_image_base64
            image_b64 = await get_camera_image_base64("Camera_Car", timeout_sec=15.0)
            if image_b64:
                image_ref = self._artifact_store(state).save_base64_image("camera_car_rgb", image_b64, created_by_node="observe_node", metadata={"camera_name": "Camera_Car"})
        observation = Observation(description=description, image_ref=image_ref)
        status = "OBSERVED"
        return {"observation": dump_model(observation), "current_status": status, "last_execution": self._execution(state, "observe_node", status, started)}

    async def _reason_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        out = await self.brain.reason(state, artifact_store=self._artifact_store(state))
        decision = out["prediction"]
        record = DecisionRecord(reasoning=decision.reasoning, call_module=decision.call_module, module_params=decision.module_params, latency_sec=float(out["latency"]), model=out.get("model", ""))
        status = "REASONED"
        return {"decision": dump_model(record), "module_params": decision.module_params, "current_status": status, "last_execution": self._execution(state, "reason_node", status, started, message=decision.reasoning)}

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
        nav_state = NavigationState(**{**navigation, "current_goal_rank": next_rank, "current_goal_pose_index": 0, "nav_goal": NavGoal(**goal), "nav_goal_pose_source": "major_nav", "nav_move_source": "major_nav", "force_initialpose": False})
        status = "MAJOR_NAV_CONTEXT_READY"
        return {"navigation": dump_model(nav_state), "current_status": status, "last_execution": self._execution(state, "major_nav_node", status, started, message=f"rank {next_rank}")}

    def _route_major_nav(self, state: CommanderState) -> Literal["nav_move_node", "nav_home_node"]:
        return "nav_home_node" if state.get("task_complete") or state.get("current_status") == "MAJOR_NAV_EXHAUSTED" else "nav_move_node"

    def _route_nav_move(self, state: CommanderState) -> Literal["observe_node", "update_memory_node"]:
        return "observe_node" if (state.get("navigation", {}) or {}).get("nav_move_source") == "bootstrap" else "update_memory_node"

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
            nav_result = NavResult(goal=NavGoal(**goal), arrived=True, plan_ready=True, attempt=1, events=events, message="mock nav arrived")
            navigation["result"] = dump_model(nav_result)
            status = "NAV_COMPLETED"
            return {"navigation": navigation, "current_status": status, "last_execution": self._execution(state, "nav_move_node", status, started, message=nav_result.message)}
        payload = self._nav_runner_payload(state, goal, rank, goal_pose_index)
        result = await self._run_nav_move_runner(payload)
        success = bool(result.get("success", False))
        nav_result = NavResult(goal=NavGoal(**goal), arrived=success, plan_ready=bool(result.get("plan_ready", False)), attempt=int(payload.get("attempt", 1)), events=result.get("events", []) or [], message=str(result.get("message", "")))
        navigation["result"] = dump_model(nav_result)
        status = "NAV_COMPLETED" if success else "NAV_FAILED"
        return {"navigation": navigation, "current_status": status, "last_execution": self._execution(state, "nav_move_node", status, started, success=success, message=nav_result.message)}

    def _nav_runner_payload(self, state: CommanderState, goal: dict[str, Any], rank: int, goal_pose_index: int) -> dict[str, Any]:
        legacy_heading = os.getenv("NAV_GOAL_HEADING_TOLERANCE_DEG")
        return {
            "goal_pose": goal,
            "publish_initialpose": bool((state.get("navigation", {}) or {}).get("force_initialpose", False)),
            "initial_pose": self._default_initial_pose(),
            "plan_timeout_sec": float(os.getenv("NAV_PLAN_TIMEOUT_SEC", "8")),
            "arrival_timeout_sec": float(os.getenv("NAV_ARRIVAL_TIMEOUT_SEC", "180")),
            "publish_interval_sec": float(os.getenv("NAV_PUBLISH_INTERVAL_SEC", "0.1")),
            "goal_tolerance_m": float(os.getenv("NAV_GOAL_TOLERANCE_M", str(goal_tolerance_m_default()))),
            "goal_heading_tolerance_rad": float(os.getenv("NAV_GOAL_HEADING_TOLERANCE_RAD", str(math.radians(float(legacy_heading)) if legacy_heading is not None else goal_heading_tolerance_rad_default()))),
            "status_topic": "/nav_move/status",
            "attempt": 1,
            "rank": rank,
            "goal_pose_index": goal_pose_index,
            "goal_pose_source": goal.get("goal_pose_source", ""),
            "source": (state.get("navigation", {}) or {}).get("nav_move_source", "reason_loop"),
        }

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
            payload["status_topic"] = "/nav_home/status"
            result = await self._run_nav_move_runner(payload)
            events = result.get("events", []) or []
            success = bool(result.get("success", False))
            message = str(result.get("message", "home navigation completed" if success else "home navigation failed"))
        nav_result = NavResult(goal=NavGoal(**goal), arrived=success, plan_ready=success, attempt=1, events=events, message=message)
        navigation = {**(state.get("navigation", {}) or {}), "nav_move_source": "nav_home", "nav_goal": goal, "result": dump_model(nav_result)}
        status = "NAV_HOME_COMPLETED" if success else "NAV_HOME_FAILED"
        return {"navigation": navigation, "task_complete": True, "current_status": status, "last_execution": self._execution(state, "nav_home_node", status, started, success=success, message=message)}

    async def _car_grasp_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        from agents.grasp_agent import GraspAgent
        from commander.camera import get_camera_rgbd_base64

        store = self._artifact_store(state)
        object_id = (state.get("requested_object", {}) or {}).get("id", "")
        if not object_id:
            status = "GRASP_FAILED"
            return {"grasp_result": dump_model(GraspResult(success=False)), "current_status": status, "last_execution": self._execution(state, "car_grasp_node", status, started, success=False, error="missing object id")}
        rgb_ref = depth_ref = None
        rgbd: dict[str, str] = {}
        if self.use_mock:
            rgbd = {"camera_name": "Camera_Car", "rgb_base64": "", "depth_base64": ""}
        else:
            rgbd = await get_camera_rgbd_base64("Camera_Car", timeout_sec=15.0) or {}
            if not rgbd:
                status = "GRASP_FAILED"
                return {"grasp_result": dump_model(GraspResult(object_id=object_id, success=False)), "current_status": status, "last_execution": self._execution(state, "car_grasp_node", status, started, success=False, error="missing RGBD")}
            rgb_ref = store.save_base64_image("camera_car_rgb", rgbd["rgb_base64"], created_by_node="car_grasp_node", metadata={"camera_name": "Camera_Car"})
            depth_ref = store.save_base64_image("camera_car_depth", rgbd["depth_base64"], created_by_node="car_grasp_node", metadata={"camera_name": "Camera_Car"})
        params = {**(state.get("module_params", {}) or {}), "object_id": object_id, "camera_name": "Camera_Car", "rgb_base64": rgbd.get("rgb_base64", ""), "depth_base64": rgbd.get("depth_base64", "")}
        request_ref = store.save_json("a2a_raw_request", {"object_id": object_id, "camera_name": "Camera_Car", "rgb_ref": artifact_ref_json(rgb_ref), "depth_ref": artifact_ref_json(depth_ref)}, created_by_node="car_grasp_node")
        agent = GraspAgent(http_client=self.http_client, use_mock=self.use_mock)
        result = await agent.execute(params, state.get("context_id", ""))
        success = bool(result.get("success", False))
        payload = result.get("result", {}) if isinstance(result.get("result", {}), dict) else {}
        raw_ref = store.save_json("a2a_raw_result", payload, created_by_node="car_grasp_node", metadata={"request_artifact_id": request_ref.artifact_id})
        valid_ref = None
        if payload.get("valid_grasp_poses_camera"):
            valid_ref = store.save_json("debug_payload", payload.get("valid_grasp_poses_camera", []), created_by_node="car_grasp_node", metadata={"kind": "valid_grasp_poses_camera"})
        grasp = GraspResult(
            object_id=payload.get("object_id") or object_id,
            camera_name=payload.get("camera_name", "Camera_Car"),
            success=success,
            grasp_confidence=payload.get("grasp_confidence"),
            num_candidate_grasps=payload.get("num_candidate_grasps"),
            num_valid_grasps=payload.get("num_valid_grasps"),
            best_grasp_pose_camera=payload.get("best_grasp_pose_camera", {}),
            valid_grasp_poses_ref=valid_ref,
            object_reference_center_camera=payload.get("object_reference_center_camera", []),
            rgb_ref=rgb_ref,
            depth_ref=depth_ref,
            raw_result_ref=raw_ref,
            a2a_task_id=str(result.get("a2a_task_id", "")),
        )
        status = "GRASP_READY" if success else "GRASP_FAILED"
        return {"grasp_result": dump_model(grasp), "current_status": status, "last_execution": self._execution(state, "car_grasp_node", status, started, success=success)}

    async def _car_approach_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        from agents.car_approach_agent import CarApproachAgent

        store = self._artifact_store(state)
        params = dict(state.get("module_params", {}) or {})
        params.setdefault("grasp_result", state.get("grasp_result", {}) or {})
        agent = CarApproachAgent(use_mock=self.use_mock)
        result = await agent.execute(params, state.get("context_id", ""))
        success = bool(result.get("success", False))
        payload = result.get("result", {}) if isinstance(result.get("result", {}), dict) else {}
        raw_ref = store.save_json("approach_raw_result", payload, created_by_node="car_approach_node")
        approach = ApproachResult(
            success=success,
            status_code=str(payload.get("status_code", "APPROACH_SUCCESS" if success else "APPROACH_FAIL")),
            phase=str(payload.get("phase", "")),
            message=str(payload.get("message") or payload.get("error") or ""),
            next_agent=payload.get("next_agent"),
            nav_result=payload.get("nav_result", {}) if isinstance(payload.get("nav_result", {}), dict) else {},
            arm_result=payload.get("arm_result", {}) if isinstance(payload.get("arm_result", {}), dict) else {},
            arm_base_alignment_result=payload.get("arm_base_alignment_result", {}) if isinstance(payload.get("arm_base_alignment_result", {}), dict) else {},
            selected_solution=payload.get("selected_solution", {}) if isinstance(payload.get("selected_solution", {}), dict) else {},
            raw_result_ref=raw_ref,
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
        ranks = {}
        for group in item_info.get("group_ranking", []) or []:
            rank = int(group.get("rank", len(ranks) + 1) or len(ranks) + 1)
            ranks[str(rank)] = dict(group)
        return {"target_instance_key": item_info.get("target_instance_key", ""), "center_world": item_info.get("center_world", []), "current_goal_rank": current_rank, "rank_order": [int(k) for k in ranks.keys()], "ranks": ranks, "updated_at": time.time()}

    def _target_center_world_to_map_xy(self, item_info: Dict[str, Any]) -> tuple[float | None, float | None]:
        center_world = item_info.get("center_world", [])
        if not isinstance(center_world, list) or len(center_world) < 3:
            return None, None
        return 6.0 - float(center_world[2]), float(center_world[0]) - 3.0

    def _goal_pose_from_ros_map(self, item_info: Dict[str, Any], goal_pose_ros: Any) -> tuple[Dict[str, Any], str]:
        if not isinstance(goal_pose_ros, list) or len(goal_pose_ros) < 2:
            return {}, "missing goal_pose_ros_map"
        try:
            goal_x = float(goal_pose_ros[0]); goal_y = float(goal_pose_ros[1])
        except (TypeError, ValueError):
            return {}, f"invalid goal_pose_ros_map={goal_pose_ros}"
        yaw = 0.0
        target_map_x, target_map_y = self._target_center_world_to_map_xy(item_info)
        if target_map_x is not None and target_map_y is not None:
            yaw = math.atan2(target_map_y - goal_y, target_map_x - goal_x)
        goal_pose = {"x": goal_x, "y": goal_y, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": math.sin(yaw / 2.0), "qw": math.cos(yaw / 2.0), "yaw": yaw}
        if target_map_x is not None and target_map_y is not None:
            goal_pose["face_target_x"] = target_map_x; goal_pose["face_target_y"] = target_map_y
        return goal_pose, ""

    def _goal_pose_for_rank(self, item_info: Dict[str, Any], rank: int) -> tuple[Dict[str, Any], str]:
        groups = item_info.get("group_ranking", []) or []
        idx = rank - 1
        if idx < 0 or idx >= len(groups):
            return {}, f"rank={rank} out of range"
        group = groups[idx] or {}
        goal, err = self._goal_pose_from_ros_map(item_info, group.get("best_goal_pose_ros_map", []))
        if err:
            return {}, f"rank={rank} {err}"
        goal.update({"goal_rank": rank, "goal_pose_index": 0, "goal_pose_source": "rank_best", "orientation_group": group.get("orientation_group"), "grasp_confidence": group.get("best_confidence"), "map_feasible": group.get("map_feasible"), "selection_mode": group.get("selection_mode", "")})
        return goal, ""

    def _default_initial_pose(self) -> Dict[str, Any]:
        return {"x": 3.4133476128639803, "y": -3.040367824880008, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0, "covariance": [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06853892060437211]}

    async def _run_nav_move_runner(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import shlex
        safe_payload = shlex.quote(json.dumps(payload))
        ros_py = "/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages"
        ros_lib = "/opt/ros/humble/lib"
        cmd = (
            "unset VIRTUAL_ENV PYTHONPATH PYTHONHOME && "
            "source /opt/ros/humble/setup.bash && "
            "source /workspaces/install/setup.bash 2>/dev/null || true && "
            f"export LD_LIBRARY_PATH={ros_lib}:${{LD_LIBRARY_PATH:-}} && "
            f"PYTHONPATH={ros_py} /usr/bin/python3 -m commander.nav_move_runner --payload {safe_payload}"
        )
        proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, executable="/bin/bash")
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            return {"success": False, "plan_ready": False, "message": stderr.decode().strip() or "nav_move_runner failed", "events": []}
        try:
            return json.loads(stdout.decode().strip() or "{}")
        except json.JSONDecodeError:
            return {"success": False, "plan_ready": False, "message": "Invalid nav_move_runner output", "events": []}

    async def aclose(self) -> None:
        await self.http_client.aclose()
        if self._checkpoint_cm is not None:
            await self._checkpoint_cm.__aexit__(None, None, None)
        elif self._checkpoint_conn is not None:
            self._checkpoint_conn.close()
