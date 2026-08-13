from __future__ import annotations

import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .storage.artifact_store import ArtifactStore
from .contracts import NodeExecution, dump_model
from .logger import TraceLogger
from .flows.chat import ChatFlowMixin, ChatReply, RelatedObjectSelection, TaskClassification
from .flows.pick import PickFlowMixin
from .state import CommanderState


class Orchestrator(ChatFlowMixin, PickFlowMixin):
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
        # Ablation A1: no VLM brain is constructed. The observe/reason VLM step is
        # removed and the pick loop is a fixed flow wired by plain graph edges
        # in _build_graph (no model in the decision path).
        self.logger = trace_logger
        self.http_client = httpx.AsyncClient(timeout=120.0)
        self.use_mock = use_mock
        self._classifier_model = None
        self._related_object_model = None
        self._chat_model = None
        self._artifact_stores: dict[str, ArtifactStore] = {}
        self._transient_base64: dict[str, dict[str, str]] = {}
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

    def _context_id(self, state: CommanderState | dict[str, Any]) -> str:
        return str(state.get("context_id", "") or "default")

    def _remember_transient_base64(self, state: CommanderState | dict[str, Any], kind: str, encoded: str) -> str:
        key = f"{kind}:{uuid.uuid4().hex}"
        self._transient_base64.setdefault(self._context_id(state), {})[key] = encoded
        return key

    def _load_transient_base64(self, state: CommanderState | dict[str, Any], key: str) -> str:
        try:
            return self._transient_base64[self._context_id(state)][key]
        except KeyError as exc:
            raise KeyError(f"transient payload not found: {key}") from exc

    def _init_ollama_chat_models(self) -> None:
        from langchain_openai import ChatOpenAI

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        classifier_model_name = os.getenv("OLLAMA_CLASSIFIER_MODEL", "gemma3:1b")
        chat_model_name = os.getenv("OLLAMA_CHAT_MODEL", os.getenv("OLLAMA_MODEL", "gemma4:31b"))
        classifier_base_model = ChatOpenAI(
            model=classifier_model_name,
            openai_api_key="ollama",
            openai_api_base=f"{base_url}/v1",
            temperature=0,
        )
        self._classifier_model = classifier_base_model.with_structured_output(TaskClassification)
        self._related_object_model = classifier_base_model.with_structured_output(RelatedObjectSelection)
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
        # Ablation A1: observe_node + reason_node (the VLM observe-reason step) are
        # removed. The pick loop is wired as a fixed flow with plain edges.
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
        workflow.add_edge("nav_home_node", "goodbye_node")
        workflow.add_edge("car_grasp_node", "car_approach_node")
        # Fixed flow: after arriving at a viewpoint, always attempt grasp
        # (via world-position refresh) -> car_grasp -> car_approach.
        workflow.add_edge("nav_move_node", "update_item_info_2_node")
        # Fixed flow: a failed approach records a retry, then advances to the
        # next ranked viewpoint (via world-position refresh) -> major_nav.
        workflow.add_edge("update_memory_node", "update_item_info_1_node")

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
            "car_approach_node",
            self._route_car_approach,
            {"end": "nav_home_node", "update_memory_node": "update_memory_node"},
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

    async def aclose(self) -> None:
        await self.http_client.aclose()
        if self._checkpoint_cm is not None:
            await self._checkpoint_cm.__aexit__(None, None, None)
        elif self._checkpoint_conn is not None:
            self._checkpoint_conn.close()
