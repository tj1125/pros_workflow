from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..contracts import DecisionRecord, dump_model
from ..state import CommanderState
from ..object_catalog import load_graspable_objects, valid_object_index

logger = logging.getLogger(__name__)

_GOODBYE_TOKENS = {"bye", "exit", "quit", "q", "再見", "掰掰", "結束"}
_PICK_TASK_PATTERN = re.compile(r"\b(pick up|pick|grab|grasp|get|fetch|take|hold)\b|抓取|拾取|拿|抓|夾|取")


class TaskClassification(BaseModel):
    intent: Literal["general_chat", "specific_task"] = Field(
        description="Route to general_chat for casual conversation, specific_task for picking tasks."
    )
    selected_object_index: int = 0
    reasoning: str = ""


class ChatReply(BaseModel):
    reply: str


class RelatedObjectSelection(BaseModel):
    related_object_indices: list[int] = Field(
        default_factory=list,
        description="1-based object indices from the provided config that should be considered for the request.",
    )
    reasoning: str = ""


class ChatFlowMixin:
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
        objects = load_graspable_objects()
        try:
            classification = self._mock_task_classification(human_reply, objects) if self.use_mock else await self._llm_task_classification(human_reply, objects)
        except Exception as exc:
            logger.error("[task_classification_node] classifier failed: %s", exc, exc_info=True)
            classification = self._mock_task_classification(human_reply, objects)
        selected_index = valid_object_index(classification.selected_object_index, objects)
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
        return {
            "human_reply": "",
            "history_buffer": [entry],
            "current_status": status,
            "last_execution": self._execution(state, "chat_memory_node", status, started),
        }

    async def _goodbye_node(self, state: CommanderState) -> Dict[str, Any]:
        started = time.time()
        context_id = str(state.get("context_id", "") or "default")
        try:
            self._artifact_store(state).clear_artifacts()
            if hasattr(self, "_transient_base64"):
                self._transient_base64.pop(context_id, None)
        except Exception as exc:
            logger.warning("[goodbye_node] session artifact cleanup failed: %s", exc)
        message = "對話及任務結束，祝您有美好的一天～"
        print(f"\n{message}", flush=True)
        status = "GOODBYE_SENT"
        return {"current_status": status, "last_execution": self._execution(state, "goodbye_node", status, started, message=message)}

    def _route_human_reply(self, state: CommanderState) -> Literal["task_classification_node", "goodbye_node"]:
        return "goodbye_node" if self._is_goodbye_reply(state.get("human_reply", "")) else "task_classification_node"

    def _route_task_classification(self, state: CommanderState) -> Literal["ai_reply_node", "input_node"]:
        selected = valid_object_index(state.get("selected_object_index", 0), load_graspable_objects())
        return "input_node" if state.get("task_intent") == "specific_task" and selected else "ai_reply_node"

    @staticmethod
    def _is_goodbye_reply(reply: str) -> bool:
        return reply.strip().casefold() in _GOODBYE_TOKENS

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
