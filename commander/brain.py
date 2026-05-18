import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from .prompts import SYSTEM_PROMPT
from .state import CommanderState

logger = logging.getLogger(__name__)


class BrainDecision(BaseModel):
    """Structured JSON output from the VLM reasoning step."""

    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(description="Short reasoning about the current scene and chosen action")
    call_module: Literal[
        "nav_agent",
        "major_nav_node",
        "grasp_agent",
        "car_approach_agent",
        "DONE",
    ] = Field(description="The agent module to invoke, or DONE if the task is complete")
    module_params: Dict[str, Any] = Field(default_factory=dict)


class Brain:
    """VLM reasoning hub over the narrow LangGraph state schema."""

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self._raw_model = None
        self._model = None
        if not use_mock:
            self._init_model()

    def _init_model(self) -> None:
        provider = os.getenv("VLM_PROVIDER", "google").lower()
        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            self._raw_model = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
            method = os.getenv("GOOGLE_STRUCTURED_METHOD", "json_schema")
            self._model = self._raw_model.with_structured_output(BrainDecision, method=method)
            logger.info("[Brain] Using Google Gemini: %s structured_method=%s", model_name, method)
        elif provider == "ollama":
            from langchain_openai import ChatOpenAI

            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model_name = os.getenv("OLLAMA_MODEL", "gemma4:26b")
            self._raw_model = ChatOpenAI(
                model=model_name,
                openai_api_key="ollama",
                openai_api_base=f"{base_url}/v1",
                temperature=0,
            )
            method = os.getenv("OLLAMA_STRUCTURED_METHOD", "json_schema")
            self._model = self._raw_model.with_structured_output(BrainDecision, method=method)
            logger.info("[Brain] Using Ollama: %s @ %s structured_method=%s", model_name, base_url, method)
        else:
            raise ValueError(f"[Brain] Unknown VLM_PROVIDER: {provider}")

    def _build_prompt(self, state: CommanderState, artifact_store: Any | None = None, image_loader: Any | None = None) -> list | str:
        task = state.get("task", {}) or {}
        requested = state.get("requested_object", {}) or {}
        selected = state.get("selected_instance", {}) or {}
        item_info = state.get("item_info", {}) or {}
        navigation = state.get("navigation", {}) or {}
        observation = state.get("observation", {}) or {}
        history = list(state.get("history_buffer", []) or [])[-6:]
        session_summary = str(state.get("session_summary", "") or "")
        retry = int(state.get("retry_count", 0) or 0)

        history_summary = "\n".join(self._format_history_entry(item) for item in history) or "  (no recent node history)"
        success_criteria = task.get("success_criteria") or []
        success_text = "\n".join(f"- {item}" for item in success_criteria) or "- Complete the requested robot task safely."
        nav_result = navigation.get("result") or {}
        current_rank = int(navigation.get("current_goal_rank", 1) or 1)
        groups = item_info.get("group_ranking", []) or []
        available_goal_ranks = []
        for idx, group in enumerate(groups, 1):
            if isinstance(group, dict):
                available_goal_ranks.append(int(group.get("rank", idx) or idx))
        next_rank_available = any(rank > current_rank for rank in available_goal_ranks)
        failure_hint = self._failure_hint(state)

        text_content = (
            "## Task\n"
            f"Original request: {task.get('original_user_request', '')}\n"
            f"Normalized task: {task.get('normalized_task', '')}\n"
            f"Done policy: {task.get('done_policy', '')}\n"
            f"Success criteria:\n{success_text}\n\n"
            "## Requested Object\n"
            f"ID: {requested.get('id', '')}\n"
            f"Label: {requested.get('label', '')}\n\n"
            "## Selected Instance\n"
            f"Instance key: {selected.get('instance_key', '')}\n"
            f"Center world: {selected.get('center_world', item_info.get('center_world', []))}\n"
            f"Primary camera: {selected.get('primary_camera', item_info.get('primary_camera_id', ''))}\n\n"
            "## Navigation\n"
            f"Current rank: {current_rank}\n"
            f"Available ranked goal poses: {available_goal_ranks or 'unknown'}\n"
            f"Next major_nav rank available: {next_rank_available}\n"
            f"Current nav goal source: {navigation.get('nav_goal_pose_source', '')}\n"
            f"Last result: {json.dumps(nav_result, ensure_ascii=False)}\n\n"
            "## Visual Safety Checklist\n"
            "Before choosing grasp_agent, inspect the image for foreground blockers. "
            "If the target is partly hidden by a cup, bottle, container, table edge, wall, chair part, "
            "or the gripper itself, choose major_nav_node when Next major_nav rank available is True. "
            "If recent grasp or approach failed at this rank, choose major_nav_node instead of repeating grasp_agent. "
            "Only choose grasp_agent when the target body and gripper approach corridor are clearly unobstructed.\n"
            f"Recent failure hint: {failure_hint or '(none)'}\n\n"
            "## Current Observation\n"
            f"{observation.get('description', 'Camera image unavailable.')}\n\n"
            "## Session Summary\n"
            f"{session_summary or '(empty)'}\n\n"
            "## Recent Node History (bounded to 6)\n"
            f"{history_summary}\n\n"
            f"## Retry Count\n{retry}\n\n"
            "Decide the next action. Output exactly one JSON object with this schema:\n"
            '{"reasoning":"short reason","call_module":"major_nav_node|grasp_agent|car_approach_agent|DONE","module_params":{}}\n'
            "Use DONE only when the task success criteria and done policy are satisfied."
        )

        image_b64 = ""
        image_key = str(observation.get("image_key", "") or "")
        image_ref = observation.get("image_ref")
        if image_key and image_loader is not None:
            try:
                image_b64 = image_loader(image_key)
            except Exception as exc:
                logger.warning("[Brain] Failed to load transient observation image: %s", exc)
        elif artifact_store is not None and image_ref:
            try:
                image_b64 = artifact_store.load_base64(image_ref)
            except Exception as exc:
                logger.warning("[Brain] Failed to load observation image artifact: %s", exc)
        if image_b64:
            return [
                {"type": "text", "text": text_content},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ]
        return text_content


    @staticmethod
    def _failure_hint(state: CommanderState) -> str:
        approach = state.get("approach_result", {}) or {}
        if approach and approach.get("success") is False:
            phase = str(approach.get("phase", "") or "unknown")
            message = str(approach.get("message", "") or approach.get("status_code", "") or "approach failed")
            return f"latest approach failed at current rank: phase={phase}; message={message}"
        grasp = state.get("grasp_result", {}) or {}
        if grasp and grasp.get("success") is False:
            return "latest grasp failed at current rank; do not repeat the same view if another ranked goal exists"
        return ""

    @staticmethod
    def _format_history_entry(entry: Dict[str, Any]) -> str:
        result = entry.get("result", "")
        if isinstance(result, dict):
            result = json.dumps(result, ensure_ascii=False, sort_keys=True)
        key_facts = entry.get("key_facts") or {}
        facts_text = ""
        if isinstance(key_facts, dict) and key_facts:
            facts_text = " | KeyFacts: " + ", ".join(f"{key}={value}" for key, value in key_facts.items())
        return f"  - Action: {entry.get('action')}, Result: {result}, Success: {entry.get('success')}{facts_text}"

    async def reason(self, state: CommanderState, artifact_store: Any | None = None, image_loader: Any | None = None) -> Dict[str, Any]:
        start = time.time()
        if self.use_mock:
            decision = await self._mock_reason(state)
        else:
            decision = await self._llm_reason(state, artifact_store=artifact_store, image_loader=image_loader)
        latency = time.time() - start
        logger.info("[Brain] Decision: call_module=%s, latency=%.2fs", decision.call_module, latency)
        return {"prediction": decision, "latency": latency, "model": self._model_name()}

    def _model_name(self) -> str:
        if self.use_mock:
            return "mock"
        return os.getenv("GEMINI_MODEL") or os.getenv("OLLAMA_MODEL") or "configured-vlm"

    async def _llm_reason(self, state: CommanderState, artifact_store: Any | None = None, image_loader: Any | None = None) -> BrainDecision:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=self._build_prompt(state, artifact_store=artifact_store, image_loader=image_loader)),
        ]
        result = await asyncio.get_event_loop().run_in_executor(None, self._model.invoke, messages)
        return self._coerce_structured_decision(result)

    @staticmethod
    def _coerce_structured_decision(result: Any) -> BrainDecision:
        if isinstance(result, BrainDecision):
            return result
        if isinstance(result, dict):
            return BrainDecision.model_validate(result)
        if hasattr(result, "content"):
            raise ValueError(
                "VLM provider returned free-form message content instead of BrainDecision structured output. "
                "Check OLLAMA_STRUCTURED_METHOD/GOOGLE_STRUCTURED_METHOD and model support."
            )
        return BrainDecision.model_validate(result)


    async def _mock_reason(self, state: CommanderState) -> BrainDecision:
        await asyncio.sleep(0.5)
        approach = state.get("approach_result", {}) or {}
        arm_result = approach.get("arm_result", {}) if isinstance(approach.get("arm_result", {}), dict) else {}
        if approach.get("success") and arm_result.get("success", True):
            return BrainDecision(reasoning="Approach and arm execution succeeded; task success criteria are satisfied.", call_module="DONE", module_params={})

        count = int(state.get("retry_count", 0) or 0)
        requested = state.get("requested_object", {}) or {}
        object_id = requested.get("id") or requested.get("label") or "target_object"
        if count == 0:
            return BrainDecision(reasoning="Bootstrap navigation is complete. Generate a grasp pose for the requested object.", call_module="grasp_agent", module_params={"object_id": object_id})
        if count == 1:
            return BrainDecision(reasoning="The previous action did not mark the task done. Try one alternate navigation viewpoint.", call_module="nav_agent", module_params={})
        return BrainDecision(reasoning="No further action is useful after the bounded retry loop.", call_module="DONE", module_params={})
