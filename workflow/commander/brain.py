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
            self._model = self._with_structured_output(method)
            logger.info("[Brain] Using Google Gemini: %s structured_method=%s", model_name, method)
        elif provider == "ollama":
            from langchain_openai import ChatOpenAI

            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model_name = os.getenv("OLLAMA_MODEL", "gemma3:12b")
            self._raw_model = ChatOpenAI(
                model=model_name,
                openai_api_key="ollama",
                openai_api_base=f"{base_url}/v1",
                temperature=0,
            )
            method = os.getenv("OLLAMA_STRUCTURED_METHOD", "json_schema")
            self._model = self._with_structured_output(method)
            logger.info("[Brain] Using Ollama: %s @ %s structured_method=%s", model_name, base_url, method)
        else:
            raise ValueError(f"[Brain] Unknown VLM_PROVIDER: {provider}")

    def _with_structured_output(self, method: str) -> Any:
        if self._raw_model is None:
            raise RuntimeError("[Brain] Raw model is not initialized.")
        try:
            return self._raw_model.with_structured_output(BrainDecision, method=method, include_raw=True)
        except TypeError as exc:
            if "include_raw" not in str(exc):
                raise
            logger.warning("[Brain] Structured output wrapper lacks include_raw support; raw retry fallback remains enabled.")
            return self._raw_model.with_structured_output(BrainDecision, method=method)

    def _build_prompt(self, state: CommanderState, artifact_store: Any | None = None, image_loader: Any | None = None) -> list | str:
        task = state.get("task", {}) or {}
        requested = state.get("requested_object", {}) or {}
        selected = state.get("selected_instance", {}) or {}
        item_info = state.get("item_info", {}) or {}
        navigation = state.get("navigation", {}) or {}
        observation = state.get("observation", {}) or {}
        history = list(state.get("history_buffer", []) or [])[-6:]

        history_summary = "\n".join(self._format_history_entry(item) for item in history) or "  (no recent node history)"
        nav_result = navigation.get("result") or {}
        current_rank = int(navigation.get("current_goal_rank", 1) or 1)
        groups = item_info.get("group_ranking", []) or []
        available_goal_ranks = []
        for idx, group in enumerate(groups, 1):
            if isinstance(group, dict):
                available_goal_ranks.append(int(group.get("rank", idx) or idx))
        next_rank_available = any(rank > current_rank for rank in available_goal_ranks)
        failure_hint = self._failure_hint(state)
        target_label = str(requested.get("label") or requested.get("id") or task.get("normalized_task", "") or "target object")
        target_id = str(requested.get("id") or "")
        target_text = f"{target_label} ({target_id})" if target_id and target_id != target_label else target_label
        instance_key = selected.get("instance_key") or item_info.get("target_instance_key") or "unknown"
        nav_arrived = nav_result.get("arrived", "unknown") if nav_result else "unknown"
        nav_message = nav_result.get("message", "") if nav_result else ""
        grasp = state.get("grasp_result", {}) or {}
        grasp_success = grasp.get("success", "unknown") if grasp else "unknown"
        pose_ready = bool(grasp.get("best_grasp_pose_camera")) if grasp else "unknown"
        approach = state.get("approach_result", {}) or {}
        approach_success = approach.get("success", "unknown") if approach else "unknown"
        approach_phase = approach.get("phase") or approach.get("status_code") or "none"
        arm_result = approach.get("arm_result", {}) if isinstance(approach.get("arm_result", {}), dict) else {}
        arm_success = bool(arm_result.get("success", False)) if approach and arm_result else "unknown"

        text_content = (
            "## Task\n"
            f"Target: {target_text}\n"
            f"Target instance: {instance_key}\n"
            f"Done policy: {task.get('done_policy', '') or 'Complete the requested robot task safely.'}\n\n"
            "## Current State\n"
            f"Current viewpoint rank: {current_rank}\n"
            f"Next viewpoint available: {next_rank_available}\n"
            f"Last navigation: arrived={nav_arrived}, message={nav_message or 'none'}\n"
            f"Last grasp: success={grasp_success}, pose_ready={pose_ready}\n"
            f"Last approach: success={approach_success}, phase={approach_phase}, arm_success={arm_success}\n"
            f"Recent failure: {failure_hint or 'none'}\n\n"
            "## Current Observation\n"
            f"{observation.get('description', 'Camera image unavailable.')}\n\n"
            "## Recent Action History\n"
            f"{history_summary}\n\n"
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
        provider = os.getenv("VLM_PROVIDER", "google").lower()
        if provider == "ollama":
            return os.getenv("OLLAMA_MODEL", "gemma3:12b")
        if provider == "google":
            return os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return "configured-vlm"

    async def _llm_reason(self, state: CommanderState, artifact_store: Any | None = None, image_loader: Any | None = None) -> BrainDecision:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=self._build_prompt(state, artifact_store=artifact_store, image_loader=image_loader)),
        ]
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, self._model.invoke, messages)
        except Exception as exc:
            if self._raw_model is None or not self._is_structured_parse_error(exc):
                raise
            logger.warning("[Brain] Structured output parser failed; retrying with raw response parsing: %s", exc)
            result = await asyncio.get_event_loop().run_in_executor(None, self._raw_model.invoke, messages)
        return self._coerce_structured_decision(result)

    @staticmethod
    def _coerce_structured_decision(result: Any) -> BrainDecision:
        if isinstance(result, BrainDecision):
            return result
        if isinstance(result, dict):
            if "parsed" in result or "raw" in result or "parsing_error" in result:
                parsed = result.get("parsed")
                if parsed is not None:
                    return Brain._coerce_structured_decision(parsed)
                raw = result.get("raw")
                if raw is not None:
                    try:
                        return Brain._coerce_structured_decision(raw)
                    except Exception as exc:
                        parsing_error = result.get("parsing_error")
                        if parsing_error is not None:
                            raise ValueError(f"Brain structured output parsing failed: {parsing_error}") from exc
                        raise
            return BrainDecision.model_validate(Brain._normalize_decision_payload(result))
        if hasattr(result, "content"):
            payload = Brain._extract_json_object(getattr(result, "content"))
            return Brain._coerce_structured_decision(payload)
        return BrainDecision.model_validate(result)

    @staticmethod
    def _normalize_decision_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(payload)
        if "module_params" not in data and "params" in data:
            data["module_params"] = data.pop("params")
        return data

    @staticmethod
    def _extract_json_object(content: Any) -> Dict[str, Any]:
        text = Brain._message_content_to_text(content)
        if not text:
            raise ValueError("VLM provider returned empty message content instead of BrainDecision JSON.")

        candidates = [text]
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                candidates.append("\n".join(lines[1:-1]).strip())

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload

        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload

        raise ValueError(
            "VLM provider returned free-form message content instead of BrainDecision JSON. "
            "Check OLLAMA_STRUCTURED_METHOD/GOOGLE_STRUCTURED_METHOD and model support."
        )

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def _is_structured_parse_error(exc: Exception) -> bool:
        message = str(exc)
        return (
            "BrainDecision" in message
            or "Invalid JSON" in message
            or "Failed to parse" in message
            or "structured output" in message.lower()
        )


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
