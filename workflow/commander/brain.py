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
        "nav_to_next_candidate_goal_pose",
        "grasp_pipeline",
    ] = Field(description="The pipeline to invoke")
    module_params: Dict[str, Any] = Field(default_factory=dict)


class Brain:
    """VLM reasoning hub over the narrow LangGraph state schema."""

    def __init__(self):
        self._raw_model = None
        self._model = None
        self._init_model()

    def _init_model(self) -> None:
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

        viewpoint_status = self._current_viewpoint_status(state, current_rank)

        text_content = (
            "Task\n"
            f"Target: {target_text}\n"
            f"Target instance: {instance_key}\n"
            "\nCurrent State\n"
            f"Current viewpoint rank: {current_rank}\n"
            f"Next viewpoint available: {next_rank_available}\n"
            f"Last navigation: arrived={nav_arrived}, message={nav_message or 'none'}\n"
            f"Last grasp: success={grasp_success}, pose_ready={pose_ready}\n"
            f"Last approach: success={approach_success}, phase={approach_phase}, arm_success={arm_success}\n"
            f"Recent failure: {failure_hint or 'none'}\n\n"
            f"This Viewpoint (rank {current_rank}) — authoritative for the switch decision\n"
            f"{viewpoint_status}\n\n"
            "Current Observation\n"
            f"{observation.get('description', 'Camera image unavailable.')}\n\n"
            "Recent Action History (all viewpoints; context only — never switch because a DIFFERENT viewpoint failed)\n"
            f"{history_summary}\n\n"
            "Decide the next action. Output exactly one JSON object with this schema:\n"
            '{"reasoning":"short reason","call_module":"nav_to_next_candidate_goal_pose|grasp_pipeline","module_params":{}}'
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
    def _current_viewpoint_status(state: CommanderState, current_rank: int) -> str:
        """Summarise grasp/approach attempts already made at the CURRENT viewpoint.

        Read from history_buffer (each attempt records ``at_rank`` in its key_facts).
        The brain must base its retry-vs-switch decision on THIS, not on failures
        recorded at earlier viewpoints.
        """
        attempts: list[tuple[str, Any, str]] = []
        for entry in state.get("history_buffer", []) or []:
            facts = entry.get("key_facts") or {}
            if "at_rank" not in facts:
                continue
            if int(facts.get("at_rank", -1) or -1) != current_rank:
                continue
            if "approach_phase" in facts or "approach_success" in facts:
                attempts.append(("approach", facts.get("approach_success"), str(facts.get("approach_phase", "") or "")))
            elif "grasp_success" in facts:
                attempts.append(("grasp", facts.get("grasp_success"), ""))
        if not attempts:
            return (
                f"No grasp has been attempted from this viewpoint (rank {current_rank}) yet. "
                "Judge ONLY the live image below: is the target graspable and not occluded "
                "from here? Failures at earlier viewpoints are irrelevant to this fresh view."
            )
        kind, ok, phase = attempts[-1]
        phase_txt = f", phase={phase}" if phase else ""
        return (
            f"{len(attempts)} grasp attempt(s) already made from this viewpoint; "
            f"latest {kind} attempt: success={ok}{phase_txt}. "
            "Base retry-vs-switch on THIS viewpoint's failure phase (see the agent rules)."
        )

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
        decision = await self._llm_reason(state, artifact_store=artifact_store, image_loader=image_loader)
        latency = time.time() - start
        logger.info("[Brain] Decision: call_module=%s, latency=%.2fs", decision.call_module, latency)
        return {"prediction": decision, "latency": latency, "model": self._model_name()}

    def _model_name(self) -> str:
        return os.getenv("OLLAMA_MODEL", "gemma3:12b")

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
            "Check OLLAMA_STRUCTURED_METHOD and model support."
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
