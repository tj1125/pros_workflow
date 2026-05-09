import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .state import CommanderState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schema for structured VLM output
# ---------------------------------------------------------------------------

class BrainDecision(BaseModel):
    """Structured JSON output from the VLM reasoning step."""

    reasoning: str = Field(
        description="Step-by-step reasoning about the current scene and why this action is chosen"
    )
    call_module: Literal[
        "nav_agent", "grasp_agent", "car_approach_agent", "DONE"
    ] = Field(description="The agent module to invoke, or DONE if the task is complete")
    module_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters to pass to the selected agent module",
    )


# ---------------------------------------------------------------------------
# Brain: VLM reasoning hub
# ---------------------------------------------------------------------------

from .prompts import SYSTEM_PROMPT

class Brain:
    """
    VLM reasoning hub — wraps Gemini (via langchain-google-genai) or Ollama.
    Accepts current LangGraph state and returns a BrainDecision.
    """

    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self._model = None

        if not use_mock:
            self._init_model()

    def _init_model(self) -> None:
        """Initialise the LLM based on VLM_PROVIDER env var."""
        provider = os.getenv("VLM_PROVIDER", "google").lower()

        if provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI

            model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            self._model = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            ).with_structured_output(BrainDecision)
            logger.info(f"[Brain] Using Google Gemini: {model_name}")

        elif provider == "ollama":
            from langchain_openai import ChatOpenAI

            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            model_name = os.getenv("OLLAMA_MODEL", "llama3.2-vision")
            # Ollama exposes OpenAI-compatible API
            self._model = ChatOpenAI(
                model=model_name,
                openai_api_key="ollama",
                openai_api_base=f"{base_url}/v1",
                temperature=0,
            ).with_structured_output(BrainDecision)
            logger.info(f"[Brain] Using Ollama: {model_name} @ {base_url}")

        else:
            raise ValueError(f"[Brain] Unknown VLM_PROVIDER: {provider}")

    def _build_prompt(self, state: CommanderState) -> list | str:
        """Build the VLM prompt from current state."""
        obs = state.get("current_observation", {})
        history = state.get("history_buffer", [])
        retry = state.get("retry_count", 0)
        target = state.get("target_object", {})

        history_summary = "\n".join(
            [self._format_history_entry(h) for h in history]
        ) or "  (no history yet)"

        obs_desc = obs.get("description", "Camera image unavailable (mock mode)")
        target_label = target.get("label") or target.get("id") or "unspecified"
        target_pos = target.get("position_3d", "unknown")

        text_content = (
            f"## Target Object\nLabel: {target_label}\n3D Position (approx): {target_pos}\n\n"
            f"## Current Observation\n{obs_desc}\n\n"
            f"## Action History (last 3)\n{history_summary}\n\n"
            f"## Retry Count\n{retry}\n\n"
            "Decide the next action. Output valid JSON."
        )

        image_b64 = obs.get("image_base64")
        if image_b64:
            return [
                {"type": "text", "text": text_content},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
            ]
        return text_content

    @staticmethod
    def _format_history_entry(entry: Dict[str, Any]) -> str:
        """Format one memory entry for the Brain prompt."""
        result = entry.get("result", "")
        if isinstance(result, dict):
            result = json.dumps(result, ensure_ascii=False, sort_keys=True)

        key_facts = entry.get("key_facts") or {}
        if isinstance(key_facts, dict) and key_facts:
            facts_text = ", ".join(
                f"{key}={value}" for key, value in key_facts.items()
            )
            facts_text = f" | KeyFacts: {facts_text}"
        else:
            facts_text = ""

        return (
            f"  - Action: {entry.get('action')}, Result: {result}, "
            f"Success: {entry.get('success')}{facts_text}"
        )

    async def reason(self, state: CommanderState) -> Dict[str, Any]:
        """
        Run VLM inference on the current state and return a BrainDecision.
        Returns a dict with 'prediction' (BrainDecision) and 'latency' (float).
        """
        start = time.time()

        if self.use_mock:
            decision = await self._mock_reason(state)
        else:
            decision = await self._llm_reason(state)

        latency = time.time() - start
        logger.info(
            f"[Brain] Decision: call_module={decision.call_module}, "
            f"latency={latency:.2f}s"
        )
        return {"prediction": decision, "latency": latency}

    async def _llm_reason(self, state: CommanderState) -> BrainDecision:
        """Real VLM call via LangChain structured output."""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=self._build_prompt(state)),
        ]
        result = await asyncio.get_event_loop().run_in_executor(
            None, self._model.invoke, messages
        )
        return result

    async def _mock_reason(self, state: CommanderState) -> BrainDecision:
        """Deterministic mock sequence for local testing without GPU/VLM."""
        await asyncio.sleep(0.5)  # Simulate VLM latency
        count = state.get("retry_count", 0)

        if count == 0:
            return BrainDecision(
                reasoning="Scene is occluded. Navigate to a better vantage point.",
                call_module="nav_agent",
                module_params={"target_point": [0.5, 0.5, 0.0]},
            )
        elif count == 1:
            return BrainDecision(
                reasoning="Position improved. Generate grasp pose for target object.",
                call_module="grasp_agent",
                module_params={"object_id": "target_object"},
            )
        elif count == 2:
            return BrainDecision(
                reasoning="Grasp pose received. Begin base approach sequence.",
                call_module="car_approach_agent",
                module_params={"target_id": "apple"},
            )
        else:
            return BrainDecision(
                reasoning=(
                    "Car approach succeeded and the grasp was verified. Task is done."
                ),
                call_module="DONE",
                module_params={},
            )
