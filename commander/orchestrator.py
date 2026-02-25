import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, Literal

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from langgraph.graph import END, StateGraph

from .brain import Brain
from .logger import TraceLogger
from .state import CommanderState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orchestrator: LangGraph stateful graph with 5 nodes
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Builds and runs the LangGraph decision graph.

    Graph topology:
        observe_node → reason_node → route_node
            route_node --[nav_agent]--> execute_node → update_memory_node → observe_node
            route_node --[DONE]-------> END
    """

    def __init__(self, trace_logger: TraceLogger, use_mock: bool = True):
        self.logger = trace_logger
        self.brain = Brain(use_mock=use_mock)
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Agent Node URLs: read from env, fall back to localhost (local dev mode)
        self._agent_urls = {
            "nav_agent":      os.getenv("INF_NAV_URL", "http://localhost:9001"),
            "grasp_agent":    os.getenv("INF_GRASP_URL", "http://localhost:9002"),
            "approach_agent": os.getenv("INF_APPROACH_URL", ""),   # local-only
            "view_agent":     os.getenv("INF_VIEW_URL", "http://localhost:9003"),
        }

        self.use_mock = use_mock
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        workflow = StateGraph(CommanderState)

        # Add nodes
        workflow.add_node("observe_node", self._observe_node)
        workflow.add_node("reason_node", self._reason_node)
        workflow.add_node("execute_node", self._execute_node)
        workflow.add_node("update_memory_node", self._update_memory_node)

        # Entry point
        workflow.set_entry_point("observe_node")

        # Edges
        workflow.add_edge("observe_node", "reason_node")

        # Conditional routing after reasoning
        workflow.add_conditional_edges(
            "reason_node",
            self._route_decision,
            {
                "continue": "execute_node",
                "end": END,
            },
        )

        workflow.add_edge("execute_node", "update_memory_node")
        workflow.add_edge("update_memory_node", "observe_node")

        return workflow.compile()

    # ------------------------------------------------------------------
    # Node: observe
    # ------------------------------------------------------------------

    async def _observe_node(self, state: CommanderState) -> Dict[str, Any]:
        """
        Fetch current environment observation.
        In mock mode: returns a static description.
        In real mode: reads latest frame from Unity via Rosbridge (future work).
        """
        if self.use_mock:
            obs = {
                "description": (
                    f"Mock scene at step {state.get('retry_count', 0)}: "
                    "Target object visible on table with partial occlusion."
                ),
                "image_path": None,
            }
        else:
            # TODO (Day-2): subscribe to /camera/rgb/image_raw via Rosbridge WebSocket
            obs = state.get("current_observation", {"description": "No observation"})

        logger.info("[observe_node] Observation captured.")
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
    # Conditional edge after reason
    # ------------------------------------------------------------------

    def _route_decision(
        self, state: CommanderState
    ) -> Literal["continue", "end"]:
        if state.get("call_module") == "DONE" or state.get("task_complete", False):
            logger.info("[route] Task complete — ending graph.")
            return "end"
        return "continue"

    # ------------------------------------------------------------------
    # Node: execute (calls the appropriate Agent Node as A2A Client)
    # ------------------------------------------------------------------

    async def _execute_node(self, state: CommanderState) -> Dict[str, Any]:
        """
        Dispatch to the correct Agent Node.
        Each agent is a Python class/function that may call an A2A Server on RTX 3090.
        """
        module = state.get("call_module", "")
        params = state.get("module_params", {})
        context_id = state.get("context_id", uuid.uuid4().hex)

        logger.info(f"[execute_node] Dispatching → {module}")

        # Import agent modules lazily to keep dependency graph clean
        start = time.time()
        result_text = ""
        success = False

        if self.use_mock:
            result_text, success = await self._mock_execute(module, params)
        else:
            result_text, success = await self._real_execute(
                module, params, context_id
            )

        exec_latency = time.time() - start
        logger.info(
            f"[execute_node] {module} finished in {exec_latency:.2f}s, "
            f"success={success}"
        )

        return {
            "agent_result": result_text,
            "current_status": "EXECUTED",
            "_exec_latency": exec_latency,  # temp field, consumed by update_memory
        }

    async def _mock_execute(
        self, module: str, params: Dict[str, Any]
    ) -> tuple[str, bool]:
        """Mock agent execution for local testing."""
        await asyncio.sleep(0.8)
        messages = {
            "nav_agent":      f"[MockNAV] Navigated to {params.get('target_point', 'default')}",
            "grasp_agent":    f"[MockGrasp] Grasp pose generated for {params.get('object_id', 'object')}",
            "approach_agent": f"[MockApproach] Arm approaching {params.get('target_id', 'target')}",
            "view_agent":     "[MockView] View adjusted successfully",
        }
        text = messages.get(module, f"[Mock] Unknown module: {module}")
        return text, True

    async def _real_execute(
        self, module: str, params: Dict[str, Any], context_id: str
    ) -> tuple[str, bool]:
        """
        Real agent execution.
        For nav / grasp / view: use A2AClient to call the GPU inference server.
        For approach: execute local control logic (no GPU inference server defined).
        """
        if module == "approach_agent":
            from agents.approach_agent import ApproachAgent
            agent = ApproachAgent()
            result = await agent.execute(params, context_id)
            return result["result"], result["success"]

        # Agents that call RTX 3090 via A2A
        module_to_agent = {
            "nav_agent":   ("agents.nav_agent",   "NavAgent"),
            "grasp_agent": ("agents.grasp_agent", "GraspAgent"),
            "view_agent":  ("agents.view_agent",  "ViewAgent"),
        }

        if module not in module_to_agent:
            return f"Unknown module: {module}", False

        mod_path, class_name = module_to_agent[module]
        import importlib
        mod = importlib.import_module(mod_path)
        agent_cls = getattr(mod, class_name)
        agent = agent_cls(http_client=self.http_client)
        result = await agent.execute(params, context_id)
        return result["result"], result["success"]

    # ------------------------------------------------------------------
    # Node: update memory
    # ------------------------------------------------------------------

    async def _update_memory_node(self, state: CommanderState) -> Dict[str, Any]:
        """
        Append the last action to history_buffer (capped at 3 by Reducer),
        update retry_count, and write the trace log entry.
        """
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
            "success": bool(result),
        }

        # Write structured trace log
        self.logger.log_trace(
            agent_called=module,
            reasoning=reasoning,
            decision_latency=decision_latency,
            execution_latency=exec_latency,
            success=bool(result),
            context_id=context_id,
            extra_info={"result": result},
        )

        logger.info("[update_memory_node] History updated and trace logged.")

        return {
            "history_buffer": [mem_entry],
            "retry_count": state.get("retry_count", 0) + 1,
            "current_status": "MEMORY_UPDATED",
        }

    async def aclose(self) -> None:
        """Clean up the shared httpx client."""
        await self.http_client.aclose()
