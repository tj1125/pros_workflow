import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, Literal

import httpx
from langgraph.graph import END, StateGraph

from .brain import Brain
from .logger import TraceLogger
from .state import CommanderState

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Builds and runs the LangGraph decision graph.

    Graph topology:
        observe_node → reason_node
            reason_node --[nav_agent]-------> nav_node ┐
            reason_node --[grasp_agent]-----> grasp_node ├─→ update_memory_node → observe_node
            reason_node --[approach_agent]--> approach_node ┘
            reason_node --[view_agent]------> view_node ┘
            reason_node --[DONE]-----------> END
    """

    def __init__(self, trace_logger: TraceLogger, use_mock: bool = True):
        self.logger = trace_logger
        self.brain = Brain(use_mock=use_mock)
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.use_mock = use_mock
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> Any:
        workflow = StateGraph(CommanderState)

        # Core nodes
        workflow.add_node("observe_node", self._observe_node)
        workflow.add_node("reason_node", self._reason_node)
        workflow.add_node("update_memory_node", self._update_memory_node)

        # Individual Agent nodes (A2A Clients)
        workflow.add_node("nav_node", self._nav_node)
        workflow.add_node("grasp_node", self._grasp_node)
        workflow.add_node("approach_node", self._approach_node)
        workflow.add_node("view_node", self._view_node)

        # Entry point
        workflow.set_entry_point("observe_node")

        # Fixed edges
        workflow.add_edge("observe_node", "reason_node")
        workflow.add_edge("nav_node", "update_memory_node")
        workflow.add_edge("grasp_node", "update_memory_node")
        workflow.add_edge("approach_node", "update_memory_node")
        workflow.add_edge("view_node", "update_memory_node")
        workflow.add_edge("update_memory_node", "observe_node")

        # Conditional routing: reason_node → agent node or END
        workflow.add_conditional_edges(
            "reason_node",
            self._route_decision,
            {
                "nav_node":      "nav_node",
                "grasp_node":    "grasp_node",
                "approach_node": "approach_node",
                "view_node":     "view_node",
                "end":           END,
            },
        )

        return workflow.compile()

    # ------------------------------------------------------------------
    # Node: observe
    # ------------------------------------------------------------------

    async def _observe_node(self, state: CommanderState) -> Dict[str, Any]:
        """Fetch current environment observation (mock or Rosbridge)."""
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
    # Conditional edge
    # ------------------------------------------------------------------

    def _route_decision(
        self, state: CommanderState
    ) -> Literal["nav_node", "grasp_node", "approach_node", "view_node", "end"]:
        module = state.get("call_module", "")
        if module == "DONE" or state.get("task_complete", False):
            logger.info("[route] Task complete — ending graph.")
            return "end"
        mapping = {
            "nav_agent":      "nav_node",
            "grasp_agent":    "grasp_node",
            "approach_agent": "approach_node",
            "view_agent":     "view_node",
        }
        return mapping.get(module, "end")

    # ------------------------------------------------------------------
    # Agent nodes (each is an A2A Client calling RTX 3090)
    # ------------------------------------------------------------------

    async def _nav_node(self, state: CommanderState) -> Dict[str, Any]:
        """Nav Agent Node: navigate robot base via INF_NAV (A2A Server on RTX 3090)."""
        from agents.nav_agent import NavAgent
        agent = NavAgent(http_client=self.http_client)
        return await self._run_agent(agent, state)

    async def _grasp_node(self, state: CommanderState) -> Dict[str, Any]:
        """GraspGen Agent Node: generate 6-DoF grasp pose via INF_GRASP (A2A Server)."""
        from agents.grasp_agent import GraspAgent
        agent = GraspAgent(http_client=self.http_client)
        return await self._run_agent(agent, state)

    async def _approach_node(self, state: CommanderState) -> Dict[str, Any]:
        """Approach Agent Node: guide arm to pre-grasp point (local control, no GPU)."""
        from agents.approach_agent import ApproachAgent
        agent = ApproachAgent()
        return await self._run_agent(agent, state, has_http=False)

    async def _view_node(self, state: CommanderState) -> Dict[str, Any]:
        """View Agent Node: adjust camera/arm posture via INF_VIEW (A2A Server)."""
        from agents.view_agent import ViewAgent
        agent = ViewAgent(http_client=self.http_client)
        return await self._run_agent(agent, state)

    async def _run_agent(
        self,
        agent: Any,
        state: CommanderState,
        has_http: bool = True,
    ) -> Dict[str, Any]:
        """Common runner: call agent.execute() and return state updates."""
        params = state.get("module_params", {})
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
            "current_status": "EXECUTED",
            "_exec_latency": exec_latency,
        }

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

        mem_entry = {
            "action": module,
            "reasoning": reasoning,
            "result": result,
            "success": bool(result),
        }

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
