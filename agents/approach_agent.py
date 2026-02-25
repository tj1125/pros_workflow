"""
approach_agent.py — Approach Agent Node

Role in the system:
  - LangGraph Node: called by the Orchestrator via Conditional Edge
  - Local control only: no corresponding GPU inference server defined in flow-chart.md

Guides the robot arm to approach the pre-grasp point.
In real mode, sends control commands via ROS/Rosbridge or direct hardware API.
"""

import asyncio
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ApproachAgent:
    """
    Approach Agent Node: drives the arm toward the pre-grasp point.

    No A2A call to RTX 3090 (not defined in flow-chart.md).
    Executes local kinematics / ROS control logic.
    """

    AGENT_NAME = "Approach Agent"

    def __init__(self):
        self._use_mock = os.getenv("MOCK_MODE", "true").lower() == "true"
        self._rosbridge_url = os.getenv("ROSBRIDGE_URL", "ws://localhost:9090")

    async def execute(
        self, params: Dict[str, Any], context_id: str = ""
    ) -> Dict[str, Any]:
        if self._use_mock:
            return await self._mock_execute(params)
        return await self._real_execute(params)

    async def _mock_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(0.7)
        target = params.get("target_id", "target")
        logger.info(f"[{self.AGENT_NAME}] Mock: approaching {target}")
        return {
            "result": f"[MockApproach] Arm moved to pre-grasp position for '{target}'",
            "success": True,
        }

    async def _real_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Real execution: publish an approach command via Rosbridge.
        TODO (Day-2): implement WebSocket publish to /arm_controller/command
        """
        logger.warning(
            f"[{self.AGENT_NAME}] Real execution not yet implemented. "
            "Falling back to mock."
        )
        return await self._mock_execute(params)
