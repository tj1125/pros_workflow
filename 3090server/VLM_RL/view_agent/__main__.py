"""
view_agent/__main__.py — A2A Server Entrypoint for ViewAgent

Starts the View Agent SAC Policy inference A2A server on Port 8007.

Usage:
    cd /path/to/VLM_RL/3090server/VLM_RL
    conda activate a2a_vlm_view
    python -m view_agent
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

# Allow importing sibling packages (a2a_utils, view_agent) when run directly.
_AGENT_SERVICES_ROOT = Path(__file__).parent.parent
if str(_AGENT_SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_SERVICES_ROOT))

from view_agent.agent_executor import ViewAgentExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = 8007
EXTERNAL_IP = os.getenv("EXTERNAL_IP", "140.116.82.226")

agent_card = AgentCard(
    name="View Agent Server",
    description=(
        "Receives up to 3 stacked RGB + depth frames, current joint state, and "
        "last action history. Runs CLIP+DINOv2 temporal feature extraction and a "
        "SAC policy network to produce a 6-DOF arm joint delta for active viewpoint "
        "adjustment, resolving mild gripper-target occlusion."
    ),
    url=f"http://{EXTERNAL_IP}:{PORT}/",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="sac_view_policy",
            name="SAC Viewpoint Adjustment Policy",
            description=(
                "Given temporal RGB frames and robot state, return a 6-DOF "
                "arm joint delta action that improves the observation viewpoint."
            ),
            tags=["rl", "sac", "clip", "dinov2", "viewpoint", "occlusion"],
            examples=["adjust arm to improve view of target object"],
            input_modes=["text"],
            output_modes=["text"],
        )
    ],
    default_input_modes=["text"],
    default_output_modes=["text"],
)


def main() -> None:
    logger.info(f"Starting ViewAgent A2A Server on port {PORT}...")
    executor = ViewAgentExecutor()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)
    uvicorn.run(app.build(), host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
