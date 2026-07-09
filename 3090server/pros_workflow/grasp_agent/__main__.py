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


SERVER_ROOT = Path(__file__).parent.parent
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from tool.runtime.env import load_env

load_env()  # populate EXTERNAL_IP (and INF_* URLs) from 3090server/pros_workflow/.env

from grasp_agent.agent_executor import GraspAgentExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = 8007
EXTERNAL_IP = os.getenv("EXTERNAL_IP", "140.116.82.226")

agent_card = AgentCard(
    name="Grasp Agent Server",
    description=(
        "Receives Camera_Car RGBD plus a target object id, runs YOLO and SAM segmentation, "
        "then returns the highest-confidence 6-DoF grasp pose."
    ),
    url=f"http://{EXTERNAL_IP}:{PORT}/",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="generate_grasp_pose",
            name="RGBD Grasp Pose Generation",
            description="Generate the highest-confidence grasp pose from a single RGBD observation.",
            tags=["grasp", "rgbd", "yolo", "sam"],
            examples=["generate grasp pose for object 'doll' from Camera_Car RGBD"],
            input_modes=["data", "file"],
            output_modes=["data"],
        )
    ],
    default_input_modes=["data", "file"],
    default_output_modes=["data"],
)


def main() -> None:
    logger.info("Starting GraspAgent A2A Server on port %d ...", PORT)
    executor = GraspAgentExecutor()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)
    uvicorn.run(app.build(), host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
