"""
__main__.py — A2A Server Entrypoint for GetItemInfoAgent.

Usage:
    cd /path/to/pros_workflow/3090server/pros_workflow
    conda activate get_item_info_agent
    python -m get_item_info_agent
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

# Allow importing sibling packages (a2a_utils, get_item_info_agent) when run directly.
_AGENT_SERVICES_ROOT = Path(__file__).parent.parent
if str(_AGENT_SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_SERVICES_ROOT))

from get_item_info_agent.agent_executor import GetItemInfoExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the project .env (if present) so EXTERNAL_IP can be configured there instead of
# being hardcoded or passed on every launch. An explicit shell env var still wins.
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

PORT = int(os.getenv("GET_ITEM_INFO_LEGACY_PORT", "8008"))
EXTERNAL_IP = os.getenv("EXTERNAL_IP", "127.0.0.1")

agent_card = AgentCard(
    name="Get Item Info Agent",
    description=(
        "Receives a stereo image pair and a YOLO class name, runs the full perception pipeline "
        "(YOLO → SAM → Triangulation → DepthAnything → SAM3D → GraspGen), "
        "and returns the 3D world position and ranked goal poses as JSON."
    ),
    url=f"http://{EXTERNAL_IP}:{PORT}/",
    version="2.0.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="get_item_info",
            name="3D Item Info Estimation",
            description=(
                "Estimate 3D world position and ranked goal poses for a target object "
                "from a stereo RGB image pair."
            ),
            tags=["3d", "spatial", "grasp", "stereo"],
            examples=["get 3D goal pose for object class 'doll'"],
            # Part 0: JSON text; Part 1 & 2: image bytes (base64 or inline data)
            input_modes=["text", "data"],
            output_modes=["text"],
        )
    ],
    default_input_modes=["text", "data"],
    default_output_modes=["text"],
)


def main() -> None:
    logger.info("Starting GetItemInfoAgent A2A Server on port %d ...", PORT)
    executor = GetItemInfoExecutor()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)
    uvicorn.run(app.build(), host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
