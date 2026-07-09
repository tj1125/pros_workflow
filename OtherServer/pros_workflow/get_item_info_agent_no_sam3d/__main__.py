"""
__main__.py — A2A Server Entrypoint for GetItemInfoAgentNoSam3D.

Usage:
    cd /path/to/pros_workflow/3090server/pros_workflow
    conda activate <your_env>
    python -m get_item_info_agent_no_sam3d
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

_AGENT_SERVICES_ROOT = Path(__file__).parent.parent
if str(_AGENT_SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_SERVICES_ROOT))

from get_item_info_agent_no_sam3d.agent_executor import GetItemInfoNoSam3dExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the project .env (if present) so EXTERNAL_IP can be configured there instead of
# being hardcoded or passed on every launch. An explicit shell env var still wins.
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

PORT = 8006
EXTERNAL_IP = os.getenv("EXTERNAL_IP", "127.0.0.1")

agent_card = AgentCard(
    name="Get Item Info Agent No SAM3D",
    description=(
        "Receives multi-view RGB images plus /world_position_data observations, "
        "runs SAM + geometry-only multi-camera size fusion without SAM3D, "
        "and returns 3D item info and ranked goal poses as JSON."
    ),
    url=f"http://{EXTERNAL_IP}:{PORT}/",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="get_item_info_no_sam3d",
            name="3D Item Info Estimation Without SAM3D",
            description=(
                "Estimate 3D world position, fused object size, and ranked goal poses "
                "from multi-view RGB images and /world_position_data."
            ),
            tags=["3d", "spatial", "grasp", "multiview", "sam", "geometry"],
            examples=["get 3D goal pose for object class 'doll' from /world_position_data"],
            input_modes=["data", "file"],
            output_modes=["data"],
        )
    ],
    default_input_modes=["data", "file"],
    default_output_modes=["data"],
)


def main() -> None:
    logger.info("Starting GetItemInfoAgentNoSam3D A2A Server on port %d ...", PORT)
    executor = GetItemInfoNoSam3dExecutor()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)
    uvicorn.run(app.build(), host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
