"""
__main__.py — A2A Server Entrypoint for GetItemInfoAgent

Starts the GetItemInfo A2A server.
Usage: python -m get_item_info_agent
"""

import logging
import sys
from pathlib import Path

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

# Allow importing the local packages if run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from get_item_info_agent.agent_executor import GetItemInfoExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = 8006

agent_card = AgentCard(
    name="Get Item Info Agent Server",
    description="Receives camera and 2D bounding box, returns 3D world position and size estimation.",
    url=f"http://0.0.0.0:{PORT}/",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="estimate_3d",
            name="3D Object Estimation",
            description="Estimate 3D world position and size from 2D bounding box.",
            tags=["3d", "spatial"],
            examples=["get 3d position of the detected object"],
            input_modes=["text"],
            output_modes=["text"],
        )
    ],
    default_input_modes=["text"],
    default_output_modes=["text"],
)

def main():
    logger.info(f"Starting GetItemInfoAgent A2A Server on port {PORT}...")
    executor = GetItemInfoExecutor()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)
    uvicorn.run(app.build(), host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()
