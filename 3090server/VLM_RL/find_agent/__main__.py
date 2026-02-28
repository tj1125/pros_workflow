"""
__main__.py — A2A Server Entrypoint for FindAgent

Starts the FindAgent A2A server.
Usage: python -m find_agent
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

from find_agent.agent_executor import FindAgentExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = 8005

agent_card = AgentCard(
    name="Find Agent Server",
    description="Receives multi-camera RGB images, runs YOLO detection, and returns globally-numbered candidate objects.",
    url=f"http://0.0.0.0:{PORT}/",
    version="1.0.0",
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        AgentSkill(
            id="yolo_detect",
            name="YOLO Object Detection",
            description="Run YOLO on multi-camera images and return detections.",
            input_modes=["text"],
            output_modes=["text"],
        )
    ],
    default_input_modes=["text"],
    default_output_modes=["text"],
)

def main():
    logger.info(f"Starting FindAgent A2A Server on port {PORT}...")
    executor = FindAgentExecutor()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)
    uvicorn.run(app.build(), host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()
