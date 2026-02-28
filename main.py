"""
main.py — VLM-RL Grasping System Entry Point

Starts the LangGraph orchestration loop.
All four agent nodes run within this single process.
"""

import asyncio
import logging
import os
import sys
import uuid

import click
from dotenv import load_dotenv

from commander.logger import TraceLogger
from commander.orchestrator import Orchestrator


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class MissingConfigError(Exception):
    """Raised when required environment variables are not set."""


def _validate_env(mock_mode: bool) -> None:
    """Check required env vars for non-mock operation."""
    if mock_mode:
        return

    provider = os.getenv("VLM_PROVIDER", "google").lower()
    if provider == "google" and not os.getenv("GOOGLE_API_KEY"):
        raise MissingConfigError(
            "GOOGLE_API_KEY is required when VLM_PROVIDER=google. "
            "Set it in .env or use --mock flag."
        )
    if provider == "ollama" and not os.getenv("OLLAMA_BASE_URL"):
        raise MissingConfigError(
            "OLLAMA_BASE_URL is required when VLM_PROVIDER=ollama."
        )


@click.command()
@click.option(
    "--mock",
    "mock_mode",
    is_flag=True,
    default=None,
    help="Run in mock mode (no VLM or GPU calls)",
)
@click.option(
    "--no-mock",
    "no_mock_mode",
    is_flag=True,
    default=False,
    help="Force real VLM and GPU calls",
)
@click.option(
    "--max-steps",
    default=10,
    show_default=True,
    help="Maximum number of Observe-Reason-Act cycles",
)
@click.option(
    "--log-file",
    default=None,
    help="Path to JSONL trace log file (overrides TRACE_LOG_FILE env var)",
)
def main(
    mock_mode: bool,
    no_mock_mode: bool,
    max_steps: int,
    log_file: str,
) -> None:
    """
    VLM-RL Multi-Agent Active Perception Grasping System.

    Runs a LangGraph stateful loop where Brain (VLM) orchestrates
    Nav / GraspGen / Approach / View agent nodes to complete a grasping task.
    """
    # Resolve mock mode: CLI flag > env var > default True
    if no_mock_mode:
        use_mock = False
    elif mock_mode:
        use_mock = True
    else:
        use_mock = os.getenv("MOCK_MODE", "true").lower() == "true"

    log_path = log_file or os.getenv("TRACE_LOG_FILE", "logs/trace_logger.jsonl")

    try:
        _validate_env(use_mock)
    except MissingConfigError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    asyncio.run(_run(use_mock=use_mock, max_steps=max_steps, log_path=log_path))


async def _run(use_mock: bool, max_steps: int, log_path: str) -> None:
    """Async main loop for the LangGraph orchestration."""
    mode_label = "MOCK" if use_mock else "REAL"
    logger.info(f"Starting VLM-RL system [{'MOCK' if use_mock else 'REAL'} MODE]")

    trace_logger = TraceLogger(log_file=log_path)
    orchestrator = Orchestrator(trace_logger=trace_logger, use_mock=use_mock)

    # Build initial LangGraph state
    context_id = uuid.uuid4().hex
    initial_state = {
        "task_description": "",  # filled by input_node
        "current_observation": {"description": "System initialising..."},
        "reasoning": "",
        "call_module": "",
        "module_params": {},
        "history_buffer": [],
        "current_status": "INIT",
        "context_id": context_id,
        "retry_count": 0,
        "decision_latency": 0.0,
        "agent_result": "",
        "task_complete": False,
        "target_object": {},
        "candidate_objects": [],
        "find_complete": False,
        "yolo_detections": {},
        "selected_detection_id": 0,
    }

    logger.info(f"Starting LangGraph loop | context_id={context_id}")

    step = 0
    try:
        async for event in orchestrator.graph.astream(
            initial_state,
            config={"recursion_limit": max_steps * 4},
        ):
            for node_name, state_update in event.items():
                status = state_update.get("current_status", "")
                module = state_update.get("call_module", "")
                step += 1
                print(
                    f"\n[Step {step:02d}] Node='{node_name}' "
                    f"status={status} module={module}"
                )

    except Exception as e:
        logger.error(f"LangGraph execution error: {e}", exc_info=True)
    finally:
        await orchestrator.aclose()

    print("\n" + "=" * 60)
    print(f"  Task complete. Trace log: {log_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
