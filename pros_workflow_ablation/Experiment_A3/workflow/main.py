"""
main.py — VLM Grasping System Entry Point

Starts the LangGraph orchestration loop.
All four agent nodes run within this single process.
"""

import asyncio
import logging
import os
import sys
import uuid

import click

from commander import load_project_env

load_project_env()

from commander.experiment_report import ExperimentReport
from commander.logger import TraceLogger
from commander.orchestrator import Orchestrator
from commander.storage.session_store import SessionMemoryStore
from commander.state import create_initial_state


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
    VLM multi-agent grasping system.

    Runs a LangGraph stateful loop where Brain (VLM) orchestrates
    navigation, GraspGen, and car approach nodes to complete a grasping task.
    """
    # Resolve mock mode: CLI flag > env var > default True
    if no_mock_mode:
        use_mock = False
    elif mock_mode:
        use_mock = True
    else:
        use_mock = os.getenv("MOCK_MODE", "true").lower() == "true"

    log_path = log_file or os.getenv("TRACE_LOG_FILE", "")

    try:
        _validate_env(use_mock)
    except MissingConfigError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    asyncio.run(_run(use_mock=use_mock, max_steps=max_steps, log_path=log_path))


async def _run(use_mock: bool, max_steps: int, log_path: str) -> None:
    """Async main loop for the LangGraph orchestration."""
    mode_label = "MOCK" if use_mock else "REAL"
    logger.info(f"Starting VLM grasp system [{'MOCK' if use_mock else 'REAL'} MODE]")

    trace_logger = TraceLogger(log_file=log_path)
    context_id = uuid.uuid4().hex
    orchestrator = await Orchestrator.create(trace_logger=trace_logger, use_mock=use_mock)

    # Build initial LangGraph state
    initial_state = create_initial_state(context_id)
    session_store = SessionMemoryStore(context_id=context_id, initial_state=initial_state)

    logger.info(f"Starting LangGraph loop | context_id={context_id}")

    # Ablation A3 execution recorder: from get_item_info_no_sam3d_node to END.
    report = ExperimentReport(context_id=context_id)

    step = 0
    try:
        async for event in orchestrator.graph.astream(
            initial_state,
            config={"configurable": {"thread_id": context_id}, "recursion_limit": max_steps * 8},
        ):
            for node_name, state_update in event.items():
                status = state_update.get("current_status", "")
                module = (state_update.get("decision", {}) or {}).get("call_module", "")
                step += 1
                session_store.record_event(
                    step=step,
                    node_name=node_name,
                    state_update=state_update,
                )
                report.ingest(node_name, state_update)
                print(
                    f"\n[Step {step:02d}] Node='{node_name}' "
                    f"status={status} module={module}"
                )

    except Exception as e:
        logger.error(f"LangGraph execution error: {e}", exc_info=True)
    finally:
        await orchestrator.aclose()

    report_path = report.write()

    print("\n" + "=" * 60)
    print(f"  Task complete. Trace log: {log_path}")
    if report_path:
        print(f"  Experiment A3 report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
