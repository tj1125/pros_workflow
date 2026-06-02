"""
test_client.py — End-to-end test client for the VLM grasp system

Tests two scenarios:
1. Mock LangGraph loop: verifies the Orchestrator graph starts and terminates cleanly.
2. A2A connectivity check: verifies configured RTX 3090 A2A servers expose AgentCards.
   (only runs when INF_xxx_URL env vars are set)

Usage:
    python test_client.py
    python test_client.py --mock-only
"""

import asyncio
import logging
import os
import sys
import uuid

import click
import httpx
from commander import load_project_env

load_project_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Test 1: Mock LangGraph loop
# ---------------------------------------------------------------------------

async def test_mock_langgraph_loop() -> bool:
    """
    Run the full Orchestrator graph in mock mode and verify it terminates cleanly.
    Expected: graph executes at least one full mock session path and exits cleanly.
    """
    from commander.logger import TraceLogger
    from commander.orchestrator import Orchestrator
    from commander.session_store import SessionMemoryStore
    from commander.state import create_initial_state

    print("\n" + "=" * 50)
    print("Test 1: Mock LangGraph Loop")
    print("=" * 50)

    trace_logger = TraceLogger(log_file="test_trace.jsonl")
    orchestrator = await Orchestrator.create(trace_logger=trace_logger, use_mock=True)

    context_id = uuid.uuid4().hex
    initial_state = create_initial_state(context_id)
    initial_state.update(
        {
            "human_reply": "Test: grab apple on the table",
            "observation": {"description": "Test scene: apple on white table"},
        }
    )
    session_store = SessionMemoryStore(context_id=context_id, initial_state=initial_state)

    steps_executed = []
    try:
        async for event in orchestrator.graph.astream(
            initial_state,
            config={"configurable": {"thread_id": context_id}, "recursion_limit": 40},
        ):
            for node_name, state_update in event.items():
                status = state_update.get("current_status", "")
                module = (state_update.get("decision", {}) or {}).get("call_module", "")
                steps_executed.append(node_name)
                session_store.record_event(
                    step=len(steps_executed),
                    node_name=node_name,
                    state_update=state_update,
                )
                print(f"  ✓ Node='{node_name}' status={status} module={module}")

    finally:
        await orchestrator.aclose()

    success = len(steps_executed) > 0
    print(f"\n  Result: {'PASS ✅' if success else 'FAIL ❌'}")
    print(f"  Total steps: {len(steps_executed)}")
    return success


# ---------------------------------------------------------------------------
# Test 2: A2A connectivity to RTX 3090 inference servers
# ---------------------------------------------------------------------------

async def test_a2a_connectivity(server_name: str, url: str) -> bool:
    """
    Verify that an A2A inference server's AgentCard is reachable.
    Mimics the pattern from a2a-samples/test_client.py.
    """
    from a2a.client import A2ACardResolver

    print(f"\n  Testing {server_name} @ {url}")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resolver = A2ACardResolver(httpx_client=client, base_url=url)
            card = await resolver.get_agent_card()
            print(f"  ✓ AgentCard received: name='{card.name}', version={card.version}")
            return True
    except Exception as e:
        print(f"  ✗ Failed to reach {server_name}: {e}")
        return False


async def test_all_a2a_servers() -> bool:
    """Check all configured inference server endpoints."""
    print("\n" + "=" * 50)
    print("Test 2: A2A Inference Server Connectivity")
    print("=" * 50)

    servers = {
        "Find Agent": os.getenv("INF_FIND_URL", ""),
        "Get Item Info No SAM3D": os.getenv("INF_GET_ITEM_INFO_NO_SAM3D_URL", ""),
        "Get Item Info Legacy SAM3D": os.getenv("INF_GET_ITEM_INFO_URL", ""),
        "Grasp Agent": os.getenv("INF_GRASP_URL", ""),
    }

    configured = {k: v for k, v in servers.items() if v}
    if not configured:
        print("  ⚠ No INF_xxx_URL env vars set — skipping A2A connectivity tests")
        return True

    results = await asyncio.gather(
        *[test_a2a_connectivity(name, url) for name, url in configured.items()]
    )
    all_pass = all(results)
    print(f"\n  Result: {'PASS ✅' if all_pass else 'PARTIAL/FAIL ❌'}")
    return all_pass


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--mock-only",
    is_flag=True,
    default=False,
    help="Only run mock LangGraph loop test (skip A2A server tests)",
)
def main(mock_only: bool) -> None:
    """VLM grasp system end-to-end test client."""
    results = []

    async def _run():
        r1 = await test_mock_langgraph_loop()
        results.append(r1)

        if not mock_only:
            r2 = await test_all_a2a_servers()
            results.append(r2)

    asyncio.run(_run())

    print("\n" + "=" * 50)
    overall = all(results)
    print(f"  Overall: {'ALL TESTS PASSED ✅' if overall else 'SOME TESTS FAILED ❌'}")
    print("=" * 50)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
