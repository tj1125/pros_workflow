from typing import Annotated, TypedDict, List, Dict, Any, Optional
import operator


def _keep_last_three(existing: List, new: List) -> List:
    """Reducer: append new items and keep only the last 3 entries."""
    combined = existing + new
    return combined[-3:]


class CommanderState(TypedDict):
    """
    LangGraph global state for the VLM-RL grasping system.
    Tracks the full lifecycle of an observation-reason-act cycle.
    """

    # Current environment observation (image data or mock description)
    current_observation: Dict[str, Any]

    # VLM reasoning output
    reasoning: str

    # Target agent module to invoke (nav_agent / grasp_agent / approach_agent / view_agent / DONE)
    call_module: str

    # Parameters to pass into the target agent
    module_params: Dict[str, Any]

    # Sliding window memory: last 3 action records
    # operator.add causes LangGraph to APPEND new entries
    history_buffer: Annotated[List[Dict[str, Any]], _keep_last_three]

    # Current pipeline status string (INIT / OBSERVED / REASONED / EXECUTED / DONE)
    current_status: str

    # Shared context ID for A2A session tracking
    context_id: str

    # Number of action cycles completed
    retry_count: int

    # Latency measured during the reasoning step (seconds)
    decision_latency: float

    # Result text returned by the last executed agent
    agent_result: str

    # Whether the task has been marked complete
    task_complete: bool
