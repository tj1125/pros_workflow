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

    # Human task description: set once at the start by the operator
    task_description: str

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

    # Confirmed target object (set by find_node after user confirmation)
    # Contains: id, label, position_3d, camera
    target_object: Dict[str, Any]

    # Candidate objects returned by find_node (before user confirmation)
    candidate_objects: List[Dict[str, Any]]

    # 1-based index pointing to the rank of the current goal pose to attempt
    current_goal_rank: int

    # Whether find_node has been completed (prevents re-running)
    find_complete: bool

    # YOLO detections from find_node: key = global detection number (1-based)
    # Value: {camera, group_id, bbox, label, conf, annotated_image_base64}
    yolo_detections: Dict[int, Dict[str, Any]]

    # User-selected detection ID (0 = user typed "no")
    selected_detection_id: int

    # nav_move routing source: bootstrap (from get_item_info) or reason_loop (from nav_node)
    nav_move_source: str

    # Computed goal pose for navigation runner
    nav_goal_pose: Dict[str, Any]

    # nav_move execution flags
    nav_plan_ready: bool
    nav_arrived: bool
    nav_attempt: int
    force_initialpose: bool

    # Navigation and action execution feedback
    nav_move_events: List[Dict[str, Any]]
    agent_success: bool
