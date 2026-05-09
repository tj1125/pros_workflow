from typing import Annotated, TypedDict, List, Dict, Any


def _keep_last_three(existing: List, new: List) -> List:
    """Reducer: append new items and keep only the last 3 entries."""
    combined = existing + new
    return combined[-3:]


LATEST_RESULT_KEYS = (
    "latest_nav_result",
    "latest_grasp_result",
    "latest_approach_result",
)


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

    # Target agent module to invoke (nav_agent / grasp_agent / car_approach_agent / DONE)
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

    # Result payload returned by the last executed agent
    agent_result: Any

    # Whether the task has been marked complete
    task_complete: bool

    # Confirmed target object (set by find_node after user confirmation)
    # Contains: id, label, position_3d, camera
    target_object: Dict[str, Any]

    # Candidate objects returned by find_node (before user confirmation)
    candidate_objects: List[Dict[str, Any]]

    # Fully-prepared target selected by find_node for get_item_info_no_sam3d_node
    selected_target: Dict[str, Any]

    # 1-based index pointing to the rank of the current goal pose to attempt
    current_goal_rank: int

    # Whether find_node has been completed (prevents re-running)
    find_complete: bool

    # Candidate detections from find_node: key = display number (1-based)
    # Value: {instance_key, center_world, camsrc, bboxes_by_camera, preview_path, ...}
    yolo_detections: Dict[int, Dict[str, Any]]

    # User-selected detection ID (0 = user typed "no")
    selected_detection_id: int

    # nav_move routing source: bootstrap (from get_item_info) or reason_loop (from nav_node)
    nav_move_source: str

    # Computed goal pose for navigation runner
    nav_goal_pose: Dict[str, Any]

    # Last AMCL pose recorded after car_approach finished moving the base
    last_car_approach_amcl_pose: Dict[str, Any]

    # Last arm base joint angle recorded after car_approach aligned the arm base
    last_arm_base_alignment_result: Dict[str, Any]

    # nav_move execution flags
    nav_plan_ready: bool
    nav_arrived: bool
    nav_attempt: int
    force_initialpose: bool

    # Navigation and action execution feedback
    nav_move_events: List[Dict[str, Any]]
    agent_success: bool

    # Latest structured results for downstream nodes and debugging
    latest_nav_result: Dict[str, Any]
    latest_grasp_result: Dict[str, Any]
    latest_approach_result: Dict[str, Any]


def create_initial_state(
    context_id: str,
    *,
    task_description: str = "",
    current_observation: Dict[str, Any] | None = None,
) -> CommanderState:
    """Build a fully-populated initial CommanderState."""
    return {
        "task_description": task_description,
        "current_observation": current_observation or {"description": "System initialising..."},
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
        "selected_target": {},
        "find_complete": False,
        "yolo_detections": {},
        "selected_detection_id": 0,
        "current_goal_rank": 1,
        "nav_move_source": "",
        "nav_goal_pose": {},
        "last_car_approach_amcl_pose": {},
        "last_arm_base_alignment_result": {},
        "nav_plan_ready": False,
        "nav_arrived": False,
        "nav_attempt": 0,
        "force_initialpose": False,
        "nav_move_events": [],
        "agent_success": False,
        "latest_nav_result": {},
        "latest_grasp_result": {},
        "latest_approach_result": {},
    }
