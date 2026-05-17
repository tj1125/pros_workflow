from __future__ import annotations

from typing import Annotated, Any, Dict, List, TypedDict


def _append_history(existing: List, new: List) -> List:
    return existing + new


def _merge_dict(existing: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing or {})
    merged.update(new or {})
    return merged


class CommanderState(TypedDict):
    """Narrow LangGraph state. Large data lives in ArtifactStore."""

    context_id: str
    interface_mode: str
    current_status: str
    task_complete: bool

    greeting_sent: bool
    human_reply: str
    task_intent: str
    selected_object_index: int
    ai_reply: str
    task: Dict[str, Any]
    requested_object: Dict[str, Any]
    selected_instance: Dict[str, Any]
    world_position: Dict[str, Any]
    room_cameras: Dict[str, Any]
    item_info: Dict[str, Any]
    navigation: Annotated[Dict[str, Any], _merge_dict]
    observation: Dict[str, Any]
    decision: Dict[str, Any]
    module_params: Dict[str, Any]
    grasp_result: Dict[str, Any]
    approach_result: Dict[str, Any]
    last_execution: Dict[str, Any]

    session_summary: str
    history_buffer: Annotated[List[Dict[str, Any]], _append_history]
    retry_count: int


def create_initial_state(context_id: str) -> CommanderState:
    """Build a fully-populated initial CommanderState for a new session."""
    return {
        "context_id": context_id,
        "interface_mode": "cli",
        "current_status": "INIT",
        "task_complete": False,
        "greeting_sent": False,
        "human_reply": "",
        "task_intent": "",
        "selected_object_index": 0,
        "ai_reply": "",
        "task": {},
        "requested_object": {},
        "selected_instance": {},
        "world_position": {},
        "room_cameras": {},
        "item_info": {},
        "navigation": {
            "current_goal_rank": 1,
            "current_goal_pose_index": 0,
            "goal_pose_db": {},
            "nav_goal": {},
            "nav_goal_pose_source": "",
            "nav_move_source": "",
            "force_initialpose": False,
            "result": {},
        },
        "observation": {"description": "System initialising..."},
        "decision": {},
        "module_params": {},
        "grasp_result": {},
        "approach_result": {},
        "last_execution": {},
        "session_summary": "",
        "history_buffer": [],
        "retry_count": 0,
    }
