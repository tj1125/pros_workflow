"""
a2a_utils/response.py — Unified A2A response helpers

Standardize the way AgentExecutors return execution results and handle errors
in a structure compatible with the official a2a-sdk.
"""

import json

from a2a.types import (
    Artifact,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message


def build_success_artifact(data: dict) -> dict:
    """
    Wrap a result dictionary into a standard A2A completed task event.
    Returns the kwargs for `event_queue.enqueue_event()`.
    
    The Commander expects `artifacts[0].parts[0].root.text` to be valid JSON.
    """
    return {
        "event": {
            "state": TaskState.completed,
            "artifacts": [
                Artifact(
                    artifact_id="result",
                    parts=[Part(root=TextPart(text=json.dumps(data)))],
                )
            ],
        }
    }


def build_error_artifact(message: str) -> dict:
    """
    Wrap an error message into a failed task event.
    """
    return {
        "event": {
            "state": TaskState.failed,
            "artifacts": [
                Artifact(
                    artifact_id="error",
                    parts=[Part(root=TextPart(text=json.dumps({"error": message})))],
                )
            ],
        }
    }
