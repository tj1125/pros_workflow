import json
import logging
from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared Pydantic response schema for all Agent Nodes
# ---------------------------------------------------------------------------

class AgentResponseFormat(BaseModel):
    """Structured response returned by each Agent Node after execution."""

    status: Literal["completed", "error"] = Field(
        description="Whether the agent completed successfully or encountered an error"
    )
    message: str = Field(
        description="Human-readable summary of what the agent did"
    )
    result_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured data from the inference result (e.g., grasp pose, nav waypoint)",
    )


class AgentFeedback(BaseModel):
    """Simple feedback record stored in the state history buffer."""

    success: bool
    status_message: str
    feedback_data: Dict[str, Any] = Field(default_factory=dict)
