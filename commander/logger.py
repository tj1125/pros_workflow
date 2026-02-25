import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import logging

logger = logging.getLogger(__name__)


class TraceLogger:
    """
    Records each Observe-Reason-Act cycle to a JSONL file.
    Each line contains timing, agent identity, reasoning, and success status.
    """

    def __init__(self, log_file: str = "trace_logger.jsonl"):
        self.log_file = log_file

    def log_trace(
        self,
        agent_called: str,
        reasoning: str,
        decision_latency: float,
        execution_latency: float,
        success: bool,
        context_id: str = "",
        task_id: str = "",
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one complete trace entry to the JSONL log file."""
        entry = {
            # ISO 8601 timestamp for easy parsing
            "iso_timestamp": datetime.now(timezone.utc).isoformat(),
            "unix_timestamp": time.time(),
            "context_id": context_id,
            "task_id": task_id,
            "agent_called": agent_called,
            "reasoning": reasoning,
            "decision_latency_sec": round(decision_latency, 4),
            "execution_latency_sec": round(execution_latency, 4),
            "success": success,
            "extra_info": extra_info or {},
        }

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error(f"[TraceLogger] Failed to write trace: {e}")

    def __repr__(self) -> str:
        return f"TraceLogger(file='{self.log_file}')"
