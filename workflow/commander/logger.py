import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import logging

logger = logging.getLogger(__name__)


class TraceLogger:
    """
    Records each Observe-Reason-Act cycle to a JSONL file.
    Each line contains timing, agent identity, reasoning, and success status.
    """

    def __init__(self, log_file: str = ""):
        self.log_file = str(log_file or "").strip()

    def log_trace(
        self,
        agent_called: str,
        reasoning: str,
        decision_latency: float,
        execution_latency: float,
        success: bool,
        context_id: str = "",
        task_id: str = "",
        trace_id: str = "",
        memory_entry: Optional[Dict[str, Any]] = None,
        state_refs: Optional[Dict[str, Any]] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one complete trace entry to the JSONL log file when enabled."""
        if not self.log_file:
            return
        entry = {
            # ISO 8601 timestamp for easy parsing
            "iso_timestamp": datetime.now(timezone.utc).isoformat(),
            "unix_timestamp": time.time(),
            "context_id": context_id,
            "task_id": task_id,
            "trace_id": trace_id,
            "agent_called": agent_called,
            "reasoning": reasoning,
            "decision_latency_sec": round(decision_latency, 4),
            "execution_latency_sec": round(execution_latency, 4),
            "success": success,
            "memory_entry": memory_entry or {},
            "state_refs": state_refs or {},
            "extra_info": extra_info or {},
        }

        try:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error(f"[TraceLogger] Failed to write trace: {e}")

    def __repr__(self) -> str:
        return f"TraceLogger(file='{self.log_file}')"
