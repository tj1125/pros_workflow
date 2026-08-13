"""Run arm approach in the Python environment that has ROS/numpy installed."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

from .runner import run_arm_approach_sync


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-id", default="")
    args = parser.parse_args()

    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
    except Exception as exc:
        print(
            json.dumps(
                {
                    "success": False,
                    "result": {
                        "success": False,
                        "status_code": "ARM_APPROACH_FAIL",
                        "phase": "payload_error",
                        "error": str(exc),
                        "next_agent": None,
                    },
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    with _redirect_fd_stdout_to_stderr(), contextlib.redirect_stdout(sys.stderr):
        result = run_arm_approach_sync(payload, context_id=args.context_id)
    print(json.dumps(_json_safe(result), ensure_ascii=False), flush=True)


@contextlib.contextmanager
def _redirect_fd_stdout_to_stderr():
    sys.stdout.flush()
    saved_stdout_fd = os.dup(1)
    try:
        os.dup2(2, 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return str(value)


if __name__ == "__main__":
    main()
