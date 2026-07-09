"""A2A response helpers that return Task artifacts with DataPart payloads."""

from __future__ import annotations

import uuid
from typing import Any

from a2a.utils import completed_task, new_data_artifact


def build_success(data: dict[str, Any], context: Any | None = None, *, name: str = "result") -> object:
    return completed_task(
        task_id=_task_id(context),
        context_id=_context_id(context),
        artifacts=[new_data_artifact(name, data)],
        history=_history(context),
    )


def build_error(message: str, context: Any | None = None) -> object:
    return completed_task(
        task_id=_task_id(context),
        context_id=_context_id(context),
        artifacts=[new_data_artifact("error", {"error": message})],
        history=_history(context),
    )


def _message(context: Any | None) -> Any | None:
    return getattr(context, "message", None) if context is not None else None


def _task_id(context: Any | None) -> str:
    message = _message(context)
    return str(getattr(message, "task_id", "") or uuid.uuid4().hex)


def _context_id(context: Any | None) -> str:
    message = _message(context)
    return str(getattr(message, "context_id", "") or uuid.uuid4().hex)


def _history(context: Any | None) -> list[Any] | None:
    message = _message(context)
    return [message] if message is not None else None
