"""Repo-local FastAPI + SSE chat interface for the Commander LangGraph."""

from __future__ import annotations

import asyncio
import builtins
import copy
import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict
from urllib.parse import quote

from pydantic import BaseModel, Field
from starlette.responses import FileResponse, HTMLResponse, StreamingResponse

try:
    from fastapi import FastAPI, HTTPException
except ModuleNotFoundError:  # pragma: no cover - lets py_compile work before dependencies are installed.
    FastAPI = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]

from .. import load_project_env
from ..contracts import NodeExecution, dump_model
from ..logger import TraceLogger
from ..object_catalog import load_graspable_objects
from ..orchestrator import Orchestrator
from ..storage.session_store import SessionMemoryStore
from ..state import CommanderState, create_initial_state
from .presenter import _assistant_message_from_update, _progress_message_from_interrupt, _progress_message_from_update


load_project_env()
logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_ALLOWED_PREVIEW_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


class CreateSessionRequest(BaseModel):
    max_steps: int = Field(default=60, ge=1, le=500)
    log_file: str | None = Field(default=None)


class MessageRequest(BaseModel):
    message: str = Field(min_length=1)


class ResumeRequest(BaseModel):
    selected_detection_id: int | None = None
    value: Any | None = None


@dataclass
class WebSession:
    session_id: str
    orchestrator: Orchestrator
    store: SessionMemoryStore
    max_steps: int
    step: int = 0
    pending_interrupt: Dict[str, Any] | None = None
    lock: asyncio.Lock | None = None
    state_subscribers: set[asyncio.Queue[Dict[str, Any]]] = field(default_factory=set)
    legacy_input_queue: list[str] = field(default_factory=list)
    legacy_input_event: threading.Event | None = None
    legacy_input_value: str = ""
    active_stream_tasks: set[asyncio.Task[Any]] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.lock is None:
            self.lock = asyncio.Lock()


ACTIVE_SESSIONS: dict[str, WebSession] = {}
_SHUTDOWN_SESSION_TIMEOUT_SEC = float(os.getenv("WEB_SESSION_SHUTDOWN_TIMEOUT", "0.5"))
_SHUTDOWN_STREAM_CANCEL_TIMEOUT_SEC = float(os.getenv("WEB_STREAM_CANCEL_TIMEOUT", "0.5"))


def _trace_log_file(log_file: str | None = None) -> str:
    return log_file or os.getenv("TRACE_LOG_FILE", "logs/trace_logger.jsonl")


def _thread_config(session: WebSession) -> Dict[str, Any]:
    return {
        "configurable": {"thread_id": session.session_id},
        "recursion_limit": max(session.max_steps * 8, 40),
    }


def _json_default(value: Any) -> str:
    return str(value)


def _sse(event: str, data: Dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=_json_default)
    return f"event: {event}\ndata: {payload}\n\n"



def _state_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    task = state.get("task", {}) or {}
    decision = state.get("decision", {}) or {}
    chat_history = [entry for entry in state.get("history_buffer", []) if isinstance(entry, dict) and entry.get("type") == "chat_turn"]
    return {
        "current_status": state.get("current_status", ""),
        "task_intent": state.get("task_intent", ""),
        "selected_object_index": state.get("selected_object_index", 0),
        "task": task,
        "requested_object": state.get("requested_object", {}),
        "selected_instance": state.get("selected_instance", {}),
        "item_info": state.get("item_info", {}),
        "navigation": state.get("navigation", {}),
        "observation": state.get("observation", {}),
        "decision": decision,
        "grasp_result": state.get("grasp_result", {}),
        "approach_result": state.get("approach_result", {}),
        "retry_count": state.get("retry_count", 0),
        "task_complete": state.get("task_complete", False),
        "pending_interrupt": state.get("pending_interrupt", {}),
        "chat_history": chat_history,
        "history_buffer": state.get("history_buffer", []),
        "session_summary": state.get("session_summary", ""),
        "task_label": task.get("normalized_task") or task.get("original_user_request", ""),
        "call_module": decision.get("call_module", ""),
    }



def _event_state_update_summary(state_update: Dict[str, Any]) -> Dict[str, Any]:
    omitted_suffixes = ("_base64",)
    omitted_keys = {"camera_images", "world_position_data"}

    def scrub(value: Any) -> Any:
        if isinstance(value, dict):
            cleaned: Dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key)
                if key_text in omitted_keys or key_text.endswith(omitted_suffixes):
                    cleaned[key_text] = "<omitted>"
                else:
                    cleaned[key_text] = scrub(item)
            return cleaned
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return scrub(state_update)


def _node_logs_path(session: WebSession) -> str:
    return str(session.store.session_dir)


def _resolved_session_dir(session: WebSession) -> Path:
    session_dir = session.store.session_dir
    if session_dir.is_absolute():
        return session_dir.resolve()
    return (_REPO_ROOT / session_dir).resolve()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _preview_url(session_id: str, preview_path: Any) -> str:
    if isinstance(preview_path, dict):
        preview_path = preview_path.get("artifact_path") or preview_path.get("path") or ""
    preview_text = str(preview_path or "").strip()
    if not preview_text:
        return ""
    return f"/api/sessions/{session_id}/preview?path={quote(preview_text, safe='')}"


def _dedupe_preview_paths(preview_paths: list[Any]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for raw_path in preview_paths:
        if isinstance(raw_path, dict):
            raw_path = raw_path.get("artifact_path") or raw_path.get("path") or ""
        path_text = str(raw_path or "").strip()
        if not path_text or path_text in seen:
            continue
        seen.add(path_text)
        deduped.append(path_text)
    return deduped


def _session_logs_dir(session_id: str) -> Path:
    return (_REPO_ROOT / "logs" / "sessions" / session_id).resolve()


def _candidate_preview_dirs(session_id: str) -> list[Path]:
    session_dir = _session_logs_dir(session_id)
    return [
        (session_dir / "artifacts" / "find_candidates").resolve(),
        (session_dir / "find_candidates").resolve(),
        (_REPO_ROOT / "logs" / "find_candidates").resolve(),
    ]


def _relative_preview_path(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _parse_candidate_preview_path(path: Path, fallback_index: int = 0) -> dict[str, Any]:
    stem = path.stem
    parts = stem.split("__")
    index = fallback_index
    if len(parts) >= 3 and parts[0].isdigit():
        index = int(parts[0])
        instance_key = parts[1]
        camera_name = "__".join(parts[2:])
    elif len(parts) >= 2 and parts[0].isdigit():
        index = int(parts[0])
        instance_key = parts[1]
        camera_name = "__".join(parts[2:]) if len(parts) > 2 else ""
    elif len(parts) >= 2:
        instance_key = parts[0]
        camera_name = "__".join(parts[1:])
    else:
        instance_key = stem
        camera_name = ""
    return {
        "index": index,
        "instance_key": instance_key,
        "camera_name": camera_name,
        "exact_instance_file": "__" not in stem,
    }


def _candidate_file_sort_key(path: Path) -> tuple[int, float, str]:
    prefix = path.stem.split("__", 1)[0]
    return (int(prefix) if prefix.isdigit() else 9999, path.stat().st_mtime, path.name)


def _matching_candidate_preview_paths(session_id: str, instance_key: str) -> list[Path]:
    matches: list[Path] = []
    seen: set[Path] = set()
    for directory in _candidate_preview_dirs(session_id):
        if not directory.exists():
            continue
        for suffix in _ALLOWED_PREVIEW_SUFFIXES:
            exact = (directory / f"{instance_key}{suffix}").resolve()
            if exact.exists() and exact.is_file() and exact not in seen:
                seen.add(exact)
                matches.append(exact)
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in _ALLOWED_PREVIEW_SUFFIXES:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            if _parse_candidate_preview_path(path).get("instance_key") != instance_key:
                continue
            seen.add(resolved)
            matches.append(resolved)
    return sorted(matches, key=_candidate_file_sort_key)


def _candidate_preview_paths(detection: Dict[str, Any], session_id: str = "") -> list[str]:
    preview_paths: list[Any] = [detection.get("preview_path", ""), detection.get("preview_ref", "")]

    raw_preview_paths = detection.get("preview_paths", [])
    if isinstance(raw_preview_paths, (list, tuple)) and raw_preview_paths:
        preview_paths.extend(raw_preview_paths)
    elif raw_preview_paths:
        preview_paths.append(raw_preview_paths)

    raw_preview_refs = detection.get("preview_refs", [])
    if isinstance(raw_preview_refs, (list, tuple)) and raw_preview_refs:
        preview_paths.extend(raw_preview_refs)
    elif raw_preview_refs:
        preview_paths.append(raw_preview_refs)

    instance_key = str(detection.get("instance_key", "") or "").strip()
    if instance_key and session_id:
        for path in _matching_candidate_preview_paths(session_id, instance_key):
            preview_paths.append(_relative_preview_path(path))

    deduped = _dedupe_preview_paths(preview_paths)
    return deduped[:1]


def _find_candidate_image_items(session_id: str) -> list[Dict[str, Any]]:
    grouped: dict[str, tuple[Path, str, bool, float, int, int]] = {}
    for priority, directory in enumerate(_candidate_preview_dirs(session_id)):
        if not directory.exists():
            continue
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in _ALLOWED_PREVIEW_SUFFIXES:
                continue
            parsed = _parse_candidate_preview_path(path)
            instance_key = str(parsed.get("instance_key", "") or path.stem)
            camera_name = str(parsed.get("camera_name", "") or "")
            exact_instance_file = bool(parsed.get("exact_instance_file", False))
            index = int(parsed.get("index", 0) or 0)
            mtime = path.stat().st_mtime
            current = grouped.get(instance_key)
            if (
                current is None
                or priority < current[5]
                or (priority == current[5] and exact_instance_file and not current[2])
                or (priority == current[5] and exact_instance_file == current[2] and mtime > current[3])
            ):
                grouped[instance_key] = (path, camera_name, exact_instance_file, mtime, index, priority)

    items: list[Dict[str, Any]] = []
    for fallback_index, (instance_key, (path, camera_name, _exact_instance_file, mtime, index, _priority)) in enumerate(
        sorted(
            grouped.items(),
            key=lambda item: (item[1][4] if item[1][4] else 9999, -item[1][3], item[0]),
        ),
        start=1,
    ):
        relative_path = _relative_preview_path(path)
        stat = path.stat()
        items.append(
            {
                "id": index or fallback_index,
                "name": path.name,
                "path": relative_path,
                "preview_path": relative_path,
                "preview_paths": [relative_path],
                "preview_url": _preview_url(session_id, relative_path),
                "instance_key": instance_key,
                "label": instance_key,
                "camera": camera_name,
                "mtime": mtime,
                "size": stat.st_size,
            }
        )
    return items


def _resolve_preview_file(session: WebSession | None, raw_path: str, *, session_id: str = "") -> Path:
    preview_text = str(raw_path or "").strip()
    if not preview_text:
        raise HTTPException(status_code=404, detail="Preview path is empty.")

    input_path = Path(preview_text)
    logs_dir = (_REPO_ROOT / "logs").resolve()
    find_candidates_dir = (logs_dir / "find_candidates").resolve()
    session_dir = _resolved_session_dir(session) if session is not None else (_session_logs_dir(session_id) if session_id else None)
    allowed_roots = [find_candidates_dir]
    if session_dir is not None:
        allowed_roots.extend([
            (session_dir / "artifacts").resolve(),
            (session_dir / "find_candidates").resolve(),
        ])

    candidates: list[Path] = []
    if input_path.is_absolute():
        candidates.append(input_path.resolve())
    else:
        candidates.extend([
            (_REPO_ROOT / input_path).resolve(),
            (logs_dir / input_path).resolve(),
        ])
        if session_dir is not None:
            candidates.extend([
                (session_dir / input_path).resolve(),
                (session_dir / "artifacts" / input_path).resolve(),
                (session_dir / "artifacts" / "find_candidates" / input_path.name).resolve(),
                (session_dir / "find_candidates" / input_path.name).resolve(),
            ])
        if session_id:
            candidates.extend((directory / input_path.name).resolve() for directory in _candidate_preview_dirs(session_id))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.suffix.lower() not in _ALLOWED_PREVIEW_SUFFIXES:
            continue
        if not candidate.exists() or not candidate.is_file():
            continue
        if any(_is_relative_to(candidate, root) for root in allowed_roots):
            return candidate

    raise HTTPException(status_code=404, detail=f"Preview not found: {preview_text}")


def _attach_preview_urls_to_interrupt(
    session: WebSession,
    pending: Dict[str, Any],
) -> Dict[str, Any]:
    enriched = copy.deepcopy(pending)
    for interrupt_item in enriched.get("interrupts", []) or []:
        value = interrupt_item.get("value", {})
        if not isinstance(value, dict):
            continue
        for detection in value.get("detections", []) or []:
            if not isinstance(detection, dict):
                continue
            preview_paths = _candidate_preview_paths(detection, session.session_id)
            if preview_paths:
                detection["preview_paths"] = preview_paths
                detection["preview_path"] = detection.get("preview_path") or preview_paths[0]
                detection["preview_urls"] = [
                    url
                    for url in (_preview_url(session.session_id, path) for path in preview_paths)
                    if url
                ]
                detection["preview_url"] = detection["preview_urls"][0] if detection["preview_urls"] else ""
    return enriched


def _session_state_payload(session: WebSession) -> Dict[str, Any]:
    pending = session.pending_interrupt
    if pending:
        pending = _attach_preview_urls_to_interrupt(session, pending)
    return {
        "session_id": session.session_id,
        "state": _state_summary(session.store.current_state),
        "pending_interrupt": pending,
        "logs_path": _node_logs_path(session),
    }


def _publish_session_state(session: WebSession) -> None:
    if not session.state_subscribers:
        return

    payload = _session_state_payload(session)
    for queue in list(session.state_subscribers):
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass



def _interrupt_to_dict(item: Any) -> Dict[str, Any]:
    value = getattr(item, "value", item)
    interrupt_id = getattr(item, "id", getattr(item, "interrupt_id", ""))
    return {
        "id": interrupt_id() if callable(interrupt_id) else interrupt_id,
        "value": value,
    }



def _record_session_greeting(session: WebSession, state: CommanderState) -> str:
    message = "嗨～有什麼需要幫忙的嗎？"
    state_update = {
        "current_status": "GREETING_SENT",
        "greeting_sent": True,
        "last_execution": dump_model(
            NodeExecution(
                node_name="greeting_node",
                status="GREETING_SENT",
                success=True,
                message=message,
            )
        ),
    }
    session.step += 1
    session.store.record_event(
        step=session.step,
        node_name="greeting_node",
        state_update=state_update,
    )
    return message



def _prepare_message_state(session: WebSession, message: str) -> Dict[str, Any]:
    state = copy.deepcopy(session.store.current_state)
    state.update(
        {
            "interface_mode": "api",
            "human_reply": "",
            "task_intent": "",
            "selected_object_index": 0,
            "ai_reply": "",
            "current_status": "API_MESSAGE_RECEIVED",
            "task": {},
            "requested_object": {},
            "selected_instance": {},
            "world_position": {},
            "room_cameras": {},
            "item_info": {},
            "navigation": {},
            "observation": {},
            "decision": {},
            "module_params": {},
            "grasp_result": {},
            "approach_result": {},
            "last_execution": {},
            "task_complete": False,
            "retry_count": 0,
            "history_buffer": [],
            "session_summary": "",
        }
    )
    return state


def _legacy_prompt_type(prompt: str) -> str:
    prompt_text = str(prompt or "")
    if (
        "候選照片" in prompt_text
        or "候選目標" in prompt_text
        or "或 no" in prompt_text
        or "輸入 no" in prompt_text
        or "未找到" in prompt_text
    ):
        return "detection_selection"
    return "legacy_input"


def _legacy_detection_items(session_id: str) -> list[Dict[str, Any]]:
    now = time.time()
    recent_files: list[Path] = []
    seen: set[Path] = set()
    for directory in _candidate_preview_dirs(session_id):
        if not directory.exists():
            continue
        directory_files: list[Path] = []
        for path in directory.iterdir():
            resolved = path.resolve()
            if resolved in seen:
                continue
            if not path.is_file() or path.suffix.lower() not in _ALLOWED_PREVIEW_SUFFIXES:
                continue
            if now - path.stat().st_mtime <= 120.0:
                seen.add(resolved)
                directory_files.append(path)
        if directory_files:
            recent_files.extend(directory_files)
            break

    items: list[Dict[str, Any]] = []
    for fallback_index, path in enumerate(sorted(recent_files, key=_candidate_file_sort_key), start=1):
        parsed = _parse_candidate_preview_path(path, fallback_index=fallback_index)
        index = int(parsed.get("index", 0) or fallback_index)
        instance_key = str(parsed.get("instance_key", "") or path.stem)
        camera_name = str(parsed.get("camera_name", "") or "")
        relative_path = _relative_preview_path(path)
        items.append(
            {
                "id": index,
                "label": instance_key,
                "instance_key": instance_key,
                "camera": camera_name,
                "preview_path": relative_path,
                "preview_paths": [relative_path],
                "preview_url": _preview_url(session_id, relative_path),
            }
        )
    return items


def _legacy_pending_interrupt(session: WebSession, prompt: str) -> Dict[str, Any]:
    prompt_text = str(prompt or "").strip() or "請輸入回覆。"
    prompt_type = _legacy_prompt_type(prompt_text)
    display_message = (
        "請選擇正確的目標候選照片，或選擇 No valid target"
        if prompt_type == "detection_selection"
        else prompt_text
    )
    value: Dict[str, Any] = {
        "type": prompt_type,
        "message": display_message,
        "legacy_prompt": True,
        "legacy_prompt_text": prompt_text,
    }

    if prompt_type == "detection_selection":
        value["detections"] = _legacy_detection_items(session.session_id)

    return {
        "interrupts": [{"id": uuid.uuid4().hex, "value": value}],
        "type": prompt_type,
    }


def _legacy_input_answer(request: ResumeRequest, pending: Dict[str, Any]) -> str:
    if request.value is not None:
        return str(request.value)

    interrupt_value = (
        pending.get("interrupts", [{}])[0].get("value", {})
        if pending.get("interrupts")
        else {}
    )
    pending_type = (
        str(interrupt_value.get("type", "") or "")
        if isinstance(interrupt_value, dict)
        else str(pending.get("type", "") or "")
    )

    if request.selected_detection_id is not None:
        selected = int(request.selected_detection_id)
        if pending_type == "detection_selection" and selected == 0:
            return "no"
        return str(selected)
    return ""


async def _stream_graph(
    session: WebSession,
    graph_input: Any,
) -> AsyncIterator[str]:
    if session.lock is None:
        session.lock = asyncio.Lock()

    event_loop = asyncio.get_running_loop()
    original_input = builtins.input
    current_task = asyncio.current_task()
    if current_task is not None:
        session.active_stream_tasks.add(current_task)

    def bridged_input(prompt: str = "") -> str:
        if session.legacy_input_queue:
            return session.legacy_input_queue.pop(0)

        pending = _legacy_pending_interrupt(session, prompt)
        input_event = threading.Event()
        session.legacy_input_event = input_event
        session.legacy_input_value = ""
        session.pending_interrupt = _attach_preview_urls_to_interrupt(session, pending)
        event_loop.call_soon_threadsafe(_publish_session_state, session)
        input_event.wait()
        answer = session.legacy_input_value
        session.legacy_input_event = None
        session.legacy_input_value = ""
        session.pending_interrupt = None
        event_loop.call_soon_threadsafe(_publish_session_state, session)
        return answer

    builtins.input = bridged_input
    try:
        async with session.lock:
            try:
                async for event in session.orchestrator.graph.astream(
                    graph_input,
                    config=_thread_config(session),
                    stream_mode="updates",
                ):
                    if "__interrupt__" in event:
                        interrupts = [_interrupt_to_dict(item) for item in event["__interrupt__"]]
                        primary = interrupts[0]["value"] if interrupts else {}
                        pending = {
                            "interrupts": interrupts,
                            "type": primary.get("type", "interrupt") if isinstance(primary, dict) else "interrupt",
                        }
                        pending = _attach_preview_urls_to_interrupt(session, pending)
                        session.pending_interrupt = pending
                        session.step += 1
                        state_update = {
                            "current_status": "INTERRUPTED",
                            "awaiting_input_type": pending["type"],
                            "pending_interrupt": pending,
                            "node_execution_log": {
                                "trace_id": uuid.uuid4().hex,
                                "node_name": "__interrupt__",
                                "status": "INTERRUPTED",
                                "success": True,
                                "reasoning": "Graph interrupted for external user input.",
                                "extra_info": pending,
                            },
                        }
                        session.store.record_event(
                            step=session.step,
                            node_name="__interrupt__",
                            state_update=state_update,
                        )
                        _publish_session_state(session)
                        progress_message = _progress_message_from_interrupt(pending)
                        if progress_message:
                            yield _sse(
                                "progress_message",
                                {
                                    "session_id": session.session_id,
                                    "step": session.step,
                                    "node_name": "__interrupt__",
                                    "message": progress_message,
                                },
                            )
                        yield _sse(
                            "interrupt",
                            {
                                "session_id": session.session_id,
                                "step": session.step,
                                "interrupt": pending,
                                "logs_path": _node_logs_path(session),
                            },
                        )
                        return

                    for node_name, state_update in event.items():
                        if not isinstance(state_update, dict):
                            continue
                        session.step += 1
                        session.store.record_event(
                            step=session.step,
                            node_name=node_name,
                            state_update=state_update,
                        )
                        _publish_session_state(session)
                        yield _sse(
                            "node_update",
                            {
                                "session_id": session.session_id,
                                "step": session.step,
                                "node_name": node_name,
                                "status": state_update.get("current_status", ""),
                                "state_update": _event_state_update_summary(state_update),
                                "logs_path": _node_logs_path(session),
                            },
                        )

                        assistant_message = _assistant_message_from_update(node_name, state_update)
                        if assistant_message:
                            yield _sse(
                                "assistant_message",
                                {
                                    "session_id": session.session_id,
                                    "step": session.step,
                                    "node_name": node_name,
                                    "message": assistant_message,
                                },
                            )

                        progress_message = _progress_message_from_update(
                            node_name,
                            session.store.current_state,
                            state_update,
                        )
                        if progress_message:
                            yield _sse(
                                "progress_message",
                                {
                                    "session_id": session.session_id,
                                    "step": session.step,
                                    "node_name": node_name,
                                    "status": state_update.get("current_status", ""),
                                    "message": progress_message,
                                },
                            )

                        if node_name == "chat_memory_node":
                            session.pending_interrupt = None
                            _publish_session_state(session)
                            yield _sse(
                                "done",
                                {
                                    "session_id": session.session_id,
                                    "state": _state_summary(session.store.current_state),
                                    "logs_path": _node_logs_path(session),
                                },
                            )
                            return

                session.pending_interrupt = None
                _publish_session_state(session)
                yield _sse(
                    "done",
                    {
                        "session_id": session.session_id,
                        "state": _state_summary(session.store.current_state),
                        "logs_path": _node_logs_path(session),
                    },
                )
            except Exception as exc:
                logger.error("[web_server] Stream failed: %s", exc, exc_info=True)
                _publish_session_state(session)
                yield _sse(
                    "error",
                    {
                        "session_id": session.session_id,
                        "error": str(exc),
                        "logs_path": _node_logs_path(session),
                    },
                )
    finally:
        builtins.input = original_input
        if current_task is not None:
            session.active_stream_tasks.discard(current_task)


def _cleanup_logs_dir() -> None:
    """Delete the logs/ folder on web shutdown (normal exit or interrupt)."""
    logs_dir = (_REPO_ROOT / "logs").resolve()
    try:
        shutil.rmtree(logs_dir, ignore_errors=True)
        logger.info("[web_server] Removed logs directory on shutdown: %s", logs_dir)
    except Exception as exc:
        logger.warning("[web_server] Failed to remove logs directory %s: %s", logs_dir, exc)


def create_app() -> Any:
    if FastAPI is None:
        raise RuntimeError(
            "fastapi is not installed. Rebuild the Docker image or install workflow dependencies."
        )

    app = FastAPI(title="VLM 居家機器人 Chat", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (_STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/static/styles.css")
    async def styles() -> FileResponse:
        return FileResponse(_STATIC_DIR / "styles.css")

    @app.get("/static/app.js")
    async def script() -> FileResponse:
        return FileResponse(_STATIC_DIR / "app.js")

    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        return {"ok": True, "active_sessions": len(ACTIVE_SESSIONS)}

    @app.get("/api/objects")
    async def objects() -> Dict[str, Any]:
        return {"objects": load_graspable_objects()}

    @app.post("/api/sessions")
    async def create_session(request: CreateSessionRequest) -> Dict[str, Any]:
        session_id = uuid.uuid4().hex
        trace_logger = TraceLogger(log_file=_trace_log_file(request.log_file))
        orchestrator = await Orchestrator.create(trace_logger=trace_logger)
        initial_state = create_initial_state(session_id)
        initial_state.update(
            {
                "interface_mode": "api",
                "greeting_sent": True,
                "current_status": "INIT",
            }
        )
        store = SessionMemoryStore(context_id=session_id, initial_state=initial_state)
        session = WebSession(
            session_id=session_id,
            orchestrator=orchestrator,
            store=store,
            max_steps=request.max_steps,
        )
        greeting = _record_session_greeting(session, initial_state)
        ACTIVE_SESSIONS[session_id] = session
        return {
            "session_id": session_id,
            "greeting": greeting,
            "logs_path": _node_logs_path(session),
            "objects": load_graspable_objects(),
        }

    @app.get("/api/sessions/{session_id}/state")
    async def get_state(session_id: str) -> Dict[str, Any]:
        session = _get_session(session_id)
        return _session_state_payload(session)

    @app.get("/api/sessions/{session_id}/events")
    async def session_events(session_id: str) -> StreamingResponse:
        session = _get_session(session_id)

        async def event_stream() -> AsyncIterator[str]:
            queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=4)
            session.state_subscribers.add(queue)
            try:
                yield _sse("state", _session_state_payload(session))
                while True:
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=25.0)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    if payload.get("shutdown"):
                        break
                    yield _sse("state", payload)
            finally:
                session.state_subscribers.discard(queue)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/sessions/{session_id}/preview")
    async def preview(session_id: str, path: str) -> FileResponse:
        session = ACTIVE_SESSIONS.get(session_id)
        preview_file = _resolve_preview_file(session, path, session_id=session_id)
        return FileResponse(preview_file)

    @app.get("/api/sessions/{session_id}/find-candidates")
    async def find_candidates(session_id: str) -> Dict[str, Any]:
        _get_session(session_id)
        return {"items": _find_candidate_image_items(session_id)}

    @app.post("/api/sessions/{session_id}/messages/stream")
    async def stream_message(session_id: str, request: MessageRequest) -> StreamingResponse:
        session = _get_session(session_id)
        if session.pending_interrupt:
            async def error_stream() -> AsyncIterator[str]:
                yield _sse(
                    "error",
                    {
                        "session_id": session_id,
                        "error": "Session is waiting for resume input.",
                        "pending_interrupt": session.pending_interrupt,
                    },
                )

            return StreamingResponse(error_stream(), media_type="text/event-stream")

        session.legacy_input_queue.append(request.message.strip())
        graph_input = _prepare_message_state(session, request.message)
        return StreamingResponse(
            _stream_graph(session, graph_input),
            media_type="text/event-stream",
        )

    @app.post("/api/sessions/{session_id}/resume/stream")
    async def resume_stream(session_id: str, request: ResumeRequest) -> StreamingResponse:
        session = _get_session(session_id)
        if not session.pending_interrupt:
            async def error_stream() -> AsyncIterator[str]:
                yield _sse(
                    "error",
                    {
                        "session_id": session_id,
                        "error": "Session has no pending interrupt.",
                    },
                )

            return StreamingResponse(error_stream(), media_type="text/event-stream")

        if session.legacy_input_event is None:
            async def error_stream() -> AsyncIterator[str]:
                yield _sse(
                    "error",
                    {
                        "session_id": session_id,
                        "error": "Session is not waiting for legacy input.",
                    },
                )

            return StreamingResponse(error_stream(), media_type="text/event-stream")

        resume_value = _legacy_input_answer(request, session.pending_interrupt)
        session.pending_interrupt = None
        session.legacy_input_value = resume_value
        session.legacy_input_event.set()
        _publish_session_state(session)

        async def accepted_stream() -> AsyncIterator[str]:
            yield _sse(
                "input_accepted",
                {
                    "session_id": session_id,
                    "value": resume_value,
                },
            )

        return StreamingResponse(accepted_stream(), media_type="text/event-stream")

    @app.on_event("shutdown")
    async def shutdown() -> None:
        close_tasks = [
            _close_session_for_shutdown(session)
            for session in list(ACTIVE_SESSIONS.values())
        ]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        ACTIVE_SESSIONS.clear()
        if os.getenv("WEB_KEEP_LOGS", "0").strip().lower() not in {"1", "true", "yes", "on"}:
            _cleanup_logs_dir()

    return app


async def _close_session_for_shutdown(session: WebSession) -> None:
    if session.legacy_input_event is not None:
        session.legacy_input_value = "bye"
        session.legacy_input_event.set()
    session.pending_interrupt = None

    current_task = asyncio.current_task()
    stream_tasks = [
        task
        for task in list(session.active_stream_tasks)
        if task is not current_task and not task.done()
    ]
    for task in stream_tasks:
        task.cancel()

    for queue in list(session.state_subscribers):
        try:
            queue.put_nowait({"shutdown": True})
        except asyncio.QueueFull:
            pass

    if stream_tasks:
        done, pending = await asyncio.wait(
            stream_tasks,
            timeout=_SHUTDOWN_STREAM_CANCEL_TIMEOUT_SEC,
        )
        for task in pending:
            logger.warning(
                "[web_server] Stream task for session %s did not cancel within %.2fs.",
                session.session_id,
                _SHUTDOWN_STREAM_CANCEL_TIMEOUT_SEC,
            )
        for task in done:
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.warning(
                    "[web_server] Stream task for session %s ended during shutdown: %s",
                    session.session_id,
                    exc,
                )

    try:
        await asyncio.wait_for(
            session.orchestrator.aclose(),
            timeout=_SHUTDOWN_SESSION_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "[web_server] Timed out closing session %s during shutdown.",
            session.session_id,
        )
    except Exception as exc:
        logger.warning(
            "[web_server] Failed to close session %s during shutdown: %s",
            session.session_id,
            exc,
        )


def _get_session(session_id: str) -> WebSession:
    session = ACTIVE_SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return session


if FastAPI is not None:
    app = create_app()
else:
    async def app(scope: Dict[str, Any], receive: Any, send: Any) -> None:
        raise RuntimeError(
            "fastapi is not installed. Rebuild the Docker image or install workflow dependencies."
        )
