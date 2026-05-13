"""Repo-local FastAPI + SSE chat interface for the Commander LangGraph."""

from __future__ import annotations

import asyncio
import builtins
import copy
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict
from urllib.parse import quote

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from starlette.responses import FileResponse, HTMLResponse, StreamingResponse

try:
    from fastapi import FastAPI, HTTPException
except ModuleNotFoundError:  # pragma: no cover - lets py_compile work before uv sync.
    FastAPI = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]

from .logger import TraceLogger
from .orchestrator import Orchestrator, _load_graspable_objects
from .session_store import SessionMemoryStore
from .state import CommanderState, create_initial_state


load_dotenv(override=True)
logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ALLOWED_PREVIEW_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


class CreateSessionRequest(BaseModel):
    mock: bool | None = Field(default=None)
    max_steps: int = Field(default=60, ge=1, le=500)
    log_file: str | None = Field(default=None)


class MessageRequest(BaseModel):
    message: str = Field(min_length=1)


class ResumeRequest(BaseModel):
    selected_detection_id: int | None = None
    selected_object_index: int | None = None
    value: Any | None = None


@dataclass
class WebSession:
    session_id: str
    orchestrator: Orchestrator
    store: SessionMemoryStore
    max_steps: int
    use_mock: bool
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


def _resolve_mock_mode(requested: bool | None) -> bool:
    if requested is not None:
        return requested
    return os.getenv("MOCK_MODE", "true").lower() == "true"


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
    return {
        "current_status": state.get("current_status", ""),
        "task_intent": state.get("task_intent", ""),
        "selected_object_index": state.get("selected_object_index", 0),
        "task_description": state.get("task_description", ""),
        "target_object": state.get("target_object", {}),
        "call_module": state.get("call_module", ""),
        "retry_count": state.get("retry_count", 0),
        "task_complete": state.get("task_complete", False),
        "awaiting_input_type": state.get("awaiting_input_type", ""),
        "pending_interrupt": state.get("pending_interrupt", {}),
        "chat_history_buffer": state.get("chat_history_buffer", []),
    }


def _event_state_update_summary(state_update: Dict[str, Any]) -> Dict[str, Any]:
    omitted = {"camera_images", "image_base64", "world_position_db"}
    summary: Dict[str, Any] = {}
    for key, value in state_update.items():
        if key in omitted:
            summary[key] = "<omitted>"
        elif key == "selected_target" and isinstance(value, dict):
            summary[key] = {
                item_key: item_value
                for item_key, item_value in value.items()
                if item_key not in {"camera_images", "world_position_data"}
            }
        else:
            summary[key] = value
    return summary


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
        preview_path = preview_path.get("artifact_path", "")
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


def _candidate_preview_paths(detection: Dict[str, Any]) -> list[str]:
    preview_paths: list[Any] = [detection.get("preview_path", "")]

    raw_preview_paths = detection.get("preview_paths", [])
    if isinstance(raw_preview_paths, (list, tuple)) and raw_preview_paths:
        preview_paths.append(raw_preview_paths[0])
    elif raw_preview_paths:
        preview_paths.append(raw_preview_paths)

    instance_key = str(detection.get("instance_key", "") or "").strip()
    if instance_key:
        find_candidates_dir = (_REPO_ROOT / "logs" / "find_candidates").resolve()
        for suffix in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            candidate = find_candidates_dir / f"{instance_key}{suffix}"
            if not candidate.exists() or not candidate.is_file():
                continue
            try:
                preview_paths.append(str(candidate.relative_to(_REPO_ROOT)))
            except ValueError:
                preview_paths.append(str(candidate))
            break

    deduped = _dedupe_preview_paths(preview_paths)
    return deduped[:1]


def _find_candidate_image_items(session_id: str) -> list[Dict[str, Any]]:
    find_candidates_dir = (_REPO_ROOT / "logs" / "find_candidates").resolve()
    if not find_candidates_dir.exists():
        return []

    grouped: dict[str, tuple[Path, str, bool, float]] = {}
    for path in find_candidates_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in _ALLOWED_PREVIEW_SUFFIXES:
            continue
        stem = path.stem
        exact_instance_file = "__" not in stem
        instance_key, camera_name = (stem.split("__", 1) + [""])[:2] if "__" in stem else (stem, "")
        mtime = path.stat().st_mtime
        current = grouped.get(instance_key)
        if (
            current is None
            or (exact_instance_file and not current[2])
            or (exact_instance_file == current[2] and mtime > current[3])
        ):
            grouped[instance_key] = (path, camera_name, exact_instance_file, mtime)

    items: list[Dict[str, Any]] = []
    for instance_key, (path, camera_name, _exact_instance_file, mtime) in sorted(
        grouped.items(),
        key=lambda item: item[1][3],
        reverse=True,
    ):
        try:
            relative_path = str(path.relative_to(_REPO_ROOT))
        except ValueError:
            relative_path = str(path)
        stat = path.stat()
        items.append(
            {
                "name": path.name,
                "path": relative_path,
                "preview_url": _preview_url(session_id, relative_path),
                "instance_key": instance_key,
                "camera": camera_name,
                "mtime": mtime,
                "size": stat.st_size,
            }
        )
    return items


def _resolve_preview_file(session: WebSession | None, raw_path: str) -> Path:
    preview_text = str(raw_path or "").strip()
    if not preview_text:
        raise HTTPException(status_code=404, detail="Preview path is empty.")

    input_path = Path(preview_text)
    logs_dir = (_REPO_ROOT / "logs").resolve()
    find_candidates_dir = (logs_dir / "find_candidates").resolve()
    session_dir = _resolved_session_dir(session) if session is not None else None
    allowed_roots = [find_candidates_dir]
    if session_dir is not None:
        allowed_roots.append(session_dir)

    candidates: list[Path] = []
    if input_path.is_absolute():
        candidates.append(input_path.resolve())
    else:
        candidates.extend([
            (_REPO_ROOT / input_path).resolve(),
            (logs_dir / input_path).resolve(),
        ])
        if session_dir is not None:
            candidates.append((session_dir / input_path).resolve())

    for candidate in candidates:
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
            preview_paths = _candidate_preview_paths(detection)
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


def _assistant_message_from_update(node_name: str, state_update: Dict[str, Any]) -> str:
    if node_name == "greeting_node" and state_update.get("current_status") == "GREETING_SENT":
        return (
            state_update.get("node_execution_log", {})
            .get("extra_info", {})
            .get("message", "")
        )
    if node_name == "ai_reply_node":
        return str(state_update.get("ai_reply", "") or "")
    if node_name == "goodbye_node":
        return (
            state_update.get("node_execution_log", {})
            .get("extra_info", {})
            .get("message", "")
        )
    if node_name == "input_node" and state_update.get("task_description"):
        return f"任務已確認：{state_update['task_description']}"
    return ""


def _progress_message_from_update(
    node_name: str,
    merged_state: Dict[str, Any],
    state_update: Dict[str, Any],
) -> str:
    status = str(state_update.get("current_status", "") or "")
    if not status or status in {"GREETING_SENT", "GREETING_SKIPPED", "AI_REPLY_SENT", "GOODBYE_SENT"}:
        return ""

    if node_name == "human_reply_node":
        return "已收到你的訊息。接下來會判斷這是一般聊天，還是需要機器人執行的抓取任務。"

    if node_name == "task_classification_node":
        intent = str(state_update.get("task_intent", "") or "")
        if intent == "general_chat":
            return "已判斷為一般聊天。接下來產生聊天回覆。"
        try:
            selected = int(state_update.get("selected_object_index", 0) or 0)
        except (TypeError, ValueError):
            selected = 0
        selected_text = f"目標編號 {selected}" if selected else "目標物尚未完全確認"
        return f"已判斷為機器人抓取任務，{selected_text}。接下來確認任務目標。"

    if node_name == "chat_memory_node":
        return "已更新聊天記憶。這一輪一般聊天已完成。"

    if node_name == "input_node":
        if status == "INPUT_RECEIVED":
            task = state_update.get("task_description", merged_state.get("task_description", ""))
            return f"已確認任務：{task}。接下來搜尋偵測到的候選目標 instance。"
        return "尚未確認合法的任務目標。接下來會結束這輪任務。"

    if node_name == "find_node":
        if status == "TARGET_SELECTED_FROM_WORLD_POSITION":
            target = state_update.get("selected_target", {}) or {}
            name = target.get("instance_key") or target.get("label") or "目標 instance"
            return f"已選定候選目標：{name}。接下來整理目標的 3D 資訊與導航候選位置。"
        return "沒有找到可用的目標 instance。接下來返回結束流程。"

    if node_name == "get_item_info_no_sam3d_node":
        if status == "ITEM_INFO_NO_SAM3D_READY":
            return "已取得目標 3D 資訊與導航候選位置。接下來開始導航到目標附近。"
        return "目標 3D 資訊整理失敗。接下來返回 home 並結束任務。"

    if node_name == "nav_move_node":
        if status == "NAV_COMPLETED":
            return "已完成導航移動。接下來觀察目前環境，確認下一步動作。"
        return "導航移動沒有成功完成。接下來記錄結果並重新評估或返回 home。"

    if node_name == "observe_node":
        return "已取得目前環境觀察。接下來由模型推理下一步行動。"

    if node_name == "reason_node":
        module = str(state_update.get("call_module", "") or "")
        if module == "DONE":
            return "已完成推理，任務達成結束條件。接下來返回 home。"
        if module in {"nav_agent", "major_nav_agent", "major_nav_node"}:
            return "已完成推理，決定調整導航位置。接下來準備下一個導航候選點。"
        if module in {"grasp_agent", "approach_agent", "car_approach_agent"}:
            return "已完成推理，決定進入抓取/靠近流程。接下來更新目標資訊。"
        return "已完成推理。接下來依照模型決策執行下一個節點。"

    if node_name == "update_item_info_1_node":
        if merged_state.get("world_position_target_changed", False):
            return "已更新目標位置，且偵測到目標位置改變。接下來重新整理 3D 資訊。"
        if merged_state.get("world_position_update_reason", "") == "target_missing":
            return "已更新目標位置，但目標消失。接下來返回 home。"
        return "已更新目標位置資訊。接下來準備下一個導航候選點。"

    if node_name == "major_nav_node":
        if status == "MAJOR_NAV_CONTEXT_READY":
            rank = merged_state.get("current_goal_rank", "")
            return f"已準備第 {rank} 組導航候選點。接下來執行導航移動。"
        return "已沒有更多導航候選點。接下來返回 home 並結束任務。"

    if node_name == "update_item_info_2_node":
        if merged_state.get("world_position_target_changed", False):
            return "已更新目標位置，且偵測到目標位置改變。接下來重新整理 3D 資訊。"
        if merged_state.get("world_position_update_reason", "") == "target_missing":
            return "已更新目標位置，但目標消失。接下來返回 home。"
        return "已更新目標位置資訊。接下來執行抓取與靠近流程。"

    if node_name == "car_grasp_node":
        return "已完成抓取規劃。接下來依照抓取結果控制車體與手臂靠近。"

    if node_name == "car_approach_node":
        return "已完成靠近/執行動作。接下來更新任務記憶並重新觀察環境。"

    if node_name == "update_memory_node":
        return "已更新任務記憶。接下來重新觀察環境，確認任務是否完成。"

    if node_name == "nav_home_node":
        if status == "NAV_HOME_COMPLETED":
            return "已完成返回 home。接下來結束對話與任務。"
        return "返回 home 沒有成功完成。接下來仍會進入結束流程。"

    return ""


def _progress_message_from_interrupt(pending: Dict[str, Any]) -> str:
    interrupt_type = str(pending.get("type", "") or "")
    if interrupt_type == "detection_selection":
        return "已找到候選目標 instance。接下來需要你在聊天卡片中選擇正確目標，或選擇 No valid target。"
    if interrupt_type == "object_selection":
        return "目前還沒有確認要抓取的物品。接下來需要你在聊天卡片中選擇目標物。"
    return "流程暫停等待你的輸入。接下來請在聊天卡片中完成選擇。"


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
        "node_execution_log": session.orchestrator._node_execution_log(
            state,
            "greeting_node",
            "GREETING_SENT",
            time.time(),
            reasoning="Greeting displayed to the web user during session creation.",
            extra_info={"message": message, "interface_mode": "api"},
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
            "awaiting_input_type": "",
            "pending_interrupt": {},
            "current_status": "API_MESSAGE_RECEIVED",
            "task_description": "",
            "target_object": {},
            "candidate_objects": [],
            "selected_target": {},
            "yolo_detections": {},
            "selected_detection_id": 0,
            "find_complete": False,
            "call_module": "",
            "module_params": {},
            "reasoning": "",
            "agent_result": "",
            "agent_success": False,
            "task_complete": False,
            "retry_count": 0,
            "history_buffer": [],
        }
    )
    return state


def _legacy_prompt_type(prompt: str) -> str:
    prompt_text = str(prompt or "")
    if "輸入 no" in prompt_text or "未找到" in prompt_text:
        return "detection_selection"
    if "目標物編號" in prompt_text:
        return "object_selection"
    return "legacy_input"


def _legacy_detection_items(session_id: str) -> list[Dict[str, Any]]:
    find_candidates_dir = (_REPO_ROOT / "logs" / "find_candidates").resolve()
    if not find_candidates_dir.exists():
        return []

    now = time.time()
    recent_files: list[Path] = []
    for path in find_candidates_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in _ALLOWED_PREVIEW_SUFFIXES:
            continue
        if now - path.stat().st_mtime <= 120.0:
            recent_files.append(path)

    items: list[Dict[str, Any]] = []
    for index, path in enumerate(sorted(recent_files, key=lambda item: item.stat().st_mtime), start=1):
        stem = path.stem
        instance_key = stem.split("__", 1)[0]
        camera_name = stem.split("__", 1)[1] if "__" in stem else ""
        try:
            relative_path = str(path.relative_to(_REPO_ROOT))
        except ValueError:
            relative_path = str(path)
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
        "請於左側欄位選擇目標物"
        if prompt_type == "detection_selection"
        else prompt_text
    )
    value: Dict[str, Any] = {
        "type": prompt_type,
        "message": display_message,
        "legacy_prompt": True,
        "legacy_prompt_text": prompt_text,
    }

    if prompt_type == "object_selection":
        value["objects"] = [
            {
                "index": index,
                "id": obj.get("id", ""),
                "label": obj.get("label", obj.get("id", "")),
            }
            for index, obj in enumerate(_load_graspable_objects(), start=1)
        ]
    elif prompt_type == "detection_selection":
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
    if request.selected_object_index is not None:
        return str(int(request.selected_object_index))
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


def create_app() -> Any:
    if FastAPI is None:
        raise RuntimeError("fastapi is not installed. Run `uv sync` or install project dependencies.")

    app = FastAPI(title="VLM 居家機器人 Chat", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _INDEX_HTML

    @app.get("/api/health")
    async def health() -> Dict[str, Any]:
        return {"ok": True, "active_sessions": len(ACTIVE_SESSIONS)}

    @app.get("/api/objects")
    async def objects() -> Dict[str, Any]:
        return {"objects": _load_graspable_objects()}

    @app.post("/api/sessions")
    async def create_session(request: CreateSessionRequest) -> Dict[str, Any]:
        session_id = uuid.uuid4().hex
        use_mock = _resolve_mock_mode(request.mock)
        trace_logger = TraceLogger(log_file=_trace_log_file(request.log_file))
        orchestrator = Orchestrator(trace_logger=trace_logger, use_mock=use_mock)
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
            use_mock=use_mock,
        )
        greeting = _record_session_greeting(session, initial_state)
        ACTIVE_SESSIONS[session_id] = session
        return {
            "session_id": session_id,
            "greeting": greeting,
            "mock": use_mock,
            "logs_path": _node_logs_path(session),
            "objects": _load_graspable_objects(),
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
        preview_file = _resolve_preview_file(session, path)
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


_INDEX_HTML = r"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>VLM 居家機器人</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --line: #dfe7ef;
      --ink: #24313f;
      --muted: #667085;
      --accent: #0f766e;
      --accent-hover: #0d9488;
      --accent-ink: #ffffff;
      --user-bg: #e0f2fe;
      --ai-bg: #ffffff;
      --loading-dot: #0f766e;
      --progress-bg: #ecfeff;
      --system-bg: #f8fafc;
      --choice-bg: #ffffff;
      --preview-bg: #eef7f6;
      --shadow: 0 6px 20px rgba(15, 23, 42, .06);
      --warm-shadow: 0 4px 12px rgba(15, 118, 110, .25);
      --font-main: "PingFang TC", "Microsoft JhengHei", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--font-main);
      background: var(--bg);
      color: var(--ink);
      height: 100vh;
      overflow: hidden;
    }
    header {
      height: 70px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 30px;
      background: var(--panel);
      border-bottom: 2px solid var(--line);
      box-shadow: var(--shadow);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 15px;
      min-width: 0;
    }
    .brand-mark {
      width: 36px;
      height: 36px;
      border-radius: 12px;
      background: var(--accent);
      color: var(--accent-ink);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 20px;
      flex: none;
      box-shadow: var(--warm-shadow);
    }
    .brand strong {
      white-space: nowrap;
      font-size: 1.3rem;
      letter-spacing: 0;
      color: var(--ink);
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
      min-width: 0;
    }
    .status-tag {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      max-width: 380px;
      padding: 5px 12px;
      border-radius: 15px;
      border: 1px solid #bfe8e2;
      background: #ecfeff;
      color: var(--accent);
      font-size: 14px;
    }
    .status-tag::before {
      content: "";
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, .14);
      flex: none;
    }
    #session {
      color: inherit;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    main {
      height: calc(100vh - 70px);
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      gap: 0;
    }
    section {
      background: var(--panel);
      min-height: 0;
    }
    .chat {
      background: var(--bg);
      border-right: 2px solid var(--line);
      height: 100%;
      min-height: 0;
      overflow: hidden;
    }
    .chat-workspace {
      height: 100%;
      min-height: 0;
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
    }
    .conversation {
      min-width: 0;
      min-height: 0;
      display: flex;
      flex-direction: column;
    }
    .candidate-menu[hidden] {
      display: none;
    }
    .candidate-menu[hidden] + .conversation {
      grid-column: 1 / -1;
    }
    .candidate-menu {
      min-width: 0;
      min-height: 0;
      padding: 18px;
      overflow-y: auto;
      background: var(--panel);
      border-right: 2px solid var(--line);
    }
    .candidate-menu h3 {
      margin: 0 0 14px;
      padding-bottom: 10px;
      border-bottom: 2px solid var(--line);
      font-size: 15px;
      color: var(--ink);
      letter-spacing: 0;
    }
    .candidate-list {
      display: grid;
      gap: 10px;
    }
    .messages {
      flex: 1;
      padding: 40px 56px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 24px;
      scroll-behavior: smooth;
    }
    .msg {
      max-width: 78%;
      padding: 18px 24px;
      border-radius: 25px;
      font-size: 16px;
      line-height: 1.65;
      white-space: pre-wrap;
      box-shadow: var(--shadow);
      animation: msgIn .4s cubic-bezier(.175, .885, .32, 1.275) both;
    }
    @keyframes msgIn {
      from { opacity: 0; transform: translateY(15px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .user {
      align-self: flex-end;
      background: var(--user-bg);
      border-bottom-right-radius: 8px;
      color: #075985;
    }
    .assistant {
      align-self: flex-start;
      background: var(--ai-bg);
      border: 1px solid var(--line);
      border-bottom-left-radius: 8px;
    }
    .progress {
      align-self: center;
      background: var(--progress-bg);
      border: 1px solid #bae6fd;
      color: var(--muted);
      font-size: 14px;
      max-width: min(86%, 760px);
      padding: 8px 18px;
      border-radius: 20px;
      margin: 12px 0;
      box-shadow: none;
    }
    .system {
      align-self: center;
      background: var(--system-bg);
      color: var(--muted);
      border: 1px solid var(--line);
      font-size: 13px;
      max-width: min(86%, 760px);
      padding: 10px 18px;
      border-radius: 999px;
      box-shadow: none;
    }
    .typing-indicator {
      align-self: flex-start;
      display: flex;
      gap: 7px;
      width: fit-content;
      padding: 18px 24px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 25px;
      margin-bottom: 12px;
      box-shadow: var(--shadow);
      animation: msgIn .4s cubic-bezier(.175, .885, .32, 1.275) both;
    }
    .typing-indicator .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--loading-dot);
      opacity: .7;
      animation: dotBounce 1.4s infinite ease-in-out;
    }
    .typing-indicator .dot:nth-child(2) { animation-delay: .2s; }
    .typing-indicator .dot:nth-child(3) { animation-delay: .4s; }
    @keyframes dotBounce {
      0%, 80%, 100% { transform: scale(.8); opacity: .5; }
      40% { transform: scale(1.2); opacity: 1; }
    }
    .input-container {
      padding: 28px 56px;
      background: var(--panel);
      border-top: 2px solid var(--line);
    }
    form {
      display: flex;
      gap: 15px;
      max-width: none;
      margin: 0 auto;
    }
    input, button {
      font: inherit;
    }
    input {
      flex: 1;
      min-width: 0;
      background: var(--bg);
      border: 2px solid var(--line);
      color: var(--ink);
      padding: 18px 28px;
      border-radius: 35px;
      font-size: 16px;
      outline: none;
      transition: all .3s;
    }
    input::placeholder {
      color: var(--muted);
    }
    input:focus {
      border-color: var(--accent);
      background: #ffffff;
      box-shadow: 0 0 12px rgba(15, 118, 110, .16);
    }
    button {
      background: var(--accent);
      color: var(--accent-ink);
      border: none;
      border-radius: 35px;
      min-width: 60px;
      padding: 0 20px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      font-weight: 600;
      font-size: 16px;
      cursor: pointer;
      transition: all .2s;
      box-shadow: none;
    }
    .send-icon {
      width: 20px;
      height: 20px;
      flex: none;
    }
    button:hover {
      background: var(--accent-hover);
      transform: scale(1.05);
    }
    button:active { transform: scale(.95); }
    button:disabled {
      background: #94a3b8;
      color: #f8fafc;
      opacity: 1;
      cursor: not-allowed;
      transform: none;
      filter: grayscale(.18);
    }
    button:disabled:hover {
      background: #94a3b8;
      transform: none;
    }
    aside {
      background: var(--panel);
      padding: 30px;
      overflow-y: auto;
    }
    .panel {
      padding: 0;
      margin-bottom: 18px;
      border-bottom: 0;
    }
    aside h3 {
      margin: 0 0 25px;
      padding-bottom: 10px;
      border-bottom: 2px solid var(--line);
      font-size: 16px;
      color: var(--ink);
      letter-spacing: 0;
      text-align: center;
    }
    #task {
      padding: 15px;
      min-height: 52px;
      border-radius: 15px;
      background: var(--bg);
      border: 2px solid transparent;
      color: #0f766e;
      font-size: 14px;
      line-height: 1.5;
      font-weight: 700;
      box-shadow: var(--shadow);
    }
    .timeline {
      overflow: auto;
      padding: 0;
    }
    .node {
      position: relative;
      padding: 15px;
      font-size: 14px;
      background: var(--bg);
      border: 2px solid transparent;
      border-radius: 15px;
      margin-bottom: 15px;
      color: var(--muted);
      box-shadow: none;
      transition: all .3s;
    }
    .node:last-child {
      border-color: var(--accent);
      background: #ecfeff;
      color: #0f766e;
      font-weight: 700;
      box-shadow: var(--shadow);
    }
    .node strong {
      display: block;
      margin-bottom: 4px;
      color: inherit;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .node span {
      color: var(--muted);
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .interrupt-message {
      width: min(980px, 100%);
      max-width: min(94%, 980px);
      white-space: normal;
      padding: 0;
      overflow: hidden;
    }
    .interrupt-card {
      padding: 18px;
      opacity: 1;
      transform: translateY(0);
      transition: opacity .18s ease, transform .18s ease;
    }
    .interrupt-card.closing {
      opacity: 0;
      transform: translateY(8px);
      pointer-events: none;
    }
    .interrupt-title {
      font-size: 15px;
      line-height: 1.45;
      margin-bottom: 12px;
      color: var(--ink);
    }
    .choice-grid {
      display: grid;
      gap: 10px;
    }
    .empty-choice-note {
      padding: 13px 15px;
      background: var(--progress-bg);
      border: 1px solid #bae6fd;
      border-radius: 15px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }
    .manual-input {
      display: flex;
      gap: 10px;
      margin-bottom: 10px;
    }
    .manual-input input {
      padding: 12px 14px;
      border-radius: 15px;
      font-size: 14px;
    }
    .manual-input button {
      min-height: 46px;
      border-radius: 15px;
      flex: none;
    }
    .choice {
      width: 100%;
      text-align: left;
      margin-top: 0;
      background: var(--choice-bg);
      color: var(--ink);
      border: 2px solid var(--line);
      border-radius: 15px;
      padding: 13px 15px;
      box-shadow: none;
    }
    .choice:hover {
      background: #ffffff;
      border-color: var(--accent);
      transform: translateY(-1px);
      box-shadow: 0 8px 18px rgba(15, 23, 42, .08);
    }
    .choice:active { transform: scale(.985); }
    .choice.selected {
      background: #ccfbf1;
      border-color: var(--accent);
      transform: scale(.985);
    }
    .choice.faded { opacity: .35; }
    .choice:disabled { cursor: default; }
    .detection-card {
      display: grid;
      grid-template-columns: minmax(220px, 38%) minmax(0, 1fr);
      gap: 14px;
      align-items: stretch;
      padding: 10px;
      background: #f8fbff;
    }
    .preview-gallery {
      min-width: 0;
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }
    .preview-gallery.multi {
      grid-template-columns: repeat(auto-fit, minmax(128px, 1fr));
      max-height: 260px;
      overflow-y: auto;
      padding-right: 2px;
    }
    .preview {
      min-width: 0;
      min-height: 118px;
      aspect-ratio: 16 / 9;
      border: 2px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: var(--preview-bg);
      color: var(--muted);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
      text-align: center;
    }
    .preview img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      background: #ffffff;
      display: block;
    }
    .candidate-menu .interrupt-card {
      padding: 0;
    }
    .candidate-menu .interrupt-title {
      margin-bottom: 10px;
      font-size: 13px;
      font-weight: 700;
    }
    .candidate-menu .choice-grid {
      gap: 8px;
    }
    .candidate-menu .detection-card {
      grid-template-columns: 1fr;
      padding: 8px;
    }
    .candidate-menu .preview-gallery.multi {
      grid-template-columns: 1fr;
      max-height: none;
    }
    .candidate-menu .preview {
      min-height: 130px;
    }
    .choice-title {
      font-weight: 700;
      margin-bottom: 6px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .choice-meta {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }
    code { color: var(--muted); }
    @media (max-width: 1100px) {
      main { grid-template-columns: minmax(360px, 1fr) 310px; }
      .chat-workspace { grid-template-columns: 260px minmax(0, 1fr); }
      .messages { padding: 28px; }
      aside { padding: 24px 22px; }
      .detection-card { grid-template-columns: minmax(190px, 40%) minmax(0, 1fr); }
    }
    @media (max-width: 880px) {
      body { overflow: auto; }
      main { grid-template-columns: 1fr; height: auto; min-height: calc(100vh - 70px); }
      .chat { min-height: calc(100vh - 70px); border-right: 0; }
      .chat-workspace { grid-template-columns: 1fr; }
      .candidate-menu { max-height: 340px; border-right: 0; border-bottom: 2px solid var(--line); }
      aside { max-height: 420px; border-top: 2px solid var(--line); }
      .detection-card { grid-template-columns: minmax(170px, 40%) minmax(0, 1fr); }
      header { height: auto; min-height: 70px; align-items: flex-start; padding: 12px 16px; }
      .header-actions { flex-wrap: wrap; justify-content: flex-end; }
      .messages { padding: 24px 16px; }
      .msg { max-width: 92%; }
      .input-container { padding: 16px; }
      #session { max-width: 180px; }
    }
    @media (max-width: 560px) {
      header { flex-direction: column; align-items: stretch; }
      .header-actions { justify-content: flex-start; }
      .brand strong { font-size: 1.05rem; white-space: normal; }
      form { flex-direction: column; }
      button { min-height: 50px; }
      .detection-card { grid-template-columns: 1fr; }
      .preview { min-height: 160px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span class="brand-mark">H</span>
      <strong>VLM 居家機器人</strong>
    </div>
    <div class="header-actions">
      <span class="status-tag">系統運行中 <code id="session">starting...</code></span>
    </div>
  </header>
  <main>
    <section class="chat">
      <div class="chat-workspace">
        <div id="candidate-menu" class="candidate-menu" hidden>
          <h3>候選照片</h3>
          <div id="candidate-list" class="candidate-list"></div>
        </div>
        <div class="conversation">
          <div id="messages" class="messages"></div>
          <div class="input-container">
            <form id="chat-form">
              <input id="message-input" autocomplete="off" placeholder="對居家機器人下達指令..." />
              <button id="send-button" type="submit" aria-label="送出" title="送出">
                <svg class="send-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M4 12L20 4L16 20L12 13L4 12Z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
                </svg>
              </button>
            </form>
          </div>
        </div>
      </div>
    </section>
	    <aside>
	      <section class="panel">
	        <h3>任務流程監控</h3>
	        <div id="task">準備就緒，等待任務指令。</div>
	      </section>
	      <section class="timeline" id="timeline"></section>
	    </aside>
  </main>
  <script>
    let sessionId = "";
    let activeInterruptCard = null;
    let activeInterruptSignature = "";
    let currentInterruptValue = null;
    let candidateMenuSuppressed = false;
    let streamInFlight = false;
    let loadingIndicator = null;
    let stateEvents = null;
    const candidateMenu = document.getElementById("candidate-menu");
    const candidateList = document.getElementById("candidate-list");
    const messages = document.getElementById("messages");
    const timeline = document.getElementById("timeline");
    const timelinePanel = timeline.closest("aside");
    const task = document.getElementById("task");
    const form = document.getElementById("chat-form");
    const input = document.getElementById("message-input");
    const sendButton = document.getElementById("send-button");

    function showLoading() {
      if (loadingIndicator) return;
      loadingIndicator = document.createElement("div");
      loadingIndicator.className = "typing-indicator";
      loadingIndicator.setAttribute("aria-label", "等待回覆中");
      loadingIndicator.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
      messages.appendChild(loadingIndicator);
      messages.scrollTop = messages.scrollHeight;
    }

    function hideLoading() {
      if (!loadingIndicator) return;
      loadingIndicator.remove();
      loadingIndicator = null;
    }

    function addMessage(kind, text) {
      hideLoading();
      const div = document.createElement("div");
      div.className = `msg ${kind}`;
      div.textContent = text;
      messages.appendChild(div);
      messages.scrollTop = messages.scrollHeight;
    }

    function scrollTimelineToBottom() {
      requestAnimationFrame(() => {
        timeline.scrollTop = timeline.scrollHeight;
        if (timelinePanel) timelinePanel.scrollTop = timelinePanel.scrollHeight;
      });
    }

    function addNode(event) {
      const div = document.createElement("div");
      div.className = "node";
      div.innerHTML = `<strong>${event.step}. ${event.node_name}</strong><span>${event.status || ""}</span>`;
      timeline.appendChild(div);
      scrollTimelineToBottom();
      const update = event.state_update || {};
      if (event.node_name === "task_classification_node") {
        task.textContent = `intent=${update.task_intent || ""}, selected=${update.selected_object_index || 0}`;
      }
      if (event.node_name === "input_node" && update.task_description) {
        task.textContent = update.task_description;
      }
    }

    function updateSendButtonState() {
      sendButton.disabled = streamInFlight || currentInterruptValue !== null;
    }

    function parseSseBuffer(buffer, onEvent, flush = false) {
      buffer = buffer.replace(/\r\n/g, "\n");
      const parts = buffer.split("\n\n");
      const rest = parts.pop();
      for (const part of parts) {
        let event = "message";
        let data = "";
        for (const line of part.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:")) data += line.slice(5).trimStart();
        }
        if (data) onEvent(event, JSON.parse(data));
      }
      if (flush && rest.trim()) {
        let event = "message";
        let data = "";
        for (const line of rest.split("\n")) {
          if (line.startsWith("event:")) event = line.slice(6).trim();
          if (line.startsWith("data:")) data += line.slice(5).trimStart();
        }
        if (data) onEvent(event, JSON.parse(data));
        return "";
      }
      return rest;
    }

    async function streamPost(url, body) {
      streamInFlight = true;
      updateSendButtonState();
      showLoading();
      const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
      });
      if (!response.ok || !response.body) {
        hideLoading();
        addMessage("system", `HTTP error ${response.status}`);
        streamInFlight = false;
        updateSendButtonState();
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream: true});
        buffer = parseSseBuffer(buffer, handleEvent);
      }
      buffer += decoder.decode();
      parseSseBuffer(buffer, handleEvent, true);
      hideLoading();
      streamInFlight = false;
      updateSendButtonState();
    }

    async function resumePost(url, body) {
      const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
      });
      if (!response.ok || !response.body) {
        addMessage("system", `HTTP error ${response.status}`);
        return;
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, {stream: true});
        buffer = parseSseBuffer(buffer, handleEvent);
      }
      buffer += decoder.decode();
      parseSseBuffer(buffer, handleEvent, true);
    }

    function handleEvent(type, data) {
      if (type === "node_update") addNode(data);
      if (type === "assistant_message") addMessage("assistant", data.message);
      if (type === "progress_message") {
        addMessage("progress", data.message);
        showLoading();
      }
      if (type === "interrupt") renderInterrupt(data.interrupt, {showChat: true});
      if (type === "done") {
        activeInterruptCard = null;
        activeInterruptSignature = "";
        currentInterruptValue = null;
        renderCandidateMenu(null);
        updateSendButtonState();
        addMessage("system", `Done. Logs: ${data.logs_path}`);
      }
      if (type === "error") addMessage("system", `Error: ${data.error}`);
    }

    function interruptValue(interrupt) {
      return (interrupt && interrupt.interrupts && interrupt.interrupts[0] && interrupt.interrupts[0].value) || {};
    }

    function interruptSignature(interrupt) {
      const value = interruptValue(interrupt);
      const detections = (value.detections || []).map((item) => item.id).join(",");
      const objects = (value.objects || []).map((item) => item.index || item.id || item.label || "").join(",");
      return `${value.type || ""}|${value.message || ""}|d:${detections}|o:${objects}`;
    }

    function localPreviewUrl(previewPath) {
      if (!previewPath) return "";
      if (typeof previewPath === "object") {
        previewPath = previewPath.artifact_path || previewPath.path || "";
      }
      if (!previewPath) return "";
      return `/api/sessions/${sessionId}/preview?path=${encodeURIComponent(String(previewPath))}`;
    }

    function previewUrls(item) {
      const urls = [];
      const addUrl = (url) => {
        if (url && !urls.includes(url)) urls.push(url);
      };
      const addPath = (path) => {
        const url = localPreviewUrl(path);
        if (url) addUrl(url);
      };
      const addMany = (value, handler) => {
        if (Array.isArray(value)) {
          for (const entry of value) handler(entry);
        } else {
          handler(value);
        }
      };
      addMany(item.preview_urls || [], addUrl);
      addUrl(item.preview_url || "");
      addMany(item.preview_paths || [], addPath);
      addPath(item.preview_path);
      return urls;
    }

    function makePreviewFrame(item, url, index) {
      const preview = document.createElement("div");
      preview.className = "preview";
      const image = document.createElement("img");
      const label = item.instance_key || item.label || "candidate";
      image.alt = `${label} preview ${index + 1}`;
      image.onerror = () => {
        image.remove();
        preview.textContent = "Preview missing";
      };
      preview.appendChild(image);
      image.src = url;
      return preview;
    }

    function makePreviewGallery(item) {
      const urls = previewUrls(item);
      const gallery = document.createElement("div");
      gallery.className = `preview-gallery${urls.length > 1 ? " multi" : ""}`;
      if (!urls.length) {
        const preview = document.createElement("div");
        preview.className = "preview";
        preview.textContent = "No preview";
        gallery.appendChild(preview);
        return gallery;
      }
      urls.forEach((url, index) => {
        gallery.appendChild(makePreviewFrame(item, url, index));
      });
      return gallery;
    }

    function renderCandidateMenu(value = currentInterruptValue) {
      candidateList.replaceChildren();
      const detections = (value && value.detections) || [];
      const objects = (value && value.objects) || [];
      const hasPrompt = Boolean(value && value.type);

      if (detections.length || objects.length || hasPrompt) {
        const card = document.createElement("div");
        card.className = "interrupt-card";
        fillInterruptCard(card, value);
        candidateList.appendChild(card);
        candidateMenu.hidden = false;
        activeInterruptCard = card;
        return;
      }

      if (candidateMenuSuppressed) {
        candidateMenu.hidden = true;
        return;
      }

      candidateMenu.hidden = true;
    }

    function makeManualInput(value) {
      const row = document.createElement("div");
      row.className = "manual-input";
      const manual = document.createElement("input");
      manual.type = "text";
      manual.placeholder = value.type === "detection_selection"
        ? "輸入編號或 no"
        : (value.type === "object_selection" ? "輸入編號" : "輸入回覆");
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = "送出";
      const submit = () => {
        const text = manual.value.trim();
        if (!text) return;
        submitInterruptChoice({value: text}, button);
      };
      button.onclick = submit;
      manual.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          submit();
        }
      });
      row.appendChild(manual);
      row.appendChild(button);
      requestAnimationFrame(() => manual.focus());
      return row;
    }

    function makeChoiceButton(item, mode) {
      const button = document.createElement("button");
      button.type = "button";
      if (mode === "detection") {
        button.className = "choice detection-card";
        button.appendChild(makePreviewGallery(item));

        const detail = document.createElement("div");
        const title = document.createElement("div");
        title.className = "choice-title";
        title.textContent = `[${item.id}] ${item.instance_key || item.label || "target"}`;
        const meta = document.createElement("div");
        meta.className = "choice-meta";
        const cameraText = (item.target_camera_names || item.camsrc || []).join(", ") || "camera unknown";
        meta.textContent = `${cameraText} · center=${JSON.stringify(item.center_world || [])}`;
        detail.appendChild(title);
        detail.appendChild(meta);
        button.appendChild(detail);
        button.onclick = () => submitInterruptChoice({selected_detection_id: item.id}, button);
        return button;
      }

      button.className = "choice";
      if (mode === "object") {
        button.textContent = `[${item.index}] ${item.label || item.id}`;
        button.onclick = () => submitInterruptChoice({selected_object_index: item.index}, button);
      } else {
        button.textContent = "No valid target";
        button.onclick = () => submitInterruptChoice({selected_detection_id: 0, selected_object_index: 0}, button);
      }
      return button;
    }

    function fillInterruptCard(card, value) {
      const title = document.createElement("div");
      title.className = "interrupt-title";
      title.textContent = value.message || "Input required";
      card.appendChild(title);

      const detections = value.detections || [];
      const objects = value.objects || [];

      if (value.type === "legacy_input") {
        card.appendChild(makeManualInput(value));
        return;
      }

      if (value.type === "detection_selection" || value.type === "object_selection") {
        card.appendChild(makeManualInput(value));
      }

      const grid = document.createElement("div");
      grid.className = "choice-grid";
      card.appendChild(grid);

      for (const item of detections) grid.appendChild(makeChoiceButton(item, "detection"));
      for (const item of objects) grid.appendChild(makeChoiceButton(item, "object"));
      if (value.type !== "object_selection") grid.appendChild(makeChoiceButton({}, "none"));

      if (!detections.length && !objects.length) {
        const note = document.createElement("div");
        note.className = "empty-choice-note";
        note.textContent = "目前沒有可顯示的候選項。";
        card.insertBefore(note, grid);
      }
    }

    function submitInterruptChoice(payload, button) {
      const card = button ? button.closest(".interrupt-card") : activeInterruptCard;
      if (!card) {
        resumePost(`/api/sessions/${sessionId}/resume/stream`, payload);
        return;
      }
	      const buttons = card.querySelectorAll("button");
      for (const item of buttons) {
        item.disabled = true;
        if (item !== button) item.classList.add("faded");
      }
      if (button) button.classList.add("selected");
      candidateMenuSuppressed = true;
      setTimeout(() => {
        card.classList.add("closing");
      }, 120);
      setTimeout(() => {
        const message = card.closest(".interrupt-message");
        if (message) {
          message.remove();
        } else {
          card.remove();
        }
        if (!message && activeInterruptCard) {
          const activeMessage = activeInterruptCard.closest(".interrupt-message");
          if (activeMessage) activeMessage.remove();
          activeInterruptCard = null;
        }
        if (activeInterruptCard === card) {
          activeInterruptCard = null;
          activeInterruptSignature = "";
        }
        currentInterruptValue = null;
        candidateMenuSuppressed = true;
        renderCandidateMenu(null);
        updateSendButtonState();
      }, 320);
      resumePost(`/api/sessions/${sessionId}/resume/stream`, payload);
    }

    function renderInterrupt(interrupt, options = {}) {
      hideLoading();
      const value = interruptValue(interrupt);
      const signature = interruptSignature(interrupt);
      candidateMenuSuppressed = false;
      currentInterruptValue = value;
      renderCandidateMenu(value);
      updateSendButtonState();
      if (options.showChat === false) return;
      if (activeInterruptSignature === signature) {
        messages.scrollTop = messages.scrollHeight;
        return;
      }
      activeInterruptSignature = signature;
      addMessage("assistant", value.message || "請在左側候選照片中選擇目標。");
      messages.scrollTop = messages.scrollHeight;
    }

    function syncState(data, options = {}) {
	      const state = data.state || {};
	      if (state.task_description) task.textContent = state.task_description;
      const hasPending = Object.prototype.hasOwnProperty.call(data, "pending_interrupt");
      const pending = hasPending ? data.pending_interrupt : (state.pending_interrupt || null);
      if (pending && pending.interrupts && pending.interrupts.length) {
        renderInterrupt(pending, {showChat: options.showChat !== false});
      } else {
        currentInterruptValue = null;
        activeInterruptCard = null;
        activeInterruptSignature = "";
        renderCandidateMenu(null);
        updateSendButtonState();
      }
    }

    function connectStateEvents() {
      if (!sessionId || typeof EventSource === "undefined") return;
      if (stateEvents) stateEvents.close();
      stateEvents = new EventSource(`/api/sessions/${sessionId}/events`);
      stateEvents.addEventListener("state", (event) => {
        syncState(JSON.parse(event.data), {showChat: true});
      });
      stateEvents.onerror = () => {
      };
    }

    window.addEventListener("beforeunload", () => {
      if (stateEvents) stateEvents.close();
    });

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const text = input.value.trim();
      if (!text) return;
      input.value = "";
      addMessage("user", text);
      streamPost(`/api/sessions/${sessionId}/messages/stream`, {message: text});
    });

	    async function restoreSession(savedSessionId) {
	      if (!savedSessionId) return false;
	      try {
	        const response = await fetch(`/api/sessions/${savedSessionId}/state`);
	        if (!response.ok) return false;
	        const data = await response.json();
	        sessionId = savedSessionId;
	        document.getElementById("session").textContent = sessionId;
        addMessage("system", `Restored session. Logs: ${data.logs_path}`);
        syncState(data, {showChat: true});
        connectStateEvents();
        return true;
	      } catch (error) {
	        return false;
	      }
	    }

	    async function init() {
	      const savedSessionId = localStorage.getItem("vlmRlSessionId") || "";
	      if (await restoreSession(savedSessionId)) return;
	      const response = await fetch("/api/sessions", {
	        method: "POST",
	        headers: {"Content-Type": "application/json"},
	        body: JSON.stringify({})
	      });
	      const data = await response.json();
	      sessionId = data.session_id;
	      localStorage.setItem("vlmRlSessionId", sessionId);
	      document.getElementById("session").textContent = sessionId;
      addMessage("assistant", data.greeting);
      addMessage("system", `Logs: ${data.logs_path}`);
      connectStateEvents();
    }
	    init();
	  </script>
</body>
</html>
"""


if FastAPI is not None:
    app = create_app()
else:
    async def app(scope: Dict[str, Any], receive: Any, send: Any) -> None:
        raise RuntimeError("fastapi is not installed. Run `uv sync` or install project dependencies.")
