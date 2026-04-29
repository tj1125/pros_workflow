import base64
import copy
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .state import LATEST_RESULT_KEYS, _keep_last_three

logger = logging.getLogger(__name__)

_TRANSIENT_SNAPSHOT_KEYS = {"_exec_latency"}
_IMAGE_SUFFIX_MAP = {
    b"\xff\xd8\xff": ".jpg",
    b"\x89PNG\r\n\x1a\n": ".png",
    b"GIF87a": ".gif",
    b"GIF89a": ".gif",
    b"RIFF": ".webp",
}


class SessionMemoryStore:
    """Persist session-scoped memory snapshots, deltas, and artifacts under logs/."""

    def __init__(
        self,
        *,
        context_id: str,
        initial_state: Dict[str, Any],
        base_dir: str | Path = "logs",
    ) -> None:
        self.context_id = context_id
        self.base_dir = Path(base_dir)
        self.session_dir = self.base_dir / "sessions" / context_id
        self.events_dir = self.session_dir / "events"
        self.raw_results_dir = self.session_dir / "artifacts" / "raw_results"
        self.observations_dir = self.session_dir / "artifacts" / "observations"
        self.previews_dir = self.session_dir / "artifacts" / "previews"
        self.current_state = copy.deepcopy(initial_state)
        self._artifact_cache: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._latest_agent_result_artifact: Dict[str, Any] = {}

        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.raw_results_dir.mkdir(parents=True, exist_ok=True)
        self.observations_dir.mkdir(parents=True, exist_ok=True)
        self.previews_dir.mkdir(parents=True, exist_ok=True)
        self._write_latest_snapshots()

    def record_event(self, *, step: int, node_name: str, state_update: Dict[str, Any]) -> None:
        """Apply a LangGraph event update, persist deltas, and refresh latest snapshots."""
        self._merge_state_update(state_update)

        module = str(self.current_state.get("call_module", "") or "")
        status = str(self.current_state.get("current_status", "") or "")
        raw_result_ref = None
        if "agent_result" in state_update:
            raw_result_ref = self._write_raw_result_artifact(
                step=step,
                node_name=node_name,
                module=module or node_name,
                raw_result=state_update["agent_result"],
            )
            self._latest_agent_result_artifact = raw_result_ref

        sanitized_update = self._sanitize_value(
            copy.deepcopy(state_update),
            step=step,
            key_path=f"event.{node_name}",
            snapshot=False,
        )
        if raw_result_ref is not None:
            sanitized_update["agent_result_artifact"] = raw_result_ref

        self._append_jsonl(
            self.events_dir / "state_updates.jsonl",
            {
                "iso_timestamp": self._iso_timestamp(),
                "context_id": self.context_id,
                "step": step,
                "node_name": node_name,
                "module": module,
                "current_status": status,
                "state_update": sanitized_update,
            },
        )

        if node_name == "update_memory_node":
            memory_entries = state_update.get("history_buffer") or []
            if memory_entries:
                memory_entry = copy.deepcopy(memory_entries[-1])
                self._append_jsonl(
                    self.events_dir / "memory_updates.jsonl",
                    {
                        "iso_timestamp": self._iso_timestamp(),
                        "context_id": self.context_id,
                        "step": step,
                        "node_name": node_name,
                        "module": module,
                        "trace_id": memory_entry.get("trace_id", ""),
                        "memory_entry": memory_entry,
                        "latest_result_key": self._latest_result_key(module),
                    },
                )

        self._write_latest_snapshots()

    def _merge_state_update(self, state_update: Dict[str, Any]) -> None:
        for key, value in state_update.items():
            if key == "history_buffer":
                existing = self.current_state.get("history_buffer", [])
                self.current_state[key] = _keep_last_three(existing, copy.deepcopy(value))
                continue
            self.current_state[key] = copy.deepcopy(value)

    def _write_latest_snapshots(self) -> None:
        session_state = self._snapshot_state()
        sanitized_session_state = self._sanitize_value(
            session_state,
            step=0,
            key_path="session_state",
            snapshot=True,
        )

        self._write_json(self.session_dir / "session_state.latest.json", sanitized_session_state)
        self._write_json(
            self.session_dir / "history_buffer.latest.json",
            sanitized_session_state.get("history_buffer", []),
        )
        self._write_json(
            self.session_dir / "latest_results.latest.json",
            {key: sanitized_session_state.get(key, {}) for key in LATEST_RESULT_KEYS},
        )

    def _snapshot_state(self) -> Dict[str, Any]:
        state = copy.deepcopy(self.current_state)
        for key in list(state.keys()):
            if key in _TRANSIENT_SNAPSHOT_KEYS or key.startswith("_"):
                state.pop(key, None)
        if state.get("agent_result", "") not in ("", None) and self._latest_agent_result_artifact:
            state["agent_result_artifact"] = copy.deepcopy(self._latest_agent_result_artifact)
        return state

    def _sanitize_value(
        self,
        value: Any,
        *,
        step: int,
        key_path: str,
        snapshot: bool,
    ) -> Any:
        if isinstance(value, dict):
            sanitized: Dict[str, Any] = {}
            for key, item in value.items():
                if snapshot and (
                    key in _TRANSIENT_SNAPSHOT_KEYS
                    or (isinstance(key, str) and key.startswith("_"))
                ):
                    continue

                next_key_path = f"{key_path}.{key}" if key_path else key
                if isinstance(item, str) and item:
                    if key.endswith("_base64"):
                        sanitized[key] = self._persist_base64_artifact(
                            encoded=item,
                            step=step,
                            key_path=next_key_path,
                        )
                        continue
                    if key == "preview_path":
                        preview_ref = self._persist_preview_artifact(
                            preview_path=Path(item),
                            step=step,
                            key_path=next_key_path,
                        )
                        sanitized[key] = preview_ref if preview_ref is not None else item
                        continue

                sanitized[key] = self._sanitize_value(
                    item,
                    step=step,
                    key_path=next_key_path,
                    snapshot=snapshot,
                )

            if self._is_nav_move_events_key_path(key_path):
                return self._filter_persisted_nav_events(sanitized)
            return sanitized

        if isinstance(value, list):
            sanitized_list = [
                self._sanitize_value(item, step=step, key_path=key_path, snapshot=snapshot)
                for item in value
            ]
            if self._is_nav_move_events_key_path(key_path):
                return self._filter_persisted_nav_events(sanitized_list)
            return sanitized_list

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        return str(value)

    def _persist_base64_artifact(self, *, encoded: str, step: int, key_path: str) -> Dict[str, Any]:
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except Exception:
            return self._write_cached_artifact(
                data=encoded.encode("utf-8"),
                directory=self.observations_dir,
                artifact_kind="base64_text",
                suffix=".b64.txt",
                suggested_name=f"step_{step:03d}_{self._slugify(key_path)}",
            )

        suffix = self._guess_image_suffix(decoded)
        if suffix is None:
            return self._write_cached_artifact(
                data=encoded.encode("utf-8"),
                directory=self.observations_dir,
                artifact_kind="base64_text",
                suffix=".b64.txt",
                suggested_name=f"step_{step:03d}_{self._slugify(key_path)}",
            )

        return self._write_cached_artifact(
            data=decoded,
            directory=self.observations_dir,
            artifact_kind="observation_image",
            suffix=suffix,
            suggested_name=f"step_{step:03d}_{self._slugify(key_path)}",
        )

    def _persist_preview_artifact(
        self,
        *,
        preview_path: Path,
        step: int,
        key_path: str,
    ) -> Dict[str, Any] | None:
        if not preview_path.exists() or not preview_path.is_file():
            return None

        try:
            data = preview_path.read_bytes()
        except OSError as exc:
            logger.warning("[SessionMemoryStore] Failed to read preview '%s': %s", preview_path, exc)
            return None

        suffix = preview_path.suffix or ".bin"
        return self._write_cached_artifact(
            data=data,
            directory=self.previews_dir,
            artifact_kind="preview_image",
            suffix=suffix,
            suggested_name=f"step_{step:03d}_{self._slugify(key_path)}",
        )

    def _write_raw_result_artifact(
        self,
        *,
        step: int,
        node_name: str,
        module: str,
        raw_result: Any,
    ) -> Dict[str, Any]:
        payload = {
            "iso_timestamp": self._iso_timestamp(),
            "context_id": self.context_id,
            "step": step,
            "node_name": node_name,
            "module": module,
            "raw_result": raw_result,
        }
        raw_bytes = json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        filename = f"step_{step:03d}_{self._slugify(module or node_name)}.json"
        path = self.raw_results_dir / filename
        path.write_bytes(raw_bytes)
        return {
            "artifact_path": str(path.relative_to(self.base_dir)),
            "artifact_kind": "raw_result",
            "size_bytes": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }

    def _write_cached_artifact(
        self,
        *,
        data: bytes,
        directory: Path,
        artifact_kind: str,
        suffix: str,
        suggested_name: str,
    ) -> Dict[str, Any]:
        sha256 = hashlib.sha256(data).hexdigest()
        cache_key = (artifact_kind, sha256)
        if cache_key in self._artifact_cache:
            return copy.deepcopy(self._artifact_cache[cache_key])

        filename = f"{suggested_name}_{sha256[:12]}{suffix}"
        path = directory / filename
        path.write_bytes(data)
        ref = {
            "artifact_path": str(path.relative_to(self.base_dir)),
            "artifact_kind": artifact_kind,
            "size_bytes": len(data),
            "sha256": sha256,
        }
        self._artifact_cache[cache_key] = ref
        return copy.deepcopy(ref)

    @staticmethod
    def _guess_image_suffix(data: bytes) -> str | None:
        for magic, suffix in _IMAGE_SUFFIX_MAP.items():
            if data.startswith(magic):
                return suffix
        return None

    @staticmethod
    def _slugify(value: str) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in value)
        return safe.strip("_") or "artifact"

    @staticmethod
    def _latest_result_key(module: str) -> str:
        mapping = {
            "nav_agent": "latest_nav_result",
            "grasp_agent": "latest_grasp_result",
            "view_agent": "latest_view_result",
            "approach_agent": "latest_approach_result",
            "car_approach_agent": "latest_approach_result",
            "arm_approach_agent": "latest_approach_result",
        }
        return mapping.get(module, "")

    @staticmethod
    def _is_nav_move_events_key_path(key_path: str) -> bool:
        return key_path.endswith("nav_move_events")

    @staticmethod
    def _filter_persisted_nav_events(value: Any) -> Any:
        if isinstance(value, list):
            return [
                item
                for item in value
                if not (
                    isinstance(item, dict)
                    and str(item.get("event", "") or "").strip().lower() == "tracking"
                )
            ]
        if isinstance(value, dict):
            if str(value.get("event", "") or "").strip().lower() == "tracking":
                return {}
            return value
        return value

    @staticmethod
    def _iso_timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _append_jsonl(path: Path, entry: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
