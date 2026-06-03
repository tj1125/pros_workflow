from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from ..contracts import ArtifactRef, dump_model


_IMAGE_SUFFIX_MAP = {
    b"\xff\xd8\xff": ".jpg",
    b"\x89PNG\r\n\x1a\n": ".png",
    b"GIF87a": ".gif",
    b"GIF89a": ".gif",
    b"RIFF": ".webp",
}


class ArtifactStore:
    """Session-local SQLite metadata store plus file-backed artifacts."""

    def __init__(self, *, context_id: str, base_dir: str | Path = "logs") -> None:
        self.context_id = context_id
        self.base_dir = Path(base_dir)
        self.session_dir = self.base_dir / "sessions" / context_id
        self.artifacts_dir = self.session_dir / "artifacts"
        self.db_path = self.session_dir / "session.sqlite"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate_schema(self, conn: sqlite3.Connection) -> None:
        self._ensure_column(conn, "world_snapshots", "raw_payload_json", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(conn, "world_snapshots", "target_changed", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column(conn, "world_snapshots", "update_source_node", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(conn, "world_snapshots", "update_reason", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column(conn, "world_snapshots", "update_distance_m", "REAL NOT NULL DEFAULT 0.0")
        self._ensure_column(conn, "world_snapshots", "updated_at", "REAL NOT NULL DEFAULT 0.0")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                    context_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL DEFAULT '',
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    final_status TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    context_id TEXT NOT NULL,
                    original_user_request TEXT NOT NULL,
                    normalized_task TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    success_criteria_json TEXT NOT NULL,
                    done_policy TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    context_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mime_type TEXT,
                    created_by_node TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS node_runs (
                    run_id TEXT PRIMARY KEY,
                    context_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    node_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    latency_sec REAL NOT NULL,
                    route_to TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS state_patches (
                    patch_id TEXT PRIMARY KEY,
                    context_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    node_name TEXT NOT NULL,
                    state_patch_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS world_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    context_id TEXT NOT NULL,
                    step INTEGER NOT NULL DEFAULT 0,
                    raw_payload_json TEXT NOT NULL DEFAULT '',
                    candidate_count INTEGER NOT NULL,
                    selected_instance_key TEXT NOT NULL DEFAULT '',
                    target_changed INTEGER NOT NULL DEFAULT 0,
                    update_source_node TEXT NOT NULL DEFAULT '',
                    update_reason TEXT NOT NULL DEFAULT '',
                    update_distance_m REAL NOT NULL DEFAULT 0.0,
                    updated_at REAL NOT NULL DEFAULT 0.0,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observations (
                    observation_id TEXT PRIMARY KEY,
                    context_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    source_node TEXT NOT NULL,
                    camera_name TEXT NOT NULL DEFAULT '',
                    rgb_artifact_id TEXT,
                    depth_artifact_id TEXT,
                    description TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_runs (
                    agent_run_id TEXT PRIMARY KEY,
                    context_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    node_name TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    input_refs_json TEXT NOT NULL,
                    output_refs_json TEXT NOT NULL,
                    raw_result_artifact_id TEXT,
                    summary_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS nav_runs (
                    nav_run_id TEXT PRIMARY KEY,
                    context_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    rank INTEGER NOT NULL,
                    goal_pose_index INTEGER NOT NULL,
                    goal_json TEXT NOT NULL,
                    arrived INTEGER NOT NULL,
                    plan_ready INTEGER NOT NULL,
                    attempt INTEGER NOT NULL,
                    events_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS goal_poses (
                    goal_pose_id TEXT PRIMARY KEY,
                    context_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    source_node TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    goal_pose_index INTEGER NOT NULL,
                    goal_json TEXT NOT NULL,
                    goal_pose_db_json TEXT NOT NULL,
                    nav_goal_pose_source TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id TEXT PRIMARY KEY,
                    context_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    reasoning TEXT NOT NULL,
                    call_module TEXT NOT NULL,
                    module_params_json TEXT NOT NULL,
                    model TEXT NOT NULL DEFAULT '',
                    latency_sec REAL NOT NULL,
                    created_at REAL NOT NULL
                );
                """
            )
            self._migrate_schema(conn)
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions(context_id, started_at)
                VALUES (?, ?)
                """,
                (self.context_id, time.time()),
            )

    def record_task(self, task: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tasks(
                    task_id, context_id, original_user_request, normalized_task,
                    task_type, success_criteria_json, done_policy, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task.get("task_id", ""),
                    self.context_id,
                    task.get("original_user_request", ""),
                    task.get("normalized_task", ""),
                    task.get("task_type", ""),
                    json.dumps(task.get("success_criteria", []), ensure_ascii=False),
                    task.get("done_policy", ""),
                    time.time(),
                ),
            )

    def save_world_snapshot_raw(
        self,
        raw_payload: Any,
        *,
        candidate_count: int,
        selected_instance_key: str = "",
        created_by_node: str,
        target_changed: bool = False,
        update_reason: str = "",
        update_distance_m: float = 0.0,
        updated_at: float | None = None,
    ) -> str:
        snapshot_id = uuid.uuid4().hex
        timestamp = time.time() if updated_at is None else float(updated_at)
        raw_payload_json = json.dumps(raw_payload, ensure_ascii=False, default=str)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO world_snapshots(
                    snapshot_id, context_id, step, raw_payload_json, candidate_count,
                    selected_instance_key, target_changed, update_source_node,
                    update_reason, update_distance_m, updated_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    self.context_id,
                    0,
                    raw_payload_json,
                    int(candidate_count),
                    selected_instance_key,
                    1 if target_changed else 0,
                    created_by_node,
                    update_reason,
                    float(update_distance_m),
                    timestamp,
                    time.time(),
                ),
            )
        return snapshot_id

    def load_world_snapshot_raw(self, snapshot_id: str) -> Any:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT raw_payload_json FROM world_snapshots
                WHERE context_id = ? AND snapshot_id = ?
                """,
                (self.context_id, snapshot_id),
            ).fetchone()
        if row is None or not row["raw_payload_json"]:
            raise KeyError(f"world snapshot raw payload not found: {snapshot_id}")
        return json.loads(row["raw_payload_json"])

    def save_json(
        self,
        kind: str,
        payload: Any,
        *,
        created_by_node: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        data = json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        return self.save_bytes(
            kind,
            data,
            created_by_node=created_by_node,
            mime_type="application/json",
            suffix=".json",
            metadata=metadata,
        )

    def save_text(
        self,
        kind: str,
        text: str,
        *,
        created_by_node: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        return self.save_bytes(
            kind,
            text.encode("utf-8"),
            created_by_node=created_by_node,
            mime_type="text/plain",
            suffix=".txt",
            metadata=metadata,
        )

    def save_base64_image(
        self,
        kind: str,
        encoded: str,
        *,
        created_by_node: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        data = base64.b64decode(encoded, validate=True)
        suffix = self._guess_image_suffix(data) or ".img"
        mime_type = mimetypes.types_map.get(suffix.lower(), "application/octet-stream")
        return self.save_bytes(
            kind,
            data,
            created_by_node=created_by_node,
            mime_type=mime_type,
            suffix=suffix,
            metadata=metadata,
        )

    def save_file(
        self,
        kind: str,
        source_path: str | Path,
        *,
        created_by_node: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        source = Path(source_path)
        data = source.read_bytes()
        suffix = source.suffix or ".bin"
        mime_type = mimetypes.types_map.get(suffix.lower(), "application/octet-stream")
        return self.save_bytes(
            kind,
            data,
            created_by_node=created_by_node,
            mime_type=mime_type,
            suffix=suffix,
            metadata=metadata,
        )

    def save_bytes(
        self,
        kind: str,
        data: bytes,
        *,
        created_by_node: str,
        mime_type: str | None = None,
        suffix: str = ".bin",
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        sha256 = hashlib.sha256(data).hexdigest()
        artifact_id = uuid.uuid4().hex
        safe_kind = _safe_slug(kind)
        safe_node = _safe_slug(created_by_node)
        directory = self.artifacts_dir / safe_kind
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{safe_node}_{artifact_id[:12]}_{sha256[:12]}{suffix}"
        path = directory / filename
        path.write_bytes(data)
        rel_path = str(path.relative_to(self.base_dir))
        ref = ArtifactRef(
            artifact_id=artifact_id,
            kind=kind,
            path=rel_path,
            sha256=sha256,
            size_bytes=len(data),
            mime_type=mime_type,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts(
                    artifact_id, context_id, kind, path, sha256, size_bytes,
                    mime_type, created_by_node, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ref.artifact_id,
                    self.context_id,
                    ref.kind,
                    ref.path,
                    ref.sha256,
                    ref.size_bytes,
                    ref.mime_type,
                    created_by_node,
                    time.time(),
                    json.dumps(metadata or {}, ensure_ascii=False, default=str),
                ),
            )
        return ref

    def load_bytes(self, ref: ArtifactRef | dict[str, Any]) -> bytes:
        return self.resolve_path(ref).read_bytes()

    def load_text(self, ref: ArtifactRef | dict[str, Any]) -> str:
        return self.load_bytes(ref).decode("utf-8")

    def load_json(self, ref: ArtifactRef | dict[str, Any]) -> Any:
        return json.loads(self.load_text(ref))

    def load_base64(self, ref: ArtifactRef | dict[str, Any]) -> str:
        return base64.b64encode(self.load_bytes(ref)).decode("ascii")

    def resolve_path(self, ref: ArtifactRef | dict[str, Any]) -> Path:
        if isinstance(ref, dict):
            path = ref["path"]
        else:
            path = ref.path
        return self.base_dir / path

    def record_node_run(self, *, step: int, execution: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO node_runs(
                    run_id, context_id, step, node_name, status, success,
                    latency_sec, route_to, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    self.context_id,
                    step,
                    execution.get("node_name", ""),
                    execution.get("status", ""),
                    1 if execution.get("success", False) else 0,
                    float(execution.get("latency_sec", 0.0) or 0.0),
                    execution.get("route_to", ""),
                    execution.get("error", ""),
                    time.time(),
                ),
            )

    def record_state_patch(self, *, step: int, node_name: str, state_patch: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO state_patches(
                    patch_id, context_id, step, node_name, state_patch_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    self.context_id,
                    step,
                    node_name,
                    json.dumps(state_patch, ensure_ascii=False, default=str),
                    time.time(),
                ),
            )

    def record_world_snapshot(
        self,
        *,
        step: int,
        snapshot_id: str = "",
        candidate_count: int = 0,
        selected_instance_key: str = "",
        target_changed: bool = False,
        update_source_node: str = "",
        update_reason: str = "",
        update_distance_m: float = 0.0,
        updated_at: float = 0.0,
    ) -> None:
        with self._connect() as conn:
            if snapshot_id:
                cursor = conn.execute(
                    """
                    UPDATE world_snapshots
                    SET step = ?, candidate_count = ?, selected_instance_key = ?,
                        target_changed = ?, update_source_node = ?, update_reason = ?,
                        update_distance_m = ?, updated_at = ?
                    WHERE context_id = ? AND snapshot_id = ?
                    """,
                    (
                        step,
                        int(candidate_count),
                        selected_instance_key,
                        1 if target_changed else 0,
                        update_source_node,
                        update_reason,
                        float(update_distance_m),
                        float(updated_at or time.time()),
                        self.context_id,
                        snapshot_id,
                    ),
                )
                if cursor.rowcount:
                    return
            conn.execute(
                """
                INSERT INTO world_snapshots(
                    snapshot_id, context_id, step, raw_payload_json, candidate_count,
                    selected_instance_key, target_changed, update_source_node,
                    update_reason, update_distance_m, updated_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id or uuid.uuid4().hex,
                    self.context_id,
                    step,
                    "",
                    int(candidate_count),
                    selected_instance_key,
                    1 if target_changed else 0,
                    update_source_node,
                    update_reason,
                    float(update_distance_m),
                    float(updated_at or time.time()),
                    time.time(),
                ),
            )

    def record_observation(
        self,
        *,
        step: int,
        source_node: str,
        camera_name: str = "",
        rgb_artifact_id: str | None = None,
        depth_artifact_id: str | None = None,
        description: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO observations(
                    observation_id, context_id, step, source_node, camera_name,
                    rgb_artifact_id, depth_artifact_id, description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    self.context_id,
                    step,
                    source_node,
                    camera_name,
                    rgb_artifact_id,
                    depth_artifact_id,
                    description,
                    time.time(),
                ),
            )

    def record_agent_run(
        self,
        *,
        step: int,
        node_name: str,
        agent_name: str,
        success: bool,
        input_refs: dict[str, Any] | None = None,
        output_refs: dict[str, Any] | None = None,
        raw_result_artifact_id: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_runs(
                    agent_run_id, context_id, step, node_name, agent_name, success,
                    input_refs_json, output_refs_json, raw_result_artifact_id,
                    summary_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    self.context_id,
                    step,
                    node_name,
                    agent_name,
                    1 if success else 0,
                    json.dumps(input_refs or {}, ensure_ascii=False, default=str),
                    json.dumps(output_refs or {}, ensure_ascii=False, default=str),
                    raw_result_artifact_id,
                    json.dumps(summary or {}, ensure_ascii=False, default=str),
                    time.time(),
                ),
            )

    def record_nav_run(
        self,
        *,
        step: int,
        rank: int,
        goal_pose_index: int,
        goal: dict[str, Any],
        arrived: bool,
        plan_ready: bool,
        attempt: int,
        events: list[dict[str, Any]],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO nav_runs(
                    nav_run_id, context_id, step, rank, goal_pose_index, goal_json,
                    arrived, plan_ready, attempt, events_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    self.context_id,
                    step,
                    int(rank),
                    int(goal_pose_index),
                    json.dumps(goal, ensure_ascii=False, default=str),
                    1 if arrived else 0,
                    1 if plan_ready else 0,
                    int(attempt),
                    json.dumps(_filter_tracking_events(events), ensure_ascii=False, default=str),
                    time.time(),
                ),
            )

    def record_goal_pose(
        self,
        *,
        step: int,
        source_node: str,
        rank: int,
        goal_pose_index: int,
        goal: dict[str, Any],
        goal_pose_db: dict[str, Any],
        nav_goal_pose_source: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO goal_poses(
                    goal_pose_id, context_id, step, source_node, rank, goal_pose_index,
                    goal_json, goal_pose_db_json, nav_goal_pose_source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    self.context_id,
                    step,
                    source_node,
                    int(rank),
                    int(goal_pose_index),
                    json.dumps(goal or {}, ensure_ascii=False, default=str),
                    json.dumps(goal_pose_db or {}, ensure_ascii=False, default=str),
                    nav_goal_pose_source,
                    time.time(),
                ),
            )

    def record_decision(self, *, step: int, decision: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO decisions(
                    decision_id, context_id, step, reasoning, call_module,
                    module_params_json, model, latency_sec, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    self.context_id,
                    step,
                    decision.get("reasoning", ""),
                    decision.get("call_module", ""),
                    json.dumps(decision.get("module_params", {}), ensure_ascii=False, default=str),
                    decision.get("model", ""),
                    float(decision.get("latency_sec", 0.0) or 0.0),
                    time.time(),
                ),
            )

    @staticmethod
    def _guess_image_suffix(data: bytes) -> str | None:
        for magic, suffix in _IMAGE_SUFFIX_MAP.items():
            if data.startswith(magic):
                return suffix
        return None


def _safe_slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in value)
    return safe.strip("_") or "artifact"


def _filter_tracking_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if str((event or {}).get("event", "") or "").strip().lower() != "tracking"
    ]


def artifact_ref_id(ref: ArtifactRef | dict[str, Any] | None) -> str | None:
    if not ref:
        return None
    if isinstance(ref, dict):
        return ref.get("artifact_id")
    return ref.artifact_id


def artifact_ref_json(ref: ArtifactRef | None) -> dict[str, Any]:
    return dump_model(ref)

