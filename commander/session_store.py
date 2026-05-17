from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict

from .artifact_store import ArtifactStore, artifact_ref_id

logger = logging.getLogger(__name__)

_FORBIDDEN_STATE_KEYS = {
    "camera_images",
    "image_base64",
    "rgb_base64",
    "depth_base64",
    "world_position_data",
    "agent_result",
}


class SessionMemoryStore:
    """Persist session-scoped rows and artifact refs.

    The SQLite session database is the only session timeline. JSON may still
    exist as artifact payload files, but state patches and node history are not
    mirrored into JSON logs.
    """

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
        self.artifact_store = ArtifactStore(context_id=context_id, base_dir=self.base_dir)
        self.current_state = copy.deepcopy(initial_state)

    def record_event(self, *, step: int, node_name: str, state_update: Dict[str, Any]) -> None:
        """Apply a LangGraph update and write searchable SQLite rows."""
        self._reject_forbidden_state(state_update, key_path=node_name)
        self._merge_state_update(state_update)

        sanitized_update = self._sanitize_value(copy.deepcopy(state_update), snapshot=False)
        execution = sanitized_update.get("last_execution") or {}
        if execution:
            self.artifact_store.record_node_run(step=step, execution=execution)
        else:
            self.artifact_store.record_node_run(
                step=step,
                execution={
                    "node_name": node_name,
                    "status": sanitized_update.get("current_status", ""),
                    "success": True,
                    "latency_sec": 0.0,
                },
            )

        if "task" in sanitized_update and sanitized_update["task"]:
            self.artifact_store.record_task(sanitized_update["task"])

        self._record_domain_rows(step=step, node_name=node_name, update=sanitized_update)
        self.artifact_store.record_state_patch(
            step=step,
            node_name=node_name,
            state_patch=sanitized_update,
        )

    def _merge_state_update(self, state_update: Dict[str, Any]) -> None:
        for key, value in state_update.items():
            if key == "history_buffer":
                existing = self.current_state.get(key, [])
                self.current_state[key] = copy.deepcopy(existing) + copy.deepcopy(value)
                continue
            if key == "navigation" and isinstance(value, dict):
                merged = dict(self.current_state.get("navigation", {}) or {})
                merged.update(copy.deepcopy(value))
                self.current_state[key] = merged
                continue
            self.current_state[key] = copy.deepcopy(value)

    def _record_domain_rows(self, *, step: int, node_name: str, update: Dict[str, Any]) -> None:
        world_position = update.get("world_position") or {}
        if world_position:
            raw_ref = world_position.get("raw_payload_ref") or {}
            self.artifact_store.record_world_snapshot(
                step=step,
                raw_artifact_id=artifact_ref_id(raw_ref) or "",
                candidate_count=int(world_position.get("candidate_count", 0) or 0),
                selected_instance_key=world_position.get("selected_instance_key", ""),
            )

        observation = update.get("observation") or {}
        if observation:
            image_ref = observation.get("image_ref") or {}
            self.artifact_store.record_observation(
                step=step,
                source_node=node_name,
                camera_name=observation.get("camera_name", ""),
                rgb_artifact_id=artifact_ref_id(image_ref),
                description=observation.get("description", ""),
            )

        room_cameras = update.get("room_cameras") or {}
        for camera_name, camera in room_cameras.items():
            rgb_ref = (camera or {}).get("rgb_ref") or {}
            self.artifact_store.record_observation(
                step=step,
                source_node=node_name,
                camera_name=str(camera_name),
                rgb_artifact_id=artifact_ref_id(rgb_ref),
                description="room_camera",
            )

        decision = update.get("decision") or {}
        if decision:
            self.artifact_store.record_decision(step=step, decision=decision)

        navigation = update.get("navigation") or {}
        nav_result = navigation.get("result") or {}
        nav_goal = navigation.get("nav_goal") or nav_result.get("goal") or {}
        if nav_result:
            self.artifact_store.record_nav_run(
                step=step,
                rank=int(navigation.get("current_goal_rank", nav_goal.get("goal_rank", 0)) or 0),
                goal_pose_index=int(navigation.get("current_goal_pose_index", nav_goal.get("goal_pose_index", 0)) or 0),
                goal=nav_goal,
                arrived=bool(nav_result.get("arrived", False)),
                plan_ready=bool(nav_result.get("plan_ready", False)),
                attempt=int(nav_result.get("attempt", 0) or 0),
                events=nav_result.get("events", []) or [],
            )

        for result_key, agent_name in (
            ("item_info", "GetItemInfoNoSam3D Agent"),
            ("grasp_result", "GraspGen Agent"),
            ("approach_result", "Car Approach Agent"),
        ):
            payload = update.get(result_key) or {}
            if not payload:
                continue
            raw_ref = payload.get("raw_result_ref") or {}
            self.artifact_store.record_agent_run(
                step=step,
                node_name=node_name,
                agent_name=agent_name,
                success=bool(payload.get("success", True)),
                output_refs={"raw_result_ref": raw_ref},
                raw_result_artifact_id=artifact_ref_id(raw_ref),
                summary=payload,
            )


    def _sanitize_value(self, value: Any, *, snapshot: bool) -> Any:
        if isinstance(value, dict):
            sanitized: Dict[str, Any] = {}
            for key, item in value.items():
                if snapshot and str(key).startswith("_"):
                    continue
                sanitized[key] = self._sanitize_value(item, snapshot=snapshot)
            return sanitized
        if isinstance(value, list):
            return [self._sanitize_value(item, snapshot=snapshot) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _reject_forbidden_state(self, value: Any, *, key_path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                next_path = f"{key_path}.{key}"
                if key in _FORBIDDEN_STATE_KEYS or str(key).endswith("_base64"):
                    raise ValueError(f"Forbidden raw data key in state update: {next_path}")
                self._reject_forbidden_state(item, key_path=next_path)
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                self._reject_forbidden_state(item, key_path=f"{key_path}[{idx}]")

