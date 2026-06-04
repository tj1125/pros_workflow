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
            state_patch=self._compact_state_patch(sanitized_update),
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
            self.artifact_store.record_world_snapshot(
                step=step,
                snapshot_id=str(world_position.get("snapshot_id", "") or ""),
                candidate_count=int(world_position.get("candidate_count", 0) or 0),
                selected_instance_key=world_position.get("selected_instance_key", ""),
                target_changed=bool(world_position.get("target_changed", False)),
                update_source_node=world_position.get("update_source_node", node_name),
                update_reason=world_position.get("update_reason", ""),
                update_distance_m=float(world_position.get("update_distance_m", 0.0) or 0.0),
                updated_at=float(world_position.get("updated_at", 0.0) or 0.0),
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
        goal_pose_db = navigation.get("goal_pose_db") or {}
        if node_name in {"get_item_info_no_sam3d_node", "major_nav_node"} and (nav_goal or goal_pose_db):
            self.artifact_store.record_goal_pose(
                step=step,
                source_node=node_name,
                rank=int(navigation.get("current_goal_rank", nav_goal.get("goal_rank", 0) if isinstance(nav_goal, dict) else 0) or 0),
                goal_pose_index=int(navigation.get("current_goal_pose_index", nav_goal.get("goal_pose_index", 0) if isinstance(nav_goal, dict) else 0) or 0),
                goal=nav_goal if isinstance(nav_goal, dict) else {},
                goal_pose_db=self._compact_goal_pose_db(goal_pose_db if isinstance(goal_pose_db, dict) else {}),
                nav_goal_pose_source=navigation.get("nav_goal_pose_source", ""),
            )
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
                summary=self._compact_agent_summary(result_key, payload),
            )

    def _compact_state_patch(self, update: Dict[str, Any]) -> Dict[str, Any]:
        compact = copy.deepcopy(update)
        room_cameras = compact.get("room_cameras")
        if isinstance(room_cameras, dict):
            compact["room_cameras"] = {
                str(name): {
                    key: camera.get(key)
                    for key in ("camera_name", "topic", "bbox", "preview_path", "preview_ref", "rgb_ref")
                    if isinstance(camera, dict) and key in camera and camera.get(key) not in ({}, [], "", None)
                }
                for name, camera in room_cameras.items()
                if isinstance(camera, dict)
            }

        item_info = compact.get("item_info")
        if isinstance(item_info, dict) and item_info:
            compact["item_info"] = self._compact_item_info(item_info)

        navigation = compact.get("navigation")
        if isinstance(navigation, dict):
            navigation = copy.deepcopy(navigation)
            if isinstance(navigation.get("goal_pose_db"), dict):
                navigation["goal_pose_db"] = self._compact_goal_pose_db(navigation["goal_pose_db"])
            result = navigation.get("result")
            if isinstance(result, dict) and isinstance(result.get("events"), list):
                result["event_count"] = len(result.get("events") or [])
                result.pop("events", None)
            compact["navigation"] = navigation

        grasp_result = compact.get("grasp_result")
        if isinstance(grasp_result, dict) and grasp_result:
            compact["grasp_result"] = self._compact_grasp_result(grasp_result)

        approach_result = compact.get("approach_result")
        if isinstance(approach_result, dict) and approach_result:
            compact["approach_result"] = self._compact_approach_result(approach_result)

        return compact

    @staticmethod
    def _compact_item_info(payload: Dict[str, Any]) -> Dict[str, Any]:
        group_ranking = payload.get("group_ranking") or []
        compact = {
            key: payload.get(key)
            for key in (
                "center_world",
                "center_world_coordinate_frame",
                "primary_camera_id",
                "target_instance_key",
                "target_topic_key",
                "goal_pose_path",
                "raw_result_ref",
                "a2a_task_id",
            )
            if payload.get(key) not in ({}, [], "", None)
        }
        compact["group_count"] = len(group_ranking) if isinstance(group_ranking, list) else 0
        if isinstance(group_ranking, list) and group_ranking:
            compact["top_group"] = _compact_mapping(
                group_ranking[0],
                keys=("rank", "orientation_group", "best_confidence", "best_goal_pose_ros_map", "map_feasible", "selection_mode"),
            )
        return compact

    @staticmethod
    def _compact_goal_pose_db(payload: Dict[str, Any]) -> Dict[str, Any]:
        ranks = payload.get("ranks") if isinstance(payload.get("ranks"), dict) else {}
        compact = {
            key: payload.get(key)
            for key in ("target_instance_key", "center_world", "current_goal_rank", "rank_order", "updated_at")
            if payload.get(key) not in ({}, [], "", None)
        }
        compact["rank_count"] = len(ranks)
        current_rank = str(payload.get("current_goal_rank", "1") or "1")
        top = ranks.get(current_rank) or next(iter(ranks.values()), {}) if ranks else {}
        if isinstance(top, dict) and top:
            compact["current_rank_summary"] = _compact_mapping(
                top,
                keys=("rank", "orientation_group", "best_confidence", "best_goal_pose_ros_map", "map_feasible", "selection_mode"),
            )
        return compact

    @staticmethod
    def _compact_grasp_result(payload: Dict[str, Any]) -> Dict[str, Any]:
        compact = {
            key: payload.get(key)
            for key in (
                "object_id",
                "camera_name",
                "success",
                "target_instance_key",
                "bbox_xyxy",
                "detection_confidence",
                "target_selection",
                "grasp_confidence",
                "num_candidate_grasps",
                "num_valid_grasps",
                "best_grasp_pose_camera",
                "object_reference_center_camera",
                "raw_result_ref",
                "a2a_task_id",
            )
            if payload.get(key) not in ({}, [], "", None)
        }
        if isinstance(payload.get("valid_grasp_poses_camera"), list):
            compact["valid_grasp_pose_count"] = len(payload["valid_grasp_poses_camera"])
        return compact

    @staticmethod
    def _compact_approach_result(payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: payload.get(key)
            for key in (
                "success",
                "status_code",
                "phase",
                "message",
                "next_agent",
                "selected_solution",
                "closest_solution",
                "selected_solution_source",
                "fallback_to_closest_solution",
                "raw_result_ref",
            )
            if payload.get(key) not in ({}, [], "", None)
        }

    def _compact_agent_summary(self, result_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if result_key == "item_info":
            return self._compact_item_info(payload)
        if result_key == "grasp_result":
            return self._compact_grasp_result(payload)
        if result_key == "approach_result":
            return self._compact_approach_result(payload)
        return _compact_mapping(payload, keys=tuple(payload.keys()))

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



def _compact_mapping(mapping: Any, *, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        return {}
    return {key: mapping.get(key) for key in keys if mapping.get(key) not in ({}, [], "", None)}
