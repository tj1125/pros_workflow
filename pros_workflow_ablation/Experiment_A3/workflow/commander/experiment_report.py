"""
commander/experiment_report.py — Experiment A3 execution recorder.

Ablation A3 (``w/o get_item_info_no_sam3d_node``) removes object-information
estimation. The full system asks the no-SAM3D item-info A2A service for object
size, object yaw, grasp-direction grouping and grasp-feasibility ranked goal
poses. In this ablation ``get_item_info_no_sam3d_node`` instead synthesises
candidate goal poses from the target centre alone, at a fixed stand-off distance
in a fixed set of directions (see ``commander/flows/pick.py``).

This module records a complete run from ``get_item_info_no_sam3d_node`` to the
end of the workflow and appends one entry to the shared result file
``<workflow>/result/result.json`` (each run gets the next ``A3_NNN`` id) so the
ablation results can be analysed: task success rate, completion time, slowest
node, how many candidate goal poses were produced/tried, which candidate rank
finally worked, the VLM (reason) decisions, where failures happen, ...

The purpose of the ablation is to show that object-information estimation helps
candidate goal-pose generation, observation-angle selection, grasp feasibility
and overall task success rate. Comparing these reports against the full-system
runs makes that contribution measurable.

Usage (see main.py / web/app.py)::

    reporter = ExperimentReport(context_id=context_id)
    async for event in graph.astream(...):
        for node_name, state_update in event.items():
            reporter.ingest(node_name, state_update)
    path = reporter.write()   # only writes once the run reaches START_NODE
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

START_NODE = "get_item_info_no_sam3d_node"
END_NODE = "END"

# Nodes whose latency reflects an external A2A agent round-trip. NOTE: in the A3
# ablation, get_item_info_no_sam3d_node no longer calls an A2A service (it is
# pure local geometry), so it is intentionally NOT mapped here.
_A2A_NODE_TO_AGENT = {
    "car_grasp_node": "grasp_agent",
    "car_approach_node": "car_approach_agent",
}

# All runs accumulate into one file under the workflow root, not under logs/.
_WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
RESULT_FILENAME = "result.json"


def _load_results(path: Path) -> List[Dict[str, Any]]:
    """Read the accumulated result list; tolerate a missing/corrupt file."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _next_experiment_id(prefix: str, existing: List[Dict[str, Any]]) -> str:
    """Next ``<prefix>_NNN`` id, continuing from the highest already stored."""
    highest = 0
    for entry in existing:
        if not isinstance(entry, dict):
            continue
        eid = str(entry.get("experiment_id", ""))
        if eid.startswith(f"{prefix}_"):
            try:
                highest = max(highest, int(eid.rsplit("_", 1)[-1]))
            except ValueError:
                continue
    return f"{prefix}_{highest + 1:03d}"


class ExperimentReport:
    """Accumulates LangGraph node events into a single Experiment A3 report."""

    def __init__(
        self,
        *,
        context_id: str = "",
        scene_id: str = "",
        result_file: str = "",
        mode: str = "fixed_distance_no_item_info",
    ) -> None:
        self.context_id = context_id
        self.scene_id = scene_id or os.getenv("EXPERIMENT_SCENE_ID", "")
        self.result_path = Path(
            result_file
            or os.getenv("EXPERIMENT_A3_RESULT_FILE", "")
            or (_WORKFLOW_ROOT / "result" / RESULT_FILENAME)
        )
        self.mode = mode

        # Running snapshot of the latest value of every state slice (the graph
        # only emits the keys a node actually returned, so we merge them).
        self._latest: Dict[str, Any] = {}

        self._started = False
        self.node_sequence: List[str] = []
        self.node_latency_sec: Dict[str, float] = {}
        self.node_visit_count: Dict[str, int] = {}
        self.total_time: float = 0.0

        self.reason_actions: List[str] = []
        self.a2a_latency_sec: Dict[str, float] = {}
        self.vlm_latency_sec: float = 0.0

        self.ranked_goal_count: int = 0
        self.goal_pose_mode: str = ""
        self._tried_ranks: "set[int]" = set()
        self.success_goal_rank: Optional[int] = None

        self.nav_success: bool = False
        self.nav_time_sec: float = 0.0
        self.memory_update_count: int = 0

    @property
    def started(self) -> bool:
        """True once the run has reached the report window start node."""
        return self._started

    # ------------------------------------------------------------------ ingest
    def ingest(self, node_name: str, state_update: Dict[str, Any]) -> None:
        if not isinstance(state_update, dict):
            return
        # Merge into the running snapshot regardless of where we are in the run.
        for key, value in state_update.items():
            self._latest[key] = value

        # The report window opens at the first get_item_info_no_sam3d_node.
        if node_name == START_NODE:
            self._started = True
        if not self._started:
            return

        execution = state_update.get("last_execution", {}) or {}
        latency = float(execution.get("latency_sec", 0.0) or 0.0)

        self.node_sequence.append(node_name)
        self.node_latency_sec[node_name] = round(self.node_latency_sec.get(node_name, 0.0) + latency, 4)
        self.node_visit_count[node_name] = self.node_visit_count.get(node_name, 0) + 1
        self.total_time = round(self.total_time + latency, 4)

        agent_key = _A2A_NODE_TO_AGENT.get(node_name)
        if agent_key:
            self.a2a_latency_sec[agent_key] = round(self.a2a_latency_sec.get(agent_key, 0.0) + latency, 4)

        if node_name == "reason_node":
            self.vlm_latency_sec = round(self.vlm_latency_sec + latency, 4)
            module = str((state_update.get("decision", {}) or {}).get("call_module", "") or "")
            self.reason_actions.append(module)

        if node_name == START_NODE:
            item_info = state_update.get("item_info", {}) or {}
            groups = item_info.get("group_ranking", []) or []
            self.ranked_goal_count = max(self.ranked_goal_count, len(groups))
            if groups and isinstance(groups[0], dict):
                self.goal_pose_mode = str(groups[0].get("selection_mode", "") or "") or self.goal_pose_mode

        if node_name == "nav_move_node":
            navigation = self._latest.get("navigation", {}) or {}
            rank = int(navigation.get("current_goal_rank", 1) or 1)
            self._tried_ranks.add(rank)
            self.nav_time_sec = round(self.nav_time_sec + latency, 4)
            nav_result = navigation.get("result", {}) or {}
            if nav_result.get("arrived") is True:
                self.nav_success = True

        if node_name == "car_approach_node":
            approach = state_update.get("approach_result", {}) or {}
            if approach.get("success") is True:
                navigation = self._latest.get("navigation", {}) or {}
                self.success_goal_rank = int(navigation.get("current_goal_rank", 1) or 1)

        if node_name == "update_memory_node":
            self.memory_update_count += 1

    # ----------------------------------------------------------------- summary
    def _grasp_metrics(self) -> Dict[str, Any]:
        grasp = self._latest.get("grasp_result", {}) or {}
        candidate = grasp.get("num_candidate_grasps")
        valid = grasp.get("num_valid_grasps")
        return {
            "grasp_success": bool(grasp.get("success", False)),
            "grasp_pose_ready": bool(grasp.get("best_grasp_pose_camera")),
            "grasp_candidate_count": {
                "num_candidate_grasps": candidate,
                "num_valid_grasps": valid,
            },
        }

    def _approach_metrics(self) -> Dict[str, Any]:
        approach = self._latest.get("approach_result", {}) or {}
        arm_result = approach.get("arm_result", {}) if isinstance(approach.get("arm_result", {}), dict) else {}
        return {
            "approach_success": bool(approach.get("success", False)),
            "arm_success": bool(arm_result.get("success", False)),
        }

    def _task_success(self) -> bool:
        approach = self._latest.get("approach_result", {}) or {}
        return approach.get("success") is True

    def _final_status(self, task_success: bool) -> str:
        if task_success:
            return "SUCCESS"
        current_status = str(self._latest.get("current_status", ""))
        if not self._tried_ranks and not self._latest.get("item_info"):
            return "NO_VALID_GOAL"
        if current_status in {"MAJOR_NAV_EXHAUSTED"}:
            return "GRASP_FAILED" if self._latest.get("grasp_result") else "FAILED"
        return "FAILED"

    def _failure_reason(self, task_success: bool) -> str:
        if task_success:
            return ""
        approach = self._latest.get("approach_result", {}) or {}
        grasp = self._latest.get("grasp_result", {}) or {}
        current_status = str(self._latest.get("current_status", ""))
        if not self._latest.get("item_info"):
            return "no valid goal pose (could not project target center)"
        if self._tried_ranks and not self.nav_success:
            return "navigation failed"
        if approach and approach.get("success") is False:
            phase = str(approach.get("phase", "") or approach.get("status_code", "") or "approach failed")
            return f"approach failed ({phase})"
        if grasp and grasp.get("success") is False:
            return "grasp failed"
        if current_status == "MAJOR_NAV_EXHAUSTED":
            return "target not graspable from any candidate goal pose"
        return "unknown failure"

    def to_dict(self) -> Dict[str, Any]:
        task = self._latest.get("task", {}) or {}
        requested = self._latest.get("requested_object", {}) or {}
        task_instruction = (
            task.get("original_user_request")
            or task.get("normalized_task")
            or requested.get("label")
            or requested.get("id")
            or ""
        )
        task_success = self._task_success()
        report: Dict[str, Any] = {
            "experiment": "A3",
            "ablation": "no_item_info_fixed_offset_feasible_goals",
            "ablation_description": (
                "remove object size / yaw / grasp-direction grouping / ranked goal poses; "
                "push a fixed stand-off (default 0.6 m) out from the target center in 8 directions "
                "and keep only the points with no black/occupied cell within the robot radius "
                "(default 0.25 m) on the ROS keepout map (white+gray allowed; infeasible directions dropped)"
            ),
            "mode": self.mode,
            "context_id": self.context_id,
            "scene_id": self.scene_id,
            "task_instruction": task_instruction,
            "start_node": START_NODE,
            "end_node": END_NODE,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            # --- 10 core metrics ------------------------------------------------
            "task_success": task_success,
            "total_time_from_get_item_info_to_END": round(self.total_time, 4),
            "node_sequence": list(self.node_sequence),
            "node_latency_sec": dict(self.node_latency_sec),
            "node_visit_count": dict(self.node_visit_count),
            "reason_actions": list(self.reason_actions),
            "ranked_goal_count": self.ranked_goal_count,
            "tried_goal_count": len(self._tried_ranks),
            "success_goal_rank": self.success_goal_rank,
            # --- supporting metrics --------------------------------------------
            "vlm_latency_sec": self.vlm_latency_sec,
            "a2a_latency_sec": dict(self.a2a_latency_sec),
            "nav_success": self.nav_success,
            "nav_time_sec": self.nav_time_sec,
            "memory_update_count": self.memory_update_count,
            "goal_pose_mode": self.goal_pose_mode or "ablation_fixed_offset",
            "ablation_goal_settings": _ablation_goal_settings(),
        }
        report.update(self._grasp_metrics())
        report.update(self._approach_metrics())
        report["final_status"] = self._final_status(task_success)
        report["failure_reason"] = self._failure_reason(task_success)
        return report

    # ------------------------------------------------------------------- write
    def write(self) -> str:
        """Append this run to the shared result file with the next ``A3_NNN`` id.

        Returns an empty string if the run never reached ``START_NODE`` (e.g. a
        chat-only turn), so callers can skip empty reports.
        """
        if not self._started:
            return ""
        try:
            self.result_path.parent.mkdir(parents=True, exist_ok=True)
            results = _load_results(self.result_path)
            experiment_id = _next_experiment_id("A3", results)
            results.append({"experiment_id": experiment_id, **self.to_dict()})
            with open(self.result_path, "w", encoding="utf-8") as handle:
                json.dump(results, handle, ensure_ascii=False, indent=2)
            # Keep results editable by the host user even when this runs as
            # root inside Docker (best-effort; ignore if we are not the owner).
            try:
                os.chmod(self.result_path.parent, 0o777)
                os.chmod(self.result_path, 0o666)
            except OSError:
                pass
            return str(self.result_path)
        except OSError as exc:  # pragma: no cover - disk failures only
            return f"<failed to write Experiment A3 result: {exc}>"


def _ablation_goal_settings() -> Dict[str, Any]:
    """Snapshot of the A3 candidate goal-pose settings for the report."""
    def _f(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)) or default)
        except (TypeError, ValueError):
            return default

    return {
        "mode": str(os.getenv("ABLATION_GOAL_MODE", "fixed_offset") or "fixed_offset").strip().lower(),
        "directions": max(1, int(_f("ABLATION_GOAL_DIRECTIONS", 8))),
        "start_deg": _f("ABLATION_GOAL_START_DEG", 0.0),
        "distance_m": _f("ABLATION_GOAL_DISTANCE_M", 0.6),
        "robot_radius_m": _f("ABLATION_ROBOT_RADIUS_M", 0.25),
        "black_threshold": int(_f("ABLATION_BLACK_THRESHOLD", 50)),
        "sector_centroid_r_min_m": _f("ABLATION_GOAL_R_MIN_M", 0.4),
        "sector_centroid_r_max_m": _f("ABLATION_GOAL_R_MAX_M", 1.5),
    }
