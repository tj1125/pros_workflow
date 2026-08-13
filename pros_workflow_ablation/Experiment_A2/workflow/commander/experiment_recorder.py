"""Experiment A2 execution recorder.

Records one complete execution trace per pick task — from the first
``get_item_info_no_sam3d_node`` up to graph END (``goodbye_node``) — and appends
one entry to the shared result file ``<workflow>/result/result.json`` (each task
gets the next ``A2_NNN`` id). The report captures the metrics needed to evaluate
the "ranked goal poses → single nearest goal pose" ablation: task success, timing,
node path/visits, VLM decisions, A2A latencies, candidate-goal-pose usage and the
failure stage.

The recorder is a passive observer of LangGraph node updates: both ``main.py`` and
the web app funnel every node update through :class:`SessionMemoryStore`, which
calls :meth:`ExperimentRecorder.observe` after merging the update into the running
state. It never raises into the workflow — any internal error is logged and
swallowed so a recording bug can never break a robot run.

It also works for the un-ablated baseline (flag off): there ``ranked_goal_count``
reflects the real number of perception candidates and ``tried_goal_count`` /
``success_goal_rank`` show how far down the ranking the system had to go, which is
exactly the comparison the ablation is meant to surface.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_COMMANDER_DIR = Path(__file__).resolve().parent
_WORKFLOW_ROOT = _COMMANDER_DIR.parent


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def experiment_recording_enabled() -> bool:
    return _env_flag("EXPERIMENT_RECORD_ENABLED", True)


RESULT_FILENAME = "result.json"


def _resolve_output_dir() -> Path:
    configured = os.getenv("EXPERIMENT_A2_RESULT_DIR", "").strip()
    if not configured:
        return _WORKFLOW_ROOT / "result"
    path = Path(configured)
    return path if path.is_absolute() else (_WORKFLOW_ROOT / path)


def _load_results(path: Path) -> list:
    """Read the accumulated result list; tolerate a missing/corrupt file."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _next_experiment_id(prefix: str, existing: list) -> str:
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


class ExperimentRecorder:
    """Accumulate one pick-task execution trace and emit a JSON report at END."""

    START_NODE = "get_item_info_no_sam3d_node"
    TERMINAL_NODES = {"goodbye_node"}

    def __init__(
        self,
        *,
        context_id: str,
        output_dir: Optional[Path] = None,
        experiment_prefix: Optional[str] = None,
        scene_id: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.context_id = str(context_id or "")
        self.enabled = experiment_recording_enabled() if enabled is None else bool(enabled)
        self.output_dir = Path(output_dir) if output_dir is not None else _resolve_output_dir()
        self.experiment_prefix = experiment_prefix or os.getenv("EXPERIMENT_ID_PREFIX", "A2")
        self.scene_id = scene_id if scene_id is not None else os.getenv("EXPERIMENT_SCENE_ID", "")
        self._task_seq = 0
        self._reset_task_state()

    # ------------------------------------------------------------------ public

    def observe(self, node_name: str, state_update: Dict[str, Any], current_state: Dict[str, Any]) -> None:
        """Record a single LangGraph node update. Never raises."""
        if not self.enabled:
            return
        try:
            self._observe(node_name, state_update or {}, current_state or {})
        except Exception:  # pragma: no cover - defensive: must not break a run
            logger.warning("[ExperimentRecorder] observe failed for node=%s", node_name, exc_info=True)

    def finalize(self) -> Optional[Dict[str, Any]]:
        """Write the report for the current task if one is in progress. Never raises."""
        if not self.enabled or not self._active or self._finalized:
            return None
        try:
            self._finalized = True
            self._wall_end = time.time()
            report = self._build_report()
            self._write_report(report)
            self._reset_task_state()
            return report
        except Exception:  # pragma: no cover - defensive
            logger.warning("[ExperimentRecorder] finalize failed", exc_info=True)
            self._reset_task_state()
            return None

    # ----------------------------------------------------------------- internal

    def _reset_task_state(self) -> None:
        self._active = False
        self._finalized = False
        self._wall_start: Optional[float] = None
        self._wall_end: Optional[float] = None
        self._experiment_id = ""
        self._started_iso = ""
        self._node_sequence: list[str] = []
        self._node_latency: Dict[str, float] = {}
        self._node_visits: Dict[str, int] = {}
        self._node_timeline: list[Dict[str, Any]] = []
        self._total_latency = 0.0
        self._reason_actions: list[str] = []
        self._vlm_latency = 0.0
        self._a2a_latency: Dict[str, float] = {"get_item_info_agent": 0.0, "grasp_agent": 0.0}
        self._ranked_goal_count = 0
        self._ablation: Dict[str, Any] = {}
        self._tried_ranks: set[int] = set()
        self._nav_latency = 0.0
        self._nav_arrived_ever = False
        self._nav_last_arrived: Optional[bool] = None
        self._grasp_attempted = False
        self._grasp_success: Optional[bool] = None
        self._grasp_pose_ready: Optional[bool] = None
        self._grasp_candidate_count: Optional[int] = None
        self._grasp_valid_count: Optional[int] = None
        self._approach_attempted = False
        self._approach_success: Optional[bool] = None
        self._approach_phase = ""
        self._approach_status_code = ""
        self._approach_message = ""
        self._arm_success: Optional[bool] = None
        self._memory_update_count = 0
        self._success_goal_rank: Optional[int] = None
        self._statuses: list[str] = []
        self._last_status = ""
        self._task_instruction = ""

    def _begin_task(self) -> None:
        self._reset_task_state()
        self._active = True
        self._task_seq += 1
        self._experiment_id = f"{self.experiment_prefix}_{self._task_seq:03d}"
        self._wall_start = time.time()
        self._started_iso = datetime.now(timezone.utc).isoformat()

    def _observe(self, node_name: str, state_update: Dict[str, Any], current_state: Dict[str, Any]) -> None:
        if node_name in ("", "__interrupt__"):
            return
        if not self._active:
            if node_name != self.START_NODE:
                return
            self._begin_task()

        exec_info = state_update.get("last_execution") or current_state.get("last_execution") or {}
        latency = float(exec_info.get("latency_sec", 0.0) or 0.0)
        status = str(
            exec_info.get("status")
            or state_update.get("current_status")
            or current_state.get("current_status")
            or ""
        )
        success = bool(exec_info.get("success", True))

        self._node_sequence.append(node_name)
        self._node_latency[node_name] = round(self._node_latency.get(node_name, 0.0) + latency, 4)
        self._node_visits[node_name] = self._node_visits.get(node_name, 0) + 1
        self._total_latency += latency
        self._node_timeline.append(
            {
                "step": len(self._node_timeline) + 1,
                "node": node_name,
                "status": status,
                "success": success,
                "latency_sec": round(latency, 4),
            }
        )
        if status:
            self._statuses.append(status)
            self._last_status = status

        task = current_state.get("task") or {}
        if task and not self._task_instruction:
            self._task_instruction = str(task.get("normalized_task") or task.get("original_user_request") or "")

        navigation = current_state.get("navigation") or {}
        item_info = current_state.get("item_info") or {}

        if node_name == self.START_NODE:
            self._a2a_latency["get_item_info_agent"] = round(self._a2a_latency["get_item_info_agent"] + latency, 4)
            groups = item_info.get("group_ranking")
            self._ranked_goal_count = len(groups) if isinstance(groups, list) else 0
            ablation = navigation.get("ablation")
            if isinstance(ablation, dict) and ablation:
                self._ablation = dict(ablation)
        elif node_name == "reason_node":
            decision = current_state.get("decision") or {}
            module = str(decision.get("call_module") or "")
            if module:
                self._reason_actions.append(module)
            self._vlm_latency = round(self._vlm_latency + float(decision.get("latency_sec", 0.0) or 0.0), 4)
        elif node_name == "nav_move_node":
            nav_goal = navigation.get("nav_goal") or {}
            rank = int(navigation.get("current_goal_rank", nav_goal.get("goal_rank", 1) if isinstance(nav_goal, dict) else 1) or 1)
            self._tried_ranks.add(rank)
            self._nav_latency = round(self._nav_latency + latency, 4)
            nav_result = navigation.get("result") or {}
            arrived = bool(nav_result.get("arrived", False))
            self._nav_last_arrived = arrived
            if arrived:
                self._nav_arrived_ever = True
        elif node_name == "car_grasp_node":
            self._a2a_latency["grasp_agent"] = round(self._a2a_latency["grasp_agent"] + latency, 4)
            self._grasp_attempted = True
            grasp = current_state.get("grasp_result") or {}
            self._grasp_success = bool(grasp.get("success", False))
            self._grasp_pose_ready = bool(grasp.get("best_grasp_pose_camera"))
            self._grasp_candidate_count = grasp.get("num_candidate_grasps")
            self._grasp_valid_count = grasp.get("num_valid_grasps")
        elif node_name == "car_approach_node":
            self._approach_attempted = True
            approach = current_state.get("approach_result") or {}
            self._approach_success = bool(approach.get("success", False))
            self._approach_phase = str(approach.get("phase", "") or "")
            self._approach_status_code = str(approach.get("status_code", "") or "")
            self._approach_message = str(approach.get("message", "") or "")
            arm = approach.get("arm_result")
            self._arm_success = bool(arm.get("success", False)) if isinstance(arm, dict) else None
            if self._approach_success:
                self._success_goal_rank = int(navigation.get("current_goal_rank", 1) or 1)
        elif node_name == "update_memory_node":
            self._memory_update_count += 1

        if node_name in self.TERMINAL_NODES:
            self.finalize()

    def _derive_outcome(self) -> tuple[bool, str, str]:
        """Return (task_success, final_status, failure_reason)."""
        had_goal = bool(self._tried_ranks) or self._grasp_attempted
        if self._approach_success:
            return True, "SUCCESS", ""
        if self._grasp_attempted and self._grasp_success is False:
            # No usable grasp pose ⇒ approach can never succeed; grasp is the root cause.
            return False, "GRASP_FAILED", "grasp_failed"
        if self._approach_attempted and self._approach_success is False:
            phase = (self._approach_phase or self._approach_status_code or "").lower()
            if "ik" in phase:
                reason = "ik_failed"
            elif "arm" in phase:
                reason = "arm_failed"
            else:
                reason = f"approach_failed:{self._approach_phase or self._approach_status_code or 'unknown'}"
            return False, "APPROACH_FAILED", reason
        if "ITEM_INFO_NO_SAM3D_FAILED" in self._statuses:
            return False, "ITEM_INFO_FAILED", "item_info_failed"
        if any(s in self._statuses for s in ("TARGET_LOST_IN_WORLD_POSITION", "TARGET_LOST")):
            return False, "TARGET_LOST", "target_not_visible"
        if self._tried_ranks and not self._nav_arrived_ever:
            return False, "NAV_FAILED", "navigation_failed"
        if "MAJOR_NAV_EXHAUSTED" in self._statuses:
            return False, "NO_VALID_GOAL", "all_ranked_goals_exhausted"
        if not had_goal:
            return False, "NO_VALID_GOAL", "no_valid_goal_pose"
        return False, "FAILED", "unknown"

    def _build_report(self) -> Dict[str, Any]:
        # Same schema as the A0/A1/A3 ExperimentReport: A0/A1 base fields plus the
        # ablation-specific detail kept in extra fields (mirrors A3's
        # ablation_description / ablation_goal_settings). The full ablation dict is
        # preserved verbatim under "ablation_selection".
        task_success, final_status, failure_reason = self._derive_outcome()
        applied = bool(self._ablation.get("applied"))
        mode = "single_nearest_goal_pose" if applied else "full_ranked_candidates"
        return {
            "experiment_id": self._experiment_id,
            "experiment": "A2",
            "ablation": "ranked_goal_poses_single_nearest" if applied else "ranked_goal_poses_full_baseline",
            "ablation_description": (
                "remove multi-candidate stand-off + ranking + fail-and-switch; keep only the "
                "single goal pose whose docking position is nearest the target in the ROS map XY plane"
            ),
            "mode": mode,
            "context_id": self.context_id,
            "scene_id": self.scene_id,
            "task_instruction": self._task_instruction,
            "start_node": self.START_NODE,
            "end_node": "END",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            # --- 10 core metrics ---
            "task_success": task_success,
            "total_time_from_get_item_info_to_END": round(self._total_latency, 4),
            "node_sequence": [*self._node_sequence, "END"],
            "node_latency_sec": dict(self._node_latency),
            "node_visit_count": dict(self._node_visits),
            "reason_actions": list(self._reason_actions),
            "ranked_goal_count": self._ranked_goal_count,
            "tried_goal_count": len(self._tried_ranks),
            "success_goal_rank": self._success_goal_rank,
            # --- supporting metrics ---
            "vlm_latency_sec": round(self._vlm_latency, 4),
            "a2a_latency_sec": dict(self._a2a_latency),
            "nav_success": self._nav_arrived_ever,
            "nav_time_sec": round(self._nav_latency, 4),
            "memory_update_count": self._memory_update_count,
            "ablation_selection": {
                "variable": "ranked_goal_poses",
                "mode": mode,
                "env_flag": "ABLATION_SINGLE_NEAREST_GOAL",
                **self._ablation,
            },
            "grasp_success": self._grasp_success,
            "grasp_pose_ready": self._grasp_pose_ready,
            "grasp_candidate_count": {
                "num_candidate_grasps": self._grasp_candidate_count,
                "num_valid_grasps": self._grasp_valid_count,
            },
            "approach_success": self._approach_success,
            "arm_success": self._arm_success,
            "final_status": final_status,
            "failure_reason": failure_reason,
        }

    def _write_report(self, report: Dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.output_dir / RESULT_FILENAME
        results = _load_results(result_path)
        # Assign the id from what is already stored, so numbering is continuous
        # across tasks, sessions and processes (not just within this recorder).
        report["experiment_id"] = _next_experiment_id(self.experiment_prefix, results)
        results.append(report)
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
        # Keep results editable by the host user even when this runs as root
        # inside Docker (best-effort; ignore if we are not the owner).
        try:
            os.chmod(self.output_dir, 0o777)
            os.chmod(result_path, 0o666)
        except OSError:
            pass
        logger.info(
            "[ExperimentRecorder] appended %s to %s (success=%s, status=%s, tried=%d/%d ranks)",
            report["experiment_id"],
            result_path,
            report["task_success"],
            report["final_status"],
            report["tried_goal_count"],
            report["ranked_goal_count"],
        )
