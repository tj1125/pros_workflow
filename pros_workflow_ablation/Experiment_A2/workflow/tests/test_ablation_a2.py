"""Experiment A2 tests: single-nearest-goal ablation + execution recorder.

These intentionally avoid importing the LangGraph orchestrator so they run with
nothing heavier than pydantic. The orchestrator-level routing/guard behaviour for
the collapsed (single-candidate) ranking is already covered by
``test_decision_safety_guard...`` and ``test_langgraph_route_matrix`` in
``test_refactor_contracts.py``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from commander.ablation import collapse_to_nearest_goal, single_nearest_goal_enabled
from commander.contracts import NavigationState, dump_model
from commander.experiment_recorder import ExperimentRecorder


def _group(rank: int, goal_xy, confidence=0.5, orientation_group=0) -> dict:
    return {
        "rank": rank,
        "orientation_group": orientation_group,
        "best_goal_pose_ros_map": list(goal_xy),
        "best_confidence": confidence,
        "map_feasible": True,
        "selection_mode": "free_map",
    }


# --------------------------------------------------------------------- ablation


def test_collapse_picks_nearest_docking_pose() -> None:
    # Target at map (0,0). rank 1 is far (confidence-best), rank 3 is nearest.
    groups = [
        _group(1, (5.0, 0.0), confidence=0.9),
        _group(2, (3.0, 4.0), confidence=0.7),
        _group(3, (0.5, 0.0), confidence=0.4),
    ]
    collapsed, meta = collapse_to_nearest_goal(groups, (0.0, 0.0))

    assert len(collapsed) == 1
    assert meta["applied"] is True
    assert meta["perception_group_count"] == 3
    assert meta["dropped_group_count"] == 2
    assert meta["selected_perception_rank"] == 3
    assert abs(meta["selected_distance_m"] - 0.5) < 1e-6
    # Surviving candidate is relabeled rank 1 so the rest of the workflow treats
    # it as the only goal pose, but keeps its docking pose and provenance.
    assert collapsed[0]["rank"] == 1
    assert collapsed[0]["best_goal_pose_ros_map"] == [0.5, 0.0]
    assert collapsed[0]["ablation_original_rank"] == 3
    assert collapsed[0]["ablation_single_nearest_goal"] is True


def test_collapse_handles_empty_and_missing_target() -> None:
    empty, meta = collapse_to_nearest_goal([], (0.0, 0.0))
    assert empty == []
    assert meta["applied"] is False

    groups = [_group(1, (5.0, 0.0)), _group(2, (1.0, 0.0))]
    collapsed, meta = collapse_to_nearest_goal(groups, (None, None))
    # No target XY → fall back to the perception's own rank 1.
    assert len(collapsed) == 1
    assert meta["selected_perception_rank"] == 1
    assert meta["selection_metric"] == "fallback_perception_rank_1_no_target_xy"


def test_single_nearest_goal_flag_reads_env() -> None:
    saved = os.environ.get("ABLATION_SINGLE_NEAREST_GOAL")
    try:
        os.environ.pop("ABLATION_SINGLE_NEAREST_GOAL", None)
        assert single_nearest_goal_enabled() is False  # default OFF keeps baseline intact
        os.environ["ABLATION_SINGLE_NEAREST_GOAL"] = "true"
        assert single_nearest_goal_enabled() is True
        os.environ["ABLATION_SINGLE_NEAREST_GOAL"] = "0"
        assert single_nearest_goal_enabled() is False
    finally:
        if saved is None:
            os.environ.pop("ABLATION_SINGLE_NEAREST_GOAL", None)
        else:
            os.environ["ABLATION_SINGLE_NEAREST_GOAL"] = saved


def test_navigation_state_carries_ablation_metadata() -> None:
    nav = NavigationState(current_goal_rank=1, ablation={"applied": True, "perception_group_count": 4})
    dumped = dump_model(nav)
    assert dumped["ablation"]["applied"] is True
    assert dumped["ablation"]["perception_group_count"] == 4
    # Baseline keeps the field present but empty.
    assert dump_model(NavigationState())["ablation"] == {}


# --------------------------------------------------------------------- recorder


def _merge(state: dict, update: dict) -> dict:
    """Mirror SessionMemoryStore._merge_state_update for the fields we exercise."""
    for key, value in update.items():
        if key == "history_buffer":
            state[key] = list(state.get(key, [])) + list(value)
        elif key == "navigation" and isinstance(value, dict):
            merged = dict(state.get("navigation", {}) or {})
            merged.update(value)
            state[key] = merged
        else:
            state[key] = value
    return state


def _exec(node: str, status: str, latency: float, success: bool = True) -> dict:
    return {"node_name": node, "status": status, "success": success, "latency_sec": latency}


def _drive(recorder: ExperimentRecorder, events: list[tuple[str, dict]]) -> dict:
    state: dict = {}
    for node, update in events:
        _merge(state, update)
        recorder.observe(node, update, state)
    return state


def _success_events() -> list[tuple[str, dict]]:
    return [
        (
            "get_item_info_no_sam3d_node",
            {
                "current_status": "ITEM_INFO_NO_SAM3D_READY",
                "task": {"normalized_task": "幫我拿桌子旁邊的熊"},
                "item_info": {"group_ranking": [_group(1, (0.5, 0.0))]},
                "navigation": {
                    "current_goal_rank": 1,
                    "nav_goal": {"x": 0.5, "y": 0.0, "goal_rank": 1},
                    "ablation": {"applied": True, "perception_group_count": 4, "selected_perception_rank": 3},
                },
                "last_execution": _exec("get_item_info_no_sam3d_node", "ITEM_INFO_NO_SAM3D_READY", 8.42),
            },
        ),
        (
            "nav_move_node",
            {
                "current_status": "NAV_COMPLETED",
                "navigation": {"result": {"arrived": True, "message": "arrived"}},
                "last_execution": _exec("nav_move_node", "NAV_COMPLETED", 15.73),
            },
        ),
        ("observe_node", {"current_status": "OBSERVED", "last_execution": _exec("observe_node", "OBSERVED", 1.12)}),
        (
            "reason_node",
            {
                "current_status": "REASONED",
                "decision": {"call_module": "grasp_agent", "latency_sec": 3.85},
                "last_execution": _exec("reason_node", "REASONED", 3.85),
            },
        ),
        (
            "car_grasp_node",
            {
                "current_status": "GRASP_READY",
                "grasp_result": {"success": True, "best_grasp_pose_camera": {"x": 1}, "num_candidate_grasps": 20, "num_valid_grasps": 5},
                "last_execution": _exec("car_grasp_node", "GRASP_READY", 9.64),
            },
        ),
        (
            "car_approach_node",
            {
                "current_status": "APPROACH_COMPLETED",
                "approach_result": {"success": True, "phase": "", "arm_result": {"success": True}},
                "last_execution": _exec("car_approach_node", "APPROACH_COMPLETED", 3.55),
            },
        ),
        (
            "nav_home_node",
            {"current_status": "NAV_HOME_COMPLETED", "task_complete": True, "last_execution": _exec("nav_home_node", "NAV_HOME_COMPLETED", 2.0)},
        ),
        ("goodbye_node", {"current_status": "GOODBYE", "last_execution": _exec("goodbye_node", "GOODBYE", 0.01)}),
    ]


def test_recorder_success_report() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        recorder = ExperimentRecorder(context_id="ctxsuccess", output_dir=Path(tmp), enabled=True)
        _drive(recorder, _success_events())

        results = json.loads((Path(tmp) / "result.json").read_text(encoding="utf-8"))
        assert len(results) == 1
        report = results[0]
        assert report["experiment_id"] == "A2_001"

        assert report["task_success"] is True
        assert report["final_status"] == "SUCCESS"
        assert report["failure_reason"] == ""
        assert report["start_node"] == "get_item_info_no_sam3d_node"
        assert report["node_sequence"][0] == "get_item_info_no_sam3d_node"
        assert report["node_sequence"][-1] == "END"
        assert abs(report["total_time_from_get_item_info_to_END"] - (8.42 + 15.73 + 1.12 + 3.85 + 9.64 + 3.55 + 2.0 + 0.01)) < 1e-6
        assert report["reason_actions"] == ["grasp_agent"]
        assert report["ranked_goal_count"] == 1
        assert report["tried_goal_count"] == 1
        assert report["success_goal_rank"] == 1
        assert report["a2a_latency_sec"] == {"get_item_info_agent": 8.42, "grasp_agent": 9.64}
        assert report["vlm_latency_sec"] == 3.85
        assert report["nav_success"] is True
        assert report["nav_time_sec"] == 15.73
        assert report["grasp_candidate_count"]["num_candidate_grasps"] == 20
        assert report["arm_success"] is True
        assert report["memory_update_count"] == 0
        assert report["experiment"] == "A2"
        assert report["mode"] == "single_nearest_goal_pose"
        assert report["ablation_selection"]["selected_perception_rank"] == 3
        assert report["node_visit_count"]["get_item_info_no_sam3d_node"] == 1

        # A second finished task appends A2_002 to the same result file.
        recorder2 = ExperimentRecorder(context_id="ctxsuccess2", output_dir=Path(tmp), enabled=True)
        _drive(recorder2, _success_events())
        results = json.loads((Path(tmp) / "result.json").read_text(encoding="utf-8"))
        assert [entry["experiment_id"] for entry in results] == ["A2_001", "A2_002"]


def test_recorder_grasp_failure_report() -> None:
    events = [
        (
            "get_item_info_no_sam3d_node",
            {
                "current_status": "ITEM_INFO_NO_SAM3D_READY",
                "item_info": {"group_ranking": [_group(1, (0.5, 0.0))]},
                "navigation": {"current_goal_rank": 1, "nav_goal": {"goal_rank": 1}, "ablation": {"applied": True, "perception_group_count": 4}},
                "last_execution": _exec("get_item_info_no_sam3d_node", "ITEM_INFO_NO_SAM3D_READY", 8.0),
            },
        ),
        ("nav_move_node", {"current_status": "NAV_COMPLETED", "navigation": {"result": {"arrived": True}}, "last_execution": _exec("nav_move_node", "NAV_COMPLETED", 12.0)}),
        ("observe_node", {"current_status": "OBSERVED", "last_execution": _exec("observe_node", "OBSERVED", 1.0)}),
        ("reason_node", {"current_status": "REASONED", "decision": {"call_module": "grasp_agent", "latency_sec": 3.0}, "last_execution": _exec("reason_node", "REASONED", 3.0)}),
        ("car_grasp_node", {"current_status": "GRASP_FAILED", "grasp_result": {"success": False}, "last_execution": _exec("car_grasp_node", "GRASP_FAILED", 7.0, success=False)}),
        # grasp produced no pose ⇒ approach also runs and fails, but grasp is the root cause.
        ("car_approach_node", {"current_status": "APPROACH_FAILED", "approach_result": {"success": False, "phase": "no_grasp_pose"}, "last_execution": _exec("car_approach_node", "APPROACH_FAILED", 1.0, success=False)}),
        ("update_memory_node", {"current_status": "MEMORY_UPDATED", "last_execution": _exec("update_memory_node", "MEMORY_UPDATED", 0.1)}),
        ("observe_node", {"current_status": "OBSERVED", "last_execution": _exec("observe_node", "OBSERVED", 1.0)}),
        ("reason_node", {"current_status": "REASONED", "decision": {"call_module": "DONE", "latency_sec": 2.0}, "last_execution": _exec("reason_node", "REASONED", 2.0)}),
        ("nav_home_node", {"current_status": "NAV_HOME_COMPLETED", "task_complete": True, "last_execution": _exec("nav_home_node", "NAV_HOME_COMPLETED", 2.0)}),
        ("goodbye_node", {"current_status": "GOODBYE", "last_execution": _exec("goodbye_node", "GOODBYE", 0.01)}),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        recorder = ExperimentRecorder(context_id="ctxgraspfail", output_dir=Path(tmp), enabled=True)
        _drive(recorder, events)
        report = json.loads((Path(tmp) / "result.json").read_text(encoding="utf-8"))[0]

    assert report["task_success"] is False
    assert report["final_status"] == "GRASP_FAILED"
    assert report["failure_reason"] == "grasp_failed"
    assert report["success_goal_rank"] is None
    assert report["tried_goal_count"] == 1
    assert report["memory_update_count"] == 1
    assert report["node_visit_count"]["reason_node"] == 2
    assert report["reason_actions"] == ["grasp_agent", "DONE"]


def test_recorder_baseline_multi_candidate_rank2_success() -> None:
    """Un-ablated baseline: rank 1 fails, fallback to rank 2 which succeeds."""
    events = [
        (
            "get_item_info_no_sam3d_node",
            {
                "current_status": "ITEM_INFO_NO_SAM3D_READY",
                "item_info": {"group_ranking": [_group(1, (5.0, 0.0)), _group(2, (3.0, 0.0)), _group(3, (1.0, 0.0)), _group(4, (0.2, 0.0))]},
                "navigation": {"current_goal_rank": 1, "nav_goal": {"goal_rank": 1}, "ablation": {}},
                "last_execution": _exec("get_item_info_no_sam3d_node", "ITEM_INFO_NO_SAM3D_READY", 8.0),
            },
        ),
        ("nav_move_node", {"current_status": "NAV_COMPLETED", "navigation": {"result": {"arrived": True}}, "last_execution": _exec("nav_move_node", "NAV_COMPLETED", 12.0)}),
        ("observe_node", {"current_status": "OBSERVED", "last_execution": _exec("observe_node", "OBSERVED", 1.0)}),
        ("reason_node", {"current_status": "REASONED", "decision": {"call_module": "grasp_agent", "latency_sec": 3.0}, "last_execution": _exec("reason_node", "REASONED", 3.0)}),
        ("car_grasp_node", {"current_status": "GRASP_READY", "grasp_result": {"success": True, "best_grasp_pose_camera": {"x": 1}}, "last_execution": _exec("car_grasp_node", "GRASP_READY", 7.0)}),
        ("car_approach_node", {"current_status": "APPROACH_FAILED", "approach_result": {"success": False, "phase": "ik_failed"}, "last_execution": _exec("car_approach_node", "APPROACH_FAILED", 2.0, success=False)}),
        ("update_memory_node", {"current_status": "MEMORY_UPDATED", "last_execution": _exec("update_memory_node", "MEMORY_UPDATED", 0.1)}),
        ("observe_node", {"current_status": "OBSERVED", "last_execution": _exec("observe_node", "OBSERVED", 1.0)}),
        ("reason_node", {"current_status": "REASONED", "decision": {"call_module": "major_nav_node", "latency_sec": 3.0}, "last_execution": _exec("reason_node", "REASONED", 3.0)}),
        # major_nav advances to rank 2; reset grasp/approach in real graph (not needed for the recorder).
        ("major_nav_node", {"current_status": "MAJOR_NAV_CONTEXT_READY", "navigation": {"current_goal_rank": 2, "nav_goal": {"goal_rank": 2}}, "last_execution": _exec("major_nav_node", "MAJOR_NAV_CONTEXT_READY", 0.2)}),
        ("nav_move_node", {"current_status": "NAV_COMPLETED", "navigation": {"result": {"arrived": True}}, "last_execution": _exec("nav_move_node", "NAV_COMPLETED", 11.0)}),
        ("update_memory_node", {"current_status": "MEMORY_UPDATED", "last_execution": _exec("update_memory_node", "MEMORY_UPDATED", 0.1)}),
        ("observe_node", {"current_status": "OBSERVED", "last_execution": _exec("observe_node", "OBSERVED", 1.0)}),
        ("reason_node", {"current_status": "REASONED", "decision": {"call_module": "grasp_agent", "latency_sec": 3.0}, "last_execution": _exec("reason_node", "REASONED", 3.0)}),
        ("car_grasp_node", {"current_status": "GRASP_READY", "grasp_result": {"success": True, "best_grasp_pose_camera": {"x": 1}}, "last_execution": _exec("car_grasp_node", "GRASP_READY", 7.0)}),
        ("car_approach_node", {"current_status": "APPROACH_COMPLETED", "approach_result": {"success": True, "arm_result": {"success": True}}, "last_execution": _exec("car_approach_node", "APPROACH_COMPLETED", 3.0)}),
        ("nav_home_node", {"current_status": "NAV_HOME_COMPLETED", "task_complete": True, "last_execution": _exec("nav_home_node", "NAV_HOME_COMPLETED", 2.0)}),
        ("goodbye_node", {"current_status": "GOODBYE", "last_execution": _exec("goodbye_node", "GOODBYE", 0.01)}),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        recorder = ExperimentRecorder(context_id="ctxbaseline", output_dir=Path(tmp), enabled=True)
        _drive(recorder, events)
        report = json.loads((Path(tmp) / "result.json").read_text(encoding="utf-8"))[0]

    assert report["task_success"] is True
    assert report["ranked_goal_count"] == 4
    assert report["tried_goal_count"] == 2
    assert report["success_goal_rank"] == 2
    assert report["mode"] == "full_ranked_candidates"
    assert report["reason_actions"] == ["grasp_agent", "major_nav_node", "grasp_agent"]
    assert report["memory_update_count"] == 2


def test_recorder_disabled_writes_nothing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        recorder = ExperimentRecorder(context_id="ctxoff", output_dir=Path(tmp), enabled=False)
        _drive(recorder, _success_events())
        assert not (Path(tmp) / "result.json").exists()


def run_all() -> None:
    test_collapse_picks_nearest_docking_pose()
    test_collapse_handles_empty_and_missing_target()
    test_single_nearest_goal_flag_reads_env()
    test_navigation_state_carries_ablation_metadata()
    test_recorder_success_report()
    test_recorder_grasp_failure_report()
    test_recorder_baseline_multi_candidate_rank2_success()
    test_recorder_disabled_writes_nothing()


if __name__ == "__main__":
    run_all()
    print(json.dumps({"ok": True, "tests": 8}, ensure_ascii=False))
