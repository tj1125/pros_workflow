from __future__ import annotations

import asyncio
import builtins
import json
import sqlite3
import sys
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

from a2a.types import DataPart, Part
from a2a.utils import completed_task, new_agent_parts_message, new_data_artifact

from agents.a2a_adapter import extract_result_payload, require_agent_card_modes
from commander.brain import Brain, BrainDecision
from commander.storage.artifact_store import ArtifactStore
from commander.storage.session_store import SessionMemoryStore
from commander.object_catalog import lexical_related_object_options, load_graspable_objects
from commander.orchestrator import Orchestrator
from commander.state import _append_history, create_initial_state
from commander.web.app import _legacy_prompt_type


def test_history_reducer_appends_all_entries() -> None:
    assert _append_history([{"i": i} for i in range(4)], [{"i": i} for i in range(4, 9)]) == [
        {"i": 0},
        {"i": 1},
        {"i": 2},
        {"i": 3},
        {"i": 4},
        {"i": 5},
        {"i": 6},
        {"i": 7},
        {"i": 8},
    ]


def test_task_classifier_matches_id_label_and_keeps_chat_general() -> None:
    objects = load_graspable_objects()
    assert [obj["id"] for obj in objects] == ["apple", "box", "coffee", "cup", "doll", "gaobear", "hpb", "xbox"]
    assert all(set(obj) == {"id", "label"} for obj in objects)

    router = Orchestrator.__new__(Orchestrator)

    apple = router._mock_task_classification("The human wants to pick up apple.", objects)
    assert apple.intent == "specific_task"

    doll = router._mock_task_classification("請拿褐色小熊玩偶", objects)
    assert doll.intent == "specific_task"

    chat = router._mock_task_classification("hello", objects)
    assert chat.intent == "general_chat"


def test_legacy_prompt_type_only_uses_detection_selection() -> None:
    assert _legacy_prompt_type("請輸入候選照片編號 (1-2) 或 no：") == "detection_selection"
    assert _legacy_prompt_type("請輸入目標物編號 (1-8)：") == "legacy_input"


def test_item_info_uses_all_room1_upload_cameras() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)

    camera_names = orchestrator._item_info_room_camera_names()
    assert camera_names == [
        "Camera_Room1_12",
        "Camera_Room1_13",
        "Camera_Room1_14",
        "Camera_Room1_15",
    ]

    camera_names.append("Camera_Room1_16")
    assert orchestrator._item_info_room_camera_names() == [
        "Camera_Room1_12",
        "Camera_Room1_13",
        "Camera_Room1_14",
        "Camera_Room1_15",
    ]


def test_related_object_options_use_llm_config_indices() -> None:
    objects = load_graspable_objects()
    captured: dict[str, str] = {}

    def _invoke(messages: list[object]) -> SimpleNamespace:
        captured["prompt"] = "\n".join(str(getattr(message, "content", "")) for message in messages)
        return SimpleNamespace(related_object_indices=[5, 6], reasoning="doll-like objects")

    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.use_mock = False
    orchestrator._related_object_model = SimpleNamespace(invoke=_invoke)

    options = asyncio.run(orchestrator._resolve_related_object_options("pick doll", objects))

    assert [option["id"] for option in options[:2]] == ["doll", "gaobear"]
    assert options[1]["match_reason"] == "llm_related_config_description"
    assert "id=" not in captured["prompt"]
    assert "aliases" not in captured["prompt"]


def test_lexical_related_object_options_use_labels_only() -> None:
    objects = load_graspable_objects()
    doll = next(obj for obj in objects if obj.get("id") == "doll")
    id_only_object = next(
        obj for obj in objects
        if str(obj.get("id", "")).casefold() not in str(obj.get("label", "")).casefold()
    )

    assert [option["id"] for option in lexical_related_object_options(str(doll["label"]), objects)] == ["doll"]
    assert lexical_related_object_options(str(id_only_object["id"]), objects) == []


def test_update_item_info_accepts_nearby_instance_id_reassignment() -> None:
    selected = {"item_id": "doll", "instance_id": 2, "instance_key": "doll_2", "center_world": [1.0, 0.2, 3.0]}
    candidates = [
        {"item_id": "doll", "instance_id": 3, "instance_key": "doll_3", "center_world": [1.02, 0.2, 3.01]},
        {"item_id": "doll", "instance_id": 5, "instance_key": "doll_5", "center_world": [1.0, 0.2, 3.0]},
    ]

    refreshed, match_info = Orchestrator._find_selected_candidate(candidates, selected)

    assert refreshed["instance_key"] == "doll_3"
    assert match_info["id_reassigned"] is True
    assert match_info["instance_id_delta"] == 1


def test_update_item_info_rejects_far_or_large_id_delta_candidates() -> None:
    selected = {"item_id": "doll", "instance_id": 2, "instance_key": "doll_2", "center_world": [1.0, 0.2, 3.0]}
    candidates = [
        {"item_id": "doll", "instance_id": 3, "instance_key": "doll_3", "center_world": [1.04, 0.2, 3.0]},
        {"item_id": "doll", "instance_id": 4, "instance_key": "doll_4", "center_world": [1.0, 0.2, 3.0]},
        {"item_id": "gaobear", "instance_id": 3, "instance_key": "gaobear_3", "center_world": [1.0, 0.2, 3.0]},
    ]

    refreshed, match_info = Orchestrator._find_selected_candidate(candidates, selected)

    assert refreshed is None
    assert match_info["match_mode"] == "missing"


def test_item_info_no_sam3d_target_selection_respects_selected_instance() -> None:
    server_root = Path(__file__).resolve().parents[2] / "3090server" / "VLM_RL"
    if str(server_root) not in sys.path:
        sys.path.insert(0, str(server_root))

    from get_item_info_agent_no_sam3d.pipeline.steps.topic_input import (
        parse_world_position_data,
        select_target_object,
        world_position_instance_key,
    )

    objects = parse_world_position_data(
        {
            "doll": [
                {
                    "item": "doll",
                    "id": 237,
                    "world_x": 1.07,
                    "world_y": 0.51,
                    "world_z": 7.58,
                    "camsrc": ["Camera_Room1_12"],
                    "bbox": [[10, 10, 40, 40]],
                },
                {
                    "item": "doll",
                    "id": 238,
                    "world_x": 1.59,
                    "world_y": 0.51,
                    "world_z": 7.52,
                    "camsrc": ["Camera_Room1_12"],
                    "bbox": [[50, 10, 90, 40]],
                },
            ]
        }
    )

    selected = select_target_object(
        objects,
        "doll",
        target_instance_id=238,
        target_instance_key="doll_238",
        target_topic_key="doll",
        target_center_world=[1.59, 0.51, 7.52],
    )

    assert selected.item_id == 238
    assert world_position_instance_key(selected) == "doll_238"


def test_artifact_store_json_text_bytes_and_db_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ArtifactStore(context_id="ctx", base_dir=tmp)
        json_ref = store.save_json("debug_payload", {"hello": "world"}, created_by_node="test")
        text_ref = store.save_text("debug_payload", "hello", created_by_node="test")
        bytes_ref = store.save_bytes("debug_payload", b"abc", created_by_node="test", mime_type="application/octet-stream")

        assert store.load_json(json_ref) == {"hello": "world"}
        assert store.load_text(text_ref) == "hello"
        assert store.load_bytes(bytes_ref) == b"abc"

        db_path = Path(tmp) / "sessions" / "ctx" / "session.sqlite"
        con = sqlite3.connect(db_path)
        rows = con.execute("select artifact_id, sha256, size_bytes from artifacts").fetchall()
        con.close()
        assert {row[0] for row in rows} == {json_ref.artifact_id, text_ref.artifact_id, bytes_ref.artifact_id}
        assert all(row[1] and row[2] >= 3 for row in rows)


def test_artifact_store_register_file_and_clear_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ArtifactStore(context_id="ctx-clear", base_dir=tmp)
        preview_path = store.artifacts_dir / "find_candidates" / "candidate.jpg"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_bytes(b"preview")
        legacy_dir = store.session_dir / "find_candidates"
        legacy_dir.mkdir(parents=True, exist_ok=True)
        (legacy_dir / "old.jpg").write_bytes(b"old")

        ref = store.register_file("find_candidates", preview_path, created_by_node="find_node")
        assert store.resolve_path(ref).exists()

        store.clear_artifacts()

        assert store.artifacts_dir.exists()
        assert not store.resolve_path(ref).exists()
        assert not legacy_dir.exists()
        con = sqlite3.connect(Path(tmp) / "sessions" / "ctx-clear" / "session.sqlite")
        count = con.execute("select count(*) from artifacts").fetchone()[0]
        con.close()
        assert count == 0


def test_world_snapshot_raw_and_goal_pose_are_sqlite_rows() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ArtifactStore(context_id="ctx-db", base_dir=tmp)
        snapshot_id = store.save_world_snapshot_raw(
            {"data": "world-position-json"},
            candidate_count=3,
            selected_instance_key="doll_1",
            created_by_node="find_node",
            update_reason="db_created",
        )
        assert store.load_world_snapshot_raw(snapshot_id) == {"data": "world-position-json"}

        store.record_world_snapshot(
            step=5,
            snapshot_id=snapshot_id,
            candidate_count=3,
            selected_instance_key="doll_1",
            update_source_node="find_node",
            update_reason="db_created",
        )
        store.record_goal_pose(
            step=6,
            source_node="get_item_info_no_sam3d_node",
            rank=1,
            goal_pose_index=0,
            goal={"x": 1.0, "y": 2.0, "qz": 0.0, "qw": 1.0},
            goal_pose_db={"target_instance_key": "doll_1"},
            nav_goal_pose_source="rank_best",
        )

        con = sqlite3.connect(Path(tmp) / "sessions" / "ctx-db" / "session.sqlite")
        world_row = con.execute(
            "select step, raw_payload_json, selected_instance_key, update_reason from world_snapshots where snapshot_id=?",
            (snapshot_id,),
        ).fetchone()
        artifact_rows = con.execute("select count(*) from artifacts where kind='world_position_raw'").fetchone()[0]
        goal_row = con.execute("select goal_json, goal_pose_db_json from goal_poses").fetchone()
        con.close()

        assert world_row[0] == 5
        assert json.loads(world_row[1]) == {"data": "world-position-json"}
        assert world_row[2] == "doll_1"
        assert world_row[3] == "db_created"
        assert artifact_rows == 0
        assert json.loads(goal_row[0])["x"] == 1.0
        assert json.loads(goal_row[1])["target_instance_key"] == "doll_1"


def test_session_store_rejects_raw_state_blobs() -> None:
    state = create_initial_state("ctx-raw")
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionMemoryStore(context_id="ctx-raw", initial_state=state, base_dir=tmp)
        try:
            store.record_event(step=1, node_name="bad_node", state_update={"camera_images": {"cam": "..."}})
        except ValueError as exc:
            assert "Forbidden raw data key" in str(exc)
        else:
            raise AssertionError("raw camera_images state was not rejected")

        try:
            store.record_event(step=2, node_name="bad_node", state_update={"observation": {"image_base64": "..."}})
        except ValueError as exc:
            assert "Forbidden raw data key" in str(exc)
        else:
            raise AssertionError("nested image_base64 state was not rejected")


def test_brain_accepts_only_structured_decisions() -> None:
    decision = Brain._coerce_structured_decision({
        "reasoning": "ready",
        "call_module": "grasp_agent",
        "module_params": {"object_id": "doll"},
    })
    assert isinstance(decision, BrainDecision)
    assert decision.call_module == "grasp_agent"

    fenced = Brain._coerce_structured_decision(
        SimpleNamespace(content='```json\n{"reasoning":"ready","call_module":"grasp_agent","module_params":{}}\n```')
    )
    assert fenced.call_module == "grasp_agent"

    wrapper = Brain._coerce_structured_decision({
        "raw": SimpleNamespace(content='```json\n{"reasoning":"ready","call_module":"grasp_agent","params":{"object_id":"doll"}}\n```'),
        "parsed": None,
        "parsing_error": ValueError("provider parser rejected markdown fences"),
    })
    assert wrapper.module_params == {"object_id": "doll"}

    try:
        Brain._coerce_structured_decision(SimpleNamespace(content="ready to grasp"))
    except ValueError as exc:
        assert "BrainDecision JSON" in str(exc)
    else:
        raise AssertionError("free-form Brain output was accepted")


def test_a2a_card_validation_rejects_wrong_service() -> None:
    wrong_card = SimpleNamespace(
        name="Get Item Info Agent",
        default_input_modes=["text", "data"],
        default_output_modes=["text"],
        skills=[SimpleNamespace(id="get_item_info", input_modes=["text", "data"], output_modes=["text"])],
    )
    try:
        require_agent_card_modes(
            wrong_card,
            input_modes={"data", "file"},
            output_modes={"data"},
            skill_ids={"get_item_info_no_sam3d"},
        )
    except ValueError as exc:
        assert "not the expected A2A service" in str(exc)
        assert "get_item_info_no_sam3d" in str(exc)
    else:
        raise AssertionError("wrong AgentCard was accepted")


def test_a2a_card_validation_accepts_required_modes() -> None:
    card = SimpleNamespace(
        name="Get Item Info Agent No SAM3D",
        defaultInputModes=["data", "file"],
        defaultOutputModes=["data"],
        skills=[SimpleNamespace(id="get_item_info_no_sam3d", inputModes=["data", "file"], outputModes=["data"])],
    )
    require_agent_card_modes(
        card,
        input_modes={"data", "file"},
        output_modes={"data"},
        skill_ids={"get_item_info_no_sam3d"},
    )


def test_a2a_adapter_accepts_direct_message_and_task_artifact() -> None:
    message = new_agent_parts_message([Part(root=DataPart(data={"ok": True}))], context_id="ctx", task_id="task-1")
    payload, task_id = extract_result_payload(SimpleNamespace(root=SimpleNamespace(result=message)))
    assert payload == {"ok": True}
    assert task_id == "task-1"

    task = completed_task(
        task_id="task-2",
        context_id="ctx",
        artifacts=[new_data_artifact("result", {"value": 42})],
        history=None,
    )
    payload, task_id = extract_result_payload(SimpleNamespace(root=SimpleNamespace(result=task)))
    assert payload == {"value": 42}
    assert task_id == "task-2"


def test_brain_prompt_keeps_human_context_compact_and_omits_retry_count() -> None:
    state = create_initial_state("ctx-prompt")
    state["retry_count"] = 7
    state["requested_object"] = {"id": "doll", "label": "brown teddy bear"}
    state["selected_instance"] = {"instance_key": "doll_1"}
    state["navigation"]["current_goal_rank"] = 1
    state["navigation"]["result"] = {"arrived": True, "message": "arrived at rank 1"}
    state["item_info"] = {
        "center_world": [2.0, 0.5, 4.0],
        "group_ranking": [
            {"rank": 1, "best_goal_pose_ros_map": [1.0, 0.0]},
            {"rank": 2, "best_goal_pose_ros_map": [1.5, 0.5]},
        ],
    }
    state["approach_result"] = {
        "success": False,
        "phase": "grasp_verification_failed",
        "message": "target still free after grasp",
    }

    prompt = Brain(use_mock=True)._build_prompt(state)
    assert isinstance(prompt, str)
    assert "## Current State" in prompt
    assert "Target: brown teddy bear (doll)" in prompt
    assert "Target instance: doll_1" in prompt
    assert "Next viewpoint available: True" in prompt
    assert "Last navigation: arrived=True, message=arrived at rank 1" in prompt
    assert "latest approach failed at current rank" in prompt
    assert "Retry Count" not in prompt
    assert "retry_count" not in prompt
    assert "Grasp Feasibility Check" not in prompt
    assert "Available ranked goal poses" not in prompt


def test_brain_owns_decision_and_history_surfaces_failure_phase() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)

    # The hardcoded grasp->major_nav override has been removed: the VLM owns the
    # grasp-vs-switch-viewpoint decision and judges it from the execution history.
    assert not hasattr(orchestrator, "_apply_decision_safety_guard")
    assert not hasattr(orchestrator, "_should_force_major_nav_after_failed_attempt")

    # A before-the-car-moves approach failure is surfaced with its phase and the
    # viewpoint rank, so the VLM can decide to switch viewpoints by itself.
    state = create_initial_state("ctx-history")
    state["navigation"]["current_goal_rank"] = 2
    state["approach_result"] = {"success": False, "phase": "no_feasible_sample", "message": "no feasible grasp"}
    _, approach_facts = orchestrator._memory_summary(state)
    assert approach_facts["at_rank"] == 2
    assert approach_facts["approach_phase"] == "no_feasible_sample"
    assert approach_facts["approach_success"] is False

    # A grasp service that returns no pose is likewise recorded as a failure.
    grasp_state = create_initial_state("ctx-history-grasp")
    grasp_state["grasp_result"] = {"success": False}
    _, grasp_facts = orchestrator._memory_summary(grasp_state)
    assert grasp_facts["at_rank"] == 1
    assert grasp_facts["grasp_success"] is False


def test_pre_move_failure_backstop_switches_after_two_failures() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    state = create_initial_state("ctx-backstop")
    state["navigation"]["current_goal_rank"] = 1
    state["item_info"] = {
        "center_world": [2.0, 0.5, 4.0],
        "group_ranking": [
            {"rank": 1, "best_goal_pose_ros_map": [1.0, 0.0]},
            {"rank": 2, "best_goal_pose_ros_map": [1.5, 0.5]},
        ],
    }
    grasp_decision = BrainDecision(reasoning="looks graspable", call_module="car_approach_agent", module_params={"object_id": "doll"})

    # One before-the-car-moves failure at this viewpoint: the VLM still owns it.
    state["history_buffer"] = [
        {"action": "car_approach_agent", "key_facts": {"at_rank": 1, "approach_success": False, "approach_phase": "no_feasible_sample"}},
    ]
    assert orchestrator._consecutive_pre_move_failures_at_viewpoint(state) == 1
    assert orchestrator._apply_pre_move_backstop(state, grasp_decision).call_module == "car_approach_agent"

    # Two consecutive before-the-car-moves failures: the backstop forces a switch.
    state["history_buffer"] = [
        {"action": "car_approach_agent", "key_facts": {"at_rank": 1, "approach_success": False, "approach_phase": "no_grasp"}},
        {"action": "car_approach_agent", "key_facts": {"at_rank": 1, "grasp_success": False}},
    ]
    assert orchestrator._consecutive_pre_move_failures_at_viewpoint(state) == 2
    switched = orchestrator._apply_pre_move_backstop(state, grasp_decision)
    assert switched.call_module == "major_nav_node"
    assert switched.module_params["backstop_overridden_from"] == "car_approach_agent"

    # A failure AFTER the car moved breaks the streak and does not count.
    state["history_buffer"] = [
        {"action": "car_approach_agent", "key_facts": {"at_rank": 1, "approach_success": False, "approach_phase": "no_feasible_sample"}},
        {"action": "car_approach_agent", "key_facts": {"at_rank": 1, "approach_success": False, "approach_phase": "arm_sequence_failed"}},
    ]
    assert orchestrator._consecutive_pre_move_failures_at_viewpoint(state) == 0
    assert orchestrator._apply_pre_move_backstop(state, grasp_decision).call_module == "car_approach_agent"

    # No remaining viewpoint to fall back to: the VLM's choice stands.
    state["item_info"]["group_ranking"] = [{"rank": 1, "best_goal_pose_ros_map": [1.0, 0.0]}]
    state["history_buffer"] = [
        {"action": "car_approach_agent", "key_facts": {"at_rank": 1, "grasp_success": False}},
        {"action": "car_approach_agent", "key_facts": {"at_rank": 1, "grasp_success": False}},
    ]
    assert orchestrator._apply_pre_move_backstop(state, grasp_decision).call_module == "car_approach_agent"


def test_langgraph_general_chat_returns_to_input_without_replaying_reply() -> None:
    async def _run() -> list[str]:
        orchestrator = await Orchestrator.create(
            trace_logger=SimpleNamespace(log_trace=lambda **kwargs: None),
            use_mock=True,
        )
        original_input = builtins.input
        builtins.input = lambda prompt="": "bye"
        context_id = uuid.uuid4().hex
        state = create_initial_state(context_id)
        state.update({"human_reply": "hello"})
        nodes: list[str] = []
        try:
            async for event in orchestrator.graph.astream(
                state,
                config={"configurable": {"thread_id": context_id}, "recursion_limit": 20},
            ):
                nodes.extend(event.keys())
        finally:
            builtins.input = original_input
            await orchestrator.aclose()
        return nodes

    nodes = asyncio.run(_run())
    assert nodes == [
        "greeting_node",
        "human_reply_node",
        "task_classification_node",
        "ai_reply_node",
        "chat_memory_node",
        "human_reply_node",
        "goodbye_node",
    ]


def test_langgraph_route_matrix() -> None:
    async def _run() -> None:
        orchestrator = await Orchestrator.create(
            trace_logger=SimpleNamespace(log_trace=lambda **kwargs: None),
            use_mock=True,
        )
        try:
            cases = [
                ("human_reply.goodbye", orchestrator._route_human_reply({"human_reply": "bye"}), "goodbye_node"),
                ("human_reply.continue", orchestrator._route_human_reply({"human_reply": "hello"}), "task_classification_node"),
                ("task_classification.specific", orchestrator._route_task_classification({"task_intent": "specific_task", "selected_object_index": 0}), "input_node"),
                ("task_classification.general", orchestrator._route_task_classification({"task_intent": "general_chat", "selected_object_index": 0}), "ai_reply_node"),
                ("find.selected", orchestrator._route_find({"selected_instance": {"instance_key": "apple_1"}}), "get_item_info_no_sam3d_node"),
                ("find.empty", orchestrator._route_find({"selected_instance": {}}), "end"),
                ("item_info.ready", orchestrator._route_get_item_info_no_sam3d({"current_status": "ITEM_INFO_NO_SAM3D_READY", "navigation": {"nav_goal": {"x": 1}}}), "nav_move_node"),
                ("item_info.no_goal", orchestrator._route_get_item_info_no_sam3d({"current_status": "ITEM_INFO_NO_SAM3D_READY", "navigation": {}}), "nav_home_node"),
                ("item_info.failed", orchestrator._route_get_item_info_no_sam3d({"current_status": "ITEM_INFO_NO_SAM3D_FAILED", "navigation": {"nav_goal": {"x": 1}}}), "nav_home_node"),
                ("update_info_1.missing", orchestrator._route_update_item_info_1({"world_position": {"update_reason": "target_missing"}}), "nav_home_node"),
                ("update_info_1.changed", orchestrator._route_update_item_info_1({"world_position": {"target_changed": True}}), "get_item_info_no_sam3d_node"),
                ("update_info_1.unchanged", orchestrator._route_update_item_info_1({"world_position": {"target_changed": False}}), "major_nav_node"),
                ("update_info_2.missing", orchestrator._route_update_item_info_2({"world_position": {"update_reason": "target_missing"}}), "nav_home_node"),
                ("update_info_2.changed", orchestrator._route_update_item_info_2({"world_position": {"target_changed": True}}), "get_item_info_no_sam3d_node"),
                ("update_info_2.unchanged", orchestrator._route_update_item_info_2({"world_position": {"target_changed": False}}), "car_grasp_node"),
                ("decision.done", orchestrator._route_decision({"decision": {"call_module": "DONE"}}), "end"),
                ("decision.task_complete", orchestrator._route_decision({"task_complete": True, "decision": {"call_module": "nav_agent"}}), "end"),
                ("decision.nav_agent", orchestrator._route_decision({"decision": {"call_module": "nav_agent"}}), "major_nav_node"),
                ("decision.major_nav_node", orchestrator._route_decision({"decision": {"call_module": "major_nav_node"}}), "major_nav_node"),
                ("decision.grasp_agent", orchestrator._route_decision({"decision": {"call_module": "grasp_agent"}}), "car_grasp_node"),
                ("decision.approach_agent", orchestrator._route_decision({"decision": {"call_module": "approach_agent"}}), "car_grasp_node"),
                ("decision.unknown", orchestrator._route_decision({"decision": {"call_module": "noop"}}), "end"),
                ("major_nav.exhausted", orchestrator._route_major_nav({"current_status": "MAJOR_NAV_EXHAUSTED"}), "nav_home_node"),
                ("major_nav.task_complete", orchestrator._route_major_nav({"task_complete": True}), "nav_home_node"),
                ("major_nav.ready", orchestrator._route_major_nav({"current_status": "MAJOR_NAV_CONTEXT_READY"}), "nav_move_node"),
                ("nav_move.bootstrap", orchestrator._route_nav_move({"navigation": {"nav_move_source": "bootstrap"}}), "observe_node"),
                ("nav_move.major_nav", orchestrator._route_nav_move({"navigation": {"nav_move_source": "major_nav"}}), "update_memory_node"),
                ("nav_move.empty", orchestrator._route_nav_move({"navigation": {}}), "update_memory_node"),
                ("car_approach.success", orchestrator._route_car_approach({"approach_result": {"success": True}}), "end"),
                ("car_approach.failed", orchestrator._route_car_approach({"approach_result": {"success": False}}), "update_memory_node"),
                ("car_approach.empty", orchestrator._route_car_approach({"approach_result": {}}), "update_memory_node"),
            ]
            for name, got, expected in cases:
                assert got == expected, f"{name}: got {got}, expected {expected}"
        finally:
            await orchestrator.aclose()

    asyncio.run(_run())


def run_all() -> None:
    test_history_reducer_appends_all_entries()
    test_task_classifier_matches_id_label_and_keeps_chat_general()
    test_legacy_prompt_type_only_uses_detection_selection()
    test_item_info_uses_all_room1_upload_cameras()
    test_related_object_options_use_llm_config_indices()
    test_lexical_related_object_options_use_labels_only()
    test_update_item_info_accepts_nearby_instance_id_reassignment()
    test_update_item_info_rejects_far_or_large_id_delta_candidates()
    test_item_info_no_sam3d_target_selection_respects_selected_instance()
    test_artifact_store_json_text_bytes_and_db_rows()
    test_artifact_store_register_file_and_clear_artifacts()
    test_world_snapshot_raw_and_goal_pose_are_sqlite_rows()
    test_session_store_rejects_raw_state_blobs()
    test_brain_accepts_only_structured_decisions()
    test_a2a_card_validation_rejects_wrong_service()
    test_a2a_card_validation_accepts_required_modes()
    test_a2a_adapter_accepts_direct_message_and_task_artifact()
    test_brain_prompt_keeps_human_context_compact_and_omits_retry_count()
    test_brain_owns_decision_and_history_surfaces_failure_phase()
    test_pre_move_failure_backstop_switches_after_two_failures()
    test_langgraph_general_chat_returns_to_input_without_replaying_reply()
    test_langgraph_route_matrix()


if __name__ == "__main__":
    run_all()
    print(json.dumps({"ok": True, "tests": 22, "run_id": uuid.uuid4().hex}, ensure_ascii=False))
