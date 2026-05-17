from __future__ import annotations

import json
import sqlite3
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

from a2a.types import DataPart, Part
from a2a.utils import completed_task, new_agent_parts_message, new_data_artifact

from agents.a2a_adapter import extract_result_payload, require_agent_card_modes
from commander.artifact_store import ArtifactStore
from commander.session_store import SessionMemoryStore
from commander.orchestrator import Orchestrator, _load_graspable_objects
from commander.state import _append_history, create_initial_state
from commander.web_server import _legacy_prompt_type


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


def test_task_classifier_matches_alias_and_keeps_chat_general() -> None:
    objects = _load_graspable_objects()
    router = Orchestrator.__new__(Orchestrator)

    teddy = router._mock_task_classification("The human wants to pick up the brown teddy bear.", objects)
    assert teddy.intent == "specific_task"
    assert teddy.selected_object_index == 1

    chat = router._mock_task_classification("hello", objects)
    assert chat.intent == "general_chat"
    assert chat.selected_object_index == 0


def test_legacy_prompt_type_only_uses_detection_selection() -> None:
    assert _legacy_prompt_type("請輸入候選照片編號 (1-2) 或 no：") == "detection_selection"
    assert _legacy_prompt_type("請輸入目標物編號 (1-3)：") == "legacy_input"


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


def run_all() -> None:
    test_history_reducer_appends_all_entries()
    test_task_classifier_matches_alias_and_keeps_chat_general()
    test_legacy_prompt_type_only_uses_detection_selection()
    test_artifact_store_json_text_bytes_and_db_rows()
    test_session_store_rejects_raw_state_blobs()
    test_a2a_card_validation_rejects_wrong_service()
    test_a2a_card_validation_accepts_required_modes()
    test_a2a_adapter_accepts_direct_message_and_task_artifact()


if __name__ == "__main__":
    run_all()
    print(json.dumps({"ok": True, "tests": 8, "run_id": uuid.uuid4().hex}, ensure_ascii=False))
