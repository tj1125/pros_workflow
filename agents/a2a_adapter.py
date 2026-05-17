"""A2A protocol helpers for Commander-side clients."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def data_part(data: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "data", "data": data}


def file_part(
    *,
    name: str,
    data_base64: str,
    mime_type: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "kind": "file",
        "file": {"name": name, "mimeType": mime_type, "bytes": data_base64},
        "metadata": metadata or {},
    }


def require_agent_card_modes(
    agent_card: Any,
    *,
    input_modes: set[str],
    output_modes: set[str],
    skill_ids: set[str] | None = None,
) -> None:
    skills = list(_field(agent_card, "skills") or [])
    available_skill_ids = {str(_field(skill, "id") or "") for skill in skills}
    if skill_ids and not (skill_ids & available_skill_ids):
        raise ValueError(
            "AgentCard is not the expected A2A service: "
            f"name={_field(agent_card, 'name')!r} "
            f"expected_skill_ids={sorted(skill_ids)} "
            f"advertised_skill_ids={sorted(skill_id for skill_id in available_skill_ids if skill_id)}"
        )

    available_input = _string_set(_field(agent_card, "default_input_modes", "defaultInputModes"))
    available_output = _string_set(_field(agent_card, "default_output_modes", "defaultOutputModes"))
    for skill in skills:
        available_input.update(_string_set(_field(skill, "input_modes", "inputModes")))
        available_output.update(_string_set(_field(skill, "output_modes", "outputModes")))

    missing_input = input_modes - available_input
    missing_output = output_modes - available_output
    if missing_input or missing_output:
        raise ValueError(
            "AgentCard does not advertise required A2A modes: "
            f"missing_input={sorted(missing_input)} "
            f"missing_output={sorted(missing_output)} "
            f"available_input={sorted(available_input)} "
            f"available_output={sorted(available_output)}"
        )


def _field(obj: Any, *names: str) -> Any:
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return None
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _string_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    try:
        return {str(item) for item in value if str(item)}
    except TypeError:
        return {str(value)} if str(value) else set()


def extract_result_payload(response: Any) -> tuple[dict[str, Any], str]:
    """Normalize direct Message or stateful Task/Artifact responses."""
    root = getattr(response, "root", response)
    if hasattr(root, "error") and getattr(root, "error"):
        return {"error": str(getattr(root, "error"))}, ""

    result = getattr(root, "result", root)
    task_id = str(getattr(result, "id", "") or getattr(result, "task_id", "") or "")
    payloads: list[dict[str, Any]] = []

    artifacts = getattr(result, "artifacts", None) or []
    for artifact in artifacts:
        payloads.extend(_payloads_from_parts(getattr(artifact, "parts", None) or []))

    if hasattr(result, "parts"):
        payloads.extend(_payloads_from_parts(getattr(result, "parts", None) or []))

    status = getattr(result, "status", None)
    status_message = getattr(status, "message", None) if status is not None else None
    if status_message is not None:
        payloads.extend(_payloads_from_parts(getattr(status_message, "parts", None) or []))

    if not payloads:
        return {"error": "A2A response did not contain Message parts or Task artifacts."}, task_id

    for payload in payloads:
        if "error" not in payload:
            return payload, task_id
    return payloads[0], task_id


def _payloads_from_parts(parts: list[Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for part in parts:
        root = getattr(part, "root", part)
        kind = getattr(root, "kind", "")
        if kind == "data" or hasattr(root, "data"):
            data = getattr(root, "data", None)
            if isinstance(data, dict):
                payloads.append(data)
            continue
        if kind == "text" or hasattr(root, "text"):
            text = str(getattr(root, "text", "") or "").strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                payloads.append({"text": text})
            else:
                payloads.append(parsed if isinstance(parsed, dict) else {"data": parsed})
    return payloads
