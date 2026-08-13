from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
_OBJECTS_CONFIG_PATH = _CONFIG_DIR / "objects.yaml"


def load_graspable_objects() -> list[dict[str, Any]]:
    with _OBJECTS_CONFIG_PATH.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    objects = payload.get("graspable_objects", [])
    return objects if isinstance(objects, list) else []


def valid_object_index(value: Any, objects: list[dict[str, Any]]) -> int:
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return 0
    return idx if 1 <= idx <= len(objects) else 0


def normalize_match_text(value: Any) -> str:
    text = str(value or "").casefold().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def object_terms(obj: dict[str, Any]) -> list[str]:
    raw_terms: list[Any] = [obj.get("label", "")]
    terms: list[str] = []
    for term in raw_terms:
        normalized = normalize_match_text(term)
        if normalized and normalized not in terms:
            terms.append(normalized)
    return terms


def object_option(obj: dict[str, Any], index: int, *, reason: str, score: int) -> dict[str, Any]:
    object_id = str(obj.get("id") or obj.get("label") or "").strip()
    return {
        "id": object_id,
        "label": str(obj.get("label") or object_id),
        "index": int(index),
        "match_reason": reason,
        "match_score": int(score),
    }


def lexical_related_object_options(
    text: str,
    objects: list[dict[str, Any]],
    *,
    selected_index: int = 0,
) -> list[dict[str, Any]]:
    selected_index = valid_object_index(selected_index, objects)
    normalized_text = normalize_match_text(text)
    selected_terms = set(object_terms(objects[selected_index - 1])) if selected_index else set()
    selected_tokens = {token for term in selected_terms for token in term.split() if token}

    options: list[dict[str, Any]] = []
    seen: set[str] = set()
    for idx, obj in enumerate(objects, 1):
        terms = object_terms(obj)
        tokens = {token for term in terms for token in term.split() if token}
        reason = ""
        score = 0
        if idx == selected_index:
            reason = "selected_object"
            score = 100
        elif normalized_text and any(term and term in normalized_text for term in terms):
            reason = "direct_text_match"
            score = 80
        elif selected_tokens and tokens & selected_tokens:
            reason = "shared_config_token"
            score = 50
        if not reason:
            continue
        option = object_option(obj, idx, reason=reason, score=score)
        if option["id"] and option["id"] not in seen:
            seen.add(option["id"])
            options.append(option)
    return options
