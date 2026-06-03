from __future__ import annotations

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
