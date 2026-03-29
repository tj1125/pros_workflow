from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

from get_item_info_agent_no_sam3d.pipeline.types import BoundingBox, TopicObservation, WorldPositionObject


def normalize_label(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("_", " ").replace("-", " ").split())


def canonical_camera_name(name: str) -> str:
    raw = str(name).strip()
    match = re.search(r"camera[_-]?room1[_-]?(\d+)", raw, flags=re.IGNORECASE)
    if match:
        return f"Camera_Room1_{int(match.group(1))}"
    match = re.search(r"camera[_-]?(\d+)", raw, flags=re.IGNORECASE)
    if match:
        return f"Camera_{int(match.group(1))}"
    return raw


def _to_bbox(item: Any) -> BoundingBox | None:
    if not isinstance(item, (list, tuple)) or len(item) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in item]
    except Exception:
        return None
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _dedupe_bboxes(bboxes: list[Any]) -> list[BoundingBox]:
    seen: set[tuple[float, float, float, float]] = set()
    result: list[BoundingBox] = []
    for item in bboxes:
        bbox = _to_bbox(item)
        if bbox is None:
            continue
        key = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
        if key in seen:
            continue
        seen.add(key)
        result.append(bbox)
    return result


def parse_world_position_data(payload: Any) -> list[WorldPositionObject]:
    data = payload
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise ValueError("world_position_data must resolve to a JSON object.")

    objects: list[WorldPositionObject] = []
    for topic_key in sorted(data.keys(), key=lambda key: str(key)):
        entries = data[topic_key]
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            label = normalize_label(entry.get("item", ""))
            if not label:
                continue
            item_id = int(entry.get("id", -1))
            center_world_unity = np.array(
                [
                    float(entry.get("world_x")),
                    float(entry.get("world_y")),
                    float(entry.get("world_z")),
                ],
                dtype=np.float32,
            )
            camsrc = [canonical_camera_name(camera_name) for camera_name in entry.get("camsrc", [])]
            unique_bboxes = _dedupe_bboxes(list(entry.get("bbox", [])))
            observations: dict[str, TopicObservation] = {}
            for camera_id, bbox in zip(camsrc, unique_bboxes[: len(camsrc)]):
                if camera_id in observations:
                    continue
                observations[camera_id] = TopicObservation(camera_id=camera_id, bbox=bbox)
            objects.append(
                WorldPositionObject(
                    label=label,
                    item_id=item_id,
                    center_world_unity=center_world_unity,
                    observations=observations,
                    topic_key=str(topic_key),
                )
            )
    if not objects:
        raise RuntimeError("No usable objects were found in world_position_data.")
    return objects

