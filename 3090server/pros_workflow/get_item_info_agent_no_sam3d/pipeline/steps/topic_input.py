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


def _coerce_instance_id(value: Any) -> int | None:
    try:
        instance_id = int(value)
    except (TypeError, ValueError):
        return None
    return instance_id if instance_id >= 0 else None


def _coerce_center_world(value: Any) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        return np.asarray([float(value[0]), float(value[1]), float(value[2])], dtype=np.float32)
    except (TypeError, ValueError):
        return None


def world_position_instance_key(obj: WorldPositionObject) -> str:
    return f"{obj.label}_{int(obj.item_id)}"


def select_target_object(
    objects: list[WorldPositionObject],
    target_label: str,
    *,
    target_instance_id: Any = None,
    target_instance_key: str = "",
    target_topic_key: str = "",
    target_center_world: Any = None,
) -> WorldPositionObject:
    normalized_target = normalize_label(target_label)
    target_candidates = [obj for obj in objects if obj.label == normalized_target]
    if not target_candidates:
        available = sorted({obj.label for obj in objects})
        raise RuntimeError(f"Target '{normalized_target}' not found in world_position_data. Available: {available}")

    requested_key = str(target_instance_key or "").strip()
    requested_topic = str(target_topic_key or "").strip()
    requested_instance_id = _coerce_instance_id(target_instance_id)

    if requested_key:
        for obj in target_candidates:
            keys = {
                world_position_instance_key(obj),
                f"{normalize_label(obj.topic_key)}_{int(obj.item_id)}",
                f"{obj.topic_key}:{obj.label}:{int(obj.item_id)}",
            }
            if requested_key in keys:
                return obj

    if requested_instance_id is not None:
        id_matches = [obj for obj in target_candidates if int(obj.item_id) == requested_instance_id]
        if requested_topic:
            for obj in id_matches:
                if str(obj.topic_key) == requested_topic or normalize_label(obj.topic_key) == normalize_label(requested_topic):
                    return obj
        if id_matches:
            return id_matches[0]

    requested_center = _coerce_center_world(target_center_world)
    if requested_center is not None:
        nearest = min(
            target_candidates,
            key=lambda obj: float(np.linalg.norm(np.asarray(obj.center_world_unity, dtype=np.float32) - requested_center)),
        )
        nearest_distance = float(np.linalg.norm(np.asarray(nearest.center_world_unity, dtype=np.float32) - requested_center))
        if nearest_distance <= 0.03:
            return nearest

    return target_candidates[0]


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

