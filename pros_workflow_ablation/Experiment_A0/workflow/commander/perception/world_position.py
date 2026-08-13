"""Helpers for parsing /world_position_data into local candidate records."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


def normalize_label(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("_", " ").replace("-", " ").split())


def normalized_item_id(label: str) -> str:
    return normalize_label(label).replace(" ", "_")


def canonical_camera_name(name: str) -> str:
    raw = str(name).strip()
    match = re.search(r"camera[_-]?room1[_-]?(\d+)", raw, flags=re.IGNORECASE)
    if match:
        return f"Camera_Room1_{int(match.group(1))}"
    match = re.search(r"camera[_-]?(\d+)", raw, flags=re.IGNORECASE)
    if match:
        return f"Camera_{int(match.group(1))}"
    return raw


def instance_key(item_id: str, instance_id: int) -> str:
    return f"{normalized_item_id(item_id)}_{int(instance_id)}"


def bbox_area(bbox: List[float]) -> float:
    if len(bbox) != 4:
        return 0.0
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _parse_bbox(item: Any) -> Optional[List[float]]:
    if not isinstance(item, (list, tuple)) or len(item) != 4:
        return None
    try:
        return [float(v) for v in item]
    except Exception:
        return None


def parse_world_position_payload(payload: Any) -> List[Dict[str, Any]]:
    data = payload
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise ValueError("world_position_data must resolve to a JSON object.")

    results: List[Dict[str, Any]] = []
    for topic_key in sorted(data.keys(), key=lambda key: str(key)):
        entries = data[topic_key]
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            item_id = normalized_item_id(entry.get("item", ""))
            if not item_id:
                continue
            try:
                current_instance_id = int(entry.get("id"))
                center_world = [
                    float(entry.get("world_x")),
                    float(entry.get("world_y")),
                    float(entry.get("world_z")),
                ]
            except Exception:
                continue

            camsrc = [canonical_camera_name(camera_name) for camera_name in entry.get("camsrc", [])]
            bboxes_by_camera: Dict[str, List[float]] = {}
            for camera_name, bbox in zip(camsrc, entry.get("bbox", [])):
                parsed_bbox = _parse_bbox(bbox)
                if parsed_bbox is None or camera_name in bboxes_by_camera:
                    continue
                bboxes_by_camera[camera_name] = parsed_bbox

            results.append(
                {
                    "item_id": item_id,
                    "instance_id": current_instance_id,
                    "instance_key": instance_key(item_id, current_instance_id),
                    "topic_key": str(topic_key),
                    "center_world": center_world,
                    "camsrc": [camera_name for camera_name in camsrc if camera_name in bboxes_by_camera],
                    "bboxes_by_camera": bboxes_by_camera,
                }
            )
    return results


def find_instance(
    payload: Any,
    item_id: str,
    instance_id: int,
) -> Optional[Dict[str, Any]]:
    wanted_item = normalized_item_id(item_id)
    wanted_instance = int(instance_id)
    for candidate in parse_world_position_payload(payload):
        if candidate["item_id"] == wanted_item and int(candidate["instance_id"]) == wanted_instance:
            return candidate
    return None
