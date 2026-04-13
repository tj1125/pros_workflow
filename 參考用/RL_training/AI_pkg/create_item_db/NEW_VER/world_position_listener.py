import json
import os
import re
import time
from typing import Any, Dict, List, Set, Tuple

import psycopg2
import requests
import roslibpy

from image_topic_manager import ImageTopicManager

ROS_HOST = os.environ.get("ROS_HOST", "192.168.75.2")
ROS_PORT = int(os.environ.get("ROS_PORT", "9090"))
WORLD_TOPIC = os.environ.get("WORLD_TOPIC", "/world_position_data")
WORLD_TOPIC_TYPE = os.environ.get("WORLD_TOPIC_TYPE", "std_msgs/msg/String")
MOVE_THRESHOLD = float(os.environ.get("MOVE_THRESHOLD", "0.2"))
N8N_DELETE_URL = os.environ.get(
    "N8N_DELETE_URL", "http://localhost:5678/webhook/delete-object"
)

DB_NAME = os.environ.get("PGDATABASE", "item_in_house_db")
DB_USER = os.environ.get("PGUSER", "item_in_house_user")
DB_PASSWORD = os.environ.get("PGPASSWORD", "")
DB_HOST = os.environ.get("PGHOST", "localhost")
DB_PORT = int(os.environ.get("PGPORT", "5432"))


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_world_payload(raw: Any) -> List[dict]:
    if isinstance(raw, (dict, list)):
        return _normalize_world_data(raw)

    cleaned = str(raw or "").strip()
    if not cleaned:
        return []

    candidates = [cleaned]

    sanitized = cleaned.replace("'", '"')
    sanitized = re.sub(r"(\b\d+\b)\s*:", r'"\1":', sanitized)
    if sanitized and not sanitized.startswith("{") and ":" in sanitized:
        sanitized = "{" + sanitized
    if sanitized and not sanitized.endswith("}") and sanitized.startswith("{"):
        sanitized = sanitized + "}"
    if sanitized != cleaned:
        candidates.append(sanitized)

    for text in candidates:
        try:
            decoded = json.loads(text)
            return _normalize_world_data(decoded)
        except json.JSONDecodeError:
            continue

    print(f"⚠️ 無法解析 world topic payload: {raw[:80]}...")
    return []


def _normalize_world_data(decoded) -> List[dict]:
    items: List[dict] = []
    if isinstance(decoded, dict):
        for key, entries in decoded.items():
            if key == "data" and isinstance(entries, (dict, list, str)):
                items.extend(_normalize_world_data(entries))
                continue
            items.extend(_convert_entries(entries))
    elif isinstance(decoded, list):
        items.extend(_convert_entries(decoded))
    elif isinstance(decoded, str):
        try:
            return _normalize_world_data(json.loads(decoded))
        except json.JSONDecodeError:
            return []
    return items


_COORD_RE = re.compile(r"Coordinate:\s*\[([^\]]+)\]")


def _convert_entries(entries) -> List[dict]:
    if not isinstance(entries, list):
        entries = [entries]

    normalized: List[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item_name = str(entry.get("item", "")).strip()
        item_id = entry.get("id")
        if not item_name or item_id is None:
            continue
        try:
            item_id = int(item_id)
        except (ValueError, TypeError):
            continue

        normalized.append(
            {
                "item": item_name,
                "id": item_id,
                "world_x": _to_float(entry.get("world_x")),
                "world_y": _to_float(entry.get("world_y")),
                "world_z": _to_float(entry.get("world_z")),
            }
        )

    return normalized


def _coords_from_content(content: str) -> Dict[str, float]:
    match = _COORD_RE.search(content or "")
    if not match:
        return {}
    raw_values = [part.strip() for part in match.group(1).split(",")]
    numbers: List[float] = []
    for value in raw_values:
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            numbers.append(0.0)
    while len(numbers) < 3:
        numbers.append(0.0)
    return {
        "world_x": numbers[0],
        "world_y": numbers[1],
        "world_z": numbers[2],
    }


class WorldPositionListener:
    def __init__(self) -> None:
        self.client = roslibpy.Ros(host=ROS_HOST, port=ROS_PORT)
        self.image_manager = ImageTopicManager(self.client)
        self.world_topic = roslibpy.Topic(
            self.client, WORLD_TOPIC, WORLD_TOPIC_TYPE
        )
        self.known_items: Dict[str, dict] = {}
        self.pending_update_keys: Set[str] = set()

    def start(self) -> None:
        self.client.run()
        time.sleep(1)
        self.world_topic.subscribe(self._handle_world_message)
        print(f"🚀 Listening to {WORLD_TOPIC} on {ROS_HOST}:{ROS_PORT}")
        try:
            while self.client.is_connected:
                self._process_pending_updates()
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("🛑 Manual stop requested")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self.world_topic:
            try:
                self.world_topic.unsubscribe()
            except Exception:
                pass
        if self.known_items:
            keys = list(self.known_items.keys())
            self.image_manager.remove_items_by_keys(keys, delete_files=False)
            self.known_items.clear()
        self.pending_update_keys.clear()
        if self.client.is_connected:
            self.client.terminate()

    def _handle_world_message(self, message: dict) -> None:
        raw_payload = message.get("data", "")
        items = parse_world_payload(raw_payload)
        if not items:
            print("⚠️ Parsed 0 items from world payload")
            return

        key_map = {
            self.image_manager.build_key(item["item"], item["id"]): item
            for item in items
        }
        # print(f"📦 Parsed {len(items)} items | keys: {list(key_map.keys())}")

        db_state, db_keys = self._fetch_db_snapshot()
        current_keys = set(key_map.keys())
        tracked_keys = set(self.known_items.keys()) | db_keys

        removed_keys = tracked_keys - current_keys
        if removed_keys:
            print(f"🗃️ Detected removed items: {removed_keys}")
            self.image_manager.remove_items_by_keys(removed_keys)
            for key in removed_keys:
                db_path = self.image_manager._db_path(key)
                self._notify_delete(db_path)
                self.pending_update_keys.discard(key)

        for key, item in key_map.items():
            previous = self.known_items.get(key) or db_state.get(key)
            if previous is None or self._has_item_changed(previous, item):
                self.pending_update_keys.add(key)

        self.known_items = key_map
        self.image_manager.ensure_subscriptions(items)
        self._process_pending_updates()

    def _process_pending_updates(self) -> None:
        if not self.pending_update_keys or not self.known_items:
            return

        pending_items = [
            self.known_items[key]
            for key in list(self.pending_update_keys)
            if key in self.known_items
        ]
        if not pending_items:
            return

        processed = self.image_manager.prepare_and_send_updates(pending_items)
        for key in processed:
            self.pending_update_keys.discard(key)
        if processed:
            print(f"✅ Processed keys: {processed}")

    @staticmethod
    def _has_item_changed(previous: dict, current: dict) -> bool:
        for axis in ("world_x", "world_y", "world_z"):
            delta = abs(_to_float(previous.get(axis)) - _to_float(current.get(axis)))
            if delta > MOVE_THRESHOLD:
                return True
        return False

    def _notify_delete(self, image_path: str) -> None:
        payload = {"image_path": image_path}
        try:
            response = requests.post(N8N_DELETE_URL, json=payload, timeout=10)
            response.raise_for_status()
            print(f"🧹 Delete webhook OK for {image_path}")
        except Exception as exc:
            print(f"⚠️ Delete webhook failed for {image_path}: {exc}")

    def _fetch_db_snapshot(self) -> Tuple[Dict[str, dict], Set[str]]:
        records: Dict[str, dict] = {}
        keys: Set[str] = set()
        try:
            connection = psycopg2.connect(
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                host=DB_HOST,
                port=DB_PORT,
            )
        except Exception as exc:
            print(f"⚠️ Failed to connect to DB: {exc}")
            return records, keys

        try:
            with connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT image_path, content FROM object_data")
                    for image_path, content in cursor.fetchall():
                        key = os.path.splitext(os.path.basename(image_path or ""))[0]
                        keys.add(key)
                        coords = _coords_from_content(content or "")
                        if coords:
                            records[key] = coords
        except Exception as exc:
            print(f"⚠️ Failed to query DB state: {exc}")
        finally:
            connection.close()

        return records, keys


if __name__ == "__main__":
    listener = WorldPositionListener()
    listener.start()
