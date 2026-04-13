import base64
import os
import re
import time
from typing import Dict, Iterable, List

import requests
import roslibpy
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

import ollama

LOCAL_FOLDER = os.environ.get(
    "LOCAL_ITEM_FOLDER",
    "/Users/chentingjie/self-hosted-ai-starter-kit/shared/pics_db",
)
DB_IMAGE_PREFIX = os.environ.get("DB_IMAGE_PREFIX", "/data/shared/pics_db")
N8N_DB_UPDATE_URL = os.environ.get(
    "N8N_DB_UPDATE_URL", "http://localhost:5678/webhook/insert-object"
)

_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
_PROCESSOR = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
_MODEL = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(_DEVICE)
print(f"🔧 BLIP running on: {_DEVICE}")


def generate_caption(image_path: str, num_captions: int = 1) -> str:
    image = Image.open(image_path).convert("RGB")
    inputs = _PROCESSOR(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(_DEVICE)

    captions: List[str] = []
    for _ in range(num_captions):
        output = _MODEL.generate(
            pixel_values=pixel_values,
            max_length=40,
            num_beams=5,
            temperature=1.0,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
        captions.append(_PROCESSOR.decode(output[0], skip_special_tokens=True))
    return " ".join(captions)


def embed_text(text: str) -> List[float]:
    response = ollama.embeddings(model="nomic-embed-text:latest", prompt=text)
    return response.embedding


def post_json(url: str, payload: dict, purpose: str) -> dict:
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        # print(f"✅ {purpose} webhook OK ({response.status_code})")
        if response.text:
            try:
                return response.json()
            except ValueError:
                return {}
        return {}
    except Exception as exc:
        print(f"❌ {purpose} webhook failed: {exc}")
        return {}


def _format_coord_value(value: float) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    text = f"{number:.3f}"
    text = text.rstrip("0").rstrip(".")
    return text or "0"


class ImageTopicManager:
    def __init__(
        self,
        client: roslibpy.Ros,
        local_folder: str = LOCAL_FOLDER,
        db_prefix: str = DB_IMAGE_PREFIX,
    ) -> None:
        self.client = client
        self.local_folder = local_folder
        self.db_prefix = db_prefix
        self.subscriptions: Dict[str, roslibpy.Topic] = {}
        self.image_records: Dict[str, dict] = {}
        os.makedirs(self.local_folder, exist_ok=True)

    @staticmethod
    def build_key(item_name: str, item_id: int) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", item_name.strip().lower())
        slug = re.sub(r"_+", "_", slug).strip("_") or "item"
        return f"{slug}_{item_id}"

    def _file_path(self, key: str) -> str:
        return os.path.join(self.local_folder, f"{key}.jpg")

    def _db_path(self, key: str) -> str:
        return f"{self.db_prefix}/{key}.jpg"

    def ensure_subscriptions(self, items: Iterable[dict]) -> None:
        for item in items:
            key = self.build_key(item["item"], item["id"])
            if key in self.subscriptions:
                continue
            topic_name = f"/cropped/{item['item']}_{item['id']}"
            topic = roslibpy.Topic(
                self.client, topic_name, "sensor_msgs/CompressedImage"
            )
            topic.subscribe(self._make_image_callback(key, topic_name, item))
            self.subscriptions[key] = topic
            print(f"📡 Subscribed to image topic: {topic_name}")

    def remove_items_by_keys(
        self, keys: Iterable[str], delete_files: bool = True
    ) -> None:
        for key in keys:
            topic = self.subscriptions.pop(key, None)
            if topic:
                topic.unsubscribe()
                print(f"🛑 Unsubscribed from {topic.name}")

            file_path = self._file_path(key)
            if delete_files and os.path.exists(file_path):
                os.remove(file_path)
                print(f"🗑️ Deleted cached image: {file_path}")

            self.image_records.pop(key, None)

    def _make_image_callback(self, key: str, topic_name: str, item: dict):
        def callback(message: dict) -> None:
            try:
                payload = message.get("data")
                if payload is None:
                    print(f"⚠️ {topic_name} payload missing 'data'")
                    return
                if isinstance(payload, list):
                    image_bytes = bytes(payload)
                elif isinstance(payload, str):
                    image_bytes = base64.b64decode(payload)
                elif isinstance(payload, (bytes, bytearray)):
                    image_bytes = bytes(payload)
                else:
                    print(f"⚠️ {topic_name} unsupported data type: {type(payload)}")
                    return
                file_path = self._file_path(key)
                with open(file_path, "wb") as handler:
                    handler.write(image_bytes)
                self.image_records[key] = {
                    "item": item,
                    "path": file_path,
                    "topic": topic_name,
                    "updated_at": time.time(),
                }
                # print(f"📸 Stored {topic_name} image at {file_path}")
            except Exception as exc:
                print(f"❌ Failed to store image from {topic_name}: {exc}")

        return callback

    def prepare_and_send_updates(self, items: Iterable[dict]) -> List[str]:
        processed_keys: List[str] = []
        for item in items:
            item_name = item.get("item")
            item_id = item.get("id")
            if not item_name or item_id is None:
                continue

            key = self.build_key(item_name, item_id)
            file_path = self._file_path(key)
            if not os.path.exists(file_path):
                print(f"⚠️ Missing image for {item_name}_{item_id}, wait for image before update")
                continue

            try:
                print(f"🤖 Running BLIP for {item_name}_{item_id}")
                caption = generate_caption(file_path)
                caption_text = (
                    caption.strip() if isinstance(caption, str) else str(caption)
                )
                embedding = embed_text(caption_text)

                coord_x = _format_coord_value(item.get("world_x"))
                coord_y = _format_coord_value(item.get("world_y"))
                coord_line = f"[{coord_x}, {coord_y}, 0]"
                db_path = self._db_path(key)
                content = (
                    f"Coordinate: {coord_line}\n"
                    f"Caption: {caption_text}\n"
                    f"image_path: {db_path}"
                )
                payload = {
                    "image_path": db_path,
                    "content": content,
                    "embedding": embedding,
                }
                post_json(N8N_DB_UPDATE_URL, payload, "DB upsert")
                processed_keys.append(key)
            except Exception as exc:
                print(f"❌ Failed to build payload for {item_name}-{item_id}: {exc}")
        return processed_keys
