"""
commander/room_topics.py — One-shot ROS topic helpers for room-camera snapshots
and world_position_data retrieval.

This module mirrors commander/camera.py's subprocess pattern so the main app can
keep running under Python 3.12 while ROS subscriptions happen under the system
Python with rclpy available.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _ros_python_bin() -> str:
    return os.getenv("ROS_PYTHON_BIN", "/usr/bin/python3")


def _ros_setup_scripts() -> list[str]:
    return [
        os.getenv("ROS_SETUP_BASH", "/opt/ros/humble/setup.bash"),
        os.getenv("ROS_OVERLAY_SETUP_BASH", "/workspaces/install/setup.bash"),
    ]


def _topic_subprocess_cmd(mode: str, topic_name: str, timeout_sec: float) -> str:
    script = shlex.quote(os.path.abspath(__file__))
    python_bin = shlex.quote(_ros_python_bin())
    quoted_topic = shlex.quote(topic_name)
    quoted_timeout = shlex.quote(str(timeout_sec))
    parts = ["unset VIRTUAL_ENV PYTHONHOME PYTHONPATH"]
    for setup_script in _ros_setup_scripts():
        if setup_script:
            quoted = shlex.quote(setup_script)
            parts.append(f"if [ -f {quoted} ]; then source {quoted}; fi")
    parts.append(f"export ROS_DOMAIN_ID={shlex.quote(os.getenv('ROS_DOMAIN_ID', '1'))}")
    parts.append(
        f"exec {python_bin} {script} {shlex.quote(mode)} {quoted_topic} --timeout {quoted_timeout}"
    )
    return " && ".join(parts)


async def get_topic_string_message(
    topic_name: str = "/world_position_data",
    timeout_sec: float = 5.0,
) -> Optional[str]:
    loop = asyncio.get_event_loop()

    def _capture() -> Optional[str]:
        try:
            result = subprocess.run(
                ["/bin/bash", "-lc", _topic_subprocess_cmd("string_topic", topic_name, timeout_sec)],
                capture_output=True,
                text=True,
                timeout=timeout_sec + 2.0,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            logger.error("[Topic] String topic capture failed: %s", result.stderr.strip())
            return None
        except Exception as exc:
            logger.error("[Topic] String topic subprocess error: %s", exc)
            return None

    return await loop.run_in_executor(None, _capture)


async def get_compressed_image_topic_base64(
    topic_name: str,
    timeout_sec: float = 10.0,
) -> Optional[str]:
    loop = asyncio.get_event_loop()

    def _capture() -> Optional[str]:
        try:
            result = subprocess.run(
                ["/bin/bash", "-lc", _topic_subprocess_cmd("compressed_image_topic", topic_name, timeout_sec)],
                capture_output=True,
                text=True,
                timeout=timeout_sec + 2.0,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            logger.error("[Topic] Image topic capture failed: %s", result.stderr.strip())
            return None
        except Exception as exc:
            logger.error("[Topic] Image topic subprocess error: %s", exc)
            return None

    return await loop.run_in_executor(None, _capture)


def save_preview_bbox_annotated(image_base64: str, bbox: list[float], output_path: Path) -> bool:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "image_base64": image_base64,
            "bbox": bbox,
            "output_path": str(output_path),
        }
    )
    try:
        result = subprocess.run(
            [_ros_python_bin(), os.path.abspath(__file__), "annotate_preview"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        if result.returncode == 0:
            return True
        logger.error("[Topic] Preview annotate failed: %s", result.stderr.strip())
        return False
    except Exception as exc:
        logger.error("[Topic] Preview annotate subprocess error: %s", exc)
        return False


def save_preview_crop(image_base64: str, bbox: list[float], output_path: Path) -> bool:
    """Backward-compatible wrapper; now saves the full image with a red bbox."""
    return save_preview_bbox_annotated(image_base64, bbox, output_path)


if __name__ == "__main__":
    import argparse
    import base64
    import io
    import sys
    import time

    parser = argparse.ArgumentParser(description="One-shot ROS topic helpers.")
    parser.add_argument(
        "mode",
        choices=("string_topic", "compressed_image_topic", "annotate_preview", "crop_preview"),
    )
    parser.add_argument("topic_name", nargs="?", default="")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    if args.mode in ("annotate_preview", "crop_preview"):
        from PIL import Image, ImageDraw

        payload = json.load(sys.stdin)
        image_bytes = base64.b64decode(payload["image_base64"])
        bbox = [float(v) for v in payload.get("bbox", [])]
        output_path = Path(payload["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = img.convert("RGB")
            left, top, right, bottom = [int(round(v)) for v in bbox[:4]]
            left = max(0, min(rgb.width, left))
            top = max(0, min(rgb.height, top))
            right = max(0, min(rgb.width, right))
            bottom = max(0, min(rgb.height, bottom))
            if args.mode == "crop_preview":
                if right <= left or bottom <= top:
                    preview = rgb
                else:
                    preview = rgb.crop((left, top, right, bottom))
            else:
                preview = rgb.copy()
                if right > left and bottom > top:
                    draw = ImageDraw.Draw(preview)
                    line_width = max(3, min(preview.width, preview.height) // 200)
                    draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=line_width)
            preview.save(output_path, format="JPEG", quality=95)
        print(str(output_path))
        raise SystemExit(0)

    import rclpy
    from rclpy.node import Node

    class _OnceSubscriber(Node):
        def __init__(self, topic_name: str, timeout: float, mode: str):
            super().__init__(f"vlm_topic_{int(time.time())}")
            self._timeout = timeout
            self._mode = mode
            self._message = None

            if mode == "string_topic":
                from std_msgs.msg import String

                self.create_subscription(String, topic_name, self._callback, 10)
            else:
                from sensor_msgs.msg import CompressedImage

                self.create_subscription(CompressedImage, topic_name, self._callback, 10)

        def _callback(self, msg):
            if self._message is None:
                self._message = msg

        def read_once(self) -> str:
            deadline = time.monotonic() + self._timeout
            while time.monotonic() < deadline and self._message is None:
                rclpy.spin_once(self, timeout_sec=0.1)
            if self._message is None:
                print("ERROR: topic timed out", file=sys.stderr)
                return ""

            if self._mode == "string_topic":
                return str(self._message.data)

            return base64.b64encode(bytes(self._message.data)).decode()

    rclpy.init()
    node = _OnceSubscriber(args.topic_name, timeout=args.timeout, mode=args.mode)
    try:
        result = node.read_once()
        if result:
            print(result)
    finally:
        node.destroy_node()
        rclpy.shutdown()
