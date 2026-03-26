"""
commander/camera.py — On-demand Unity Camera Image Retrieval

Two responsibilities in one file:
  1. When IMPORTED by the Python 3.12 venv (main app):
     `get_camera_image_base64()` / `get_camera_rgbd_base64()` spawn this file as a
     subprocess using the
     system Python 3.10 (which has rclpy compiled against it).

  2. When run DIRECTLY via `/usr/bin/python3 camera.py <camera_name>`:
     Acts as a ROS 2 client, fires the Trigger service, waits for the
     CompressedImage topic(s), and prints the Base64 string / JSON payload to stdout.
"""

import asyncio
import json
import logging
import os
import shlex
import subprocess
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def _ros_python_bin() -> str:
    """Return the Python executable that has ROS 2 Python packages installed."""
    return os.getenv("ROS_PYTHON_BIN", "/usr/bin/python3")


def _ros_setup_scripts() -> list[str]:
    """Return ROS setup scripts to source before running the ROS-side helper."""
    return [
        os.getenv("ROS_SETUP_BASH", "/opt/ros/humble/setup.bash"),
        os.getenv("ROS_OVERLAY_SETUP_BASH", "/workspaces/nav_install/setup.bash"),
    ]


def _camera_subprocess_cmd(camera_name: str, mode: str) -> str:
    """Build a ROS-aware bash command for the camera helper subprocess."""
    script = shlex.quote(os.path.abspath(__file__))
    camera_arg = shlex.quote(camera_name)
    python_bin = shlex.quote(_ros_python_bin())
    mode_arg = "" if mode == "rgb" else " --mode rgbd"
    parts = [
        "unset VIRTUAL_ENV PYTHONHOME PYTHONPATH",
    ]
    for setup_script in _ros_setup_scripts():
        if setup_script:
            quoted = shlex.quote(setup_script)
            parts.append(f"if [ -f {quoted} ]; then source {quoted}; fi")
    parts.append(f"export ROS_DOMAIN_ID={shlex.quote(os.getenv('ROS_DOMAIN_ID', '1'))}")
    parts.append(f"exec {python_bin} {script} {camera_arg}{mode_arg}")
    return " && ".join(parts)


# ---------------------------------------------------------------------------
# Public API — called by orchestrator (Python 3.12 context)
# ---------------------------------------------------------------------------

async def get_camera_image_base64(
    camera_name: str = "Camera_Car",
    timeout_sec: float = 10.0,
) -> Optional[str]:
    """
    Spawn a Python 3.10 subprocess to fetch a single ROS 2 camera image.
    Returns the image as a Base64-encoded string, or None on failure.
    """
    loop = asyncio.get_event_loop()

    def _capture() -> Optional[str]:
        try:
            result = subprocess.run(
                ["/bin/bash", "-lc", _camera_subprocess_cmd(camera_name, mode="rgb")],
                capture_output=True,
                text=True,
                timeout=timeout_sec + 2.0,
            )
            if result.returncode == 0 and result.stdout.strip():
                logger.info("[Camera] Image received successfully.")
                return result.stdout.strip()
            logger.error(f"[Camera] Capture failed: {result.stderr.strip()}")
            return None
        except Exception as exc:
            logger.error(f"[Camera] Subprocess error: {exc}")
            return None

    return await loop.run_in_executor(None, _capture)


async def get_camera_rgbd_base64(
    camera_name: str = "Camera_Car",
    timeout_sec: float = 10.0,
) -> Optional[Dict[str, str]]:
    """
    Spawn a Python 3.10 subprocess to fetch a single ROS 2 camera RGBD observation.
    Returns a dict with Base64-encoded rgb/depth images, or None on failure.
    """
    loop = asyncio.get_event_loop()

    def _capture() -> Optional[Dict[str, str]]:
        try:
            result = subprocess.run(
                ["/bin/bash", "-lc", _camera_subprocess_cmd(camera_name, mode="rgbd")],
                capture_output=True,
                text=True,
                timeout=timeout_sec + 2.0,
            )
            if result.returncode == 0 and result.stdout.strip():
                try:
                    payload = json.loads(result.stdout.strip())
                except json.JSONDecodeError as exc:
                    logger.error(f"[Camera] RGBD payload decode failed: {exc}")
                    return None
                if payload.get("rgb_base64") and payload.get("depth_base64"):
                    logger.info("[Camera] RGBD received successfully.")
                    return payload
            logger.error(f"[Camera] RGBD capture failed: {result.stderr.strip()}")
            return None
        except Exception as exc:
            logger.error(f"[Camera] RGBD subprocess error: {exc}")
            return None

    return await loop.run_in_executor(None, _capture)


# ---------------------------------------------------------------------------
# ROS 2 client — runs in Python 3.10 subprocess
# ---------------------------------------------------------------------------

def _camera_key(name: str) -> str:
    key = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name.strip())
    if not key:
        return "camera"
    return f"cam_{key}" if key[0].isdigit() else key


if __name__ == "__main__":
    import argparse
    import base64
    import sys
    import time

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage
    from std_srvs.srv import Trigger

    class _Client(Node):
        def __init__(self, camera_name: str, timeout: float, mode: str):
            super().__init__(f"vlm_cam_{int(time.time())}")
            self._camera_name = camera_name
            key = _camera_key(camera_name)
            self._svc = f"/capture_image/{key}"
            self._rgb_topic = f"/capture_image/{key}/rgb/compressed"
            self._depth_topic = f"/capture_image/{key}/depth/compressed"
            self._timeout = timeout
            self._mode = mode
            self._client = self.create_client(Trigger, self._svc)
            self._rgb_msg: Optional[CompressedImage] = None
            self._depth_msg: Optional[CompressedImage] = None
            self._waiting = False
            self.create_subscription(CompressedImage, self._rgb_topic, self._rgb_cb, 10)
            self.create_subscription(CompressedImage, self._depth_topic, self._depth_cb, 10)

        def _rgb_cb(self, msg):
            if self._waiting and self._rgb_msg is None:
                self._rgb_msg = msg

        def _depth_cb(self, msg):
            if self._waiting and self._depth_msg is None:
                self._depth_msg = msg

        def capture_base64(self) -> str:
            if not self._client.wait_for_service(timeout_sec=self._timeout):
                print(f"ERROR: {self._svc} not available", file=sys.stderr)
                return ""
            self._rgb_msg = None
            self._depth_msg = None
            self._waiting = True
            future = self._client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=self._timeout)
            resp = future.result()
            if not resp or not resp.success:
                print(f"ERROR: Trigger failed — {getattr(resp, 'message', 'timeout')}", file=sys.stderr)
                return ""
            deadline = time.monotonic() + self._timeout
            while time.monotonic() < deadline:
                if self._rgb_msg is not None and (
                    self._mode == "rgb" or self._depth_msg is not None
                ):
                    break
                rclpy.spin_once(self, timeout_sec=0.1)
            self._waiting = False
            if self._rgb_msg is None:
                print("ERROR: RGB image topic timed out", file=sys.stderr)
                return ""
            rgb_b64 = base64.b64encode(bytes(self._rgb_msg.data)).decode()
            if self._mode == "rgb":
                return rgb_b64

            if self._depth_msg is None:
                print("ERROR: Depth image topic timed out", file=sys.stderr)
                return ""

            depth_b64 = base64.b64encode(bytes(self._depth_msg.data)).decode()
            return json.dumps(
                {
                    "camera_name": self._camera_name,
                    "rgb_base64": rgb_b64,
                    "depth_base64": depth_b64,
                }
            )

    parser = argparse.ArgumentParser(description="Capture Unity camera RGB or RGBD images.")
    parser.add_argument("camera_name", nargs="?", default="Camera_Car")
    parser.add_argument("--mode", choices=("rgb", "rgbd"), default="rgb")
    args = parser.parse_args()

    rclpy.init()
    node = _Client(args.camera_name, timeout=10.0, mode=args.mode)
    try:
        result = node.capture_base64()
        if result:
            print(result)
    finally:
        node.destroy_node()
        rclpy.shutdown()
