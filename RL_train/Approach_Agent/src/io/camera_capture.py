from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class AmclPoseSnapshot:
    stamp_sec: float | None
    position_xyz: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]


@dataclass
class CameraRgbdSnapshot:
    camera_name: str
    rgb_bytes: bytes
    depth_bytes: bytes
    rgb_format: str
    depth_format: str
    amcl_pose: AmclPoseSnapshot | None


def _camera_key(name: str) -> str:
    key = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name.strip())
    if not key:
        return "camera"
    return f"cam_{key}" if key[0].isdigit() else key


def _stamp_to_seconds(stamp: Any) -> float | None:
    if stamp is None:
        return None
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except Exception:
        return None


def capture_rgbd_snapshot(
    camera_name: str = "Camera_Car",
    *,
    timeout_sec: float = 15.0,
    amcl_topic: str = "/amcl_pose",
) -> CameraRgbdSnapshot:
    try:
        import rclpy
        from geometry_msgs.msg import PoseWithCovarianceStamped
        from rclpy.node import Node
        from sensor_msgs.msg import CompressedImage
        from std_srvs.srv import Trigger
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Missing ROS Python runtime. Please run this stage inside the ROS-enabled project container."
        ) from exc

    class _CaptureNode(Node):
        def __init__(self) -> None:
            super().__init__(f"approach_agent_capture_{int(time.time())}")
            key = _camera_key(camera_name)
            self._service_name = f"/capture_image/{key}"
            self._rgb_topic = f"/capture_image/{key}/rgb/compressed"
            self._depth_topic = f"/capture_image/{key}/depth/compressed"
            self._timeout_sec = float(timeout_sec)
            self._client = self.create_client(Trigger, self._service_name)
            self._rgb_msg: CompressedImage | None = None
            self._depth_msg: CompressedImage | None = None
            self._latest_amcl_msg: PoseWithCovarianceStamped | None = None
            self._waiting_capture = False
            self.create_subscription(CompressedImage, self._rgb_topic, self._on_rgb, 10)
            self.create_subscription(CompressedImage, self._depth_topic, self._on_depth, 10)
            self.create_subscription(PoseWithCovarianceStamped, amcl_topic, self._on_amcl, 10)

        def _on_rgb(self, msg: CompressedImage) -> None:
            if self._waiting_capture and self._rgb_msg is None:
                self._rgb_msg = msg

        def _on_depth(self, msg: CompressedImage) -> None:
            if self._waiting_capture and self._depth_msg is None:
                self._depth_msg = msg

        def _on_amcl(self, msg: PoseWithCovarianceStamped) -> None:
            self._latest_amcl_msg = msg

        def capture(self) -> CameraRgbdSnapshot:
            if not self._client.wait_for_service(timeout_sec=self._timeout_sec):
                raise RuntimeError(f"{self._service_name} is not available.")

            self._rgb_msg = None
            self._depth_msg = None
            self._waiting_capture = True
            future = self._client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=self._timeout_sec)
            response = future.result()
            if response is None or not response.success:
                raise RuntimeError(
                    f"Camera capture trigger failed: {getattr(response, 'message', 'timeout')}"
                )

            deadline = time.monotonic() + self._timeout_sec
            while time.monotonic() < deadline:
                if self._rgb_msg is not None and self._depth_msg is not None:
                    break
                rclpy.spin_once(self, timeout_sec=0.1)
            self._waiting_capture = False

            if self._rgb_msg is None:
                raise RuntimeError("RGB topic timed out.")
            if self._depth_msg is None:
                raise RuntimeError("Depth topic timed out.")

            amcl_snapshot: AmclPoseSnapshot | None = None
            if self._latest_amcl_msg is not None:
                position = self._latest_amcl_msg.pose.pose.position
                orientation = self._latest_amcl_msg.pose.pose.orientation
                amcl_snapshot = AmclPoseSnapshot(
                    stamp_sec=_stamp_to_seconds(self._latest_amcl_msg.header.stamp),
                    position_xyz=(float(position.x), float(position.y), float(position.z)),
                    orientation_xyzw=(
                        float(orientation.x),
                        float(orientation.y),
                        float(orientation.z),
                        float(orientation.w),
                    ),
                )

            return CameraRgbdSnapshot(
                camera_name=camera_name,
                rgb_bytes=bytes(self._rgb_msg.data),
                depth_bytes=bytes(self._depth_msg.data),
                rgb_format=str(getattr(self._rgb_msg, "format", "")),
                depth_format=str(getattr(self._depth_msg, "format", "")),
                amcl_pose=amcl_snapshot,
            )

    rclpy.init()
    node = _CaptureNode()
    try:
        return node.capture()
    finally:
        node.destroy_node()
        rclpy.shutdown()
