#!/usr/bin/env python3
import argparse
import io
import sys
import time
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from PIL import Image
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import Trigger


class TestCameraClient(Node):
    def __init__(
        self,
        camera_name: str,
        output_dir: Path,
        service_name: str,
        rgb_topic: str,
        depth_topic: str,
        capture_depth: bool,
        timeout_sec: float,
    ):
        super().__init__("test_camera_client")
        self.camera_name = camera_name
        self.output_dir = output_dir
        self.service_name = service_name
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.capture_depth = capture_depth
        self.timeout_sec = timeout_sec

        self.client = self.create_client(Trigger, self.service_name)

        self.rgb_msg: Optional[CompressedImage] = None
        self.depth_msg: Optional[CompressedImage] = None
        self.waiting_capture = False

        self.rgb_sub = self.create_subscription(
            CompressedImage, self.rgb_topic, self._on_rgb, 10
        )
        self.depth_sub = self.create_subscription(
            CompressedImage, self.depth_topic, self._on_depth, 10
        )

    def _on_rgb(self, msg: CompressedImage):
        if self.waiting_capture and self.rgb_msg is None:
            self.rgb_msg = msg

    def _on_depth(self, msg: CompressedImage):
        if self.waiting_capture and self.depth_msg is None:
            self.depth_msg = msg

    def capture(self) -> bool:
        self.get_logger().info(
            f"等待服務 {self.service_name} (Unity 是否有啟動?)...."
        )

        if not self.client.wait_for_service(timeout_sec=self.timeout_sec):
            self.get_logger().error("找不到服務，請確認 Unity 端已經啟動並連上 rosbridge。")
            return False

        self.rgb_msg = None
        self.depth_msg = None
        self.waiting_capture = True

        self.get_logger().info(
            f"請求拍照: service={self.service_name}, rgb_topic={self.rgb_topic}, depth={self.capture_depth}"
        )
        future = self.client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.timeout_sec)

        response = future.result()
        if response is None:
            self.waiting_capture = False
            self.get_logger().error("呼叫服務失敗 (沒有收到回應)。")
            return False

        if not response.success:
            self.waiting_capture = False
            self.get_logger().error(f"拍照觸發失敗: {response.message}")
            return False

        self.get_logger().info(f"觸發成功: {response.message}")

        deadline = time.monotonic() + self.timeout_sec
        while time.monotonic() < deadline:
            has_rgb = self.rgb_msg is not None
            has_depth = (not self.capture_depth) or (self.depth_msg is not None)
            if has_rgb and has_depth:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        self.waiting_capture = False

        if self.rgb_msg is None:
            self.get_logger().error("逾時：沒有收到 RGB topic。")
            return False
        if self.capture_depth and self.depth_msg is None:
            self.get_logger().error("逾時：沒有收到 Depth topic。")
            return False

        self.save_images()
        return True

    def save_images(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        rgb_path = self.output_dir / f"rgb_{self.camera_name}.png"
        depth_path = self.output_dir / f"depth_{self.camera_name}.png"

        try:
            rgb_bytes = bytes(self.rgb_msg.data)
            with Image.open(io.BytesIO(rgb_bytes)) as img:
                img.convert("RGB").save(rgb_path, format="PNG")
            self.get_logger().info(f"RGB 已存檔: {rgb_path}")
        except Exception as exc:
            self.get_logger().error(f"RGB 存檔發生錯誤: {exc}")

        if self.capture_depth and self.depth_msg is not None:
            try:
                depth_bytes = bytes(self.depth_msg.data)
                with open(depth_path, "wb") as depth_file:
                    depth_file.write(depth_bytes)
                self.get_logger().info(f"Depth 已存檔: {depth_path}")
            except Exception as exc:
                self.get_logger().error(f"Depth 存檔發生錯誤: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="測試 Trigger 版 Unity 相機 client"
    )
    parser.add_argument("--camera", "-c", type=str, required=True, help="相機名稱，例如 1_1")
    args = parser.parse_args()

    output_dir = Path(f"/workspaces/VLM_RL/test_img/{args.camera}").expanduser().resolve()
    camera_key = "".join(
        ch if (ch.isalnum() or ch == "_") else "_"
        for ch in args.camera.strip()
    )
    if not camera_key:
        camera_key = "camera"
    elif camera_key[0].isdigit():
        camera_key = f"cam_{camera_key}"

    service_name = f"/capture_image/{camera_key}"
    rgb_topic = f"/capture_image/{camera_key}/rgb/compressed"
    depth_topic = f"/capture_image/{camera_key}/depth/compressed"

    rclpy.init()
    node = TestCameraClient(
        camera_name=args.camera,
        output_dir=output_dir,
        service_name=service_name,
        rgb_topic=rgb_topic,
        depth_topic=depth_topic,
        capture_depth=True,
        timeout_sec=15.0,
    )

    try:
        ok = node.capture()
    except KeyboardInterrupt:
        ok = False
    finally:
        node.destroy_node()
        rclpy.shutdown()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
