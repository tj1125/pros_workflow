#!/usr/bin/env python3
import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from PIL import Image
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import Trigger


DEFAULT_CAMERA_NAME = "Camera_Car"
DEFAULT_TIMEOUT_SEC = 15.0


def camera_key(name: str) -> str:
    key = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name.strip())
    if not key:
        return "camera"
    if key[0].isdigit():
        return f"cam_{key}"
    return key


class RecvCameraCarRGBD(Node):
    def __init__(
        self,
        camera_name: str,
        output_dir: Path,
        service_name: str,
        rgb_topic: str,
        depth_topic: str,
        timeout_sec: float,
    ):
        super().__init__("recv_camera_car_rgbd")
        self.camera_name = camera_name
        self.output_dir = output_dir
        self.service_name = service_name
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.timeout_sec = timeout_sec

        self.client = self.create_client(Trigger, self.service_name)
        self.rgb_msg: Optional[CompressedImage] = None
        self.depth_msg: Optional[CompressedImage] = None
        self.waiting_capture = False

        self.rgb_sub = self.create_subscription(CompressedImage, self.rgb_topic, self._on_rgb, 10)
        self.depth_sub = self.create_subscription(CompressedImage, self.depth_topic, self._on_depth, 10)

    def _on_rgb(self, msg: CompressedImage) -> None:
        if self.waiting_capture and self.rgb_msg is None:
            self.rgb_msg = msg

    def _on_depth(self, msg: CompressedImage) -> None:
        if self.waiting_capture and self.depth_msg is None:
            self.depth_msg = msg

    def capture(self) -> bool:
        self.get_logger().info(f"等待服務 {self.service_name} ...")
        if not self.client.wait_for_service(timeout_sec=self.timeout_sec):
            self.get_logger().error("找不到服務，請確認 Unity 端已啟動並已連上 ROS。")
            return False

        self.rgb_msg = None
        self.depth_msg = None
        self.waiting_capture = True

        self.get_logger().info(
            f"請求拍照: service={self.service_name}, rgb_topic={self.rgb_topic}, depth_topic={self.depth_topic}"
        )
        future = self.client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.timeout_sec)
        response = future.result()

        if response is None:
            self.waiting_capture = False
            self.get_logger().error("呼叫服務失敗，沒有收到回應。")
            return False
        if not response.success:
            self.waiting_capture = False
            self.get_logger().error(f"拍照觸發失敗: {response.message}")
            return False

        self.get_logger().info(f"觸發成功: {response.message}")
        deadline = time.monotonic() + self.timeout_sec
        while time.monotonic() < deadline:
            if self.rgb_msg is not None and self.depth_msg is not None:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        self.waiting_capture = False
        if self.rgb_msg is None:
            self.get_logger().error("逾時：沒有收到 RGB topic。")
            return False
        if self.depth_msg is None:
            self.get_logger().error("逾時：沒有收到 Depth topic。")
            return False

        self.save_images()
        return True

    def save_images(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        rgb_path = self.output_dir / f"{self.camera_name}_rgb.png"
        depth_path = self.output_dir / f"{self.camera_name}_depth.png"
        depth_npy_path = self.output_dir / f"{self.camera_name}_depth.npy"
        meta_path = self.output_dir / f"{self.camera_name}_meta.json"

        try:
            rgb_bytes = bytes(self.rgb_msg.data)
            with Image.open(io.BytesIO(rgb_bytes)) as img:
                rgb = img.convert("RGB")
                rgb.save(rgb_path, format="PNG")
            self.get_logger().info(f"RGB 已存檔: {rgb_path}")
        except Exception as exc:
            self.get_logger().error(f"RGB 存檔發生錯誤: {exc}")
            return

        depth_min = None
        depth_max = None
        depth_shape = None
        try:
            depth_bytes = bytes(self.depth_msg.data)
            depth_path.write_bytes(depth_bytes)
            self.get_logger().info(f"Depth 已存檔: {depth_path}")

            with Image.open(io.BytesIO(depth_bytes)) as depth_img:
                depth_arr = np.array(depth_img)
            if depth_arr.dtype == np.uint16:
                depth_m = depth_arr.astype(np.float32) / 1000.0
            else:
                depth_m = depth_arr.astype(np.float32)
            np.save(depth_npy_path, depth_m)
            depth_min = float(np.nanmin(depth_m))
            depth_max = float(np.nanmax(depth_m))
            depth_shape = list(depth_m.shape)
            self.get_logger().info(f"Depth numpy 已存檔: {depth_npy_path}")
        except Exception as exc:
            self.get_logger().warning(f"Depth 解碼失敗，僅保留原始 png: {exc}")

        meta = {
            "camera_name": self.camera_name,
            "service_name": self.service_name,
            "rgb_topic": self.rgb_topic,
            "depth_topic": self.depth_topic,
            "rgb_format": self.rgb_msg.format if self.rgb_msg is not None else None,
            "depth_format": self.depth_msg.format if self.depth_msg is not None else None,
            "depth_shape": depth_shape,
            "depth_min_m": depth_min,
            "depth_max_m": depth_max,
            "saved_at_unix": time.time(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        self.get_logger().info(f"Meta 已存檔: {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="接收 Unity Camera_Room1_* 的 RGBD 圖。")
    parser.add_argument("--start", type=int, default=1, help="起始相機編號，預設 1。")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SEC, help="等待服務與 topic 的 timeout 秒數。")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="輸出資料夾，預設為本檔所在資料夾。",
    )
    args = parser.parse_args()

    rclpy.init()
    camera_index = args.start
    while True:
        camera_name = f"Camera_Room1_{camera_index}"
        key = camera_key(camera_name)
        service_name = f"/capture_image/{key}"
        rgb_topic = f"/capture_image/{key}/rgb/compressed"
        depth_topic = f"/capture_image/{key}/depth/compressed"

        node = RecvCameraCarRGBD(
            camera_name=camera_name,
            output_dir=args.out_dir.expanduser().resolve(),
            service_name=service_name,
            rgb_topic=rgb_topic,
            depth_topic=depth_topic,
            timeout_sec=args.timeout,
        )

        try:
            ok = node.capture()
            if not ok:
                print(f"相機 {camera_name} 接收失敗，停止。")
                break
        except KeyboardInterrupt:
            ok = False
            break
        finally:
            node.destroy_node()

        camera_index += 1

    rclpy.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
