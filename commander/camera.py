import asyncio
import base64
import logging
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import Trigger

logger = logging.getLogger(__name__)

class SimpleCameraClient(Node):
    """
    On-demand camera client.
    Creates a node, requests an image via Trigger, waits for the response on the topic, and then closes.
    """
    def __init__(self, camera_name: str, timeout_sec: float = 10.0):
        super().__init__(f"simple_camera_client_{int(time.time())}")
        self.camera_name = camera_name
        self.timeout_sec = timeout_sec

        camera_key = "".join(
            ch if (ch.isalnum() or ch == "_") else "_"
            for ch in camera_name.strip()
        )
        if not camera_key:
            camera_key = "camera"
        elif camera_key[0].isdigit():
            camera_key = f"cam_{camera_key}"
            
        self.service_name = f"/capture_image/{camera_key}"
        self.rgb_topic = f"/capture_image/{camera_key}/rgb/compressed"

        self.client = self.create_client(Trigger, self.service_name)
        self.rgb_msg: Optional[CompressedImage] = None
        self.waiting_capture = False

        self.rgb_sub = self.create_subscription(
            CompressedImage, self.rgb_topic, self._on_rgb, 10
        )

    def _on_rgb(self, msg: CompressedImage):
        if self.waiting_capture and self.rgb_msg is None:
            self.rgb_msg = msg

    def capture_base64(self) -> Optional[str]:
        """Trigger camera and wait for RGB image, returning it as a base64 string."""
        if not self.client.wait_for_service(timeout_sec=self.timeout_sec):
            logger.error(f"[Camera] Service {self.service_name} not available. Is Unity running?")
            return None

        self.rgb_msg = None
        self.waiting_capture = True
        
        logger.info(f"[Camera] Requesting image from Unity ({self.camera_name})...")
        future = self.client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.timeout_sec)

        response = future.result()
        if response is None or not response.success:
            logger.error(f"[Camera] Failed to trigger image capture: {response.message if response else 'timeout'}")
            self.waiting_capture = False
            return None

        # Wait for image on topic
        deadline = time.monotonic() + self.timeout_sec
        while time.monotonic() < deadline:
            if self.rgb_msg is not None:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        self.waiting_capture = False

        if self.rgb_msg is None:
            logger.error("[Camera] Timed out waiting for RGB image over topic.")
            return None

        logger.info("[Camera] Image received successfully.")
        return base64.b64encode(bytes(self.rgb_msg.data)).decode('utf-8')


async def get_camera_image_base64(camera_name: str = "Camera_Car", timeout_sec: float = 10.0) -> Optional[str]:
    """
    Run the ROS 2 blocking camera client in a background thread 
    so it doesn't block the asyncio event loop of LangGraph.
    """
    loop = asyncio.get_event_loop()
    
    def _capture():
        if not rclpy.ok():
            rclpy.init()
            
        node = SimpleCameraClient(camera_name, timeout_sec)
        try:
            return node.capture_base64()
        finally:
            node.destroy_node()
            
    return await loop.run_in_executor(None, _capture)
