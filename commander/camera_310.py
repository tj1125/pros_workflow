import sys
import time
import base64
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import Trigger

class SimpleCameraClient(Node):
    def __init__(self, camera_name: str, timeout_sec: float):
        super().__init__(f"simple_camera_client_{int(time.time())}")
        self.timeout_sec = timeout_sec

        camera_key = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in camera_name.strip())
        if not camera_key:
            camera_key = "camera"
        elif camera_key[0].isdigit():
            camera_key = f"cam_{camera_key}"
            
        self.service_name = f"/capture_image/{camera_key}"
        self.rgb_topic = f"/capture_image/{camera_key}/rgb/compressed"

        self.client = self.create_client(Trigger, self.service_name)
        self.rgb_msg = None
        self.waiting_capture = False

        self.rgb_sub = self.create_subscription(CompressedImage, self.rgb_topic, self._on_rgb, 10)

    def _on_rgb(self, msg):
        if self.waiting_capture and self.rgb_msg is None:
            self.rgb_msg = msg

    def capture_base64(self):
        if not self.client.wait_for_service(timeout_sec=self.timeout_sec):
            print(f"ERROR: Service {self.service_name} not available", file=sys.stderr)
            return ""

        self.rgb_msg = None
        self.waiting_capture = True
        
        future = self.client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.timeout_sec)

        response = future.result()
        if response is None or not response.success:
            print("ERROR: Trigger failed or timeout", file=sys.stderr)
            return ""

        deadline = time.monotonic() + self.timeout_sec
        while time.monotonic() < deadline:
            if self.rgb_msg is not None:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        self.waiting_capture = False
        if self.rgb_msg is None:
            print("ERROR: Wait for image topic timed out", file=sys.stderr)
            return ""

        return base64.b64encode(bytes(self.rgb_msg.data)).decode('utf-8')

def main():
    camera_name = sys.argv[1] if len(sys.argv) > 1 else "Camera_Car"
    rclpy.init()
    node = SimpleCameraClient(camera_name, 10.0)
    try:
        b64 = node.capture_base64()
        if b64:
            print(b64)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
