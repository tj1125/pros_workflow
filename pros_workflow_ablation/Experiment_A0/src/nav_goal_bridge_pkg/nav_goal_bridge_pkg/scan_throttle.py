from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanThrottle(Node):
    def __init__(self) -> None:
        super().__init__("scan_throttle_node")
        self.declare_parameter("input_topic", "/scan_tmp")
        self.declare_parameter("output_topic", "/scan")
        self.declare_parameter("target_rate_hz", 10.0)

        input_topic = str(self.get_parameter("input_topic").value)
        output_topic = str(self.get_parameter("output_topic").value)
        target_rate_hz = float(self.get_parameter("target_rate_hz").value)
        if target_rate_hz <= 0.0:
            self.get_logger().warn("target_rate_hz must be positive; using 10.0 Hz")
            target_rate_hz = 10.0

        self._latest_scan: LaserScan | None = None
        self._latest_scan_received_at = 0.0
        self._last_publish_scan_received_at = 0.0

        self._publisher = self.create_publisher(
            LaserScan, output_topic, qos_profile_sensor_data
        )
        self.create_subscription(
            LaserScan, input_topic, self._scan_callback, qos_profile_sensor_data
        )
        self.create_timer(1.0 / target_rate_hz, self._publish_latest_scan)

        self.get_logger().info(
            "scan_throttle_node ready: "
            f"{input_topic} -> {output_topic} at {target_rate_hz:.2f} Hz"
        )

    def _scan_callback(self, msg: LaserScan) -> None:
        self._latest_scan = msg
        self._latest_scan_received_at = time.monotonic()

    def _publish_latest_scan(self) -> None:
        if self._latest_scan is None:
            return
        if self._latest_scan_received_at == self._last_publish_scan_received_at:
            return
        self._last_publish_scan_received_at = self._latest_scan_received_at
        self._latest_scan.header.stamp = self.get_clock().now().to_msg()
        self._publisher.publish(self._latest_scan)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScanThrottle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
