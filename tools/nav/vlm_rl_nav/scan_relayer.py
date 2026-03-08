import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


class ScanRelayer(Node):
    def __init__(self) -> None:
        super().__init__("scan_relayer")
        self.declare_parameter("timestamp_backdate_ms", 0.0)
        self.declare_parameter("max_publish_hz", 8.0)
        backdate_ms = float(self.get_parameter("timestamp_backdate_ms").value)
        self._backdate = Duration(seconds=0.0, nanoseconds=int(backdate_ms * 1_000_000))
        max_publish_hz = float(self.get_parameter("max_publish_hz").value)
        self._min_publish_period = None
        if max_publish_hz > 0.0:
            self._min_publish_period = Duration(seconds=1.0 / max_publish_hz)
        self._last_publish_time = None

        self.sub = self.create_subscription(
            LaserScan,
            "/scan_tmp",
            self.scan_callback,
            rclpy.qos.qos_profile_sensor_data,
        )

        sensor_scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.pub = self.create_publisher(LaserScan, "/scan", sensor_scan_qos)
        self.get_logger().info(
            "Scan Relayer started: /scan_tmp -> /scan "
            f"(syncing timestamps to container time - {backdate_ms:.1f} ms, "
            f"max_publish_hz={max_publish_hz:.1f}, qos=BEST_EFFORT)"
        )

    def scan_callback(self, msg: LaserScan) -> None:
        now = self.get_clock().now()
        if (
            self._min_publish_period is not None
            and self._last_publish_time is not None
            and now - self._last_publish_time < self._min_publish_period
        ):
            return

        msg.header.stamp = (now - self._backdate).to_msg()
        self.pub.publish(msg)
        self._last_publish_time = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScanRelayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
