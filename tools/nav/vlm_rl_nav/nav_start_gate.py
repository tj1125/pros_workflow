import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


class NavStartGate(Node):
    def __init__(self) -> None:
        super().__init__("nav_start_gate")
        self.declare_parameter("check_period_sec", 0.2)
        self.declare_parameter("stable_duration_sec", 1.5)
        self.declare_parameter("max_tf_age_sec", 0.25)
        self.declare_parameter("amcl_pose_timeout_sec", 1.5)
        self.declare_parameter("initial_pose_publish_period_sec", 5.0)
        self.declare_parameter("wait_for_scan_before_initial_pose", True)
        self.declare_parameter("scan_timeout_sec", 1.5)
        self.declare_parameter("initial_x", 3.112286942135556)
        self.declare_parameter("initial_y", -3.2084078403508274)
        self.declare_parameter("initial_z", 0.0)
        self.declare_parameter("initial_qx", 0.0)
        self.declare_parameter("initial_qy", 0.0)
        self.declare_parameter("initial_qz", -0.016265232678593755)
        self.declare_parameter("initial_qw", 0.9998677123529448)
        self.declare_parameter(
            "initial_covariance",
            [
                0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.06853892060437211,
            ],
        )

        self._check_period = float(self.get_parameter("check_period_sec").value)
        self._stable_duration = Duration(
            seconds=float(self.get_parameter("stable_duration_sec").value)
        )
        self._max_tf_age = Duration(
            seconds=float(self.get_parameter("max_tf_age_sec").value)
        )
        self._amcl_pose_timeout = Duration(
            seconds=float(self.get_parameter("amcl_pose_timeout_sec").value)
        )
        self._initial_pose_period = Duration(
            seconds=float(self.get_parameter("initial_pose_publish_period_sec").value)
        )
        self._wait_for_scan_before_initial_pose = bool(
            self.get_parameter("wait_for_scan_before_initial_pose").value
        )
        self._scan_timeout = Duration(
            seconds=float(self.get_parameter("scan_timeout_sec").value)
        )

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)
        self._last_amcl_pose_time = None
        self._last_scan_time = None
        self._stable_since = None
        self._last_initial_pose_pub = None
        self._release_requested = False
        self._initial_pose_msg = self._build_initial_pose_msg()

        initial_pose_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            "/initialpose",
            initial_pose_qos,
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self._amcl_pose_callback,
            10,
        )
        self.create_subscription(
            LaserScan,
            "/scan",
            self._scan_callback,
            qos_profile_sensor_data,
        )
        self.create_timer(self._check_period, self._check_ready)

        self.get_logger().info(
            "Nav start gate waiting for fresh AMCL pose and TF chain stability"
        )

    def _amcl_pose_callback(self, _msg: PoseWithCovarianceStamped) -> None:
        self._last_amcl_pose_time = self.get_clock().now()

    def _scan_callback(self, _msg: LaserScan) -> None:
        self._last_scan_time = self.get_clock().now()

    def _build_initial_pose_msg(self) -> PoseWithCovarianceStamped:
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = float(self.get_parameter("initial_x").value)
        msg.pose.pose.position.y = float(self.get_parameter("initial_y").value)
        msg.pose.pose.position.z = float(self.get_parameter("initial_z").value)
        msg.pose.pose.orientation.x = float(self.get_parameter("initial_qx").value)
        msg.pose.pose.orientation.y = float(self.get_parameter("initial_qy").value)
        msg.pose.pose.orientation.z = float(self.get_parameter("initial_qz").value)
        msg.pose.pose.orientation.w = float(self.get_parameter("initial_qw").value)
        covariance = list(self.get_parameter("initial_covariance").value)
        if len(covariance) == 36:
            msg.pose.covariance = [float(v) for v in covariance]
        return msg

    def _publish_initial_pose_if_needed(self, now: Time) -> None:
        if self._last_amcl_pose_time is not None:
            return
        if self._wait_for_scan_before_initial_pose:
            if self._last_scan_time is None:
                return
            if now - self._last_scan_time > self._scan_timeout:
                return
        try:
            self._lookup_fresh_transform("odom", "base_footprint")
        except TransformException:
            return
        if self._last_initial_pose_pub is not None and now - self._last_initial_pose_pub < self._initial_pose_period:
            return

        self._initial_pose_msg.header.stamp = now.to_msg()
        self._initial_pose_pub.publish(self._initial_pose_msg)
        self._last_initial_pose_pub = now
        self.get_logger().info("Published bootstrap /initialpose for AMCL")

    def _lookup_fresh_transform(self, target_frame: str, source_frame: str):
        transform = self._tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            Time(),
            timeout=Duration(seconds=0.05),
        )
        stamp = Time.from_msg(transform.header.stamp)
        age = self.get_clock().now() - stamp
        if age > self._max_tf_age:
            raise TransformException(
                f"{target_frame}->{source_frame} too old ({age.nanoseconds / 1e9:.3f}s)"
            )
        return transform

    def _check_ready(self) -> None:
        if self._release_requested:
            return

        now = self.get_clock().now()
        self._publish_initial_pose_if_needed(now)

        if self._last_amcl_pose_time is None:
            self._stable_since = None
            return

        if now - self._last_amcl_pose_time > self._amcl_pose_timeout:
            self._stable_since = None
            self.get_logger().warn("AMCL pose is stale, still waiting")
            return

        try:
            self._lookup_fresh_transform("map", "odom")
            self._lookup_fresh_transform("odom", "base_footprint")
            self._lookup_fresh_transform("map", "base_footprint")
        except TransformException as exc:
            self._stable_since = None
            self.get_logger().info(f"Still waiting for stable TF: {exc}")
            return

        if self._stable_since is None:
            self._stable_since = now
            self.get_logger().info("TF chain looks good, verifying stability window")
            return

        if now - self._stable_since >= self._stable_duration:
            self.get_logger().info("TF chain stable, releasing navigation startup")
            self._release_requested = True

    @property
    def release_requested(self) -> bool:
        return self._release_requested


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavStartGate()
    try:
        while rclpy.ok() and not node.release_requested:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
