import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class CmdVelAdapter(Node):
    def __init__(self) -> None:
        super().__init__("cmd_vel_adapter")
        self.declare_parameter("max_linear_velocity", 0.26)
        self.declare_parameter("max_angular_velocity", 1.0)
        self.declare_parameter("max_wheel_command", 20.0)
        self.declare_parameter("cmd_vel_timeout_sec", 0.5)
        self.declare_parameter("publish_period_sec", 0.05)

        max_linear_velocity = abs(float(self.get_parameter("max_linear_velocity").value))
        max_angular_velocity = abs(float(self.get_parameter("max_angular_velocity").value))
        self._max_wheel_command = abs(float(self.get_parameter("max_wheel_command").value))
        self._linear_scale = (
            self._max_wheel_command / max_linear_velocity
            if max_linear_velocity > 0.0
            else 0.0
        )
        self._angular_scale = (
            self._max_wheel_command / max_angular_velocity
            if max_angular_velocity > 0.0
            else 0.0
        )
        self._cmd_vel_timeout = Duration(
            seconds=float(self.get_parameter("cmd_vel_timeout_sec").value)
        )
        self._last_cmd_vel_time = None
        self._stopped_for_timeout = False

        self._front_pub = self.create_publisher(Float32MultiArray, "/car_C_front_wheel", 10)
        self._rear_pub = self.create_publisher(Float32MultiArray, "/car_C_rear_wheel", 10)

        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_callback, 10)
        self.create_timer(
            float(self.get_parameter("publish_period_sec").value),
            self._watchdog_timer_callback,
        )

        self._publish_wheel_command(0.0, 0.0)
        self.get_logger().info(
            "CmdVel adapter started: /cmd_vel -> /car_C_front_wheel,/car_C_rear_wheel "
            f"(max_linear_velocity={max_linear_velocity:.3f}, "
            f"max_angular_velocity={max_angular_velocity:.3f}, "
            f"linear_scale={self._linear_scale:.3f}, "
            f"angular_scale={self._angular_scale:.3f}, "
            f"max_wheel_command={self._max_wheel_command:.3f}, "
            f"timeout={self._cmd_vel_timeout.nanoseconds / 1e9:.2f}s)"
        )

    def _clamp(self, value: float) -> float:
        return max(-self._max_wheel_command, min(self._max_wheel_command, value))

    def _publish_wheel_command(self, left: float, right: float) -> None:
        left = self._clamp(left)
        right = self._clamp(right)

        front_msg = Float32MultiArray()
        rear_msg = Float32MultiArray()
        front_msg.data = [left, right]
        rear_msg.data = [left, right]
        self._front_pub.publish(front_msg)
        self._rear_pub.publish(rear_msg)

    def _cmd_vel_callback(self, msg: Twist) -> None:
        left = (
            float(msg.linear.x) * self._linear_scale
            - float(msg.angular.z) * self._angular_scale
        )
        right = (
            float(msg.linear.x) * self._linear_scale
            + float(msg.angular.z) * self._angular_scale
        )
        self._last_cmd_vel_time = self.get_clock().now()
        self._stopped_for_timeout = False
        self._publish_wheel_command(left, right)

    def _watchdog_timer_callback(self) -> None:
        if self._last_cmd_vel_time is None:
            return
        if self.get_clock().now() - self._last_cmd_vel_time <= self._cmd_vel_timeout:
            return
        if self._stopped_for_timeout:
            return

        self._publish_wheel_command(0.0, 0.0)
        self._stopped_for_timeout = True
        self.get_logger().warn("cmd_vel timed out, publishing STOP to wheel topics")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_wheel_command(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
