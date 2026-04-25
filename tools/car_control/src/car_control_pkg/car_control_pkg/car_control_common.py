import time
from functools import lru_cache
from pathlib import Path as FilePath

import rclpy
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String

from car_control_pkg.utils import get_action_mapping, parse_control_signal


_DEFAULT_APPROACH_STOP_XY_TOLERANCE_M = 0.10
_DEFAULT_ALIGN_STOP_YAW_TOLERANCE_RAD = 0.017453292519943295
_DEFAULT_SLOW_APPROACH_DISTANCE_M = 1.0


def _nested_get(payload: dict, *keys, default=None):
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


@lru_cache(maxsize=1)
def _load_shared_mapper_params() -> dict:
    try:
        nav_share_dir = FilePath(get_package_share_directory("nav_goal_bridge_pkg"))
    except PackageNotFoundError:
        return {}

    params_path = nav_share_dir / "config" / "mapper_params.yaml"
    try:
        with params_path.open(encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}


def _shared_goal_tolerances() -> tuple[float, float]:
    payload = _load_shared_mapper_params()
    xy_tolerance = float(
        _nested_get(
            payload,
            "car_control_node",
            "ros__parameters",
            "approach_stop_xy_tolerance_m",
            default=_nested_get(
                payload,
                "controller_server",
                "ros__parameters",
                "general_goal_checker",
                "xy_goal_tolerance",
                default=_DEFAULT_APPROACH_STOP_XY_TOLERANCE_M,
            ),
        )
    )
    yaw_tolerance = _nested_get(
        payload,
        "car_control_node",
        "ros__parameters",
        "align_stop_yaw_tolerance_rad",
        default=None,
    )
    if yaw_tolerance is None:
        yaw_tolerance = _nested_get(
            payload,
            "controller_server",
            "ros__parameters",
            "general_goal_checker",
            "yaw_goal_tolerance",
            default=_DEFAULT_ALIGN_STOP_YAW_TOLERANCE_RAD,
        )
    return xy_tolerance, float(yaw_tolerance)


def _shared_slow_approach_distance_m() -> float:
    payload = _load_shared_mapper_params()
    distance_m = _nested_get(
        payload,
        "car_control_node",
        "ros__parameters",
        "slow_approach_distance_m",
        default=_DEFAULT_SLOW_APPROACH_DISTANCE_M,
    )
    return float(distance_m)


class CarControlPublishers:
    """Class to manage common car control publishers and methods"""

    @staticmethod
    def create_publishers(node):
        """Create and return common publishers for car control"""
        rear_wheel_pub = node.create_publisher(
            Float32MultiArray, "car_C_rear_wheel", 10
        )
        front_wheel_pub = node.create_publisher(
            Float32MultiArray, "car_C_front_wheel", 10
        )

        return rear_wheel_pub, front_wheel_pub

    @staticmethod
    def create_control_subscription(node, callback):
        """Create subscription for car control signals"""
        return node.create_subscription(String, "car_control_signal", callback, 10)

    @staticmethod
    def publish_control(node, action, rear_wheel_pub, front_wheel_pub=None):
        """
        If the action is a string, it will be converted to a velocity array using the action mapping.
        If the action is a list, it will be used as the velocity array directly.
        """
        if not isinstance(action, str):
            vel = [action[0], action[1], action[0], action[1]]

        else:
            vel = get_action_mapping(action)

        if front_wheel_pub is None:
            # Only rear wheel publisher is available
            rear_msg = Float32MultiArray()
            rear_msg.data = vel  # Use entire velocity array [0:4]
            rear_wheel_pub.publish(rear_msg)
            node.get_logger().debug(f"Publishing all control data to rear wheel: {vel}")
        else:
            # Both publishers are available
            rear_msg = Float32MultiArray()
            front_msg = Float32MultiArray()
            front_msg.data = vel[0:2]
            rear_msg.data = vel[2:4]
            rear_wheel_pub.publish(rear_msg)
            front_wheel_pub.publish(front_msg)
            node.get_logger().debug(
                f"Publishing split control data: front={vel[0:2]}, rear={vel[2:4]}"
            )


class BaseCarControlNode(Node):
    """Base class for car control nodes providing common functionality"""

    def __init__(self, node_name, enable_nav_subscribers=False):
        super().__init__(node_name)
        xy_tolerance, yaw_tolerance = _shared_goal_tolerances()
        slow_approach_distance_m = _shared_slow_approach_distance_m()
        self.declare_parameter("approach_stop_xy_tolerance_m", xy_tolerance)
        self.declare_parameter("align_stop_yaw_tolerance_rad", yaw_tolerance)
        self.declare_parameter("slow_approach_distance_m", slow_approach_distance_m)

        # Create common publishers
        self.rear_wheel_pub, self.front_wheel_pub = (
            CarControlPublishers.create_publishers(self)
        )

        # Publisher to clear the plan topic
        self.plan_clear_pub = self.create_publisher(Path, '/plan', 10)

        # Create subscription to control signals
        self.subscription = CarControlPublishers.create_control_subscription(
            self, self.key_callback
        )

        # Navigation data storage
        self.latest_amcl_pose = None
        self.latest_goal_pose = None
        self.latest_global_plan = None
        self.latest_cmd_vel = None

        # Create navigation data subscribers if enabled
        if enable_nav_subscribers:
            self._create_navigation_subscribers()

    def clear_plan(self):
        """
        Clear the /plan topic by publishing an empty Path message
        and resetting internal stored plan.
        """
        empty = Path()
        empty.header.stamp = self.get_clock().now().to_msg()
        empty.header.frame_id = ''
        empty.poses = []
        self.plan_clear_pub.publish(empty)
        self.latest_global_plan = None
        self.get_logger().info('Cleared /plan topic')
        
    def _create_navigation_subscribers(self):
        """Create all subscribers needed for navigation"""
        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._amcl_callback, 10
        )

        self.goal_pose_sub = self.create_subscription(
            PoseStamped, "/goal_pose", self._goal_pose_callback, 10
        )

        self.plan_sub = self.create_subscription(
            Path, "/received_global_plan", self._global_plan_callback, 1
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist, "/cmd_vel", self.cmd_vel_callback, 10
        )

        self.get_logger().info("Navigation subscribers created")

    # Callback methods for navigation data
    def _amcl_callback(self, msg):
        """Store latest AMCL pose"""
        self.latest_amcl_pose = msg

    def _goal_pose_callback(self, msg):
        """Store latest AMCL pose"""
        self.latest_goal_pose = msg

    def _global_plan_callback(self, msg):
        """Store latest global plan"""
        self.latest_global_plan = msg

    def get_goal_pose(self):
        """Get goal position or None if unavailable"""
        if self.latest_goal_pose is None:
            return None

        try:
            return self.latest_goal_pose.pose.position
        except AttributeError:
            # Handle cases where the message structure is unexpected
            self.get_logger().warn("Goal pose has unexpected structure")
            return None

    def get_goal_orientation(self):
        """Get goal orientation or None if unavailable"""
        if self.latest_goal_pose is None:
            return None

        try:
            return self.latest_goal_pose.pose.orientation
        except AttributeError:
            self.get_logger().warn("Goal pose has unexpected structure")
            return None

    # Helper methods for navigation data access
    def get_car_position_and_orientation(self):
        """
        Get current car position and orientation (with staleness safety check)

        Returns:
            Tuple containing (position, orientation) or (None, None) if data unavailable or stale
        """
        if self.latest_amcl_pose:
            now = self.get_clock().now()
            pose_time = rclpy.time.Time.from_msg(self.latest_amcl_pose.header.stamp)
            staleness = (now - pose_time).nanoseconds / 1e9
            
            if staleness > 3.0:
                self.get_logger().warn(
                    f"Safety Stop: AMCL pose is stale by {staleness:.2f} seconds! Aborting current control.",
                    throttle_duration_sec=2.0
                )
                return None, None

            position = self.latest_amcl_pose.pose.pose.position
            orientation = self.latest_amcl_pose.pose.pose.orientation
            return position, orientation
        return None, None

    def cmd_vel_callback(self, msg: Twist):
        wheel_distance = 0.5
        max_speed = 30.0
        min_speed = -30.0
        v = msg.linear.x
        omega = msg.angular.z

        v_left = v - (wheel_distance / 2.0) * omega
        v_right = v + (wheel_distance / 2.0) * omega

        v_left = max(min_speed, min(max_speed, v_left))
        v_right = max(min_speed, min(max_speed, v_right))

        speed_msg = Float32MultiArray()
        speed_msg.data = [v_left, v_right]
        self.latest_cmd_vel = [v_left, v_right]

    def get_cmd_vel_data(self):
        return self.latest_cmd_vel

    def get_path_points(self, include_orientation=True):
        path_points = []

        plan_to_use = self.latest_global_plan
        if plan_to_use and plan_to_use.poses:
            for pose in plan_to_use.poses:
                pos = pose.pose.position
                if include_orientation:
                    # Return both position and orientation data
                    orient = pose.pose.orientation
                    path_points.append(
                        {
                            "position": [pos.x, pos.y, pos.z],
                            "orientation": [orient.x, orient.y, orient.z, orient.w],
                        }
                    )
                else:
                    path_points.append([pos.x, pos.y])
        return path_points

    # Common methods for all car control nodes
    def key_callback(self, msg):
        """Parse control signal and delegate to handle_command"""
        mode, command = parse_control_signal(msg.data)  # Parse signal
        if mode is None or command is None:
            return

        # Call the handle_command method that derived classes implement
        self.handle_command(mode, command)

    def publish_control(self, action):
        """Common method to publish control actions"""
        CarControlPublishers.publish_control(
            self, action, self.rear_wheel_pub, self.front_wheel_pub
        )

    def publish_stop_burst(self, repeat: int = 5, interval_sec: float = 0.05):
        """Publish STOP multiple times to make sure the base receives the halt command."""
        repeat = max(1, int(repeat))
        interval_sec = max(0.0, float(interval_sec))
        for idx in range(repeat):
            self.publish_control("STOP")
            if idx < repeat - 1 and interval_sec > 0.0:
                time.sleep(interval_sec)

    # If you inherit from this class, you must implement this method
    def handle_command(self, mode, command):
        """Handle parsed commands - to be implemented by subclasses"""
        # Default implementation does nothing
        pass
