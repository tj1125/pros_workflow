import json

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from action_interface.action import NavGoal
from std_msgs.msg import String
from car_control_pkg.car_action_server import NavigationActionServer
from car_control_pkg.car_control_common import BaseCarControlNode
from car_control_pkg.car_manual import ManualControlNode
from car_control_pkg.nav2_utils import cal_distance, calculate_goal_heading_error


class AutoNavStarter(Node):
    def __init__(self, car_control_node):
        super().__init__('auto_nav_starter')
        self.car_control_node = car_control_node
        self.nav_action_client = ActionClient(self, NavGoal, 'nav_action_server')
        self.nav_result_pub = self.create_publisher(String, "/auto_nav/result", 10)
        self.approach_stop_xy_tolerance_m = float(
            self.car_control_node.get_parameter("approach_stop_xy_tolerance_m").value
        )
        self.align_stop_yaw_tolerance_rad = float(
            self.car_control_node.get_parameter("align_stop_yaw_tolerance_rad").value
        )
        
        self.goal_pose_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.goal_pose_callback, 10
        )
        self.plan_sub = self.create_subscription(
            Path, '/received_global_plan', self.plan_callback, 10
        )
        self.navigating = False
        self._active_goal_key = None
        self._completed_goal_key = None
        self._latest_goal_key = None

    @staticmethod
    def _goal_key(goal_pose, goal_orientation):
        if goal_pose is None or goal_orientation is None:
            return None
        return (
            round(float(goal_pose.x), 3),
            round(float(goal_pose.y), 3),
            round(float(goal_orientation.z), 3),
            round(float(goal_orientation.w), 3),
        )

    def goal_pose_callback(self, msg):
        goal_key = self._goal_key(msg.pose.position, msg.pose.orientation)
        if goal_key is None:
            return

        self._latest_goal_key = goal_key
        if goal_key != self._completed_goal_key:
            self._completed_goal_key = None

        if self.navigating and goal_key != self._active_goal_key:
            self._active_goal_key = goal_key
            self.get_logger().info(
                '導航中收到新 /goal_pose，沿用目前自動導航 action 並更新目標'
            )

    def _current_goal_is_satisfied(self, car_position, car_orientation, goal_pose, goal_orientation):
        if not car_position or not car_orientation or goal_pose is None or goal_orientation is None:
            return False

        target_distance = cal_distance(
            [car_position.x, car_position.y],
            [goal_pose.x, goal_pose.y],
        )
        heading_error = calculate_goal_heading_error(
            [car_orientation.z, car_orientation.w],
            [goal_orientation.z, goal_orientation.w],
        )
        return (
            target_distance <= self.approach_stop_xy_tolerance_m
            and abs(heading_error) <= self.align_stop_yaw_tolerance_rad
        )

    def plan_callback(self, msg):
        if not msg.poses:
            return

        car_position, car_orientation = self.car_control_node.get_car_position_and_orientation()
        goal_pose = self.car_control_node.get_goal_pose()
        goal_orientation = self.car_control_node.get_goal_orientation()
        goal_key = self._latest_goal_key or self._goal_key(goal_pose, goal_orientation)
        if goal_key is None:
            return

        if self.navigating:
            if goal_key != self._active_goal_key:
                self._active_goal_key = goal_key
                self.get_logger().info(
                    '導航中收到新路徑，沿用目前自動導航 action 並更新目標'
                )
            return

        if goal_key == self._completed_goal_key:
            self.get_logger().debug('Skipping already completed auto-nav goal')
            return

        if self._current_goal_is_satisfied(
            car_position, car_orientation, goal_pose, goal_orientation
        ):
            self._completed_goal_key = goal_key
            return

        self.get_logger().info('收到新路徑，啟動全自動導航 (Auto Navigation)')
        self.start_auto_nav(goal_key=goal_key)

    def _publish_nav_result(self, success: bool, message: str):
        payload = {
            "success": bool(success),
            "message": str(message),
            "source": "auto_nav_starter",
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.nav_result_pub.publish(msg)

    def start_auto_nav(self, goal_key=None):
        self.navigating = True
        self._active_goal_key = goal_key
        
        # 確認並發送導航目標
        if self.nav_action_client.wait_for_server(timeout_sec=1.0):
            nav_goal_msg = NavGoal.Goal()
            nav_goal_msg.mode = 'Manual_Nav'
            self.nav_send_goal_future = self.nav_action_client.send_goal_async(nav_goal_msg)
            self.nav_send_goal_future.add_done_callback(self.nav_goal_response_callback)
        else:
            self.get_logger().warn('nav_action_server 不可用')
            self._publish_nav_result(False, "nav_action_server unavailable")
            self.navigating = False
            self._active_goal_key = None

    def nav_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('自動導航請求被拒絕')
            self._publish_nav_result(False, "navigation goal rejected")
            self.navigating = False
            self._active_goal_key = None
            return
            
        self.get_logger().info('自動導航請求已接受，開始導航...')
        self.nav_result_future = goal_handle.get_result_async()
        self.nav_result_future.add_done_callback(self.nav_get_result_callback)

    def nav_get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'自動導航結束: {result.message}')
        self._publish_nav_result(getattr(result, "success", False), result.message)
        if getattr(result, "success", False):
            completed_goal_key = self._goal_key(
                self.car_control_node.get_goal_pose(),
                self.car_control_node.get_goal_orientation(),
            )
            self._completed_goal_key = completed_goal_key or self._active_goal_key
            if str(result.message).startswith("Navigation goal reached successfully."):
                self.car_control_node.clear_plan()
        self.navigating = False
        self._active_goal_key = None


def main(args=None):
    rclpy.init(args=args)
    car_control_node = BaseCarControlNode(
        node_name="car_control_node", enable_nav_subscribers=True
    )
    manual_control_node = ManualControlNode()
    action_server = NavigationActionServer(car_control_node=car_control_node)
    auto_starter = AutoNavStarter(car_control_node=car_control_node)
    
    executor = MultiThreadedExecutor()
    executor.add_node(car_control_node)
    executor.add_node(action_server)
    executor.add_node(manual_control_node)
    executor.add_node(auto_starter)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        action_server.get_logger().info("Keyboard interrupt, shutting down...")
    finally:
        auto_starter.destroy_node()
        action_server.destroy_node()
        manual_control_node.destroy_node()
        car_control_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
