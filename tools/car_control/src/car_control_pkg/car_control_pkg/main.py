import json

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from nav_msgs.msg import Path
from action_interface.action import NavGoal, ArmGoal
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
        self.arm_action_client = ActionClient(self, ArmGoal, 'arm_action_server')
        self.nav_result_pub = self.create_publisher(String, "/auto_nav/result", 10)
        self.approach_stop_xy_tolerance_m = float(
            self.car_control_node.get_parameter("approach_stop_xy_tolerance_m").value
        )
        self.align_stop_yaw_tolerance_rad = float(
            self.car_control_node.get_parameter("align_stop_yaw_tolerance_rad").value
        )
        
        self.plan_sub = self.create_subscription(Path, '/received_global_plan', self.plan_callback, 10)
        self.navigating = False

    def plan_callback(self, msg):
        if not msg.poses:
            return
            
        if not self.navigating:
            # 檢查是否已經到達終點
            car_position, car_orientation = self.car_control_node.get_car_position_and_orientation()
            goal_pose = self.car_control_node.get_goal_pose()
            goal_orientation = self.car_control_node.get_goal_orientation()
            if car_position and car_orientation and goal_pose and goal_orientation:
                target_distance = cal_distance(
                    [car_position.x, car_position.y],
                    [goal_pose.x, goal_pose.y],
                )
                heading_error = calculate_goal_heading_error(
                    [car_orientation.z, car_orientation.w],
                    [goal_orientation.z, goal_orientation.w],
                )
                if (
                    target_distance <= self.approach_stop_xy_tolerance_m
                    and abs(heading_error) <= self.align_stop_yaw_tolerance_rad
                ):
                    return
                    
            self.get_logger().info('收到新路徑，啟動全自動導航 (Auto Navigation)')
            self.start_auto_nav()

    def _publish_nav_result(self, success: bool, message: str):
        payload = {
            "success": bool(success),
            "message": str(message),
            "source": "auto_nav_starter",
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.nav_result_pub.publish(msg)

    def start_auto_nav(self):
        self.navigating = True
        
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

        # 確認並發送手臂目標
        if self.arm_action_client.wait_for_server(timeout_sec=1.0):
            arm_goal_msg = ArmGoal.Goal()
            arm_goal_msg.mode = 'catch'
            self.arm_action_client.send_goal_async(arm_goal_msg)
        else:
            self.get_logger().warn('arm_action_server 不可用 (不影響導航)')

    def nav_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('自動導航請求被拒絕')
            self._publish_nav_result(False, "navigation goal rejected")
            self.navigating = False
            return
            
        self.get_logger().info('自動導航請求已接受，開始導航...')
        self.nav_result_future = goal_handle.get_result_async()
        self.nav_result_future.add_done_callback(self.nav_get_result_callback)

    def nav_get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'自動導航結束: {result.message}')
        self._publish_nav_result(getattr(result, "success", False), result.message)
        self.navigating = False


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
