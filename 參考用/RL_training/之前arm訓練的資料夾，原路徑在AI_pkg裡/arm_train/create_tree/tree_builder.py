import rclpy
from rclpy.node import Node
from action_interface.action import ArmGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.action import ActionClient
from db_helper import DBHelper
import queue
import math


class TreeBuilder(Node):
    def __init__(self):
        super().__init__('arm_tree_builder')
        self.db = DBHelper()
        self.angle_sub = self.create_subscription(
            JointTrajectoryPoint,
            '/robot_arm',
            self.angle_callback,
            10
        )
        self.arm_client = ActionClient(self, ArmGoal, '/arm_action_server')
        self.latest_angles = None
        self.queue = queue.Queue()
        self.max_depth = 20
        self.directions = ["up", "down", "left", "right", "forward", "backward"]
        self.opposite = {
            "up": "down", "down": "up",
            "left": "right", "right": "left",
            "forward": "backward", "backward": "forward"
        }
        self.visited_keys = set()

    def angle_callback(self, msg):
        self.latest_angles = [round(a * 180 / math.pi) for a in msg.positions]

    def wait_for_result(self):
        while self.latest_angles is None:
            rclpy.spin_once(self)
        result = self.latest_angles
        self.latest_angles = None
        return result

    def reset_to_angles(self, angles):
        print(f"[RESET] Resetting arm to angles: {angles}")
        msg = JointTrajectoryPoint()
        msg.positions = [a * math.pi / 180 for a in angles]
        self.angle_callback(msg)  # 模擬直接接收 callback 資訊

    def send_arm_command(self, direction):
        goal_msg = ArmGoal.Goal()
        goal_msg.mode = direction
        self.arm_client.wait_for_server()
        future = self.arm_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f"Goal {direction} rejected")
            return None

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result().result
        if result.success:
            return self.wait_for_result()
        else:
            return None

    def run(self):
        reset_q = [90, 30, 160, 180]  # degrees
        root_key = self.make_key(reset_q)
        root_id = self.db.get_or_create_node(root_key, reset_q, [0]*6)
        self.visited_keys.add(root_key)
        print(f"[VISITED] {root_key} (id={root_id})")
        self.queue.put((root_id, reset_q, 0, None))  # (node_id, angles, depth, from_dir)

        while not self.queue.empty():
            node_id, angles, depth, from_dir = self.queue.get()
            if depth >= self.max_depth:
                continue

            self.reset_to_angles(angles)

            for direction in self.directions:
                if from_dir and direction == self.opposite[from_dir]:
                    continue
                if self.db.edge_exists(node_id, direction):
                    continue
                print(f"Exploring {direction} from node {node_id}...")
                result = self.send_arm_command(direction)
                if result:
                    new_key = self.make_key(result[:4])
                    if new_key in self.visited_keys:
                        print(f"[SKIP] Already visited {new_key}")
                        existing_id = self.db.get_node_id_by_key(new_key)
                        self.db.insert_edge(node_id, existing_id, direction)
                        continue
                    new_id = self.db.get_or_create_node(new_key, result[:4], [0]*6)
                    self.db.insert_edge(node_id, new_id, direction)
                    self.queue.put((new_id, result[:4], depth + 1, direction))
                    self.visited_keys.add(new_key)
                    print(f"[VISITED] {new_key} (id={new_id})")

    def make_key(self, angles):
        return f"q1_{angles[0]}_q2_{angles[1]}_q3_{angles[2]}_q4_{angles[3]}"


def main():
    rclpy.init()
    node = TreeBuilder()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()