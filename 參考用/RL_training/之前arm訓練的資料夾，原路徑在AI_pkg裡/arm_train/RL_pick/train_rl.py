# train_rl.py
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Float32MultiArray
from action_interface.action import ArmGoal
from db_helper import DBHelper
import numpy as np
import pickle
import random

class RLTrainer(Node):
    def __init__(self):
        super().__init__('rl_trainer')

        # DB helper，載入 tree
        self.db = DBHelper()

        # Action client
        self._arm_client = ActionClient(self, ArmGoal, '/arm_action_server')

        # cubes position subscriber
        self.obs = None
        self.create_subscription(
            Float32MultiArray,
            '/cubes_coordinates',
            self.obs_callback,
            10
        )

        # Q‐table 存放在 { node_id: {direction: q_value, …}, …}
        self.Q = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

        # tree 方向列表
        self.directions = ["up","down","left","right","forward","backward"]

    def obs_callback(self, msg: Float32MultiArray):
        self.obs = np.array(msg.data)

    def send_arm_command(self, direction: str) -> bool:
        """把 direction (up/down/…) 當作一個 Goal 送給 ArmActionServer"""
        goal = ArmGoal.Goal()
        goal.mode = direction
        self._arm_client.wait_for_server()
        send_fut = self._arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut)
        handle = send_fut.result()
        if not handle.accepted:
            self.get_logger().warn(f"Goal {direction} rejected")
            return False

        res_fut = handle.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        return res_fut.result().result.success

    def train(self, episodes=1000):
        for ep in range(episodes):
            # reset to root node & get observation
            current_node = 1  # 假設 root 節點 id = 1
            obs = None
            while obs is None:
                rclpy.spin_once(self)
                obs = self.obs

            done = False
            steps = 0
            while not done and steps<100:
                # ε‐greedy 選 action
                if current_node not in self.Q:
                    self.Q[current_node] = {d:0.0 for d in self.directions}

                if random.random() < self.epsilon:
                    action = random.choice(self.directions)
                else:
                    # pick best direction
                    action = max(self.Q[current_node], key=self.Q[current_node].get)

                # 執行 action
                success = self.send_arm_command(action)
                if not success:
                    reward = -1.0
                    new_obs = obs
                    next_node = current_node
                else:
                    # 等待新座標
                    new_obs = None
                    while new_obs is None:
                        rclpy.spin_once(self)
                        new_obs = self.obs

                    # reward = 距離變化
                    old_dist = np.linalg.norm(obs[:3]-obs[3:])
                    new_dist = np.linalg.norm(new_obs[:3]-new_obs[3:])
                    reward = (old_dist - new_dist)  + (1.0 if new_dist<old_dist else -0.5)

                    # 根據 tree 找下個 node_id
                    # edges table: from_node, to_node, direction
                    next_node = self.db.get_next_node(current_node, action)

                # Q update
                best_next = max(self.Q.get(next_node, {d:0.0 for d in self.directions}).values())
                self.Q[current_node][action] += self.alpha*(reward + self.gamma*best_next - self.Q[current_node][action])

                current_node, obs = next_node, new_obs
                if new_dist<0.1:  # 假設夾到目標就結束
                    done = True
                steps += 1

            if ep%50==0:
                self.get_logger().info(f"Episode {ep} done")

        # 存 Q-table
        with open('qtable.pkl','wb') as f:
            pickle.dump(self.Q, f)
        self.get_logger().info("Training complete, Q-table saved to qtable.pkl")


def main(args=None):
    rclpy.init(args=args)
    trainer = RLTrainer()
    try:
        trainer.train()
    finally:
        trainer.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()