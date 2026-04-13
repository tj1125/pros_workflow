import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import queue
import threading
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
import time

class ArmEnv(gym.Env):
    def __init__(self, cooldown=1):  # ✅ cooldown 改為 n 秒
        super().__init__()
        self.observation_space = Box(low=-180, high=180, shape=(3,), dtype=np.float32)  # dx, dy, dz
        self.action_space = Discrete(9)  # 8 actions + 1 (no-op)

        self.state = None
        self.prev_distance = None
        self.step_count = 0
        self.max_steps = 500
        self.global_step = 0  # 累積 step 計數

        self.last_action = None  # ✅ 記錄上一步 action
        self.smoothness_penalty = 0.1  # ✅ 平滑懲罰項，可調整大小

        rclpy.init(args=None)
        self.node = rclpy.create_node('arm_rl_env_node')

        self.arm_publisher = self.node.create_publisher(String, '/arm_control_signal', 10)
        self.obs_subscriber = self.node.create_subscription(Float32MultiArray, '/cubes_coordinates', self.obs_callback, 10)
        
        self.obs_queue = queue.Queue()
        self.ros_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.ros_thread.start()

        self.cooldown = cooldown

    def obs_callback(self, msg):
        data = list(msg.data)
        self.obs_queue.put(data)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.last_action = None  # ✅ Reset 的時候清掉 last_action

        print("[Env] Resetting: sending reset signal to all joints.")
        for joint_id in range(4):
            control_signal = String()
            control_signal.data = f"{joint_id}:b"
            self.arm_publisher.publish(control_signal)

        obs = self.obs_queue.get()
        dx, dy, dz = obs[3] - obs[0], obs[4] - obs[1], obs[5] - obs[2]
        
        self.state = np.array([dx, dy, dz], dtype=np.float32)
        self.prev_distance = np.linalg.norm([dx, dy, dz])
        
        return self.state, {}

    def step(self, action):
        time.sleep(self.cooldown)

        self.step_count += 1
        self.global_step += 1

        if action == 8:
            print("[Env] No-op action: no control sent.")
        else:
            joint_id = action // 2
            direction = "i" if action % 2 == 0 else "k"

            control_signal = String()
            control_signal.data = f"{joint_id}:{direction}"
            self.arm_publisher.publish(control_signal)
            print(f"[Env] Sent control: {control_signal.data} (👉 Unity 記得每次加/減 15 度)")

        try:
            obs = self.obs_queue.get(timeout=5.0)
        except queue.Empty:
            print("[Env] Timeout waiting Unity")
            return self.state, 0.0, True, False, {}

        dx, dy, dz = obs[3] - obs[0], obs[4] - obs[1], obs[5] - obs[2]
        current_distance = np.linalg.norm([dx, dy, dz])

        reward_distance = self.prev_distance - current_distance

        # ✅ 平滑懲罰項
        if action != self.last_action:
            reward_smoothness = -self.smoothness_penalty
        else:
            reward_smoothness = 0.0

        reward = reward_distance + reward_smoothness
        self.prev_distance = current_distance
        self.last_action = action  # 記錄下來

        self.state = np.array([dx, dy, dz], dtype=np.float32)

        terminated = current_distance < 0.1
        truncated = self.step_count >= self.max_steps

        return self.state, reward, terminated, truncated, {}

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
