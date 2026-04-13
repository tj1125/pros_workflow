import math
import time
from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import Float32MultiArray, String, Bool
from std_srvs.srv import Empty

from action_interface.action import ArmGoal  # Replace with your actual action definition


class UnityArmEnv(gym.Env, Node):
    """
    Gym environment for a Unity-based robot arm using discrete actions via ROS2.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    OPPOSITE_ACTIONS = {
        "forward": "backward",
        "backward": "forward",
        "left": "right",
        "right": "left",
        "up": "down",
        "down": "up",
        "elbow_forward": "elbow_backward",
        "elbow_backward": "elbow_forward",
        "elbow_left": "elbow_right",
        "elbow_right": "elbow_left",
        "none": "none",
    }

    def __init__(self, max_steps=80, seed=0, reward_cfg: Optional[dict] = None):
        if not rclpy.ok():
            rclpy.init()

        super().__init__("unity_arm_env_client")

        self.directions = [
            "up", "down", "left", "right", "forward", "backward",
            "elbow_backward", "elbow_forward", "elbow_left", "elbow_right", "none"
        ]
        self.action_space = gym.spaces.Discrete(len(self.directions))
        # The observation is the difference between target and gripper coordinates (dx, dy, dz)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

        self.current_obs = None
        self.current_step = 0

        # State variables for reward calculation
        self.initial_distance = None
        self.previous_distance = None
        self.previous_action = None
        self.prev_prev_action = None
        self._milestones_hit = set()
        self._distance_window = deque(maxlen=5)

        self.reward_cfg = {
            "dist_scale": 2.0,
            "delta_smooth_kappa": 2.0,
            "success_threshold": 0.3,
            "success_bonus": 5.0,
            "milestone_fractions": (0.75, 0.5, 0.25),
            "milestone_bonus": 0.5,
            "smooth_penalty": -0.05,
            "jitter_penalty": -0.05,
            "energy_penalty": -0.02,
            "time_penalty": -0.01,
            "hf_lambda": 0.5,
            "hf_weight": 1.0,
            "align_weight": 1.0,
            "safety_penalty": 0.0,
            "stagnation_penalty": -0.5,
            "stagnation_window": 5,
            "stagnation_tolerance": 0.01,
        }
        if reward_cfg:
            self.reward_cfg.update(reward_cfg)

        self._distance_window = deque(maxlen=int(self.reward_cfg["stagnation_window"]))

        self.max_steps = max_steps

        # Subscribe to cube coordinates
        self._subscribe_coords()
        
        # Removed: reset publisher for initial joint angles

        # Publisher to signal Unity scene reset
        self.reset_unity_publisher = self.create_publisher(Bool, '/reset_unity', 10)

        self._action_client = ActionClient(self, ArmGoal, "/arm_action_server")

        # External callers control when to reset; no auto-reset here.

    def _obs_callback(self, msg):
        raw_obs = np.array(msg.data, dtype=np.float32)
        if raw_obs.shape == (6,):
            gripper_coords = raw_obs[0:3]
            target_coords = raw_obs[3:6]
            self.current_obs = target_coords - gripper_coords
        else:
            # Log an error if the shape is not as expected
            self.get_logger().error(f"Received observation with unexpected shape: {raw_obs.shape}")

    def _subscribe_coords(self):
        try:
            if hasattr(self, 'subscription') and self.subscription is not None:
                # Clean up any previous subscription to avoid duplicate callbacks
                self.destroy_subscription(self.subscription)
        except Exception:
            pass
        self.subscription = self.create_subscription(
            Float32MultiArray,
            "/cubes_coordinates",
            self._obs_callback,
            10
        )

    def trigger_reset_signal(self):
        try:
            self.reset_unity_publisher.publish(Bool(data=True))
            # self.get_logger().info("Manual /reset_unity signal sent")
        except Exception as e:
            self.get_logger().error(f"Failed to publish manual /reset_unity: {e}")

    def step(self, action: str):
        self.current_step += 1
        self.get_logger().info(f"Step {self.current_step}")

        # --- 1. Send action to environment ---
        direction = action
        if direction not in self.directions:
            self.get_logger().error(f"Invalid action: '{direction}'. Valid actions are: {self.directions}")
            # On invalid action, return a large penalty and end the episode
            return self.current_obs, -10.0, True, False, {"error": "Invalid action"}

        goal_msg = ArmGoal.Goal()
        goal_msg.mode = direction
        self.get_logger().info(f"Sending direction: {direction}")
        self._action_client.wait_for_server()
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected")
            return self.current_obs, -10.0, True, False, {"error": "Goal rejected"}

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        # Wait for the next observation to be updated by the callback
        rclpy.spin_once(self, timeout_sec=1.0) # Give time for obs to update
        obs = self.current_obs

        # --- 2. Calculate Reward based on the new state ---
        reward_components = {}
        total_reward = 0.0
        current_distance = float(np.linalg.norm(obs))
        reward_components["distance"] = current_distance

        prev_distance = self.previous_distance if self.previous_distance is not None else current_distance
        delta_raw = 0.0
        if self.initial_distance is not None and self.initial_distance > 1e-6:
            delta_raw = (prev_distance - current_distance) / max(self.initial_distance, 1e-6)
        kappa = float(self.reward_cfg["delta_smooth_kappa"])
        delta_smooth = math.tanh(kappa * delta_raw) if kappa > 0 else delta_raw
        r_dist = self.reward_cfg["dist_scale"] * delta_smooth
        total_reward += r_dist
        reward_components["r_dist"] = r_dist
        reward_components["delta"] = delta_raw
        reward_components["delta_smooth"] = delta_smooth

        is_success = current_distance < self.reward_cfg["success_threshold"]
        if is_success:
            total_reward += self.reward_cfg["success_bonus"]
            reward_components["r_success"] = self.reward_cfg["success_bonus"]

        if self.initial_distance is not None:
            for frac in self.reward_cfg["milestone_fractions"]:
                if frac in self._milestones_hit:
                    continue
                if current_distance <= frac * self.initial_distance:
                    total_reward += self.reward_cfg["milestone_bonus"]
                    reward_components.setdefault("r_milestone", 0.0)
                    reward_components["r_milestone"] += self.reward_cfg["milestone_bonus"]
                    self._milestones_hit.add(frac)

        if self.previous_action is not None and action != self.previous_action:
            total_reward += self.reward_cfg["smooth_penalty"]
            reward_components["r_smooth"] = self.reward_cfg["smooth_penalty"]

        jitter_penalty = 0.0
        if self.prev_prev_action is not None and self.previous_action is not None:
            opp = self.OPPOSITE_ACTIONS.get(self.prev_prev_action)
            if opp is not None and self.previous_action == opp and action == self.prev_prev_action:
                jitter_penalty = self.reward_cfg["jitter_penalty"]
                total_reward += jitter_penalty
                reward_components["r_jitter"] = jitter_penalty

        if action != "none":
            total_reward += self.reward_cfg["energy_penalty"]
            reward_components["r_energy"] = self.reward_cfg["energy_penalty"]

        total_reward += self.reward_cfg["time_penalty"]
        reward_components["r_time"] = self.reward_cfg["time_penalty"]

        reward_components["r_hf"] = 0.0
        reward_components["r_align"] = 0.0
        reward_components["r_safety"] = 0.0

        self._distance_window.append(current_distance)
        if len(self._distance_window) == self._distance_window.maxlen:
            tolerance = float(self.reward_cfg["stagnation_tolerance"])
            if max(self._distance_window) - min(self._distance_window) <= tolerance:
                penalty = self.reward_cfg["stagnation_penalty"]
                total_reward += penalty
                reward_components["r_stagnation"] = penalty

        self.prev_prev_action = self.previous_action
        self.previous_action = action
        self.previous_distance = current_distance

        terminated = is_success
        truncated = False

        info = {
            "distance": current_distance,
            "delta_distance": prev_distance - current_distance,
            "is_success": bool(is_success),
            "reward_components": reward_components,
        }

        if (not is_success) and self.current_step >= self.max_steps:
            # self.get_logger().info("Max steps reached without success; forcing reset")
            if hasattr(self, "trigger_reset_signal"):
                try:
                    self.trigger_reset_signal()
                except Exception:
                    pass
            reset_obs, reset_info = self.reset()
            if reset_info is None:
                reset_info = {}
            else:
                reset_info = dict(reset_info)
            reset_info["forced_reset"] = True
            return reset_obs, total_reward, False, False, reset_info

        return obs, total_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.get_logger().info("Resetting environment...")

        # Send an init pose command via the action server (same as step logic)
        try:
            goal_msg = ArmGoal.Goal()
            goal_msg.mode = "init_pose"
            # self.get_logger().info("Sending init_pose to /arm_action_server")
            self._action_client.wait_for_server()
            send_goal_future = self._action_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, send_goal_future)
            goal_handle = send_goal_future.result()

            if not goal_handle or not goal_handle.accepted:
                self.get_logger().warn("init_pose goal rejected; proceeding to wait for observation")
            else:
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future)
                # Give a moment for obs to update after reset action
                rclpy.spin_once(self, timeout_sec=1.0)
        except Exception as e:
            self.get_logger().error(f"Failed to send init_pose: {e}")

        # Prepare to receive a fresh observation
        self.current_obs = None

        # Publish reset signal to Unity
        try:
            # self.get_logger().info("Publishing True to /reset_unity")
            self.reset_unity_publisher.publish(Bool(data=True))
        except Exception as e:
            self.get_logger().error(f"Failed to publish /reset_unity: {e}")

        # Wait for a new observation after reset
        self.get_logger().info("Waiting for observation after reset...")
        # Give Unity a brief moment
        time.sleep(0.5)
        # Keep existing subscription to avoid missing the first post-reset message
        start = time.monotonic()
        timeout_s = 5.0
        while self.current_obs is None and (time.monotonic() - start) < timeout_s:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.current_obs is None:
            self.get_logger().warn("Timeout waiting for /cubes_coordinates after reset; proceeding with zeros.")
            self.current_obs = np.zeros(3, dtype=np.float32)

        # Initialize reward-related state
        self.initial_distance = np.linalg.norm(self.current_obs)
        self.previous_distance = self.initial_distance
        self.previous_action = "none"
        self.prev_prev_action = None
        self._milestones_hit = set()
        self._distance_window.clear()

        self.get_logger().info("Reset complete.")
        return self.current_obs, {}

    def render(self, mode="human"):
        pass

    def close(self):
        self.get_logger().info("Shutting down ROS2 node.")
        self.destroy_node()
        rclpy.shutdown()
