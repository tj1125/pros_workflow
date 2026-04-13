from __future__ import annotations

import math
import time
from typing import Any, Dict, Optional, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - fallback for older gym
    import gym
    from gym import spaces

import numpy as np


class SimpleArmEnv(gym.Env):
    """Lightweight discrete arm environment with reward shaping aligned to Unity env."""

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 30}

    OPPOSITE_ACTIONS = {
        0: 1,
        1: 0,
        2: 3,
        3: 2,
        4: 5,
        5: 4,
        6: 6,
    }

    def __init__(
        self,
        action_scale: float = 0.05,
        workspace_low: Tuple[float, float, float] = (-1.0, -1.0, 0.0),
        workspace_high: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        success_threshold: float = 0.3,
        max_steps: int = 80,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
        reward_cfg: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -2.0, -2.0], dtype=np.float32),
            high=np.array([2.0, 2.0, 2.0], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

        self.action_scale = float(action_scale)
        self.workspace_low = np.array(workspace_low, dtype=np.float32)
        self.workspace_high = np.array(workspace_high, dtype=np.float32)
        self.success_threshold = float(success_threshold)
        self.max_steps = int(max_steps)
        self._rng = np.random.default_rng(seed)
        self.render_mode = render_mode

        self._arm_pos = np.zeros(3, dtype=np.float32)
        self._target_pos = np.zeros(3, dtype=np.float32)
        self._step_count = 0

        self.initial_distance: Optional[float] = None
        self.previous_distance: Optional[float] = None
        self.previous_action: Optional[int] = None
        self.prev_prev_action: Optional[int] = None
        self._milestones_hit: set[float] = set()
        self._distance_window: deque[float] = deque(maxlen=5)

        self.reward_cfg: Dict[str, float] = {
            "dist_scale": 2.0,
            "delta_smooth_kappa": 2.0,
            "success_bonus": 5.0,
            "milestone_bonus": 0.5,
            "smooth_penalty": -0.05,
            "jitter_penalty": -0.05,
            "energy_penalty": -0.02,
            "time_penalty": -0.01,
            "stagnation_penalty": -0.5,
            "stagnation_window": 5,
            "stagnation_tolerance": 0.01,
        }
        if reward_cfg:
            self.reward_cfg.update(reward_cfg)
        self._distance_window = deque(maxlen=int(self.reward_cfg["stagnation_window"]))

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)
        return [seed]

    def _sample_workspace(self) -> np.ndarray:
        return self._rng.uniform(self.workspace_low, self.workspace_high).astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        return (self._target_pos - self._arm_pos).astype(np.float32)

    def _compute_delta(self, dist: float) -> Tuple[float, float]:
        prev_dist = self.previous_distance if self.previous_distance is not None else dist
        if self.initial_distance is None or self.initial_distance < 1e-6:
            return 0.0, 0.0
        delta_raw = (prev_dist - dist) / max(self.initial_distance, 1e-6)
        kappa = float(self.reward_cfg["delta_smooth_kappa"])
        delta_smooth = math.tanh(kappa * delta_raw) if kappa > 0 else delta_raw
        return delta_raw, delta_smooth

    # ---------------------------------------------------------------------
    # Gym API
    # ---------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed(seed)
        self._arm_pos = self._sample_workspace()
        self._target_pos = self._sample_workspace()
        obs = self._get_obs()

        dist = float(np.linalg.norm(obs))
        self.initial_distance = dist
        self.previous_distance = dist
        self.previous_action = 6  # treat as no-op
        self.prev_prev_action = None
        self._milestones_hit.clear()
        self._step_count = 0
        self._distance_window.clear()

        info = {"distance": dist}
        return obs, info

    def step(self, action: int):
        self._step_count += 1

        delta = np.zeros(3, dtype=np.float32)
        if action == 0:
            delta[0] += self.action_scale
        elif action == 1:
            delta[0] -= self.action_scale
        elif action == 2:
            delta[1] += self.action_scale
        elif action == 3:
            delta[1] -= self.action_scale
        elif action == 4:
            delta[2] += self.action_scale
        elif action == 5:
            delta[2] -= self.action_scale

        self._arm_pos = np.clip(self._arm_pos + delta, self.workspace_low, self.workspace_high)

        obs = self._get_obs()
        dist = float(np.linalg.norm(obs))
        delta_raw, delta_smooth = self._compute_delta(dist)

        reward = self.reward_cfg["dist_scale"] * delta_smooth
        reward_components = {
            "distance": dist,
            "delta": delta_raw,
            "delta_smooth": delta_smooth,
            "r_dist": reward,
        }

        success = dist < self.success_threshold
        if success:
            reward += self.reward_cfg["success_bonus"]
            reward_components["r_success"] = self.reward_cfg["success_bonus"]

        if self.initial_distance is not None:
            for frac in (0.75, 0.5, 0.25):
                if frac in self._milestones_hit:
                    continue
                if dist <= frac * self.initial_distance:
                    reward += self.reward_cfg["milestone_bonus"]
                    reward_components.setdefault("r_milestone", 0.0)
                    reward_components["r_milestone"] += self.reward_cfg["milestone_bonus"]
                    self._milestones_hit.add(frac)

        if self.previous_action is not None and action != self.previous_action:
            reward += self.reward_cfg["smooth_penalty"]
            reward_components["r_smooth"] = self.reward_cfg["smooth_penalty"]

        if self.prev_prev_action is not None and self.previous_action is not None:
            opposite = self.OPPOSITE_ACTIONS.get(self.prev_prev_action)
            if opposite is not None and self.previous_action == opposite and action == self.prev_prev_action:
                reward += self.reward_cfg["jitter_penalty"]
                reward_components["r_jitter"] = self.reward_cfg["jitter_penalty"]

        if action != 6:
            reward += self.reward_cfg["energy_penalty"]
            reward_components["r_energy"] = self.reward_cfg["energy_penalty"]

        reward += self.reward_cfg["time_penalty"]
        reward_components["r_time"] = self.reward_cfg["time_penalty"]

        self._distance_window.append(dist)
        if len(self._distance_window) == self._distance_window.maxlen:
            tolerance = float(self.reward_cfg["stagnation_tolerance"])
            if max(self._distance_window) - min(self._distance_window) <= tolerance:
                penalty = self.reward_cfg["stagnation_penalty"]
                reward += penalty
                reward_components["r_stagnation"] = penalty

        info = {
            "distance": dist,
            "reward_components": reward_components,
            "is_success": success,
        }

        self.prev_prev_action = self.previous_action
        self.previous_action = action
        self.previous_distance = dist

        terminated = success
        truncated = self._step_count >= self.max_steps and not terminated

        time.sleep(0.5)

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Arm: {self._arm_pos}, Target: {self._target_pos}, d={np.linalg.norm(self._get_obs()):.3f}")
        elif self.render_mode == "ansi":
            return f"Arm: {self._arm_pos}, Target: {self._target_pos}"

    def close(self):
        pass


def make_single_env(**kwargs):
    return SimpleArmEnv(**kwargs)
