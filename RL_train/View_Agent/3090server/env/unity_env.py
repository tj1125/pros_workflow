"""
env/unity_env.py — Gymnasium Environment Wrapper for Unity (via Rosbridge)

Wraps the Unity digital twin as a gymnasium.Env for SAC training.
Communication uses Rosbridge WebSocket (same as TrajectoryRecorder).

Observation: np.ndarray (FRAME_STACK, FUSED_DIM) — stacked visual features
Action:      np.ndarray (6,) — 6-DOF joint delta in [-1, 1] (scaled by action_scale)
Reward:      float — physics-based reward (occlusion, centering, stability)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from collector.trajectory_recorder import TrajectoryRecorder
from collector.randomizer import EnvRandomizer
from env.obs_builder import ObsBuilder, FRAME_STACK, FUSED_DIM
from rl.reward_fn import RewardFn

logger = logging.getLogger(__name__)

ACTION_SCALE  = 0.05    # Max delta per joint per step (rad)
MAX_STEPS     = 80      # Episode length
CONTROL_HZ    = 10      # Hz (must match config)


class UnityViewEnv(gym.Env):
    """
    Gymnasium environment for View Agent SAC training.

    Reset: randomizes Unity scene, clears ObsBuilder buffers.
    Step:  publishes action to Unity, waits one control period,
           reads new observation, computes physics reward.

    Observation space: Box(FRAME_STACK, FUSED_DIM)   — visual feature stack
    Action space:      Box(-1, 1, (6,))              — normalized joint deltas
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict):
        super().__init__()
        self._cfg         = config
        env_cfg           = config["env"]
        sac_cfg           = config["sac"]
        col_cfg           = config["collection"]

        # Rosbridge
        self._url         = env_cfg["rosbridge_url"]
        self._action_topic = env_cfg["action_topic"]
        self._control_period = 1.0 / CONTROL_HZ

        # Helpers
        self._obs_builder = ObsBuilder()
        self._randomizer  = EnvRandomizer(self._url)
        self._reward_fn   = RewardFn(sac_cfg)

        # Recorder (reused for spinning Rosbridge)
        self._recorder    = TrajectoryRecorder(config)

        # Gym spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(FRAME_STACK, FUSED_DIM),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self._step_count = 0
        self._prev_physics: Optional[Dict] = None
        self._connected   = False

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if not self._connected:
            self._recorder.connect()
            self._connected = True

        # Randomize Unity scene
        env_seed = seed if seed is not None else int(np.random.randint(0, 2**31))
        self._randomizer.randomize(seed=env_seed)
        self._randomizer.wait_for_ready(timeout_sec=2.0)

        # Clear observation buffers
        self._obs_builder.reset()

        self._step_count  = 0
        self._prev_physics = None

        # Collect initial observation
        self._recorder._spin(timeout=0.1)
        self._obs_builder.update(
            self._recorder._latest_rgb_pil(),
            self._recorder._latest_joints,
        )

        obs = self._obs_builder.get_obs_stack()
        return obs.astype(np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        t0 = time.monotonic()

        # Scale action from [-1,1] to rad
        scaled_action = (action * ACTION_SCALE).tolist()

        # Publish action to Unity
        self._publish_action(scaled_action)

        # Wait one control period
        elapsed = time.monotonic() - t0
        if elapsed < self._control_period:
            time.sleep(self._control_period - elapsed)

        # Spin Rosbridge for new messages
        self._recorder._spin(timeout=0.02)

        # Build new observation
        self._obs_builder.update(
            self._recorder._latest_rgb_pil(),
            self._recorder._latest_joints,
            history_action=scaled_action,
        )
        obs = self._obs_builder.get_obs_stack()

        # Physics reward
        physics = self._get_physics_dict()
        reward  = self._reward_fn.compute(physics, prev_physics=self._prev_physics)
        self._prev_physics = physics

        self._step_count += 1
        terminated = bool(physics.get("occlusion_rate", 1.0) < 0.05)   # Clear view
        truncated  = self._step_count >= MAX_STEPS

        info = {"physics": physics, "step": self._step_count}
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._connected:
            self._recorder.disconnect()
            self._connected = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _publish_action(self, action: list) -> None:
        cmd = {
            "op": "publish",
            "topic": self._action_topic,
            "msg": {"data": json.dumps(action)},
        }
        try:
            self._recorder._conn.send(json.dumps(cmd))
        except Exception as e:
            logger.warning(f"[UnityEnv] Failed to publish action: {e}")

    def _get_physics_dict(self) -> dict:
        p = self._recorder._latest_physics
        return {
            "occlusion_rate":    p.occlusion_rate,
            "centering_score":   p.centering_score,
            "chassis_stability": p.chassis_stability,
        }
