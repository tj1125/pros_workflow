"""
rl/replay_buffer.py — Temporal Experience Replay Buffer

Stores transitions with stacked observation frames for time-aware
SAC training.

Transition schema:
    obs_stack    : (FRAME_STACK, FUSED_DIM) feature stack at time t
    action       : (6,)  delta joint action
    reward       : scalar
    next_obs_stack : (FRAME_STACK, FUSED_DIM) feature stack at time t+1
    done         : bool
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict


class TemporalReplayBuffer:
    """
    Fixed-capacity circular replay buffer for temporal SAC training.

    Usage:
        buf = TemporalReplayBuffer(capacity=100_000, obs_shape=(3, 1280), action_dim=6)
        buf.add(obs, action, reward, next_obs, done)
        batch = buf.sample(256)
    """

    def __init__(
        self,
        capacity:   int,
        obs_shape:  tuple,     # (FRAME_STACK, FUSED_DIM)
        action_dim: int = 6,
        device:     str = "cpu",
    ):
        self._capacity   = capacity
        self._obs_shape  = obs_shape
        self._action_dim = action_dim
        self._device     = device
        self._ptr        = 0
        self._size       = 0

        # Pre-allocate numpy arrays for efficiency
        self._obs       = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_obs  = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions   = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards   = np.zeros((capacity,), dtype=np.float32)
        self._dones     = np.zeros((capacity,), dtype=np.float32)

    def add(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        """Add a single transition to the buffer."""
        self._obs[self._ptr]      = obs
        self._next_obs[self._ptr] = next_obs
        self._actions[self._ptr]  = action
        self._rewards[self._ptr]  = reward
        self._dones[self._ptr]    = float(done)

        self._ptr  = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a random mini-batch of transitions.

        Returns dict with keys: state, action, reward, next_state, done
        All tensors are on self._device.
        """
        idx = np.random.randint(0, self._size, size=batch_size)

        def to_tensor(arr):
            return torch.tensor(arr, dtype=torch.float32, device=self._device)

        return {
            "state":      to_tensor(self._obs[idx]),
            "action":     to_tensor(self._actions[idx]),
            "reward":     to_tensor(self._rewards[idx]),
            "next_state": to_tensor(self._next_obs[idx]),
            "done":       to_tensor(self._dones[idx]),
        }

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        """True when buffer has enough samples to start training."""
        return self._size > 0
