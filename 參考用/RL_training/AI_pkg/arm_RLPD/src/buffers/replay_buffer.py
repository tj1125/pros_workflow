from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_obs: np.ndarray
    done: np.ndarray
    is_offline: np.ndarray


class ReplayBuffer:
    """
    Unified replay buffer supporting offline+online storage and symmetric sampling.

    - Stores transitions as numpy arrays for speed.
    - Balanced sampling: offline:online ~= 1:1 when both are available.
    - Falls back gracefully if one side is empty or insufficient.
    """

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...] = (), dtype=np.float32):
        self.capacity = int(capacity)
        self.obs_buf = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.next_obs_buf = np.zeros((capacity, *obs_shape), dtype=dtype)
        # discrete actions stored as int64 scalar or vector
        self.action_buf = np.zeros((capacity, *action_shape), dtype=np.int64 if action_shape == () else np.float32)
        self.reward_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.offline_flag = np.zeros((capacity, 1), dtype=np.int8)

        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done, is_offline: bool):
        i = self.idx
        self.obs_buf[i] = obs
        self.action_buf[i] = action
        self.reward_buf[i] = reward
        self.next_obs_buf[i] = next_obs
        self.done_buf[i] = done
        self.offline_flag[i] = 1 if is_offline else 0

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _sample_indices(self, n: int) -> np.ndarray:
        return np.random.randint(0, self.size, size=n)

    def _indices_by_flag(self, flag: int) -> np.ndarray:
        if self.size == 0:
            return np.array([], dtype=np.int64)
        flags = self.offline_flag[: self.size, 0]
        return np.where(flags == flag)[0]

    def sample(self, batch_size: int, balanced: bool = True) -> Transition:
        assert self.size > 0, "ReplayBuffer is empty"
        if not balanced:
            idx = self._sample_indices(batch_size)
            return self._gather(idx)

        half = batch_size // 2
        rem = batch_size - half

        offline_idx_all = self._indices_by_flag(1)
        online_idx_all = self._indices_by_flag(0)

        if len(offline_idx_all) == 0 or len(online_idx_all) == 0:
            idx = self._sample_indices(batch_size)
            return self._gather(idx)

        off_choice = np.random.choice(offline_idx_all, size=min(half, len(offline_idx_all)), replace=len(offline_idx_all) < half)
        on_choice = np.random.choice(online_idx_all, size=min(rem, len(online_idx_all)), replace=len(online_idx_all) < rem)
        idx = np.concatenate([off_choice, on_choice])
        if len(idx) < batch_size:
            extra = np.random.choice(np.arange(self.size), size=batch_size - len(idx), replace=True)
            idx = np.concatenate([idx, extra])
        np.random.shuffle(idx)
        return self._gather(idx)

    def _gather(self, idx: np.ndarray) -> Transition:
        return Transition(
            obs=self.obs_buf[idx],
            action=self.action_buf[idx],
            reward=self.reward_buf[idx],
            next_obs=self.next_obs_buf[idx],
            done=self.done_buf[idx],
            is_offline=self.offline_flag[idx],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "size": self.size,
            "idx": self.idx,
            "obs_buf": self.obs_buf[: self.size],
            "action_buf": self.action_buf[: self.size],
            "reward_buf": self.reward_buf[: self.size],
            "next_obs_buf": self.next_obs_buf[: self.size],
            "done_buf": self.done_buf[: self.size],
            "offline_flag": self.offline_flag[: self.size],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ReplayBuffer":
        size = int(d["size"])
        obs_shape = d["obs_buf"].shape[1:]
        action_shape = d["action_buf"].shape[1:]
        buf = ReplayBuffer(capacity=int(d["capacity"]), obs_shape=obs_shape, action_shape=action_shape)
        buf.size = size
        buf.idx = int(d.get("idx", size % buf.capacity))
        buf.obs_buf[:size] = d["obs_buf"]
        buf.action_buf[:size] = d["action_buf"]
        buf.reward_buf[:size] = d["reward_buf"]
        buf.next_obs_buf[:size] = d["next_obs_buf"]
        buf.done_buf[:size] = d["done_buf"]
        buf.offline_flag[:size] = d["offline_flag"]
        return buf

