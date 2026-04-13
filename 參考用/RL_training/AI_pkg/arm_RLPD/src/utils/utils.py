from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional
import csv
import time

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - tensorboard optional
    SummaryWriter = None  # type: ignore


def td_target(
    reward: torch.Tensor,
    done: torch.Tensor,
    next_q_values: torch.Tensor,
    gamma: float,
    entropy_term: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute TD target for discrete SAC-style update.
    next_q_values: expectation under next-policy, i.e., sum_a pi(a|s') [Q_min(s',a) - alpha*log pi(a|s')]
    entropy_term: optional term already incorporated into next_q_values; kept for clarity.
    """
    with torch.no_grad():
        not_done = 1.0 - done
        return reward + gamma * not_done * next_q_values


class CSVLogger:
    def __init__(self, path: str, fieldnames: Optional[list] = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.file = open(path, "w", newline="")
        self.fieldnames = fieldnames or ["step", "reward", "loss"]
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.file.flush()

    def log(self, row: Dict[str, Any]):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


def maybe_load_checkpoint(path: str, device: str = "cpu") -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location=device)


def save_checkpoint(path: str, state: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def reward_shaping(distance: float, success_threshold: float, success_bonus: float = 1.0) -> float:
    reward = -distance
    if distance < success_threshold:
        reward += success_bonus
    return reward


def create_summary_writer(log_dir: Optional[str]):
    if log_dir is None or SummaryWriter is None:
        return None
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)
