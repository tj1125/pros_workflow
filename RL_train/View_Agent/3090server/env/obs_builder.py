"""
env/obs_builder.py — Observation Vector Builder

Transforms raw sensor data (PIL images, joint angles, IMU, history action)
into the normalized observation tensors fed to the SAC policy.

Output:
    obs_stack  : np.ndarray (FRAME_STACK, FUSED_DIM) — stacked visual features
    state_vec  : np.ndarray (STATE_DIM,)             — full state for SAC
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional

import numpy as np

from models.feature_extractor import FeatureExtractor

FRAME_STACK  = 3
FUSED_DIM    = 1280   # CLIP(512) + DINOv2(768)
JOINT_DIM    = 6
HISTORY_DIM  = 6
TEMPORAL_DIM = 512    # After TemporalEncoder
STATE_DIM    = TEMPORAL_DIM + JOINT_DIM + HISTORY_DIM   # 524


class ObsBuilder:
    """
    Maintains a rolling frame buffer and builds observation vectors.

    Usage:
        builder = ObsBuilder()
        builder.update(rgb_image, joint_angles, history_action)
        obs_stack = builder.get_obs_stack()   # (FRAME_STACK, FUSED_DIM)
    """

    def __init__(self, device: Optional[str] = None):
        self._extractor = FeatureExtractor(device=device)
        # Circular buffer of per-frame features
        self._frame_buf: collections.deque = collections.deque(
            maxlen=FRAME_STACK
        )
        # Initialize with zero frames
        for _ in range(FRAME_STACK):
            self._frame_buf.append(np.zeros(FUSED_DIM, dtype=np.float32))

        self._last_joints: np.ndarray = np.zeros(JOINT_DIM, dtype=np.float32)
        self._last_action: np.ndarray = np.zeros(HISTORY_DIM, dtype=np.float32)

    def reset(self) -> None:
        """Clear frame buffer (call at the start of each episode)."""
        self._frame_buf = collections.deque(
            [np.zeros(FUSED_DIM, dtype=np.float32)] * FRAME_STACK,
            maxlen=FRAME_STACK
        )
        self._last_joints = np.zeros(JOINT_DIM, dtype=np.float32)
        self._last_action = np.zeros(HISTORY_DIM, dtype=np.float32)

    def update(
        self,
        rgb_image,
        joint_angles: Dict[str, float],
        history_action: Optional[List[float]] = None,
    ) -> None:
        """
        Update internal buffers with the latest observation.

        Args:
            rgb_image     : PIL.Image (current RGB frame)
            joint_angles  : dict of joint_name -> angle (rad)
            history_action: last 6-DOF action executed
        """
        # Extract visual feature and push to frame buffer
        feat = self._extractor.extract(rgb_image)   # (FUSED_DIM,)
        self._frame_buf.append(feat)

        # Update joint state
        joints_list = list(joint_angles.values())[:JOINT_DIM]
        joints_list += [0.0] * max(0, JOINT_DIM - len(joints_list))
        self._last_joints = np.array(joints_list, dtype=np.float32)

        # Update history action
        if history_action is not None:
            hist = list(history_action)[:HISTORY_DIM]
            hist += [0.0] * max(0, HISTORY_DIM - len(hist))
            self._last_action = np.array(hist, dtype=np.float32)

    def get_obs_stack(self) -> np.ndarray:
        """
        Return the stacked frame features for TemporalEncoder input.

        Returns:
            np.ndarray of shape (FRAME_STACK, FUSED_DIM)
        """
        return np.stack(list(self._frame_buf), axis=0)

    def get_joints(self) -> np.ndarray:
        """Return the latest joint state vector (JOINT_DIM,)."""
        return self._last_joints.copy()

    def get_history_action(self) -> np.ndarray:
        """Return the latest history action vector (HISTORY_DIM,)."""
        return self._last_action.copy()
