"""
collector/snapshot.py — Single-frame snapshot dataclass

Stores one synchronized observation frame containing:
  - RGB image (as bytes, PNG-encoded)
  - Depth image (as bytes, PNG-encoded)
  - Joint angles (6-DOF)
  - IMU data (linear acceleration + angular velocity)
  - Physics metrics from Unity (occlusion, centering, stability)
  - The action that was executed at this step
  - Timestamp
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PhysicsMetrics:
    """Physical ground-truth metrics reported by Unity each frame."""
    # Occlusion rate: fraction of target object occluded (0=clear, 1=fully occluded)
    occlusion_rate: float = 0.0
    # Centering score: how centered the target is in camera frame (1=center, 0=edge)
    centering_score: float = 0.0
    # Chassis stability: 1=stable, 0=oscillating/tilted (from IMU)
    chassis_stability: float = 1.0


@dataclass
class Snapshot:
    """
    One synchronized observation timestep during trajectory recording.

    All images are stored as PNG-encoded bytes to keep dtype-agnostic.
    """
    # Step index within the trajectory
    step_idx: int = 0

    # Wall-clock timestamp when this snapshot was captured
    timestamp: float = field(default_factory=time.time)

    # RGB image as PNG bytes (shape: H x W x 3)
    rgb_bytes: bytes = b""

    # Depth image as PNG bytes (shape: H x W, single-channel float32 stored as uint16)
    depth_bytes: bytes = b""

    # Joint angles in radians, keyed by joint name
    joint_angles: Dict[str, float] = field(default_factory=dict)

    # IMU: linear acceleration [ax, ay, az] m/s²
    linear_acceleration: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # IMU: angular velocity [wx, wy, wz] rad/s
    angular_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    # Physics metrics published by Unity
    physics: PhysicsMetrics = field(default_factory=PhysicsMetrics)

    # Action executed at this step (6-DOF joint delta in radians)
    action: List[float] = field(default_factory=lambda: [0.0] * 6)

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for HDF5 metadata storage)."""
        return {
            "step_idx": self.step_idx,
            "timestamp": self.timestamp,
            "joint_angles": self.joint_angles,
            "linear_acceleration": self.linear_acceleration,
            "angular_velocity": self.angular_velocity,
            "occlusion_rate": self.physics.occlusion_rate,
            "centering_score": self.physics.centering_score,
            "chassis_stability": self.physics.chassis_stability,
            "action": self.action,
        }
