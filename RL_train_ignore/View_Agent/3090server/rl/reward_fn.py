"""
rl/reward_fn.py — Reward Function for View Agent SAC Training

Combines two reward sources:
  1. Real-time physics metrics from Unity (immediate, dense feedback)
  2. Offline Teacher VLM score (r_total, sparse, high-quality)

The weights between the two sources are configured in train_config.yaml.
"""

from __future__ import annotations

from typing import Dict, Optional


class RewardFn:
    """
    Computes the blended reward for one environment step.

    Usage:
        reward_fn = RewardFn(config["sac"])
        reward = reward_fn.compute(physics_metrics, teacher_reward)
    """

    def __init__(self, sac_config: dict):
        self._physics_w = float(sac_config.get("physics_reward_weight", 0.5))
        self._teacher_w = float(sac_config.get("teacher_reward_weight", 0.5))
        self._use_teacher = bool(sac_config.get("use_teacher_reward", True))

    def compute(
        self,
        physics: Dict[str, float],
        teacher_reward: Optional[float] = None,
        prev_physics: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute blended step reward.

        Args:
            physics       : Current frame physics metrics dict:
                              {occlusion_rate, centering_score, chassis_stability}
            teacher_reward: r_total from Teacher VLM (None if not available yet)
            prev_physics  : Previous frame physics dict (for progress delta)

        Returns:
            Scalar float reward
        """
        physics_r = self._physics_reward(physics, prev_physics)

        if self._use_teacher and teacher_reward is not None:
            return (
                self._physics_w * physics_r
                + self._teacher_w * float(teacher_reward)
            )
        return physics_r

    @staticmethod
    def _physics_reward(
        current: Dict[str, float],
        prev: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute physics-based reward from Unity ground-truth metrics.

        Components:
          - Centering bonus   : higher centering_score is better (+0.3 max)
          - Stability bonus   : penalize low chassis_stability (-0.2 max)
          - Progress bonus    : reward reduction in occlusion_rate vs prev frame

        All components are normalized to roughly [-1, 1].
        """
        occlusion  = float(current.get("occlusion_rate", 0.0))
        centering  = float(current.get("centering_score", 0.0))
        stability  = float(current.get("chassis_stability", 1.0))

        # Penalize high occlusion
        occ_penalty = -occlusion * 0.4

        # Reward high centering
        center_bonus = centering * 0.3

        # Reward chassis stability
        stab_bonus = (stability - 0.5) * 0.2

        # Progress reward: improvement in occlusion vs previous frame
        progress = 0.0
        if prev is not None:
            prev_occ = float(prev.get("occlusion_rate", occlusion))
            delta_occ = prev_occ - occlusion   # positive = improvement
            progress = delta_occ * 0.5

        return occ_penalty + center_bonus + stab_bonus + progress
