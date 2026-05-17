"""
models/temporal_encoder.py — 3-Frame Temporal Fusion Encoder

Takes FRAME_STACK stacked visual feature vectors (one per frame)
and produces a single compact temporal representation.

Architecture: Flatten → Linear(3*1280, 1024) → ReLU → Linear(1024, 512) → LayerNorm
"""

from __future__ import annotations

import torch
import torch.nn as nn

FRAME_STACK = 3
FUSED_DIM   = 1280    # CLIP(512) + DINOv2(768) per frame
TEMPORAL_DIM = 512    # Output dim


class TemporalEncoder(nn.Module):
    """
    Fuses FRAME_STACK × FUSED_DIM visual feature vectors into a
    single TEMPORAL_DIM-dimensional representation.

    Input:  Tensor of shape (batch, FRAME_STACK, FUSED_DIM)
    Output: Tensor of shape (batch, TEMPORAL_DIM)
    """

    def __init__(
        self,
        frame_stack: int  = FRAME_STACK,
        fused_dim:   int  = FUSED_DIM,
        temporal_dim: int = TEMPORAL_DIM,
        hidden_dim:  int  = 1024,
    ):
        super().__init__()
        in_dim = frame_stack * fused_dim
        self.net = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, temporal_dim),
            nn.LayerNorm(temporal_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, FRAME_STACK, FUSED_DIM) or (FRAME_STACK, FUSED_DIM)
        Returns:
            (batch, TEMPORAL_DIM)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)          # Add batch dim
        return self.net(x)
