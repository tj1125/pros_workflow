"""
view_agent/policy_service.py — SAC Policy Inference Service

Responsibilities:
  1. Lazy-load CLIP (openai/clip-vit-base-patch32) and DINOv2
     (facebook/dinov2-base) vision encoders once on first call.
  2. Extract per-frame visual features and fuse 3 temporal frames
     through TemporalEncoder (linear projection).
  3. Forward through a pre-trained SAC Actor network to produce a
     6-DOF joint delta action (in radians).

Input dict keys (from agent_executor.py):
    rgb_frames     : List[PIL.Image] — up to 3 frames, oldest→newest
    depth_frames   : List[PIL.Image] — up to 3 depth frames (grayscale)
    joint_state    : Dict[str, float] — joint name → angle (rad)
    history_action : List[float] — last 6-DOF delta executed

Output dict:
    delta_joints   : List[float] — 6 joint displacement increments (rad)
    confidence     : float       — SAC action log-probability proxy ∈ [0,1]
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLIP_MODEL_ID = os.getenv("CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
DINOV2_MODEL_ID = os.getenv("DINOV2_MODEL_ID", "facebook/dinov2-base")
CLIP_FEAT_DIM = 512     # CLIP ViT-B/32 visual output dim
DINOV2_FEAT_DIM = 768   # DINOv2-base CLS token dim
FUSED_DIM = CLIP_FEAT_DIM + DINOV2_FEAT_DIM  # 1280 per frame
TEMPORAL_DIM = 512       # TemporalEncoder output
JOINT_DIM = 6            # 6-DOF arm
HISTORY_DIM = JOINT_DIM  # one past action
STATE_DIM = TEMPORAL_DIM + JOINT_DIM + HISTORY_DIM  # 524
FRAME_STACK = 3


class PolicyService:
    """
    SAC Policy inference service (singleton-safe, lazy-loaded).

    Usage:
        svc = PolicyService()
        result = svc.infer(obs)  # obs: {rgb_frames, depth_frames, ...}
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        # Checkpoint for the SAC Actor weights
        self._ckpt_path = checkpoint_path or os.getenv(
            "VIEW_AGENT_CHECKPOINT", ""
        )
        # Lazy-loaded models
        self._clip_model = None
        self._clip_processor = None
        self._dinov2_model = None
        self._dinov2_processor = None
        self._temporal_encoder = None
        self._actor = None
        self._device = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run SAC policy inference.

        Args:
            obs: dict with keys:
                rgb_frames     : List[PIL.Image], length ≤ FRAME_STACK
                depth_frames   : List[PIL.Image], length ≤ FRAME_STACK
                joint_state    : Dict[str, float]
                history_action : List[float] (6 values)

        Returns:
            {"delta_joints": List[float], "confidence": float}
        """
        self._lazy_load()

        import torch

        rgb_frames: List = obs.get("rgb_frames", [])
        joint_state: Dict = obs.get("joint_state", {})
        history_action: List[float] = obs.get("history_action", [0.0] * JOINT_DIM)

        # -- Step 1: Extract per-frame visual features -------------------
        frame_feats = []
        for img in self._pad_frames(rgb_frames):
            clip_f = self._extract_clip(img)
            dino_f = self._extract_dinov2(img)
            frame_feats.append(np.concatenate([clip_f, dino_f]))  # [1280]

        # Stack and encode temporal context: [FRAME_STACK, 1280] -> [512]
        frame_tensor = torch.tensor(
            np.stack(frame_feats), dtype=torch.float32
        ).to(self._device)  # [3, 1280]
        temporal_feat = self._temporal_encoder(frame_tensor)  # [512]

        # -- Step 2: Concatenate state vector ----------------------------
        joints_arr = np.array(
            list(joint_state.values())[:JOINT_DIM] +
            [0.0] * max(0, JOINT_DIM - len(joint_state)),
            dtype=np.float32
        )
        hist_arr = np.array(
            history_action[:JOINT_DIM] +
            [0.0] * max(0, JOINT_DIM - len(history_action)),
            dtype=np.float32
        )
        state_vec = torch.cat([
            temporal_feat,
            torch.tensor(joints_arr, dtype=torch.float32).to(self._device),
            torch.tensor(hist_arr,  dtype=torch.float32).to(self._device),
        ]).unsqueeze(0)  # [1, STATE_DIM]

        # -- Step 3: SAC Actor forward ------------------------------------
        with torch.no_grad():
            mu, log_std = self._actor(state_vec)
            # Tanh squash + scale to ±0.05 rad per step
            delta = torch.tanh(mu) * 0.05
            # Confidence: proxy via mean action magnitude closeness to 0
            confidence = float(1.0 - delta.abs().mean().cpu())

        delta_joints = delta.squeeze(0).cpu().numpy().tolist()
        logger.debug(f"PolicyService: delta_joints={delta_joints}")
        return {
            "delta_joints": delta_joints,
            "confidence": max(0.0, min(1.0, confidence)),
        }

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    def _lazy_load(self) -> None:
        """Load models on first call; skip if already loaded."""
        if self._loaded:
            return

        import torch
        from transformers import (
            CLIPModel, CLIPProcessor,
            AutoModel, AutoImageProcessor,
        )

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"PolicyService: loading models on {self._device}...")

        # CLIP
        logger.info(f"  Loading CLIP: {CLIP_MODEL_ID}")
        self._clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        self._clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(self._device)
        self._clip_model.eval()

        # DINOv2
        logger.info(f"  Loading DINOv2: {DINOV2_MODEL_ID}")
        self._dinov2_processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
        self._dinov2_model = AutoModel.from_pretrained(DINOV2_MODEL_ID).to(self._device)
        self._dinov2_model.eval()

        # TemporalEncoder (Linear: FRAME_STACK * FUSED_DIM → TEMPORAL_DIM)
        self._temporal_encoder = TemporalEncoderHead(
            in_dim=FRAME_STACK * FUSED_DIM, out_dim=TEMPORAL_DIM
        ).to(self._device)

        # SAC Actor
        self._actor = SACActorHead(
            state_dim=STATE_DIM, action_dim=JOINT_DIM
        ).to(self._device)

        # Load checkpoint if provided
        if self._ckpt_path and Path(self._ckpt_path).exists():
            logger.info(f"  Loading checkpoint: {self._ckpt_path}")
            ckpt = torch.load(self._ckpt_path, map_location=self._device)
            if "temporal_encoder" in ckpt:
                self._temporal_encoder.load_state_dict(ckpt["temporal_encoder"])
            if "actor" in ckpt:
                self._actor.load_state_dict(ckpt["actor"])
            logger.info("  Checkpoint loaded successfully.")
        else:
            logger.warning(
                "PolicyService: no checkpoint provided. "
                "Running with random weights — for inference testing only."
            )
            self._temporal_encoder.eval()
            self._actor.eval()

        self._loaded = True
        logger.info("PolicyService: all models loaded.")

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------
    def _extract_clip(self, image) -> np.ndarray:
        """Extract CLIP visual features. Returns [CLIP_FEAT_DIM] numpy array."""
        import torch
        inputs = self._clip_processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            feats = self._clip_model.get_image_features(**inputs)  # [1, 512]
        return feats.squeeze(0).cpu().numpy()

    def _extract_dinov2(self, image) -> np.ndarray:
        """Extract DINOv2 CLS token. Returns [DINOV2_FEAT_DIM] numpy array."""
        import torch
        inputs = self._dinov2_processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._dinov2_model(**inputs)
            feats = out.last_hidden_state[:, 0, :]  # CLS token [1, 768]
        return feats.squeeze(0).cpu().numpy()

    @staticmethod
    def _pad_frames(frames: list, n: int = FRAME_STACK) -> list:
        """Pad frame list to exactly n frames by repeating the oldest frame."""
        if not frames:
            from PIL import Image
            blank = Image.fromarray(
                np.zeros((224, 224, 3), dtype=np.uint8)
            )
            return [blank] * n
        while len(frames) < n:
            frames = [frames[0]] + frames
        return frames[-n:]


# ---------------------------------------------------------------------------
# Lightweight network heads (defined here to avoid extra files)
# ---------------------------------------------------------------------------

class TemporalEncoderHead:
    """
    Flattens [FRAME_STACK, FUSED_DIM] -> [TEMPORAL_DIM] via a 2-layer MLP.
    Kept simple so it can be trained from scratch during Behavior Cloning.
    """

    def __init__(self, in_dim: int, out_dim: int):
        import torch.nn as nn
        self._net = nn.Sequential(
            nn.Flatten(),                   # [FRAME_STACK * FUSED_DIM]
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim),
            nn.LayerNorm(out_dim),
        )

    def to(self, device):
        self._net = self._net.to(device)
        return self

    def eval(self):
        self._net.eval()
        return self

    def load_state_dict(self, state_dict):
        self._net.load_state_dict(state_dict)

    def state_dict(self):
        return self._net.state_dict()

    def __call__(self, x):
        import torch
        # x: [FRAME_STACK, FUSED_DIM]  -> add batch dim
        return self._net(x.unsqueeze(0)).squeeze(0)


class SACActorHead:
    """
    Gaussian Actor for SAC: outputs (mu, log_std) for 6-DOF action.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, state_dim: int, action_dim: int):
        import torch.nn as nn
        self._trunk = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self._mu_head = nn.Linear(256, action_dim)
        self._log_std_head = nn.Linear(256, action_dim)

    def to(self, device):
        self._trunk = self._trunk.to(device)
        self._mu_head = self._mu_head.to(device)
        self._log_std_head = self._log_std_head.to(device)
        return self

    def eval(self):
        self._trunk.eval()
        self._mu_head.eval()
        self._log_std_head.eval()
        return self

    def load_state_dict(self, state_dict: dict):
        import torch
        trunk_sd = {k.replace("trunk.", ""): v
                    for k, v in state_dict.items() if k.startswith("trunk.")}
        mu_sd = {k.replace("mu.", ""): v
                 for k, v in state_dict.items() if k.startswith("mu.")}
        log_std_sd = {k.replace("log_std.", ""): v
                      for k, v in state_dict.items() if k.startswith("log_std.")}
        if trunk_sd:
            self._trunk.load_state_dict(trunk_sd)
        if mu_sd:
            self._mu_head.load_state_dict(mu_sd)
        if log_std_sd:
            self._log_std_head.load_state_dict(log_std_sd)

    def state_dict(self) -> dict:
        sd = {}
        sd.update({f"trunk.{k}": v for k, v in self._trunk.state_dict().items()})
        sd.update({f"mu.{k}": v for k, v in self._mu_head.state_dict().items()})
        sd.update({f"log_std.{k}": v for k, v in self._log_std_head.state_dict().items()})
        return sd

    def __call__(self, state):
        import torch
        h = self._trunk(state)
        mu = self._mu_head(h)
        log_std = self._log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std
