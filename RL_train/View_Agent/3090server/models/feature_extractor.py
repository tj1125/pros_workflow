"""
models/feature_extractor.py — CLIP + DINOv2 Dual-Tower Feature Extractor

Extracts a fused visual feature vector from a single RGB image using:
  - CLIP ViT-B/32 (512-dim visual features)
  - DINOv2-base   (768-dim CLS token)
  - Concatenated: 1280-dim fused feature

Used during both training (feature pre-computation) and inference.
Models are loaded lazily and shared across calls.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CLIP_MODEL_ID   = os.getenv("CLIP_MODEL_ID",   "openai/clip-vit-base-patch32")
DINOV2_MODEL_ID = os.getenv("DINOV2_MODEL_ID", "facebook/dinov2-base")
CLIP_DIM        = 512
DINOV2_DIM      = 768
FUSED_DIM       = CLIP_DIM + DINOV2_DIM   # 1280


class FeatureExtractor:
    """
    Dual-tower visual encoder: CLIP + DINOv2.

    extract(image) -> np.ndarray of shape [FUSED_DIM=1280]

    Usage:
        fe = FeatureExtractor()
        feat = fe.extract(pil_image)   # [1280]
    """

    def __init__(self, device: Optional[str] = None):
        self._device = device  # Resolved on first use
        self._clip_model     = None
        self._clip_processor = None
        self._dino_model     = None
        self._dino_processor = None
        self._loaded         = False

    def extract(self, image) -> np.ndarray:
        """
        Extract fused CLIP+DINOv2 feature from a single PIL Image.

        Args:
            image: PIL.Image (any size / mode; will be converted to RGB internally)

        Returns:
            np.ndarray of shape (FUSED_DIM,) = (1280,)
        """
        self._lazy_load()
        import torch

        if hasattr(image, "convert"):
            image = image.convert("RGB")

        clip_feat = self._clip_encode(image)   # (512,)
        dino_feat = self._dino_encode(image)   # (768,)
        return np.concatenate([clip_feat, dino_feat])

    def extract_batch(self, images: list) -> np.ndarray:
        """
        Extract features from a list of PIL Images.

        Returns:
            np.ndarray of shape (N, FUSED_DIM)
        """
        return np.stack([self.extract(img) for img in images])

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    def _lazy_load(self) -> None:
        if self._loaded:
            return

        import torch
        from transformers import (
            CLIPModel, CLIPProcessor,
            AutoModel, AutoImageProcessor,
        )

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        device = self._device

        logger.info(f"[FeatureExtractor] Loading CLIP ({CLIP_MODEL_ID}) on {device}")
        self._clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        self._clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()

        logger.info(f"[FeatureExtractor] Loading DINOv2 ({DINOV2_MODEL_ID}) on {device}")
        self._dino_processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
        self._dino_model = AutoModel.from_pretrained(DINOV2_MODEL_ID).to(device).eval()

        self._loaded = True
        logger.info("[FeatureExtractor] Models loaded.")

    def _clip_encode(self, image) -> np.ndarray:
        import torch
        inputs = self._clip_processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            feat = self._clip_model.get_image_features(**inputs)  # (1, 512)
        return feat.squeeze(0).cpu().numpy()

    def _dino_encode(self, image) -> np.ndarray:
        import torch
        inputs = self._dino_processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out  = self._dino_model(**inputs)
            feat = out.last_hidden_state[:, 0, :]  # CLS token (1, 768)
        return feat.squeeze(0).cpu().numpy()
