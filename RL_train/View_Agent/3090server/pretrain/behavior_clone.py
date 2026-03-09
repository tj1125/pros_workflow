"""
pretrain/behavior_clone.py — Behavior Cloning Pre-training (Stage 2)

Trains the TemporalEncoder + Actor to imitate the Teacher VLM's
expert actions (a_expert) using supervised MSE loss.

This pre-training step gives the SAC policy a sensible starting point,
avoiding the initial random-walk phase during RL fine-tuning.

Input data: HDF5 trajectories with labels/a_expert and labels/r_total
Output:     checkpoints/pretrain/best.pt
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.feature_extractor import FeatureExtractor
from models.temporal_encoder import TemporalEncoder
from models.sac_policy import Actor

logger = logging.getLogger(__name__)

FRAME_STACK  = 3
FUSED_DIM    = 1280
TEMPORAL_DIM = 512
JOINT_DIM    = 6
HISTORY_DIM  = 6
STATE_DIM    = TEMPORAL_DIM + JOINT_DIM + HISTORY_DIM


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BCDataset(Dataset):
    """
    Loads pre-extracted feature stacks + a_expert labels from HDF5 files.

    Features must be pre-computed (run extract features first for efficiency).
    If 'features' group is absent, falls back to on-the-fly extraction (slow).
    """

    def __init__(self, hdf5_paths: List[Path], frame_stack: int = FRAME_STACK):
        self._samples: List[Tuple] = []  # (obs_stack, joints, history, a_expert)
        self._frame_stack = frame_stack
        extractor = None

        for p in hdf5_paths:
            with h5py.File(p, "r") as f:
                if "labels" not in f or "a_expert" not in f["labels"]:
                    continue
                n = int(f.attrs.get("n_steps", f["actions"].shape[0]))
                actions   = f["actions"][:]             # (N, 6)
                a_expert  = f["labels/a_expert"][:]     # (N, 6)
                joints    = f["joints"][:]              # (N, 6)
                physics   = f["physics"][:]             # (N, 3)

                # Try to use pre-computed visual features
                if "features" in f:
                    raw_feats = f["features"][:]        # (N, FUSED_DIM)
                    has_feats = True
                else:
                    has_feats = False
                    if extractor is None:
                        extractor = FeatureExtractor()
                    raw_feats = self._extract_feats(f, extractor, n)

            # Build sliding windows
            for end_idx in range(frame_stack - 1, n):
                if np.any(np.isnan(a_expert[end_idx])):
                    continue   # Skip un-labeled steps

                start_idx = end_idx - frame_stack + 1
                obs_stack = raw_feats[start_idx:end_idx + 1]   # (FRAME_STACK, FUSED_DIM)
                joint_vec = joints[end_idx]                     # (6,)
                hist_act  = actions[max(0, end_idx - 1)]        # (6,)
                target    = a_expert[end_idx]                   # (6,)

                self._samples.append(
                    (obs_stack.astype(np.float32),
                     joint_vec.astype(np.float32),
                     hist_act.astype(np.float32),
                     target.astype(np.float32))
                )

        logger.info(f"[BCDataset] Loaded {len(self._samples)} training samples.")

    @staticmethod
    def _extract_feats(f: h5py.File, extractor: FeatureExtractor, n: int) -> np.ndarray:
        """Fallback: extract features from raw images (slow)."""
        from io import BytesIO
        from PIL import Image
        feats = []
        for i in range(n):
            rgb_b = bytes(f["rgb"][i])
            img = Image.open(BytesIO(rgb_b)).convert("RGB") if rgb_b else \
                  Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            feats.append(extractor.extract(img))
        return np.stack(feats)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        obs, joints, hist, target = self._samples[idx]
        return {
            "obs_stack": torch.tensor(obs),
            "joints":    torch.tensor(joints),
            "history":   torch.tensor(hist),
            "a_expert":  torch.tensor(target),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BehaviorCloneTrainer:
    """
    Trains TemporalEncoder + Actor via MSE behavior cloning.

    Usage:
        trainer = BehaviorCloneTrainer(config)
        trainer.train()
    """

    def __init__(self, config: dict):
        self._cfg   = config["pretrain"]
        self._feat_cfg = config["features"]
        device_str   = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_str)

        self._encoder = TemporalEncoder(
            frame_stack  = self._feat_cfg["frame_stack"],
            fused_dim    = self._feat_cfg["fused_dim"],
            temporal_dim = self._feat_cfg["temporal_dim"],
        ).to(self._device)

        self._actor = Actor(
            state_dim  = STATE_DIM,
            action_dim = 6,
            hidden_dim = 256,
        ).to(self._device)

        self._opt = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._actor.parameters()),
            lr=float(self._cfg["learning_rate"]),
        )

        ckpt_dir = Path(self._cfg["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_dir = ckpt_dir

    def train(self) -> None:
        data_dir  = Path(self._cfg["data_dir"])
        all_paths = list(data_dir.glob("*.h5"))
        if not all_paths:
            raise FileNotFoundError(f"No HDF5 files found in {data_dir}")

        # Train/val split
        random.shuffle(all_paths)
        val_n  = max(1, int(len(all_paths) * float(self._cfg["val_split"])))
        val_p  = all_paths[:val_n]
        train_p = all_paths[val_n:]

        train_ds = BCDataset(train_p)
        val_ds   = BCDataset(val_p)

        bs = int(self._cfg["batch_size"])
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                                  num_workers=2)

        loss_fn     = nn.MSELoss()
        best_val    = float("inf")
        patience    = int(self._cfg["early_stopping_patience"])
        no_improve  = 0
        save_every  = int(self._cfg.get("save_every_n_epochs", 5))

        for epoch in range(1, int(self._cfg["epochs"]) + 1):
            # --- Train -------------------------------------------------------
            self._encoder.train(); self._actor.train()
            train_losses = []
            for batch in train_loader:
                obs   = batch["obs_stack"].to(self._device)   # (B, 3, 1280)
                joints = batch["joints"].to(self._device)     # (B, 6)
                hist   = batch["history"].to(self._device)    # (B, 6)
                target = batch["a_expert"].to(self._device)   # (B, 6)

                temporal = self._encoder(obs)                  # (B, 512)
                state    = torch.cat([temporal, joints, hist], dim=-1)  # (B, 524)
                mu, _    = self._actor(state)
                pred_act = torch.tanh(mu)

                loss = loss_fn(pred_act, target)
                self._opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self._encoder.parameters()) + list(self._actor.parameters()), 1.0
                )
                self._opt.step()
                train_losses.append(loss.item())

            # --- Validation --------------------------------------------------
            self._encoder.eval(); self._actor.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    obs    = batch["obs_stack"].to(self._device)
                    joints = batch["joints"].to(self._device)
                    hist   = batch["history"].to(self._device)
                    target = batch["a_expert"].to(self._device)
                    temporal = self._encoder(obs)
                    state    = torch.cat([temporal, joints, hist], dim=-1)
                    mu, _    = self._actor(state)
                    pred_act = torch.tanh(mu)
                    val_losses.append(loss_fn(pred_act, target).item())

            train_loss = np.mean(train_losses)
            val_loss   = np.mean(val_losses)
            logger.info(
                f"Epoch {epoch:3d}/{self._cfg['epochs']} "
                f"train={train_loss:.5f}  val={val_loss:.5f}"
            )

            # Early stopping
            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0
                self._save("best.pt")
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"[BC] Early stopping at epoch {epoch}.")
                    break

            if epoch % save_every == 0:
                self._save(f"epoch_{epoch:04d}.pt")

        logger.info(f"[BC] Training complete. Best val loss: {best_val:.5f}")

    def _save(self, filename: str) -> None:
        path = self._ckpt_dir / filename
        torch.save({
            "temporal_encoder": self._encoder.state_dict(),
            "actor":            self._actor.state_dict(),
        }, path)
        logger.info(f"[BC] Saved checkpoint → {path}")
