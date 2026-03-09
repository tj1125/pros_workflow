"""
rl/sac_trainer.py — SAC Reinforcement Learning Training Loop (Stage 3b)

Runs the full SAC fine-tuning loop:
  1. Warm-up: collect random transitions
  2. Training: actor-critic updates from replay buffer
  3. Evaluation: periodic episode rollout
  4. Checkpointing: save best and periodic snapshots
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from env.unity_env import UnityViewEnv
from models.temporal_encoder import TemporalEncoder
from models.sac_policy import SACAgent
from rl.replay_buffer import TemporalReplayBuffer

logger = logging.getLogger(__name__)


class SACTrainer:
    """
    SAC training loop that integrates UnityViewEnv, TemporalEncoder,
    SACAgent, and TemporalReplayBuffer.

    The TemporalEncoder is part of the observation pipeline; its
    gradients are NOT backpropagated during SAC (frozen after BC pretrain).
    The Actor and Critic are updated by the SAC algorithm.
    """

    def __init__(self, config: dict):
        self._cfg     = config
        sac_cfg       = config["sac"]
        feat_cfg      = config["features"]
        device_str    = "cuda" if torch.cuda.is_available() else "cpu"
        self._device  = torch.device(device_str)

        # -- Environment ---------------------------------------------------
        self._env = UnityViewEnv(config)

        # -- TemporalEncoder (frozen after BC pretrain) -------------------
        self._encoder = TemporalEncoder(
            frame_stack  = feat_cfg["frame_stack"],
            fused_dim    = feat_cfg["fused_dim"],
            temporal_dim = feat_cfg["temporal_dim"],
        ).to(self._device).eval()

        # -- SAC Agent ---------------------------------------------------
        self._agent = SACAgent(
            state_dim      = feat_cfg["state_dim"],
            action_dim     = sac_cfg["action_dim"],
            lr             = float(sac_cfg["learning_rate"]),
            gamma          = float(sac_cfg["gamma"]),
            tau            = float(sac_cfg["tau"]),
            auto_alpha     = bool(sac_cfg["auto_alpha"]),
            target_entropy = -sac_cfg["action_dim"] * float(sac_cfg["target_entropy_scale"]),
            device         = device_str,
        )

        # -- Replay Buffer -----------------------------------------------
        obs_shape = (feat_cfg["frame_stack"], feat_cfg["fused_dim"])
        self._buf = TemporalReplayBuffer(
            capacity   = int(sac_cfg["replay_buffer_size"]),
            obs_shape  = obs_shape,
            action_dim = sac_cfg["action_dim"],
            device     = device_str,
        )

        # -- Config params -----------------------------------------------
        self._total_steps       = int(sac_cfg["total_env_steps"])
        self._warmup_steps      = int(sac_cfg["warmup_steps"])
        self._batch_size        = int(sac_cfg["batch_size"])
        self._update_every      = int(sac_cfg["update_every_n_steps"])
        self._updates_per_step  = int(sac_cfg["updates_per_step"])
        self._save_every        = int(sac_cfg["save_every_n_steps"])
        self._eval_every        = int(sac_cfg["eval_every_n_steps"])
        self._eval_episodes     = int(sac_cfg["eval_episodes"])

        self._ckpt_dir = Path(sac_cfg["checkpoint_dir"])
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Load BC pretrain checkpoint if available
        bc_ckpt = sac_cfg.get("pretrain_checkpoint", "")
        if bc_ckpt and Path(bc_ckpt).exists():
            self._load_bc_checkpoint(bc_ckpt)

        self._best_eval_reward = -float("inf")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def train(self) -> None:
        logger.info(
            f"[SAC] Starting training: {self._total_steps} steps, "
            f"device={self._device}, warmup={self._warmup_steps}"
        )

        obs, _ = self._env.reset()
        episode_reward = 0.0
        episode_steps  = 0
        episode_count  = 0
        t_start        = time.monotonic()

        for global_step in range(1, self._total_steps + 1):
            # Encode observation to state vector
            state = self._encode_obs(obs)

            # Sample action
            if global_step <= self._warmup_steps:
                action = self._env.action_space.sample()
            else:
                with torch.no_grad():
                    s_tensor = torch.tensor(state, dtype=torch.float32,
                                            device=self._device).unsqueeze(0)
                    action = self._agent.actor.deterministic_action(s_tensor)
                    action = action.squeeze(0).cpu().numpy()

            # Step environment
            next_obs, reward, terminated, truncated, info = self._env.step(action)
            done = terminated or truncated

            # Store transition
            next_state = self._encode_obs(next_obs)
            self._buf.add(obs, action, reward, next_obs, float(done))

            obs = next_obs if not done else self._env.reset()[0]
            episode_reward += reward
            episode_steps  += 1

            if done:
                logger.info(
                    f"[SAC] Episode {episode_count+1} done | "
                    f"steps={episode_steps} | reward={episode_reward:.3f}"
                )
                episode_reward = 0.0
                episode_steps  = 0
                episode_count += 1

            # -- SAC update ------------------------------------------
            if (global_step > self._warmup_steps
                    and global_step % self._update_every == 0
                    and len(self._buf) >= self._batch_size):
                for _ in range(self._updates_per_step):
                    batch = self._buf.sample(self._batch_size)
                    # Encode obs stacks in batch to state vectors
                    batch["state"]      = self._encode_batch(batch["state"])
                    batch["next_state"] = self._encode_batch(batch["next_state"])
                    metrics = self._agent.update(batch)

                if global_step % 1000 == 0:
                    logger.info(
                        f"[SAC] Step {global_step} | "
                        f"critic={metrics['critic_loss']:.4f} | "
                        f"actor={metrics['actor_loss']:.4f} | "
                        f"alpha={metrics['alpha']:.4f}"
                    )

            # -- Evaluation ------------------------------------------
            if global_step % self._eval_every == 0:
                avg_reward = self._evaluate()
                logger.info(
                    f"[SAC] Eval @ step {global_step}: avg_reward={avg_reward:.3f}"
                )
                if avg_reward > self._best_eval_reward:
                    self._best_eval_reward = avg_reward
                    self._agent.save(str(self._ckpt_dir / "best.pt"))
                    logger.info(f"[SAC] New best model saved: {avg_reward:.3f}")

            # -- Periodic checkpoint ----------------------------------
            if global_step % self._save_every == 0:
                self._agent.save(str(self._ckpt_dir / f"step_{global_step:07d}.pt"))

        elapsed = time.monotonic() - t_start
        logger.info(f"[SAC] Training complete in {elapsed/3600:.2f}h.")
        self._env.close()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate(self) -> float:
        """Run deterministic evaluation episodes and return mean reward."""
        total_reward = 0.0
        for _ in range(self._eval_episodes):
            obs, _ = self._env.reset()
            done   = False
            ep_r   = 0.0
            while not done:
                state = self._encode_obs(obs)
                with torch.no_grad():
                    s_t = torch.tensor(state, dtype=torch.float32,
                                       device=self._device).unsqueeze(0)
                    action = self._agent.actor.deterministic_action(s_t)
                    action = action.squeeze(0).cpu().numpy()
                obs, r, terminated, truncated, _ = self._env.step(action)
                ep_r  += r
                done   = terminated or truncated
            total_reward += ep_r
        return total_reward / self._eval_episodes

    # ------------------------------------------------------------------
    # Observation encoding helpers
    # ------------------------------------------------------------------
    def _encode_obs(self, obs_stack: np.ndarray) -> np.ndarray:
        """
        Encode (FRAME_STACK, FUSED_DIM) obs_stack through TemporalEncoder.
        Returns (STATE_DIM,) numpy array.
        """
        t = torch.tensor(obs_stack, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            temporal = self._encoder(t.unsqueeze(0)).squeeze(0)   # (512,)
        return temporal.cpu().numpy()

    def _encode_batch(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of obs_stacks: (B, FRAME_STACK, FUSED_DIM) -> (B, TEMPORAL_DIM).
        """
        with torch.no_grad():
            return self._encoder(obs_batch)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------
    def _load_bc_checkpoint(self, path: str) -> None:
        """Load BC pretrain weights into encoder and actor."""
        ckpt = torch.load(path, map_location=self._device)
        if "temporal_encoder" in ckpt:
            self._encoder.load_state_dict(ckpt["temporal_encoder"])
            logger.info(f"[SAC] Loaded TemporalEncoder from BC checkpoint: {path}")
        if "actor" in ckpt:
            self._agent.actor.load_state_dict(ckpt["actor"])
            logger.info(f"[SAC] Loaded Actor from BC checkpoint: {path}")
