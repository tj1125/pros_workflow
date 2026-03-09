"""
models/sac_policy.py — SAC Actor-Critic Network

Implements the full Soft Actor-Critic (SAC) network for 6-DOF
arm viewpoint adjustment:

  - Actor   : Gaussian policy, outputs (mu, log_std) → tanh-squashed action
  - Critic  : Twin Q-networks (to mitigate overestimation bias)
  - SACAgent: Wraps Actor + Twin-Critics with alpha auto-tuning

State space  : [temporal_feat(512), joints(6), history_action(6)] = 524-dim
Action space : 6-DOF joint delta ∈ [-action_scale, +action_scale] rad
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ------------------------------------------------------------------
# Constants (should match train_config.yaml)
# ------------------------------------------------------------------
STATE_DIM    = 524   # temporal(512) + joints(6) + history(6)
ACTION_DIM   = 6
HIDDEN_DIM   = 256
LOG_STD_MIN  = -5.0
LOG_STD_MAX  = 2.0


class Actor(nn.Module):
    """
    Gaussian Actor for SAC.

    Forward pass returns (mu, log_std) before tanh squashing.
    Use sample() to get reparameterized action + log_prob.
    """

    def __init__(
        self,
        state_dim:  int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head      = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_std) — unbounded."""
        h = self.trunk(state)
        mu      = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterized sample with tanh squashing.

        Returns:
            action   : tanh-squashed action (batch, action_dim)
            log_prob : log probability of the action (batch,)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        x = dist.rsample()                       # Reparameterized sample

        # Tanh squash
        action = torch.tanh(x)

        # Log-prob with change of variables: log π(a|s) = log π(u|s) - Σ log(1 - tanh²(u))
        log_prob = dist.log_prob(x).sum(dim=-1)
        log_prob -= (2 * (math.log(2) - x - F.softplus(-2 * x))).sum(dim=-1)

        return action, log_prob

    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """Return the deterministic (mean) action for evaluation."""
        mu, _ = self.forward(state)
        return torch.tanh(mu)


class Critic(nn.Module):
    """
    Twin Q-Network Critic for SAC.

    Returns Q1 and Q2 estimates to mitigate overestimation bias.
    """

    def __init__(
        self,
        state_dim:  int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        in_dim = state_dim + action_dim

        def make_q():
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1 = make_q()
        self.q2 = make_q()

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (Q1, Q2) for soft backup."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Convenience: return only Q1 (used in policy update)."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class SACAgent(nn.Module):
    """
    Full SAC Agent: Actor + Twin-Critic + Target-Twin-Critic + auto-alpha.

    This class is used during training (sac_trainer.py).
    Only the Actor weights are shipped to the inference server.
    """

    def __init__(
        self,
        state_dim:     int   = STATE_DIM,
        action_dim:    int   = ACTION_DIM,
        hidden_dim:    int   = HIDDEN_DIM,
        lr:            float = 3e-4,
        gamma:         float = 0.99,
        tau:           float = 0.005,
        auto_alpha:    bool  = True,
        target_entropy: float = -ACTION_DIM,
        device:        str   = "cpu",
    ):
        super().__init__()
        self.gamma  = gamma
        self.tau    = tau
        self.device = device

        # Networks
        self.actor          = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic         = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target  = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Entropy coefficient auto-tuning
        self.auto_alpha     = auto_alpha
        self.target_entropy = target_entropy
        self.log_alpha      = torch.tensor(0.0, requires_grad=auto_alpha, device=device)
        self.alpha          = self.log_alpha.exp().item()
        self.alpha_opt      = torch.optim.Adam([self.log_alpha], lr=lr) if auto_alpha else None

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def update(self, batch: dict) -> dict:
        """
        One SAC gradient step from a batch sampled from replay buffer.

        Args:
            batch: dict with keys {state, action, reward, next_state, done}
                   all torch.Tensor on self.device

        Returns:
            dict of loss metrics for logging
        """
        s   = batch["state"]
        a   = batch["action"]
        r   = batch["reward"].unsqueeze(-1)
        s2  = batch["next_state"]
        done = batch["done"].unsqueeze(-1)

        alpha = self.log_alpha.exp().detach()

        # -- Critic update -----------------------------------------------
        with torch.no_grad():
            a2, log_pi2 = self.actor.sample(s2)
            q1_t, q2_t  = self.critic_target(s2, a2)
            min_q_target = torch.min(q1_t, q2_t) - alpha * log_pi2.unsqueeze(-1)
            y = r + self.gamma * (1.0 - done) * min_q_target

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # -- Actor update -----------------------------------------------
        a_new, log_pi_new = self.actor.sample(s)
        q1_pi = self.critic.q1_forward(s, a_new)
        actor_loss = (alpha * log_pi_new - q1_pi.squeeze(-1)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # -- Alpha update -----------------------------------------------
        alpha_loss = torch.tensor(0.0)
        if self.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_pi_new.detach() + self.target_entropy)
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        # -- Soft target update -----------------------------------------
        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_t.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha_loss":  alpha_loss.item() if self.auto_alpha else 0.0,
            "alpha":       self.alpha,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Save all network weights for resuming training or inference."""
        torch.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha":     self.log_alpha,
        }, path)

    def load(self, path: str) -> None:
        """Load checkpoint weights."""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        if "log_alpha" in ckpt:
            self.log_alpha.data = ckpt["log_alpha"].data
            self.alpha = self.log_alpha.exp().item()
