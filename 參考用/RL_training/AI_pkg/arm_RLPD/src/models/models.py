from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: Tuple[int, ...] = (128, 128), layer_norm: bool = False):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiscreteActor(nn.Module):
    """
    Actor for discrete action spaces.
    Outputs logits over actions; helper to sample and compute log-prob.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: Tuple[int, ...] = (128, 128), layer_norm: bool = False):
        super().__init__()
        self.n_actions = n_actions
        self.backbone = MLP(obs_dim, n_actions, hidden_dims, layer_norm)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.backbone(obs)

    def dist(self, obs: torch.Tensor):
        logits = self.forward(obs)
        return torch.distributions.Categorical(logits=logits)

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        d = self.dist(obs)
        if deterministic:
            a = torch.argmax(d.logits, dim=-1)
        else:
            a = d.sample()
        logp = d.log_prob(a)
        return a, logp


class QNetwork(nn.Module):
    """
    Q-network for discrete actions. Predicts Q-values for all actions.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dims: Tuple[int, ...] = (256, 256), layer_norm: bool = True):
        super().__init__()
        self.net = MLP(obs_dim, n_actions, hidden_dims, layer_norm)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QEnsemble(nn.Module):
    """
    Ensemble of Q-networks.
    """

    def __init__(self, obs_dim: int, n_actions: int, n_critics: int = 2, hidden_dims: Tuple[int, ...] = (256, 256), layer_norm: bool = True):
        super().__init__()
        self.critics = nn.ModuleList([QNetwork(obs_dim, n_actions, hidden_dims, layer_norm) for _ in range(n_critics)])

    def forward(self, obs: torch.Tensor) -> List[torch.Tensor]:
        return [q(obs) for q in self.critics]

    def min_q(self, obs: torch.Tensor) -> torch.Tensor:
        qs = self.forward(obs)
        q_stack = torch.stack(qs, dim=0)
        return torch.min(q_stack, dim=0).values

