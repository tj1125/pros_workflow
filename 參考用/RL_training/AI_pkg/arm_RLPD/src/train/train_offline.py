import os
import json
import pickle
import argparse
import shutil
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from src.buffers.replay_buffer import ReplayBuffer
from src.models.models import DiscreteActor, QEnsemble
from src.utils.utils import (
    CSVLogger,
    td_target,
    save_checkpoint,
    maybe_load_checkpoint,
    create_summary_writer,
)


def _scaled_human_reward(entry: Dict[str, Any]) -> float:
    if "human_reward" in entry and entry["human_reward"] is not None:
        try:
            return float(entry["human_reward"])
        except (TypeError, ValueError):
            return 0.0
    raw = entry.get("human_level")
    if raw is not None:
        try:
            level = int(raw)
            if 1 <= level <= 7:
                return 0.5 * (level - 4)
        except (TypeError, ValueError):
            return 0.0
    raw = entry.get("human_score")
    if raw is None:
        return 0.0
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if val in (-1.0, 1.0):
        return 1.5 * val
    if -3.0 <= val <= 3.0:
        return 0.5 * val
    return 0.0


def soft_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    """Polyak averaging update for target networks."""
    with torch.no_grad():
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.mul_(1.0 - tau)
            tgt_param.data.add_(tau * src_param.data)


def load_offline_buffer(path: str, capacity: int, obs_dim: int, action_shape=()) -> ReplayBuffer:
    buf = ReplayBuffer(capacity=capacity, obs_shape=(obs_dim,), action_shape=action_shape)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Offline data not found: {path}")

    def _try_jsonl(p: str) -> bool:
        try:
            loaded_any = False
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t = json.loads(line)
                    # Prefer integer action; fall back to action_id for string labels
                    action_val = t.get("action")
                    if isinstance(action_val, str):
                        action_val = t.get("action_id")
                    if action_val is None:
                        raise ValueError("Missing action/action_id in JSONL transition")
                    reward_val = float(t.get("reward", 0.0))
                    reward_val += _scaled_human_reward(t)
                    buf.add(
                        np.array(t["obs"], dtype=np.float32),
                        int(action_val),
                        reward_val,
                        np.array(t["next_obs"], dtype=np.float32),
                        float(t["done"]),
                        True,
                    )
                    loaded_any = True
            return loaded_any
        except Exception:
            return False

    if path.endswith(".pkl"):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "obs_buf" in data:
                return ReplayBuffer.from_dict(data)
            elif isinstance(data, list):
                for t in data:
                    reward_val = float(t.get("reward", 0.0))
                    reward_val += _scaled_human_reward(t)
                    buf.add(t["obs"], t["action"], reward_val, t["next_obs"], t["done"], True)
                return buf
            else:
                raise RuntimeError("Unsupported pickle structure. Expect dict with obs_buf or list of transitions.")
        except Exception:
            # Fallback: try parse same file as JSONL (in case it was saved as text)
            if _try_jsonl(path):
                return buf
            # Or try a sibling .jsonl with same stem
            jsonl_path = os.path.splitext(path)[0] + ".jsonl"
            if os.path.exists(jsonl_path) and _try_jsonl(jsonl_path):
                return buf
            raise RuntimeError(f"Failed to load offline data from {path}. File may be empty/corrupted. Try providing a valid .pkl (ReplayBuffer dict or list) or .jsonl.")

    if path.endswith(".jsonl"):
        ok = _try_jsonl(path)
        if not ok:
            raise RuntimeError(f"Failed to parse JSONL offline data: {path}")
        return buf

    # Unknown extension: try JSONL then pickle
    if _try_jsonl(path):
        return buf
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "obs_buf" in data:
            return ReplayBuffer.from_dict(data)
        elif isinstance(data, list):
            for t in data:
                reward_val = float(t.get("reward", 0.0))
                reward_val += _scaled_human_reward(t)
                buf.add(t["obs"], t["action"], reward_val, t["next_obs"], t["done"], True)
            return buf
    except Exception:
        pass
    raise RuntimeError(f"Unrecognized offline data format: {path}")


def train_offline(
    data_path: str = "data/offline_buffer.jsonl",
    obs_dim: int = 3,
    n_actions: int = 7,
    capacity: int = 100000,
    batch_size: int = 256,
    n_critics: int = 2,
    critic_lr: float = 3e-7,
    actor_lr: float = 1e-4,
    gamma: float = 0.99,
    alpha: float = 0.2,
    total_updates: int = 10000,
    device: str = "mps",
    log_path: str = "data/offline_log.csv",
    ckpt_path: str = "data/offline_ckpt.pt",
    tau: float = 0.01,
    target_update_freq: int = 1,
    max_grad_norm: float = 5.0,
    reward_clip: float = 5.0,
    normalize_rewards: bool = True,
    awac_beta: float = 1.0,
    awac_weight_clip: float = 20.0,
    awac_entropy_coef: float | None = None,
    tb_log_dir: Optional[str] = None,
):
    # Resolve device preference (default to MPS on Mac if available)
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    device = torch.device(device)

    # Load offline buffer
    buffer = load_offline_buffer(data_path, capacity=capacity, obs_dim=obs_dim)
    assert buffer.size > 0, f"No offline data found at {data_path}"

    # Reward stabilisation: clip and normalise to tame extreme targets
    rewards_view = buffer.reward_buf[: buffer.size]
    raw_mean = float(rewards_view.mean())
    raw_std = float(rewards_view.std() + 1e-6)
    if reward_clip is not None and reward_clip > 0:
        np.clip(rewards_view, -reward_clip, reward_clip, out=rewards_view)
    clipped_mean = float(rewards_view.mean())
    clipped_std = float(rewards_view.std() + 1e-6)
    if normalize_rewards and clipped_std > 1e-6:
        rewards_view -= clipped_mean
        rewards_view /= clipped_std
        reward_desc = (
            f"raw mean {raw_mean:.3f}, std {raw_std:.3f} → clipped mean {clipped_mean:.3f}, std {clipped_std:.3f}, normalised"
        )
    else:
        reward_desc = f"raw mean {raw_mean:.3f}, std {raw_std:.3f} (clipped mean {clipped_mean:.3f}, std {clipped_std:.3f})"
    print(f"[offline] Loaded {buffer.size} transitions; reward stats: {reward_desc}")

    actor = DiscreteActor(obs_dim, n_actions).to(device)
    critics = QEnsemble(obs_dim, n_actions, n_critics=n_critics).to(device)
    target_critics = QEnsemble(obs_dim, n_actions, n_critics=n_critics).to(device)
    target_critics.load_state_dict(critics.state_dict())
    for p in target_critics.parameters():
        p.requires_grad_(False)

    opt_actor = Adam(actor.parameters(), lr=actor_lr)
    opt_critic = Adam(critics.parameters(), lr=critic_lr)

    if tb_log_dir:
        shutil.rmtree(tb_log_dir, ignore_errors=True)
    logger = CSVLogger(log_path, fieldnames=["update", "critic_loss", "actor_loss"])
    tb_writer = create_summary_writer(tb_log_dir)

    entropy_coef = alpha if awac_entropy_coef is None else awac_entropy_coef

    # Optionally resume
    state = maybe_load_checkpoint(ckpt_path, device=str(device))
    start_update = 0
    if state is not None:
        actor.load_state_dict(state["actor"])  # type: ignore
        critics.load_state_dict(state["critics"])  # type: ignore
        opt_actor.load_state_dict(state["opt_actor"])  # type: ignore
        opt_critic.load_state_dict(state["opt_critic"])  # type: ignore
        start_update = int(state.get("update", 0))

    for update in range(start_update, total_updates):
        t = buffer.sample(batch_size, balanced=False)
        obs = torch.as_tensor(t.obs, device=device)
        actions = torch.as_tensor(t.action, device=device, dtype=torch.long)
        if actions.dim() == 1:
            actions = actions.view(-1, 1)
        else:
            actions = actions.long()
        rewards = torch.as_tensor(t.reward, device=device)
        next_obs = torch.as_tensor(t.next_obs, device=device)
        done = torch.as_tensor(t.done, device=device)

        # Critic target with SAC-style entropy term
        with torch.no_grad():
            logits_next = actor(next_obs)
            q_next_min = target_critics.min_q(next_obs)
            logp_next = F.log_softmax(logits_next, dim=-1)
            v_next = torch.sum(torch.softmax(logits_next, dim=-1) * (q_next_min - alpha * logp_next), dim=-1, keepdim=True)
            target_q = td_target(rewards, done, v_next, gamma)

        # Critic loss: MSE over chosen actions
        q_values_list = critics(obs)
        critic_loss = 0.0
        for q in q_values_list:
            q_a = q.gather(1, actions)
            critic_loss = critic_loss + F.mse_loss(q_a, target_q)

        opt_critic.zero_grad()
        critic_loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            clip_grad_norm_(critics.parameters(), max_grad_norm)
        opt_critic.step()

        if (update + 1) % target_update_freq == 0:
            soft_update(critics, target_critics, tau)

        # Actor loss: Advantage-weighted behaviour cloning with entropy regularisation
        logits = actor(obs)
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        q_min = critics.min_q(obs)
        with torch.no_grad():
            q_act = q_min.gather(1, actions)
            v = (p * q_min).sum(dim=-1, keepdim=True)
            adv = q_act - v
            weights = torch.exp(adv / awac_beta)
            if awac_weight_clip is not None and awac_weight_clip > 0:
                weights = torch.clamp(weights, max=awac_weight_clip)
        logp_actions = logp.gather(1, actions)
        entropy = -(p * logp).sum(dim=-1, keepdim=True)
        actor_loss = -(weights * logp_actions).mean() - entropy_coef * entropy.mean()

        opt_actor.zero_grad()
        actor_loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            clip_grad_norm_(actor.parameters(), max_grad_norm)
        opt_actor.step()

        step_idx = update + 1
        if tb_writer is not None:
            tb_writer.add_scalar("offline/critic_loss", float(critic_loss.item()), step_idx)
            tb_writer.add_scalar("offline/actor_loss", float(actor_loss.item()), step_idx)

        if step_idx % 100 == 0:
            critic_loss_val = float(critic_loss.item())
            actor_loss_val = float(actor_loss.item())
            logger.log({
                "update": step_idx,
                "critic_loss": critic_loss_val,
                "actor_loss": actor_loss_val,
            })
            print(f"[offline] update {step_idx:05d} | critic_loss={critic_loss_val:.4f} | actor_loss={actor_loss_val:.4f}")
            save_checkpoint(ckpt_path, {
                "actor": actor.state_dict(),
                "critics": critics.state_dict(),
                "opt_actor": opt_actor.state_dict(),
                "opt_critic": opt_critic.state_dict(),
                "update": step_idx,
            })

    logger.close()
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/offline_buffer.jsonl")
    parser.add_argument("--obs_dim", type=int, default=3)
    parser.add_argument("--n_actions", type=int, default=11)
    parser.add_argument("--capacity", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_critics", type=int, default=2)
    parser.add_argument("--critic_lr", type=float, default=3e-7)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--total_updates", type=int, default=10000)
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--log_path", type=str, default="data/offline_log.csv")
    parser.add_argument("--ckpt_path", type=str, default="data/offline_ckpt.pt")
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--target_update_freq", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--reward_clip", type=float, default=5.0)
    parser.add_argument("--normalize_rewards", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--awac_beta", type=float, default=1.0)
    parser.add_argument("--awac_weight_clip", type=float, default=20.0)
    parser.add_argument("--awac_entropy_coef", type=float, default=None)
    parser.add_argument("--tb_log_dir", type=str, default="runs/offline")
    args = parser.parse_args()

    train_offline(
        data_path=args.data_path,
        obs_dim=args.obs_dim,
        n_actions=args.n_actions,
        capacity=args.capacity,
        batch_size=args.batch_size,
        n_critics=args.n_critics,
        critic_lr=args.critic_lr,
        actor_lr=args.actor_lr,
        gamma=args.gamma,
        alpha=args.alpha,
        total_updates=args.total_updates,
        device=args.device,
        log_path=args.log_path,
        ckpt_path=args.ckpt_path,
        tau=args.tau,
        target_update_freq=args.target_update_freq,
        max_grad_norm=args.max_grad_norm,
        reward_clip=args.reward_clip,
        normalize_rewards=args.normalize_rewards,
        awac_beta=args.awac_beta,
        awac_weight_clip=args.awac_weight_clip,
        awac_entropy_coef=args.awac_entropy_coef,
        tb_log_dir=args.tb_log_dir,
    )
