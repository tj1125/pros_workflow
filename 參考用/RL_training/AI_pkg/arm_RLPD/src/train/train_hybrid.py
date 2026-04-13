import os
import json
import pickle
import argparse
import time
import shutil
from typing import Optional, Tuple, Dict, Any

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

ACTION_ID_TO_STR = {
    0: "none",
    1: "forward",
    2: "backward",
    3: "right",
    4: "left",
    5: "up",
    6: "down",
    7: "elbow_backward",
    8: "elbow_forward",
    9: "elbow_left",
    10: "elbow_right",
}

STR_TO_ACTION_ID = {v: k for k, v in ACTION_ID_TO_STR.items()}


def load_offline(path: str, capacity: int, obs_dim: int) -> ReplayBuffer:
    buf = ReplayBuffer(capacity=capacity, obs_shape=(obs_dim,), action_shape=())
    if not os.path.exists(path):
        return buf

    def _try_jsonl(p: str) -> bool:
        try:
            loaded_any = False
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t = json.loads(line)
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
                raise RuntimeError("Unsupported pickle structure for offline data.")
        except Exception:
            if _try_jsonl(path):
                return buf
            jsonl_path = os.path.splitext(path)[0] + ".jsonl"
            if os.path.exists(jsonl_path) and _try_jsonl(jsonl_path):
                return buf
            return buf

    if path.endswith(".jsonl"):
        _ = _try_jsonl(path)
        return buf

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
    return buf


def create_env(env_type: str, max_steps: int, seed: int):
    if env_type == "unity":
        from src.envs.unity_arm_env import UnityArmEnv

        return UnityArmEnv(max_steps=max_steps, seed=seed)
    elif env_type == "simple":
        from src.envs.arm_env import SimpleArmEnv

        return SimpleArmEnv(max_steps=max_steps, seed=seed)
    else:
        raise ValueError(f"Unknown env_type '{env_type}'. Use 'unity' or 'simple'.")


def reset_env(env) -> Tuple[np.ndarray, Dict[str, Any]]:
    state = env.reset()
    if isinstance(state, tuple):
        if len(state) == 2:
            return state  # obs, info
        return state[0], {}
    return state, {}


def train_hybrid(
    offline_path: str = "data/offline_buffer.jsonl",
    obs_dim: int = 3,
    n_actions: int = 11,
    online_capacity: int = 200000,
    batch_size: int = 256,
    utd_ratio: int = 4,
    total_env_steps: int = 2000,
    start_learning: int = 1000,
    offline_fraction: float = 0.3,
    n_critics: int = 2,
    gamma: float = 0.99,
    alpha: float = 0.2,
    critic_lr: float = 3e-7,
    actor_lr: float = 1e-4,
    device: str = "mps",
    log_path: str = "data/hybrid_log.csv",
    ckpt_path: str = "data/hybrid_ckpt.pt",
    save_ckpt_path: Optional[str] = None,
    env_type: str = "unity",
    max_steps: int = 80,
    seed: int = 0,
    tau: float = 0.01,
    target_update_freq: int = 1,
    max_grad_norm: float = 5.0,
    reward_clip: Optional[float] = 5.0,
    normalize_online_rewards: bool = False,
    reward_scale: float = 1.0,
    distance_reward_coef: float = 0.0,
    distance_penalty_threshold: float = 0.0,
    distance_penalty_coef: float = 0.0,
    print_step_metrics: bool = True,
    tb_log_dir: Optional[str] = None,
):
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    device = torch.device(device)

    env = create_env(env_type=env_type, max_steps=max_steps, seed=seed)
    offline_buf = load_offline(offline_path, capacity=online_capacity, obs_dim=obs_dim)
    online_buf = ReplayBuffer(capacity=online_capacity, obs_shape=(obs_dim,), action_shape=())

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
    logger = CSVLogger(log_path, fieldnames=["step", "episode_return", "critic_loss", "actor_loss", "episode_len"])
    tb_writer = create_summary_writer(tb_log_dir)

    entropy_coef = alpha

    ckpt = maybe_load_checkpoint(ckpt_path, device=str(device))
    global_step = 0
    if ckpt is not None:
        actor.load_state_dict(ckpt["actor"])  # type: ignore
        critics.load_state_dict(ckpt["critics"])  # type: ignore
        target_critics.load_state_dict(critics.state_dict())
        opt_actor.load_state_dict(ckpt["opt_actor"])  # type: ignore
        opt_critic.load_state_dict(ckpt["opt_critic"])  # type: ignore
        global_step = int(ckpt.get("step", 0))

    obs, _ = reset_env(env)
    episode_return = 0.0
    episode_len = 0
    last_reward_val: Optional[float] = None
    last_distance_val: Optional[float] = None
    last_critic_loss_val: Optional[float] = None
    last_actor_loss_val: Optional[float] = None
    update_counter = 0
    last_episode_return: float = 0.0

    def _convert_action_id_to_env_cmd(action_id: int):
        if env_type == "unity":
            return ACTION_ID_TO_STR.get(int(action_id), "none")
        elif env_type == "simple":
            return int(action_id)
        else:
            raise ValueError(f"Unsupported env_type {env_type}")

    save_path = save_ckpt_path or ckpt_path

    try:
        while global_step < total_env_steps:
            obs_tensor = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_tensor, _ = actor.act(obs_tensor, deterministic=False)
            action_id = int(action_tensor.squeeze(0).item())
            env_action = _convert_action_id_to_env_cmd(action_id)

            step_out = env.step(env_action)

            forced_reset = False
            if isinstance(step_out, tuple) and len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                done_flag = bool(terminated or truncated)
            elif isinstance(step_out, tuple) and len(step_out) == 4:
                next_obs, reward, done_flag, info = step_out  # type: ignore
                terminated = bool(done_flag)
                truncated = False
            else:
                raise RuntimeError("Unexpected step() return format from environment")

            distance_val = None
            if isinstance(info, dict):
                distance_val = float(info.get("distance", 0.0))
                forced_reset = bool(info.get("forced_reset", False))
            else:
                info = {}
            if forced_reset:
                done_flag = True

            reward_val = float(reward)
            if normalize_online_rewards and reward_clip is not None and reward_clip > 0:
                reward_val = float(np.clip(reward_val, -reward_clip, reward_clip))

            if distance_val is not None:
                if distance_reward_coef != 0.0:
                    reward_val += distance_reward_coef * distance_val
                if distance_penalty_coef > 0.0 and distance_penalty_threshold > 0.0 and distance_val > distance_penalty_threshold:
                    penalty = distance_penalty_coef * (distance_val - distance_penalty_threshold)
                    reward_val -= penalty

            if reward_scale != 1.0:
                reward_val *= reward_scale

            online_buf.add(
                obs.astype(np.float32),
                int(action_id),
                reward_val,
                np.asarray(next_obs, dtype=np.float32),
                float(done_flag),
                is_offline=False,
            )

            obs = np.asarray(next_obs, dtype=np.float32)
            episode_return += reward_val
            episode_len += 1
            global_step += 1

            last_reward_val = reward_val
            last_distance_val = distance_val

            if tb_writer is not None:
                tb_writer.add_scalar("hybrid/step_reward", reward_val, global_step)
                if distance_val is not None:
                    tb_writer.add_scalar("hybrid/distance", distance_val, global_step)

            if print_step_metrics:
                msg = f"[hybrid] step {global_step:06d} action={action_id} reward={reward_val:.3f}"
                if distance_val is not None:
                    msg += f" distance={distance_val:.3f}"
                print(msg)

            time.sleep(1.0)

            if online_buf.size >= start_learning and (offline_buf.size > 0 or online_buf.size >= batch_size):
                offline_fraction_clamped = min(max(offline_fraction, 0.0), 1.0)
                for _ in range(utd_ratio):
                    offline_batch = int(round(batch_size * offline_fraction_clamped))
                    offline_batch = min(offline_batch, batch_size - 1) if batch_size > 1 else offline_batch
                    if offline_buf.size == 0:
                        t_on = online_buf.sample(batch_size, balanced=False)
                        obs_b = torch.as_tensor(t_on.obs, device=device)
                        act_arr = t_on.action
                        rew_b = torch.as_tensor(t_on.reward, device=device)
                        nobs_b = torch.as_tensor(t_on.next_obs, device=device)
                        done_b = torch.as_tensor(t_on.done, device=device)
                    else:
                        offline_batch = max(1, offline_batch)
                        online_batch = max(1, batch_size - offline_batch)
                        t_off = offline_buf.sample(offline_batch, balanced=False)
                        t_on = online_buf.sample(online_batch, balanced=False)
                        obs_b = torch.as_tensor(np.concatenate([t_off.obs, t_on.obs], axis=0), device=device)
                        act_arr = np.concatenate([t_off.action, t_on.action], axis=0)
                        rew_b = torch.as_tensor(np.concatenate([t_off.reward, t_on.reward], axis=0), device=device)
                        nobs_b = torch.as_tensor(np.concatenate([t_off.next_obs, t_on.next_obs], axis=0), device=device)
                        done_b = torch.as_tensor(np.concatenate([t_off.done, t_on.done], axis=0), device=device)

                    act_b = torch.as_tensor(act_arr, device=device, dtype=torch.long)
                    if act_b.dim() == 1:
                        act_b = act_b.view(-1, 1)

                    with torch.no_grad():
                        logits_next = actor(nobs_b)
                        logp_next = F.log_softmax(logits_next, dim=-1)
                        p_next = logp_next.exp()
                        q_next_min = target_critics.min_q(nobs_b)
                        v_next = (p_next * (q_next_min - alpha * logp_next)).sum(dim=-1, keepdim=True)
                        target = td_target(rew_b, done_b, v_next, gamma)

                    q_values_list = critics(obs_b)
                    critic_loss = 0.0
                    for q in q_values_list:
                        qa = q.gather(1, act_b)
                        critic_loss = critic_loss + F.mse_loss(qa, target)
                    opt_critic.zero_grad()
                    critic_loss.backward()
                    if max_grad_norm is not None and max_grad_norm > 0:
                        clip_grad_norm_(critics.parameters(), max_grad_norm)
                    opt_critic.step()

                    if (global_step + 1) % target_update_freq == 0:
                        with torch.no_grad():
                            for src, tgt in zip(critics.parameters(), target_critics.parameters()):
                                tgt.data.mul_(1.0 - tau)
                                tgt.data.add_(tau * src.data)

                    logits = actor(obs_b)
                    logp = F.log_softmax(logits, dim=-1)
                    p = logp.exp()
                    q_min = critics.min_q(obs_b)
                    actor_loss = -(p * (q_min - entropy_coef * logp)).sum(dim=-1).mean()
                    opt_actor.zero_grad()
                    actor_loss.backward()
                    if max_grad_norm is not None and max_grad_norm > 0:
                        clip_grad_norm_(actor.parameters(), max_grad_norm)
                    opt_actor.step()

                    loss_step = update_counter + 1
                    last_critic_loss_val = float(critic_loss.item())
                    last_actor_loss_val = float(actor_loss.item())
                    if tb_writer is not None:
                        tb_writer.add_scalar("hybrid/critic_loss", last_critic_loss_val, loss_step)
                        tb_writer.add_scalar("hybrid/actor_loss", last_actor_loss_val, loss_step)

                    update_counter = loss_step

            if done_flag:
                last_episode_return = episode_return
                if not forced_reset and hasattr(env, "trigger_reset_signal"):
                    try:
                        env.trigger_reset_signal()  # type: ignore[call-arg]
                    except Exception:
                        pass
                logger.log({
                    "step": global_step,
                    "episode_return": float(episode_return),
                    "episode_len": int(episode_len),
                })
                if tb_writer is not None:
                    tb_writer.add_scalar("hybrid/episode_return", float(episode_return), global_step)
                    tb_writer.add_scalar("hybrid/episode_len", float(episode_len), global_step)
                if print_step_metrics:
                    print(f"[hybrid] episode end | steps={episode_len} return={episode_return:.3f}")
                if forced_reset:
                    obs = np.asarray(next_obs, dtype=np.float32)
                    info = {}
                else:
                    obs, info = reset_env(env)
                episode_return = 0.0
                episode_len = 0
                continue

            if global_step % 500 == 0 and global_step > 0:
                save_checkpoint(save_path, {
                    "actor": actor.state_dict(),
                    "critics": critics.state_dict(),
                    "opt_actor": opt_actor.state_dict(),
                    "opt_critic": opt_critic.state_dict(),
                    "step": global_step,
                })

    finally:
        logger.close()
        if tb_writer is not None:
            final_step = global_step
            if last_reward_val is not None:
                tb_writer.add_scalar("hybrid/final_step_reward", last_reward_val, final_step)
            if last_distance_val is not None:
                tb_writer.add_scalar("hybrid/final_step_distance", last_distance_val, final_step)
            if last_critic_loss_val is not None:
                tb_writer.add_scalar("hybrid/final_critic_loss", last_critic_loss_val, final_step)
            if last_actor_loss_val is not None:
                tb_writer.add_scalar("hybrid/final_actor_loss", last_actor_loss_val, final_step)
            final_episode_metric = episode_return if episode_len > 0 else last_episode_return
            tb_writer.add_scalar("hybrid/final_episode_return", final_episode_metric, final_step)
            tb_writer.flush()
        if tb_writer is not None:
            tb_writer.close()
        save_checkpoint(save_path, {
            "actor": actor.state_dict(),
            "critics": critics.state_dict(),
            "opt_actor": opt_actor.state_dict(),
            "opt_critic": opt_critic.state_dict(),
            "step": global_step,
        })
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline_path", type=str, default="data/offline_buffer.jsonl")
    parser.add_argument("--obs_dim", type=int, default=3)
    parser.add_argument("--n_actions", type=int, default=11)
    parser.add_argument("--online_capacity", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--utd_ratio", type=int, default=4)
    parser.add_argument("--total_env_steps", type=int, default=2000)
    parser.add_argument("--start_learning", type=int, default=1000)
    parser.add_argument("--offline_fraction", type=float, default=0.3)
    parser.add_argument("--n_critics", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--critic_lr", type=float, default=3e-7)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--log_path", type=str, default="data/hybrid_log.csv")
    parser.add_argument("--ckpt_path", type=str, default="data/offline_ckpt.pt")
    parser.add_argument("--save_ckpt_path", type=str, default="data/hybrid_ckpt.pt")
    parser.add_argument("--env_type", type=str, default="unity", choices=["unity", "simple"])
    parser.add_argument("--max_steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--target_update_freq", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--reward_clip", type=float, default=5.0)
    parser.add_argument("--normalize_online_rewards", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=False)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--distance_reward_coef", type=float, default=-0.05)
    parser.add_argument("--distance_penalty_threshold", type=float, default=0.8)
    parser.add_argument("--distance_penalty_coef", type=float, default=0.3)
    parser.add_argument("--print_step_metrics", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=True)
    parser.add_argument("--tb_log_dir", type=str, default="runs/hybrid")
    args = parser.parse_args()

    train_hybrid(
        offline_path=args.offline_path,
        obs_dim=args.obs_dim,
        n_actions=args.n_actions,
        online_capacity=args.online_capacity,
        batch_size=args.batch_size,
        utd_ratio=args.utd_ratio,
        total_env_steps=args.total_env_steps,
        start_learning=args.start_learning,
        offline_fraction=args.offline_fraction,
        n_critics=args.n_critics,
        gamma=args.gamma,
        alpha=args.alpha,
        critic_lr=args.critic_lr,
        actor_lr=args.actor_lr,
        device=args.device,
        log_path=args.log_path,
        ckpt_path=args.ckpt_path,
        save_ckpt_path=args.save_ckpt_path,
        env_type=args.env_type,
        max_steps=args.max_steps,
        seed=args.seed,
        tau=args.tau,
        target_update_freq=args.target_update_freq,
        max_grad_norm=args.max_grad_norm,
        reward_clip=args.reward_clip,
        normalize_online_rewards=args.normalize_online_rewards,
        reward_scale=args.reward_scale,
        distance_reward_coef=args.distance_reward_coef,
        distance_penalty_threshold=args.distance_penalty_threshold,
        distance_penalty_coef=args.distance_penalty_coef,
        print_step_metrics=args.print_step_metrics,
        tb_log_dir=args.tb_log_dir,
    )
