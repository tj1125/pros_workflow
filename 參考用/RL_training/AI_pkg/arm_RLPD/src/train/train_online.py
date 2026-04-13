from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.envs.make_envs import make_vector_envs
from src.buffers.replay_buffer import ReplayBuffer
from src.models.models import DiscreteActor, QEnsemble
from src.utils.utils import CSVLogger, td_target, save_checkpoint, maybe_load_checkpoint


def train_online(
    num_envs: int = 3,
    obs_dim: int = 3,
    n_actions: int = 7,
    buffer_capacity: int = 200000,
    total_env_steps: int = 3_000,
    start_learning: int = 1000,
    batch_size: int = 256,
    utd_ratio: int = 1,
    gamma: float = 0.99,
    alpha: float = 0.2,
    critic_lr: float = 3e-7,
    actor_lr: float = 1e-4,
    device: str = "cpu",
    log_path: str = "data/online_log.csv",
    ckpt_path: str = "data/online_ckpt.pt",
):
    device = torch.device(device)

    env = make_vector_envs(num_envs=num_envs)
    buffer = ReplayBuffer(capacity=buffer_capacity, obs_shape=(obs_dim,), action_shape=())

    actor = DiscreteActor(obs_dim, n_actions).to(device)
    critics = QEnsemble(obs_dim, n_actions, n_critics=2).to(device)
    opt_actor = Adam(actor.parameters(), lr=actor_lr)
    opt_critic = Adam(critics.parameters(), lr=critic_lr)
    logger = CSVLogger(log_path, fieldnames=["step", "reward", "critic_loss", "actor_loss"])

    state = env.reset()
    # gym.vector returns (obs, info) for gymnasium
    if isinstance(state, tuple):
        obs, _ = state
    else:
        obs = state
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    global_step = 0

    # Optional resume
    ckpt = maybe_load_checkpoint(ckpt_path, device=str(device))
    if ckpt is not None:
        actor.load_state_dict(ckpt["actor"])  # type: ignore
        critics.load_state_dict(ckpt["critics"])  # type: ignore
        opt_actor.load_state_dict(ckpt["opt_actor"])  # type: ignore
        opt_critic.load_state_dict(ckpt["opt_critic"])  # type: ignore

    while global_step < total_env_steps:
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        with torch.no_grad():
            a, _ = actor.act(obs_t, deterministic=False)
        actions = a.cpu().numpy()

        step_out = env.step(actions)
        # Gymnasium: (obs, reward, terminated, truncated, info)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            next_obs, reward, terminated, truncated, info = step_out
            done = np.logical_or(terminated, truncated).astype(np.float32)
        else:
            next_obs, reward, done, info = step_out  # type: ignore
            done = done.astype(np.float32)

        # Store transitions
        for i in range(num_envs):
            buffer.add(
                obs[i].astype(np.float32),
                int(actions[i]),
                float(reward[i]),
                next_obs[i].astype(np.float32),
                float(done[i]),
                is_offline=False,
            )
        obs = next_obs
        episode_rewards += reward
        global_step += num_envs

        # Log episodic reward when episode ends (approx via done flag)
        if np.any(done > 0.5):
            for i in np.where(done > 0.5)[0]:
                logger.log({"step": global_step, "reward": float(episode_rewards[i])})
                episode_rewards[i] = 0.0

        # Learn
        if buffer.size >= start_learning:
            for _ in range(utd_ratio):
                t = buffer.sample(batch_size, balanced=False)
                o = torch.as_tensor(t.obs, device=device)
                a = torch.as_tensor(t.action.squeeze(-1), device=device, dtype=torch.long)
                r = torch.as_tensor(t.reward, device=device)
                no = torch.as_tensor(t.next_obs, device=device)
                d = torch.as_tensor(t.done, device=device)

                with torch.no_grad():
                    logits_next = actor(no)
                    logp_next = F.log_softmax(logits_next, dim=-1)
                    p_next = logp_next.exp()
                    q_next_min = critics.min_q(no)
                    v_next = (p_next * (q_next_min - alpha * logp_next)).sum(dim=-1, keepdim=True)
                    target = td_target(r, d, v_next, gamma)

                q_values_list = critics(o)
                critic_loss = 0.0
                for q in q_values_list:
                    qa = q.gather(1, a.view(-1, 1))
                    critic_loss = critic_loss + F.mse_loss(qa, target)
                opt_critic.zero_grad()
                critic_loss.backward()
                opt_critic.step()

                logits = actor(o)
                logp = F.log_softmax(logits, dim=-1)
                p = logp.exp()
                q_min = critics.min_q(o)
                actor_loss = -(p * (q_min - alpha * logp)).sum(dim=-1).mean()
                opt_actor.zero_grad()
                actor_loss.backward()
                opt_actor.step()

            if global_step % 500 == 0:
                save_checkpoint(ckpt_path, {
                    "actor": actor.state_dict(),
                    "critics": critics.state_dict(),
                    "opt_actor": opt_actor.state_dict(),
                    "opt_critic": opt_critic.state_dict(),
                    "step": global_step,
                })

    logger.close()


if __name__ == "__main__":
    train_online()
