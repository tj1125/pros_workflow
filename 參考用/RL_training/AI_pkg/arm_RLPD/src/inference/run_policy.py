import argparse
import csv
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.models.models import DiscreteActor
from src.utils.utils import maybe_load_checkpoint

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


def create_env(env_type: str, max_steps: int, seed: int):
    if env_type == "unity":
        from src.envs.unity_arm_env import UnityArmEnv

        return UnityArmEnv(max_steps=max_steps, seed=seed)
    if env_type == "simple":
        from src.envs.arm_env import SimpleArmEnv

        return SimpleArmEnv(max_steps=max_steps, seed=seed)
    raise ValueError(f"Unknown env_type '{env_type}'. Use 'unity' or 'simple'.")


def reset_env(env) -> tuple[np.ndarray, Dict[str, Any]]:
    state = env.reset()
    if isinstance(state, tuple) and len(state) == 2:
        return state
    return state, {}


def convert_action(action_id: int, env_type: str) -> Any:
    if env_type == "unity":
        return ACTION_ID_TO_STR.get(int(action_id), "none")
    return int(action_id)


def run_inference(
    ckpt_path: str,
    episodes: int = 5,
    obs_dim: int = 3,
    n_actions: int = 11,
    device: str = "cpu",
    env_type: str = "unity",
    max_steps: int = 100,
    deterministic: bool = True,
    log_path: Optional[str] = None,
    seed: int = 0,
    print_steps: bool = True,
) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    ckpt = maybe_load_checkpoint(ckpt_path, device=str(torch_device))
    if ckpt is None or "actor" not in ckpt:
        raise FileNotFoundError(f"Checkpoint '{ckpt_path}' not found or missing 'actor' weights")

    actor = DiscreteActor(obs_dim, n_actions).to(torch_device)
    actor.load_state_dict(ckpt["actor"])  # type: ignore[arg-type]
    actor.eval()

    env = create_env(env_type=env_type, max_steps=max_steps, seed=seed)

    results: list[dict[str, Any]] = []
    completed = 0
    obs, info = reset_env(env)

    while completed < episodes:
        steps = 0
        episode_return = 0.0
        success = False
        while True:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=torch_device).unsqueeze(0)
            with torch.no_grad():
                action_tensor, _ = actor.act(obs_tensor, deterministic=deterministic)
            action_id = int(action_tensor.squeeze(0).item())
            env_action = convert_action(action_id, env_type)

            step_out = env.step(env_action)
            forced_reset = False
            if isinstance(step_out, tuple) and len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
            elif isinstance(step_out, tuple) and len(step_out) == 4:
                next_obs, reward, terminated, info = step_out  # type: ignore[misc]
                truncated = False
            else:
                raise RuntimeError("Unexpected return signature from environment step")

            if isinstance(info, dict):
                forced_reset = bool(info.get("forced_reset", False))
            else:
                info = {}

            steps += 1
            reward_val = float(reward)
            episode_return += reward_val
            if print_steps:
                distance = info.get("distance")
                if distance is None and isinstance(next_obs, np.ndarray):
                    distance = float(np.linalg.norm(next_obs))
                dist_str = f" distance={distance:.3f}" if distance is not None else ""
                print(
                    f"[inference] step {steps:03d} action={action_id} reward={reward_val:.3f}{dist_str}"
                )

            done = bool(terminated or truncated or forced_reset)
            if not done and steps >= max_steps:
                done = True

            if done:
                success = bool(terminated)
                results.append(
                    {
                        "episode": completed + 1,
                        "steps": steps,
                        "return": episode_return,
                        "success": success,
                    }
                )
                print(
                    f"[inference] episode {completed + 1} end | steps={steps} return={episode_return:.3f} success={success}"
                )
                completed += 1
                if completed >= episodes:
                    break
                if forced_reset:
                    obs = next_obs
                    info = {}
                else:
                    obs, info = reset_env(env)
                break
            else:
                obs = next_obs
                info = info
        else:
            continue
        if completed >= episodes:
            break

    if log_path:
        fieldnames = ["episode", "steps", "return", "success"]
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    if hasattr(env, "close"):
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with a trained SAC/RLPD policy")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint path containing actor weights")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--obs_dim", type=int, default=3, help="Observation dimension")
    parser.add_argument("--n_actions", type=int, default=11, help="Number of discrete actions")
    parser.add_argument("--device", type=str, default="mps", choices=["cpu", "cuda", "mps"], help="Torch device")
    parser.add_argument("--env_type", type=str, default="unity", choices=["unity", "simple"], help="Environment backend")
    parser.add_argument("--max_steps", type=int, default=80, help="Max steps per episode")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy output")
    parser.add_argument("--log_path", type=str, default=None, help="Optional CSV to store results")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed")
    parser.add_argument("--no_print_steps", action="store_true", help="Disable per-step logging")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        ckpt_path=args.ckpt_path,
        episodes=args.episodes,
        obs_dim=args.obs_dim,
        n_actions=args.n_actions,
        device=args.device,
        env_type=args.env_type,
        max_steps=args.max_steps,
        deterministic=args.deterministic,
        log_path=args.log_path,
        seed=args.seed,
        print_steps=not args.no_print_steps,
    )
