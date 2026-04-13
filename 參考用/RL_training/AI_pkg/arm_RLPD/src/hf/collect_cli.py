import argparse
import os
import sys
import json
from typing import Optional, Tuple

import numpy as np

from src.envs.unity_arm_env import UnityArmEnv
from src.hf.hf_data_collector import HFDataCollector


def get_obs_info(env) -> Tuple[np.ndarray, dict]:
    state = env.reset()
    if isinstance(state, tuple):
        obs, info = state
    else:
        obs, info = state, {}
    return obs, info


def main():
    parser = argparse.ArgumentParser(description="Interactive human-feedback data collector for Unity Arm")
    parser.add_argument("--out", type=str, default="data/offline_buffer.jsonl", help="Output JSONL path")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=0)
    # --render argument is removed as rendering is handled by Unity
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"Saving to: {args.out}")

    print("Initializing UnityArmEnv (this may take a moment)...")
    # Instantiate the Unity Arm environment. Seed is passed to the env which is used in reset().
    env = UnityArmEnv(max_steps=args.max_steps, seed=args.seed)
    hf = HFDataCollector()

    for ep in range(args.episodes):
        obs, _ = get_obs_info(env)
        done = False
        step = 0
        print(f"\nEpisode {ep+1}/{args.episodes}")
        while not done and step < args.max_steps:
            # Rendering is handled by the Unity simulation itself

            print(f"Obs (dx,dy,dz) = {np.round(obs,3)}  | distance={np.linalg.norm(obs):.3f}")
            try:
                text = input("Instruction (e.g., 'move right/up/forward/stay'): ").strip()
            except EOFError:
                print("\nEOF received; exiting.")
                return

            action_id = hf.parse_text_to_action(text)

            if action_id == 0:
                print("動作為 'none'，跳過此步驟。")
                continue

            action_str = hf.ACTION_MAP.get(action_id, "none")

            print(f'正在執行 "{action_str}"...')

            step_out = env.step(action_str)
            # Gymnasium: (obs, reward, terminated, truncated, info)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                done_flag = terminated or truncated
            else:
                next_obs, reward, done_flag, info = step_out  # type: ignore

            # Get human feedback score (1-7 scale or g/b shorthand)
            human_level: Optional[float] = None
            while human_level is None:
                try:
                    score_str = input("這個動作的評分？(1-7 / g=好 / b=壞 / Enter=略過): ").strip().lower()
                except EOFError:
                    print("\nEOF received; exiting.")
                    return

                if score_str == "":
                    break
                if score_str in {"g", "good", "y", "yes"}:
                    human_level = 7
                elif score_str in {"b", "bad", "n", "no"}:
                    human_level = 1
                else:
                    try:
                        lvl = int(score_str)
                        if 1 <= lvl <= 7:
                            human_level = lvl
                        else:
                            print("請輸入 1-7、g 或 b。")
                    except ValueError:
                        print("請輸入 1-7、g 或 b。")

            # Persist one transition
            hf.record(
                obs,
                action_id,
                reward,
                next_obs,
                done_flag,
                args.out,
                text,
                human_feedback=human_level,
            )

            obs = next_obs
            done = done_flag
            step += 1

    env.close()
    print("\nCollection finished.")


if __name__ == "__main__":
    main()
