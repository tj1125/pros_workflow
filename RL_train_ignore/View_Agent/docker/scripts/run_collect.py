"""
scripts/run_collect.py — Stage 1: Data Collection Entry Point

Collects trajectory data from Unity/Rosbridge and saves to HDF5.

Usage:
    python scripts/run_collect.py --config configs/train_config.yaml
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from collector.trajectory_recorder import TrajectoryRecorder
from collector.randomizer import EnvRandomizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def make_random_action_fn(action_dim: int = 6, scale: float = 0.05):
    """Returns a callable that produces a random action each call."""
    def fn():
        return [random.uniform(-scale, scale) for _ in range(action_dim)]
    return fn


def main():
    parser = argparse.ArgumentParser(description="View Agent — Data Collection")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--n",      type=int, default=None,
                        help="Number of trajectories (overrides config)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    col_cfg = config["collection"]
    n_traj  = args.n or col_cfg["num_trajectories"]
    duration = col_cfg["trajectory_duration_sec"]
    rand_every = col_cfg.get("randomize_every_n", 1)
    action_scale = col_cfg.get("action_scale", 0.05)

    recorder   = TrajectoryRecorder(config)
    randomizer = EnvRandomizer(config["env"]["rosbridge_url"])
    action_fn  = make_random_action_fn(6, action_scale)

    logger.info(f"Starting collection: {n_traj} trajectories × {duration}s each")
    recorder.connect()

    try:
        for i in range(n_traj):
            if i % rand_every == 0:
                seed = random.randint(0, 2**31 - 1)
                randomizer.randomize(seed=seed)
                randomizer.wait_for_ready()

            traj_id = recorder.record_trajectory(action_fn, duration_sec=duration, env_seed=seed)
            logger.info(f"[{i+1}/{n_traj}] Trajectory {traj_id} saved.")
    finally:
        recorder.disconnect()

    logger.info("Collection complete.")


if __name__ == "__main__":
    main()
