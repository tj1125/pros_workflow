"""
scripts/run_train.py — Stage 3b: SAC Reinforcement Learning Training Entry Point

Usage:
    python scripts/run_train.py --config configs/train_config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.sac_trainer import SACTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="View Agent — SAC RL Training")
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    trainer = SACTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
