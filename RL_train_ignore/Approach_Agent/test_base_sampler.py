"""Legacy CLI shim for the runtime car approach agent.

Runtime code lives in agents.car_approach. This file keeps the old command path
working without duplicating the agent implementation under RL_train.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.car_approach.base_sampler import ApproachAgentRunConfig, main, run_approach_agent

__all__ = ["ApproachAgentRunConfig", "main", "run_approach_agent"]


if __name__ == "__main__":
    main()
