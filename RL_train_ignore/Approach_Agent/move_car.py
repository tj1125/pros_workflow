"""Legacy CLI/import shim for base approach car movement."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.car_approach.move_car import *  # noqa: F401,F403
from agents.car_approach.move_car import main


if __name__ == "__main__":
    main()
