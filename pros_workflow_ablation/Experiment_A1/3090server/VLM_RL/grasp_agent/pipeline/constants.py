from __future__ import annotations

from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
SERVER_ROOT = AGENT_ROOT.parent
DEFAULT_CONFIG = AGENT_ROOT / "configs" / "runtime.default.yaml"
