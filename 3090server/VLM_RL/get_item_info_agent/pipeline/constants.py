from __future__ import annotations

from pathlib import Path

# Root of this agent package (VLM_RL/3090server/VLM_RL/get_item_info_agent)
AGENT_ROOT = Path(__file__).resolve().parents[1]

# Convenience aliases that downstream code may reference
DEFAULT_SCENE_CONFIG = AGENT_ROOT / "configs" / "scene.default.yaml"
