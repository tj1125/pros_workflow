from __future__ import annotations

from pathlib import Path

# Root of this agent package (pros_workflow/OtherServer/pros_workflow/get_item_info_agent)
AGENT_ROOT = Path(__file__).resolve().parents[1]

# Convenience aliases that downstream code may reference
DEFAULT_SCENE_CONFIG = AGENT_ROOT / "configs" / "scene.default.yaml"
