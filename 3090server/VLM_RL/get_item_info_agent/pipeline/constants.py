from __future__ import annotations

from pathlib import Path

# Root of the original get_item_info project (where models/ and vendor/ live).
# Adjust this path if the project is moved.
GET_ITEM_INFO_ROOT = Path("/home/tjchen/workspace/get_item_info")

# Root of this agent package
AGENT_ROOT = Path(__file__).resolve().parents[1]

# Convenience aliases that downstream code may reference
DEFAULT_SCENE_CONFIG = GET_ITEM_INFO_ROOT / "configs" / "scene.default.yaml"
