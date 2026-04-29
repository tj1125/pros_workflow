import asyncio
from agents.approach.runner import run_base_approach_sync

params = {
    "grasp_result": {
        "object_id": "test",
        "valid_grasp_poses_camera": [{"matrix_4x4": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]
    },
    "show_gui": False,
    "run_rule_navigation": False,
    "allow_missing_amcl": True,
}

result = run_base_approach_sync(params, context_id="test")
print("Result:", result)
