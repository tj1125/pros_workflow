from agents.approach.runner import _grasp_payload_from_params
params = {"grasp_result": {"object_id": "test", "valid_grasp_poses_camera": []}}
print("grasp_payload:", _grasp_payload_from_params(params))
