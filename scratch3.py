def _grasp_payload_from_params(params) -> dict | None:
    for key in ("grasp_result_payload", "grasp_result", "latest_grasp_result"):
        payload = params.get(key)
        if isinstance(payload, dict):
            nested_result = payload.get("result")
            if isinstance(nested_result, dict):
                return nested_result
            return payload
    return None

params = {"grasp_result": {"object_id": "test", "valid_grasp_poses_camera": []}}
print("grasp_payload:", _grasp_payload_from_params(params))
params2 = {}
print("grasp_payload empty:", _grasp_payload_from_params(params2))
