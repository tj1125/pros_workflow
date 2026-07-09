# Get Item Info Agent No SAM3D

`get_item_info_agent_no_sam3d` is the RTX 3090 A2A item-info server used by the current main flow. It does not run SAM3D mesh reconstruction; instead it uses multi-view RGB, bounding boxes, `/world_position_data`, and geometry fusion to estimate target info and produce ranked goal poses.

## Pipeline

```text
world_position_data + multi-view RGB + bboxes
  -> topic_input parsing
  -> SAM / geometry size fusion
  -> obstacle and map feasibility check
  -> GraspGen candidate formatting
  -> ranked goal_pose output
```

Core modules:

- `agent_executor.py`: A2A request/response wrapper.
- `pipeline/pipeline.py`: the main pipeline.
- `pipeline/steps/topic_input.py`: parses `/world_position_data` and camera observations.
- `pipeline/steps/goal_pose.py`: produces the Nav2/Unity goal-pose ranking.
- `configs/scene.default.yaml`: camera parameters, model paths, and keepout-map settings.

## Environment

- Suggested conda env: `get_item_info_no_sam3d`
- Port: 8006
- Requires CUDA and the SAM/GraspGen model weights.

Install under `3090server/pros_workflow`:

```bash
conda create -n get_item_info_no_sam3d python=3.11 -y
conda activate get_item_info_no_sam3d
pip install -r requirements.txt
pip install --no-build-isolation -e tool/graspgen_runtime/pointnet2_ops
```

This service uses the shared root `requirements.txt` and the shared GraspGen runtime at `tool/graspgen_runtime`; it does not depend on the `get_item_info_agent` folder.

## Running

```bash
cd /path/to/pros_workflow/3090server/pros_workflow
conda activate get_item_info_no_sam3d
EXTERNAL_IP=<gpu-host> python -m get_item_info_agent_no_sam3d
```

Commander `.env`:

```env
INF_GET_ITEM_INFO_NO_SAM3D_URL=http://<gpu-host>:8006
```

## A2A Request

`parts[0].text` is JSON; `parts[1..N]` are RGB images (base64 or inline data) in the same order as `camera_names`:

```json
{
  "yolo_class": "apple",
  "target_item_id": "apple",
  "target_instance_id": 1,
  "target_instance_key": "apple_1",
  "target_topic_key": "apple",
  "target_label": "red apple",
  "selected_camera": "Camera_Room1_12",
  "camera_names": ["Camera_Room1_12", "Camera_Room1_13"],
  "center_world": [1.2, 0.4, 2.8],
  "bboxes_by_camera": {
    "Camera_Room1_12": [100, 100, 300, 300]
  },
  "world_position_data": {
    "data": "{\"...\":\"...\"}"
  }
}
```

## A2A Response

```json
{
  "center_world": [1.2, 0.4, 2.8],
  "center_world_coordinate_frame": "unity_world",
  "primary_camera_id": "Camera_Room1_12",
  "target_instance_key": "apple_1",
  "target_object": {
    "label": "red apple",
    "id": 1,
    "instance_id": 1,
    "instance_key": "apple_1"
  },
  "group_ranking": [
    {
      "rank": 1,
      "orientation_group": "front",
      "best_confidence": 0.87,
      "best_goal_pose_ros_map": [2.5, 3.1],
      "best_goal_pose_unity": [],
      "map_feasible": true,
      "selection_mode": "rank_best",
      "grasp_goal_poses": []
    }
  ],
  "goal_pose_path": "/tmp/get_item_info_no_sam3d_xxxxx/goal_pose.json",
  "objects": [],
  "num_matched_objects": 1
}
```

The Commander converts `group_ranking[*].best_goal_pose_ros_map` into a Nav2 `/goal_pose`, and tries the next candidate rank by rank in `major_nav_node`.
