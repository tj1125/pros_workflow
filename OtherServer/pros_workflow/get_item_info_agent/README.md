# Get Item Info Agent Server (Legacy SAM3D)

`get_item_info_agent` is the SAM3D / full-3D perception service. It receives multi-view RGB images and a target class and runs:

```text
RGB images -> YOLO -> SAM -> Triangulation -> DepthAnything -> SAM3D
  -> Pose Alignment -> GraspGen -> Goal Pose
```

This service provides the full 3D pipeline for cases that need SAM3D mesh reconstruction. It is disabled by default and uses port `8008` (the main flow uses `get_item_info_agent_no_sam3d` on :8006).

## Environment

- Suggested conda env: `get_item_info_agent`
- Main deps: `torch`, `segment_anything`, `ultralytics`, `pytorch3d`, `GraspGen`
- Port: 8008

```bash
conda create -n get_item_info_agent python=3.11 -y
conda activate get_item_info_agent

pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r get_item_info_agent/requirements.txt
pip install --no-build-isolation -e get_item_info_agent/vendor/graspgen_runtime/pointnet2_ops
```

## Running

Under `OtherServer/pros_workflow`:

```bash
conda activate get_item_info_agent
EXTERNAL_IP=<gpu-host> python -m get_item_info_agent
```

Set this only when intentionally running the legacy client:

```env
INF_GET_ITEM_INFO_URL=http://<gpu-host>:8008
```

## A2A Request

`parts[0].text` is JSON; `parts[1..N]` are RGB images (base64 or inline data) in the same order as `camera_names`:

```json
{
  "yolo_class": "apple",
  "scene_config": "get_item_info_agent/configs/scene.default.yaml",
  "selected_camera": "Camera_Room1_12",
  "camera_names": ["Camera_Room1_12", "Camera_Room1_13"]
}
```

## A2A Response

```json
{
  "center_world": [1.23, -0.45, 0.89],
  "group_ranking": [
    {
      "rank": 1,
      "best_confidence": 0.87,
      "best_goal_pose_ros_map": [2.5, 3.1],
      "best_pose_unity": [],
      "best_pose_matrix_unity": [],
      "best_goal_pose_unity": []
    }
  ],
  "goal_pose_path": "/tmp/get_item_info_xxxxx/goal_pose.json"
}
```
