# Get Item Info Agent No SAM3D

`get_item_info_agent_no_sam3d` 是現行主流程使用的 RTX 3090 A2A item-info server。它不跑 SAM3D mesh reconstruction，而是使用多視角 RGB、bbox、`/world_position_data` 與幾何融合，估計目標資訊並產生 ranked goal poses。

## Pipeline

```text
world_position_data + multi-view RGB + bboxes
  -> topic_input parsing
  -> SAM / geometry size fusion
  -> obstacle and map feasibility check
  -> GraspGen candidate formatting
  -> ranked goal_pose output
```

核心程式：

- `agent_executor.py`: A2A request/response wrapper。
- `pipeline/pipeline.py`: 主 pipeline。
- `pipeline/steps/topic_input.py`: 解析 `/world_position_data` 與相機觀測。
- `pipeline/steps/goal_pose.py`: 產生 Nav2/Unity goal pose ranking。
- `configs/scene.default.yaml`: camera parameters、模型與 keepout map 設定。

## 執行環境

- Conda 環境建議：`get_item_info_no_sam3d`
- Port：8006
- 需要 CUDA 與 SAM/GraspGen 相關模型權重。

在 `3090server/VLM_RL` 下安裝：

```bash
conda create -n get_item_info_no_sam3d python=3.11 -y
conda activate get_item_info_no_sam3d
pip install -r get_item_info_agent/requirements.txt
pip install --no-build-isolation -e get_item_info_agent/vendor/graspgen_runtime/pointnet2_ops
```

此服務目前共用 `get_item_info_agent` 的 GraspGen runtime 與部分 requirements；若未來拆出專屬 requirements，請同步更新本文件。

## 啟動

```bash
cd /path/to/VLM_RL/3090server/VLM_RL
conda activate get_item_info_no_sam3d
EXTERNAL_IP=192.168.1.10 python -m get_item_info_agent_no_sam3d
```

Commander `.env`：

```env
INF_GET_ITEM_INFO_NO_SAM3D_URL=http://192.168.1.10:8006
```

## A2A Request

`parts[0].text` 是 JSON，`parts[1..N]` 是與 `camera_names` 同順序的 RGB image base64 或 inline data：

```json
{
  "yolo_class": "apple",
  "target_item_id": "apple",
  "target_instance_id": 1,
  "target_instance_key": "apple_1",
  "target_topic_key": "apple",
  "target_label": "紅色蘋果",
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
    "label": "紅色蘋果",
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

Commander 會將 `group_ranking[*].best_goal_pose_ros_map` 轉成 Nav2 `/goal_pose`，並在 `major_nav_node` 中逐 rank 嘗試下一個候選。
