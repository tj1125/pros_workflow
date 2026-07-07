# Get Item Info Agent Server (Legacy SAM3D)

`get_item_info_agent` 是保留版 SAM3D/full 3D 感知服務。它接收多視角 RGB 影像與目標類別，執行：

```text
RGB images -> YOLO -> SAM -> Triangulation -> DepthAnything -> SAM3D
  -> Pose Alignment -> GraspGen -> Goal Pose
```

本服務提供需要 SAM3D mesh reconstruction 的完整 3D pipeline，預設不啟用，使用 port `8008`（Commander 主流程用的是 `get_item_info_agent_no_sam3d` :8006）。

## 執行環境

- Conda 環境建議：`get_item_info_agent`
- 主要依賴：`torch`, `segment_anything`, `ultralytics`, `pytorch3d`, `GraspGen`
- Port：8008

```bash
conda create -n get_item_info_agent python=3.11 -y
conda activate get_item_info_agent

pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r get_item_info_agent/requirements.txt
pip install --no-build-isolation -e get_item_info_agent/vendor/graspgen_runtime/pointnet2_ops
```

## 啟動

在 `3090server/pros_workflow` 下：

```bash
conda activate get_item_info_agent
EXTERNAL_IP=192.168.1.10 python -m get_item_info_agent
```

只有刻意跑 legacy client 時才設定：

```env
INF_GET_ITEM_INFO_URL=http://192.168.1.10:8008
```

## A2A Request

`parts[0].text` 是 JSON，`parts[1..N]` 是與 `camera_names` 同順序的 RGB image base64 或 inline data：

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
