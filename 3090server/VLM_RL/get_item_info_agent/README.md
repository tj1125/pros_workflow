# Get Item Info Agent Server (Legacy SAM3D)

`get_item_info_agent` 是保留版 SAM3D/full 3D 感知服務。它接收多視角 RGB 影像與目標類別，執行：

```text
RGB images -> YOLO -> SAM -> Triangulation -> DepthAnything -> SAM3D
  -> Pose Alignment -> GraspGen -> Goal Pose
```

現行 Commander 主流程改用 `get_item_info_agent_no_sam3d`。本服務保留給需要 SAM3D mesh reconstruction 的實驗。兩者預設都使用 port `8006`，同一台 host 上不能同時用同一 port 啟動。

## 執行環境

- Conda 環境建議：`get_item_info_agent`
- 主要依賴：`torch`, `segment_anything`, `ultralytics`, `pytorch3d`, `GraspGen`
- Port：8006

```bash
conda create -n get_item_info_agent python=3.11 -y
conda activate get_item_info_agent

pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r get_item_info_agent/requirements.txt
pip install --no-build-isolation -e get_item_info_agent/vendor/graspgen_runtime/pointnet2_ops
```

## 啟動

在 `3090server/VLM_RL` 下：

```bash
conda activate get_item_info_agent
EXTERNAL_IP=192.168.1.10 python -m get_item_info_agent
```

Commander `.env` legacy fallback：

```env
INF_GET_ITEM_INFO_URL=http://192.168.1.10:8006
```

## A2A Request

`parts[0].text` 是 JSON，`parts[1..N]` 是與 `camera_names` 同順序的 RGB image base64 或 inline data：

```json
{
  "yolo_class": "doll",
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
