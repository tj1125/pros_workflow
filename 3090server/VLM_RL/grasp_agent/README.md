# Grasp Agent Server

`grasp_agent` 是一個 A2A Agent Server，負責接收 `Camera_Car` 的 RGBD 與 `object_id`，在 3090 端執行：

`RGBD -> YOLO(best.pt) -> SAM -> 目標點雲 -> GraspGen -> valid grasps -> 優先選離夾爪中點最近的 grasp pose`

## A2A Request

client 送到 server 的 `parts[0].text` 內容是單一 JSON：

```json
{
  "object_id": "doll",
  "camera_name": "Camera_Car",
  "rgb_base64": "...",
  "depth_base64": "..."
}
```

## A2A Response

server 會回傳 JSON，重點欄位包含：

- `bbox_xyxy`
- `detection_confidence`
- `grasp_confidence`
- `gripper_midpoint_camera_xyz`
- `grasp_distance_to_gripper_midpoint_m`
- `grasp_distance_to_camera_m`
- `best_grasp_pose_camera`
- `object_reference_center_camera`

`best_grasp_pose_camera.frame` 目前是 `camera`，因為這版只需要 Camera_Car 內參，不使用外參。
`best_grasp_pose_camera` 會先以 `gripper_midpoint_camera_xyz` 為 reference，選 grasp position 最近者，再用 `grasp_confidence` 當次要排序。

## Tool

會被多個 agents 共用的工具，放在 `3090server/VLM_RL/tool/`：

- `tool/vision/yolo.py`: find / get_item_info / grasp 共用 YOLO helper
- `tool/vision/sam.py`: get_item_info / grasp 共用 SAM helper
- `tool/grasp/graspgen.py`: get_item_info / grasp 共用 GraspGen helper

大型 vendor 目錄目前仍沿用現有 `get_item_info_agent/vendor/` 內容；`grasp_agent` 會優先讀 config / env 指定路徑，若 `graspgen_root` 不存在，會 fallback 到 `get_item_info_agent/vendor/graspgen_runtime`。

## Config

預設 config 在：

- `grasp_agent/configs/runtime.default.yaml`

可用以下 env 覆蓋：

- `GRASP_YOLO_WEIGHTS`
- `GRASP_SAM_CHECKPOINT`
- `GRASP_GRASPGEN_ROOT`
- `GRASP_GRIPPER_CONFIG`
- `GRASP_CAMERA_INTRINSICS`

## 啟動

在 `3090server/VLM_RL` 下：

```bash
python -m grasp_agent
```

服務 port 是 `8007`。
