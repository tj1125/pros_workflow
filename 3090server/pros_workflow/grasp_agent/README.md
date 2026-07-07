# Grasp Agent Server

`grasp_agent` 是一個 A2A Agent Server，負責接收 `Camera_Car` 的 RGBD 與 `object_id`，在 3090 端執行：

`RGBD -> YOLOv26(yolov26_best.pt) -> SAM -> 目標點雲 -> GraspGen -> valid grasps -> 回傳所有可行 grasp pose（並保留 best grasp）`

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
- `num_valid_grasps`
- `best_grasp_pose_camera`
- `valid_grasp_poses_camera`
- `object_reference_center_camera`

`best_grasp_pose_camera.frame` 目前是 `camera`，因為這版只需要 Camera_Car 內參，不使用外參。
`valid_grasp_poses_camera` 會回傳所有可行 grasp，排序規則是先以 `gripper_midpoint_camera_xyz` 為 reference，選 grasp position 最近者，再用 `grasp_confidence` 當次要排序。
`best_grasp_pose_camera` 會維持相容，等於 `valid_grasp_poses_camera` 的第一個 pose。

## Tool

會被多個 agents 共用的工具，放在 `3090server/pros_workflow/tool/`：

- `tool/vision/yolo.py`: find / get_item_info / grasp 共用 YOLO helper
- `tool/vision/sam.py`: get_item_info / grasp 共用 SAM helper
- `tool/grasp/graspgen.py`: get_item_info / grasp 共用 GraspGen helper

GraspGen runtime 目前以 `get_item_info_agent/vendor/graspgen_runtime` 為 canonical source；多個 agents 共用同一套 helper 與 root resolver，只有顯式設定有效外部路徑時才會覆蓋。

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

在 `3090server/pros_workflow` 下：

```bash
python -m grasp_agent
```

服務 port 是 `8007`。
