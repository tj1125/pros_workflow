# Grasp Agent Server

`grasp_agent` is an A2A Agent Server that receives `Camera_Car` RGBD and an `object_id` and runs, on the 3090:

`RGBD -> YOLOv26 (yolov26_best.pt) -> SAM -> target point cloud -> GraspGen -> valid grasps -> return all feasible grasp poses (keeping the best grasp)`

## A2A Request

The `parts[0].text` the client sends to the server is a single JSON object:

```json
{
  "object_id": "doll",
  "camera_name": "Camera_Car",
  "rgb_base64": "...",
  "depth_base64": "..."
}
```

## A2A Response

The server returns JSON; the key fields include:

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

`best_grasp_pose_camera.frame` is currently `camera`, because this version only needs the Camera_Car intrinsics and uses no extrinsics.
`valid_grasp_poses_camera` returns all feasible grasps, sorted first by nearest grasp position relative to `gripper_midpoint_camera_xyz`, then by `grasp_confidence` as a secondary key.
`best_grasp_pose_camera` is kept for compatibility and equals the first pose in `valid_grasp_poses_camera`.

## Tool

Helpers shared across services live in `3090server/pros_workflow/tool/`:

- `tool/vision/yolo.py`: YOLO helper shared by get_item_info / grasp.
- `tool/vision/sam.py`: SAM helper shared by get_item_info / grasp.
- `tool/grasp/graspgen.py`: GraspGen helper shared by get_item_info / grasp.

The GraspGen runtime uses `get_item_info_agent/vendor/graspgen_runtime` as the canonical source; the services share one set of helpers and one root resolver, overridden only when a valid external path is set explicitly.

## Config

Default config:

- `grasp_agent/configs/runtime.default.yaml`

Overridable via env vars:

- `GRASP_YOLO_WEIGHTS`
- `GRASP_SAM_CHECKPOINT`
- `GRASP_GRASPGEN_ROOT`
- `GRASP_GRIPPER_CONFIG`
- `GRASP_CAMERA_INTRINSICS`

## Running

Under `3090server/pros_workflow`:

```bash
python -m grasp_agent
```

The service port is `8007`.
