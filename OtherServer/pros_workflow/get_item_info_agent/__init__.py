"""
get_item_info_agent — A2A server for 3D item info estimation.

Pipeline: stereo RGB → YOLO → SAM → Triangulation → DepthAnything
          → SAM3D → Pose Alignment → GraspGen → goal_pose JSON

Input  (A2A message, 3 parts):
  [0] text  : JSON {
                  "yolo_class": "doll",
                  "scene_config": "...",
                  "selected_camera": "Camera_Room1_1",
                  "camera_names": ["Camera_Room1_1", "Camera_Room1_2", "Camera_Room1_3"]
              }
  [1..N]    : uploaded RGB image bytes/base64 text in the same order as camera_names

Output (A2A response):
  JSON { "center_world": [...], "group_ranking": [...], "goal_pose_path": "..." }
"""

__version__ = "2.0.0"
