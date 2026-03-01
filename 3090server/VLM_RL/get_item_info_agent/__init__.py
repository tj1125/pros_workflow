"""
get_item_info_agent — A2A server for 3D item info estimation.

Pipeline: stereo RGB → YOLO → SAM → Triangulation → DepthAnything
          → SAM3D → Pose Alignment → GraspGen → goal_pose JSON

Input  (A2A message, 3 parts):
  [0] text  : JSON { "yolo_class": "doll", "scene_config": "..." }
  [1] data  : camera-A image bytes (PNG/JPG)
  [2] data  : camera-B image bytes (PNG/JPG)

Output (A2A response):
  JSON { "center_world": [...], "group_ranking": [...], "goal_pose_path": "..." }
"""

__version__ = "2.0.0"
