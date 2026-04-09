#!/usr/bin/env python3
"""
visualize_grasps.py
載入 inference_result.npz，在 MeshCat 中顯示：
  - 場景點雲（灰色）
  - 物件點雲（綠色）
  - 無碰撞抓取姿態（品質色）
  - 碰撞抓取姿態（紅色，最多 20 個）
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# GraspGen vendor path (meshcat_utils, robot)
# ------------------------------------------------------------------
AGENT_ROOT      = Path("/home/tjchen/workspace/VLM_RL/3090server/VLM_RL/get_item_info_agent")
GRASPGEN_VENDOR = AGENT_ROOT / "vendor/graspgen_runtime"

for p in [str(GRASPGEN_VENDOR), str(GRASPGEN_VENDOR / "pointnet2_ops")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import trimesh.transformations as tra  # type: ignore
from grasp_gen.utils.meshcat_utils import (  # type: ignore
    create_visualizer,
    get_color_from_score,
    visualize_grasp,
    visualize_pointcloud,
)

DEFAULT_RESULT_NPZ = Path(
    "/home/tjchen/workspace/VLM_RL/RL_train/View_Agent/test_tmp/pics/Camera_Room1_test/"
    "four_camera_eye_level_experiment_outputs/Camera_Room1_12/"
    "doll_eye_level_grasp_result.npz"
)
GRIPPER_CONFIG = AGENT_ROOT.parent / "models/graspgen_checkpoints/graspgen_robotiq_2f_140.yml"
SCENE_MAX_DISP = 2000000
SCENE_POINT_SIZE = 0.004
OBJECT_POINT_SIZE = 0.007


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize GraspGen results in MeshCat.")
    parser.add_argument("result_npz", nargs="?", default=str(DEFAULT_RESULT_NPZ))
    return parser.parse_args()


def main():
    args = parse_args()
    result_npz = Path(args.result_npz).expanduser().resolve()
    if not result_npz.exists():
        raise FileNotFoundError(result_npz)

    # 1. Load results
    data = np.load(str(result_npz), allow_pickle=True)
    all_grasps  = data["all_grasps"]            # (N, 4, 4)
    all_scores  = data["all_scores"]            # (N,)
    coll_mask   = data["collision_free_mask"]   # (N,) bool
    pc_obj      = data["pc_object"]             # (M, 3)
    pc_scene    = data["pc_scene"]              # (K, 3)
    pc_obj_raw  = data["pc_object_raw"] if "pc_object_raw" in data.files else pc_obj
    obj_colors  = data["pc_object_colors"] if "pc_object_colors" in data.files else None
    obj_raw_colors = data["pc_object_raw_colors"] if "pc_object_raw_colors" in data.files else obj_colors
    scene_colors = data["pc_scene_colors"] if "pc_scene_colors" in data.files else None

    free_grasps = all_grasps[coll_mask]
    free_scores = all_scores[coll_mask]
    coll_grasps = all_grasps[~coll_mask]

    print(f"Loaded: {len(all_grasps)} total grasps, "
          f"{len(free_grasps)} collision-free, "
          f"{len(coll_grasps)} colliding")
    print(f"Result NPZ: {result_npz}")
    print(f"Object PC: {len(pc_obj)} filtered pts  |  {len(pc_obj_raw)} display pts  |  Scene PC: {len(pc_scene)} pts")

    # 2. Get gripper name
    from grasp_gen.grasp_server import load_grasp_cfg  # type: ignore
    cfg = load_grasp_cfg(str(GRIPPER_CONFIG))
    gripper_name = cfg.data.gripper_name
    print(f"Gripper: {gripper_name}")

    # 3. Create visualizer (connects to running meshcat-server on port 6000)
    vis = create_visualizer()
    vis.delete()   # clear previous scene
    print(f"MeshCat URL: {vis.url()}")

    # 4. Scene point cloud — gray
    if len(pc_scene) > 0:
        # Downsample for display
        idx = (np.random.choice(len(pc_scene), SCENE_MAX_DISP, replace=False)
               if len(pc_scene) > SCENE_MAX_DISP else np.arange(len(pc_scene)))
        visualize_pointcloud(
            vis,
            "scene_pc",
            pc_scene[idx],
            scene_colors[idx] if scene_colors is not None else [128, 128, 128],
            size=SCENE_POINT_SIZE,
        )

    # 5. Object point cloud — green
    if len(pc_obj_raw) > 0:
        visualize_pointcloud(
            vis,
            "object_pc",
            pc_obj_raw,
            obj_raw_colors if obj_raw_colors is not None else [0, 220, 80],
            size=OBJECT_POINT_SIZE,
        )

    # 6. Collision-free grasps — quality colour (blue→red)
    free_colors = get_color_from_score(free_scores, use_255_scale=True)
    for i, (grasp, score_color) in enumerate(zip(free_grasps, free_colors)):
        visualize_grasp(
            vis, f"collision_free/{i:03d}",
            grasp, color=score_color,
            gripper_name=gripper_name, linewidth=0.8,
        )

    # 7. Colliding grasps — red (up to 20)
    for i, grasp in enumerate(coll_grasps[:20]):
        visualize_grasp(
            vis, f"colliding/{i:03d}",
            grasp, color=[220, 40, 40],
            gripper_name=gripper_name, linewidth=0.3,
        )

    print(f"\nVisualization ready!")
    print(f"  → Green cloud   : object ({len(pc_obj_raw)} display pts)")
    print(f"  → Gray cloud    : scene background")
    print(f"  → Colored arms  : {len(free_grasps)} collision-free grasps (blue=low, red=high quality)")
    print(f"  → Red arms(dim) : {min(len(coll_grasps), 20)} colliding grasps")
    print(f"\nOpen browser at: {vis.url()}")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
