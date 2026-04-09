#!/usr/bin/env python3
"""
GraspGen inference script with collision filtering.

Usage:
    conda run -n a2a_vlm_find python test_graspgen.py [--class-name doll]

Inputs:
  - RGB:   test_tmp/pics/Camera_Room1/Camera_Room1_1_rgb.png
  - Depth: test_tmp/pics/Camera_Room1/Camera_Room1_1_depth.png
  - Cam:   get_item_info_agent/data/camera_parameter/Intrinsics/Camera_Room1_1.yaml

Output:
  - test_tmp/pics/Camera_Room1/inference_result.npz
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import yaml

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
AGENT_ROOT      = Path("/home/tjchen/workspace/VLM_RL/3090server/VLM_RL/get_item_info_agent")
# Use the vendor copy of GraspGen (no webdataset / dataset_utils dependency)
GRASPGEN_VENDOR = AGENT_ROOT / "vendor/graspgen_runtime"
YOLO_WEIGHTS    = AGENT_ROOT.parent / "models/yolo/pure720.pt"
SAM_CHECKPOINT  = AGENT_ROOT.parent / "models/segmentation/sam_vit_b_01ec64.pth"
GRIPPER_CONFIG  = AGENT_ROOT.parent / "models/graspgen_checkpoints/graspgen_robotiq_2f_140.yml"
INTRINSICS_YAML = AGENT_ROOT / "data/camera_parameter/Intrinsics/Camera_Room1_1.yaml"
RGB_PATH    = Path("/home/tjchen/workspace/VLM_RL/RL_train/View_Agent/test_tmp/pics/Camera_Room1/Camera_Room1_1_rgb.png")
DEPTH_PATH  = Path("/home/tjchen/workspace/VLM_RL/RL_train/View_Agent/test_tmp/pics/Camera_Room1/Camera_Room1_1_depth.png")
OUTPUT_PATH = Path("/home/tjchen/workspace/VLM_RL/RL_train/View_Agent/test_tmp/pics/Camera_Room1/inference_result.npz")

# Use vendor GraspGen (stripped of training/webdataset deps)
for p in [str(GRASPGEN_VENDOR), str(GRASPGEN_VENDOR / "pointnet2_ops")]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("GRASPGEN_NO_VIS", "1")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(AGENT_ROOT / ".cache/torch_extensions"))


# ------------------------------------------------------------------
# Camera intrinsics
# ------------------------------------------------------------------
def load_intrinsics(yaml_path: Path) -> Tuple[float, float, float, float]:
    """Read fx, fy, cx, cy from camera_matrix YAML.
    K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    d = cfg["camera_matrix"]["data"]
    return float(d[0]), float(d[4]), float(d[2]), float(d[5])


# ------------------------------------------------------------------
# Step 1: YOLO → bounding box
# ------------------------------------------------------------------
def detect_bbox(rgb_path: Path, class_name: str) -> np.ndarray:
    """Return [x1, y1, x2, y2] for the highest-confidence detection."""
    from ultralytics import YOLO  # type: ignore

    model = YOLO(str(YOLO_WEIGHTS))
    results = model(str(rgb_path), verbose=False)
    if not results:
        raise ValueError("YOLO returned no results.")

    result = results[0]
    names = {int(k): v for k, v in result.names.items()}
    target = class_name.lower()
    best_box, best_conf = None, -1.0

    for box in result.boxes:
        label = str(names.get(int(box.cls.item()), "")).lower()
        if label == target:
            conf = float(box.conf.item())
            if conf > best_conf:
                best_conf = conf
                best_box = box.xyxy[0].cpu().numpy()

    if best_box is None:
        available = sorted({v.lower() for v in names.values()})
        raise ValueError(f"Class '{class_name}' not found. YOLO known classes: {available}")

    print(f"[YOLO] '{class_name}' conf={best_conf:.3f}  box={best_box}")
    return best_box


# ------------------------------------------------------------------
# Step 2: SAM → binary mask
# ------------------------------------------------------------------
def segment_mask(rgb_path: Path, bbox: np.ndarray) -> np.ndarray:
    """Return a bool (H, W) mask via SAM box-prompt."""
    from segment_anything import SamPredictor, sam_model_registry  # type: ignore

    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=str(SAM_CHECKPOINT)).to("cuda")
    predictor = SamPredictor(sam)

    img_bgr = cv2.imread(str(rgb_path))
    predictor.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    masks, scores, _ = predictor.predict(
        point_coords=None, point_labels=None,
        box=bbox.astype(np.float32)[None, :], multimask_output=True,
    )
    best = masks[int(np.argmax(scores))].astype(bool)
    print(f"[SAM]  mask pixels={best.sum()}")
    return best


# ------------------------------------------------------------------
# Step 3: Depth loading → metres
# ------------------------------------------------------------------
def load_depth_m(depth_path: Path) -> np.ndarray:
    """Load a depth image as float32 in metres."""
    if depth_path.suffix.lower() == ".npy":
        depth_m = np.load(str(depth_path)).astype(np.float32)
        depth_m = np.flipud(depth_m).copy()
        print(f"[Depth] Loaded {depth_path.name} and flipped vertically")
        return depth_m

    raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot read: {depth_path}")

    print(f"[Depth] dtype={raw.dtype}  min={raw.min()}  max={raw.max()}")
    if raw.dtype == np.uint16:
        depth_m = raw.astype(np.float32) / 1000.0   # mm → m
        depth_m = np.flipud(depth_m).copy()
        print(f"[Depth] Loaded {depth_path.name} and flipped vertically")
        return depth_m
    if raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[2] >= 3:
        # Unity shader OutputMode 6 packs 16-bit millimeter depth into R/G.
        # OpenCV loads PNG as BGR(A), so use channel 2 as high byte and 1 as low byte.
        high_byte = raw[..., 2].astype(np.uint16)
        low_byte = raw[..., 1].astype(np.uint16)
        depth_mm = (high_byte << 8) | low_byte
        depth_m = depth_mm.astype(np.float32) / 1000.0
        depth_m = np.flipud(depth_m).copy()
        print(f"[Depth] Decoded Unity OutputMode 6 from {depth_path.name} and flipped vertically")
        return depth_m
    if raw.ndim == 3:
        raise ValueError(
            f"Unsupported depth image format {raw.shape} {raw.dtype}. "
            "Expected uint16 PNG, float32 .npy, or Unity OutputMode 6 RGB PNG."
        )
    # uint8 single-channel fallback for legacy captures.
    print("[Depth] WARNING: uint8 single-channel depth detected; assuming 0-255 = 0-5 m")
    depth_m = raw.astype(np.float32) / 255.0 * 5.0
    depth_m = np.flipud(depth_m).copy()
    print(f"[Depth] Loaded {depth_path.name} and flipped vertically")
    return depth_m


# ------------------------------------------------------------------
# Step 4: Point cloud from depth + mask (no renderer dependency)
# ------------------------------------------------------------------
def depth_mask_to_pointclouds(
    depth_m: np.ndarray,
    mask_obj: np.ndarray,
    rgb: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Project depth → 3-D XYZ and split into object / scene clouds.
    Avoids GraspGen's renderer import (which pulls webdataset etc.).
    """
    H, W = depth_m.shape
    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    xs, ys = np.meshgrid(xs, ys)

    Z = depth_m
    valid = (Z > 0.01) & (Z < 10.0)          # keep 1 cm – 10 m range

    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy
    XYZ = np.stack([X, Y, Z], axis=-1)       # H×W×3

    all_pts  = XYZ[valid]                    # N×3
    all_cols = rgb.reshape(-1, 3)[valid.ravel()] if rgb is not None else None

    obj_mask_flat  = mask_obj[valid]
    obj_pts   = all_pts[obj_mask_flat]
    obj_cols  = all_cols[obj_mask_flat] if all_cols is not None else None
    scene_pts = all_pts[~obj_mask_flat]
    scene_cols = all_cols[~obj_mask_flat] if all_cols is not None else None

    print(f"[PC]   object={len(obj_pts)}  scene={len(scene_pts)}")
    return scene_pts, obj_pts, scene_cols, obj_cols


# ------------------------------------------------------------------
# Step 5: GraspGen inference
# ------------------------------------------------------------------
def run_grasp_inference(
    object_pc: np.ndarray,
    object_colors: np.ndarray = None,
    num_grasps: int = 200,
    topk: int = 100,
):
    import trimesh.transformations as tra  # type: ignore
    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg  # type: ignore
    from grasp_gen.utils.point_cloud_utils import (  # type: ignore
        point_cloud_outlier_removal,
        point_cloud_outlier_removal_with_color,
    )
    from grasp_gen.utils.meshcat_utils import get_color_from_score  # type: ignore

    # Outlier removal
    filtered_colors = None
    if object_colors is not None:
        pc_t, removed_t, color_t, _ = point_cloud_outlier_removal_with_color(
            torch.from_numpy(object_pc), torch.from_numpy(object_colors)
        )
        filtered_colors = color_t.numpy()
    else:
        pc_t, removed_t = point_cloud_outlier_removal(torch.from_numpy(object_pc))
    pc_filtered = pc_t.numpy()
    print(f"[Filter] kept={len(pc_filtered)}  removed={len(removed_t)}")
    if len(pc_filtered) == 0:
        raise RuntimeError("Object PC empty after outlier removal.")

    # GraspGen
    cfg     = load_grasp_cfg(str(GRIPPER_CONFIG))
    sampler = GraspGenSampler(cfg)
    grasps_t, conf_t = GraspGenSampler.run_inference(
        pc_filtered, sampler,
        grasp_threshold=-1.0, num_grasps=num_grasps, topk_num_grasps=topk,
    )
    if len(grasps_t) == 0:
        raise RuntimeError("GraspGen returned no grasps.")

    grasps = grasps_t.cpu().numpy()
    conf   = conf_t.cpu().numpy()
    grasps[:, 3, 3] = 1.0
    print(f"[Grasp] {len(grasps)} grasps  score=[{conf.min():.3f}, {conf.max():.3f}]")

    # Centre both point cloud and grasps
    T_sub = tra.translation_matrix(-pc_filtered.mean(axis=0))
    pc_c  = tra.transform_points(pc_filtered, T_sub)
    grasps_c = np.array([T_sub @ g for g in grasps])
    scores = get_color_from_score(conf, use_255_scale=True)

    return pc_c, filtered_colors, grasps_c, conf, scores, T_sub, cfg


# ------------------------------------------------------------------
# Step 6: Collision filtering
# ------------------------------------------------------------------
def filter_collisions(
    scene_pc: np.ndarray,
    grasps_c: np.ndarray,
    T_center: np.ndarray,
    cfg,
    collision_threshold: float = 0.02,
    max_scene_pts: int = 8192,
) -> Tuple[np.ndarray, np.ndarray]:
    import trimesh.transformations as tra  # type: ignore
    from grasp_gen.utils.point_cloud_utils import filter_colliding_grasps  # type: ignore
    from grasp_gen.robot import get_gripper_info  # type: ignore

    scene_c = tra.transform_points(scene_pc, T_center)

    # Downsample for speed
    if len(scene_c) > max_scene_pts:
        idx = np.random.choice(len(scene_c), max_scene_pts, replace=False)
        scene_ds = scene_c[idx]
        print(f"[Scene] {len(scene_c)} → {len(scene_ds)} pts (downsampled)")
    else:
        scene_ds = scene_c
        print(f"[Scene] {len(scene_c)} pts")

    gripper_info = get_gripper_info(cfg.data.gripper_name)
    coll_mesh    = gripper_info.collision_mesh
    print(f"[Gripper] {cfg.data.gripper_name}  mesh_verts={len(coll_mesh.vertices)}")

    t0 = time.time()
    mask = filter_colliding_grasps(
        scene_pc=scene_ds,
        grasp_poses=grasps_c,
        gripper_collision_mesh=coll_mesh,
        collision_threshold=collision_threshold,
    )
    print(f"[Collision] {mask.sum()}/{len(grasps_c)} free  ({time.time()-t0:.1f}s)")
    return mask, scene_c


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="GraspGen + Collision Filtering")
    p.add_argument("--class-name",       default="doll")
    p.add_argument("--num-grasps",       type=int,   default=200)
    p.add_argument("--topk",             type=int,   default=100)
    p.add_argument("--collision-thresh", type=float, default=0.02)
    return p.parse_args()


def main():
    args      = parse_args()
    t0_global = time.time()

    print("=" * 60)
    print("GraspGen Inference + Collision Filtering")
    print("=" * 60)

    # 1. Camera
    fx, fy, cx, cy = load_intrinsics(INTRINSICS_YAML)
    print(f"[Camera] fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")

    # 2. YOLO
    bbox = detect_bbox(RGB_PATH, args.class_name)

    # 3. SAM
    mask = segment_mask(RGB_PATH, bbox)

    # 4. Depth
    depth_m = load_depth_m(DEPTH_PATH)
    rgb_bgr = cv2.imread(str(RGB_PATH))
    rgb     = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    # Resize depth if needed
    if depth_m.shape[:2] != rgb.shape[:2]:
        depth_m = cv2.resize(depth_m, (rgb.shape[1], rgb.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        mask    = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        print(f"[Depth] Resized depth+mask to {depth_m.shape}")

    valid_depth = depth_m[depth_m > 0]
    valid_px = len(valid_depth)
    if valid_px == 0:
        raise RuntimeError("Depth image has no valid pixels after loading/flipping.")
    print(f"[Depth] Valid px={valid_px}  "
          f"range=[{valid_depth.min():.3f}, {valid_depth.max():.3f}] m")

    # 5. Point clouds (self-contained, no renderer dependency)
    scene_pc, object_pc, scene_colors, object_colors = depth_mask_to_pointclouds(
        depth_m, mask, rgb, fx, fy, cx, cy
    )

    if len(object_pc) == 0:
        raise RuntimeError("Object point cloud is empty – check YOLO/SAM/depth alignment.")

    # 6. Grasp inference
    pc_c, obj_colors_c, grasps_c, conf, scores, T_center, cfg = run_grasp_inference(
        object_pc, object_colors, num_grasps=args.num_grasps, topk=args.topk
    )
    import trimesh.transformations as tra  # type: ignore
    object_pc_raw_c = tra.transform_points(object_pc, T_center)

    # 7. Collision filtering
    coll_mask, scene_c = filter_collisions(
        scene_pc, grasps_c, T_center, cfg,
        collision_threshold=args.collision_thresh,
    )

    free_grasps = grasps_c[coll_mask]
    free_conf   = conf[coll_mask]

    print("\n" + "=" * 60)
    print(f"RESULT: {len(free_grasps)} collision-free / {len(grasps_c)} total grasps")
    print(f"Total time: {time.time()-t0_global:.1f}s")
    print("=" * 60)

    # 8. Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_data = dict(
        all_grasps=grasps_c,
        all_scores=conf,
        collision_free_mask=coll_mask,
        collision_free_grasps=free_grasps,
        collision_free_scores=free_conf,
        pc_object=pc_c,
        pc_object_raw=object_pc_raw_c,
        pc_scene=scene_c,
        camera_intrinsics=np.array([fx, fy, cx, cy]),
    )
    if obj_colors_c is not None:
        save_data["pc_object_colors"] = obj_colors_c
    if object_colors is not None:
        save_data["pc_object_raw_colors"] = object_colors
    if scene_colors is not None:
        save_data["pc_scene_colors"] = scene_colors

    np.savez(str(OUTPUT_PATH), **save_data)
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
