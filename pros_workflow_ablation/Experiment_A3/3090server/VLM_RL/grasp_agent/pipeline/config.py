from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import yaml

from tool.grasp.graspgen import resolve_graspgen_runtime_root


REQUIRED_SECTIONS = ("camera", "models", "runtime")


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_runtime_config(path: Path) -> tuple[dict[str, Any], Path]:
    cfg_path = path.expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Grasp config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Grasp config must be a mapping: {cfg_path}")

    for section in REQUIRED_SECTIONS:
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"Missing section '{section}' in grasp config: {cfg_path}")

    camera = cfg["camera"]
    models = cfg["models"]
    runtime = cfg["runtime"]

    for key in ("camera_name", "intrinsics_path"):
        if key not in camera:
            raise ValueError(f"Missing key 'camera.{key}' in grasp config.")
    for key in ("yolo_weights", "sam_seg_checkpoint", "graspgen_root", "gripper_config"):
        if key not in models:
            raise ValueError(f"Missing key 'models.{key}' in grasp config.")
    for key in (
        "device",
        "sam_model_type",
        "depth_scale",
        "yolo_conf_threshold",
        "min_depth_m",
        "max_depth_m",
        "object_point_stride",
        "scene_point_stride",
        "object_voxel_size_m",
        "scene_voxel_size_m",
        "grasp_threshold",
        "num_grasps",
        "topk_num_grasps",
        "collision_threshold",
        "max_collision_scene_points",
        "num_collision_samples",
    ):
        if key not in runtime:
            raise ValueError(f"Missing key 'runtime.{key}' in grasp config.")

    base_dir = cfg_path.parent
    env_overrides = {
        ("models", "yolo_weights"): os.getenv("GRASP_YOLO_WEIGHTS"),
        ("models", "sam_seg_checkpoint"): os.getenv("GRASP_SAM_CHECKPOINT"),
        ("models", "graspgen_root"): os.getenv("GRASP_GRASPGEN_ROOT"),
        ("models", "gripper_config"): os.getenv("GRASP_GRIPPER_CONFIG"),
        ("camera", "intrinsics_path"): os.getenv("GRASP_CAMERA_INTRINSICS"),
    }
    for (section, key), override in env_overrides.items():
        if override:
            cfg[section][key] = override

    for section, key in (
        ("camera", "intrinsics_path"),
        ("models", "yolo_weights"),
        ("models", "sam_seg_checkpoint"),
        ("models", "graspgen_root"),
        ("models", "gripper_config"),
    ):
        cfg[section][key] = resolve_path(cfg[section][key], base_dir)

    cfg["models"]["graspgen_root"] = resolve_graspgen_runtime_root(cfg["models"]["graspgen_root"])

    return cfg, cfg_path


def validate_required_paths(cfg: dict[str, Any]) -> None:
    checks = [
        ("camera.intrinsics_path", cfg["camera"]["intrinsics_path"]),
        ("models.yolo_weights", cfg["models"]["yolo_weights"]),
        ("models.sam_seg_checkpoint", cfg["models"]["sam_seg_checkpoint"]),
        ("models.graspgen_root", cfg["models"]["graspgen_root"]),
        ("models.gripper_config", cfg["models"]["gripper_config"]),
    ]
    for name, path in checks:
        if not Path(path).exists():
            raise FileNotFoundError(f"Required path missing: {name} -> {path}")


def validate_runtime_device(cfg: dict[str, Any]) -> str:
    device = str(cfg["runtime"]["device"]).lower()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for grasp inference but is not available.")
    return device
