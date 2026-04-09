from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from get_item_info_agent_no_sam3d.pipeline.constants import AGENT_ROOT


REQUIRED_SECTIONS = ("camera", "models", "alignment", "map", "runtime")


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _require_keys(data: dict[str, Any], keys: list[str], section: str) -> None:
    for key in keys:
        if key not in data:
            raise ValueError(f"Missing key '{section}.{key}' in scene config.")


def load_scene_config(path: Path) -> tuple[dict[str, Any], Path]:
    cfg_path = path.expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Scene config not found: {cfg_path}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Scene config must be a mapping: {cfg_path}")

    for section in REQUIRED_SECTIONS:
        if section not in cfg or not isinstance(cfg[section], dict):
            raise ValueError(f"Missing section '{section}' in scene config: {cfg_path}")

    camera = cfg["camera"]
    models = cfg["models"]
    alignment = cfg["alignment"]
    map_cfg = cfg["map"]
    runtime = cfg["runtime"]

    _require_keys(camera, ["camera_parameter_dir"], "camera")
    _require_keys(
        models,
        [
            "sam_seg_checkpoint",
            "sam3d_root",
            "graspgen_root",
            "gripper_config",
        ],
        "models",
    )
    _require_keys(
        alignment,
        [
            "normal",
            "origin",
            "scan_y_step",
            "weight_shape",
            "rotate_z",
            "voxel",
            "width",
            "height",
            "padding",
        ],
        "alignment",
    )
    _require_keys(map_cfg, ["map_pgm", "map_yaml", "unity_map_origin", "offset", "white_threshold"], "map")
    _require_keys(
        runtime,
        [
            "device",
            "sam_model_type",
            "height_scale",
            "sam_seed",
            "grasp_threshold",
            "num_grasps",
            "topk_num_grasps",
            "num_sample_points",
        ],
        "runtime",
    )

    base_dir = cfg_path.parent
    path_fields = [
        ("camera", "camera_parameter_dir"),
        ("models", "sam_seg_checkpoint"),
        ("models", "sam3d_root"),
        ("models", "graspgen_root"),
        ("models", "gripper_config"),
        ("map", "map_pgm"),
        ("map", "map_yaml"),
    ]
    for section, key in path_fields:
        resolved = resolve_path(cfg[section][key], base_dir)
        cfg[section][key] = resolved

    return cfg, cfg_path


def validate_required_paths(cfg: dict[str, Any]) -> None:
    checks = [
        ("camera.camera_parameter_dir", cfg["camera"]["camera_parameter_dir"]),
        ("models.sam_seg_checkpoint", cfg["models"]["sam_seg_checkpoint"]),
        ("models.sam3d_root", cfg["models"]["sam3d_root"]),
        ("models.graspgen_root", cfg["models"]["graspgen_root"]),
        ("models.gripper_config", cfg["models"]["gripper_config"]),
        ("map.map_pgm", cfg["map"]["map_pgm"]),
        ("map.map_yaml", cfg["map"]["map_yaml"]),
    ]
    for name, path in checks:
        if not Path(path).exists():
            raise FileNotFoundError(f"Required path missing: {name} -> {path}")

    gripper_cfg = yaml.safe_load(Path(cfg["models"]["gripper_config"]).read_text(encoding="utf-8"))
    eval_ckpt = Path(cfg["models"]["gripper_config"]).parent / gripper_cfg["eval"]["checkpoint"]
    dis_ckpt = Path(cfg["models"]["gripper_config"]).parent / gripper_cfg["discriminator"]["checkpoint"]
    if not eval_ckpt.exists():
        raise FileNotFoundError(f"Missing GraspGen generator checkpoint: {eval_ckpt}")
    if not dis_ckpt.exists():
        raise FileNotFoundError(f"Missing GraspGen discriminator checkpoint: {dis_ckpt}")


def prepare_runtime_imports(cfg: dict[str, Any]) -> None:
    sam_root = Path(cfg["models"]["sam3d_root"]).resolve()
    graspgen_root = Path(cfg["models"]["graspgen_root"]).resolve()
    pointnet2_root = graspgen_root / "pointnet2_ops"
    camera_root = sam_root / "Camera_3D_Localization"
    camera_src = camera_root / "src"

    if "CONDA_PREFIX" not in os.environ:
        os.environ["CONDA_PREFIX"] = str(Path(sys.executable).resolve().parents[1])
    os.environ.setdefault("GRASPGEN_NO_VIS", "1")
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR",
        str(AGENT_ROOT / ".cache" / "torch_extensions"),
    )
    Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)

    for path in (sam_root, camera_root, camera_src, graspgen_root, pointnet2_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def validate_runtime_device(cfg: dict[str, Any]) -> str:
    device = str(cfg["runtime"]["device"]).lower()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full pipeline but is not available.")
    return device
