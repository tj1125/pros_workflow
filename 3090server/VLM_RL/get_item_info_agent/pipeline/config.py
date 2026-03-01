from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from pipeline.constants import GET_ITEM_INFO_ROOT


REQUIRED_SECTIONS = ("camera", "models", "alignment", "map", "runtime")


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    """Resolve a path relative to base_dir if not already absolute."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _require_keys(data: dict[str, Any], keys: list[str], section: str) -> None:
    """Raise if any required key is missing from a config section."""
    for key in keys:
        if key not in data:
            raise ValueError(f"Missing key '{section}.{key}' in scene config.")


def load_scene_config(path: Path) -> tuple[dict[str, Any], Path]:
    """Load and validate the scene YAML config file."""
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

    _require_keys(camera, ["camera_a", "camera_b", "camera_parameter_dir", "rgb_dir"], "camera")
    _require_keys(
        models,
        [
            "yolo_weights",
            "sam_seg_checkpoint",
            "depthanything_weights",
            "sam3d_root",
            "sam3d_config",
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
            "weight_depth",
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
            "depth_input_size",
            "base_ratio",
            "height_scale",
            "sam_seed",
            "grasp_threshold",
            "num_grasps",
            "topk_num_grasps",
            "num_sample_points",
        ],
        "runtime",
    )

    # Resolve all path fields relative to the scene config directory.
    base_dir = cfg_path.parent
    path_fields = [
        ("camera", "camera_parameter_dir"),
        ("camera", "rgb_dir"),
        ("models", "yolo_weights"),
        ("models", "sam_seg_checkpoint"),
        ("models", "depthanything_weights"),
        ("models", "sam3d_root"),
        ("models", "sam3d_config"),
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
    """Check that all model and data paths referenced in the config actually exist."""
    checks = [
        ("camera.camera_parameter_dir", cfg["camera"]["camera_parameter_dir"]),
        ("camera.rgb_dir", cfg["camera"]["rgb_dir"]),
        ("models.yolo_weights", cfg["models"]["yolo_weights"]),
        ("models.sam_seg_checkpoint", cfg["models"]["sam_seg_checkpoint"]),
        ("models.depthanything_weights", cfg["models"]["depthanything_weights"]),
        ("models.sam3d_root", cfg["models"]["sam3d_root"]),
        ("models.sam3d_config", cfg["models"]["sam3d_config"]),
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
    """Add the vendor/third-party roots from the original get_item_info project to sys.path."""
    sam_root = Path(cfg["models"]["sam3d_root"]).resolve()
    graspgen_root = Path(cfg["models"]["graspgen_root"]).resolve()
    pointnet2_root = graspgen_root / "pointnet2_ops"
    # pydeps lives inside the original project's vendor directory
    pydeps_root = GET_ITEM_INFO_ROOT / "vendor" / "pydeps"
    camera_root = sam_root / "Camera_3D_Localization"
    camera_src = camera_root / "src"

    if "CONDA_PREFIX" not in os.environ:
        os.environ["CONDA_PREFIX"] = str(Path(sys.executable).resolve().parents[1])
    os.environ.setdefault("GRASPGEN_NO_VIS", "1")
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR",
        str(GET_ITEM_INFO_ROOT / ".cache" / "torch_extensions"),
    )
    Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)

    for path in (pydeps_root, sam_root, camera_root, camera_src, graspgen_root, pointnet2_root):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def resolve_input_images(cfg: dict[str, Any], image_a: Path | None, image_b: Path | None) -> tuple[Path, Path]:
    """Resolve image paths; falls back to rgb_dir defaults if not provided.

    NOTE: Unlike the original CLI version, this function does NOT restrict paths
    to be under GET_ITEM_INFO_ROOT, so callers can pass temporary file paths (e.g. /tmp/*).
    """
    rgb_dir = Path(cfg["camera"]["rgb_dir"]).resolve()
    path_a = image_a.expanduser().resolve() if image_a else None
    path_b = image_b.expanduser().resolve() if image_b else None

    if path_a is None:
        matches = sorted(rgb_dir.glob("rgb_1_1*.png"))
        if not matches:
            raise FileNotFoundError(f"No rgb_1_1*.png in {rgb_dir}")
        path_a = matches[0]
    if path_b is None:
        matches = sorted(rgb_dir.glob("rgb_1_2*.png"))
        if not matches:
            raise FileNotFoundError(f"No rgb_1_2*.png in {rgb_dir}")
        path_b = matches[0]

    if not path_a.exists():
        raise FileNotFoundError(path_a)
    if not path_b.exists():
        raise FileNotFoundError(path_b)

    return path_a, path_b


def validate_runtime_device(cfg: dict[str, Any]) -> str:
    """Return the configured device string; raises if CUDA is requested but unavailable."""
    device = str(cfg["runtime"]["device"]).lower()
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full pipeline but is not available.")
    return device
