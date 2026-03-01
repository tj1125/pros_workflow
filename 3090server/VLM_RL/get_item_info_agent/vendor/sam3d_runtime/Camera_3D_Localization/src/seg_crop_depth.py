#!/usr/bin/env python3
"""Generate depth maps for segmented RGB crops using Depth Anything V2."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
import cv2
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "result"
SEG_CROP_RGB_DIR = RESULTS_DIR / "seg_crop_rgb"
SEG_CROP_DEPTH_DIR = RESULTS_DIR / "seg_crop_depth"
SEG_MASK_DIR = RESULTS_DIR / "seg_mask"
YOLO_MASK_DIR = RESULTS_DIR / "yolo_mask"
RGB_IMAGE_DIR = PROJECT_ROOT / "rgb_image"
DEFAULT_MODEL = PROJECT_ROOT / "model" / "depth" / "depth_anything_v2_vitb.pth"

# Ensure local src package is importable when running as a script.
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def infer_encoder_name(model_path: Path) -> str:
    name = model_path.stem.lower()
    for key in MODEL_CONFIGS:
        if key in name:
            return key
    raise ValueError(f"Cannot infer encoder from model name: {model_path.name}")


def load_depth_model(model_path: Path, device: torch.device):
    try:
        from src.depth_anything_v2.dpt import DepthAnythingV2
    except ImportError as exc:  # pragma: no cover - runtime dependency
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError as exc2:
            raise RuntimeError(
                "depth_anything_v2 is not installed. Place the Depth-Anything-V2 "
                "code under Camera_3D_Localization/src/depth_anything_v2 or add the "
                "repo to PYTHONPATH before running depth inference."
            ) from exc2

    encoder = infer_encoder_name(model_path)
    config = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**config)

    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]

    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing or unexpected:
        print(f"Warning: missing keys={len(missing)}, unexpected keys={len(unexpected)}")

    model = model.to(device).eval()
    return model


def normalize_to_uint16(depth: np.ndarray) -> np.ndarray:
    mask = depth > 0
    if not np.any(mask):
        return np.zeros_like(depth, dtype=np.uint16)
    vals = depth[mask]
    depth_min = float(vals.min())
    depth_max = float(vals.max())
    denom = max(depth_max - depth_min, 1e-6)
    depth_norm = (depth - depth_min) / denom
    depth_norm[~mask] = 0.0
    depth_norm = 1.0 - depth_norm
    depth_norm[~mask] = 0.0
    return (depth_norm * 65535.0).astype(np.uint16)


def normalize_to_uint8(depth: np.ndarray) -> np.ndarray:
    mask = depth > 0
    if not np.any(mask):
        return np.zeros_like(depth, dtype=np.uint8)
    vals = depth[mask]
    depth_min = float(vals.min())
    depth_max = float(vals.max())
    denom = max(depth_max - depth_min, 1e-6)
    depth_norm = (depth - depth_min) / denom
    depth_norm[~mask] = 0.0
    depth_norm = 1.0 - depth_norm
    depth_norm[~mask] = 0.0
    return (depth_norm * 255.0).astype(np.uint8)


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.float32)
    mask = depth > 0
    if not np.any(mask):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    vals = depth[mask]
    depth_min = float(vals.min())
    depth_max = float(vals.max())
    denom = max(depth_max - depth_min, 1e-6)
    depth_norm = (depth - depth_min) / denom
    depth_norm[~mask] = 0.0
    depth_norm = 1.0 - depth_norm
    depth_norm[~mask] = 0.0
    depth_u8 = (depth_norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


def parse_camera_id(path: Path) -> Optional[str]:
    match = re.search(r"rgb_(\d+_\d+)", path.stem)
    if match:
        return match.group(1)
    return None


def infer_depth_image(model, image_bgr: np.ndarray, input_size: Optional[int]) -> np.ndarray:
    if input_size is None:
        return model.infer_image(image_bgr)
    try:
        return model.infer_image(image_bgr, input_size=input_size)
    except TypeError:
        return model.infer_image(image_bgr)


def resolve_rgb_from_mask(mask_path: Path, rgb_dir: Path) -> Path:
    stem = mask_path.stem.replace("_seg_mask", "")
    candidates = sorted(rgb_dir.glob(f"{stem}.*"))
    if not candidates:
        raise FileNotFoundError(f"RGB image not found for mask {mask_path}")
    return candidates[0]


def resolve_mask_from_seg_crop(seg_crop_path: Path, mask_dir: Path) -> Path:
    stem = seg_crop_path.stem.replace("_seg_crop_rgb", "")
    candidates = sorted(mask_dir.glob(f"{stem}_seg_mask.*"))
    if not candidates:
        raise FileNotFoundError(f"Seg mask not found for seg crop {seg_crop_path}")
    return candidates[0]


def resolve_yolo_mask_from_seg_crop(seg_crop_path: Path, mask_dir: Path) -> Optional[Path]:
    stem = seg_crop_path.stem.replace("_seg_crop_rgb", "")
    candidates = sorted(mask_dir.glob(f"{stem}_mask.*"))
    if not candidates:
        return None
    return candidates[0]


def mask_and_crop(image_bgr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if image_bgr.shape[:2] != mask.shape[:2]:
        raise ValueError("Mask and RGB image have different sizes.")
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise ValueError("Empty mask; cannot crop.")
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    crop = image_bgr[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    crop[crop_mask == 0] = 0
    return crop, crop_mask


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        raise ValueError("Mask is empty.")
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    return x1, y1, x2, y2


def generate_depth_for_seg_crops(
    seg_crop_paths: Iterable[Path],
    output_dir: Path,
    model_path: Path = DEFAULT_MODEL,
    device: str = "cuda",
    input_size: Optional[int] = 518,
    depth_bits: int = 16,
    mask_paths: Optional[Iterable[Path]] = None,
    mask_dir: Path = SEG_MASK_DIR,
    yolo_mask_dir: Path = YOLO_MASK_DIR,
) -> List[Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device_t = torch.device(device)

    model_path = model_path.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Depth model not found: {model_path}")

    model = load_depth_model(model_path, device_t)

    seg_crop_paths = [Path(p) for p in seg_crop_paths]
    mask_paths_list = [Path(p) for p in mask_paths] if mask_paths is not None else None

    outputs = []
    for idx, seg_crop_path in enumerate(seg_crop_paths):
        seg_crop_path = seg_crop_path.expanduser().resolve()
        if not seg_crop_path.exists():
            raise FileNotFoundError(f"Seg crop image not found: {seg_crop_path}")

        image_bgr = cv2.imread(str(seg_crop_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {seg_crop_path}")

        if mask_paths_list is not None:
            mask_path = mask_paths_list[idx]
        else:
            mask_path = resolve_mask_from_seg_crop(seg_crop_path, mask_dir.expanduser().resolve())

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        x1, y1, x2, y2 = mask_bbox(mask)
        mask_crop = mask[y1:y2, x1:x2]
        if mask_crop.shape[:2] != image_bgr.shape[:2]:
            yolo_mask_path = resolve_yolo_mask_from_seg_crop(
                seg_crop_path, yolo_mask_dir.expanduser().resolve()
            )
            if yolo_mask_path is not None and yolo_mask_path.exists():
                yolo_mask = cv2.imread(str(yolo_mask_path), cv2.IMREAD_GRAYSCALE)
                if yolo_mask is None:
                    raise RuntimeError(f"Failed to read YOLO mask: {yolo_mask_path}")
                yx1, yy1, yx2, yy2 = mask_bbox(yolo_mask)
                mask_crop = mask[yy1:yy2, yx1:yx2]
                if mask_crop.shape[:2] == image_bgr.shape[:2]:
                    x1, y1, x2, y2 = yx1, yy1, yx2, yy2

        if mask_crop.shape[:2] != image_bgr.shape[:2]:
            target_h, target_w = image_bgr.shape[:2]
            full_h, full_w = mask.shape[:2]
            center_y = (y1 + y2) / 2.0
            center_x = (x1 + x2) / 2.0
            crop_y1 = int(round(center_y - target_h / 2.0))
            crop_x1 = int(round(center_x - target_w / 2.0))
            crop_y2 = crop_y1 + target_h
            crop_x2 = crop_x1 + target_w

            if crop_y1 < 0:
                crop_y2 -= crop_y1
                crop_y1 = 0
            if crop_x1 < 0:
                crop_x2 -= crop_x1
                crop_x1 = 0
            if crop_y2 > full_h:
                delta = crop_y2 - full_h
                crop_y1 = max(0, crop_y1 - delta)
                crop_y2 = full_h
            if crop_x2 > full_w:
                delta = crop_x2 - full_w
                crop_x1 = max(0, crop_x1 - delta)
                crop_x2 = full_w

            mask_crop = mask[crop_y1:crop_y2, crop_x1:crop_x2]
            if mask_crop.shape[:2] != image_bgr.shape[:2]:
                raise ValueError(
                    f"Seg crop {seg_crop_path} shape {image_bgr.shape[:2]} does not match "
                    f"mask crop {mask_crop.shape[:2]} from {mask_path}."
                )

        with torch.no_grad():
            depth = infer_depth_image(model, image_bgr, input_size)
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()

        depth[mask_crop == 0] = 0.0

        stem = seg_crop_path.stem.replace("_seg_crop_rgb", "")
        depth_png = output_dir / f"{stem}_seg_crop_depth.png"

        if depth_bits == 8:
            depth_vis = normalize_to_uint8(depth)
        else:
            depth_vis = normalize_to_uint16(depth)
        cv2.imwrite(str(depth_png), depth_vis)

        outputs.append(depth_png)

    return outputs


def generate_depth_for_seg_masks(
    mask_paths: Iterable[Path],
    output_dir: Path,
    model_path: Path = DEFAULT_MODEL,
    device: str = "cuda",
    input_size: Optional[int] = 518,
    depth_bits: int = 16,
    rgb_paths: Optional[Iterable[Path]] = None,
    rgb_dir: Path = RGB_IMAGE_DIR,
) -> List[Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device_t = torch.device(device)

    model_path = model_path.expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Depth model not found: {model_path}")

    model = load_depth_model(model_path, device_t)

    mask_paths = [Path(p) for p in mask_paths]
    rgb_paths_list = [Path(p) for p in rgb_paths] if rgb_paths is not None else None

    outputs = []
    for idx, mask_path in enumerate(mask_paths):
        mask_path = mask_path.expanduser().resolve()
        if not mask_path.exists():
            raise FileNotFoundError(f"Seg mask not found: {mask_path}")

        if rgb_paths_list is not None:
            rgb_path = rgb_paths_list[idx]
        else:
            rgb_path = resolve_rgb_from_mask(mask_path, rgb_dir.expanduser().resolve())

        image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {rgb_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        x1, y1, x2, y2 = mask_bbox(mask)

        with torch.no_grad():
            depth = infer_depth_image(model, image_bgr, input_size)
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()

        if depth.shape != mask.shape:
            depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]))

        depth[mask == 0] = 0.0
        crop_mask = mask[y1:y2, x1:x2]
        depth = depth[y1:y2, x1:x2]

        stem = mask_path.stem.replace("_seg_mask", "")
        depth_png = output_dir / f"{stem}_seg_crop_depth.png"

        if depth_bits == 8:
            depth_vis = normalize_to_uint8(depth)
        else:
            depth_vis = normalize_to_uint16(depth)
        cv2.imwrite(str(depth_png), depth_vis)

        outputs.append(depth_png)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate depth maps for seg-crop RGB images.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Depth model weights.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument("--input-size", type=int, default=518, help="Model input size.")
    parser.add_argument("--depth-bits", type=int, choices=[8, 16], default=8)
    parser.add_argument("--seg-crop-dir", type=Path, default=SEG_CROP_RGB_DIR)
    parser.add_argument("--mask-dir", type=Path, default=SEG_MASK_DIR)
    parser.add_argument("--rgb-dir", type=Path, default=RGB_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=SEG_CROP_DEPTH_DIR)
    parser.add_argument("--images", nargs="*", type=Path, default=None)
    parser.add_argument("--mask-paths", nargs="*", type=Path, default=None)
    parser.add_argument("--rgb-paths", nargs="*", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = []
    if args.images:
        seg_crop_paths = [Path(p) for p in args.images]
        outputs = generate_depth_for_seg_crops(
            seg_crop_paths,
            output_dir=args.output_dir,
            model_path=args.model,
            device=args.device,
            input_size=args.input_size,
            depth_bits=args.depth_bits,
        )
    else:
        if args.mask_paths:
            mask_paths = [Path(p) for p in args.mask_paths]
        else:
            mask_paths = sorted(args.mask_dir.glob("*_seg_mask.*"))
        if not mask_paths:
            raise FileNotFoundError(f"No seg masks in {args.mask_dir}")

        rgb_paths = [Path(p) for p in args.rgb_paths] if args.rgb_paths else None
        outputs = generate_depth_for_seg_masks(
            mask_paths,
            output_dir=args.output_dir,
            model_path=args.model,
            device=args.device,
            input_size=args.input_size,
            depth_bits=args.depth_bits,
            rgb_paths=rgb_paths,
            rgb_dir=args.rgb_dir,
        )
    print(f"Saved depth outputs ({len(outputs)}): {args.output_dir}")


if __name__ == "__main__":
    main()
