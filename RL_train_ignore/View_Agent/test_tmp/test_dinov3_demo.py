import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms import functional as TF


DEFAULT_REPO_DIR = Path(__file__).resolve().parent / "dinov3"
DEFAULT_YOLO_WEIGHTS = Path(__file__).resolve().parents[3] / "3090server" / "VLM_RL" / "models" / "yolo" / "pure720.pt"
DEFAULT_DINOV3_WEIGHTS = Path(__file__).resolve().parent / "checkpoints" / "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DEFAULT_SAM_WEIGHTS = Path(__file__).resolve().parents[3] / "3090server" / "VLM_RL" / "models" / "segmentation" / "sam_vit_b_01ec64.pth"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINOv3 two-frame bbox tracking demo.")
    parser.add_argument("--frame1", type=Path, required=True, help="Path to frame 1.")
    parser.add_argument("--frame2", type=Path, required=True, help="Path to frame 2.")
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR, help="Local dinov3 repo path.")
    parser.add_argument("--arch", type=str, default="dinov3_vitb16", help="Backbone entrypoint from hubconf.py.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_DINOV3_WEIGHTS,
        help="Checkpoint path or URL for the DINOv3 backbone.",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=DEFAULT_YOLO_WEIGHTS,
        help="YOLO checkpoint used to detect the target object on frame 1.",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default="doll",
        help="YOLO class name used for the initial frame-1 detection.",
    )
    parser.add_argument(
        "--sam-weights",
        type=Path,
        default=DEFAULT_SAM_WEIGHTS,
        help="SAM checkpoint used to refine the frame-1 YOLO box into a mask.",
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="vit_b",
        help="SAM model type registered in segment_anything.",
    )
    parser.add_argument(
        "--use-sam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SAM to refine the frame-1 YOLO box into a mask before DINOv3 matching.",
    )
    parser.add_argument(
        "--short-side",
        type=int,
        default=448,
        help="Resize each frame so the short side matches this value.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cpu or cuda:0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "pics" / "dinov3_tracking_demo.png",
        help="Where to save the visualization.",
    )
    return parser.parse_args()


def load_model(repo_dir: Path, arch: str, weights: Path | None, device: torch.device) -> torch.nn.Module:
    repo_dir = repo_dir.expanduser().resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"dinov3 repo not found: {repo_dir}")

    weights_arg = str(weights.expanduser().resolve()) if weights is not None else None
    sys.path.insert(0, str(repo_dir))
    try:
        backbones = importlib.import_module("dinov3.hub.backbones")
        factory = getattr(backbones, arch)
        if weights_arg is not None:
            model = factory(pretrained=True, weights=weights_arg)
        else:
            model = factory(pretrained=False)
    finally:
        if sys.path and sys.path[0] == str(repo_dir):
            sys.path.pop(0)

    model.eval()
    return model.to(device)


def detect_best_bbox(image_path: Path, weights_path: Path, class_name: str) -> tuple[Box, float]:
    from ultralytics import YOLO  # type: ignore

    weights_path = weights_path.expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    model = YOLO(str(weights_path))
    results = model(str(image_path), verbose=False)
    if not results:
        raise RuntimeError(f"No YOLO result for image: {image_path}")

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        raise ValueError(f"No YOLO boxes predicted for image: {image_path}")

    names = {int(k): v for k, v in result.names.items()}
    target = class_name.lower()
    best_box = None
    best_conf = -1.0
    for box in result.boxes:
        cls_id = int(box.cls.item())
        label = str(names.get(cls_id, cls_id)).lower()
        if label != target:
            continue
        conf = float(box.conf.item())
        if conf > best_conf:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            best_box = Box(x1, y1, x2, y2)
            best_conf = conf

    if best_box is None:
        available = sorted({str(v).lower() for v in names.values()})
        raise ValueError(f"No class '{class_name}' detection in {image_path}. Available classes: {available}")

    return best_box, best_conf


def segment_with_sam(image: Image.Image, box: Box, model_type: str, checkpoint: Path, device: str) -> tuple[np.ndarray, float]:
    from segment_anything import SamPredictor, sam_model_registry  # type: ignore

    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
    if model_type not in sam_model_registry:
        raise ValueError(f"Unsupported SAM model type: {model_type}")

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint)).to(device=device)
    predictor = SamPredictor(sam)
    image_np = np.asarray(image.convert("RGB"))
    predictor.set_image(image_np)
    box_np = np.array([box.x1, box.y1, box.x2, box.y2], dtype=np.float32)
    masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=box_np[None, :], multimask_output=True)
    if masks is None or len(masks) == 0:
        raise RuntimeError("SAM returned no masks for the YOLO bbox prompt.")
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool), float(scores[best_idx])


def resize_to_patch_multiple(image: Image.Image, short_side: int, patch_size: int) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("invalid image size")

    if width < height:
        new_width = round_up(short_side, patch_size)
        new_height = round_up(height * new_width / width, patch_size)
    else:
        new_height = round_up(short_side, patch_size)
        new_width = round_up(width * new_height / height, patch_size)
    return image.resize((new_width, new_height), Image.Resampling.BICUBIC)


def round_up(value: float, multiple: int) -> int:
    return int(np.ceil(value / multiple) * multiple)


def extract_patch_features(model: torch.nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        feats = model.get_intermediate_layers(image_tensor, n=1, reshape=True)[0]
    feats = feats[0].permute(1, 2, 0).contiguous()
    return F.normalize(feats, dim=-1, p=2)


def scale_box(box: Box, src_size: tuple[int, int], dst_size: tuple[int, int]) -> Box:
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    scale_x = dst_w / src_w
    scale_y = dst_h / src_h
    return Box(box.x1 * scale_x, box.y1 * scale_y, box.x2 * scale_x, box.y2 * scale_y)


def clamp_box(box: Box, width: int, height: int) -> Box:
    x1 = min(max(box.x1, 0.0), width - 1.0)
    y1 = min(max(box.y1, 0.0), height - 1.0)
    x2 = min(max(box.x2, x1 + 1.0), float(width))
    y2 = min(max(box.y2, y1 + 1.0), float(height))
    return Box(x1, y1, x2, y2)


def image_box_to_patch_box(box: Box, patch_size: int, feat_w: int, feat_h: int) -> tuple[int, int, int, int]:
    x1 = int(np.floor(box.x1 / patch_size))
    y1 = int(np.floor(box.y1 / patch_size))
    x2 = int(np.ceil(box.x2 / patch_size))
    y2 = int(np.ceil(box.y2 / patch_size))
    x1 = int(np.clip(x1, 0, feat_w - 1))
    y1 = int(np.clip(y1, 0, feat_h - 1))
    x2 = int(np.clip(max(x2, x1 + 1), 1, feat_w))
    y2 = int(np.clip(max(y2, y1 + 1), 1, feat_h))
    return x1, y1, x2, y2


def patch_box_to_image_box(x1: int, y1: int, x2: int, y2: int, patch_size: int) -> Box:
    return Box(x1 * patch_size, y1 * patch_size, x2 * patch_size, y2 * patch_size)


def box_to_query_embedding(features: torch.Tensor, patch_box: tuple[int, int, int, int]) -> torch.Tensor:
    x1, y1, x2, y2 = patch_box
    target = features[y1:y2, x1:x2]
    if target.numel() == 0:
        raise ValueError("bbox does not overlap any feature patches")
    query = target.mean(dim=(0, 1))
    return F.normalize(query, dim=0, p=2)


def image_mask_to_patch_mask(mask: np.ndarray, feat_h: int, feat_w: int) -> torch.Tensor:
    mask_tensor = torch.from_numpy(mask.astype(np.float32))[None, None, :, :]
    patch_mask = F.interpolate(mask_tensor, size=(feat_h, feat_w), mode="nearest")[0, 0] > 0.5
    return patch_mask


def mask_to_query_embedding(features: torch.Tensor, patch_mask: torch.Tensor, fallback_patch_box: tuple[int, int, int, int]) -> torch.Tensor:
    selected = features[patch_mask]
    if selected.numel() == 0:
        return box_to_query_embedding(features, fallback_patch_box)
    query = selected.mean(dim=0)
    return F.normalize(query, dim=0, p=2)


def patch_mask_extent(patch_mask: torch.Tensor, fallback_patch_box: tuple[int, int, int, int]) -> tuple[int, int]:
    ys, xs = torch.where(patch_mask)
    if ys.numel() == 0:
        return max(1, fallback_patch_box[2] - fallback_patch_box[0]), max(1, fallback_patch_box[3] - fallback_patch_box[1])
    width = int(xs.max().item() - xs.min().item() + 1)
    height = int(ys.max().item() - ys.min().item() + 1)
    return max(1, width), max(1, height)


def similarity_map(features: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    sim = torch.einsum("hwd,d->hw", features, query)
    sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
    return sim


def localize_bbox_from_similarity(sim_map: torch.Tensor, patch_box_size: tuple[int, int]) -> tuple[int, int, int, int]:
    box_w, box_h = patch_box_size
    sim = sim_map.unsqueeze(0).unsqueeze(0)
    pooled = F.avg_pool2d(sim, kernel_size=(box_h, box_w), stride=1)
    flat_idx = pooled.view(-1).argmax().item()
    out_w = pooled.shape[-1]
    top = flat_idx // out_w
    left = flat_idx % out_w
    return left, top, left + box_w, top + box_h


def localize_bbox_from_peak(
    sim_map: torch.Tensor,
    patch_box_size: tuple[int, int],
    peak_threshold: float = 0.72,
) -> tuple[int, int, int, int]:
    box_w, box_h = patch_box_size
    feat_h, feat_w = sim_map.shape

    peak_idx = int(sim_map.view(-1).argmax().item())
    peak_y = peak_idx // feat_w
    peak_x = peak_idx % feat_w
    peak_value = float(sim_map[peak_y, peak_x].item())

    active = sim_map >= (peak_value * peak_threshold)
    if not bool(active[peak_y, peak_x]):
        active[peak_y, peak_x] = True

    ys, xs = torch.where(active)
    if ys.numel() == 0:
        cx = float(peak_x)
        cy = float(peak_y)
    else:
        weights = sim_map[ys, xs]
        weight_sum = float(weights.sum().item())
        if weight_sum <= 1e-8:
            cx = float(xs.float().mean().item())
            cy = float(ys.float().mean().item())
        else:
            cx = float((xs.float() * weights).sum().item() / weight_sum)
            cy = float((ys.float() * weights).sum().item() / weight_sum)

    left = int(round(cx - (box_w - 1) / 2.0))
    top = int(round(cy - (box_h - 1) / 2.0))
    left = int(np.clip(left, 0, max(0, feat_w - box_w)))
    top = int(np.clip(top, 0, max(0, feat_h - box_h)))
    return left, top, left + box_w, top + box_h


def localize_adaptive_bbox_from_peak(
    sim_map: torch.Tensor,
    fallback_patch_box: tuple[int, int, int, int],
    min_scale: float = 0.75,
    max_scale: float = 1.8,
    peak_threshold: float = 0.72,
) -> tuple[int, int, int, int]:
    feat_h, feat_w = sim_map.shape
    base_w = max(1, fallback_patch_box[2] - fallback_patch_box[0])
    base_h = max(1, fallback_patch_box[3] - fallback_patch_box[1])

    peak_idx = int(sim_map.view(-1).argmax().item())
    peak_y = peak_idx // feat_w
    peak_x = peak_idx % feat_w
    peak_value = float(sim_map[peak_y, peak_x].item())

    active = sim_map >= (peak_value * peak_threshold)
    if not bool(active[peak_y, peak_x]):
        active[peak_y, peak_x] = True

    ys, xs = torch.where(active)
    if ys.numel() == 0:
        return localize_bbox_from_peak(sim_map, (base_w, base_h), peak_threshold=peak_threshold)

    weights = sim_map[ys, xs]
    weight_sum = float(weights.sum().item())
    if weight_sum <= 1e-8:
        cx = float(xs.float().mean().item())
        cy = float(ys.float().mean().item())
    else:
        cx = float((xs.float() * weights).sum().item() / weight_sum)
        cy = float((ys.float() * weights).sum().item() / weight_sum)

    extent_w = int(xs.max().item() - xs.min().item() + 1)
    extent_h = int(ys.max().item() - ys.min().item() + 1)
    box_w = int(np.clip(extent_w, max(1, int(round(base_w * min_scale))), min(feat_w, int(round(base_w * max_scale)))))
    box_h = int(np.clip(extent_h, max(1, int(round(base_h * min_scale))), min(feat_h, int(round(base_h * max_scale)))))

    left = int(round(cx - (box_w - 1) / 2.0))
    top = int(round(cy - (box_h - 1) / 2.0))
    left = int(np.clip(left, 0, max(0, feat_w - box_w)))
    top = int(np.clip(top, 0, max(0, feat_h - box_h)))
    return left, top, left + box_w, top + box_h


def draw_box(image: Image.Image, box: Box, color: tuple[int, int, int], width: int = 4) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=color, width=width)
    return canvas


def make_heatmap_overlay(image: Image.Image, sim_map: torch.Tensor) -> Image.Image:
    sim_np = sim_map.detach().cpu().numpy().astype(np.float32)
    lo = float(np.percentile(sim_np, 5))
    hi = float(np.percentile(sim_np, 99))
    sim_np = np.clip((sim_np - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    sim_np = np.power(sim_np, 0.6)

    heat = Image.fromarray((sim_np * 255).astype(np.uint8), mode="L").resize(image.size, Image.Resampling.BILINEAR)
    heat_np = np.asarray(heat, dtype=np.float32) / 255.0

    overlay = np.zeros((heat_np.shape[0], heat_np.shape[1], 3), dtype=np.float32)
    overlay[..., 0] = np.clip((heat_np - 0.25) / 0.75, 0.0, 1.0)
    overlay[..., 1] = np.clip(1.0 - np.abs(heat_np - 0.55) / 0.28, 0.0, 1.0)
    overlay[..., 2] = np.clip((0.45 - heat_np) / 0.45, 0.0, 1.0)

    base = np.asarray(image, dtype=np.float32) / 255.0
    alpha = np.clip(heat_np[..., None] * 0.85, 0.15, 0.85)
    blended = (1.0 - alpha) * base + alpha * overlay
    blended = (blended * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def make_mask_overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int] = (0, 255, 0)) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = 0.55 * overlay[mask_bool] + 0.45 * np.array(color, dtype=np.float32)
    return Image.fromarray(overlay.clip(0, 255).astype(np.uint8))


def make_triptych(frame1: Image.Image, frame2_pred: Image.Image, heatmap: Image.Image) -> Image.Image:
    width = frame1.width + frame2_pred.width + heatmap.width
    height = max(frame1.height, frame2_pred.height, heatmap.height)
    canvas = Image.new("RGB", (width, height), color=(18, 18, 18))
    canvas.paste(frame1, (0, 0))
    canvas.paste(frame2_pred, (frame1.width, 0))
    canvas.paste(heatmap, (frame1.width + frame2_pred.width, 0))
    return canvas


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    input_box, input_conf = detect_best_bbox(args.frame1, args.yolo_weights, args.target_class)
    model = load_model(args.repo_dir, args.arch, args.weights, device)
    patch_size = int(model.patch_size)

    frame1_original = Image.open(args.frame1).convert("RGB")
    frame2_original = Image.open(args.frame2).convert("RGB")
    frame1_mask = None
    sam_score = None
    if args.use_sam:
        frame1_mask, sam_score = segment_with_sam(
            frame1_original,
            input_box,
            args.sam_model_type,
            args.sam_weights,
            args.device,
        )
    frame1_resized = resize_to_patch_multiple(frame1_original, args.short_side, patch_size)
    frame2_resized = resize_to_patch_multiple(frame2_original, args.short_side, patch_size)

    frame1_tensor = TF.normalize(TF.to_tensor(frame1_resized), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)
    frame2_tensor = TF.normalize(TF.to_tensor(frame2_resized), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)

    feats1 = extract_patch_features(model, frame1_tensor)
    feats2 = extract_patch_features(model, frame2_tensor)

    frame1_box_resized = clamp_box(scale_box(input_box, frame1_original.size, frame1_resized.size), *frame1_resized.size)
    feat_h, feat_w = feats1.shape[:2]
    patch_box1 = image_box_to_patch_box(frame1_box_resized, patch_size, feat_w, feat_h)
    if frame1_mask is not None:
        frame1_mask_resized = np.array(
            Image.fromarray(frame1_mask.astype(np.uint8) * 255, mode="L").resize(frame1_resized.size, Image.Resampling.NEAREST)
        ) > 127
        patch_mask1 = image_mask_to_patch_mask(frame1_mask_resized, feat_h, feat_w)
        query = mask_to_query_embedding(feats1, patch_mask1, patch_box1)
    else:
        patch_mask1 = None
        query = box_to_query_embedding(feats1, patch_box1)
    sim_map = similarity_map(feats2, query)

    patch_box2 = localize_adaptive_bbox_from_peak(sim_map, patch_box1)
    frame2_box_resized = clamp_box(patch_box_to_image_box(*patch_box2, patch_size), *frame2_resized.size)
    frame2_box_original = clamp_box(scale_box(frame2_box_resized, frame2_resized.size, frame2_original.size), *frame2_original.size)

    if frame1_mask is not None:
        frame1_vis = draw_box(make_mask_overlay(frame1_original, frame1_mask), input_box, (0, 255, 0))
    else:
        frame1_vis = draw_box(frame1_original, input_box, (0, 255, 0))
    frame2_vis = draw_box(frame2_original, frame2_box_original, (255, 128, 0))
    heatmap = draw_box(make_heatmap_overlay(frame2_original, sim_map), frame2_box_original, (255, 255, 255))
    triptych = make_triptych(frame1_vis, frame2_vis, heatmap)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    triptych.save(args.output)

    print(f"device: {device}")
    print(f"arch: {args.arch}")
    print(f"weights: {args.weights.expanduser().resolve() if args.weights is not None else 'random_init'}")
    print(f"yolo weights: {args.yolo_weights.expanduser().resolve()}")
    print(f"use_sam: {args.use_sam}")
    if args.use_sam:
        print(f"sam weights: {args.sam_weights.expanduser().resolve()}")
        print(f"sam model type: {args.sam_model_type}")
    print(f"target class: {args.target_class}")
    print(f"frame1 YOLO conf: {input_conf:.4f}")
    if sam_score is not None:
        print(f"frame1 SAM score: {sam_score:.4f}")
    print(f"frame1 size: {frame1_original.size}")
    print(f"frame2 size: {frame2_original.size}")
    print(f"patch size: {patch_size}")
    print("frame1 bbox (YOLO):", [round(input_box.x1, 2), round(input_box.y1, 2), round(input_box.x2, 2), round(input_box.y2, 2)])
    print(
        "frame2 bbox (pred):",
        [
            round(frame2_box_original.x1, 2),
            round(frame2_box_original.y1, 2),
            round(frame2_box_original.x2, 2),
            round(frame2_box_original.y2, 2),
        ],
    )
    print(f"visualization saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
