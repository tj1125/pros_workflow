import argparse
import functools
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms as TVT
from torchvision.transforms import functional as TVTF

from test_dinov3_demo import (
    DEFAULT_DINOV3_WEIGHTS,
    DEFAULT_REPO_DIR,
    DEFAULT_SAM_WEIGHTS,
    DEFAULT_YOLO_WEIGHTS,
    Box,
    box_to_query_embedding,
    clamp_box,
    image_box_to_patch_box,
    load_model,
    localize_adaptive_bbox_from_peak,
    patch_box_to_image_box,
    scale_box,
    similarity_map,
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_IMAGE_DIR = Path("/home/tjchen/workspace/VLM_RL/RL_train/View_Agent/test_tmp/pics/Camera_Room1")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "pics" / "dinov3_segtrack_notebook_pipeline.png"


def forward(
    model: nn.Module,
    img: torch.Tensor,  # [3, H, W] already normalized for the model
) -> torch.Tensor:
    with torch.inference_mode():
        feats = model.get_intermediate_layers(img.unsqueeze(0), n=1, reshape=True)[0]  # [1, D, h, w]
    feats = feats.movedim(-3, -1)  # [1, h, w, D]
    feats = F.normalize(feats, dim=-1, p=2)
    return feats.squeeze(0)  # [h, w, D]


def mask_to_rgb(mask: np.ndarray | torch.Tensor, num_masks: int) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    mask = mask.astype(np.int64, copy=False)
    background = mask == 0
    palette = np.array(
        [
            [31, 119, 180],
            [255, 127, 14],
            [44, 160, 44],
            [214, 39, 40],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [127, 127, 127],
            [188, 189, 34],
            [23, 190, 207],
            [174, 199, 232],
            [255, 187, 120],
            [152, 223, 138],
            [255, 152, 150],
            [197, 176, 213],
            [196, 156, 148],
            [247, 182, 210],
            [199, 199, 199],
            [219, 219, 141],
            [158, 218, 229],
        ],
        dtype=np.uint8,
    )

    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    if num_masks <= 1:
        return rgb

    for idx in range(1, num_masks):
        rgb[mask == idx] = palette[(idx - 1) % len(palette)]
    rgb[background] = 0
    return rgb


def mask_id_to_color(mask_id: int) -> tuple[int, int, int]:
    palette = mask_to_rgb(np.array([[mask_id]], dtype=np.int32), max(mask_id + 1, 2))
    return tuple(int(v) for v in palette[0, 0])


class SAMRefiner:
    def __init__(self, model_type: str, checkpoint: Path, device: str):
        from segment_anything import SamPredictor, sam_model_registry  # type: ignore

        checkpoint = checkpoint.expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        if model_type not in sam_model_registry:
            raise ValueError(f"Unsupported SAM model type: {model_type}")

        sam = sam_model_registry[model_type](checkpoint=str(checkpoint)).to(device=device)
        self.predictor = SamPredictor(sam)

    def set_image(self, image: Image.Image) -> None:
        self.predictor.set_image(np.asarray(image.convert("RGB")))

    def predict_box(self, box: Box) -> tuple[np.ndarray, float]:
        box_np = np.array([box.x1, box.y1, box.x2, box.y2], dtype=np.float32)
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_np[None, :],
            multimask_output=True,
        )
        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM returned no masks for the box prompt.")
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(bool), float(scores[best_idx])


class ResizeToMultiple(nn.Module):
    def __init__(self, short_side: int, multiple: int):
        super().__init__()
        self.short_side = short_side
        self.multiple = multiple

    def _round_up(self, side: float) -> int:
        return math.ceil(side / self.multiple) * self.multiple

    def forward(self, img: Image.Image) -> torch.Tensor:
        old_width, old_height = TVTF.get_image_size(img)
        if old_width > old_height:
            new_height = self._round_up(self.short_side)
            new_width = self._round_up(old_width * new_height / old_height)
        else:
            new_width = self._round_up(self.short_side)
            new_height = self._round_up(old_height * new_width / old_width)
        return TVTF.resize(img, [new_height, new_width], interpolation=TVT.InterpolationMode.BICUBIC)


def maybe_mark_dynamic(tensor: torch.Tensor, dims: int | tuple[int, ...]) -> None:
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and hasattr(dynamo, "maybe_mark_dynamic"):
        dynamo.maybe_mark_dynamic(tensor, dims)


def propagate(
    current_features: torch.Tensor,  # [h", w", D]
    context_features: torch.Tensor,  # [t, h, w, D]
    context_probs: torch.Tensor,  # [t, h, w, M]
    neighborhood_mask: torch.Tensor,  # [h", w", h, w]
    topk: int,
    temperature: float,
) -> torch.Tensor:
    _, h, w, _ = context_probs.shape

    dot = torch.einsum(
        "ijd, tuvd -> ijtuv",
        current_features,
        context_features,
    )
    dot = torch.where(
        neighborhood_mask[:, :, None, :, :],
        dot,
        -torch.inf,
    )

    dot = dot.flatten(2, -1).flatten(0, 1)
    k = min(topk, dot.shape[1])
    k_th_largest = torch.topk(dot, dim=1, k=k).values
    dot = torch.where(dot >= k_th_largest[:, -1:], dot, -torch.inf)

    weights = F.softmax(dot / temperature, dim=1)
    current_probs = torch.mm(weights, context_probs.flatten(0, 2))
    current_probs = current_probs / current_probs.sum(dim=1, keepdim=True)
    return current_probs.unflatten(0, (h, w))


@functools.lru_cache()
def make_neighborhood_mask(h: int, w: int, size: float, shape: str, device: str) -> torch.Tensor:
    ij = torch.stack(
        torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=device),
            torch.arange(w, dtype=torch.float32, device=device),
            indexing="ij",
        ),
        dim=-1,
    )
    if shape == "circle":
        ord_value = 2
    elif shape == "square":
        ord_value = torch.inf
    else:
        raise ValueError(f"Invalid {shape=}")

    norm = torch.linalg.vector_norm(
        ij[:, :, None, None, :] - ij[None, None, :, :, :],
        ord=ord_value,
        dim=-1,
    )
    return norm <= size


def postprocess_probs(
    probs: torch.Tensor,  # [B, M, H, W]
) -> torch.Tensor:
    vmin = probs.flatten(2, 3).min(dim=2).values
    vmax = probs.flatten(2, 3).max(dim=2).values
    probs = (probs - vmin[:, :, None, None]) / (vmax[:, :, None, None] - vmin[:, :, None, None])
    return torch.nan_to_num(probs, nan=0.0)


def label_mask_to_patch_probs(
    label_mask: np.ndarray,
    feat_h: int,
    feat_w: int,
    num_masks: int,
    device: torch.device,
) -> torch.Tensor:
    mask_tensor = torch.from_numpy(label_mask).to(device=device, dtype=torch.long)
    patch_mask = F.interpolate(
        mask_tensor[None, None, :, :].float(),
        (feat_h, feat_w),
        mode="nearest-exact",
    )[0, 0].long()
    return F.one_hot(patch_mask, num_masks).float()


def compress_mask_labels(label_mask: np.ndarray) -> np.ndarray:
    compact = np.zeros_like(label_mask, dtype=np.int32)
    next_id = 1
    for old_id in sorted(int(v) for v in np.unique(label_mask) if int(v) != 0):
        compact[label_mask == old_id] = next_id
        next_id += 1
    return compact


def translate_box(box: Box | None, dx: int, dy: int) -> Box | None:
    if box is None:
        return None
    return Box(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy)


def crop_image(image: Image.Image, crop_box: Box) -> Image.Image:
    return image.crop((int(crop_box.x1), int(crop_box.y1), int(crop_box.x2), int(crop_box.y2)))


def crop_mask(label_mask: np.ndarray, crop_box: Box) -> np.ndarray:
    return label_mask[int(crop_box.y1):int(crop_box.y2), int(crop_box.x1):int(crop_box.x2)]


def paste_mask(label_mask: np.ndarray, crop_box: Box, frame_height: int, frame_width: int) -> np.ndarray:
    full_mask = np.zeros((frame_height, frame_width), dtype=np.int32)
    full_mask[int(crop_box.y1):int(crop_box.y2), int(crop_box.x1):int(crop_box.x2)] = label_mask
    return full_mask


def center_crop_box_from_anchor(anchor_box: Box, crop_width: int, crop_height: int, image_width: int, image_height: int) -> Box:
    cx = (anchor_box.x1 + anchor_box.x2) / 2.0
    cy = (anchor_box.y1 + anchor_box.y2) / 2.0
    left = int(round(cx - crop_width / 2.0))
    top = int(round(cy - crop_height / 2.0))
    left = int(np.clip(left, 0, max(0, image_width - crop_width)))
    top = int(np.clip(top, 0, max(0, image_height - crop_height)))
    return Box(float(left), float(top), float(left + crop_width), float(top + crop_height))


def box_from_binary_mask(mask: np.ndarray, pad: int, width: int, height: int) -> Box | None:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None

    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(width, int(xs.max()) + 1 + pad)
    y2 = min(height, int(ys.max()) + 1 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return Box(float(x1), float(y1), float(x2), float(y2))


def refine_mask_with_sam(
    image: Image.Image,
    coarse_mask: np.ndarray,
    num_masks: int,
    sam_refiner: SAMRefiner,
    box_pad: int,
) -> tuple[np.ndarray, list[Box | None], list[float]]:
    refined_mask = np.zeros_like(coarse_mask, dtype=np.int32)
    object_boxes: list[Box | None] = [None] * num_masks
    object_scores: list[float] = [0.0] * num_masks
    occupied = np.zeros_like(coarse_mask, dtype=bool)

    sam_refiner.set_image(image)
    for obj_id in range(1, num_masks):
        coarse_obj_mask = coarse_mask == obj_id
        bbox = box_from_binary_mask(coarse_obj_mask, pad=box_pad, width=image.width, height=image.height)
        if bbox is None:
            continue

        try:
            sam_mask, sam_score = sam_refiner.predict_box(bbox)
        except RuntimeError:
            sam_mask = coarse_obj_mask
            sam_score = 0.0

        sam_mask = np.asarray(sam_mask, dtype=bool)
        if not np.any(sam_mask):
            sam_mask = coarse_obj_mask
        sam_mask &= ~occupied
        if not np.any(sam_mask):
            continue

        refined_mask[sam_mask] = obj_id
        occupied |= sam_mask
        object_boxes[obj_id] = box_from_binary_mask(sam_mask, pad=0, width=image.width, height=image.height)
        object_scores[obj_id] = sam_score

    return refined_mask, object_boxes, object_scores


def detect_all_bboxes(
    image: Image.Image,
    weights_path: Path,
    conf_threshold: float = 0.25,
    upscale_factor: float = 1.0,
    min_box_size: int = 5,
    allowed_labels: set[str] | None = None,
) -> list[tuple[Box, str, float]]:
    from ultralytics import YOLO  # type: ignore

    weights_path = weights_path.expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    run_image = image
    if upscale_factor > 1.0:
        run_image = image.resize(
            (int(image.width * upscale_factor), int(image.height * upscale_factor)),
            Image.Resampling.LANCZOS,
        )

    model = YOLO(str(weights_path))
    results = model(run_image, conf=conf_threshold, verbose=False)
    if not results:
        raise RuntimeError("No YOLO result returned for the first frame.")

    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    names = {int(k): v for k, v in result.names.items()}
    bboxes: list[tuple[Box, str, float]] = []
    for box in result.boxes:
        cls_id = int(box.cls.item())
        label = str(names.get(cls_id, cls_id))
        if allowed_labels is not None and label.lower() not in allowed_labels:
            continue

        conf = float(box.conf.item())
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        if upscale_factor > 1.0:
            x1 /= upscale_factor
            y1 /= upscale_factor
            x2 /= upscale_factor
            y2 /= upscale_factor

        bbox = Box(x1, y1, x2, y2)
        if bbox.width < min_box_size or bbox.height < min_box_size:
            continue
        bboxes.append((bbox, label, conf))

    bboxes.sort(key=lambda item: item[2], reverse=True)
    return bboxes


def detect_anchor_bbox(
    image: Image.Image,
    weights_path: Path,
    anchor_label: str,
    conf_threshold: float,
    upscale_factor: float,
    min_box_size: int,
) -> tuple[Box, str, float]:
    detections = detect_all_bboxes(
        image,
        weights_path,
        conf_threshold=conf_threshold,
        upscale_factor=upscale_factor,
        min_box_size=min_box_size,
        allowed_labels={anchor_label.lower()},
    )
    if not detections:
        raise ValueError(f"YOLO found no '{anchor_label}' box in the first frame.")
    return detections[0]


def render_init_preview(
    image: Image.Image,
    init_mask: np.ndarray,
    detections: list[tuple[Box, str, float]],
    output_path: Path,
) -> None:
    overlay = Image.fromarray(mask_to_rgb(init_mask, int(init_mask.max()) + 1)).convert("RGBA")
    alpha = Image.fromarray(np.where(init_mask > 0, 120, 0).astype(np.uint8), mode="L")
    overlay.putalpha(alpha)

    preview = image.convert("RGBA")
    preview.alpha_composite(overlay)

    draw = ImageDraw.Draw(preview)
    for obj_idx, (bbox, label, conf) in enumerate(detections, start=1):
        color = mask_id_to_color(obj_idx)
        draw.rectangle([bbox.x1, bbox.y1, bbox.x2, bbox.y2], outline=color, width=3)
        draw.text((bbox.x1, max(0.0, bbox.y1 - 14.0)), f"{obj_idx}:{label} {conf:.2f}", fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.convert("RGB").save(output_path)


def build_first_mask_with_yolo_sam(
    first_frame: Image.Image,
    yolo_weights: Path,
    yolo_conf: float,
    yolo_upscale: float,
    min_box_size: int,
    max_objects: int,
    sam_refiner: SAMRefiner,
    yolo_classes: set[str] | None,
) -> tuple[np.ndarray, list[tuple[Box, str, float]]]:
    detections = detect_all_bboxes(
        first_frame,
        yolo_weights,
        conf_threshold=yolo_conf,
        upscale_factor=yolo_upscale,
        min_box_size=min_box_size,
        allowed_labels=yolo_classes,
    )
    if not detections:
        raise ValueError("YOLO found no usable boxes for the first frame.")

    detections = detections[:max_objects]
    init_mask = np.zeros((first_frame.height, first_frame.width), dtype=np.int32)
    kept_detections: list[tuple[Box, str, float]] = []
    next_id = 1

    sam_refiner.set_image(first_frame)
    for bbox, label, conf in detections:
        sam_mask, sam_score = sam_refiner.predict_box(bbox)
        obj_mask = np.asarray(sam_mask, dtype=bool)
        obj_mask &= init_mask == 0
        if int(obj_mask.sum()) == 0:
            continue

        refined_bbox = box_from_binary_mask(obj_mask, pad=0, width=first_frame.width, height=first_frame.height)
        if refined_bbox is None:
            continue

        init_mask[obj_mask] = next_id
        kept_detections.append((refined_bbox, f"{label}|sam:{sam_score:.2f}", conf))
        next_id += 1

    if next_id == 1:
        raise ValueError("SAM produced no usable first-frame masks from the YOLO detections.")

    return init_mask, kept_detections


def infer_first_mask_path(image_dir: Path) -> Path | None:
    candidates = [
        image_dir / "Camera_Room1_1_mask.png",
        image_dir / "Camera_Room1_1_seg.png",
        image_dir / "first_mask.png",
        image_dir / "mask.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_mask_labels(mask_path: Path, frame_size: tuple[int, int]) -> np.ndarray:
    mask_img = Image.open(mask_path)
    frame_width, frame_height = frame_size

    if mask_img.size == (frame_width, frame_height):
        pass
    elif mask_img.height == frame_height and mask_img.width % frame_width == 0:
        # Some local outputs concatenate masks horizontally. Use the first panel as the first-frame mask.
        mask_img = mask_img.crop((0, 0, frame_width, frame_height))
    else:
        raise ValueError(
            f"Mask size {mask_img.size} does not match frame size {(frame_width, frame_height)}."
        )

    if mask_img.mode in {"L", "I", "I;16"}:
        return np.array(mask_img, dtype=np.int32)

    if mask_img.mode == "P":
        return np.array(mask_img, dtype=np.int32)

    mask_rgb = np.array(mask_img.convert("RGB"), dtype=np.uint8)
    flat = mask_rgb.reshape(-1, 3)
    unique_colors, inverse = np.unique(flat, axis=0, return_inverse=True)
    non_black = int(np.count_nonzero(np.any(unique_colors != 0, axis=1)))

    if non_black > 256:
        raise ValueError(
            f"Mask image contains {non_black} non-background colors after cropping to one frame. "
            "This looks like a visualization/overlay, not a label mask. "
            "Please provide a single-frame segmentation mask where each object uses one solid color or one integer label."
        )

    color_ids = np.zeros(len(unique_colors), dtype=np.int32)
    next_id = 1
    for idx, color in enumerate(unique_colors):
        if np.all(color == 0):
            color_ids[idx] = 0
        else:
            color_ids[idx] = next_id
            next_id += 1
    return color_ids[inverse].reshape(mask_rgb.shape[:2])


def load_frames(image_dir: Path) -> list[Image.Image]:
    image_files = sorted(image_dir.glob("Camera_Room1_*_rgb.png"))
    if not image_files:
        raise FileNotFoundError(f"No Camera_Room1_*_rgb.png files found in {image_dir}")
    return [Image.open(path).convert("RGB") for path in image_files]


def make_visualization(
    frames: list[Image.Image],
    mask_predictions: torch.Tensor,
    num_masks: int,
    object_boxes: list[list[Box | None]] | None = None,
    crop_boxes: list[Box] | None = None,
) -> Image.Image:
    frame_width, frame_height = frames[0].size
    canvas = Image.new("RGB", (frame_width * len(frames), frame_height), color=(255, 255, 255))

    for idx, frame in enumerate(frames):
        pred_mask = mask_predictions[idx].cpu().numpy()
        overlay = Image.fromarray(mask_to_rgb(pred_mask, num_masks)).convert("RGBA")
        alpha = Image.fromarray(np.where(pred_mask > 0, 120, 0).astype(np.uint8), mode="L")
        overlay.putalpha(alpha)

        frame_rgba = frame.convert("RGBA")
        frame_rgba.alpha_composite(overlay)

        draw = ImageDraw.Draw(frame_rgba)
        if crop_boxes is not None:
            crop_box = crop_boxes[idx]
            draw.rectangle([crop_box.x1, crop_box.y1, crop_box.x2, crop_box.y2], outline=(255, 255, 255), width=2)

        for obj_idx in range(1, num_masks):
            color = mask_id_to_color(obj_idx)
            bbox = None if object_boxes is None else object_boxes[idx][obj_idx]
            if bbox is None:
                ys, xs = np.where(pred_mask == obj_idx)
                if ys.size == 0:
                    continue
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
            else:
                x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, max(0, y1 - 12)), str(obj_idx), fill=color)

        canvas.paste(frame_rgba.convert("RGB"), (idx * frame_width, 0))

    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DINOv3 segmentation tracking notebook pipeline on Camera_Room1 frames.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Directory containing Camera_Room1_*_rgb.png frames.")
    parser.add_argument("--first-mask", type=Path, default=None, help="Optional override for the first-frame segmentation mask. If omitted, the script builds one using YOLO + SAM.")
    parser.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR, help="Local dinov3 repo path.")
    parser.add_argument("--arch", type=str, default="dinov3_vitb16", help="Backbone entrypoint from dinov3.hub.backbones.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_DINOV3_WEIGHTS, help="DINOv3 checkpoint path.")
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS, help="YOLO checkpoint used to detect the first-frame boxes.")
    parser.add_argument("--yolo-conf", type=float, default=0.5, help="YOLO confidence threshold for the first-frame detections.")
    parser.add_argument("--yolo-upscale", type=float, default=1.0, help="Optional image upscale factor before YOLO. Increase this for very small objects.")
    parser.add_argument("--yolo-classes", type=str, default=None, help="Optional comma-separated YOLO class whitelist for the first frame, for example 'doll,bottle'.")
    parser.add_argument("--min-box-size", type=int, default=5, help="Drop YOLO boxes smaller than this many pixels.")
    parser.add_argument("--max-objects", type=int, default=10, help="Maximum number of YOLO+SAM objects kept in the initial mask.")
    parser.add_argument("--sam-weights", type=Path, default=DEFAULT_SAM_WEIGHTS, help="SAM checkpoint used to refine the first-frame YOLO boxes.")
    parser.add_argument("--sam-model-type", type=str, default="vit_b", help="SAM model type.")
    parser.add_argument("--sam-box-pad", type=int, default=8, help="Padding in pixels added around each tracked coarse mask before SAM refinement.")
    parser.add_argument("--anchor-label", type=str, default="doll", help="YOLO class name used to center the crop on each frame.")
    parser.add_argument("--crop-width", type=int, default=500, help="Crop width around the tracked anchor object.")
    parser.add_argument("--crop-height", type=int, default=300, help="Crop height around the tracked anchor object.")
    parser.add_argument("--short-side", type=int, default=960, help="Resize short side before DINOv3 feature extraction. Notebook default is 960.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device, for example cpu or cuda:0.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Visualization output path.")
    parser.add_argument("--neighborhood-size", type=float, default=12.0, help="Neighborhood size for propagation.")
    parser.add_argument("--neighborhood-shape", type=str, default="circle", choices=["circle", "square"], help="Neighborhood shape.")
    parser.add_argument("--topk", type=int, default=5, help="Top-k similar patches for propagation.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for propagation softmax.")
    parser.add_argument("--max-context", type=int, default=5, help="Maximum number of historical frames kept in the context queue.")
    parser.add_argument("--save-mask-npy", type=Path, default=None, help="Optional path to save raw predicted label masks as a .npy file.")
    parser.add_argument("--save-prob-npy", type=Path, default=None, help="Optional path to save propagated per-class probabilities as a .npy file. This can be large.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    frames = load_frames(args.image_dir)
    num_frames = len(frames)
    original_width, original_height = frames[0].size
    print(f"Loaded {num_frames} frames, original size: {original_width}x{original_height}")
    if args.crop_width > original_width or args.crop_height > original_height:
        raise ValueError("Crop size must fit inside the original frame size.")

    sam_refiner = SAMRefiner(args.sam_model_type, args.sam_weights, args.device)

    first_frame_pil = frames[0]
    anchor_bbox, anchor_label, anchor_conf = detect_anchor_bbox(
        first_frame_pil,
        args.yolo_weights,
        args.anchor_label,
        args.yolo_conf,
        args.yolo_upscale,
        args.min_box_size,
    )
    crop_boxes: list[Box] = [
        center_crop_box_from_anchor(anchor_bbox, args.crop_width, args.crop_height, original_width, original_height)
    ]
    print(f"Anchor object: {anchor_label} {anchor_conf:.2f}, initial crop: {crop_boxes[0]}")
    first_crop_frame = crop_image(first_frame_pil, crop_boxes[0])

    yolo_classes = None
    if args.yolo_classes:
        yolo_classes = {item.strip().lower() for item in args.yolo_classes.split(",") if item.strip()}

    first_mask_path = args.first_mask
    init_detections: list[tuple[Box, str, float]] = []
    if first_mask_path is not None:
        if not first_mask_path.exists():
            raise FileNotFoundError(f"First-frame mask not found: {first_mask_path}")
        full_first_mask = load_mask_labels(first_mask_path, (original_width, original_height))
        first_crop_mask = compress_mask_labels(crop_mask(full_first_mask, crop_boxes[0]))
        print(f"Using provided first mask: {first_mask_path}")
    else:
        first_crop_mask, init_detections = build_first_mask_with_yolo_sam(
            first_crop_frame,
            args.yolo_weights,
            args.yolo_conf,
            args.yolo_upscale,
            args.min_box_size,
            args.max_objects,
            sam_refiner,
            yolo_classes,
        )
        init_preview_path = args.output.with_name(f"{args.output.stem}_init_crop_mask.png")
        render_init_preview(first_crop_frame, first_crop_mask, init_detections, init_preview_path)
        print(f"Generated first-frame crop mask with YOLO+SAM. Preview saved to: {init_preview_path}")

    model = load_model(args.repo_dir, args.arch, args.weights, device)
    patch_size = int(model.patch_size)

    transform = TVT.Compose(
        [
            ResizeToMultiple(short_side=args.short_side, multiple=patch_size),
            TVT.ToTensor(),
            TVT.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    full_first_frame = transform(first_frame_pil).to(device)
    maybe_mark_dynamic(full_first_frame, (1, 2))
    full_first_feats = forward(model, full_first_frame)
    full_resized_height, full_resized_width = full_first_frame.shape[1:]
    full_feat_h, full_feat_w = full_first_feats.shape[:2]
    scaled_anchor_bbox = scale_box(anchor_bbox, first_frame_pil.size, (full_resized_width, full_resized_height))
    scaled_anchor_bbox = clamp_box(scaled_anchor_bbox, full_resized_width, full_resized_height)
    anchor_patch_box = image_box_to_patch_box(scaled_anchor_bbox, patch_size, full_feat_w, full_feat_h)
    anchor_query = box_to_query_embedding(full_first_feats, anchor_patch_box)

    crop_mask_height, crop_mask_width = first_crop_mask.shape
    num_masks = int(first_crop_mask.max() + 1)
    print(f"Crop mask size: {crop_mask_width}x{crop_mask_height}, num_masks: {num_masks}")

    first_crop_tensor = transform(first_crop_frame).to(device)
    maybe_mark_dynamic(first_crop_tensor, (1, 2))
    first_feats = forward(model, first_crop_tensor)
    crop_feat_h, crop_feat_w = first_feats.shape[:2]
    first_probs = label_mask_to_patch_probs(first_crop_mask, crop_feat_h, crop_feat_w, num_masks, device)

    mask_predictions = torch.zeros([num_frames, original_height, original_width], dtype=torch.uint8)
    mask_predictions[0] = torch.from_numpy(
        paste_mask(first_crop_mask, crop_boxes[0], original_height, original_width)
    ).to(dtype=torch.uint8)
    tracked_boxes: list[list[Box | None]] = [[None] * num_masks for _ in range(num_frames)]
    for obj_id in range(1, num_masks):
        local_box = box_from_binary_mask(first_crop_mask == obj_id, pad=0, width=crop_mask_width, height=crop_mask_height)
        tracked_boxes[0][obj_id] = translate_box(local_box, int(crop_boxes[0].x1), int(crop_boxes[0].y1))

    mask_probabilities = None
    if args.save_prob_npy is not None:
        mask_probabilities = torch.zeros([num_frames, num_masks, crop_mask_height, crop_mask_width], dtype=torch.float32)
        mask_probabilities[0] = F.one_hot(torch.from_numpy(first_crop_mask).long(), num_masks).movedim(-1, -3).float()

    features_queue: list[torch.Tensor] = []
    probs_queue: list[torch.Tensor] = []

    neighborhood_mask = make_neighborhood_mask(
        crop_feat_h,
        crop_feat_w,
        size=args.neighborhood_size,
        shape=args.neighborhood_shape,
        device=str(device),
    )

    start = time.perf_counter()
    for frame_idx in range(1, num_frames):
        current_full_frame = transform(frames[frame_idx]).to(device)
        maybe_mark_dynamic(current_full_frame, (1, 2))
        current_full_feats = forward(model, current_full_frame)
        sim_map = similarity_map(current_full_feats, anchor_query)
        current_anchor_patch_box = localize_adaptive_bbox_from_peak(sim_map, anchor_patch_box)
        current_anchor_resized = patch_box_to_image_box(*current_anchor_patch_box, patch_size)
        current_anchor_box = scale_box(
            current_anchor_resized,
            (current_full_frame.shape[2], current_full_frame.shape[1]),
            frames[frame_idx].size,
        )
        current_anchor_box = clamp_box(current_anchor_box, original_width, original_height)
        current_crop_box = center_crop_box_from_anchor(
            current_anchor_box,
            args.crop_width,
            args.crop_height,
            original_width,
            original_height,
        )
        crop_boxes.append(current_crop_box)
        current_crop_frame = crop_image(frames[frame_idx], current_crop_box)
        current_crop_tensor = transform(current_crop_frame).to(device)
        maybe_mark_dynamic(current_crop_tensor, (1, 2))
        current_feats = forward(model, current_crop_tensor)

        context_feats = torch.stack([first_feats, *features_queue], dim=0)
        context_probs = torch.stack([first_probs, *probs_queue], dim=0)
        maybe_mark_dynamic(context_feats, 0)
        maybe_mark_dynamic(context_probs, (0, 3))

        current_probs = propagate(
            current_feats,
            context_feats,
            context_probs,
            neighborhood_mask,
            args.topk,
            args.temperature,
        )

        features_queue.append(current_feats)
        probs_queue.append(current_probs)
        if len(features_queue) > args.max_context:
            features_queue.pop(0)
        if len(probs_queue) > args.max_context:
            probs_queue.pop(0)

        current_probs = F.interpolate(
            current_probs.movedim(-1, -3)[None, :, :, :],
            size=(crop_mask_height, crop_mask_width),
            mode="nearest",
        )
        current_probs = postprocess_probs(current_probs).squeeze(0).detach().cpu()
        coarse_pred = torch.argmax(current_probs, dim=0).to(dtype=torch.uint8).numpy()
        refined_mask, refined_boxes, _ = refine_mask_with_sam(
            current_crop_frame,
            coarse_pred,
            num_masks,
            sam_refiner,
            args.sam_box_pad,
        )

        if not np.any(refined_mask):
            refined_mask = coarse_pred.astype(np.int32)
            for obj_id in range(1, num_masks):
                refined_boxes[obj_id] = box_from_binary_mask(
                    refined_mask == obj_id,
                    pad=0,
                    width=crop_mask_width,
                    height=crop_mask_height,
                )

        if mask_probabilities is not None:
            mask_probabilities[frame_idx] = current_probs
        full_refined_mask = paste_mask(refined_mask, current_crop_box, original_height, original_width)
        mask_predictions[frame_idx] = torch.from_numpy(full_refined_mask).to(dtype=torch.uint8)
        tracked_boxes[frame_idx] = [
            translate_box(box, int(current_crop_box.x1), int(current_crop_box.y1))
            for box in refined_boxes
        ]
        probs_queue[-1] = label_mask_to_patch_probs(refined_mask, crop_feat_h, crop_feat_w, num_masks, device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    print(f"Tracking finished in {elapsed:.2f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    vis = make_visualization(frames, mask_predictions, num_masks, object_boxes=tracked_boxes, crop_boxes=crop_boxes)
    vis.save(args.output)
    print(f"Visualization saved to: {args.output}")

    if args.save_mask_npy is not None:
        args.save_mask_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_mask_npy, mask_predictions.numpy())
        print(f"Raw mask predictions saved to: {args.save_mask_npy}")

    if args.save_prob_npy is not None:
        args.save_prob_npy.parent.mkdir(parents=True, exist_ok=True)
        if mask_probabilities is None:
            raise RuntimeError("Internal error: probability buffer was not initialized.")
        np.save(args.save_prob_npy, mask_probabilities.numpy())
        print(f"Raw mask probabilities saved to: {args.save_prob_npy}")


if __name__ == "__main__":
    main()
