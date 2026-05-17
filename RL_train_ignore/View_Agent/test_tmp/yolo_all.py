import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


DEFAULT_IMAGE_DIR = Path("/home/tjchen/workspace/VLM_RL/RL_train/View_Agent/test_tmp/pics/Camera_Room1")
DEFAULT_WEIGHTS = Path("/home/tjchen/workspace/VLM_RL/3090server/VLM_RL/models/yolo/pure720.pt")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "pics" / "yolo_pure720_all_frames"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pure720.pt YOLO inference on all Camera_Room1 frames.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="Directory containing Camera_Room1_*_rgb.png frames.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="YOLO checkpoint path.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save annotated frames and detections JSON.")
    return parser.parse_args()


def load_image_paths(image_dir: Path) -> list[Path]:
    image_paths = sorted(image_dir.glob("Camera_Room1_*_rgb.png"))
    if not image_paths:
        raise FileNotFoundError(f"No Camera_Room1_*_rgb.png files found in {image_dir}")
    return image_paths


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    colors = [
        (255, 99, 71),
        (65, 105, 225),
        (60, 179, 113),
        (255, 165, 0),
        (186, 85, 211),
        (0, 206, 209),
    ]

    for idx, det in enumerate(detections):
        color = colors[idx % len(colors)]
        box = det["bbox_xyxy"]
        x1, y1, x2, y2 = box
        label = f'{det["label"]} {det["confidence"]:.2f}'
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, max(0.0, y1 - 14.0)), label, fill=color)

    return canvas


def main() -> None:
    args = parse_args()

    from ultralytics import YOLO  # type: ignore

    weights_path = args.weights.expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    image_paths = load_image_paths(args.image_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    all_detections: dict[str, list[dict]] = {}

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        results = model(image, conf=args.conf, verbose=False)
        if not results:
            raise RuntimeError(f"No YOLO result for image: {image_path}")

        result = results[0]
        frame_detections: list[dict] = []
        if result.boxes is not None:
            names = {int(k): v for k, v in result.names.items()}
            for box in result.boxes:
                cls_id = int(box.cls.item())
                label = str(names.get(cls_id, cls_id))
                conf = float(box.conf.item())
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                frame_detections.append(
                    {
                        "label": label,
                        "confidence": conf,
                        "bbox_xyxy": [x1, y1, x2, y2],
                    }
                )

        annotated = draw_detections(image, frame_detections)
        annotated.save(args.output_dir / f"{image_path.stem}_yolo.png")
        all_detections[image_path.name] = frame_detections
        print(f"{image_path.name}: {len(frame_detections)} detections")

    detections_path = args.output_dir / "detections.json"
    detections_path.write_text(json.dumps(all_detections, indent=2))
    print(f"Annotated frames saved to: {args.output_dir}")
    print(f"Detections JSON saved to: {detections_path}")


if __name__ == "__main__":
    main()
