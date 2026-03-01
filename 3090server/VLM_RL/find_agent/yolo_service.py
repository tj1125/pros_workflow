"""
yolo_service.py — Core YOLO Detection Logic for FindAgent

Abstracts away PyTorch/Ultralytics inference and image annotation
from the A2A messaging layer.

Returns unformatted dict of globally-numbered detections.
"""

import base64
import io
import logging
from typing import Dict, Any

from PIL import Image, ImageDraw, ImageFont

from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = Path(__file__).parent.parent / "models" / "pure720.pt"


class YoloService:
    def __init__(self, model_path: str = str(_DEFAULT_MODEL)):
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            logger.info(f"[YoloService] Loaded model: {model_path}")
        except Exception as e:
            logger.error(f"[YoloService] Failed to load YOLO or weights not found: {e}")
            self._model = None

    def detect_and_annotate(
        self,
        camera_images: Dict[str, str],
        target_object: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Input:  { "Camera1": "<base64_img>", ... }
        Output: {
            "yolo_detections": { "1": {camera, bbox, label, conf}, ... },
            "composed_image_base64": "<single grid image with all boxes>",
        }
        """
        if not self._model:
            raise RuntimeError("YOLO model not initialized.")

        logger.info(f"[YoloService] Running detection on {len(camera_images)} images.")

        target_id    = target_object.get("id", "").lower()    if target_object else ""
        target_label = target_object.get("label", "")          if target_object else ""

        yolo_detections: Dict[str, Any] = {}
        annotated_frames: list = []  # one PIL image per camera
        global_id = 1

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60
            )
        except Exception:
            font = ImageFont.load_default()

        for cam_name, b64_str in camera_images.items():
            try:
                img_bytes = base64.b64decode(b64_str)
                pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                results  = self._model(pil_img, verbose=False)
                num_det  = len(results[0].boxes)
                logger.info(f"[YoloService] {cam_name}: Detected {num_det} objects.")

                draw = ImageDraw.Draw(pil_img)

                for box in results[0].boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    conf    = float(box.conf[0])
                    label   = self._model.names[int(box.cls[0])].lower()

                    # Filter by target object if specified
                    if target_id and label != target_id and label not in target_label:
                        continue

                    logger.info(f"  - ID {global_id}: {label} ({conf:.2f}) at [{x1},{y1},{x2},{y2}]")

                    # Thin border (1px), large number label (font 60px)
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
                    draw.text((x1 + 4, y1 + 4), str(global_id), fill="red", font=font)

                    yolo_detections[str(global_id)] = {
                        "camera": cam_name,
                        "bbox":   [x1, y1, x2, y2],
                        "label":  label,
                        "conf":   conf,
                    }
                    global_id += 1

                annotated_frames.append(pil_img)

            except Exception as e:
                logger.error(f"[YoloService] Error processing {cam_name}: {e}")

        # Compose all camera images into one horizontal mosaic
        composed_b64 = ""
        if annotated_frames:
            total_w = sum(f.width for f in annotated_frames)
            max_h   = max(f.height for f in annotated_frames)
            canvas  = Image.new("RGB", (total_w, max_h))
            x_off   = 0
            for frame in annotated_frames:
                canvas.paste(frame, (x_off, 0))
                x_off += frame.width
            buf = io.BytesIO()
            canvas.save(buf, format="JPEG", quality=85)
            composed_b64 = base64.b64encode(buf.getvalue()).decode()

        return {
            "yolo_detections":      yolo_detections,
            "composed_image_base64": composed_b64,
        }

