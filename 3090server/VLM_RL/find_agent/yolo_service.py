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
            "annotated_images": { "Camera1": "<base64>", "Camera2": "<base64>"}
        }
        """
        if not self._model:
            raise RuntimeError("YOLO model not initialized.")

        logger.info(f"[YoloService] Running detection on {len(camera_images)} images.")

        target_id    = target_object.get("id", "").lower()    if target_object else ""
        target_label = target_object.get("label", "")          if target_object else ""

        yolo_detections: Dict[str, Any] = {}
        annotated_images: Dict[str, str] = {}
        global_id = 1

        try:
            # 使用較細的字體DejaVuSans.ttf (預設不再加粗)
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 60
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
                drawn_text_boxes = []

                has_detection = False
                for box in results[0].boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    conf    = float(box.conf[0])
                    label   = self._model.names[int(box.cls[0])].lower()

                    # Filter by target object if specified
                    if target_id and label != target_id and label not in target_label:
                        continue

                    has_detection = True
                    logger.info(f"  - ID {global_id}: {label} ({conf:.2f}) at [{x1},{y1},{x2},{y2}]")

                    # Thin border (1px)
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=1)
                    
                    # Calculate text bounding box
                    text_str = str(global_id)
                    text_bbox = draw.textbbox((0, 0), text_str, font=font)
                    text_w = text_bbox[2] - text_bbox[0]
                    text_h = text_bbox[3] - text_bbox[1]

                    # Initial ideal text position (just above the bounding box)
                    tx = x1
                    ty = y1 - text_h - 4
                    if ty < 0:
                        ty = y2 + 4  # push DOWN BELOW the bbox if going off top edge

                    # Avoid overlapping with other text boxes by sliding horizontally
                    for _ in range(20):
                        overlap = False
                        for cx1, cy1, cx2, cy2 in drawn_text_boxes:
                            # Strict overlap formula
                            if not (tx + text_w < cx1 or tx > cx2 or ty + text_h < cy1 or ty > cy2):
                                overlap = True
                                break
                        if not overlap:
                            break
                        # Shift right
                        tx += text_w + 4

                    # Append to memory to avoid future overlaps
                    drawn_text_boxes.append((tx, ty, tx + text_w, ty + text_h))

                    # Draw edge mask/outline to make red text pop without full black rectangle
                    # This achieves visibility without an ugly black box
                    draw.text((tx-1, ty), text_str, fill="black", font=font)
                    draw.text((tx+1, ty), text_str, fill="black", font=font)
                    draw.text((tx, ty-1), text_str, fill="black", font=font)
                    draw.text((tx, ty+1), text_str, fill="black", font=font)
                    draw.text((tx, ty), text_str, fill="red", font=font)

                    yolo_detections[str(global_id)] = {
                        "camera": cam_name,
                        "bbox":   [x1, y1, x2, y2],
                        "label":  label,
                        "conf":   conf,
                    }
                    global_id += 1

                if has_detection:
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG", quality=85)
                    annotated_images[cam_name] = base64.b64encode(buf.getvalue()).decode()

            except Exception as e:
                logger.error(f"[YoloService] Error processing {cam_name}: {e}")

        return {
            "yolo_detections": yolo_detections,
            "annotated_images": annotated_images,
        }

