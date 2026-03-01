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
            "cropped_images": { "1": "<base64>", "2": "<base64>"}
        }
        """
        if not self._model:
            raise RuntimeError("YOLO model not initialized.")

        logger.info(f"[YoloService] Running detection on {len(camera_images)} images.")

        target_id    = target_object.get("id", "").lower()    if target_object else ""
        target_label = target_object.get("label", "")          if target_object else ""

        yolo_detections: Dict[str, Any] = {}
        cropped_images: Dict[str, str] = {}
        global_id = 1

        for cam_name, b64_str in camera_images.items():
            try:
                img_bytes = base64.b64decode(b64_str)
                pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                results  = self._model(pil_img, verbose=False)
                num_det  = len(results[0].boxes)
                logger.info(f"[YoloService] {cam_name}: Detected {num_det} objects.")

                for box in results[0].boxes:
                    conf    = float(box.conf[0])
                    
                    # 信心門檻 > 50%
                    if conf <= 0.5:
                        continue
                        
                    label   = self._model.names[int(box.cls[0])].lower()

                    # Filter by target object if specified
                    if target_id and label != target_id and label not in target_label:
                        continue

                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                    # Clamp coordinates to image width/height just in case
                    x1_c = max(0, min(x1, pil_img.width - 1))
                    y1_c = max(0, min(y1, pil_img.height - 1))
                    x2_c = max(0, min(x2, pil_img.width))
                    y2_c = max(0, min(y2, pil_img.height))

                    logger.info(f"  - ID {global_id}: {label} ({conf:.2f}) at [{x1_c},{y1_c},{x2_c},{y2_c}]")

                    # Crop the bounding box
                    crop_img = pil_img.crop((x1_c, y1_c, x2_c, y2_c))
                    
                    # Draw a red border around the cropped image
                    crop_draw = ImageDraw.Draw(crop_img)
                    crop_draw.rectangle([0, 0, crop_img.width - 1, crop_img.height - 1], outline="red", width=2)

                    buf = io.BytesIO()
                    crop_img.save(buf, format="JPEG", quality=85)
                    cropped_images[str(global_id)] = base64.b64encode(buf.getvalue()).decode()

                    yolo_detections[str(global_id)] = {
                        "camera": cam_name,
                        "bbox":   [x1_c, y1_c, x2_c, y2_c],
                        "label":  label,
                        "conf":   conf,
                    }
                    global_id += 1

            except Exception as e:
                logger.error(f"[YoloService] Error processing {cam_name}: {e}")

        return {
            "yolo_detections": yolo_detections,
            "cropped_images": cropped_images,
        }

