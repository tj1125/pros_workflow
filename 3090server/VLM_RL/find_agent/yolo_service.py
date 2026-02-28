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

    def detect_and_annotate(self, camera_images: Dict[str, str]) -> Dict[str, Any]:
        """
        Input: { "Camera1": "<base64_img>", "Camera2": "<base64_img>" }
        Output: { "1": {bbox, label, conf, camera, annotated_image_base64}, "2": ... }
        """
        if not self._model:
            raise RuntimeError("YOLO model not initialized.")

        logger.info(f"[YoloService] Running detection on {len(camera_images)} images.")
        
        yolo_detections = {}
        global_id = 1

        for cam_name, b64_str in camera_images.items():
            try:
                # Decode image
                img_bytes = base64.b64decode(b64_str)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                # Run YOLO inference
                results = self._model(pil_img, verbose=False)

                # Annotate image
                draw = ImageDraw.Draw(pil_img)
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                except Exception:
                    font = ImageFont.load_default()

                for box in results[0].boxes:
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    label = self._model.names[int(box.cls[0])]

                    # Draw red bounding box and ID text
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    draw.text((x1 + 4, y1 + 4), str(global_id), fill="red", font=font)

                    # Encode annotated image
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG", quality=85)
                    annotated_b64 = base64.b64encode(buf.getvalue()).decode()

                    yolo_detections[str(global_id)] = {
                        "camera": cam_name,
                        "bbox": [x1, y1, x2, y2],
                        "label": label,
                        "conf": conf,
                        "annotated_image_base64": annotated_b64,
                    }
                    global_id += 1

            except Exception as e:
                logger.error(f"[YoloService] Error processing {cam_name}: {e}")

        return yolo_detections
