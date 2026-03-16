"""
yolo_service.py — Core YOLO Detection Logic for FindAgent

Abstracts away PyTorch/Ultralytics inference and image annotation
from the A2A messaging layer.

Returns unformatted dict of globally-numbered detections.
"""

import base64
import io
import logging
import re
from typing import Dict, Any

from PIL import Image, ImageDraw, ImageFont

from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = Path(__file__).parent.parent / "models/yolo/pure720.pt"
_MIN_DET_CONF = 0.5
_MIN_GROUP_MAX_CONF = 0.6
_GROUP_SIZE = 3
_MAX_GROUP_CAMERA_INDEX = 12


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
            "annotated_images": { "1": "<base64>", "2": "<base64>"}
        }
        """
        if not self._model:
            raise RuntimeError("YOLO model not initialized.")

        logger.info(f"[YoloService] Running detection on {len(camera_images)} images.")

        target_id    = target_object.get("id", "").lower()    if target_object else ""
        target_label = target_object.get("label", "")          if target_object else ""

        camera_candidates: list[Dict[str, Any]] = []

        for cam_name, b64_str in sorted(camera_images.items()):
            try:
                img_bytes = base64.b64decode(b64_str)
                pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                results  = self._model(pil_img, verbose=False)
                num_det  = len(results[0].boxes)
                logger.info(f"[YoloService] {cam_name}: Detected {num_det} objects.")

                detections = []
                for box in results[0].boxes:
                    conf    = float(box.conf[0])

                    if conf <= _MIN_DET_CONF:
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

                    detections.append(
                        {
                            "camera": cam_name,
                            "bbox": [x1_c, y1_c, x2_c, y2_c],
                            "label": label,
                            "conf": conf,
                        }
                    )

                if not detections:
                    continue

                group_id = self._camera_group(cam_name)
                if group_id is None:
                    logger.warning(f"[YoloService] {cam_name}: camera group not recognized, skipping.")
                    continue

                avg_conf = sum(det["conf"] for det in detections) / len(detections)
                max_conf = max(det["conf"] for det in detections)
                logger.info(
                    f"[YoloService] {cam_name}: group={group_id} kept={len(detections)} "
                    f"avg_conf={avg_conf:.3f} max_conf={max_conf:.3f}"
                )

                camera_candidates.append(
                    {
                        "camera": cam_name,
                        "group_id": group_id,
                        "avg_conf": avg_conf,
                        "max_conf": max_conf,
                        "detections": detections,
                        "annotated_image": self._encode_annotated_image(pil_img, detections),
                    }
                )

            except Exception as e:
                logger.error(f"[YoloService] Error processing {cam_name}: {e}")

        best_per_group: Dict[int, Dict[str, Any]] = {}
        for candidate in camera_candidates:
            group_id = candidate["group_id"]
            current = best_per_group.get(group_id)
            if current is None:
                best_per_group[group_id] = candidate
                continue

            candidate_key = (
                float(candidate["avg_conf"]),
                float(candidate["max_conf"]),
                len(candidate["detections"]),
                str(candidate["camera"]),
            )
            current_key = (
                float(current["avg_conf"]),
                float(current["max_conf"]),
                len(current["detections"]),
                str(current["camera"]),
            )
            if candidate_key > current_key:
                best_per_group[group_id] = candidate

        yolo_detections: Dict[str, Any] = {}
        annotated_images: Dict[str, str] = {}
        selected_groups: Dict[str, Any] = {}
        global_id = 1

        for group_id in sorted(best_per_group):
            candidate = best_per_group[group_id]
            if float(candidate["max_conf"]) < _MIN_GROUP_MAX_CONF:
                logger.info(
                    f"[YoloService] Group {group_id}: skipped because max_conf={candidate['max_conf']:.3f} "
                    f"< {_MIN_GROUP_MAX_CONF:.1f}"
                )
                continue

            selected_groups[str(group_id)] = {
                "camera": candidate["camera"],
                "avg_conf": candidate["avg_conf"],
                "max_conf": candidate["max_conf"],
                "num_detections": len(candidate["detections"]),
            }

            for det in candidate["detections"]:
                det_id = str(global_id)
                logger.info(
                    f"  - ID {det_id}: group={group_id} camera={det['camera']} "
                    f"{det['label']} ({det['conf']:.2f}) at {det['bbox']}"
                )
                yolo_detections[det_id] = {
                    **det,
                    "group_id": group_id,
                    "camera_avg_conf": candidate["avg_conf"],
                    "camera_max_conf": candidate["max_conf"],
                }
                annotated_images[det_id] = candidate["annotated_image"]
                global_id += 1

        return {
            "yolo_detections": yolo_detections,
            "annotated_images": annotated_images,
            "selected_groups": selected_groups,
        }

    @staticmethod
    def _camera_group(camera_name: str) -> int | None:
        match = re.search(r"(\d+)(?!.*\d)", str(camera_name))
        if not match:
            return None
        camera_idx = int(match.group(1))
        if camera_idx < 1 or camera_idx > _MAX_GROUP_CAMERA_INDEX:
            return None
        return ((camera_idx - 1) // _GROUP_SIZE) + 1

    @staticmethod
    def _encode_annotated_image(pil_img: Image.Image, detections: list[Dict[str, Any]]) -> str:
        det_img = pil_img.copy()
        det_draw = ImageDraw.Draw(det_img)
        for det in detections:
            det_draw.rectangle(det["bbox"], outline="red", width=2)

        buf = io.BytesIO()
        det_img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
