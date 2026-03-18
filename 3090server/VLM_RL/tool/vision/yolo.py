from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class YoloDetection:
    class_id: int
    label: str
    conf: float
    bbox_xyxy: tuple[float, float, float, float]


def load_yolo_model(weights_path: str | Path):
    """Load a YOLO model from the given weights file."""
    from ultralytics import YOLO  # type: ignore

    return YOLO(str(weights_path))


def run_yolo(
    model: Any,
    source: Any,
    *,
    conf_threshold: float | None = None,
    device: str | None = None,
):
    """Run YOLO once and return the first Ultralytics result object."""
    kwargs: dict[str, Any] = {
        "source": source,
        "verbose": False,
    }
    if conf_threshold is not None:
        kwargs["conf"] = float(conf_threshold)
    if device is not None:
        kwargs["device"] = device

    results = model.predict(**kwargs)
    if not results:
        raise RuntimeError("YOLO returned no result.")
    return results[0]


def result_label_lookup(result) -> dict[int, str]:
    names = result.names
    if isinstance(names, dict):
        return {int(k): str(v).lower() for k, v in names.items()}
    return {int(idx): str(name).lower() for idx, name in enumerate(names)}


def extract_detections(
    result,
    *,
    conf_threshold: float = 0.0,
) -> list[YoloDetection]:
    """Convert an Ultralytics result into normalized detection records."""
    if result.boxes is None or len(result.boxes) == 0:
        return []

    name_lookup = result_label_lookup(result)
    detections: list[YoloDetection] = []
    for box in result.boxes:
        conf = float(box.conf.item())
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls.item())
        label = name_lookup.get(cls_id, str(cls_id)).lower()
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        detections.append(
            YoloDetection(
                class_id=cls_id,
                label=label,
                conf=conf,
                bbox_xyxy=(x1, y1, x2, y2),
            )
        )
    return detections


def select_best_bbox(
    result,
    class_name: str,
) -> tuple[tuple[float, float, float, float], float]:
    """Return the highest-confidence bbox for the requested class."""
    target = class_name.lower()
    best_detection: YoloDetection | None = None

    for detection in extract_detections(result, conf_threshold=0.0):
        if detection.label != target:
            continue
        if best_detection is None or detection.conf > best_detection.conf:
            best_detection = detection

    if best_detection is None:
        available_labels = sorted(set(result_label_lookup(result).values()))
        raise ValueError(
            f"No YOLO detection matched '{class_name}'. Available labels: {available_labels}"
        )

    return best_detection.bbox_xyxy, float(best_detection.conf)
