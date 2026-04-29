from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..pybullet_smoke import _load_yaml


@dataclass
class CameraIntrinsics:
    camera_name: str
    image_width: int
    image_height: int
    k: np.ndarray


def load_camera_intrinsics(intrinsics_path: Path) -> CameraIntrinsics:
    payload = _load_yaml(intrinsics_path)
    camera_matrix = payload.get("camera_matrix", {})
    data = camera_matrix.get("data")
    if not isinstance(data, list) or len(data) != 9:
        raise ValueError(f"Invalid camera_matrix.data in {intrinsics_path}")
    k = np.asarray(data, dtype=np.float32).reshape(3, 3)
    return CameraIntrinsics(
        camera_name=str(payload.get("camera_name", intrinsics_path.stem)),
        image_width=int(payload["image_width"]),
        image_height=int(payload["image_height"]),
        k=k,
    )
