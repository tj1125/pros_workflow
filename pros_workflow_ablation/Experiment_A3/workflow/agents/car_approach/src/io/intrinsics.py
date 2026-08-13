"""Camera intrinsics loading."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    camera_name: str
    image_width: int
    image_height: int
    k: np.ndarray


def load_camera_intrinsics(intrinsics_path: Path) -> CameraIntrinsics:
    path = Path(intrinsics_path).expanduser()
    payload = _load_yaml(path)
    camera_matrix = payload.get("camera_matrix", {})
    data = camera_matrix.get("data") if isinstance(camera_matrix, dict) else None
    if not isinstance(data, list) or len(data) != 9:
        raise ValueError(f"Invalid camera_matrix.data in {path}")
    k = np.asarray(data, dtype=np.float32).reshape(3, 3)
    return CameraIntrinsics(
        camera_name=str(payload.get("camera_name", path.stem)),
        image_width=int(payload["image_width"]),
        image_height=int(payload["image_height"]),
        k=k,
    )


def _load_yaml(path: Path) -> dict[str, object]:
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        return _parse_camera_yaml(path)

    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Camera intrinsics must be a mapping: {path}")
    return payload


def _parse_camera_yaml(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    matrix_block = _indented_block(text, "camera_matrix")
    matrix_data = _data_list(matrix_block, path)
    return {
        "image_width": int(_scalar(text, "image_width", path)),
        "image_height": int(_scalar(text, "image_height", path)),
        "camera_name": str(_scalar(text, "camera_name", path)),
        "camera_matrix": {"data": matrix_data},
    }


def _scalar(text: str, key: str, path: Path) -> object:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"Missing {key} in {path}")
    raw_value = match.group(1).strip()
    try:
        return ast.literal_eval(raw_value)
    except Exception:
        return raw_value


def _indented_block(text: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*\n((?:[ \t]+.*(?:\n|$))+)", text, flags=re.MULTILINE)
    return match.group(1) if match else ""


def _data_list(block: str, path: Path) -> list[float]:
    match = re.search(r"data:\s*\[(.*?)\]", block, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Missing camera_matrix.data in {path}")
    try:
        values = ast.literal_eval("[" + match.group(1).replace("\n", " ") + "]")
    except Exception as exc:
        raise ValueError(f"Invalid camera_matrix.data in {path}") from exc
    if not isinstance(values, list):
        raise ValueError(f"Invalid camera_matrix.data in {path}")
    return [float(value) for value in values]
