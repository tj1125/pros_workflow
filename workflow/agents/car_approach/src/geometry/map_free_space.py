"""Minimal ROS occupancy-map feasibility checks for base poses."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import coordinate_transforms as coord


@dataclass(frozen=True)
class OccupancyMapMeta:
    pgm_path: Path
    resolution_m: float
    origin_xy: tuple[float, float]
    free_thresh: float
    occupied_thresh: float
    negate: bool


@dataclass(frozen=True)
class MapFreeSpace:
    free_cell_keys: frozenset[tuple[int, int]]
    origin_xy: tuple[float, float]
    resolution_m: float
    vehicle_footprint_points_xy: np.ndarray


def build_map_free_space(
    map_yaml_path: Path,
    *,
    vehicle_length_x_m: float,
    vehicle_length_y_m: float,
) -> MapFreeSpace:
    meta = load_map_meta(map_yaml_path)
    free_cells_map_xy = load_free_cells_map_xy(meta)
    if len(free_cells_map_xy) == 0:
        raise RuntimeError(f"Map has no free cells: {map_yaml_path}")
    resolution = float(meta.resolution_m)
    origin_x, origin_y = meta.origin_xy
    free_cell_keys = frozenset(
        (
            int(round((float(cell[0]) - origin_x) / resolution)),
            int(round((float(cell[1]) - origin_y) / resolution)),
        )
        for cell in free_cells_map_xy
    )
    return MapFreeSpace(
        free_cell_keys=free_cell_keys,
        origin_xy=(origin_x, origin_y),
        resolution_m=resolution,
        vehicle_footprint_points_xy=_rectangular_footprint_points(
            length_x_m=vehicle_length_x_m,
            length_y_m=vehicle_length_y_m,
            resolution_m=resolution,
        ),
    )


def load_map_meta(yaml_path: Path) -> OccupancyMapMeta:
    path = Path(yaml_path).expanduser()
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        payload = _parse_simple_yaml(path)
    else:
        with path.open("r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Map YAML must be a mapping: {path}")

    image_field = str(payload.get("image", "map.pgm"))
    origin_raw = payload.get("origin", [0.0, 0.0, 0.0])
    return OccupancyMapMeta(
        pgm_path=(path.parent / image_field).resolve(),
        resolution_m=float(payload.get("resolution", 0.05)),
        origin_xy=(float(origin_raw[0]), float(origin_raw[1])),
        free_thresh=float(payload.get("free_thresh", 0.25)),
        occupied_thresh=float(payload.get("occupied_thresh", 0.65)),
        negate=bool(int(payload.get("negate", 0))),
    )


def load_free_cells_map_xy(meta: OccupancyMapMeta) -> np.ndarray:
    image = _load_pgm(meta.pgm_path)
    height, _ = image.shape
    pixels = image.astype(np.float32) / 255.0
    if meta.negate:
        pixels = 1.0 - pixels
    free_mask = pixels > float(meta.free_thresh)
    rows, cols = np.where(free_mask)
    map_x = cols.astype(np.float32) * meta.resolution_m + meta.origin_xy[0]
    map_y = (height - 1 - rows).astype(np.float32) * meta.resolution_m + meta.origin_xy[1]
    return np.stack([map_x, map_y], axis=1).astype(np.float32)


def ros_map_pose_is_clear(map_free_space: MapFreeSpace, ros_map_pose: object) -> bool:
    pose = coord.pose2d_from_any(ros_map_pose)
    footprint_world = coord.footprint_world_xy(
        map_free_space.vehicle_footprint_points_xy,
        pose=pose,
    )
    origin_x, origin_y = map_free_space.origin_xy
    resolution = float(map_free_space.resolution_m)
    keys = zip(
        np.rint((footprint_world[:, 0] - origin_x) / resolution).astype(np.int32),
        np.rint((footprint_world[:, 1] - origin_y) / resolution).astype(np.int32),
    )
    return all((int(key_x), int(key_y)) in map_free_space.free_cell_keys for key_x, key_y in keys)


def _rectangular_footprint_points(*, length_x_m: float, length_y_m: float, resolution_m: float) -> np.ndarray:
    step = max(float(resolution_m) * 0.5, 0.01)
    half_x = float(length_x_m) * 0.5
    half_y = float(length_y_m) * 0.5
    x_values = np.arange(-half_x, half_x + step * 0.5, step, dtype=np.float64)
    y_values = np.arange(-half_y, half_y + step * 0.5, step, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="xy")
    corners = np.asarray([[-half_x, -half_y], [-half_x, half_y], [half_x, -half_y], [half_x, half_y]], dtype=np.float64)
    grid_points = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])
    return np.unique(np.vstack([grid_points, corners]), axis=0).astype(np.float64)


def _load_pgm(pgm_path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError:
        return _load_pgm_pure_python(pgm_path)
    with Image.open(pgm_path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _load_pgm_pure_python(pgm_path: Path) -> np.ndarray:
    with pgm_path.open("rb") as fh:
        magic = fh.readline().strip()
        if magic != b"P5":
            raise ValueError(f"Only binary PGM P5 is supported without Pillow, got {magic!r}")
        line = b""
        while not line or line.startswith(b"#"):
            line = fh.readline().strip()
        width, height = (int(value) for value in line.split())
        max_value = int(fh.readline().strip())
        raw = fh.read()
    dtype = np.uint16 if max_value > 255 else np.uint8
    image = np.frombuffer(raw, dtype=dtype).reshape(height, width)
    if max_value != 255:
        image = (image.astype(np.float32) / float(max_value) * 255.0).astype(np.uint8)
    return image.astype(np.uint8)


def _parse_simple_yaml(path: Path) -> dict[str, object]:
    payload: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        raw_value = value.strip()
        try:
            payload[key.strip()] = ast.literal_eval(raw_value)
        except Exception:
            payload[key.strip()] = raw_value
    return payload
