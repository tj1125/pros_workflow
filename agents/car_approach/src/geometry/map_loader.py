"""
map_loader.py — Load a ROS-style 2-D occupancy-grid PGM map and expose its
free cells as Unity world-frame (x, z) coordinates.

Coordinate conventions
----------------------
Map frame (ROS):   x = right,  y = up   (metres from map origin)
Unity world frame: x = right,  y = up,  z = forward  (floor plane = X-Z)

The conversion used here:
    unity_x = map_x
    unity_z = map_y

(This matches the AMCL convention used in the project where the localisation
system maps Unity positions to the ROS map frame.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class OccupancyMapMeta:
    """Metadata loaded from a ROS map YAML file."""

    pgm_path: Path
    resolution_m: float          # metres per pixel
    origin_xy: tuple[float, float]  # map frame (x, y) of the bottom-left pixel
    free_thresh: float           # pixels >= free_thresh * 255 are free
    occupied_thresh: float       # pixels <= occupied_thresh * 255 are occupied
    negate: bool                 # invert pixel intensities when True


def load_map_meta(yaml_path: Path) -> OccupancyMapMeta:
    """Parse a ROS *map.yaml* file and return an :class:`OccupancyMapMeta`.

    Supports maps written by ``map_saver`` / ``slam_toolbox``.
    """
    try:
        import yaml  # type: ignore[import]
        with yaml_path.open("r", encoding="utf-8") as fh:
            payload: dict = yaml.safe_load(fh) or {}
    except ImportError:
        # Minimal fallback parser - handles the simple flat-key YAML written by
        # map_saver (no nested keys, no anchors).
        payload = _parse_simple_map_yaml(yaml_path)

    image_field = str(payload.get("image", "map.pgm"))
    pgm_path = yaml_path.parent / image_field
    resolution = float(payload.get("resolution", 0.05))
    origin_raw = payload.get("origin", [0.0, 0.0, 0.0])
    origin_xy = (float(origin_raw[0]), float(origin_raw[1]))
    free_thresh = float(payload.get("free_thresh", 0.25))
    occupied_thresh = float(payload.get("occupied_thresh", 0.65))
    negate = bool(int(payload.get("negate", 0)))

    return OccupancyMapMeta(
        pgm_path=pgm_path,
        resolution_m=resolution,
        origin_xy=origin_xy,
        free_thresh=free_thresh,
        occupied_thresh=occupied_thresh,
        negate=negate,
    )


def load_free_cells_unity_xz(meta: OccupancyMapMeta) -> np.ndarray:
    """Return an ``(N, 2)`` float32 array of free-cell centres in Unity (x, z).

    Free cells are those whose value (after normalising to [0, 1]) is above
    ``meta.free_thresh``.  Unknown cells (value ≈ 0.5 in the raw image) are
    treated as occupied/unknown and are excluded.

    Returns
    -------
    np.ndarray, shape (N, 2), dtype float32
        Each row is ``[unity_x, unity_z]`` (metres, Unity world frame).
    """
    image = _load_pgm(meta.pgm_path)    # shape (H, W), uint8

    H, W = image.shape
    pixels = image.astype(np.float32) / 255.0
    if meta.negate:
        pixels = 1.0 - pixels

    # PGM convention: 255 = white = free, 0 = black = occupied, 205 = unknown.
    # After normalisation: free > free_thresh, occupied < occupied_thresh.
    free_mask = pixels > meta.free_thresh

    # Row 0 in the image is the TOP of the map; row (H-1) is the BOTTOM.
    # In ROS map convention the origin is at the BOTTOM-LEFT corner, so we
    # must flip the row index when converting to metric coordinates.
    rows, cols = np.where(free_mask)          # image indices
    map_x = cols.astype(np.float32) * meta.resolution_m + meta.origin_xy[0]
    map_y = (H - 1 - rows).astype(np.float32) * meta.resolution_m + meta.origin_xy[1]

    # map (x, y) → Unity (x, z)
    unity_x = map_x
    unity_z = map_y
    return np.stack([unity_x, unity_z], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_pgm(pgm_path: Path) -> np.ndarray:
    """Load a PGM (P5 binary or P2 ASCII) image as a uint8 array."""
    try:
        from PIL import Image  # type: ignore[import]
        img = Image.open(pgm_path).convert("L")
        return np.asarray(img, dtype=np.uint8)
    except ImportError:
        pass
    # Pure-Python fallback for P5 (binary) PGMs only.
    return _load_pgm_pure_python(pgm_path)


def _load_pgm_pure_python(pgm_path: Path) -> np.ndarray:
    with pgm_path.open("rb") as fh:
        magic = fh.readline().strip()
        if magic != b"P5":
            raise ValueError(f"Only binary PGM (P5) is supported, got: {magic!r}")
        # skip comment lines
        line = b""
        while not line or line.startswith(b"#"):
            line = fh.readline().strip()
        W, H = (int(v) for v in line.split())
        max_val = int(fh.readline().strip())
        raw = fh.read()
    dtype = np.uint16 if max_val > 255 else np.uint8
    arr = np.frombuffer(raw, dtype=dtype).reshape(H, W)
    if max_val != 255:
        arr = (arr.astype(np.float32) / max_val * 255).astype(np.uint8)
    return arr.astype(np.uint8)


def _parse_simple_map_yaml(path: Path) -> dict:
    import ast
    payload: dict = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        raw_val = value.strip()
        try:
            payload[key.strip()] = ast.literal_eval(raw_val)
        except Exception:
            payload[key.strip()] = raw_val
    return payload
