"""
commander/nav/ablation_goals.py — Experiment A3 candidate goal-pose generation.

Ablation A3 (``w/o get_item_info_no_sam3d_node``) removes object-information
estimation. Instead of the full system's grasp-feasibility ranked goal poses,
candidate goal poses are generated purely from the target centre and the ROS
occupancy map (``keepout_map``):

  - Split the area around the target into ``num_sectors`` angular sectors
    (8 方位 by default, each 360/N degrees wide).
  - Within ``[r_min_m, r_max_m]`` of the target, keep the map cells that are
    feasible, i.e. no black/occupied cell (``pgm <= black_threshold``) lies within
    ``robot_radius_m`` of them (white and gray/unknown cells are allowed).
  - A sector counts as a candidate as long as it has at least one feasible cell
    ("只要可走就算"); infeasible sectors are dropped (so the candidate count is
    <= num_sectors).

The default mode is ``fixed_offset`` (see ``fixed_offset_feasible_candidates``):
push a fixed stand-off out from the target in 8 directions and keep the points
with no obstacle in the footprint.
  - The candidate goal pose for a sector is placed at the **centroid of that
    sector's walkable cells** ("那個方位的中間"). If that centroid happens to be
    non-walkable (e.g. an L-shaped sector), it snaps to the walkable cell in the
    sector nearest to the centroid.

The heading (yaw) is filled in later by ``goal_builder.goal_pose_from_ros_map``
from pure geometry (face the target centre); no object orientation is estimated.

This module returns plain candidate dicts (in ROS map xy); the caller wraps them
into the ``group_ranking`` contract used by the rest of the graph.
"""

from __future__ import annotations

import ast
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KeepoutMap:
    pgm: np.ndarray  # (H, W) uint8 grayscale, 255 = free/white
    resolution: float
    origin_x: float
    origin_y: float
    height: int
    width: int


def _read_pgm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic = handle.readline().strip()
        if magic not in (b"P5", b"P2"):
            raise ValueError(f"Unsupported PGM format: {magic}")
        tokens: list[bytes] = []
        while len(tokens) < 3:
            line = handle.readline()
            if not line:
                break
            line = line.split(b"#", 1)[0].strip()
            if line:
                tokens.extend(line.split())
        if len(tokens) < 3:
            raise ValueError(f"Invalid PGM header: {path}")
        width, height, _maxval = (int(tok) for tok in tokens[:3])
        if magic == b"P5":
            data = handle.read(width * height)
            if len(data) < width * height:
                raise ValueError(f"PGM data truncated: {path}")
            return np.frombuffer(data, dtype=np.uint8).reshape((height, width))
        values: list[int] = []
        for line in handle:
            line = line.split(b"#", 1)[0].strip()
            if line:
                values.extend(int(v) for v in line.split())
        if len(values) < width * height:
            raise ValueError(f"PGM data truncated: {path}")
        return np.array(values[: width * height], dtype=np.uint8).reshape((height, width))


def _read_map_yaml(path: Path) -> tuple[Path, float, tuple[float, float]]:
    image = resolution = origin = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("image:"):
            image = line.split(":", 1)[1].strip()
        elif line.startswith("resolution:"):
            resolution = float(line.split(":", 1)[1].strip())
        elif line.startswith("origin:"):
            origin = ast.literal_eval(line.split(":", 1)[1].strip())
    if image is None or resolution is None or origin is None:
        raise ValueError(f"Invalid map yaml: {path}")
    image_path = (path.parent / image).resolve()
    return image_path, float(resolution), (float(origin[0]), float(origin[1]))


@lru_cache(maxsize=4)
def load_keepout_map(yaml_path: str) -> KeepoutMap:
    path = Path(yaml_path).expanduser().resolve()
    image_path, resolution, origin = _read_map_yaml(path)
    pgm = _read_pgm(image_path)
    height, width = pgm.shape
    return KeepoutMap(
        pgm=pgm,
        resolution=resolution,
        origin_x=origin[0],
        origin_y=origin[1],
        height=height,
        width=width,
    )


def _no_black_footprint_mask(pgm: np.ndarray, radius_px: int, black_threshold: int) -> np.ndarray:
    """True for cells whose robot footprint (a disk of ``radius_px``) contains no
    black/occupied cell. A cell is "black" when ``pgm <= black_threshold``
    (occupied / keepout); white and gray/unknown cells are allowed. Computed by
    dilating the black mask and inverting."""
    black = pgm <= int(black_threshold)
    if radius_px <= 0:
        return ~black
    try:
        from scipy.ndimage import binary_dilation

        span = np.arange(-radius_px, radius_px + 1)
        yy, xx = np.meshgrid(span, span)
        disk = (xx * xx + yy * yy) <= radius_px * radius_px
        blocked = binary_dilation(black, structure=disk, border_value=0)
        return ~blocked
    except Exception as exc:  # pragma: no cover - fallback if scipy is unavailable
        logger.warning("[ablation_goals] scipy dilation unavailable (%s); using per-shift dilation", exc)
        blocked = black.copy()
        height, width = black.shape
        for dy in range(-radius_px, radius_px + 1):
            for dx in range(-radius_px, radius_px + 1):
                if dx * dx + dy * dy > radius_px * radius_px or (dx == 0 and dy == 0):
                    continue
                shifted = np.zeros_like(black)
                ys0, ys1 = max(0, dy), min(height, height + dy)
                xs0, xs1 = max(0, dx), min(width, width + dx)
                shifted[ys0 - dy : ys1 - dy, xs0 - dx : xs1 - dx] = black[ys0:ys1, xs0:xs1]
                blocked |= shifted
        return ~blocked


def _map_xy_to_px(map_x: float, map_y: float, m: KeepoutMap) -> tuple[int, int]:
    px = int(round((map_x - m.origin_x) / m.resolution))
    py = int(round((m.height - 1) - (map_y - m.origin_y) / m.resolution))
    return px, py


def fixed_offset_feasible_candidates(
    target_map_x: float,
    target_map_y: float,
    *,
    map_yaml_path: str,
    robot_radius_m: float = 0.25,
    black_threshold: int = 50,
    distance_m: float = 0.6,
    num_directions: int = 8,
    start_deg: float = 0.0,
    keep_infeasible: bool = False,
) -> list[dict[str, Any]]:
    """Push ``distance_m`` out from the target centre in ``num_directions`` fixed
    directions, and keep the points whose footprint contains no obstacle.

    Each direction's candidate is ``target_center + distance_m * (cos, sin)`` in
    ROS map metres. A candidate is feasible when there is NO black/occupied cell
    (``pgm <= black_threshold``) within ``robot_radius_m`` of the goal (white and
    gray/unknown cells are allowed); with ``robot_radius_m == 0`` only the goal
    cell itself must be non-black. By default non-feasible directions are dropped
    (``keep_infeasible=False``), so the candidate count is <= num_directions. The
    heading (face the target) is filled in later by
    ``goal_builder.goal_pose_from_ros_map``.

    Each candidate dict has: ``goal_pose_ros_map`` ([x, y]), ``direction_index``,
    ``direction_deg``, ``map_feasible`` and ``distance_m``.
    """
    num_directions = max(1, int(num_directions))
    try:
        m = load_keepout_map(map_yaml_path)
    except Exception as exc:
        logger.warning("[ablation_goals] failed to load keepout map %s: %s", map_yaml_path, exc)
        return []

    radius_px = int(round(max(0.0, robot_radius_m) / m.resolution))
    free = _no_black_footprint_mask(m.pgm, radius_px, int(black_threshold))

    candidates: list[dict[str, Any]] = []
    for index in range(num_directions):
        angle_deg = start_deg + index * (360.0 / num_directions)
        angle_rad = math.radians(angle_deg)
        goal_x = target_map_x + distance_m * math.cos(angle_rad)
        goal_y = target_map_y + distance_m * math.sin(angle_rad)
        px, py = _map_xy_to_px(goal_x, goal_y, m)
        feasible = bool(0 <= py < m.height and 0 <= px < m.width and free[py, px])
        if feasible or keep_infeasible:
            candidates.append(
                {
                    "goal_pose_ros_map": [float(goal_x), float(goal_y)],
                    "direction_index": index,
                    "direction_deg": float(angle_deg % 360.0),
                    "map_feasible": feasible,
                    "distance_m": float(distance_m),
                }
            )
    return candidates


def sector_centroid_candidates(
    target_map_x: float,
    target_map_y: float,
    *,
    map_yaml_path: str,
    robot_radius_m: float = 0.25,
    black_threshold: int = 50,
    r_min_m: float = 0.4,
    r_max_m: float = 1.5,
    num_sectors: int = 8,
    start_deg: float = 0.0,
) -> list[dict[str, Any]]:
    """Return one map-aligned candidate goal pose per feasible sector.

    A cell is feasible when no black/occupied cell is within ``robot_radius_m``.
    Each candidate dict has: ``goal_pose_ros_map`` ([x, y] in ROS map metres),
    ``sector_index``, ``sector_center_deg``, ``sector_cell_count`` and
    ``mean_distance_m``. Sectors with no feasible cell are omitted.
    """
    if num_sectors < 1:
        num_sectors = 1
    try:
        m = load_keepout_map(map_yaml_path)
    except Exception as exc:
        logger.warning("[ablation_goals] failed to load keepout map %s: %s", map_yaml_path, exc)
        return []

    radius_px = int(round(max(0.0, robot_radius_m) / m.resolution))
    free = _no_black_footprint_mask(m.pgm, radius_px, int(black_threshold))

    # Pixel bounding box covering the r_max disk around the target.
    tpx = (target_map_x - m.origin_x) / m.resolution
    tpy = (m.height - 1) - (target_map_y - m.origin_y) / m.resolution
    rad_px = r_max_m / m.resolution
    py0 = max(0, int(math.floor(tpy - rad_px)))
    py1 = min(m.height - 1, int(math.ceil(tpy + rad_px)))
    px0 = max(0, int(math.floor(tpx - rad_px)))
    px1 = min(m.width - 1, int(math.ceil(tpx + rad_px)))
    if py1 < py0 or px1 < px0:
        return []

    ys = np.arange(py0, py1 + 1)
    xs = np.arange(px0, px1 + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    # Map xy of each pixel centre (inverse of _map_xy_to_px).
    map_x = grid_x * m.resolution + m.origin_x
    map_y = (m.height - 1 - grid_y) * m.resolution + m.origin_y
    dx = map_x - target_map_x
    dy = map_y - target_map_y
    dist = np.hypot(dx, dy)

    sector_width = 360.0 / num_sectors
    valid = (dist >= r_min_m) & (dist <= r_max_m) & free[py0 : py1 + 1, px0 : px1 + 1]
    if not bool(valid.any()):
        return []
    angle_deg = np.degrees(np.arctan2(dy, dx))
    sector = np.floor(((angle_deg - start_deg + sector_width / 2.0) % 360.0) / sector_width).astype(int) % num_sectors

    candidates: list[dict[str, Any]] = []
    for index in range(num_sectors):
        selected = valid & (sector == index)
        count = int(selected.sum())
        if count == 0:
            continue
        sel_x = map_x[selected]
        sel_y = map_y[selected]
        centroid_x = float(sel_x.mean())
        centroid_y = float(sel_y.mean())
        # Keep the goal walkable: if the centroid lands on a non-free cell, snap
        # to the sector's walkable cell nearest to the centroid.
        cpx, cpy = _map_xy_to_px(centroid_x, centroid_y, m)
        centroid_free = 0 <= cpy < m.height and 0 <= cpx < m.width and bool(free[cpy, cpx])
        if not centroid_free:
            nearest = int(np.argmin((sel_x - centroid_x) ** 2 + (sel_y - centroid_y) ** 2))
            centroid_x = float(sel_x[nearest])
            centroid_y = float(sel_y[nearest])
        candidates.append(
            {
                "goal_pose_ros_map": [centroid_x, centroid_y],
                "sector_index": index,
                "sector_center_deg": float((start_deg + index * sector_width) % 360.0),
                "sector_cell_count": count,
                "mean_distance_m": round(float(dist[selected].mean()), 4),
            }
        )
    return candidates


def _map_xy_to_float_px(map_x: float, map_y: float, m: KeepoutMap) -> tuple[float, float]:
    px = (map_x - m.origin_x) / m.resolution
    py = (m.height - 1) - (map_y - m.origin_y) / m.resolution
    return px, py


def save_goal_pose_overlay(
    *,
    map_yaml_path: str,
    target_map_x: float,
    target_map_y: float,
    group_ranking: list[dict[str, Any]],
    out_path: str,
    robot_radius_m: float = 0.25,
    crop_margin_m: float = 1.5,
    scale: int = 6,
    title: str = "",
) -> str | None:
    """Render the keepout map with the aligned goal poses and save it as a PNG.

    Draws the target centre (red crosshair) and only the feasible goal poses
    (green) with their robot-footprint circle, a short heading arrow facing the
    target, and the rank label. Infeasible candidates are never drawn. The view is
    cropped to ``crop_margin_m`` around the target and upscaled by ``scale`` so the
    cluster is readable. Returns the saved path (or None on failure).
    """
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover - Pillow missing
        logger.warning("[ablation_goals] Pillow unavailable, cannot render overlay: %s", exc)
        return None
    try:
        m = load_keepout_map(map_yaml_path)
    except Exception as exc:
        logger.warning("[ablation_goals] cannot load map for overlay %s: %s", map_yaml_path, exc)
        return None

    tpx, tpy = _map_xy_to_float_px(target_map_x, target_map_y, m)
    margin_px = max(6.0, float(crop_margin_m) / m.resolution)
    x0 = int(max(0, math.floor(tpx - margin_px)))
    x1 = int(min(m.width, math.ceil(tpx + margin_px)))
    y0 = int(max(0, math.floor(tpy - margin_px)))
    y1 = int(min(m.height, math.ceil(tpy + margin_px)))
    if x1 <= x0 or y1 <= y0:
        logger.warning("[ablation_goals] target projects outside the map; overlay skipped")
        return None

    s = int(max(1, scale))
    crop = np.ascontiguousarray(m.pgm[y0:y1, x0:x1])
    img = Image.fromarray(crop, mode="L").convert("RGB")
    img = img.resize((img.width * s, img.height * s), Image.NEAREST)
    draw = ImageDraw.Draw(img)

    def to_img(map_x: float, map_y: float) -> tuple[float, float]:
        fpx, fpy = _map_xy_to_float_px(map_x, map_y, m)
        return (fpx - x0) * s, (fpy - y0) * s

    radius_px = (float(robot_radius_m) / m.resolution) * s
    marker_px = max(radius_px, 6.0 * s)  # arrow/label length stays visible even when robot_radius=0
    line_w = max(1, s // 2)
    tix, tiy = to_img(target_map_x, target_map_y)

    for group in group_ranking:
        goal = group.get("best_goal_pose_ros_map") or []
        if not isinstance(goal, (list, tuple)) or len(goal) < 2:
            continue
        # Only draw feasible goal poses; infeasible ones are never shown.
        if not bool(group.get("map_feasible", True)):
            continue
        gx, gy = float(goal[0]), float(goal[1])
        ix, iy = to_img(gx, gy)
        color = (0, 170, 0)
        # robot footprint circle (only when a robot radius is considered)
        if radius_px >= 1.0:
            draw.ellipse([ix - radius_px, iy - radius_px, ix + radius_px, iy + radius_px], outline=color, width=line_w)
        # heading arrow: candidate -> target (the goal faces the target centre)
        dx, dy = tix - ix, tiy - iy
        norm = math.hypot(dx, dy) or 1.0
        hx, hy = ix + dx / norm * marker_px, iy + dy / norm * marker_px
        draw.line([ix, iy, hx, hy], fill=color, width=line_w)
        # goal dot + rank label
        draw.ellipse([ix - 2 * s, iy - 2 * s, ix + 2 * s, iy + 2 * s], fill=color)
        draw.text((ix + 3 * s, iy - 7 * s), str(group.get("rank", "")), fill=color)

    # target centre crosshair
    draw.line([tix - 5 * s, tiy, tix + 5 * s, tiy], fill=(220, 0, 0), width=line_w)
    draw.line([tix, tiy - 5 * s, tix, tiy + 5 * s], fill=(220, 0, 0), width=line_w)
    draw.ellipse([tix - 2 * s, tiy - 2 * s, tix + 2 * s, tiy + 2 * s], fill=(220, 0, 0))

    if title:
        draw.rectangle([0, 0, img.width, 12 * s // 6 + 8], fill=(255, 255, 255))
        draw.text((4, 2), str(title), fill=(0, 0, 180))

    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        return out_path
    except OSError as exc:  # pragma: no cover - disk failures only
        logger.warning("[ablation_goals] failed to write overlay %s: %s", out_path, exc)
        return None
