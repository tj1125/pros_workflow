from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.adapters.grasp_format import grasp_orientation_group
from pipeline.types import MapInfo


def apply_y_flip(position: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip the Y axis to convert between coordinate conventions."""
    flip = np.diag([1.0, -1.0, 1.0])
    return flip @ position, flip @ rotation @ flip


def _read_map_yaml(path: Path) -> tuple[Path, float, tuple[float, float, float]]:
    """Parse a ROS map YAML file and return (image_path, resolution, origin)."""
    image = None
    resolution = None
    origin = None
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
    return image_path, float(resolution), tuple(origin)


def read_pgm(path: Path) -> tuple[np.ndarray, int]:
    """Read a PGM (P5 binary or P2 ASCII) file and return (image array, maxval)."""
    with path.open("rb") as handle:
        magic = handle.readline().strip()
        if magic not in (b"P5", b"P2"):
            raise ValueError(f"Unsupported PGM format: {magic}")

        tokens: list[bytes] = []
        while len(tokens) < 3:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith(b"#"):
                continue
            if b"#" in line:
                line = line.split(b"#", 1)[0].strip()
            if line:
                tokens.extend(line.split())

        if len(tokens) < 3:
            raise ValueError(f"Invalid PGM header: {path}")

        width, height, _maxval = map(int, tokens[:3])

        if magic == b"P5":
            data = handle.read(width * height)
            if len(data) < width * height:
                raise ValueError(f"PGM data truncated: {path}")
            image = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
            maxval = 255
        else:
            vals: list[bytes] = []
            for line in handle:
                line = line.strip()
                if not line or line.startswith(b"#"):
                    continue
                if b"#" in line:
                    line = line.split(b"#", 1)[0].strip()
                if line:
                    vals.extend(line.split())
            if len(vals) < width * height:
                raise ValueError(f"PGM data truncated: {path}")
            image = np.array([int(v) for v in vals[: width * height]], dtype=np.uint8).reshape((height, width))
            maxval = 255
    return image, maxval


def load_map_info(path: Path) -> MapInfo:
    """Load map metadata (resolution, origin, size) from a ROS map YAML."""
    image_path, resolution, origin = _read_map_yaml(path)
    pgm, _ = read_pgm(image_path)
    h, w = pgm.shape
    return MapInfo(resolution=resolution, origin=origin, width=w, height=h)


def map_to_pgm(map_x: float, map_y: float, info: MapInfo) -> tuple[float, float]:
    """Convert ROS map coordinates to PGM pixel coordinates."""
    px = (map_x - info.origin[0]) / info.resolution
    py = (info.height - 1) - (map_y - info.origin[1]) / info.resolution
    return px, py


def unity_to_map(unity_x: float, unity_z: float, unity_origin: tuple[float, float, float]) -> tuple[float, float]:
    """Convert Unity world coordinates to ROS map coordinates."""
    map_x = unity_origin[2] - unity_z
    map_y = unity_x - unity_origin[0]
    return map_x, map_y


def is_white(
    unity_point: np.ndarray,
    pgm: np.ndarray,
    info: MapInfo,
    white_threshold: int,
    unity_origin: tuple[float, float, float],
) -> bool:
    """Return True if the Unity world point falls on free (white) space in the occupancy map."""
    map_x, map_y = unity_to_map(float(unity_point[0]), float(unity_point[2]), unity_origin)
    pgm_x, pgm_y = map_to_pgm(map_x, map_y, info)
    px = int(round(pgm_x))
    py = int(round(pgm_y))
    if px < 0 or py < 0 or px >= info.width or py >= info.height:
        return False
    return int(pgm[py, px]) >= white_threshold


def compute_goal_pose(
    center_world: np.ndarray,
    grasps: np.ndarray,
    confidences: np.ndarray,
    map_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute feasible robot goal poses from GraspGen candidates, ranked by count × confidence."""
    if len(grasps) == 0 or len(confidences) == 0:
        raise RuntimeError("No grasp candidates available for goal pose computation.")
    if len(grasps) != len(confidences):
        raise ValueError("Mismatch between number of grasps and confidence scores.")

    pgm, _ = read_pgm(Path(map_cfg["map_pgm"]))
    info = load_map_info(Path(map_cfg["map_yaml"]))
    unity_origin = tuple(float(v) for v in map_cfg["unity_map_origin"])
    offset = float(map_cfg["offset"])
    white_th = int(map_cfg["white_threshold"])

    counts: dict[int, int] = {}
    best_by_group: dict[int, dict[str, float | np.ndarray | list[float]]] = {}

    for grasp_matrix, confidence in zip(grasps, confidences):
        position = np.array(grasp_matrix[:3, 3], dtype=float)
        rotation = np.array(grasp_matrix[:3, :3], dtype=float)
        group = grasp_orientation_group(grasp_matrix)
        confidence = float(confidence)

        position, rotation = apply_y_flip(position, rotation)
        world_pos = position + center_world

        approach = rotation[:, 2]
        norm = np.linalg.norm(approach)
        if norm == 0:
            continue

        direction = approach / norm
        if np.dot(direction, position) < 0:
            direction = -direction

        candidate = world_pos + direction * offset
        if not is_white(candidate, pgm, info, white_th, unity_origin):
            continue

        counts[group] = counts.get(group, 0) + 1
        current_best = best_by_group.get(group)
        if current_best is None or confidence > float(current_best["confidence"]):
            pose_matrix = np.eye(4, dtype=float)
            pose_matrix[:3, :3] = rotation
            pose_matrix[:3, 3] = world_pos
            goal_ros = unity_to_map(float(candidate[0]), float(candidate[2]), unity_origin)
            best_by_group[group] = {
                "confidence": confidence,
                "goal_unity": candidate,
                "goal_ros": [float(goal_ros[0]), float(goal_ros[1])],
                "pose_unity": world_pos,
                "pose_matrix_unity": pose_matrix,
            }

    if not counts:
        raise RuntimeError("No feasible goal pose found on free map area.")

    sorted_groups = sorted(
        counts.keys(),
        key=lambda g: (-counts[g], -float(best_by_group[g]["confidence"]), g),
    )
    group_ranking = []
    for rank, group in enumerate(sorted_groups, start=1):
        info_group = best_by_group[group]
        group_ranking.append(
            {
                "rank": int(rank),
                "best_confidence": float(info_group["confidence"]),
                "best_pose_unity": np.array(info_group["pose_unity"], dtype=float).tolist(),
                "best_pose_matrix_unity": np.array(info_group["pose_matrix_unity"], dtype=float).tolist(),
                "best_goal_pose_unity": np.array(info_group["goal_unity"], dtype=float).tolist(),
                "best_goal_pose_ros_map": [float(info_group["goal_ros"][0]), float(info_group["goal_ros"][1])],
            }
        )

    return {"group_ranking": group_ranking}
