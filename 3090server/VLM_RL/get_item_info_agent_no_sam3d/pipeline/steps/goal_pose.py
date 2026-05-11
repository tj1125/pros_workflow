from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np

from get_item_info_agent_no_sam3d.pipeline.adapters.grasp_format import grasp_orientation_group
from get_item_info_agent_no_sam3d.pipeline.types import MapInfo


def _read_map_yaml(path: Path) -> tuple[Path, float, tuple[float, float, float]]:
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
    image_path, resolution, origin = _read_map_yaml(path)
    pgm, _ = read_pgm(image_path)
    h, w = pgm.shape
    return MapInfo(resolution=resolution, origin=origin, width=w, height=h)


def map_to_pgm(map_x: float, map_y: float, info: MapInfo) -> tuple[float, float]:
    px = (map_x - info.origin[0]) / info.resolution
    py = (info.height - 1) - (map_y - info.origin[1]) / info.resolution
    return px, py


def unity_to_map(unity_x: float, unity_z: float, unity_origin: tuple[float, float, float]) -> tuple[float, float]:
    map_x = unity_origin[2] - unity_z
    map_y = unity_x - unity_origin[0]
    return map_x, map_y


def is_footprint_white(
    unity_point: np.ndarray,
    pgm: np.ndarray,
    info: MapInfo,
    white_threshold: int,
    unity_origin: tuple[float, float, float],
    robot_radius: float,
) -> bool:
    map_x, map_y = unity_to_map(float(unity_point[0]), float(unity_point[2]), unity_origin)
    pgm_x, pgm_y = map_to_pgm(map_x, map_y, info)
    radius_px = robot_radius / info.resolution
    min_px = int(np.floor(pgm_x - radius_px))
    max_px = int(np.ceil(pgm_x + radius_px))
    min_py = int(np.floor(pgm_y - radius_px))
    max_py = int(np.ceil(pgm_y + radius_px))
    if min_px < 0 or min_py < 0 or max_px >= info.width or max_py >= info.height:
        return False

    yy, xx = np.ogrid[min_py : max_py + 1, min_px : max_px + 1]
    footprint_mask = (xx - pgm_x) ** 2 + (yy - pgm_y) ** 2 <= radius_px**2
    footprint_cells = pgm[min_py : max_py + 1, min_px : max_px + 1][footprint_mask]
    return bool(np.all(footprint_cells >= white_threshold))


def compute_goal_pose(
    center_world: np.ndarray,
    grasps: np.ndarray,
    confidences: np.ndarray,
    map_cfg: dict[str, Any],
) -> dict[str, Any]:
    if len(grasps) == 0 or len(confidences) == 0:
        raise RuntimeError("No grasp candidates available for goal pose computation.")
    if len(grasps) != len(confidences):
        raise ValueError("Mismatch between number of grasps and confidence scores.")

    pgm, _ = read_pgm(Path(map_cfg["map_pgm"]))
    info = load_map_info(Path(map_cfg["map_yaml"]))
    unity_origin = tuple(float(v) for v in map_cfg["unity_map_origin"])
    offset = float(map_cfg["offset"])
    robot_radius = float(map_cfg["robot_radius"])
    white_th = int(map_cfg["white_threshold"])

    feasible_counts: dict[int, int] = {}
    feasible_best_by_group: dict[int, dict[str, float | np.ndarray | list[float] | bool | str]] = {}
    fallback_counts: dict[int, int] = {}
    fallback_best_by_group: dict[int, dict[str, float | np.ndarray | list[float] | bool | str]] = {}
    map_feasible_mask: list[bool] = []
    goal_unity_candidates: list[np.ndarray] = []
    goal_ros_candidates: list[list[float]] = []
    goal_pose_candidates_by_group: dict[int, list[dict[str, Any]]] = {}

    def append_group_goal_pose_candidate(
        *,
        group: int,
        grasp_index: int,
        confidence: float,
        world_pos: np.ndarray,
        rotation: np.ndarray,
        goal_unity: np.ndarray,
        goal_ros: list[float],
        map_feasible: bool,
        selection_mode: str,
    ) -> None:
        pose_matrix = np.eye(4, dtype=float)
        pose_matrix[:3, :3] = rotation
        pose_matrix[:3, 3] = world_pos
        goal_pose_candidates_by_group.setdefault(group, []).append(
            {
                "grasp_index": int(grasp_index),
                "confidence": float(confidence),
                "pose_unity": np.asarray(world_pos, dtype=float).tolist(),
                "pose_matrix_unity": pose_matrix.tolist(),
                "goal_pose_unity": np.asarray(goal_unity, dtype=float).tolist(),
                "goal_pose_ros_map": [float(goal_ros[0]), float(goal_ros[1])],
                "map_feasible": bool(map_feasible),
                "selection_mode": str(selection_mode),
            }
        )

    def update_group_candidate(
        counts: dict[int, int],
        best_by_group: dict[int, dict[str, float | np.ndarray | list[float] | bool | str]],
        *,
        group: int,
        confidence: float,
        candidate: np.ndarray,
        world_pos: np.ndarray,
        rotation: np.ndarray,
        map_feasible: bool,
        selection_mode: str,
    ) -> None:
        counts[group] = counts.get(group, 0) + 1
        current_best = best_by_group.get(group)
        if current_best is not None and confidence <= float(current_best["confidence"]):
            return

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
            "map_feasible": bool(map_feasible),
            "selection_mode": selection_mode,
        }

    for grasp_index, (grasp_matrix, confidence) in enumerate(zip(grasps, confidences)):
        position = np.array(grasp_matrix[:3, 3], dtype=float)
        rotation = np.array(grasp_matrix[:3, :3], dtype=float)
        group = grasp_orientation_group(grasp_matrix)
        confidence = float(confidence)
        world_pos = position + center_world

        approach = rotation[:, 2]
        norm = np.linalg.norm(approach)
        if norm == 0:
            append_group_goal_pose_candidate(
                group=group,
                grasp_index=grasp_index,
                confidence=confidence,
                world_pos=world_pos,
                rotation=rotation,
                goal_unity=np.full(3, np.nan, dtype=float),
                goal_ros=[float("nan"), float("nan")],
                map_feasible=False,
                selection_mode="invalid_approach",
            )
            map_feasible_mask.append(False)
            goal_unity_candidates.append(np.full(3, np.nan, dtype=float))
            goal_ros_candidates.append([float("nan"), float("nan")])
            continue

        direction = approach / norm
        if np.dot(direction, position) < 0:
            direction = -direction

        candidate = world_pos + direction * offset
        candidate_is_white = is_footprint_white(candidate, pgm, info, white_th, unity_origin, robot_radius)
        goal_ros = unity_to_map(float(candidate[0]), float(candidate[2]), unity_origin)
        map_feasible_mask.append(bool(candidate_is_white))
        goal_unity_candidates.append(np.asarray(candidate, dtype=float))
        goal_ros_candidates.append([float(goal_ros[0]), float(goal_ros[1])])
        selection_mode = "free_map" if candidate_is_white else "fallback_nonfree_map"
        append_group_goal_pose_candidate(
            group=group,
            grasp_index=grasp_index,
            confidence=confidence,
            world_pos=world_pos,
            rotation=rotation,
            goal_unity=candidate,
            goal_ros=[float(goal_ros[0]), float(goal_ros[1])],
            map_feasible=candidate_is_white,
            selection_mode=selection_mode,
        )

        update_group_candidate(
            fallback_counts,
            fallback_best_by_group,
            group=group,
            confidence=confidence,
            candidate=candidate,
            world_pos=world_pos,
            rotation=rotation,
            map_feasible=candidate_is_white,
            selection_mode=selection_mode,
        )

        if not candidate_is_white:
            continue

        update_group_candidate(
            feasible_counts,
            feasible_best_by_group,
            group=group,
            confidence=confidence,
            candidate=candidate,
            world_pos=world_pos,
            rotation=rotation,
            map_feasible=True,
            selection_mode="free_map",
        )

    using_map_fallback = False
    counts = feasible_counts
    best_by_group = feasible_best_by_group
    if not counts:
        if not fallback_counts:
            raise RuntimeError("No feasible goal pose candidates could be generated.")
        using_map_fallback = True
        counts = fallback_counts
        best_by_group = fallback_best_by_group

    sorted_groups = sorted(
        counts.keys(),
        key=lambda g: (-counts[g], -float(best_by_group[g]["confidence"]), g),
    )
    group_ranking = []
    for rank, group in enumerate(sorted_groups, start=1):
        info_group = best_by_group[group]
        grasp_goal_poses = sorted(
            goal_pose_candidates_by_group.get(group, []),
            key=lambda item: (-float(item.get("confidence", 0.0)), int(item.get("grasp_index", 0))),
        )
        group_ranking.append(
            {
                "rank": int(rank),
                "orientation_group": int(group),
                "best_confidence": float(info_group["confidence"]),
                "best_pose_unity": np.array(info_group["pose_unity"], dtype=float).tolist(),
                "best_pose_matrix_unity": np.array(info_group["pose_matrix_unity"], dtype=float).tolist(),
                "best_goal_pose_unity": np.array(info_group["goal_unity"], dtype=float).tolist(),
                "best_goal_pose_ros_map": [float(info_group["goal_ros"][0]), float(info_group["goal_ros"][1])],
                "map_feasible": bool(info_group["map_feasible"]),
                "selection_mode": str(info_group["selection_mode"]),
                "num_grasp_goal_poses": int(len(grasp_goal_poses)),
                "num_map_feasible_grasp_goal_poses": int(
                    sum(1 for item in grasp_goal_poses if bool(item.get("map_feasible", False)))
                ),
                "grasp_goal_poses": grasp_goal_poses,
            }
        )

    return {
        "group_ranking": group_ranking,
        "map_feasible_mask": np.asarray(map_feasible_mask, dtype=bool),
        "goal_unity_candidates": np.asarray(goal_unity_candidates, dtype=float),
        "goal_ros_candidates": np.asarray(goal_ros_candidates, dtype=float),
        "used_map_fallback": bool(using_map_fallback),
        "using_map_fallback": bool(using_map_fallback),
    }
