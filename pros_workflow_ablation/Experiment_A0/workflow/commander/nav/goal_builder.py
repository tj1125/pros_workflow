from __future__ import annotations

import math
import time
from typing import Any, Dict


def goal_pose_db_from_item_info(
    item_info: dict[str, Any],
    current_rank: int,
) -> dict[str, Any]:
    ranks: dict[str, dict[str, Any]] = {}
    for group in item_info.get("group_ranking", []) or []:
        rank = int(group.get("rank", len(ranks) + 1) or len(ranks) + 1)
        ranks[str(rank)] = dict(group)
    return {
        "target_instance_key": item_info.get("target_instance_key", ""),
        "center_world": item_info.get("center_world", []),
        "current_goal_rank": current_rank,
        "rank_order": [int(k) for k in ranks.keys()],
        "ranks": ranks,
        "updated_at": time.time(),
    }


def target_center_world_to_map_xy(
    item_info: Dict[str, Any],
    *,
    ros_map_origin_unity_x: float,
    ros_map_origin_unity_z: float,
) -> tuple[float | None, float | None]:
    center_world = item_info.get("center_world", [])
    if not isinstance(center_world, list) or len(center_world) < 2:
        return None, None
    frame = str(
        item_info.get("center_world_coordinate_frame", "unity_world")
        or "unity_world"
    ).lower()
    try:
        if frame in {"ros_map", "map", "ros_map_xy"}:
            return float(center_world[0]), float(center_world[1])
        if len(center_world) < 3:
            return None, None
        unity_x = float(center_world[0])
        unity_z = float(center_world[2])
    except (TypeError, ValueError):
        return None, None
    return ros_map_origin_unity_z - unity_z, unity_x - ros_map_origin_unity_x


def goal_pose_from_ros_map(
    item_info: Dict[str, Any],
    goal_pose_ros: Any,
    *,
    ros_map_origin_unity_x: float,
    ros_map_origin_unity_z: float,
) -> tuple[Dict[str, Any], str]:
    if not isinstance(goal_pose_ros, list) or len(goal_pose_ros) < 2:
        return {}, "missing goal_pose_ros_map"
    try:
        goal_x = float(goal_pose_ros[0])
        goal_y = float(goal_pose_ros[1])
    except (TypeError, ValueError):
        return {}, f"invalid goal_pose_ros_map={goal_pose_ros}"

    yaw = 0.0
    target_map_x, target_map_y = target_center_world_to_map_xy(
        item_info,
        ros_map_origin_unity_x=ros_map_origin_unity_x,
        ros_map_origin_unity_z=ros_map_origin_unity_z,
    )
    if target_map_x is not None and target_map_y is not None:
        yaw = math.atan2(target_map_y - goal_y, target_map_x - goal_x)

    goal_pose = {
        "x": goal_x,
        "y": goal_y,
        "z": 0.0,
        "qx": 0.0,
        "qy": 0.0,
        "qz": math.sin(yaw / 2.0),
        "qw": math.cos(yaw / 2.0),
        "yaw": yaw,
    }
    if target_map_x is not None and target_map_y is not None:
        goal_pose["face_target_x"] = target_map_x
        goal_pose["face_target_y"] = target_map_y
    return goal_pose, ""


def goal_pose_for_rank(
    item_info: Dict[str, Any],
    rank: int,
    *,
    ros_map_origin_unity_x: float,
    ros_map_origin_unity_z: float,
) -> tuple[Dict[str, Any], str]:
    groups = item_info.get("group_ranking", []) or []
    idx = rank - 1
    if idx < 0 or idx >= len(groups):
        return {}, f"rank={rank} out of range"

    group = groups[idx] or {}
    goal, err = goal_pose_from_ros_map(
        item_info,
        group.get("best_goal_pose_ros_map", []),
        ros_map_origin_unity_x=ros_map_origin_unity_x,
        ros_map_origin_unity_z=ros_map_origin_unity_z,
    )
    if err:
        return {}, f"rank={rank} {err}"

    goal.update(
        {
            "goal_rank": rank,
            "goal_pose_index": 0,
            "goal_pose_source": "rank_best",
            "orientation_group": group.get("orientation_group"),
            "grasp_confidence": group.get("best_confidence"),
            "map_feasible": group.get("map_feasible"),
            "selection_mode": group.get("selection_mode", ""),
        }
    )
    return goal, ""
