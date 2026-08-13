"""Experiment A2 ablation: ranked goal poses → single nearest goal pose.

The full system asks the no-SAM3D item-info service for several orientation-group
candidate docking poses (``group_ranking``) and lets the workflow try them in rank
order, switching to the next ranked goal pose when the VLM decides to (or when the
pre-move backstop forces it after repeated before-the-car-moves failures) — see
``major_nav_node`` and ``_apply_pre_move_backstop`` in :mod:`commander.flows.pick`.

This ablation removes that capability. When enabled, the workflow keeps only ONE
candidate — the goal pose whose docking position is closest (Euclidean distance in
the ROS map XY plane) to the target object — and discards the rest. Perception (the
A2A call itself) is left untouched, so the only variable under study is "multiple
candidate docking points + ranking". Because only a single candidate survives, the
existing rank-fallback machinery naturally exhausts after the first attempt:
``goal_pose_for_rank(rank=2)`` is out of range, so ``major_nav_node`` reports
``MAJOR_NAV_EXHAUSTED`` and neither the VLM nor the backstop can switch viewpoint.

The switch is config-driven (env ``ABLATION_SINGLE_NEAREST_GOAL``) and defaults to
OFF, so the un-ablated baseline and the existing contract tests are unaffected.
Experiment_A2's ``.env`` turns it ON.
"""

from __future__ import annotations

import math
import os
from typing import Any

ENV_FLAG = "ABLATION_SINGLE_NEAREST_GOAL"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def single_nearest_goal_enabled() -> bool:
    """Return True when the A2 single-nearest-goal ablation is active."""
    return _env_flag(ENV_FLAG, False)


def _docking_distance_to_target(group: dict[str, Any], target_xy: tuple[float, float]) -> float:
    """Euclidean distance (ROS map XY) between a group's docking pose and the target."""
    pose = group.get("best_goal_pose_ros_map")
    if not isinstance(pose, (list, tuple)) or len(pose) < 2:
        return float("inf")
    try:
        goal_x, goal_y = float(pose[0]), float(pose[1])
    except (TypeError, ValueError):
        return float("inf")
    return math.hypot(goal_x - target_xy[0], goal_y - target_xy[1])


def collapse_to_nearest_goal(
    group_ranking: list[dict[str, Any]],
    target_xy: tuple[float | None, float | None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reduce a ranked goal-pose list to the single nearest-to-target candidate.

    Args:
        group_ranking: the full ``group_ranking`` produced by the item-info service.
        target_xy: target object centre in ROS map XY (``(None, None)`` if unknown).

    Returns:
        ``(collapsed_group_ranking, ablation_meta)``. The surviving group is
        relabeled ``rank=1`` so the rest of the workflow behaves as if only one
        candidate ever existed. ``ablation_meta`` records what the perception
        produced and which candidate was kept, for the experiment report.
    """
    groups = [group for group in (group_ranking or []) if isinstance(group, dict)]
    meta: dict[str, Any] = {
        "single_nearest_goal": True,
        "perception_group_count": len(groups),
        "selection_metric": "min_euclidean_docking_to_target_ros_map_xy",
    }
    if not groups:
        meta["applied"] = False
        meta["note"] = "no perception goal candidates to collapse"
        return [], meta

    target_x, target_y = target_xy
    if target_x is None or target_y is None:
        chosen = min(groups, key=lambda group: int(group.get("rank", 1) or 1))
        meta["selection_metric"] = "fallback_perception_rank_1_no_target_xy"
        selected_distance: float | None = None
    else:
        chosen = min(groups, key=lambda group: _docking_distance_to_target(group, (target_x, target_y)))
        distance = _docking_distance_to_target(chosen, (target_x, target_y))
        selected_distance = round(distance, 4) if math.isfinite(distance) else None

    selected_rank = int(chosen.get("rank", 1) or 1)
    meta.update(
        {
            "applied": True,
            "selected_perception_rank": selected_rank,
            "selected_orientation_group": chosen.get("orientation_group"),
            "selected_distance_m": selected_distance,
            "selected_best_goal_pose_ros_map": chosen.get("best_goal_pose_ros_map"),
            "selected_best_confidence": chosen.get("best_confidence"),
            "dropped_group_count": max(0, len(groups) - 1),
        }
    )

    collapsed = dict(chosen)
    collapsed["rank"] = 1
    collapsed["ablation_single_nearest_goal"] = True
    collapsed["ablation_original_rank"] = selected_rank
    if selected_distance is not None:
        collapsed["ablation_docking_distance_m"] = selected_distance
    return [collapsed], meta
