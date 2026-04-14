"""
base_pose_sampler.py — Sample candidate robot base poses from a navigable map
and rank them by suitability for reaching a specified grasp target.

Algorithm
---------
1. Take all free cells from the occupancy map (Unity x-z frame).
2. Filter by horizontal distance to target: [min_dist_m, max_dist_m].
3. Compute the **approach direction** check:
     - If the target rotation (approach axis) is known, keep only cells whose
       base→target direction is within ±heading_tolerance_deg of the target's
       approach axis projected onto the x-z floor plane.
     - If no approach axis is provided, skip this filter (all cells pass).
4. Down-sample to at most ``num_samples`` cells using stratified random sampling.
5. Sort by: (a) approach-axis alignment error (best first), (b) distance (closest first).

Coordinate conventions
----------------------
Unity world:   X = right, Y = up, Z = forward  (floor = X-Z plane)
Map frame:     X = right, Y = up  (converted to Unity by map_loader)

The "approach axis" of the grasp pose is the local +X column of the target's
3×3 rotation matrix expressed in Unity world frame.  The robot should approach
FROM the direction opposite to this axis so that its end-effector reaches the
target along the approach axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BasePoseCandidate:
    """A single candidate robot base position with derived quantities."""

    unity_x: float
    unity_z: float
    heading_rad: float          # yaw in Unity x-z plane (atan2 to target)
    distance_to_target_m: float # horizontal 2-D distance on floor
    approach_error_deg: float   # |angle| vs target approach axis (0 = perfect)


def sample_candidate_base_poses(
    target_unity_xz: Sequence[float],
    free_cells_unity_xz: np.ndarray,
    *,
    target_approach_dir_unity_xz: Sequence[float] | None = None,
    min_distance_m: float = 0.4,
    max_distance_m: float = 2.0,
    heading_tolerance_deg: float = 10.0,
    num_samples: int = 200,
    rng_seed: int = 42,
) -> list[BasePoseCandidate]:
    """Sample and rank candidate base poses.

    Parameters
    ----------
    target_unity_xz:
        ``[tx, tz]`` — target grasp position on the Unity floor plane.
    free_cells_unity_xz:
        ``(N, 2)`` array of navigable cell centres ``[x, z]`` in Unity world.
    target_approach_dir_unity_xz:
        Optional normalised ``[ax, az]`` vector — the Unity-world direction
        along which the gripper approaches the target (= column 0 of the grasp
        rotation matrix converted to Unity).  When provided, only cells whose
        base→target direction is within ±``heading_tolerance_deg`` of this
        vector are kept.
    min_distance_m:
        Minimum 2-D floor distance from base to target (metres).
    max_distance_m:
        Maximum 2-D floor distance from base to target (metres).
    heading_tolerance_deg:
        Angular tolerance for the approach-direction filter.
    num_samples:
        Maximum number of candidates to return before IK evaluation.
    rng_seed:
        RNG seed for reproducibility.

    Returns
    -------
    list[BasePoseCandidate]
        Filtered, ranked candidates. Primary key: approach_error_deg ↑;
        secondary key: distance_to_target_m ↑ (nearer is better).
    """
    cells = np.asarray(free_cells_unity_xz, dtype=np.float32)
    if cells.ndim != 2 or cells.shape[1] != 2:
        raise ValueError(
            f"free_cells_unity_xz must be shape (N, 2), got {cells.shape}."
        )
    if len(cells) == 0:
        return []

    tx, tz = float(target_unity_xz[0]), float(target_unity_xz[1])

    # ── 1. Distance filter ────────────────────────────────────────────────────
    dx = cells[:, 0] - tx
    dz = cells[:, 1] - tz
    dist = np.sqrt(dx * dx + dz * dz)
    dist_mask = (dist >= float(min_distance_m)) & (dist <= float(max_distance_m))
    cells = cells[dist_mask]
    dx = dx[dist_mask]
    dz = dz[dist_mask]
    dist = dist[dist_mask]

    if len(cells) == 0:
        return []

    # Heading: direction from each base toward the target (Unity x-z atan2)
    headings = np.arctan2(tz - cells[:, 1], tx - cells[:, 0])  # angle in Unity x-z

    # ── 2. Approach-direction filter ─────────────────────────────────────────
    if target_approach_dir_unity_xz is not None:
        ax, az = float(target_approach_dir_unity_xz[0]), float(target_approach_dir_unity_xz[1])
        approach_norm = math.sqrt(ax * ax + az * az)
        if approach_norm > 1e-8:
            ax /= approach_norm
            az /= approach_norm
            # Unit vector from each base toward target
            base_to_target_x = (tx - cells[:, 0]) / dist
            base_to_target_z = (tz - cells[:, 1]) / dist
            # Dot product → cosine of angle between base→target and approach dir
            dot = base_to_target_x * ax + base_to_target_z * az
            dot = np.clip(dot, -1.0, 1.0)
            approach_error_rad = np.arccos(dot)
            approach_error_deg = np.degrees(approach_error_rad)
            approach_mask = approach_error_deg <= float(heading_tolerance_deg)
            cells = cells[approach_mask]
            headings = headings[approach_mask]
            dist = dist[approach_mask]
            approach_error_deg = approach_error_deg[approach_mask]
        else:
            approach_error_deg = np.zeros(len(cells), dtype=np.float32)
    else:
        approach_error_deg = np.zeros(len(cells), dtype=np.float32)

    if len(cells) == 0:
        return []

    # ── 3. Down-sample ────────────────────────────────────────────────────────
    n = len(cells)
    if n > int(num_samples):
        rng = np.random.default_rng(int(rng_seed))
        indices = rng.choice(n, size=int(num_samples), replace=False)
        cells = cells[indices]
        headings = headings[indices]
        dist = dist[indices]
        approach_error_deg = approach_error_deg[indices]

    # ── 4. Sort: approach error ↑ then distance ↑ ────────────────────────────
    sort_key = approach_error_deg * 1000.0 + dist   # primary: error, secondary: dist
    order = np.argsort(sort_key)
    cells = cells[order]
    headings = headings[order]
    dist = dist[order]
    approach_error_deg = approach_error_deg[order]

    candidates: list[BasePoseCandidate] = []
    for i in range(len(cells)):
        candidates.append(
            BasePoseCandidate(
                unity_x=float(cells[i, 0]),
                unity_z=float(cells[i, 1]),
                heading_rad=float(headings[i]),
                distance_to_target_m=float(dist[i]),
                approach_error_deg=float(approach_error_deg[i]),
            )
        )
    return candidates
