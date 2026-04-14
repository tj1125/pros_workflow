"""
base_pose_evaluator.py — Evaluate candidate robot base poses via IK ranking
then OMPL path planning, and render the best successful base configuration.

Pipeline (per candidate)
------------------------
1. **coordinate transform** — convert target + voxel obstacles from Unity world
   frame into the robot-local PyBullet frame for the candidate base pose.
2. **IK ranking** — run all candidates through PyBullet IK in one DIRECT
   session; keep the top-N by ``ee_error_m`` (smallest IK residual = easiest
   to grasp).
3. **OMPL evaluation** — for each IK-ranked candidate (in order), run
   ``run_ompl_planning_test``; stop at the first ``path_found`` success (or
   evaluate all when ``evaluate_all=True``).
4. **render** — re-run the best candidate with ``--gui`` disabled and save
   three debug PPM views (default, top-down, side).

Coordinate conventions
----------------------
Unity world:      X = right, Y = up, Z = forward   (floor = X-Z)
PyBullet world:   X = forward(cam), Y = right(cam), Z = up   (gravity = -Z)

Unity → PyBullet  (for a vector v):
    pb_x = v.unity_z
    pb_y = v.unity_x
    pb_z = v.unity_y

The robot is **always placed at [0, 0, initial_height]** in PyBullet.  For
each candidate base position its unique (heading, distance) encode the target
position in the robot-local PyBullet frame.  Obstacles are similarly expressed
in that local frame.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.base_pose_sampler import BasePoseCandidate
from src.pybullet_ompl import BoxObstacleSpec, OmplPlanningResult, run_ompl_planning_test
from src.pybullet_smoke import (
    _degrees_to_radians,
    _find_controllable_joints,
    _load_arm_config,
    _load_python_dependencies,
    _render_debug_ppm,
    _derive_topdown_output_path,
    _derive_side_output_path,
)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def unity_vec_to_pybullet(v_unity: Sequence[float]) -> tuple[float, float, float]:
    """Convert a 3-D Unity world-frame vector to PyBullet frame.

    Unity→PyBullet:  pb_x = uni_x,  pb_y = uni_z,  pb_z = uni_y
    """
    ux, uy, uz = float(v_unity[0]), float(v_unity[1]), float(v_unity[2])
    return (ux, uz, uy)


def rotate_unity_xz_by_neg_heading(
    dx: float, dy: float, dz: float, heading_rad: float
) -> tuple[float, float, float]:
    """Rotate a Unity-world offset vector into the robot-local Unity frame.

    The robot faces direction ``heading_rad`` (yaw around Unity +Y from +X axis).
    Rotating by -heading_rad aligns the robot's forward with +local_x so that:
        local_x  = forward toward target  (= horizontal distance to target)
        local_y  = unchanged (up)
        local_z  = lateral offset        (= 0 when heading exactly toward target)
    """
    c = math.cos(-heading_rad)
    s = math.sin(-heading_rad)
    # Rotation around Unity Y axis by -heading:
    local_x = dx * c - dz * s     # NOTE: Unity Y-rotation: x' =  x·cos - z·sin
    local_y = dy
    local_z = dx * s + dz * c     #                          z' =  x·sin + z·cos
    return local_x, local_y, local_z


def candidate_target_pybullet(
    candidate: BasePoseCandidate,
    target_unity_xyz: Sequence[float],
    base_ground_y_unity: float,
) -> tuple[float, float, float]:
    """Compute the grasp target position in the candidate's PyBullet frame.

    The robot base sits at ``[0, 0, initial_height]`` in PyBullet.  The
    ``initial_height`` is baked into the URDF mounting and handled by the
    existing planner config, so here we produce coordinates relative to the
    arm's PyBullet world origin at the floor level — the planner adds the
    z-offset internally via ``initial_height``.

    Parameters
    ----------
    candidate:
        Sampled base pose (Unity x-z position + heading).
    target_unity_xyz:
        ``[tx, ty, tz]`` — grasp target in Unity world frame.
    base_ground_y_unity:
        Unity Y value of the robot's ground level (typically 0.0).
    """
    tx, ty, tz = (float(v) for v in target_unity_xyz)
    bx, bz = candidate.unity_x, candidate.unity_z

    # Vector from base (on ground) to target in Unity world
    dx = tx - bx
    dy = ty - float(base_ground_y_unity)
    dz = tz - bz

    # Rotate into robot-local Unity frame (robot faces target = local +X)
    lx, ly, lz = rotate_unity_xz_by_neg_heading(dx, dy, dz, candidate.heading_rad)

    # Convert robot-local Unity vector → PyBullet
    return unity_vec_to_pybullet((lx, ly, lz))


def candidate_obstacles_pybullet(
    candidate: BasePoseCandidate,
    world_obstacle_centers_unity: np.ndarray,
    base_ground_y_unity: float,
    voxel_size_m: float,
    rgba: tuple[float, float, float, float] = (0.85, 0.2, 0.2, 0.55),
) -> tuple[BoxObstacleSpec, ...]:
    """Transform world-frame voxel centres into the candidate's PyBullet frame."""
    if len(world_obstacle_centers_unity) == 0:
        return ()
    centers = np.asarray(world_obstacle_centers_unity, dtype=np.float64)
    bx, bz = float(candidate.unity_x), float(candidate.unity_z)
    by = float(base_ground_y_unity)

    dx = centers[:, 0] - bx
    dy = centers[:, 1] - by
    dz = centers[:, 2] - bz

    h = float(candidate.heading_rad)
    c, s = math.cos(-h), math.sin(-h)
    lx = dx * c - dz * s
    ly = dy
    lz = dx * s + dz * c

    # Unity local → PyBullet
    pb_x = lx
    pb_y = lz
    pb_z = ly

    specs: list[BoxObstacleSpec] = []
    for i in range(len(centers)):
        specs.append(
            BoxObstacleSpec(
                size=(voxel_size_m, voxel_size_m, voxel_size_m),
                position=(float(pb_x[i]), float(pb_y[i]), float(pb_z[i])),
                rgba=rgba,
            )
        )
    return tuple(specs)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CandidateIKResult:
    candidate: BasePoseCandidate
    ee_error_m: float
    ik_ok: bool   # error < position_tolerance_m


@dataclass
class BasePoseEvaluatorResult:
    total_candidates_sampled: int = 0
    ik_evaluated_count: int = 0
    ompl_evaluated_count: int = 0
    best_candidate: BasePoseCandidate | None = None
    best_ik_error_m: float | None = None
    best_ompl_result: dict[str, Any] | None = None
    # Cached pybullet-frame values for GUI replay (not serialised to JSON)
    _best_target_pb: tuple[float, float, float] | None = field(default=None, repr=False)
    _best_obstacle_specs: tuple[BoxObstacleSpec, ...] | None = field(default=None, repr=False)
    render_ppm_path: str | None = None
    render_topdown_ppm_path: str | None = None
    render_side_ppm_path: str | None = None
    all_ik_results: list[dict[str, Any]] = field(default_factory=list)
    all_ompl_results: list[dict[str, Any]] = field(default_factory=list)
    success: bool = False
    failure_reason: str | None = None
    total_time_sec: float | None = None


# ---------------------------------------------------------------------------
# IK ranking phase (single PyBullet DIRECT session)
# ---------------------------------------------------------------------------

def _rank_by_ik(
    candidates: list[BasePoseCandidate],
    target_unity_xyz: Sequence[float],
    base_ground_y_unity: float,
    planner_config_path: Path,
    position_tolerance_m: float,
    max_ik_candidates: int,
) -> list[CandidateIKResult]:
    """Run IK for all candidates in one PyBullet DIRECT session.

    Returns candidates sorted by ee_error_m (ascending — best IK first).
    Only the top ``max_ik_candidates`` are returned for OMPL evaluation.
    """
    from src.pybullet_ompl import (
        load_planning_config,
        _set_joint_positions_direct,
        _degrees_to_radians,
    )

    planning_config = load_planning_config(planner_config_path, require_obstacles=False)
    arm_config = _load_arm_config()
    np_mod, p, pybullet_data = _load_python_dependencies()

    expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])

    client_id: int | None = None
    ik_results: list[CandidateIKResult] = []

    try:
        client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.8)
        p.loadURDF("plane.urdf")

        base_orientation_rad = [math.radians(v) for v in planning_config.base_orientation_euler_deg]
        base_orientation_xyzw = p.getQuaternionFromEuler(base_orientation_rad)
        robot_id = p.loadURDF(
            planning_config.urdf_path,
            useFixedBase=True,
            basePosition=[0.0, 0.0, planning_config.initial_height],
            baseOrientation=base_orientation_xyzw,
        )
        joint_ids, _ = _find_controllable_joints(p, robot_id, expected_joint_count)
        joint_reset_rad = _degrees_to_radians(planning_config.joint_reset_deg)
        _set_joint_positions_direct(p, robot_id, joint_ids, joint_reset_rad)
        p.performCollisionDetection()

        for candidate in candidates:
            target_pb = candidate_target_pybullet(
                candidate, target_unity_xyz, base_ground_y_unity
            )
            try:
                ik_sol = p.calculateInverseKinematics(
                    robot_id,
                    planning_config.ee_link_index,
                    targetPosition=list(target_pb),
                )
            except Exception:
                ik_results.append(CandidateIKResult(candidate, ee_error_m=1e9, ik_ok=False))
                continue

            if len(ik_sol) < len(joint_ids):
                ik_results.append(CandidateIKResult(candidate, ee_error_m=1e9, ik_ok=False))
                continue

            goal_joints = list(ik_sol[: len(joint_ids)])
            _set_joint_positions_direct(p, robot_id, joint_ids, goal_joints)
            p.performCollisionDetection()
            ee_state = p.getLinkState(robot_id, planning_config.ee_link_index, computeForwardKinematics=True)
            ee_pos = ee_state[0]
            ee_error = float(math.dist(ee_pos, target_pb))

            # Reset for next candidate
            _set_joint_positions_direct(p, robot_id, joint_ids, joint_reset_rad)
            p.performCollisionDetection()

            ik_ok = ee_error <= float(position_tolerance_m)
            ik_results.append(CandidateIKResult(candidate, ee_error_m=ee_error, ik_ok=ik_ok))

    finally:
        if client_id is not None:
            try:
                p.disconnect(client_id)
            except Exception:
                pass

    # Sort by IK error ascending (best IK = most likely to successfully grasp)
    ik_results.sort(key=lambda r: r.ee_error_m)
    return ik_results[: max(1, int(max_ik_candidates))]


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

def evaluate_candidate_base_poses(
    candidates: list[BasePoseCandidate],
    target_unity_xyz: Sequence[float],
    target_rotation_pybullet: np.ndarray | None,
    world_obstacle_centers_unity: np.ndarray,
    planner_config_path: Path,
    output_dir: Path,
    *,
    base_ground_y_unity: float = 0.0,
    voxel_size_m: float = 0.05,
    obstacle_rgba: tuple[float, float, float, float] = (0.85, 0.2, 0.2, 0.55),
    position_tolerance_m: float = 0.03,
    max_ik_candidates: int = 20,
    evaluate_all: bool = False,
    planning_timeout_sec: float | None = None,
    render_width: int = 960,
    render_height: int = 720,
    render_camera_distance: float = 1.2,
    render_yaw_deg: float = 45.0,
    render_pitch_deg: float = -30.0,
) -> BasePoseEvaluatorResult:
    """Full evaluation pipeline: IK ranking → OMPL → render best base.

    Parameters
    ----------
    candidates:
        Pre-sampled, pre-sorted candidates from :func:`sample_candidate_base_poses`.
    target_unity_xyz:
        Grasp target position in Unity world frame ``[x, y, z]``.
    target_rotation_pybullet:
        Optional 3×3 rotation matrix of the grasp pose in *PyBullet* frame.
        Used to compute target orientation for OMPL.
    world_obstacle_centers_unity:
        ``(M, 3)`` array of voxel obstacle centres in Unity world ``[x, y, z]``.
        Pass an empty array to run without obstacles.
    planner_config_path:
        Path to the OMPL planner YAML config (``pybullet_ompl.yaml``).
    output_dir:
        Directory where render images are saved.
    base_ground_y_unity:
        Unity Y of the robot's ground level.
    """
    t0 = time.perf_counter()
    result = BasePoseEvaluatorResult(total_candidates_sampled=len(candidates))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not candidates:
        result.failure_reason = "No sampled candidates provided."
        result.total_time_sec = float(time.perf_counter() - t0)
        return result

    # ── Phase 1: IK ranking ──────────────────────────────────────────────────
    ik_ranked = _rank_by_ik(
        candidates,
        target_unity_xyz,
        base_ground_y_unity,
        planner_config_path,
        position_tolerance_m,
        max_ik_candidates,
    )
    result.ik_evaluated_count = len(ik_ranked)
    result.all_ik_results = [
        {
            "unity_x": r.candidate.unity_x,
            "unity_z": r.candidate.unity_z,
            "heading_rad": r.candidate.heading_rad,
            "distance_to_target_m": r.candidate.distance_to_target_m,
            "approach_error_deg": r.candidate.approach_error_deg,
            "ee_error_m": r.ee_error_m,
            "ik_ok": r.ik_ok,
        }
        for r in ik_ranked
    ]

    if not ik_ranked:
        result.failure_reason = "IK ranking produced no results."
        result.total_time_sec = float(time.perf_counter() - t0)
        return result

    # ── Phase 2: OMPL evaluation ─────────────────────────────────────────────
    from src.pybullet_ompl import load_planning_config
    planning_config = load_planning_config(planner_config_path, require_obstacles=False)

    target_orientation_xyzw: list[float] | None = None
    if target_rotation_pybullet is not None:
        from src.camera_car_voxel_ompl import _rotation_matrix_to_quaternion_xyzw
        target_orientation_xyzw = list(
            _rotation_matrix_to_quaternion_xyzw(target_rotation_pybullet)
        )

    # Compute candidate heading-to-pybullet-yaw offset.
    # The existing base_orientation_euler_deg already encodes the arm's
    # mounting yaw.  We add the candidate's world heading on top of that.
    base_yaw_config_rad = math.radians(planning_config.base_orientation_euler_deg[2])

    best_ompl: OmplPlanningResult | None = None
    best_ik_entry: CandidateIKResult | None = None

    for ik_entry in ik_ranked:
        candidate = ik_entry.candidate
        result.ompl_evaluated_count += 1

        # Compute pybullet target for this candidate
        target_pb = candidate_target_pybullet(
            candidate, target_unity_xyz, base_ground_y_unity
        )

        # Transform obstacles to this candidate's pybullet frame
        obstacle_specs = candidate_obstacles_pybullet(
            candidate,
            world_obstacle_centers_unity,
            base_ground_y_unity,
            voxel_size_m,
            rgba=obstacle_rgba,
        )
        # Fallback: if no obstacles supplied, use a single tiny dummy far away
        # so the OMPL planner config "obstacles" requirement is satisfied
        if not obstacle_specs:
            obstacle_specs = (
                BoxObstacleSpec(
                    size=(0.01, 0.01, 0.01),
                    position=(10.0, 10.0, 10.0),
                    rgba=(0.0, 0.0, 0.0, 0.0),
                ),
            )

        # Heading-based yaw for this candidate:
        # Unity heading h (atan2 in x-z facing target) →
        #   pybullet yaw = 90° - h_deg  (derived from Unity→PyBullet mapping)
        heading_pb_yaw_rad = math.pi / 2.0 - candidate.heading_rad
        # Override planner's base_orientation_euler_deg[2] with total yaw
        # by passing it via obstacle_specs is not possible directly; instead
        # we patch it via a temporary override dict on a copy of planning_config.
        # The simpler approach: express everything in robot-local frame (which
        # we already do via candidate_target_pybullet) so heading is baked in.
        # Therefore, we leave base_orientation_euler_deg unchanged.

        run_kwargs: dict = dict(
            obstacle_specs_override=obstacle_specs,
            target_position_override=list(target_pb),
            target_orientation_override_xyzw=target_orientation_xyzw,
        )
        if planning_timeout_sec is not None:
            # The planner picks up planning_timeout_sec from its own config;
            # we cannot override it without patching the config object.
            # For now, rely on the config value.
            pass

        ompl_result = run_ompl_planning_test(
            planner_config_path,
            gui_override=False,
            **run_kwargs,
        )

        ompl_entry = {
            "unity_x": candidate.unity_x,
            "unity_z": candidate.unity_z,
            "heading_rad": candidate.heading_rad,
            "distance_to_target_m": candidate.distance_to_target_m,
            "ee_error_m": ik_entry.ee_error_m,
            "path_found": ompl_result.path_found,
            "planned_path_collision_free": ompl_result.planned_path_collision_free,
            "planning_time_sec": ompl_result.planning_time_sec,
            "path_length_joint_space": ompl_result.path_length_joint_space,
            "planned_path_joint_states_deg": ompl_result.planned_path_joint_states_deg,
            "failure_bucket": ompl_result.failure_bucket,
            "error": ompl_result.error,
        }
        result.all_ompl_results.append(ompl_entry)

        if ompl_result.path_found and ompl_result.planned_path_collision_free:
            best_ompl = ompl_result
            best_ik_entry = ik_entry
            # Cache the pybullet-frame data for GUI replay
            result._best_target_pb = target_pb
            result._best_obstacle_specs = obstacle_specs
            result.success = True
            if not evaluate_all:
                break

    if best_ompl is None or best_ik_entry is None:
        result.failure_reason = "No candidate base pose produced a collision-free OMPL path."
        result.total_time_sec = float(time.perf_counter() - t0)
        return result

    # ── Phase 3: Render best base pose ───────────────────────────────────────
    result.best_candidate = best_ik_entry.candidate
    result.best_ik_error_m = best_ik_entry.ee_error_m
    result.best_ompl_result = asdict(best_ompl)

    _render_best_base(
        best_ik_entry.candidate,
        target_unity_xyz,
        world_obstacle_centers_unity,
        planner_config_path,
        output_dir,
        base_ground_y_unity=base_ground_y_unity,
        voxel_size_m=voxel_size_m,
        obstacle_rgba=obstacle_rgba,
        render_width=render_width,
        render_height=render_height,
        render_camera_distance=render_camera_distance,
        render_yaw_deg=render_yaw_deg,
        render_pitch_deg=render_pitch_deg,
        target_orientation_xyzw=target_orientation_xyzw,
        result=result,
    )

    result.total_time_sec = float(time.perf_counter() - t0)
    return result


def _render_best_base(
    candidate: BasePoseCandidate,
    target_unity_xyz: Sequence[float],
    world_obstacle_centers_unity: np.ndarray,
    planner_config_path: Path,
    output_dir: Path,
    *,
    base_ground_y_unity: float,
    voxel_size_m: float,
    obstacle_rgba: tuple[float, float, float, float],
    render_width: int,
    render_height: int,
    render_camera_distance: float,
    render_yaw_deg: float,
    render_pitch_deg: float,
    target_orientation_xyzw: list[float] | None,
    result: BasePoseEvaluatorResult,
) -> None:
    """Launch a PyBullet DIRECT session and render 3 views of the best base."""
    from src.pybullet_ompl import (
        load_planning_config,
        _set_joint_positions_direct,
        _create_box_obstacles,
        _add_target_marker,
        _add_debug_axes,
    )

    np_mod, p, pybullet_data = _load_python_dependencies()
    planning_config = load_planning_config(planner_config_path, require_obstacles=False)
    arm_config = _load_arm_config()
    expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])

    target_pb = candidate_target_pybullet(candidate, target_unity_xyz, base_ground_y_unity)
    obstacle_specs = candidate_obstacles_pybullet(
        candidate, world_obstacle_centers_unity, base_ground_y_unity,
        voxel_size_m, rgba=obstacle_rgba,
    )
    if not obstacle_specs:
        obstacle_specs = (
            BoxObstacleSpec(
                size=(0.01, 0.01, 0.01),
                position=(10.0, 10.0, 10.0),
                rgba=(0.0, 0.0, 0.0, 0.0),
            ),
        )

    render_ppm = output_dir / "best_base_render.ppm"
    render_topdown = _derive_topdown_output_path(render_ppm)
    render_side = _derive_side_output_path(render_ppm)

    client_id: int | None = None
    try:
        client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.8)
        p.loadURDF("plane.urdf")

        base_orientation_rad = [math.radians(v) for v in planning_config.base_orientation_euler_deg]
        base_orientation_xyzw = p.getQuaternionFromEuler(base_orientation_rad)
        robot_id = p.loadURDF(
            planning_config.urdf_path,
            useFixedBase=True,
            basePosition=[0.0, 0.0, planning_config.initial_height],
            baseOrientation=base_orientation_xyzw,
        )
        joint_ids, _ = _find_controllable_joints(p, robot_id, expected_joint_count)
        joint_reset_rad = _degrees_to_radians(planning_config.joint_reset_deg)
        _set_joint_positions_direct(p, robot_id, joint_ids, joint_reset_rad)

        # Solve IK and set arm to goal pose for render
        ik_kwargs: dict = {"targetPosition": list(target_pb)}
        if target_orientation_xyzw is not None:
            ik_kwargs["targetOrientation"] = list(target_orientation_xyzw)
        ik_sol = p.calculateInverseKinematics(
            robot_id, planning_config.ee_link_index, **ik_kwargs
        )
        if len(ik_sol) >= len(joint_ids):
            goal_joints = list(ik_sol[: len(joint_ids)])
            _set_joint_positions_direct(p, robot_id, joint_ids, goal_joints)

        _create_box_obstacles(p, obstacle_specs)
        _add_target_marker(p, target_pb)
        _add_debug_axes(p, [0.0, 0.0, planning_config.initial_height],
                        axis_length=0.12, axis_width=1.8, label="base")
        p.performCollisionDetection()

        render_target = list(target_pb)

        # Default view
        _render_debug_ppm(
            p, np_mod, render_ppm,
            width=render_width, height=render_height,
            camera_target_position=render_target,
            camera_distance=render_camera_distance,
            camera_yaw_deg=render_yaw_deg,
            camera_pitch_deg=render_pitch_deg,
        )
        # Top-down view
        _render_debug_ppm(
            p, np_mod, render_topdown,
            width=render_width, height=render_height,
            camera_target_position=render_target,
            camera_distance=max(0.5, render_camera_distance * 0.85),
            camera_yaw_deg=0.0,
            camera_pitch_deg=-89.0,
        )
        # Side view
        _render_debug_ppm(
            p, np_mod, render_side,
            width=render_width, height=render_height,
            camera_target_position=render_target,
            camera_distance=max(0.5, render_camera_distance * 0.9),
            camera_yaw_deg=90.0,
            camera_pitch_deg=-12.0,
        )
        result.render_ppm_path = str(render_ppm)
        result.render_topdown_ppm_path = str(render_topdown)
        result.render_side_ppm_path = str(render_side)
    except Exception as exc:
        result.failure_reason = (
            (result.failure_reason or "") + f"  [render failed: {exc}]"
        )
    finally:
        if client_id is not None:
            try:
                p.disconnect(client_id)
            except Exception:
                pass

# ---------------------------------------------------------------------------
# GUI replay
# ---------------------------------------------------------------------------

def replay_best_base_gui(
    candidate: BasePoseCandidate,
    target_pb: Sequence[float],
    obstacle_specs: Sequence[BoxObstacleSpec],
    planned_path_deg: list[list[float]],
    planner_config_path: Path,
    *,
    hold_seconds: float = 120.0,
    frame_sleep_sec: float = 0.05,
    target_orientation_xyzw: list[float] | None = None,
) -> None:
    """Open a PyBullet **GUI** window and animate the planned arm path.

    The animation replays the joint trajectory found by OMPL (stored in
    ``planned_path_deg`` from :class:`OmplPlanningResult`) frame-by-frame
    so the user can watch the arm avoid obstacles on its way to the grasp target.

    Parameters
    ----------
    candidate:
        The winning base-pose candidate (used for coordinate-frame labels).
    target_pb:
        End-effector target in the candidate's PyBullet frame.
    obstacle_specs:
        Voxel obstacle boxes in the candidate's PyBullet frame.
    planned_path_deg:
        ``planned_path_joint_states_deg`` from the winning OMPL result —
        each inner list is one joint-configuration snapshot (degrees).
    planner_config_path:
        Path to ``pybullet_ompl.yaml``.
    hold_seconds:
        How long to keep the GUI open after the animation finishes.
    frame_sleep_sec:
        Sleep between animation frames (seconds).  Increase to slow down.
    """
    import time as _time

    from src.pybullet_ompl import (
        load_planning_config,
        _set_joint_positions_direct,
        _create_box_obstacles,
        _add_target_marker,
        _add_debug_axes,
    )

    np_mod, p, pybullet_data = _load_python_dependencies()
    planning_config = load_planning_config(planner_config_path, require_obstacles=False)
    arm_config = _load_arm_config()
    expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])

    target_pb_list = [float(v) for v in target_pb]

    client_id: int | None = None
    try:
        client_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.8)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.loadURDF("plane.urdf")

        base_orientation_rad = [math.radians(v) for v in planning_config.base_orientation_euler_deg]
        base_orientation_xyzw = p.getQuaternionFromEuler(base_orientation_rad)
        robot_id = p.loadURDF(
            planning_config.urdf_path,
            useFixedBase=True,
            basePosition=[0.0, 0.0, planning_config.initial_height],
            baseOrientation=base_orientation_xyzw,
        )
        joint_ids, _ = _find_controllable_joints(p, robot_id, expected_joint_count)
        joint_reset_rad = _degrees_to_radians(planning_config.joint_reset_deg)
        _set_joint_positions_direct(p, robot_id, joint_ids, joint_reset_rad)

        # Build the scene
        _create_box_obstacles(p, list(obstacle_specs))
        _add_target_marker(p, target_pb_list)
        _add_debug_axes(
            p, [0.0, 0.0, planning_config.initial_height],
            axis_length=0.12, axis_width=1.8, label="base",
        )
        _add_debug_axes(
            p, target_pb_list,
            axis_length=0.08, axis_width=1.5, label="target",
        )
        p.addUserDebugText(
            f"Best base: x={candidate.unity_x:.2f} z={candidate.unity_z:.2f}  "
            f"dist={candidate.distance_to_target_m:.2f}m  "
            f"IK err={candidate.approach_error_deg:.1f}deg",
            textPosition=[0.0, 0.0, planning_config.initial_height + 0.35],
            textColorRGB=[1.0, 1.0, 0.2],
            textSize=1.0,
        )
        # Position camera to look at target
        p.resetDebugVisualizerCamera(
            cameraDistance=1.0,
            cameraYaw=45.0,
            cameraPitch=-25.0,
            cameraTargetPosition=target_pb_list,
        )
        p.performCollisionDetection()

        print(f"[gui_replay] PyBullet GUI open — animating {len(planned_path_deg)} frames ...")
        print(f"[gui_replay] Base: unity_x={candidate.unity_x:.3f}  "
              f"unity_z={candidate.unity_z:.3f}  "
              f"heading={math.degrees(candidate.heading_rad):.1f}° "
              f"dist={candidate.distance_to_target_m:.2f} m")

        # ── Animate the planned path ────────────────────────────────────────
        for frame_idx, joint_state_deg in enumerate(planned_path_deg):
            joint_state_rad = [math.radians(v) for v in joint_state_deg]
            _set_joint_positions_direct(p, robot_id, joint_ids, joint_state_rad)
            p.stepSimulation()
            _time.sleep(max(0.0, float(frame_sleep_sec)))

        print(f"[gui_replay] Animation complete. Holding GUI open for {hold_seconds:.0f}s ...")
        # ── Hold GUI open ─────────────────────────────────────────────────
        deadline = _time.time() + float(hold_seconds)
        while _time.time() < deadline:
            p.stepSimulation()
            _time.sleep(0.016)  # ~60 fps tick

    finally:
        if client_id is not None:
            try:
                p.disconnect(client_id)
            except Exception:
                pass
