"""
run_base_pose_sampling.py — Entry-point script for the base-pose sampling
evaluator.  Reads a YAML config, optionally loads obstacle voxels from a
previously produced voxel snapshot NPZ, samples candidate base poses from a
2-D occupancy map, and runs the IK-then-OMPL evaluation pipeline.

Usage
-----
python -m scripts.run_base_pose_sampling --config configs/base_pose_sampling.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.base_pose_evaluator import (
    evaluate_candidate_base_poses,
    replay_best_base_gui,
)
from src.base_pose_sampler import sample_candidate_base_poses, BasePoseCandidate
from src.geometry.map_loader import load_free_cells_unity_xz, load_map_meta
from src.pybullet_ompl import BoxObstacleSpec, get_reset_end_effector_position
from src.pybullet_smoke import (
    APPROACH_AGENT_ROOT,
    _load_yaml,
    _resolve_input_path,
    _resolve_output_path,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _resolve(path_str: str, config_path: Path) -> Path:
    return _resolve_input_path(path_str, config_path)


def _resolve_out(path_str: str, config_path: Path) -> Path:
    return _resolve_output_path(path_str, config_path)


def _vec(raw: Any, name: str, n: int) -> tuple:
    if len(raw) != n:
        raise ValueError(f"{name} must have {n} values, got {len(raw)}.")
    return tuple(float(v) for v in raw)


def load_config(config_path: Path) -> dict[str, Any]:
    payload = _load_yaml(config_path)

    # Required
    map_yaml_path = _resolve(str(payload["map_yaml_path"]), config_path)
    planner_config_path = _resolve(str(payload["planner_config_path"]), config_path)
    target_unity_xyz = _vec(payload["target_unity_xyz"], "target_unity_xyz", 3)

    # Optional target approach direction (for ±heading_tolerance filter)
    raw_approach = payload.get("target_approach_dir_unity_xz")
    target_approach_dir: tuple | None = None
    if raw_approach:
        target_approach_dir = _vec(raw_approach, "target_approach_dir_unity_xz", 2)

    # Optional grasp debug npz
    raw_grasp_npz = payload.get("grasp_debug_npz_path")
    grasp_debug_npz_path: Path | None = None
    if raw_grasp_npz and str(raw_grasp_npz).strip().lower() not in {"", "null"}:
        grasp_debug_npz_path = _resolve(str(raw_grasp_npz), config_path)

    return dict(
        map_yaml_path=map_yaml_path,
        planner_config_path=planner_config_path,
        target_unity_xyz=target_unity_xyz,
        target_approach_dir_unity_xz=target_approach_dir,
        grasp_debug_npz_path=grasp_debug_npz_path,
        # Sampling
        base_ground_y_unity=float(payload.get("base_ground_y_unity", 0.0)),
        min_distance_m=float(payload.get("min_distance_m", 0.5)),
        max_distance_m=float(payload.get("max_distance_m", 2.0)),
        heading_tolerance_deg=float(payload.get("heading_tolerance_deg", 10.0)),
        num_samples=int(payload.get("num_samples", 200)),
        rng_seed=int(payload.get("rng_seed", 42)),
        # Voxel / obstacle
        voxel_size_m=float(payload.get("voxel_size_m", 0.05)),
        obstacle_rgba=_vec(payload.get("obstacle_rgba", [0.85, 0.2, 0.2, 0.55]), "obstacle_rgba", 4),
        # Evaluation
        position_tolerance_m=float(payload.get("position_tolerance_m", 0.03)),
        max_ik_candidates=int(payload.get("max_ik_candidates", 20)),
        evaluate_all=bool(payload.get("evaluate_all", False)),
        # Render
        render_width=int(payload.get("render_width", 960)),
        render_height=int(payload.get("render_height", 720)),
        render_camera_distance=float(payload.get("render_camera_distance", 1.2)),
        render_yaw_deg=float(payload.get("render_yaw_deg", 45.0)),
        render_pitch_deg=float(payload.get("render_pitch_deg", -30.0)),
        # Output
        output_root_dir=_resolve_out(
            str(payload.get("output_root_dir", "outputs/base_pose_sampling")),
            config_path,
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timestamped_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = root / stamp
    d.mkdir(parents=True, exist_ok=True)
    return d


from src.pybullet_ompl import get_reset_end_effector_position, get_reset_camera_transform
import scipy.spatial.transform as st

def _load_grasp_debug_data(
    npz_path: Path,
    planner_config_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load target grasp pose and obstacle point clouds from the pipeline NPZ.
    Transforms them precisely using Forward Kinematics from the camera_1 link.
    """
    with np.load(npz_path, allow_pickle=True) as npz:
        best_grasp_camera = np.asarray(npz["best_grasp_camera"], dtype=np.float64)
        if "scene_pc_camera" in npz:
            scene_pc_camera = np.asarray(npz["scene_pc_camera"], dtype=np.float64)
        else:
            scene_pc_camera = np.empty((0, 3), dtype=np.float64)

    # 1. Coordinate rotation: NPZ Camera (visual) -> PyBullet Camera Link (CAD)
    # As instructed by the "Absolute Truth Framework":
    # unity -> pybullet camera: x -> -y, y -> -z, z -> x
    rot_cam_local = np.asarray([
        [0.0, 0.0, 1.0],  # pb_link_x =  cam_z
        [-1.0, 0.0, 0.0], # pb_link_y = -cam_x
        [0.0,-1.0, 0.0]   # pb_link_z = -cam_y
    ], dtype=np.float64)

    # 2. PyBullet Forward Kinematics (FK) for camera_1
    cam_pos, cam_quat = get_reset_camera_transform(planner_config_path)
    cam_pos_arr = np.asarray(cam_pos, dtype=np.float64)
    # Scipy expects [x, y, z, w], exact same as PyBullet
    R_cam_world = st.Rotation.from_quat(cam_quat).as_matrix()

    # The total rotation from NPZ point cloud purely into PyBullet World Frame
    R_full = R_cam_world @ rot_cam_local

    # 3. Apply depth filter in the original camera frame (Z axis)
    if len(scene_pc_camera) > 0:
        print(f"[Debug] Point cloud count BEFORE depth filter: {len(scene_pc_camera)}")
        depths = scene_pc_camera[:, 2]
        valid_mask = (depths >= 0.2) & (depths <= 1.0)
        scene_pc_camera = scene_pc_camera[valid_mask]
        print(f"[Debug] Point cloud count AFTER depth filter (0.2m-1.0m): {len(scene_pc_camera)}")
        
    # 4. Transform voxel points
    if len(scene_pc_camera) > 0:
        scene_pc_camera = _voxel_downsample(scene_pc_camera, voxel_size=0.05)
        print(f"[Debug] Point cloud count AFTER voxel downsample: {len(scene_pc_camera)}")
        # R_full.T because points is (N, 3), so points @ R_full.T is standard R * p
        voxels_pb_world = (scene_pc_camera @ R_full.T) + cam_pos_arr
    else:
        voxels_pb_world = np.empty((0, 3), dtype=np.float64)

    # 5. Transform target grasp
    target_cam_xyz = best_grasp_camera[:3, 3]
    target_pb_xyz = (R_full @ target_cam_xyz) + cam_pos_arr
    
    # User's Explicit Rule for NPZ Grasp -> PB EE Grasp Align:
    # npz -> pb: x->-y, y->-z, z->x
    # Meaning: pb_x = npz_z, pb_y = -npz_x, pb_z = -npz_y
    align_grasp_to_pb_ee = np.asarray([
        [0.0, 1.0,  0.0],
        [0.0, 0.0,  1.0],
        [1.0, 0.0,  0.0]
    ], dtype=np.float64)
    target_rot_pb = (R_full @ best_grasp_camera[:3, :3]) @ align_grasp_to_pb_ee

    return target_pb_xyz, target_rot_pb, voxels_pb_world


def _pybullet_centers_to_unity_world(
    centers_pb: np.ndarray,
    robot_unity_xyz: Sequence[float],
    robot_heading_rad: float,
) -> np.ndarray:
    """Convert pybullet-frame voxel centres (robot-relative) to Unity world.

    New mapping (PyBullet → Unity local):
        uni_x = pb_x
        uni_y = pb_z
        uni_z = pb_y
    """
    # pb → unity (local robot frame)
    lx = centers_pb[:, 0]   # uni_x = pb_x
    ly = centers_pb[:, 2]   # uni_y = pb_z
    lz = centers_pb[:, 1]   # uni_z = pb_y

    h = float(robot_heading_rad)
    c, s = math.cos(h), math.sin(h)
    # Rotate around Unity Y by +h (world from robot)
    wx = lx * c - lz * s
    wy = ly
    wz = lx * s + lz * c

    bx, by, bz = (float(v) for v in robot_unity_xyz)
    world = np.stack([wx + bx, wy + by, wz + bz], axis=1).astype(np.float64)
    return world


def _voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if len(points) == 0 or voxel_size <= 0.0:
        return points
    buckets = np.floor(points / float(voxel_size)).astype(np.int32)
    _, keep_idx = np.unique(buckets, axis=0, return_index=True)
    return points[np.sort(keep_idx)]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_base_pose_sampling(config_path: Path, *, gui: bool = False) -> dict[str, Any]:
    cfg = load_config(config_path)
    output_dir = _timestamped_dir(Path(cfg["output_root_dir"]))

    report: dict[str, Any] = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "target_unity_xyz": list(cfg["target_unity_xyz"]),
    }

    # ── 1. Load map → free cells ─────────────────────────────────────────────
    map_meta = load_map_meta(Path(cfg["map_yaml_path"]))
    free_cells = load_free_cells_unity_xz(map_meta)
    report["map_yaml_path"] = str(cfg["map_yaml_path"])
    report["map_free_cell_count"] = int(len(free_cells))

    # ── 2. Load obstacles & target from grasp NPZ (if provided) ──────────────
    world_obstacles_unity: np.ndarray = np.empty((0, 3), dtype=np.float64)
    target_rotation_pybullet: np.ndarray | None = None

    if cfg.get("grasp_debug_npz_path"):
        try:
            target_pb, target_rot_pb, voxels_pb = _load_grasp_debug_data(
                Path(cfg["grasp_debug_npz_path"]),
                Path(cfg["planner_config_path"])
            )
            
            # Apply downsampling to reduce point cloud density down to 5cm box sizes
            voxel_size = float(cfg.get("voxel_size_m", 0.05))
            if len(voxels_pb) > 0:
                voxels_pb = _voxel_downsample(voxels_pb, voxel_size)
            
            # The NPZ data is local to the camera. We need the camera's global
            # pose (the robot's AMCL pose at snapshot time) to place everything
            # on the Unity map.
            rx = float(cfg.get("robot_unity_x_at_snapshot", 0.0))
            ry = float(cfg.get("robot_unity_y_at_snapshot", 0.0))
            rz = float(cfg.get("robot_unity_z_at_snapshot", 0.0))
            rh = float(cfg.get("robot_heading_rad_at_snapshot", 0.0))

            world_obstacles_unity = _pybullet_centers_to_unity_world(
                voxels_pb, (rx, ry, rz), rh
            )
            
            # Overwrite the target_unity_xyz from the config with the one computed from NPZ
            computed_target_unity = _pybullet_centers_to_unity_world(
                target_pb.reshape(1, 3), (rx, ry, rz), rh
            )[0]
            
            # Maintain Y using the fixed height from config OR the real visual height
            # Typically grasp height is close to gripper y, but let's use the explicit
            # global computed height so it's precise.
            cfg["target_unity_xyz"] = (
                float(computed_target_unity[0]),
                float(computed_target_unity[1]),
                float(computed_target_unity[2]),
            )
            report["target_unity_xyz"] = list(cfg["target_unity_xyz"])
            
            target_rotation_pybullet = target_rot_pb
            report["grasp_debug_npz_path"] = str(cfg["grasp_debug_npz_path"])
            report["voxel_count"] = int(len(world_obstacles_unity))
        except Exception as exc:
            report["voxel_load_warning"] = str(exc)

    # ── 3. Sample candidate base poses ───────────────────────────────────────
    tx, _, tz = cfg["target_unity_xyz"]
    target_xz = (tx, tz)

    candidates = sample_candidate_base_poses(
        target_unity_xz=target_xz,
        free_cells_unity_xz=free_cells,
        target_approach_dir_unity_xz=cfg["target_approach_dir_unity_xz"],
        min_distance_m=cfg["min_distance_m"],
        max_distance_m=cfg["max_distance_m"],
        heading_tolerance_deg=cfg["heading_tolerance_deg"],
        num_samples=cfg["num_samples"],
        rng_seed=cfg["rng_seed"],
    )
    report["candidates_sampled"] = len(candidates)
    print(f"[base_pose_sampling] {len(free_cells)} free cells in map → "
          f"{len(candidates)} sampled candidates after filters.")

    if not candidates:
        report["failure_reason"] = "No candidates passed the sampling filters."
        _write_report(output_dir, report)
        return report

    # ── 4. Evaluate (IK rank → OMPL) ─────────────────────────────────────────
    eval_result = evaluate_candidate_base_poses(
        candidates=candidates,
        target_unity_xyz=cfg["target_unity_xyz"],
        target_rotation_pybullet=target_rotation_pybullet,
        world_obstacle_centers_unity=world_obstacles_unity,
        planner_config_path=Path(cfg["planner_config_path"]),
        output_dir=output_dir,
        base_ground_y_unity=cfg["base_ground_y_unity"],
        voxel_size_m=cfg["voxel_size_m"],
        obstacle_rgba=tuple(cfg["obstacle_rgba"]),
        position_tolerance_m=cfg["position_tolerance_m"],
        max_ik_candidates=cfg["max_ik_candidates"],
        evaluate_all=cfg["evaluate_all"],
        render_width=cfg["render_width"],
        render_height=cfg["render_height"],
        render_camera_distance=cfg["render_camera_distance"],
        render_yaw_deg=cfg["render_yaw_deg"],
        render_pitch_deg=cfg["render_pitch_deg"],
    )

    # ── 5. Save report ────────────────────────────────────────────────────────
    report["success"] = eval_result.success
    report["ik_evaluated_count"] = eval_result.ik_evaluated_count
    report["ompl_evaluated_count"] = eval_result.ompl_evaluated_count
    report["total_time_sec"] = eval_result.total_time_sec
    report["failure_reason"] = eval_result.failure_reason

    if eval_result.best_candidate is not None:
        bc = eval_result.best_candidate
        report["best_base_pose"] = {
            "unity_x": bc.unity_x,
            "unity_z": bc.unity_z,
            "heading_rad": bc.heading_rad,
            "heading_deg": math.degrees(bc.heading_rad),
            "distance_to_target_m": bc.distance_to_target_m,
            "approach_error_deg": bc.approach_error_deg,
            "ee_error_m": eval_result.best_ik_error_m,
        }

    report["render_ppm"] = eval_result.render_ppm_path
    report["render_topdown_ppm"] = eval_result.render_topdown_ppm_path
    report["render_side_ppm"] = eval_result.render_side_ppm_path
    report["ik_results"] = eval_result.all_ik_results
    report["ompl_results"] = eval_result.all_ompl_results
    report["best_ompl_result"] = eval_result.best_ompl_result

    _write_report(output_dir, report)

    if eval_result.success:
        bc = eval_result.best_candidate
        print(
            f"[base_pose_sampling] SUCCESS — best base: "
            f"unity_x={bc.unity_x:.3f}  unity_z={bc.unity_z:.3f}  "
            f"heading={math.degrees(bc.heading_rad):.1f}°  "
            f"dist={bc.distance_to_target_m:.2f} m  "
            f"ee_err={eval_result.best_ik_error_m:.4f} m"
        )
        if eval_result.render_ppm_path:
            print(f"[base_pose_sampling] Render saved to: {eval_result.render_ppm_path}")
    else:
        print(f"[base_pose_sampling] FAILED: {eval_result.failure_reason}")

    if eval_result.success and gui:
        _do_gui_replay(
            eval_result,
            planner_config_path=Path(cfg["planner_config_path"]),
            target_orientation_xyzw=(
                list(_rotation_matrix_to_quaternion_xyzw(target_rotation_pybullet))
                if target_rotation_pybullet is not None
                else None
            ),
        )

    return report


def _write_report(output_dir: Path, report: dict[str, Any]) -> None:
    rp = output_dir / "base_pose_sampling_report.json"
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[base_pose_sampling] Report saved to: {rp}")


# ---------------------------------------------------------------------------
# GUI replay helpers
# ---------------------------------------------------------------------------

def _do_gui_replay(
    eval_result,
    *,
    planner_config_path: Path,
    target_orientation_xyzw: list[float] | None = None,
    hold_seconds: float = 120.0,
    frame_sleep_sec: float = 0.05,
) -> None:
    """Invoke GUI replay using cached data from the evaluator result."""
    if eval_result.best_candidate is None:
        print("[gui_replay] No best candidate to replay.")
        return
    if eval_result._best_target_pb is None or eval_result._best_obstacle_specs is None:
        print("[gui_replay] Missing cached pybullet data — cannot replay.")
        return
    planned_path_deg = (
        eval_result.best_ompl_result.get("planned_path_joint_states_deg") or []
        if eval_result.best_ompl_result
        else []
    )
    if not planned_path_deg:
        print("[gui_replay] No planned path stored in OMPL result — nothing to animate.")
        return
    replay_best_base_gui(
        candidate=eval_result.best_candidate,
        target_pb=eval_result._best_target_pb,
        obstacle_specs=eval_result._best_obstacle_specs,
        planned_path_deg=planned_path_deg,
        planner_config_path=planner_config_path,
        hold_seconds=hold_seconds,
        frame_sleep_sec=frame_sleep_sec,
        target_orientation_xyzw=target_orientation_xyzw,
    )


def replay_from_report(
    report_json_path: Path,
    config_path: Path,
    *,
    hold_seconds: float = 120.0,
    frame_sleep_sec: float = 0.05,
) -> None:
    """Reload a previously saved report JSON and replay the best path in GUI.

    The report must contain ``best_base_pose``, ``ompl_results`` with
    ``planned_path_joint_states_deg``, and scene obstacle definitions.
    We reconstruct everything from scratch using the original config.
    """
    if not report_json_path.exists():
        # Try to find existing reports to suggest one
        outputs_root = report_json_path.parents[2]
        found: list[Path] = []
        if outputs_root.is_dir():
            found = sorted(outputs_root.rglob("base_pose_sampling_report.json"))
        hint = ""
        if found:
            hint = (
                f"\n  Found {len(found)} existing report(s). Latest:\n"
                + "\n".join(f"    {p}" for p in found[-3:])
                + "\n\n  Use:  --replay-report <path above>"
            )
        else:
            hint = (
                "\n  Run the full pipeline first (without --replay-report):\n"
                "    python -m scripts.run_base_pose_sampling --config configs/base_pose_sampling.yaml"
            )
        raise FileNotFoundError(
            f"Report JSON not found: {report_json_path}{hint}"
        )

    print(f"[gui_replay] Loading report: {report_json_path}")
    report = json.loads(report_json_path.read_text(encoding="utf-8"))
    if not report.get("success"):
        print("[gui_replay] Report shows failure — nothing to replay.")
        return

    cfg = load_config(config_path)

    # Find the winning OMPL result entry (path_found + collision_free)
    winning_ompl: dict | None = None
    for entry in report.get("ompl_results", []):
        if entry.get("path_found") and entry.get("planned_path_collision_free", True):
            winning_ompl = entry
            break
    if winning_ompl is None or not report.get("best_ompl_result", {}).get("planned_path_joint_states_deg"):
        print("[gui_replay] Cannot find planned_path_joint_states_deg in report.")
        return

    planned_path_deg: list[list[float]] = report["best_ompl_result"]["planned_path_joint_states_deg"]
    best = report["best_base_pose"]
    candidate = BasePoseCandidate(
        unity_x=float(best["unity_x"]),
        unity_z=float(best["unity_z"]),
        heading_rad=float(best["heading_rad"]),
        distance_to_target_m=float(best["distance_to_target_m"]),
        approach_error_deg=float(best.get("approach_error_deg", 0.0)),
    )

    # Rebuild obstacle specs and targets for this candidate using grasp NPZ logic
    from src.base_pose_evaluator import candidate_obstacles_pybullet, candidate_target_pybullet

    world_obstacles_unity: np.ndarray = np.empty((0, 3), dtype=np.float64)
    if cfg.get("grasp_debug_npz_path"):
        try:
            _, _, voxels_pb = _load_grasp_debug_data(
                Path(cfg["grasp_debug_npz_path"]),
                Path(cfg["planner_config_path"])
            )
            voxel_size = float(cfg.get("voxel_size_m", 0.05))
            if len(voxels_pb) > 0:
                voxels_pb = _voxel_downsample(voxels_pb, voxel_size)
                
            rx = float(cfg.get("robot_unity_x_at_snapshot", 0.0))
            ry = float(cfg.get("robot_unity_y_at_snapshot", 0.0))
            rz = float(cfg.get("robot_unity_z_at_snapshot", 0.0))
            rh = float(cfg.get("robot_heading_rad_at_snapshot", 0.0))
            world_obstacles_unity = _pybullet_centers_to_unity_world(
                voxels_pb, (rx, ry, rz), rh
            )
        except Exception as exc:
            print(f"[gui_replay] Warning: could not load voxels: {exc}")

    obstacle_specs = candidate_obstacles_pybullet(
        candidate,
        world_obstacles_unity,
        float(cfg["base_ground_y_unity"]),
        float(cfg["voxel_size_m"]),
        rgba=tuple(cfg["obstacle_rgba"]),  # type: ignore[arg-type]
    )
    if not obstacle_specs:
        obstacle_specs = (
            BoxObstacleSpec(
                size=(0.01, 0.01, 0.01),
                position=(10.0, 10.0, 10.0),
                rgba=(0.0, 0.0, 0.0, 0.0),
            ),
        )

    target_pb = candidate_target_pybullet(
        candidate, cfg["target_unity_xyz"], float(cfg["base_ground_y_unity"])
    )

    replay_best_base_gui(
        candidate=candidate,
        target_pb=target_pb,
        obstacle_specs=obstacle_specs,
        planned_path_deg=planned_path_deg,
        planner_config_path=Path(cfg["planner_config_path"]),
        hold_seconds=hold_seconds,
        frame_sleep_sec=frame_sleep_sec,
    )


def _rotation_matrix_to_quaternion_xyzw(m: np.ndarray) -> tuple:
    """Thin wrapper — re-exports from camera_car_voxel_ompl."""
    from src.camera_car_voxel_ompl import _rotation_matrix_to_quaternion_xyzw as _impl
    return _impl(m)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sample candidate base poses from a map and evaluate "
                    "IK + OMPL feasibility for a given grasp target."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=APPROACH_AGENT_ROOT / "configs" / "base_pose_sampling.yaml",
        help="Path to the base-pose-sampling YAML config.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="After finding the best base pose, open PyBullet GUI and replay "
             "the planned arm path as a live animation.",
    )
    parser.add_argument(
        "--replay-report",
        type=Path,
        default=None,
        metavar="REPORT_JSON",
        help="Skip the sampling pipeline and replay an already-computed report "
             "JSON in PyBullet GUI mode.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=120.0,
        help="Seconds to keep the PyBullet GUI open after the animation finishes.",
    )
    parser.add_argument(
        "--frame-sleep",
        type=float,
        default=0.05,
        help="Sleep time between animation frames (seconds). Increase to slow down.",
    )
    args = parser.parse_args(argv)
    config_path = args.config.resolve()

    if args.replay_report is not None:
        # ── Replay-only mode ────────────────────────────────────────────────
        replay_from_report(
            args.replay_report.resolve(),
            config_path,
            hold_seconds=args.hold_seconds,
            frame_sleep_sec=args.frame_sleep,
        )
        return 0

    # ── Full sampling + evaluation pipeline ─────────────────────────────────
    report = run_base_pose_sampling(config_path, gui=args.gui)
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
