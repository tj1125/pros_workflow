#!/usr/bin/env python3
"""Render PyBullet debug views for car_approach base samples.

The script rebuilds the same PyBullet scene used by base_sampler, chooses a
sample (selected IK-feasible solution, collision-free closest fallback, or an
explicit sample index), places the arm URDF at that sample's base pose with the
computed IK joint angles, draws voxel obstacles, draws the car body, and marks
the offset target grasp pose with a yellow sphere plus the original grasp pose
with a gray sphere when available.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEBUG_DIR = Path(__file__).resolve().parent
CAR_APPROACH_DIR = DEBUG_DIR.parent
VLM_RL_WORKFLOW_ROOT = CAR_APPROACH_DIR.parents[1]
VLM_RL_ROOT = VLM_RL_WORKFLOW_ROOT.parent
DEFAULT_OUTPUT_DIR = DEBUG_DIR / "outputs"
DEFAULT_CONFIG_PATH = CAR_APPROACH_DIR / "configs" / "car_approach.yaml"
DEFAULT_ROS_MAP_YAML = VLM_RL_ROOT / "src" / "nav_goal_bridge_pkg" / "config" / "keepout_map.yaml"

if str(VLM_RL_WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(VLM_RL_WORKFLOW_ROOT))

base_sampler: Any = None
sample_logic: Any = None
coord: Any = None


def _ensure_car_approach_modules() -> None:
    global base_sampler, sample_logic, coord
    if base_sampler is not None and sample_logic is not None and coord is not None:
        return

    import importlib

    _install_lightweight_car_approach_package()
    base_sampler = importlib.import_module("agents.car_approach.base_sampler")
    sample_logic = importlib.import_module("agents.car_approach.sample_logic")
    coord = importlib.import_module("agents.car_approach.src.geometry.coordinate_transforms")


def _install_lightweight_car_approach_package() -> None:
    """Let this debug script import submodules without running car_approach.__init__."""
    import types

    agents_pkg = sys.modules.get("agents")
    if agents_pkg is None:
        agents_pkg = types.ModuleType("agents")
        agents_pkg.__path__ = [str(VLM_RL_WORKFLOW_ROOT / "agents")]
        agents_pkg.__package__ = "agents"
        sys.modules["agents"] = agents_pkg

    car_pkg = sys.modules.get("agents.car_approach")
    if car_pkg is None:
        car_pkg = types.ModuleType("agents.car_approach")
        car_pkg.__path__ = [str(CAR_APPROACH_DIR)]
        car_pkg.__package__ = "agents.car_approach"
        sys.modules["agents.car_approach"] = car_pkg
        setattr(agents_pkg, "car_approach", car_pkg)


def main() -> None:
    args = _parse_args()
    data = _build_sampling_debug_data(args)
    solution = _choose_solution(data["evaluation"], args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    before_paths = _render_solution_views(
        solution=_before_sample_solution(solution, data["config"], data["scene_config"]),
        voxels=data["voxels"],
        scene_config=data["scene_config"],
        config=data["config"],
        grasp_candidates_camera=data["grasp_candidates_camera"],
        output_dir=output_dir,
        prefix="before",
        width=args.width,
        height=args.height,
        voxel_render_limit=args.voxel_render_limit,
    )
    paths = {f"before_{key}": value for key, value in before_paths.items()}
    paths.update(_render_solution_views(
        solution=solution,
        voxels=data["voxels"],
        scene_config=data["scene_config"],
        config=data["config"],
        grasp_candidates_camera=data["grasp_candidates_camera"],
        output_dir=output_dir,
        prefix=args.prefix,
        width=args.width,
        height=args.height,
        voxel_render_limit=args.voxel_render_limit,
    ))
    depth_heatmap_path = _write_depth_heatmap(
        pointcloud_info=data["pointcloud_info"],
        solution=solution,
        grasp_candidates_camera=data["grasp_candidates_camera"],
        config=data["config"],
        config_path=data["config_path"],
        output_dir=output_dir,
        prefix=args.prefix,
    )
    if depth_heatmap_path is not None:
        paths["depth_heatmap"] = depth_heatmap_path
    ros_map_path = _write_ros_map_overlay(
        solution=solution,
        pointcloud_info=data["pointcloud_info"],
        config=data["config"],
        config_path=data["config_path"],
        output_dir=output_dir,
        prefix=args.prefix,
    )
    if ros_map_path is not None:
        paths["ros_map"] = ros_map_path
    summary_path = output_dir / f"{args.prefix}_summary.json"
    summary = {
        "created_at": time.time(),
        "config_path": str(data["config_path"]),
        "grasp_source": data["grasp_source"],
        "pointcloud_source": data["pointcloud_source"],
        "pointcloud_count": int(len(data["pointcloud"])),
        "voxel_count": int(len(data["voxels"])),
        "target_mask_applied": bool(data["pointcloud_info"].get("target_mask_applied", False)),
        "target_mask_source": data["pointcloud_info"].get("target_mask_source"),
        "target_bbox_applied": bool(data["pointcloud_info"].get("target_bbox_applied", False)),
        "target_bbox_source": data["pointcloud_info"].get("target_bbox_source"),
        "target_bbox_xyxy": data["pointcloud_info"].get("target_bbox_xyxy"),
        "excluded_bbox_xyxy": data["pointcloud_info"].get("excluded_bbox_xyxy"),
        "excluded_bbox_pixel_count": data["pointcloud_info"].get("excluded_bbox_pixel_count"),
        "target_exclusion_applied": bool(data["pointcloud_info"].get("target_exclusion_applied", False)),
        "target_exclusion_source": data["pointcloud_info"].get("target_exclusion_source"),
        "excluded_mask_shape": data["pointcloud_info"].get("excluded_mask_shape"),
        "excluded_mask_pixel_count": data["pointcloud_info"].get("excluded_mask_pixel_count"),
        "grasp_target_gripper_z_offset_m": float(data["config"].get("grasp_target_gripper_z_offset_m", 0.0)),
        "grasp_offset_applied_before_camera_to_pb": str(data["config"].get("grasp_pose_frame", "camera")).lower() == "camera",
        "camera_to_pb_position_mapping": "pb = [cam_x, -cam_z, cam_y] + camera_position_pb",
        "voxel_source_note": "voxels are voxelized from pointcloud after target mask/bbox exclusion when target_exclusion_applied is true",
        "sample_count": int(len(data["samples"])),
        "evaluated_sample_count": int(len(data["evaluation"]["samples"])),
        "ros_map_stats": data["ros_map_stats"],
        "visualizing_map_rejected_samples": bool(data.get("visualizing_map_rejected_samples", False)),
        "ros_map_yaml_path": str(data["config"].get("map_yaml_path", "")),
        "solution_choice": _solution_choice_label(solution, data["evaluation"]),
        "solution": _solution_summary(solution),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    summary_path.write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    # print(f"saved before main view: {paths['before_main']}")
    # print(f"saved before topdown view: {paths['before_topdown']}")
    # print(f"saved before side view: {paths['before_side']}")
    # print(f"saved main view: {paths['main']}")
    # print(f"saved topdown view: {paths['topdown']}")
    # print(f"saved side view: {paths['side']}")
    # if "depth_heatmap" in paths:
    #     print(f"saved depth heatmap: {paths['depth_heatmap']}")
    # else:
    #     print("depth heatmap skipped: no decoded depth image is available")
    # if "ros_map" in paths:
    #     print(f"saved ROS map overlay: {paths['ros_map']}")
    # else:
    #     print("ROS map overlay skipped: selected solution has no ROS map pose or map image is unavailable")
    # print(f"saved summary: {summary_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render car_approach PyBullet sample debug views.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--grasp-json", type=Path, default=None)
    parser.add_argument(
        "--session-sqlite",
        type=Path,
        default=None,
        help="Session database to read grasp_result from when no grasp JSON is available; defaults to latest logs/sessions/*/session.sqlite.",
    )
    parser.add_argument(
        "--ros-map-yaml",
        type=Path,
        default=DEFAULT_ROS_MAP_YAML,
        help="ROS occupancy map YAML used for map filtering and overlay; defaults to nav_goal_bridge keepout_map.yaml.",
    )
    parser.add_argument(
        "--solution",
        choices=("auto", "selected", "closest"),
        default="auto",
        help="auto uses selected_solution, then collision-free closest_solution.",
    )
    parser.add_argument("--sample-index", type=int, default=None, help="Render a specific evaluated sample index.")
    parser.add_argument(
        "--skip-ros-map-check",
        action="store_true",
        help="Disable ROS map feasibility filtering for offline PyBullet-only debugging.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default="after")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument(
        "--voxel-render-limit",
        type=int,
        default=0,
        help="0 renders all voxels; positive values render nearest voxels to the selected target.",
    )
    return parser.parse_args()


def _build_sampling_debug_data(args: argparse.Namespace) -> dict[str, Any]:
    _ensure_car_approach_modules()
    config_path, config = base_sampler.load_car_approach_config(args.config)
    config = dict(config)
    config["map_yaml_path"] = str(Path(args.ros_map_yaml).expanduser().resolve())
    if args.skip_ros_map_check:
        config["enable_ros_map_check"] = False
        config["allow_ros_map_check_without_reference"] = True

    scene_config = base_sampler._robot_scene_config(config, config_path)
    grasp_result_payload, grasp_payload_source = _session_grasp_payload_for_args(args, config)
    run_config = base_sampler.BaseSamplerRunConfig(
        config_path=config_path,
        grasp_json_path=args.grasp_json,
        grasp_result_payload=grasp_result_payload,
    )

    pointcloud, pointcloud_source, pointcloud_info, voxels = base_sampler.prepare_scene_pointcloud_and_voxels(
        run_config,
        config,
        config_path,
        scene_config,
    )

    grasp_candidates, grasp_source = base_sampler._load_grasps(run_config, config)
    if grasp_payload_source and grasp_source == "payload":
        grasp_source = grasp_payload_source
    if not grasp_candidates:
        raise RuntimeError("No grasp pose is available for visualization.")
    grasp_candidates_camera = grasp_candidates[: max(1, int(config["max_grasps"]))]
    grasp_candidates = base_sampler._transform_grasp_candidates_to_pybullet(
        grasp_candidates_camera,
        config,
    )
    samples = sample_logic.sample_base_points(
        grasp_candidates,
        min_backoff_m=float(config["min_backoff_m"]),
        max_backoff_m=float(config["max_backoff_m"]),
        step_m=float(config["backoff_step_m"]),
        yaw_span_deg=float(config["yaw_span_deg"]),
        yaw_step_deg=float(config["yaw_step_deg"]),
        max_samples=int(config["max_samples"]),
    )
    if not samples:
        raise RuntimeError("No base sample was generated for visualization.")
    samples_before_ros_map_filter = list(samples)

    samples, ros_map_stats = base_sampler._annotate_and_filter_samples_for_ros_map(
        samples,
        config,
        config_path,
        pointcloud_info,
    )
    visualizing_map_rejected_samples = False
    if not samples:
        samples, ros_map_stats = base_sampler._annotate_and_filter_samples_for_ros_map(
            samples_before_ros_map_filter,
            config,
            config_path,
            pointcloud_info,
            return_rejected=True,
        )
        visualizing_map_rejected_samples = bool(samples)
        if not samples:
            raise RuntimeError(
                "No base sample passed ROS map filtering and no rejected sample could be annotated. "
                f"ros_map_stats={ros_map_stats}"
            )
        ros_map_stats = dict(ros_map_stats)
        ros_map_stats["visualize_using_map_rejected_samples"] = True
        print(
            "ROS map filtering rejected every sample; visualize.py will render map-rejected samples "
            "so you can inspect where the vehicle footprint lands."
        )

    evaluation = base_sampler._evaluate_samples_in_pybullet(
        samples,
        scene_config=scene_config,
        voxel_centers=voxels,
        voxel_size_m=scene_config.voxel_size_m,
    )
    return {
        "config_path": config_path,
        "config": config,
        "scene_config": scene_config,
        "pointcloud": pointcloud,
        "pointcloud_source": pointcloud_source,
        "pointcloud_info": pointcloud_info,
        "voxels": voxels,
        "grasp_source": grasp_source,
        "grasp_candidates_camera": grasp_candidates_camera,
        "samples": samples,
        "ros_map_stats": ros_map_stats,
        "visualizing_map_rejected_samples": visualizing_map_rejected_samples,
        "evaluation": evaluation,
    }


def _session_grasp_payload_for_args(args: argparse.Namespace, config: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    import os

    explicit_grasp_json = (
        args.grasp_json is not None
        or bool(config.get("grasp_json_path"))
        or bool(os.getenv("GRASP_RESULT_JSON", "").strip())
    )
    if args.session_sqlite is None and explicit_grasp_json:
        return None, ""
    return _load_grasp_result_payload_from_session(args.session_sqlite, required=args.session_sqlite is not None)


def _load_grasp_result_payload_from_session(
    session_sqlite: Path | None,
    *,
    required: bool,
) -> tuple[dict[str, Any] | None, str]:
    db_path = Path(session_sqlite).expanduser().resolve() if session_sqlite is not None else _latest_session_sqlite()
    if db_path is None:
        if required:
            raise FileNotFoundError(f"session.sqlite not found: {session_sqlite}")
        return None, ""
    if not db_path.exists():
        if required:
            raise FileNotFoundError(f"session.sqlite not found: {db_path}")
        return None, ""

    for payload, source in _iter_grasp_payloads_from_session(db_path):
        normalized = _normalize_grasp_payload(payload)
        if normalized is not None:
            return normalized, source
    if required:
        raise FileNotFoundError(f"No usable grasp_result payload found in session sqlite: {db_path}")
    return None, ""


def _latest_session_sqlite() -> Path | None:
    sessions_dir = VLM_RL_WORKFLOW_ROOT / "logs" / "sessions"
    if not sessions_dir.exists():
        return None
    matches = [path for path in sessions_dir.glob("*/session.sqlite") if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda item: (item.stat().st_mtime, str(item))).resolve()


def _iter_grasp_payloads_from_session(db_path: Path) -> list[tuple[dict[str, Any], str]]:
    import json
    import sqlite3

    rows: list[tuple[dict[str, Any], str]] = []
    connection_attempts = [
        (f"file:{db_path}?mode=ro", True),
        (f"file:{db_path}?mode=ro&immutable=1", True),
        (str(db_path), False),
    ]
    for target, use_uri in connection_attempts:
        conn = None
        try:
            conn = sqlite3.connect(target, uri=use_uri)
            conn.row_factory = sqlite3.Row
            rows = []
            rows.extend(_grasp_payloads_from_state_patches(conn, db_path))
            rows.extend(_grasp_payloads_from_agent_runs(conn, db_path))
            if rows:
                return rows
        except sqlite3.Error:
            continue
        finally:
            if conn is not None:
                conn.close()
    return rows


def _grasp_payloads_from_state_patches(conn: Any, db_path: Path) -> list[tuple[dict[str, Any], str]]:
    import json
    import sqlite3

    try:
        records = conn.execute(
            """
            SELECT step, node_name, state_patch_json
            FROM state_patches
            WHERE state_patch_json LIKE '%grasp_result%'
            ORDER BY step DESC, created_at DESC
            LIMIT 50
            """
        ).fetchall()
    except sqlite3.Error:
        return []

    payloads: list[tuple[dict[str, Any], str]] = []
    for row in records:
        try:
            patch = json.loads(row["state_patch_json"] or "{}")
        except json.JSONDecodeError:
            continue
        payload = patch.get("grasp_result") if isinstance(patch, dict) else None
        if isinstance(payload, dict):
            artifact_payload = _load_grasp_raw_result_ref(conn, db_path, payload)
            if artifact_payload is not None:
                payloads.append((artifact_payload, f"{db_path}:state_patches.step_{row['step']}.raw_result_ref"))
            payloads.append((payload, f"{db_path}:state_patches.step_{row['step']}.{row['node_name']}"))
    return payloads


def _grasp_payloads_from_agent_runs(conn: Any, db_path: Path) -> list[tuple[dict[str, Any], str]]:
    import json
    import sqlite3

    try:
        records = conn.execute(
            """
            SELECT step, node_name, agent_name, summary_json, raw_result_artifact_id, output_refs_json
            FROM agent_runs
            WHERE agent_name LIKE '%Grasp%' OR node_name LIKE '%grasp%'
            ORDER BY step DESC, created_at DESC
            LIMIT 50
            """
        ).fetchall()
    except sqlite3.Error:
        return []

    payloads: list[tuple[dict[str, Any], str]] = []
    for row in records:
        artifact_payload = _load_artifact_json_by_id(conn, db_path, row["raw_result_artifact_id"])
        if artifact_payload is not None:
            payloads.append((artifact_payload, f"{db_path}:agent_runs.step_{row['step']}.raw_result_artifact"))
        try:
            summary = json.loads(row["summary_json"] or "{}")
        except json.JSONDecodeError:
            summary = {}
        if isinstance(summary, dict):
            artifact_payload = _load_grasp_raw_result_ref(conn, db_path, summary)
            if artifact_payload is not None:
                payloads.append((artifact_payload, f"{db_path}:agent_runs.step_{row['step']}.raw_result_ref"))
            payloads.append((summary, f"{db_path}:agent_runs.step_{row['step']}.{row['node_name']}"))
        try:
            output_refs = json.loads(row["output_refs_json"] or "{}")
        except json.JSONDecodeError:
            output_refs = {}
        raw_ref = output_refs.get("raw_result_ref") if isinstance(output_refs, dict) else None
        if isinstance(raw_ref, dict):
            artifact_payload = _load_artifact_json_by_ref(db_path, raw_ref)
            if artifact_payload is not None:
                payloads.append((artifact_payload, f"{db_path}:agent_runs.step_{row['step']}.output_refs.raw_result_ref"))
    return payloads


def _load_grasp_raw_result_ref(conn: Any, db_path: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    raw_ref = payload.get("raw_result_ref")
    if isinstance(raw_ref, dict):
        artifact_payload = _load_artifact_json_by_ref(db_path, raw_ref)
        if artifact_payload is not None:
            return artifact_payload
    artifact_id = raw_ref.get("artifact_id") if isinstance(raw_ref, dict) else None
    return _load_artifact_json_by_id(conn, db_path, artifact_id)


def _load_artifact_json_by_id(conn: Any, db_path: Path, artifact_id: Any) -> dict[str, Any] | None:
    import sqlite3

    if not artifact_id:
        return None
    try:
        row = conn.execute("SELECT path FROM artifacts WHERE artifact_id = ?", (str(artifact_id),)).fetchone()
    except sqlite3.Error:
        return None
    if row is None:
        return None
    return _load_artifact_json_by_ref(db_path, {"path": row["path"]})


def _load_artifact_json_by_ref(db_path: Path, ref: dict[str, Any]) -> dict[str, Any] | None:
    import json

    raw_path = ref.get("path")
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        logs_dir = db_path.parents[2]
        path = logs_dir / path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_grasp_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    nested = payload.get("result")
    if isinstance(nested, dict):
        payload = nested
    raw_result = payload.get("raw_result")
    candidate = raw_result if isinstance(raw_result, dict) else payload
    if isinstance(candidate.get("valid_grasp_poses_camera"), list) and candidate["valid_grasp_poses_camera"]:
        return candidate
    if isinstance(candidate.get("best_grasp_pose_camera"), dict) and candidate["best_grasp_pose_camera"]:
        return candidate
    return None


def _before_sample_solution(solution: dict[str, Any], config: dict[str, Any], scene_config: Any) -> dict[str, Any]:
    before = dict(solution)
    before["sample_source"] = "before_sampling_initial_pose"
    before["pb_base_link_xyz"] = [float(v) for v in config.get("arm_base_link_pb_xyz", [0.0, 0.0, scene_config.base_height_m])]
    before["pb_base_link_yaw_rad"] = _initial_arm_base_yaw_rad(config)
    before["pb_base_link_yaw_deg"] = math.degrees(float(before["pb_base_link_yaw_rad"]))
    before.pop("ik_joint_solution_rad", None)
    before.pop("ik_joint_solution_deg", None)
    before.pop("final_ee_position_xyz", None)
    return before


def _initial_arm_base_yaw_rad(config: dict[str, Any]) -> float:
    euler_deg = config.get("base_orientation_euler_deg", [0.0, 0.0, 0.0])
    return math.radians(float(list(euler_deg)[2]))


def _choose_solution(evaluation: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    candidates = [item for item in evaluation.get("samples", []) if isinstance(item, dict)]
    if args.sample_index is not None:
        for item in candidates:
            if int(item.get("sample_index", -1)) == int(args.sample_index):
                return item
        raise RuntimeError(f"sample_index={args.sample_index} not found among evaluated samples.")

    selected = evaluation.get("selected_solution")
    closest = evaluation.get("closest_solution")
    if args.solution == "selected":
        if isinstance(selected, dict) and selected:
            return selected
        raise RuntimeError("selected_solution is unavailable.")
    if args.solution == "closest":
        if isinstance(closest, dict) and closest:
            return closest
        raise RuntimeError("collision-free closest_solution is unavailable.")

    if isinstance(selected, dict) and selected:
        return selected
    if isinstance(closest, dict) and closest:
        return closest
    if candidates:
        return min(candidates, key=_candidate_sort_key)
    raise RuntimeError("No evaluated sample is available for visualization.")


def _candidate_sort_key(solution: dict[str, Any]) -> tuple[int, float, float, int]:
    orientation_error = solution.get("ee_orientation_error_deg")
    return (
        0 if bool(solution.get("collision_free", False)) else 1,
        float(solution.get("ee_position_error_m", float("inf"))),
        float("inf") if orientation_error is None else float(orientation_error),
        int(solution.get("sample_index", 0)),
    )


def _render_solution_views(
    *,
    solution: dict[str, Any],
    voxels: Any,
    scene_config: Any,
    config: dict[str, Any],
    grasp_candidates_camera: Sequence[Any],
    output_dir: Path,
    prefix: str,
    width: int,
    height: int,
    voxel_render_limit: int,
) -> dict[str, Path]:
    _ensure_car_approach_modules()
    import numpy as np
    import pybullet as p
    import pybullet_data

    client_id = p.connect(p.DIRECT)
    if client_id < 0:
        raise RuntimeError("PyBullet DIRECT connection failed.")
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.8)
        p.loadURDF("plane.urdf")

        target_xyz = _vector3(solution.get("target_xyz"), "solution.target_xyz")
        voxels_to_render = _select_render_voxels(np.asarray(voxels, dtype=np.float64), target_xyz, voxel_render_limit)
        _add_voxel_visuals(p, voxels_to_render, float(scene_config.voxel_size_m))

        robot_id = p.loadURDF(
            str(scene_config.urdf_path),
            useFixedBase=True,
            basePosition=[0.0, 0.0, float(scene_config.base_height_m)],
            baseOrientation=p.getQuaternionFromEuler(coord.deg_sequence_to_rad(scene_config.base_orientation_euler_deg)),
        )
        joint_ids = base_sampler._controllable_joint_ids(p, robot_id, int(scene_config.controllable_joints))
        base_xyz = _vector3(solution.get("pb_base_link_xyz"), "solution.pb_base_link_xyz")
        base_xyz[2] = float(scene_config.base_height_m)
        base_yaw = float(solution.get("pb_base_link_yaw_rad", 0.0))
        p.resetBasePositionAndOrientation(robot_id, base_xyz, p.getQuaternionFromEuler([0.0, 0.0, base_yaw]))

        joint_solution = solution.get("ik_joint_solution_rad")
        if isinstance(joint_solution, Sequence) and not isinstance(joint_solution, (str, bytes)):
            joints = [float(value) for value in joint_solution][: len(joint_ids)]
        else:
            joints = coord.deg_sequence_to_rad(scene_config.joint_reset_deg[: len(joint_ids)])
        base_sampler._set_joint_positions(p, robot_id, joint_ids, joints)
        p.performCollisionDetection()
        ee_xyz, ee_rotation = _current_ee_pose(p, robot_id, int(scene_config.ee_link_index))

        original_target_xyz = _original_grasp_position_pb(solution, grasp_candidates_camera, config)
        if original_target_xyz is not None:
            _add_original_grasp_marker(p, original_target_xyz)
        _add_target_marker(p, target_xyz)
        _add_grasp_axes(p, target_xyz, solution.get("target_rotation_matrix"))
        _add_final_ee_marker(p, ee_xyz)
        _add_ee_axes(p, ee_xyz, ee_rotation)
        _add_arm_base_marker(p, base_xyz)
        _add_car_body_visual(p, config=config, base_xyz=base_xyz, base_yaw_rad=base_yaw)

        camera_target, camera_distance = _arm_camera_target_and_distance(
            p,
            np,
            robot_id=robot_id,
            target_xyz=target_xyz,
            final_ee_xyz=ee_xyz,
        )
        paths = {
            "main": output_dir / f"{prefix}_main.ppm",
            "topdown": output_dir / f"{prefix}_topdown.ppm",
            "side": output_dir / f"{prefix}_side.ppm",
        }
        _render_ppm(p, np, paths["main"], width, height, camera_target, camera_distance, 45.0, -28.0)
        _render_ppm(p, np, paths["topdown"], width, height, camera_target, max(0.7, camera_distance * 0.85), 0.0, -89.0)
        _render_ppm(p, np, paths["side"], width, height, camera_target, max(0.7, camera_distance * 0.9), 90.0, -8.0)
        return paths
    finally:
        p.disconnect(client_id)




def _write_depth_heatmap(
    *,
    pointcloud_info: dict[str, Any],
    solution: dict[str, Any],
    grasp_candidates_camera: Sequence[Any],
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    prefix: str,
) -> Path | None:
    depth = pointcloud_info.get("_depth_metric_m")
    if depth is None:
        return None

    try:
        import numpy as np
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim != 2 or depth_arr.size == 0:
        return None

    source_mirrored = bool(pointcloud_info.get("depth_camera_x_mirrored", False))
    mirrored = True
    display_depth = np.fliplr(depth_arr)
    valid_mask = np.isfinite(display_depth) & (display_depth > 0.0)
    if not np.any(valid_mask):
        return None

    valid_depth = display_depth[valid_mask]
    near_m = float(np.nanmin(valid_depth))
    far_m = float(np.nanmax(valid_depth))
    if not math.isfinite(near_m) or not math.isfinite(far_m):
        return None
    if far_m <= near_m:
        far_m = near_m + 1e-6

    t = np.zeros(display_depth.shape, dtype=np.float32)
    t[valid_mask] = np.clip((display_depth[valid_mask] - near_m) / (far_m - near_m), 0.0, 1.0)
    heatmap = np.zeros((*display_depth.shape, 3), dtype=np.uint8)
    heatmap[:, :, 0] = np.asarray((1.0 - t) * 255.0, dtype=np.uint8)
    heatmap[:, :, 2] = np.asarray(t * 255.0, dtype=np.uint8)
    heatmap[:, :, 1] = np.asarray((1.0 - np.abs((t * 2.0) - 1.0)) * 32.0, dtype=np.uint8)
    heatmap[~valid_mask] = (0, 0, 0)

    image = Image.fromarray(heatmap, "RGB")
    height_px, width_px = display_depth.shape
    scale = max(1, min(6, int(math.ceil(900.0 / float(max(width_px, height_px))))))
    if scale > 1:
        resampling = getattr(getattr(Image, "Resampling", Image), "NEAREST")
        image = image.resize((width_px * scale, height_px * scale), resampling)

    draw = ImageDraw.Draw(image, "RGBA")
    _draw_depth_image_axes(draw, width=image.size[0], height=image.size[1])
    bbox_drawn = _draw_depth_target_bbox(
        draw,
        pointcloud_info=pointcloud_info,
        width_px=width_px,
        height_px=height_px,
        scale=float(scale),
        display_mirrored=mirrored,
    )
    target_pixel = _target_grasp_pixel_on_depth_heatmap(
        solution=solution,
        grasp_candidates_camera=grasp_candidates_camera,
        config=config,
        config_path=config_path,
    )
    target_in_image = False
    if target_pixel is not None:
        u_px, v_px, original_cam_xyz = target_pixel
        target_in_image = 0.0 <= u_px < float(width_px) and 0.0 <= v_px < float(height_px)
        if target_in_image:
            _draw_depth_target_marker(draw, u_px * float(scale), v_px * float(scale), scale=float(scale))

    print("depth heatmap info:")
    print(
        f"  output display flip horizontal: {mirrored} "
        f"source_depth_camera_x_mirrored={source_mirrored} "
        "grasp_marker=direct_camera_pose_projection"
    )
    print(f"  valid depth range: near={near_m:.4f}m far={far_m:.4f}m")
    print("  color: red=near, blue=far; axis overlay: +x right, +y down, +z into image/farther")
    print(f"  target bbox overlay: drawn={bbox_drawn}")
    if target_pixel is not None:
        u_px, v_px, original_cam_xyz = target_pixel
        print(
            "  grasp pose marker direct camera projection: "
            f"pixel=({u_px:.1f}, {v_px:.1f}) "
            f"camera_xyz=({original_cam_xyz[0]:.4f}, {original_cam_xyz[1]:.4f}, {original_cam_xyz[2]:.4f}) "
            f"drawn_white_dot={target_in_image}"
        )
    else:
        print("  target grasp pose: unavailable for depth projection")

    output_path = output_dir / f"{prefix}_depth_heatmap.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _target_grasp_pixel_on_depth_heatmap(
    *,
    solution: dict[str, Any],
    grasp_candidates_camera: Sequence[Any],
    config: dict[str, Any],
    config_path: Path,
) -> tuple[float, float, tuple[float, float, float]] | None:
    try:
        import numpy as np
        from agents.car_approach.src.io.intrinsics import load_camera_intrinsics

        raw_grasp_index = solution.get("grasp_index")
        grasp_index = int(raw_grasp_index) if raw_grasp_index is not None else None
        original_cam_xyz = None
        for candidate in grasp_candidates_camera:
            if grasp_index is None or int(getattr(candidate, "index")) == grasp_index:
                original_cam_xyz = np.asarray(getattr(candidate, "position_xyz"), dtype=np.float64).reshape(3)
                break
        if original_cam_xyz is None and grasp_candidates_camera:
            original_cam_xyz = np.asarray(getattr(grasp_candidates_camera[0], "position_xyz"), dtype=np.float64).reshape(3)
        if original_cam_xyz is None or float(original_cam_xyz[2]) <= 1e-6:
            return None
        raw_intrinsics_path = config.get("intrinsics_path")
        if not raw_intrinsics_path:
            return None
        intrinsics_path = base_sampler._resolve_input_path(str(raw_intrinsics_path), config_path.parent)
        intrinsics = load_camera_intrinsics(intrinsics_path)
        k = np.asarray(intrinsics.k, dtype=np.float64).reshape(3, 3)
        fx = float(k[0, 0])
        fy = float(k[1, 1])
        cx = float(k[0, 2])
        cy = float(k[1, 2])
        if abs(fx) <= 1e-9 or abs(fy) <= 1e-9:
            return None
        u_px = (fx * float(original_cam_xyz[0]) / float(original_cam_xyz[2])) + cx
        v_px = (fy * float(original_cam_xyz[1]) / float(original_cam_xyz[2])) + cy
        return (
            float(u_px),
            float(v_px),
            tuple(float(v) for v in original_cam_xyz),
        )
    except Exception:
        return None


def _draw_depth_target_marker(draw: Any, u_px: float, v_px: float, *, scale: float) -> None:
    radius = max(5.0, min(12.0, 5.0 * float(scale)))
    center = (float(u_px), float(v_px))
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    outline_bbox = [bbox[0] - 2.0, bbox[1] - 2.0, bbox[2] + 2.0, bbox[3] + 2.0]
    draw.ellipse(outline_bbox, fill=(0, 0, 0, 210))
    draw.ellipse(bbox, fill=(255, 255, 255, 255))
    cross = radius + 3.0
    width = max(1, int(round(float(scale))))
    draw.line([(center[0] - cross, center[1]), (center[0] + cross, center[1])], fill=(0, 0, 0, 230), width=width)
    draw.line([(center[0], center[1] - cross), (center[0], center[1] + cross)], fill=(0, 0, 0, 230), width=width)


def _draw_depth_target_bbox(
    draw: Any,
    *,
    pointcloud_info: dict[str, Any],
    width_px: int,
    height_px: int,
    scale: float,
    display_mirrored: bool,
) -> bool:
    bbox = pointcloud_info.get("excluded_bbox_xyxy") or pointcloud_info.get("target_bbox_xyxy")
    if bbox is None:
        return False
    try:
        x0, y0, x1, y1 = [float(v) for v in bbox]
    except Exception:
        return False
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0:
        x0 *= float(width_px)
        x1 *= float(width_px)
        y0 *= float(height_px)
        y1 *= float(height_px)
    left = max(0.0, min(float(width_px), min(x0, x1)))
    right = max(0.0, min(float(width_px), max(x0, x1)))
    top = max(0.0, min(float(height_px), min(y0, y1)))
    bottom = max(0.0, min(float(height_px), max(y0, y1)))
    if right <= left or bottom <= top:
        return False
    if display_mirrored:
        left, right = float(width_px) - right, float(width_px) - left
    rect = [
        left * float(scale),
        top * float(scale),
        right * float(scale),
        bottom * float(scale),
    ]
    width = max(2, int(round(2.0 * float(scale))))
    for inset in range(width):
        draw.rectangle(
            [rect[0] - inset, rect[1] - inset, rect[2] + inset, rect[3] + inset],
            outline=(255, 232, 64, 255),
        )
    return True


def _draw_depth_image_axes(draw: Any, *, width: int, height: int) -> None:
    origin = (max(12, int(width * 0.035)), max(12, int(height * 0.035)))
    axis_len = max(28, int(min(width, height) * 0.12))
    line_width = max(1, int(round(min(width, height) / 450.0)))
    text_offset = max(4, int(min(width, height) * 0.01))
    x_end = (origin[0] + axis_len, origin[1])
    y_end = (origin[0], origin[1] + axis_len)
    _draw_arrow(draw, origin, x_end, color=(255, 255, 255, 230), width=line_width)
    _draw_arrow(draw, origin, y_end, color=(255, 255, 255, 230), width=line_width)
    draw.text((x_end[0] + text_offset, x_end[1] - text_offset), "+x", fill=(255, 255, 255, 240))
    draw.text((y_end[0] + text_offset, y_end[1] - text_offset), "+y", fill=(255, 255, 255, 240))


def _draw_arrow(
    draw: Any,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    draw.line([start, end], fill=color, width=width)
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return
    ux = dx / length
    uy = dy / length
    head_len = max(5.0, float(width) * 5.0)
    head_w = max(3.0, float(width) * 3.0)
    base = (float(end[0]) - ux * head_len, float(end[1]) - uy * head_len)
    perp = (-uy, ux)
    points = [
        (float(end[0]), float(end[1])),
        (base[0] + perp[0] * head_w, base[1] + perp[1] * head_w),
        (base[0] - perp[0] * head_w, base[1] - perp[1] * head_w),
    ]
    draw.polygon(points, fill=color)

def _write_ros_map_overlay(
    *,
    solution: dict[str, Any],
    pointcloud_info: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    prefix: str,
) -> Path | None:
    base_pose = _pose2d_dict(solution.get("ros_map_base_link_pose"))
    before_pose = _before_vehicle_ros_map_pose(pointcloud_info, config)
    pose = _pose2d_dict(solution.get("ros_map_amcl_pose") or solution.get("goal_pose"))
    if pose is None and base_pose is not None:
        pose = _car_center_pose_from_arm_base_pose(base_pose, config, reference_pose=before_pose)
    if pose is None:
        return None
    raw_map_path = config.get("map_yaml_path")
    if not raw_map_path:
        return None

    try:
        from PIL import Image, ImageDraw
        from agents.car_approach.src.geometry import map_free_space
        import numpy as np
    except ImportError:
        return None

    map_yaml_path = base_sampler._resolve_input_path(str(raw_map_path), config_path.parent)
    meta = map_free_space.load_map_meta(map_yaml_path)
    image = Image.open(meta.pgm_path).convert("RGB")
    width_px, height_px = image.size
    scale = int(config.get("debug_ros_map_overlay_scale", 0) or 0)
    if scale <= 0:
        scale = max(1, min(6, int(math.ceil(900.0 / float(max(width_px, height_px))))))
    if scale > 1:
        resampling = getattr(getattr(Image, "Resampling", Image), "NEAREST")
        image = image.resize((width_px * scale, height_px * scale), resampling)
    draw = ImageDraw.Draw(image, "RGBA")
    draw_scale = float(scale)

    vehicle_length_x = float(config.get("vehicle_base_length_x_m", 0.33))
    vehicle_length_y = float(config.get("vehicle_base_length_y_m", 0.35))
    footprint_local = np.asarray(
        [
            [-vehicle_length_x * 0.5, -vehicle_length_y * 0.5],
            [vehicle_length_x * 0.5, -vehicle_length_y * 0.5],
            [vehicle_length_x * 0.5, vehicle_length_y * 0.5],
            [-vehicle_length_x * 0.5, vehicle_length_y * 0.5],
        ],
        dtype=np.float64,
    )
    before_arm_pose = _arm_base_pose_from_vehicle_center_pose(before_pose, config)
    before_px_raw = None
    before_px = None
    if before_pose is not None:
        before_px_raw, before_px = _draw_vehicle_pose_on_ros_map(
            draw,
            pose=before_pose,
            footprint_local=footprint_local,
            meta=meta,
            height_px=height_px,
            draw_scale=draw_scale,
            fill=(0, 180, 85, 90),
            outline=(0, 145, 70, 255),
            cross_color=(255, 255, 255, 255),
            heading_color=(0, 145, 70, 255),
            polygon_width=2,
            cross_radius=max(1.0, 7.0 * draw_scale / 3.0),
        )
    before_arm_px_raw = None
    if before_arm_pose is not None:
        before_arm_px_raw = _ros_xy_to_pixel(before_arm_pose["x"], before_arm_pose["y"], meta, height_px)
        before_arm_px = _scale_pixel(before_arm_px_raw, draw_scale)
        _draw_cross(draw, before_arm_px, radius=max(1.0, 6.0 * draw_scale / 3.0), color=(255, 0, 255, 255), width=1)
        if before_px is not None:
            draw.line([before_px, before_arm_px], fill=(255, 0, 255, 220), width=1)

    amcl_px_raw, amcl_px = _draw_vehicle_pose_on_ros_map(
        draw,
        pose=pose,
        footprint_local=footprint_local,
        meta=meta,
        height_px=height_px,
        draw_scale=draw_scale,
        fill=(0, 96, 255, 120),
        outline=(0, 24, 220, 255),
        cross_color=(255, 220, 0, 255),
        heading_color=(255, 220, 0, 255),
        polygon_width=2,
        cross_radius=max(1.0, 7.0 * draw_scale / 3.0),
    )

    if base_pose is not None:
        base_px = _scale_pixel(_ros_xy_to_pixel(base_pose["x"], base_pose["y"], meta, height_px), draw_scale)
        _draw_cross(draw, base_px, radius=max(1.0, 6.0 * draw_scale / 3.0), color=(0, 255, 255, 255), width=1)
        draw.line([amcl_px, base_px], fill=(0, 255, 255, 220), width=1)

    origin_x, origin_y = (float(meta.origin_xy[0]), float(meta.origin_xy[1]))
    map_max_x = origin_x + float(width_px) * float(meta.resolution_m)
    map_max_y = origin_y + float(height_px) * float(meta.resolution_m)
    in_map = 0.0 <= amcl_px_raw[0] < float(width_px) and 0.0 <= amcl_px_raw[1] < float(height_px)
    print("ROS map overlay info:")
    print(f"  map image: {meta.pgm_path}")
    print(f"  map range: x=[{origin_x:.2f}, {map_max_x:.2f}] y=[{origin_y:.2f}, {map_max_y:.2f}] resolution={meta.resolution_m:.3f}m/px inside={in_map}")
    print("===")
    if before_pose is not None:
        print("current vehicle ROS map pose")
        print(f"  position: x={before_pose['x']:.6f} y={before_pose['y']:.6f}")
        print(f"  yaw: {float(before_pose['yaw_rad']):.6f} rad ({math.degrees(float(before_pose['yaw_rad'])):.3f} deg)")
    else:
        print("current vehicle ROS map pose: unavailable")
    print("===")
    print("destination vehicle ROS map pose")
    print(f"  position: x={pose['x']:.6f} y={pose['y']:.6f}")
    print(f"  yaw: {float(pose['yaw_rad']):.6f} rad ({math.degrees(float(pose['yaw_rad'])):.3f} deg)")
    print("===")
    print(f"  after vehicle rectangle: X/front={vehicle_length_x:.2f}m Y/left={vehicle_length_y:.2f}m, drawn as the blue rotated rectangle")
    print(f"  after vehicle center from arm base_link + car_center_from_arm_base_pb_xy: x={pose['x']:.3f} y={pose['y']:.3f} yaw={math.degrees(float(pose['yaw_rad'])):.1f}deg, drawn as the yellow cross")
    if before_pose is not None:
        before_in_map = before_px_raw is not None and 0.0 <= before_px_raw[0] < float(width_px) and 0.0 <= before_px_raw[1] < float(height_px)
        print(f"  before vehicle center from reference AMCL: x={before_pose['x']:.3f} y={before_pose['y']:.3f} yaw={math.degrees(float(before_pose['yaw_rad'])):.1f}deg, drawn as the green rectangle and white cross inside={before_in_map}")
        if before_arm_pose is not None:
            before_arm_in_map = before_arm_px_raw is not None and 0.0 <= before_arm_px_raw[0] < float(width_px) and 0.0 <= before_arm_px_raw[1] < float(height_px)
            print(f"  before arm base_link: x={before_arm_pose['x']:.3f} y={before_arm_pose['y']:.3f}, drawn as the magenta cross inside={before_arm_in_map}")
    else:
        print("  before vehicle skipped: no capture/reference AMCL pose is available")
    if base_pose is not None:
        print(f"  arm base_link: x={base_pose['x']:.3f} y={base_pose['y']:.3f}, drawn as the cyan cross")

    output_path = output_dir / f"{prefix}_ros_map.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path



def _draw_vehicle_pose_on_ros_map(
    draw: Any,
    *,
    pose: dict[str, float],
    footprint_local: Any,
    meta: Any,
    height_px: int,
    draw_scale: float,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
    cross_color: tuple[int, int, int, int],
    heading_color: tuple[int, int, int, int],
    polygon_width: int,
    cross_radius: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    footprint_world = coord.footprint_world_xy(footprint_local, pose=pose)
    footprint_px = [
        _scale_pixel(_ros_xy_to_pixel(point[0], point[1], meta, height_px), draw_scale)
        for point in footprint_world
    ]
    _draw_polygon(draw, footprint_px, fill=fill, outline=outline, width=polygon_width)
    center_px_raw = _ros_xy_to_pixel(pose["x"], pose["y"], meta, height_px)
    center_px = _scale_pixel(center_px_raw, draw_scale)
    _draw_cross(draw, center_px, radius=cross_radius, color=cross_color, width=1)
    heading_end = (
        pose["x"] + math.cos(float(pose["yaw_rad"])) * 0.5,
        pose["y"] + math.sin(float(pose["yaw_rad"])) * 0.5,
    )
    draw.line(
        [center_px, _scale_pixel(_ros_xy_to_pixel(heading_end[0], heading_end[1], meta, height_px), draw_scale)],
        fill=heading_color,
        width=1,
    )
    return center_px_raw, center_px


def _before_vehicle_ros_map_pose(pointcloud_info: dict[str, Any], config: dict[str, Any]) -> dict[str, float] | None:
    reference = pointcloud_info.get("capture_amcl_pose") or config.get("reference_amcl_pose")
    if reference is None:
        return None
    try:
        pose = coord.pose2d_from_any(reference)
    except Exception:
        pose = _pose2d_dict(reference)
    if pose is None:
        return None
    return {"x": float(pose["x"]), "y": float(pose["y"]), "yaw_rad": float(pose["yaw_rad"])}



def _arm_base_pose_from_vehicle_center_pose(
    vehicle_pose: dict[str, float] | None,
    config: dict[str, Any],
) -> dict[str, float] | None:
    if vehicle_pose is None:
        return None
    try:
        import numpy as np

        offset_pb = np.asarray(config.get("car_center_from_arm_base_pb_xy", [0.0, -0.1285]), dtype=np.float64).reshape(2)
        offset_ros = coord.arm_base_local_pb_offset_to_ros_map_delta(
            offset_pb,
            _initial_arm_base_yaw_rad(config),
            reference_ros_yaw_rad=float(vehicle_pose["yaw_rad"]),
        )
        return {
            "x": float(vehicle_pose["x"] - offset_ros[0]),
            "y": float(vehicle_pose["y"] - offset_ros[1]),
            "yaw_rad": float(vehicle_pose["yaw_rad"]),
        }
    except Exception:
        return None


def _car_center_pose_from_arm_base_pose(
    base_pose: dict[str, float] | None,
    config: dict[str, Any],
    *,
    reference_pose: dict[str, float] | None,
) -> dict[str, float] | None:
    if base_pose is None:
        return None
    import numpy as np

    reference_ros_yaw = float(reference_pose["yaw_rad"]) if reference_pose is not None else 0.0
    reference_pb_yaw = _initial_arm_base_yaw_rad(config)
    candidate_pb_yaw = coord.wrap_angle_rad(float(base_pose["yaw_rad"]) - reference_ros_yaw + reference_pb_yaw)
    offset_pb = np.asarray(config.get("car_center_from_arm_base_pb_xy", [0.0, -0.1285]), dtype=np.float64).reshape(2)
    offset_ros = coord.arm_base_local_pb_offset_to_ros_map_delta(
        offset_pb,
        candidate_pb_yaw,
        reference_ros_yaw_rad=reference_ros_yaw,
    )
    return {
        "x": float(base_pose["x"] + offset_ros[0]),
        "y": float(base_pose["y"] + offset_ros[1]),
        "yaw_rad": float(base_pose["yaw_rad"]),
    }


def _ros_xy_to_pixel(x: Any, y: Any, meta: Any, height_px: int) -> tuple[float, float]:
    px = (float(x) - float(meta.origin_xy[0])) / float(meta.resolution_m)
    py = float(height_px - 1) - ((float(y) - float(meta.origin_xy[1])) / float(meta.resolution_m))
    return float(px), float(py)


def _scale_pixel(point: tuple[float, float], scale: float) -> tuple[float, float]:
    return float(point[0]) * float(scale), float(point[1]) * float(scale)


def _pose2d_dict(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    if value.get("x") is None or value.get("y") is None:
        return None
    yaw = value.get("yaw_rad", value.get("yaw", value.get("yaw_deg", 0.0)))
    if "yaw_deg" in value and "yaw_rad" not in value and "yaw" not in value:
        yaw = math.radians(float(yaw))
    return {"x": float(value["x"]), "y": float(value["y"]), "yaw_rad": float(yaw)}


def _draw_polygon(
    draw: Any,
    points: list[tuple[float, float]],
    *,
    fill: tuple[int, int, int, int] | None = None,
    outline: tuple[int, int, int, int] | None = None,
    width: int = 1,
) -> None:
    if not points:
        return
    if fill is not None:
        draw.polygon(points, fill=fill)
    if outline is not None:
        draw.line(points + [points[0]], fill=outline, width=width)


def _draw_cross(draw: Any, center: tuple[float, float], *, radius: float, color: tuple[int, int, int, int], width: int) -> None:
    x, y = center
    draw.line([(x - radius, y), (x + radius, y)], fill=color, width=width)
    draw.line([(x, y - radius), (x, y + radius)], fill=color, width=width)


def _add_voxel_visuals(p: Any, voxel_centers: Any, voxel_size_m: float) -> None:
    import numpy as np

    centers = np.asarray(voxel_centers, dtype=np.float64).reshape(-1, 3)
    if len(centers) == 0:
        return
    half_extents = [float(voxel_size_m) * 0.5] * 3
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.85, 0.1, 0.08, 0.65])
    for center in centers:
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=center.astype(float).tolist(),
        )


def _add_target_marker(p: Any, position_xyz: list[float]) -> None:
    _add_sphere(p, position_xyz, radius=0.035, rgba=[1.0, 0.85, 0.0, 1.0])


def _add_original_grasp_marker(p: Any, position_xyz: Sequence[float]) -> None:
    _add_sphere(p, position_xyz, radius=0.026, rgba=[0.55, 0.55, 0.55, 1.0])


def _add_final_ee_marker(p: Any, position_xyz: Any) -> None:
    try:
        point = _vector3(position_xyz, "final_ee_position_xyz")
    except Exception:
        return
    _add_sphere(p, point, radius=0.022, rgba=[0.0, 0.9, 0.2, 1.0])


def _add_arm_base_marker(p: Any, position_xyz: list[float]) -> None:
    _add_sphere(p, position_xyz, radius=0.025, rgba=[0.1, 0.35, 1.0, 1.0])


def _add_sphere(p: Any, position_xyz: Sequence[float], *, radius: float, rgba: Sequence[float]) -> None:
    visual = p.createVisualShape(p.GEOM_SPHERE, radius=float(radius), rgbaColor=list(rgba))
    p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=visual, basePosition=[float(v) for v in position_xyz])


def _add_car_body_visual(p: Any, *, config: dict[str, Any], base_xyz: list[float], base_yaw_rad: float) -> None:
    import numpy as np

    length_x = float(config.get("vehicle_base_length_x_m", 0.33))
    length_y = float(config.get("vehicle_base_length_y_m", 0.35))
    height = float(config.get("debug_vehicle_body_height_m", 0.06))
    offset_xy = np.asarray(config.get("car_center_from_arm_base_pb_xy", [0.0, -0.1285]), dtype=np.float64).reshape(2)
    c = math.cos(float(base_yaw_rad))
    s = math.sin(float(base_yaw_rad))
    rotated = np.asarray([c * offset_xy[0] - s * offset_xy[1], s * offset_xy[0] + c * offset_xy[1]])
    center = [float(base_xyz[0] + rotated[0]), float(base_xyz[1] + rotated[1]), max(height * 0.5, 1e-3)]
    # Vehicle local/base_footprint +X/front maps to PyBullet local -Y/+Y;
    # vehicle local +Y/left maps to PyBullet local +X/-X.
    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[length_y * 0.5, length_x * 0.5, height * 0.5],
        rgbaColor=[0.05, 0.25, 1.0, 0.45],
    )
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual,
        basePosition=center,
        baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, float(base_yaw_rad)]),
    )


def _current_ee_pose(p: Any, robot_id: int, ee_link_index: int) -> tuple[list[float], Any]:
    import numpy as np

    ee_state = p.getLinkState(robot_id, int(ee_link_index), computeForwardKinematics=True)
    position = [float(v) for v in ee_state[4]]
    quat_xyzw = [float(v) for v in ee_state[5]]
    rotation = np.asarray(p.getMatrixFromQuaternion(quat_xyzw), dtype=np.float64).reshape(3, 3)
    return position, rotation


def _original_grasp_position_pb(
    solution: dict[str, Any],
    grasp_candidates_camera: Sequence[Any],
    config: dict[str, Any],
) -> list[float] | None:
    try:
        import numpy as np

        grasp_index = int(solution.get("grasp_index"))
        frame = str(config.get("grasp_pose_frame", "camera")).lower()
        for candidate in grasp_candidates_camera:
            if int(getattr(candidate, "index")) != grasp_index:
                continue
            position = np.asarray(getattr(candidate, "position_xyz"), dtype=np.float64).reshape(3)
            if frame in {"pybullet", "pb", "scene", "world"}:
                return [float(v) for v in position]
            if frame != "camera":
                return None
            rotation = np.asarray(getattr(candidate, "rotation_matrix"), dtype=np.float64).reshape(3, 3)
            position_pb, _ = coord.camera_grasp_pose_to_pybullet(
                position,
                rotation,
                camera_position_pb_xyz=base_sampler._camera_position_pb_xyz(config),
            )
            return [float(v) for v in position_pb]
    except Exception:
        return None
    return None


def _add_grasp_axes(p: Any, target_xyz: list[float], target_rotation_matrix: Any) -> None:
    _add_pose_axes(
        p,
        target_xyz,
        target_rotation_matrix,
        length=0.12,
        radius=0.0045,
    )


def _add_ee_axes(p: Any, ee_xyz: list[float], ee_rotation_matrix: Any) -> None:
    _add_pose_axes(
        p,
        ee_xyz,
        ee_rotation_matrix,
        length=0.10,
        radius=0.004,
    )


def _add_pose_axes(
    p: Any,
    origin_xyz: Sequence[float],
    rotation_matrix: Any,
    *,
    length: float,
    radius: float,
) -> None:
    try:
        import numpy as np

        origin = np.asarray(origin_xyz, dtype=np.float64).reshape(3)
        rot = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    except Exception:
        return

    colors = (
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.85, 0.0, 1.0],
        [0.0, 0.2, 1.0, 1.0],
    )
    for axis_idx, color in enumerate(colors):
        direction = rot[:, axis_idx]
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            continue
        end = origin + direction / norm * float(length)
        _add_cylinder_segment(p, origin, end, radius=float(radius), rgba=color)
        _add_sphere(p, end.astype(float).tolist(), radius=float(radius) * 1.8, rgba=color)


def _add_cylinder_segment(p: Any, start_xyz: Any, end_xyz: Any, *, radius: float, rgba: Sequence[float]) -> None:
    import numpy as np

    start = np.asarray(start_xyz, dtype=np.float64).reshape(3)
    end = np.asarray(end_xyz, dtype=np.float64).reshape(3)
    delta = end - start
    length = float(np.linalg.norm(delta))
    if length <= 1e-9:
        return
    center = ((start + end) * 0.5).astype(float).tolist()
    direction = delta / length
    visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=float(radius),
        length=length,
        rgbaColor=list(rgba),
    )
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual,
        basePosition=center,
        baseOrientation=_quat_from_z_axis(direction),
    )


def _quat_from_z_axis(direction_xyz: Any) -> list[float]:
    import numpy as np

    direction = np.asarray(direction_xyz, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    direction = direction / norm
    z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    dot = max(-1.0, min(1.0, float(np.dot(z_axis, direction))))
    if dot > 1.0 - 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    if dot < -1.0 + 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    cross = np.cross(z_axis, direction)
    scale = math.sqrt((1.0 + dot) * 2.0)
    inv_scale = 1.0 / scale
    quat = [
        float(cross[0] * inv_scale),
        float(cross[1] * inv_scale),
        float(cross[2] * inv_scale),
        float(scale * 0.5),
    ]
    return quat


def _arm_camera_target_and_distance(
    p: Any,
    np: Any,
    *,
    robot_id: int,
    target_xyz: Sequence[float],
    final_ee_xyz: Any,
) -> tuple[list[float], float]:
    arm_points = _robot_aabb_points(p, np, robot_id)
    finite_arm = arm_points[np.all(np.isfinite(arm_points), axis=1)]
    if len(finite_arm) == 0:
        center = np.asarray([0.0, 0.0, 0.2], dtype=np.float64)
        frame_points = [center.reshape(1, 3)]
    else:
        lower = np.min(finite_arm, axis=0)
        upper = np.max(finite_arm, axis=0)
        center = 0.5 * (lower + upper)
        frame_points = [finite_arm]

    # Keep the camera centered on the arm, but size the shot large enough to include grasp markers.
    try:
        frame_points.append(np.asarray(target_xyz, dtype=np.float64).reshape(1, 3))
    except Exception:
        pass
    try:
        frame_points.append(np.asarray(_vector3(final_ee_xyz, "final_ee_position_xyz"), dtype=np.float64).reshape(1, 3))
    except Exception:
        pass

    scene = np.vstack(frame_points)
    finite_scene = scene[np.all(np.isfinite(scene), axis=1)]
    if len(finite_scene) == 0:
        return center.astype(float).tolist(), 1.2
    radius = float(np.max(np.linalg.norm(finite_scene - center.reshape(1, 3), axis=1)))
    return center.astype(float).tolist(), max(0.75, radius * 2.4 + 0.25)


def _robot_aabb_points(p: Any, np: Any, robot_id: int) -> Any:
    points = []
    for link_index in range(-1, int(p.getNumJoints(robot_id))):
        try:
            lower, upper = p.getAABB(robot_id, link_index)
        except Exception:
            continue
        points.append(lower)
        points.append(upper)
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64).reshape(-1, 3)


def _render_ppm(
    p: Any,
    np: Any,
    output_path: Path,
    width: int,
    height: int,
    camera_target_position: Sequence[float],
    camera_distance: float,
    camera_yaw_deg: float,
    camera_pitch_deg: float,
) -> None:
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=list(camera_target_position),
        distance=float(camera_distance),
        yaw=float(camera_yaw_deg),
        pitch=float(camera_pitch_deg),
        roll=0.0,
        upAxisIndex=2,
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=float(width) / float(height),
        nearVal=0.02,
        farVal=5.0,
    )
    _, _, rgb_data, _, _ = p.getCameraImage(
        width=int(width),
        height=int(height),
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
    )
    rgba = np.asarray(rgb_data, dtype=np.uint8)
    if rgba.ndim == 1:
        rgba = rgba.reshape(int(height), int(width), 4)
    rgb = rgba[:, :, :3]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(f"P6\n{int(width)} {int(height)}\n255\n".encode("ascii"))
        handle.write(rgb.tobytes())


def _select_render_voxels(voxels: Any, target_xyz: Sequence[float], limit: int) -> Any:
    import numpy as np

    centers = np.asarray(voxels, dtype=np.float64).reshape(-1, 3)
    if int(limit or 0) <= 0 or len(centers) <= int(limit or 0):
        return centers
    target = np.asarray(target_xyz, dtype=np.float64).reshape(1, 3)
    keep = np.argsort(np.linalg.norm(centers - target, axis=1), kind="stable")[: int(limit)]
    return centers[keep]


def _solution_choice_label(solution: dict[str, Any], evaluation: dict[str, Any]) -> str:
    if solution is evaluation.get("selected_solution"):
        return "selected_solution"
    if solution is evaluation.get("closest_solution"):
        return "closest_solution"
    return f"sample_index_{solution.get('sample_index', 'unknown')}"


def _solution_summary(solution: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "sample_index",
        "grasp_index",
        "grasp_rank",
        "ik_feasible",
        "collision_free",
        "ee_position_error_m",
        "ee_orientation_error_deg",
        "ik_error_xyz_m",
        "pb_base_link_xyz",
        "pb_base_link_yaw_rad",
        "pb_base_link_yaw_deg",
        "goal_pose",
        "ros_map_amcl_pose",
        "ros_map_base_link_pose",
        "ros_map_feasible",
        "ros_map_check",
        "reference_amcl_pose",
        "reference_pb_yaw_rad",
        "reference_pb_yaw_deg",
        "car_center_from_arm_base_pb_xy",
        "target_xyz",
        "target_rotation_matrix",
        "final_ee_position_xyz",
        "ik_joint_solution_rad",
        "ik_joint_solution_deg",
    )
    return {key: solution.get(key) for key in keys if key in solution}


def _vector3(value: Any, label: str) -> list[float]:
    if value is None:
        raise ValueError(f"{label} is required.")
    vals = [float(item) for item in value]
    if len(vals) != 3:
        raise ValueError(f"{label} must have exactly 3 values, got {len(vals)}.")
    return vals


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_safe(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return str(value)


if __name__ == "__main__":
    main()
