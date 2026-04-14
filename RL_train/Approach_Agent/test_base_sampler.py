import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pybullet as p
import scipy.spatial.transform as st

from scripts.run_base_pose_sampling import load_config
from src.camera_car_voxel_ompl import (
    _align_points_to_ee_anchor,
    _transform_camera_points_to_pybullet_basis,
    _transform_camera_rotation_to_pybullet_basis,
    load_camera_car_voxel_ompl_config,
)
from src.geometry.depth_backprojection import backproject_depth_to_points, decode_depth_png_bytes
from src.geometry.voxelization import crop_points_to_workspace, voxelize_points
from src.io.camera_capture import capture_rgbd_snapshot
from src.io.intrinsics import load_camera_intrinsics
from src.pybullet_ompl import (
    _add_debug_axes,
    _find_controllable_joints,
    _set_joint_positions_direct,
    load_planning_config,
)
from src.pybullet_smoke import _degrees_to_radians, _load_arm_config, _load_python_dependencies


@dataclass(frozen=True)
class GraspPoseCandidate:
    index: int
    rank: int
    grasp_confidence: float
    grasp_distance_to_gripper_midpoint_m: float
    grasp_distance_to_camera_m: float
    position_camera_xyz: np.ndarray
    rotation_camera: np.ndarray


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _preferred_grasp_result_json() -> Path:
    return (
        _repo_root()
        / "logs"
        / "sessions"
        / "fbbc240a8ec8476ea09b15d0d3339ac9"
        / "artifacts"
        / "raw_results"
        / "step_007_grasp_agent.json"
    )


def _find_latest_grasp_result_json() -> Path:
    candidates = sorted(
        (_repo_root() / "logs" / "sessions").glob("*/artifacts/raw_results/step_*_grasp_agent.json"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("No step_*_grasp_agent.json was found under VLM_RL/logs/sessions.")
    return candidates[-1]


def _resolve_grasp_result_json_path(grasp_result_json_path: Path | None = None) -> Path:
    env_override = os.getenv("GRASP_RESULT_JSON", "").strip()
    if grasp_result_json_path is not None:
        return grasp_result_json_path.expanduser().resolve()
    if env_override:
        return Path(env_override).expanduser().resolve()
    preferred = _preferred_grasp_result_json()
    if preferred.exists():
        return preferred.resolve()
    return _find_latest_grasp_result_json()


def _load_grasp_candidates_from_result_json(
    grasp_result_json_path: Path | None = None,
) -> tuple[list[GraspPoseCandidate], np.ndarray, Path]:
    json_path = _resolve_grasp_result_json_path(grasp_result_json_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    raw_result = payload.get("raw_result", payload)
    gripper_midpoint_camera_xyz = np.asarray(
        raw_result.get("gripper_midpoint_camera_xyz", [0.0, -0.04, 0.11]),
        dtype=np.float64,
    )
    if gripper_midpoint_camera_xyz.shape != (3,):
        raise ValueError(
            "gripper_midpoint_camera_xyz must have shape (3,), "
            f"got {gripper_midpoint_camera_xyz.shape}."
        )

    raw_candidates = raw_result.get("valid_grasp_poses_camera")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        best_grasp = raw_result.get("best_grasp_pose_camera")
        if not isinstance(best_grasp, dict):
            raise KeyError(f"No valid_grasp_poses_camera or best_grasp_pose_camera found in {json_path}")
        raw_candidates = [best_grasp]

    grasp_candidates: list[GraspPoseCandidate] = []
    for index, candidate in enumerate(raw_candidates):
        matrix_4x4 = candidate.get("matrix_4x4")
        if matrix_4x4 is not None:
            grasp_matrix_camera = np.asarray(matrix_4x4, dtype=np.float64)
            if grasp_matrix_camera.shape != (4, 4):
                raise ValueError(
                    f"Candidate matrix_4x4 must have shape (4, 4), got {grasp_matrix_camera.shape}."
                )
            position_camera_xyz = np.asarray(grasp_matrix_camera[:3, 3], dtype=np.float64)
            rotation_camera = np.asarray(grasp_matrix_camera[:3, :3], dtype=np.float64)
        else:
            position_camera_xyz = np.asarray(candidate["position"], dtype=np.float64)
            rotation_camera = np.asarray(candidate["rotation_matrix"], dtype=np.float64)

        if position_camera_xyz.shape != (3,):
            raise ValueError(
                f"Candidate position must have shape (3,), got {position_camera_xyz.shape}."
            )
        if rotation_camera.shape != (3, 3):
            raise ValueError(
                f"Candidate rotation_matrix must have shape (3, 3), got {rotation_camera.shape}."
            )

        grasp_candidates.append(
            GraspPoseCandidate(
                index=index,
                rank=int(candidate.get("rank", index + 1)),
                grasp_confidence=float(candidate.get("grasp_confidence", float("nan"))),
                grasp_distance_to_gripper_midpoint_m=float(
                    candidate.get("grasp_distance_to_gripper_midpoint_m", float("nan"))
                ),
                grasp_distance_to_camera_m=float(candidate.get("grasp_distance_to_camera_m", float("nan"))),
                position_camera_xyz=position_camera_xyz,
                rotation_camera=rotation_camera,
            )
        )

    grasp_candidates.sort(key=lambda item: item.rank)
    return grasp_candidates, gripper_midpoint_camera_xyz, json_path


def _get_reset_end_effector_pose(planner_config_path: Path) -> tuple[np.ndarray, np.ndarray]:
    planning_config = load_planning_config(planner_config_path)
    arm_config = _load_arm_config()
    _, p_mod, pybullet_data = _load_python_dependencies()
    expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])
    client_id: int | None = None

    try:
        client_id = p_mod.connect(p_mod.DIRECT)
        p_mod.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_mod.resetSimulation()
        p_mod.setGravity(0.0, 0.0, -9.8)
        p_mod.loadURDF("plane.urdf")

        base_orientation_rad = [math.radians(v) for v in planning_config.base_orientation_euler_deg]
        base_orientation_xyzw = p_mod.getQuaternionFromEuler(base_orientation_rad)
        robot_id = p_mod.loadURDF(
            planning_config.urdf_path,
            useFixedBase=True,
            basePosition=[0.0, 0.0, planning_config.initial_height],
            baseOrientation=base_orientation_xyzw,
        )
        controllable_joint_ids, _ = _find_controllable_joints(p_mod, robot_id, expected_joint_count)
        joint_reset_rad = _degrees_to_radians(planning_config.joint_reset_deg)
        _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, joint_reset_rad)
        p_mod.performCollisionDetection()

        if planning_config.ee_link_index >= p_mod.getNumJoints(robot_id):
            raise ValueError(f"ee_link_index {planning_config.ee_link_index} is out of range.")

        ee_state = p_mod.getLinkState(
            robot_id,
            planning_config.ee_link_index,
            computeForwardKinematics=True,
        )
        ee_position = np.asarray(ee_state[0], dtype=np.float64)
        ee_quaternion_xyzw = np.asarray(ee_state[1], dtype=np.float64)
        return ee_position, ee_quaternion_xyzw
    finally:
        if client_id is not None:
            try:
                p_mod.disconnect(client_id)
            except Exception:
                pass


def _voxel_downsample_points(points_xyz: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if len(points_xyz) == 0 or voxel_size_m <= 0.0:
        return np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    buckets = np.floor(np.asarray(points_xyz, dtype=np.float32) / float(voxel_size_m)).astype(np.int32)
    _, keep_indices = np.unique(buckets, axis=0, return_index=True)
    return np.asarray(points_xyz[np.sort(keep_indices)], dtype=np.float32)


def _capture_live_scene_voxels(
    camera_config_path: Path,
    ee_anchor_pb: np.ndarray,
    gripper_midpoint_camera_xyz: np.ndarray,
) -> np.ndarray:
    camera_cfg = load_camera_car_voxel_ompl_config(camera_config_path)
    snapshot = capture_rgbd_snapshot(
        camera_cfg.camera_name,
        timeout_sec=camera_cfg.capture_timeout_sec,
        amcl_topic=camera_cfg.amcl_topic,
    )
    intrinsics = load_camera_intrinsics(Path(camera_cfg.intrinsics_path))
    depth_metric_m = decode_depth_png_bytes(snapshot.depth_bytes)
    if camera_cfg.flip_depth_vertical:
        depth_metric_m = np.flipud(depth_metric_m).copy()

    points_camera = backproject_depth_to_points(
        depth_metric_m,
        intrinsics.k,
        min_depth_m=0.14,
        max_depth_m=3.0,
        pixel_stride=4,
    )
    if camera_cfg.crop_to_workspace_camera:
        points_camera = crop_points_to_workspace(points_camera, camera_cfg.workspace_bounds_camera)
    points_camera = _voxel_downsample_points(points_camera, voxel_size_m=0.01)
    if len(points_camera) == 0:
        raise RuntimeError("Live Camera_Car RGBD produced zero valid camera-frame points.")

    gripper_midpoint_pb = _transform_camera_points_to_pybullet_basis(
        np.asarray(gripper_midpoint_camera_xyz, dtype=np.float32).reshape(1, 3)
    ).reshape(3)
    points_camera_pb = _transform_camera_points_to_pybullet_basis(points_camera)
    points_pybullet, _ = _align_points_to_ee_anchor(
        points_camera_pb,
        gripper_midpoint_pybullet_xyz=gripper_midpoint_pb,
        ee_anchor_pybullet_xyz=ee_anchor_pb,
    )

    voxel_centers_pb = voxelize_points(
        points_pybullet,
        voxel_size_m=camera_cfg.voxel_size_m,
        max_voxels=camera_cfg.max_voxel_obstacles,
    )
    if len(voxel_centers_pb) == 0:
        raise RuntimeError("Live Camera_Car RGBD produced zero occupied voxels.")
    return np.asarray(voxel_centers_pb, dtype=np.float64)


def _transform_grasp_pose_camera_to_pybullet(
    grasp_candidate: GraspPoseCandidate,
    *,
    scene_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_pb = (
        _transform_camera_points_to_pybullet_basis(grasp_candidate.position_camera_xyz.reshape(1, 3)).reshape(3)
        + scene_translation
    )
    target_rot_pb = _transform_camera_rotation_to_pybullet_basis(grasp_candidate.rotation_camera)
    target_quat_pb = st.Rotation.from_matrix(target_rot_pb).as_quat()
    return (
        np.asarray(target_pb, dtype=np.float64),
        np.asarray(target_rot_pb, dtype=np.float64),
        np.asarray(target_quat_pb, dtype=np.float64),
    )


def _robot_collides_with_obstacles(
    p_mod,
    robot_id: int,
    obstacle_body_ids: list[int],
) -> bool:
    for obstacle_body_id in obstacle_body_ids:
        if p_mod.getClosestPoints(robot_id, obstacle_body_id, distance=0.0):
            return True
    return False


def _sample_feasible_ik_solutions(
    *,
    p_mod,
    robot_id: int,
    controllable_joint_ids: list[int],
    obstacle_body_ids: list[int],
    planning_config,
    target_pb: np.ndarray,
    target_rot_pb: np.ndarray,
    target_quat_pb: np.ndarray,
    rng_seed: int,
    num_samples: int = 500,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(int(rng_seed))
    valid_candidates: list[dict[str, object]] = []
    joint_reset_rad = _degrees_to_radians(planning_config.joint_reset_deg)
    target_pos = target_pb.tolist()

    for sample_index in range(int(num_samples)):
        distance = float(rng.uniform(0.20, 0.48))
        angle_offset = float(rng.uniform(math.radians(-10), math.radians(10)))

        v_x = target_rot_pb[:, 0]
        v_x_xy = np.array([v_x[0], v_x[1]], dtype=np.float64)
        if np.linalg.norm(v_x_xy) > 1e-4:
            v_x_xy = v_x_xy / np.linalg.norm(v_x_xy)
        else:
            v_x_xy = np.array([1.0, 0.0], dtype=np.float64)

        ray_angle = math.atan2(float(v_x_xy[1]), float(v_x_xy[0]))
        theta = ray_angle + angle_offset

        pb_bx = float(target_pos[0] - distance * math.cos(theta))
        pb_by = float(target_pos[1] - distance * math.sin(theta))
        pb_yaw = float(math.atan2(target_pos[1] - pb_by, target_pos[0] - pb_bx))

        t_base_pb = np.eye(4, dtype=np.float64)
        t_base_pb[:3, :3] = st.Rotation.from_euler("z", pb_yaw).as_matrix()
        t_base_pb[0, 3] = pb_bx
        t_base_pb[1, 3] = pb_by

        t_offset = np.eye(4, dtype=np.float64)
        t_offset[1, 3] = 0.1288
        t_offset[2, 3] = 0.071

        t_amcl_pb = t_base_pb @ np.linalg.inv(t_offset)
        amcl_pb_x = float(t_amcl_pb[0, 3])
        amcl_pb_y = float(t_amcl_pb[1, 3])
        amcl_pb_yaw = float(st.Rotation.from_matrix(t_amcl_pb[:3, :3]).as_euler("zyx")[0])

        ros_amcl_x = amcl_pb_y
        ros_amcl_y = -amcl_pb_x
        ros_amcl_yaw = amcl_pb_yaw

        new_amcl_pb_x = -ros_amcl_y
        new_amcl_pb_y = ros_amcl_x
        new_amcl_pb_yaw = ros_amcl_yaw

        t_amcl_pb_new = np.eye(4, dtype=np.float64)
        t_amcl_pb_new[:3, :3] = st.Rotation.from_euler("z", new_amcl_pb_yaw).as_matrix()
        t_amcl_pb_new[0, 3] = new_amcl_pb_x
        t_amcl_pb_new[1, 3] = new_amcl_pb_y

        t_base_pb_new = t_amcl_pb_new @ t_offset
        final_pb_bx = float(t_base_pb_new[0, 3])
        final_pb_by = float(t_base_pb_new[1, 3])
        final_pb_yaw = float(st.Rotation.from_matrix(t_base_pb_new[:3, :3]).as_euler("zyx")[0])

        p_mod.resetBasePositionAndOrientation(
            robot_id,
            [final_pb_bx, final_pb_by, planning_config.initial_height],
            p_mod.getQuaternionFromEuler([0.0, 0.0, final_pb_yaw]),
        )

        _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, joint_reset_rad)
        ik_joint_poses = None
        for _ in range(3):
            ik_joint_poses = p_mod.calculateInverseKinematics(
                robot_id,
                planning_config.ee_link_index,
                target_pos,
                target_quat_pb.tolist(),
                maxNumIterations=5000,
                residualThreshold=1e-4,
            )
            _set_joint_positions_direct(
                p_mod,
                robot_id,
                controllable_joint_ids,
                ik_joint_poses[: len(controllable_joint_ids)],
            )

        p_mod.performCollisionDetection()
        final_state = p_mod.getLinkState(
            robot_id,
            planning_config.ee_link_index,
            computeForwardKinematics=True,
        )
        final_pos = np.asarray(final_state[4], dtype=np.float64)
        dist_error = float(np.linalg.norm(target_pb - final_pos))
        collision_free = not _robot_collides_with_obstacles(p_mod, robot_id, obstacle_body_ids)

        if dist_error < 0.015 and collision_free and ik_joint_poses is not None:
            joint_solution_rad = np.asarray(ik_joint_poses[: len(controllable_joint_ids)], dtype=np.float64)
            joint_solution_deg = np.degrees(joint_solution_rad)
            valid_candidates.append(
                {
                    "sample_index": sample_index,
                    "pb_base_link_xyz": [final_pb_bx, final_pb_by, float(planning_config.initial_height)],
                    "pb_base_link_yaw_rad": final_pb_yaw,
                    "pb_base_link_yaw_deg": float(math.degrees(final_pb_yaw)),
                    "ros_map_amcl_xy": [float(ros_amcl_x), float(ros_amcl_y)],
                    "ros_map_amcl_yaw_rad": float(ros_amcl_yaw),
                    "ros_map_amcl_yaw_deg": float(math.degrees(ros_amcl_yaw)),
                    "ee_position_error_m": dist_error,
                    "ik_joint_solution_rad": joint_solution_rad.astype(float).tolist(),
                    "ik_joint_solution_deg": joint_solution_deg.astype(float).tolist(),
                }
            )

    return valid_candidates


def _timestamped_output_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    base_config_path = Path("configs/base_pose_sampling.yaml")
    camera_config_path = Path("configs/camera_car_voxel_ompl.yaml")

    cfg = load_config(base_config_path)
    planning_config = load_planning_config(Path(cfg["planner_config_path"]))
    arm_config = _load_arm_config()
    _, p_mod, pybullet_data = _load_python_dependencies()
    ee_anchor_pb, _ = _get_reset_end_effector_pose(Path(cfg["planner_config_path"]))
    grasp_candidates, gripper_midpoint_camera_xyz, grasp_result_json_path = _load_grasp_candidates_from_result_json()

    voxels_pb = _capture_live_scene_voxels(
        camera_config_path.resolve(),
        ee_anchor_pb,
        gripper_midpoint_camera_xyz,
    )
    gripper_midpoint_pb = _transform_camera_points_to_pybullet_basis(
        gripper_midpoint_camera_xyz.reshape(1, 3)
    ).reshape(3)
    scene_translation = ee_anchor_pb - gripper_midpoint_pb

    print(
        f"[test_base_sampler] grasp_json={grasp_result_json_path} | "
        f"grasps={len(grasp_candidates)} | live RGBD voxels={len(voxels_pb)}",
        flush=True,
    )

    client_id = p_mod.connect(p_mod.DIRECT)
    p_mod.setAdditionalSearchPath(pybullet_data.getDataPath())
    p_mod.resetSimulation()
    p_mod.setGravity(0.0, 0.0, -9.8)
    p_mod.loadURDF("plane.urdf")

    voxel_size = 0.05
    half_extents = [voxel_size / 2.0] * 3
    col_shape = p_mod.createCollisionShape(p_mod.GEOM_BOX, halfExtents=half_extents)
    vis_shape = p_mod.createVisualShape(
        p_mod.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=[0.8, 0.2, 0.2, 0.8],
    )
    obstacle_body_ids: list[int] = []
    for voxel_center in voxels_pb:
        obstacle_body_ids.append(
            p_mod.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_shape,
                baseVisualShapeIndex=vis_shape,
                basePosition=voxel_center.tolist(),
            )
        )

    robot_id = p_mod.loadURDF(
        planning_config.urdf_path,
        useFixedBase=True,
        basePosition=[0.0, 0.0, planning_config.initial_height],
        baseOrientation=p_mod.getQuaternionFromEuler([0.0, 0.0, 0.0]),
    )
    expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])
    controllable_joint_ids, _ = _find_controllable_joints(p_mod, robot_id, expected_joint_count)

    output_dir = _timestamped_output_dir(Path("outputs/test_base_sampler_all_grasps"))
    report: dict[str, object] = {
        "grasp_result_json_path": str(grasp_result_json_path),
        "num_grasp_candidates": len(grasp_candidates),
        "live_voxel_count": int(len(voxels_pb)),
        "scene_translation_xyz": np.asarray(scene_translation, dtype=float).tolist(),
        "gripper_midpoint_camera_xyz": np.asarray(gripper_midpoint_camera_xyz, dtype=float).tolist(),
        "results": [],
    }

    for grasp_candidate in grasp_candidates:
        target_pb, target_rot_pb, target_quat_pb = _transform_grasp_pose_camera_to_pybullet(
            grasp_candidate,
            scene_translation=scene_translation,
        )
        _add_debug_axes(
            p_mod,
            target_pb.astype(float).tolist(),
            orientation_xyzw=target_quat_pb.astype(float).tolist(),
            label=f"TARGET_{grasp_candidate.rank}",
        )
        feasible_solutions = _sample_feasible_ik_solutions(
            p_mod=p_mod,
            robot_id=robot_id,
            controllable_joint_ids=controllable_joint_ids,
            obstacle_body_ids=obstacle_body_ids,
            planning_config=planning_config,
            target_pb=target_pb,
            target_rot_pb=target_rot_pb,
            target_quat_pb=target_quat_pb,
            rng_seed=42 + grasp_candidate.rank,
            num_samples=500,
        )

        print(
            f"[grasp rank={grasp_candidate.rank:02d}] "
            f"conf={grasp_candidate.grasp_confidence:.4f} "
            f"dist_to_midpoint={grasp_candidate.grasp_distance_to_gripper_midpoint_m:.4f}m "
            f"-> feasible_ik={len(feasible_solutions)}",
            flush=True,
        )

        report["results"].append(
            {
                "index": grasp_candidate.index,
                "rank": grasp_candidate.rank,
                "grasp_confidence": grasp_candidate.grasp_confidence,
                "grasp_distance_to_gripper_midpoint_m": grasp_candidate.grasp_distance_to_gripper_midpoint_m,
                "grasp_distance_to_camera_m": grasp_candidate.grasp_distance_to_camera_m,
                "position_camera_xyz": grasp_candidate.position_camera_xyz.astype(float).tolist(),
                "rotation_camera_matrix": grasp_candidate.rotation_camera.astype(float).tolist(),
                "target_position_pybullet_xyz": target_pb.astype(float).tolist(),
                "target_rotation_pybullet_matrix": target_rot_pb.astype(float).tolist(),
                "feasible_ik_count": len(feasible_solutions),
                "feasible_ik_solutions": feasible_solutions,
            }
        )

    report_path = output_dir / "all_grasps_ik_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[test_base_sampler] Report saved to: {report_path}", flush=True)
    p_mod.disconnect(client_id)


if __name__ == "__main__":
    main()
