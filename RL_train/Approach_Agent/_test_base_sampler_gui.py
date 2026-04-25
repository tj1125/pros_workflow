import math
import os
import time
import numpy as np

from src.pybullet_ompl import _add_debug_axes, _find_controllable_joints, _set_joint_positions_direct
from src.pybullet_smoke import _degrees_to_radians

def rank_rgba(rank: int) -> tuple[float, float, float, float]:
    palette = (
        (0.95, 0.42, 0.24, 0.95),
        (0.20, 0.72, 0.40, 0.95),
        (0.40, 0.40, 0.90, 0.95),
        (0.90, 0.70, 0.10, 0.95),
        (0.70, 0.30, 0.85, 0.95),
        (0.25, 0.80, 0.85, 0.95),
        (0.95, 0.50, 0.75, 0.95),
        (0.60, 0.85, 0.30, 0.95),
        (0.50, 0.40, 0.35, 0.95),
        (0.85, 0.85, 0.85, 0.95),
    )
    if rank < 1:
        return palette[-1]
    return palette[min(rank - 1, len(palette) - 2)]

def add_gui_sphere_marker(
    p_mod,
    position_xyz: list[float],
    rgba: tuple[float, float, float, float],
    *,
    radius: float,
) -> None:
    visual_shape = p_mod.createVisualShape(
        p_mod.GEOM_SPHERE,
        radius=float(radius),
        rgbaColor=list(rgba),
    )
    p_mod.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=visual_shape,
        basePosition=[float(position_xyz[0]), float(position_xyz[1]), float(position_xyz[2])],
    )

def animate_gui_ik_solution(
    p_mod,
    *,
    robot_id: int,
    controllable_joint_ids: list[int],
    planning_config,
    base_xyz: list[float],
    base_yaw_rad: float,
    joint_solution_rad: list[float],
) -> None:
    step_count = max(1, int(os.getenv("BASE_SAMPLER_GUI_ANIMATION_STEPS", "80")))
    frame_sleep_sec = max(0.0, float(os.getenv("BASE_SAMPLER_GUI_FRAME_SLEEP_SEC", "0.025")))
    joint_reset_rad = np.asarray(_degrees_to_radians(planning_config.joint_reset_deg), dtype=np.float64)
    joint_goal_rad = np.asarray(joint_solution_rad, dtype=np.float64)
    if len(joint_reset_rad) != len(joint_goal_rad):
        _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, joint_goal_rad)
        p_mod.stepSimulation()
        return

    p_mod.resetBasePositionAndOrientation(
        robot_id,
        base_xyz,
        p_mod.getQuaternionFromEuler([0.0, 0.0, float(base_yaw_rad)]),
    )
    for frame_index in range(step_count + 1):
        alpha = float(frame_index) / float(step_count)
        smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        joint_state = joint_reset_rad + (joint_goal_rad - joint_reset_rad) * smooth_alpha
        _set_joint_positions_direct(p_mod, robot_id, controllable_joint_ids, joint_state)
        p_mod.stepSimulation()
        if frame_sleep_sec > 0.0:
            time.sleep(frame_sleep_sec)

def spin_gui(p_mod, *, hold_seconds: float, time_step: float) -> None:
    if hold_seconds > 0.0:
        deadline = time.time() + float(hold_seconds)
        while p_mod.isConnected() and time.time() < deadline:
            p_mod.stepSimulation()
            time.sleep(min(max(float(time_step), 1e-3), 0.02))
        return

    try:
        while p_mod.isConnected():
            p_mod.stepSimulation()
            time.sleep(min(max(float(time_step), 1e-3), 0.02))
    except KeyboardInterrupt:
        return

def compute_gui_camera_view(
    *,
    planning_config,
    visualization_records: list[dict],
    best_view_solution: dict | None,
) -> tuple[list[float], float, float, float]:
    points = [
        np.asarray([0.0, 0.0, float(planning_config.initial_height)], dtype=np.float64)
    ]
    for record in visualization_records:
        points.append(np.asarray(record["target_pb"], dtype=np.float64).reshape(3))
    if best_view_solution is not None:
        points.append(np.asarray(best_view_solution["pb_base_link_xyz"], dtype=np.float64).reshape(3))
        final_ee_position = best_view_solution.get("final_ee_position_xyz")
        if final_ee_position is not None:
            points.append(np.asarray(final_ee_position, dtype=np.float64).reshape(3))

    finite_points = [point for point in points if np.all(np.isfinite(point))]
    if not finite_points:
        return [0.0, 0.0, float(planning_config.initial_height)], 1.2, 90.0, 0.0

    point_arr = np.vstack(finite_points)
    lower = np.min(point_arr, axis=0)
    upper = np.max(point_arr, axis=0)
    center = 0.5 * (lower + upper)
    extent = float(np.max(upper - lower))
    camera_distance = max(1.2, extent * 2.1 + 0.7)
    camera_yaw = 90.0
    if best_view_solution is not None:
        camera_yaw = math.degrees(float(best_view_solution.get("pb_base_link_yaw_rad", 0.0))) - 90.0

    yaw_override = os.getenv("BASE_SAMPLER_GUI_CAMERA_YAW_DEG", "").strip()
    if yaw_override:
        camera_yaw = float(yaw_override)
    camera_pitch = float(os.getenv("BASE_SAMPLER_GUI_CAMERA_PITCH_DEG", "0.0"))
    camera_yaw = ((float(camera_yaw) + 180.0) % 360.0) - 180.0
    return center.astype(float).tolist(), camera_distance, camera_yaw, camera_pitch

def visualize_feasible_ik_results_in_gui(
    *,
    p_mod,
    pybullet_data,
    planning_config,
    arm_config,
    voxels_pb: np.ndarray,
    visualization_records: list[dict],
) -> None:
    expected_joint_count = int(arm_config["pybullet"]["controllable_joints"])
    hold_seconds = float(os.getenv("BASE_SAMPLER_GUI_HOLD_SEC", "0.0"))
    max_base_markers_per_grasp = int(os.getenv("BASE_SAMPLER_GUI_MAX_MARKERS_PER_GRASP", "40"))
    client_id = None

    try:
        client_id = p_mod.connect(p_mod.GUI)
        if client_id < 0:
            print("[test_base_sampler] PyBullet GUI unavailable; skipping visualization.", flush=True)
            return

        p_mod.setAdditionalSearchPath(pybullet_data.getDataPath())
        p_mod.resetSimulation()
        p_mod.setGravity(0.0, 0.0, -9.8)
        p_mod.configureDebugVisualizer(p_mod.COV_ENABLE_GUI, 0)
        p_mod.loadURDF("plane.urdf")

        print(f"[test_base_sampler] GUI drawing {len(voxels_pb)} voxels...", flush=True)
        voxel_size = 0.05
        half_extents = [voxel_size / 2.0] * 3
        voxel_collision_shape = p_mod.createCollisionShape(p_mod.GEOM_BOX, halfExtents=half_extents)
        voxel_visual_shape = p_mod.createVisualShape(p_mod.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.8, 0.2, 0.2, 0.8])
        for voxel_center in voxels_pb:
            p_mod.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=voxel_collision_shape,
                baseVisualShapeIndex=voxel_visual_shape,
                basePosition=voxel_center.astype(float).tolist(),
            )

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

        best_view_record = None
        best_view_solution = None
        total_feasible_count = 0
        camera_target = [0.0, 0.0, planning_config.initial_height]
        if visualization_records:
            camera_target = list(np.asarray(visualization_records[0]["target_pb"], dtype=float))

        target_visual_shape = p_mod.createVisualShape(p_mod.GEOM_SPHERE, radius=0.03, rgbaColor=[1.0, 0.8, 0.0, 1.0])

        for record in visualization_records:
            rank = int(record["rank"])
            target_pb = np.asarray(record["target_pb"], dtype=float)
            target_quat_pb = np.asarray(record["target_quat_pb"], dtype=float)
            feasible_solutions = list(record["feasible_solutions"])
            closest_solution = record.get("closest_solution")
            total_feasible_count += len(feasible_solutions)

            p_mod.createMultiBody(baseMass=0.0, baseVisualShapeIndex=target_visual_shape, basePosition=target_pb.astype(float).tolist())
            _add_debug_axes(p_mod, target_pb.astype(float).tolist(), orientation_xyzw=target_quat_pb.astype(float).tolist(), axis_length=0.12, axis_width=1.8, label=f"TARGET G{rank} ({len(feasible_solutions)})")

            shown_solutions = sorted(
                feasible_solutions,
                key=lambda s: float(s.get("ee_position_error_m", float("inf"))),
            )
            if closest_solution is not None and not any(s.get("sample_index") == closest_solution.get("sample_index") for s in shown_solutions):
                shown_solutions.append(closest_solution)
            
            for i, sol in enumerate(shown_solutions):
                if i >= max_base_markers_per_grasp:
                    break
                is_closest = bool(sol.get("selected_as_closest", False))
                is_best = bool(sol.get("selected_as_best", False))
                if is_best:
                    marker_rgba = (0.2, 0.9, 0.2, 0.95)
                    marker_radius = 0.075
                    best_view_solution = sol
                    best_view_record = record
                elif is_closest:
                    marker_rgba = (0.9, 0.9, 0.2, 0.95)
                    marker_radius = 0.06
                else:
                    rgba = rank_rgba(rank)
                    marker_rgba = (rgba[0], rgba[1], rgba[2], min(0.3, rgba[3]))
                    marker_radius = 0.03

                add_gui_sphere_marker(p_mod, sol["pb_base_link_xyz"], marker_rgba, radius=marker_radius)

        camera_target, camera_distance, camera_yaw, camera_pitch = compute_gui_camera_view(
            planning_config=planning_config,
            visualization_records=visualization_records,
            best_view_solution=best_view_solution,
        )

        if best_view_solution is not None and best_view_record is not None:
            joint_sol = best_view_solution.get("ik_joint_solution_rad")
            if joint_sol is not None:
                animate_gui_ik_solution(
                    p_mod,
                    robot_id=robot_id,
                    controllable_joint_ids=controllable_joint_ids,
                    planning_config=planning_config,
                    base_xyz=best_view_solution["pb_base_link_xyz"],
                    base_yaw_rad=best_view_solution["pb_base_link_yaw_rad"],
                    joint_solution_rad=joint_sol,
                )

        p_mod.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target,
        )
        
        spin_gui(p_mod, hold_seconds=hold_seconds, time_step=1.0 / 240.0)
    except Exception as exc:
        print(f"[test_base_sampler] GUI visualization failed: {exc}", flush=True)
    finally:
        if client_id is not None:
            try:
                p_mod.disconnect(client_id)
            except Exception:
                pass
