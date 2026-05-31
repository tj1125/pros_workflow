"""Grasp loading and base candidate sampling."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .src.geometry import coordinate_transforms as coord


@dataclass(frozen=True)
class GraspPoseCandidate:
    index: int
    rank: int
    grasp_confidence: float
    position_xyz: np.ndarray
    rotation_matrix: np.ndarray


def resolve_grasp_result_json_path(grasp_result_json_path: Path | None = None) -> Path:
    if grasp_result_json_path is not None:
        return Path(grasp_result_json_path).expanduser().resolve()

    env_override = os.getenv("GRASP_RESULT_JSON", "").strip()
    if env_override:
        return Path(env_override).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[2]
    matches = sorted(
        repo_root.glob("logs/sessions/**/step_*_grasp_agent.json"),
        key=lambda item: (item.stat().st_mtime, str(item)),
    )
    if matches:
        return matches[-1].resolve()
    raise FileNotFoundError(
        "No grasp result JSON found. Pass --grasp-json, set GRASP_RESULT_JSON, "
        f"or provide a session.sqlite containing grasp_result. searched={repo_root / 'logs' / 'sessions'}"
    )


def load_grasp_candidates_from_result_json(
    grasp_result_json_path: Path | None = None,
) -> tuple[list[GraspPoseCandidate], np.ndarray, Path]:
    json_path = resolve_grasp_result_json_path(grasp_result_json_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    candidates, gripper_midpoint = load_grasp_candidates_from_result_payload(
        payload,
        source_label=str(json_path),
    )
    return candidates, gripper_midpoint, json_path


def load_grasp_candidates_from_result_payload(
    payload: dict[str, object],
    *,
    source_label: str = "grasp_result_payload",
) -> tuple[list[GraspPoseCandidate], np.ndarray]:
    raw_result = payload.get("raw_result", payload)
    if not isinstance(raw_result, dict):
        raise ValueError(f"Grasp result payload must be a mapping: {source_label}")

    gripper_midpoint = _vector3(
        raw_result.get("gripper_midpoint_camera_xyz", [0.0, -0.0824, 0.023]),
        "gripper_midpoint_camera_xyz",
    )
    raw_candidates = raw_result.get("valid_grasp_poses_camera")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        best_grasp = raw_result.get("best_grasp_pose_camera")
        if not isinstance(best_grasp, dict):
            raise KeyError(f"No valid_grasp_poses_camera or best_grasp_pose_camera found in {source_label}")
        raw_candidates = [best_grasp]

    candidates: list[GraspPoseCandidate] = []
    for index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, dict):
            raise ValueError(f"Candidate #{index} in {source_label} must be a mapping.")
        position, rotation = _position_and_rotation(raw_candidate, index=index, source_label=source_label)
        candidates.append(
            GraspPoseCandidate(
                index=index,
                rank=int(raw_candidate.get("rank", index + 1)),
                grasp_confidence=float(raw_candidate.get("grasp_confidence", float("nan"))),
                position_xyz=position,
                rotation_matrix=rotation,
            )
        )
    candidates.sort(key=lambda item: item.rank)
    return candidates, gripper_midpoint


def sample_base_points(
    grasp_candidates: list[GraspPoseCandidate],
    *,
    min_backoff_m: float,
    max_backoff_m: float,
    step_m: float,
    yaw_span_deg: float,
    yaw_step_deg: float,
    max_samples: int,
) -> list[dict[str, object]]:
    distances = _distance_values(min_backoff_m, max_backoff_m, step_m)
    yaw_offsets = _yaw_offsets(yaw_span_deg, yaw_step_deg)
    samples: list[dict[str, object]] = []

    # Rank-major order: exhaust rank 1 first, then rank 2, and so on.
    # max_samples is intentionally per grasp rank so the top 10 grasps each get
    # a real attempt instead of rank 1 consuming the whole global budget.
    max_samples_per_grasp = int(max_samples)
    for grasp in grasp_candidates:
        grasp_sample_index = 0
        for distance in distances:
            for yaw_offset in yaw_offsets:
                base_xy, desired_yaw, yaw_rad = coord.grasp_backoff_base_pose(
                    grasp.position_xyz,
                    grasp.rotation_matrix,
                    backoff_distance_m=distance,
                    yaw_offset_rad=yaw_offset,
                )
                samples.append(
                    {
                        "sample_source": "grasp_backoff",
                        "sample_index": len(samples),
                        "grasp_sample_index": int(grasp_sample_index),
                        "grasp_index": int(grasp.index),
                        "grasp_rank": int(grasp.rank),
                        "grasp_confidence": float(grasp.grasp_confidence),
                        "target_xyz": grasp.position_xyz.astype(float).tolist(),
                        "target_rotation_matrix": grasp.rotation_matrix.astype(float).tolist(),
                        "pb_base_link_xyz": [float(base_xy[0]), float(base_xy[1]), 0.0],
                        "pb_base_link_yaw_rad": float(yaw_rad),
                        "pb_base_link_yaw_deg": float(coord.rad_to_deg(yaw_rad)),
                        "goal_pose": {
                            "x": float(base_xy[0]),
                            "y": float(base_xy[1]),
                            "z": 0.0,
                            "yaw_rad": float(yaw_rad),
                        },
                        "backoff_distance_m": float(distance),
                        "desired_yaw_rad": float(desired_yaw),
                        "desired_yaw_deg": float(coord.rad_to_deg(desired_yaw)),
                        "yaw_offset_rad": float(yaw_offset),
                        "yaw_offset_deg": float(coord.rad_to_deg(yaw_offset)),
                    }
                )
                grasp_sample_index += 1
                if max_samples_per_grasp > 0 and grasp_sample_index >= max_samples_per_grasp:
                    break
            if max_samples_per_grasp > 0 and grasp_sample_index >= max_samples_per_grasp:
                break
    return samples


def _position_and_rotation(
    candidate: dict[str, object],
    *,
    index: int,
    source_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    matrix_4x4 = candidate.get("matrix_4x4")
    if matrix_4x4 is not None:
        try:
            return coord.matrix4x4_position_rotation(matrix_4x4)
        except ValueError as exc:
            raise ValueError(f"Candidate #{index} matrix_4x4 must be 4x4 in {source_label}.") from exc

    position = _vector3(candidate.get("position"), f"candidate[{index}].position")
    if "rotation_matrix" in candidate:
        rotation = np.asarray(candidate["rotation_matrix"], dtype=np.float64)
    elif "quaternion_xyzw" in candidate:
        rotation = coord.quat_xyzw_to_matrix(candidate["quaternion_xyzw"])
    elif "orientation_xyzw" in candidate:
        rotation = coord.quat_xyzw_to_matrix(candidate["orientation_xyzw"])
    else:
        raise KeyError(
            f"Candidate #{index} in {source_label} needs rotation_matrix, quaternion_xyzw, "
            "orientation_xyzw, or matrix_4x4."
        )
    if rotation.shape != (3, 3):
        raise ValueError(f"Candidate #{index} rotation must be 3x3 in {source_label}.")
    return position, rotation.astype(np.float64)


def _vector3(value: object, name: str) -> np.ndarray:
    if value is None:
        raise KeyError(f"{name} is required.")
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values.")
    return arr


def _distance_values(min_backoff_m: float, max_backoff_m: float, step_m: float) -> list[float]:
    start = max(0.0, float(min_backoff_m))
    stop = max(start, float(max_backoff_m))
    step = max(float(step_m), 1e-3)
    values: list[float] = []
    value = start
    while value <= stop + 1e-9:
        values.append(float(value))
        value += step
    return values


def _yaw_offsets(yaw_span_deg: float, yaw_step_deg: float) -> list[float]:
    span = max(0.0, abs(float(yaw_span_deg)))
    step = max(abs(float(yaw_step_deg)), 0.5)
    offsets_deg = [0.0]
    value = step
    while value <= span + 1e-9:
        offsets_deg.extend([value, -value])
        value += step
    return coord.deg_sequence_to_rad(offsets_deg)
