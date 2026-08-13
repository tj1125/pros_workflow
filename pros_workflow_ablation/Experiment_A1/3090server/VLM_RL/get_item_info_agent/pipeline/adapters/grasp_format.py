from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh.transformations as tra
import yaml

from get_item_info_agent.pipeline.types import Grasp


def grasp_orientation_group(grasp: np.ndarray) -> int:
    """Assign an orientation group (1–8) based on the XZ approach direction of the grasp."""
    position = grasp[:3, 3]
    proj_x = float(position[0])
    proj_z = float(position[2])
    if np.hypot(proj_x, proj_z) == 0.0:
        return 1
    angle_deg = (np.degrees(np.arctan2(proj_z, proj_x)) + 360.0) % 360.0
    return int(angle_deg // 45.0) + 1


def save_to_isaac_grasp_format(grasps: np.ndarray, confidences: np.ndarray, output_path: Path) -> dict:
    """Serialize grasps to the Isaac Sim grasp YAML format."""
    data = {"format": "isaac_grasp", "format_version": 1.0, "grasps": {}}
    conf_list = confidences.tolist()
    assert len(grasps) == len(conf_list)

    for i, (grasp, confidence) in enumerate(zip(grasps, conf_list)):
        xyz = grasp[:3, 3].tolist()
        q = tra.quaternion_from_matrix(grasp[:3, :3])
        qw = float(q[0])
        qxyz = q[1:].tolist()
        group = grasp_orientation_group(grasp)

        data["grasps"][f"grasp_{i}"] = {
            "group": group,
            "confidence": confidence,
            "position": xyz,
            "orientation": {"w": qw, "xyz": qxyz},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if len(grasps) > 0:
        output_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")
    return data


def quaternion_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Convert a unit quaternion to a 3×3 rotation matrix."""
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        raise ValueError("Zero-norm quaternion.")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def load_grasps(path: Path) -> list[Grasp]:
    """Load grasps from an Isaac Sim grasp YAML file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    grasps = data.get("grasps") if isinstance(data, dict) else None
    if not isinstance(grasps, dict):
        raise ValueError(f"Invalid grasp_info format: {path}")

    out: list[Grasp] = []
    for name, g in grasps.items():
        pos = np.array(g["position"], dtype=float)
        qw = float(g["orientation"]["w"])
        qx, qy, qz = [float(v) for v in g["orientation"]["xyz"]]
        out.append(
            Grasp(
                name=name,
                position=pos,
                rotation=quaternion_to_matrix(qw, qx, qy, qz),
                confidence=float(g["confidence"]),
                group=int(g["group"]),
            )
        )
    return out
