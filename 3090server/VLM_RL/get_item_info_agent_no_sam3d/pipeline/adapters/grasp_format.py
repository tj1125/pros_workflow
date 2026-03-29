from __future__ import annotations

import numpy as np


def grasp_orientation_group(grasp: np.ndarray) -> int:
    """Assign an orientation group (1-8) based on the XZ approach direction of the grasp."""
    position = grasp[:3, 3]
    proj_x = float(position[0])
    proj_z = float(position[2])
    if np.hypot(proj_x, proj_z) == 0.0:
        return 1
    angle_deg = (np.degrees(np.arctan2(proj_z, proj_x)) + 360.0) % 360.0
    return int(angle_deg // 45.0) + 1
