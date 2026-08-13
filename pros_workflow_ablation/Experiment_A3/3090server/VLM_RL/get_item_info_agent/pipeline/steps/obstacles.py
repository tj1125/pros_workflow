from __future__ import annotations

import math

import numpy as np


def obstacle_cylinder_dimensions(length_m: float, height_m: float, min_size_m: float = 0.03) -> tuple[float, float]:
    diameter_m = max(float(length_m), min_size_m)
    height_m = max(float(height_m), min_size_m)
    return diameter_m, height_m


def sample_cylinder_surface(
    center_local: np.ndarray,
    diameter_m: float,
    height_m: float,
    samples_per_object: int,
) -> np.ndarray:
    radius = float(diameter_m) * 0.5
    half_height = float(height_m) * 0.5

    num_angles = max(16, int(math.sqrt(samples_per_object) * 4))
    num_heights = max(4, int(math.sqrt(samples_per_object)))
    num_radii = max(4, int(math.sqrt(samples_per_object)))

    angles = np.linspace(0.0, 2.0 * math.pi, num_angles, endpoint=False)
    ys = np.linspace(-half_height, half_height, num_heights)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    side_points = []
    for y in ys:
        ring = np.column_stack(
            [
                radius * cos_a,
                np.full_like(cos_a, y, dtype=float),
                radius * sin_a,
            ]
        )
        side_points.append(ring)

    cap_points = []
    radii = np.linspace(0.0, radius, num_radii)
    for y in (-half_height, half_height):
        for r in radii:
            disc = np.column_stack(
                [
                    r * cos_a,
                    np.full_like(cos_a, y, dtype=float),
                    r * sin_a,
                ]
            )
            cap_points.append(disc)

    return np.vstack(side_points + cap_points) + np.asarray(center_local, dtype=float)


def build_cylindrical_obstacle_scene(
    objects: list[dict[str, object]],
    target_label: str,
    target_center_world: np.ndarray,
    samples_per_object: int,
) -> np.ndarray:
    """Build obstacle cylinders in a Unity-local frame centered at the target 3D position."""
    target_center_world = np.asarray(target_center_world, dtype=float)
    scene_points: list[np.ndarray] = []

    for obj in objects:
        if str(obj.get("label", "")) == target_label:
            continue

        center_world = np.asarray(obj["center_world_unity"], dtype=float)
        diameter_m, height_m = obstacle_cylinder_dimensions(
            float(obj["obstacle_diameter_m"]),
            float(obj["obstacle_height_m"]),
        )
        center_local = center_world - target_center_world
        scene_points.append(
            sample_cylinder_surface(
                center_local,
                diameter_m=diameter_m,
                height_m=height_m,
                samples_per_object=samples_per_object,
            )
        )

    if not scene_points:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(scene_points)
