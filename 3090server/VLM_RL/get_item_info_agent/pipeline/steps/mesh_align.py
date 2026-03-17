from __future__ import annotations

from typing import Any

import numpy as np
import torch

from get_item_info_agent.pipeline.adapters.sam3d_adapter import Sam3DInference


def _reindex_submesh(vertices: np.ndarray, tris: np.ndarray, keep_face_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tris_kept = tris[keep_face_mask]
    if tris_kept.size == 0:
        raise RuntimeError("Submesh became empty while trimming the SAM3D mesh.")
    used = np.unique(tris_kept)
    new_index = -np.ones(len(vertices), dtype=int)
    new_index[used] = np.arange(len(used))
    return vertices[used], new_index[tris_kept]


def _trim_rectangular_base_by_y_profile(
    vertices: np.ndarray,
    tris: np.ndarray,
    *,
    scan_from_high_y: bool = True,
    max_base_fraction: float = 0.35,
    num_slices: int = 18,
    stable_area_ratio: float = 0.82,
    area_drop_ratio: float = 0.62,
    center_shift_ratio: float = 0.30,
    min_slice_points: int = 12,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool | int]]:
    """Remove a support block by scanning Y slices for a stable footprint followed by a sharp area drop."""
    if len(vertices) == 0 or len(tris) == 0:
        return vertices, tris, {"base_removed": False, "base_component_count": 0}

    y = vertices[:, 1]
    y_min = float(y.min())
    y_max = float(y.max())
    total_height = y_max - y_min
    if total_height <= 1e-6:
        return vertices, tris, {"base_removed": False, "base_component_count": 0}

    face_centroids = vertices[tris].mean(axis=1)
    probe_points = np.vstack([vertices, face_centroids])
    if scan_from_high_y:
        min_considered_y = y_max - total_height * max_base_fraction
        if min_considered_y >= y_max:
            return vertices, tris, {"base_removed": False, "base_component_count": 0}
        slice_edges = np.linspace(y_max, min_considered_y, num_slices + 1)
    else:
        max_considered_y = y_min + total_height * max_base_fraction
        if max_considered_y <= y_min:
            return vertices, tris, {"base_removed": False, "base_component_count": 0}
        slice_edges = np.linspace(y_min, max_considered_y, num_slices + 1)

    slice_stats: list[dict[str, float | np.ndarray] | None] = []
    for idx in range(num_slices):
        if scan_from_high_y:
            high = slice_edges[idx]
            low = slice_edges[idx + 1]
            if idx == num_slices - 1:
                mask = (probe_points[:, 1] >= low) & (probe_points[:, 1] <= high)
            else:
                mask = (probe_points[:, 1] > low) & (probe_points[:, 1] <= high)
        else:
            low = slice_edges[idx]
            high = slice_edges[idx + 1]
            if idx == num_slices - 1:
                mask = (probe_points[:, 1] >= low) & (probe_points[:, 1] <= high)
            else:
                mask = (probe_points[:, 1] >= low) & (probe_points[:, 1] < high)
        pts = probe_points[mask]
        if len(pts) < min_slice_points:
            slice_stats.append(None)
            continue

        x_span = float(pts[:, 0].max() - pts[:, 0].min())
        z_span = float(pts[:, 2].max() - pts[:, 2].min())
        area = x_span * z_span
        center_xz = np.array([pts[:, 0].mean(), pts[:, 2].mean()], dtype=float)
        slice_stats.append(
            {
                "low": low,
                "high": high,
                "area": area,
                "center_xz": center_xz,
                "x_span": x_span,
                "z_span": z_span,
            }
        )

    valid_indices = [idx for idx, stat in enumerate(slice_stats) if stat is not None]
    if not valid_indices:
        return vertices, tris, {"base_removed": False, "base_component_count": 0}

    bottom_idx = valid_indices[0]
    bottom_stat = slice_stats[bottom_idx]
    assert bottom_stat is not None
    reference_area = max(float(bottom_stat["area"]), 1e-8)
    reference_span = max(float(bottom_stat["x_span"]), float(bottom_stat["z_span"]), 1e-8)
    reference_center = np.asarray(bottom_stat["center_xz"], dtype=float)

    stable_until = bottom_idx
    for idx in valid_indices[1:]:
        stat = slice_stats[idx]
        assert stat is not None
        area_ratio = float(stat["area"]) / reference_area
        center_shift = float(np.linalg.norm(np.asarray(stat["center_xz"], dtype=float) - reference_center))
        if area_ratio >= stable_area_ratio and center_shift <= reference_span * center_shift_ratio:
            stable_until = idx
            continue
        break

    cut_idx = None
    for idx in valid_indices:
        if idx <= stable_until:
            continue
        stat = slice_stats[idx]
        assert stat is not None
        area_ratio = float(stat["area"]) / reference_area
        if area_ratio <= area_drop_ratio:
            cut_idx = idx
            break

    if cut_idx is None:
        return vertices, tris, {"base_removed": False, "base_component_count": 0}

    cut_y = float(slice_edges[cut_idx])
    if scan_from_high_y:
        base_height = y_max - cut_y
    else:
        base_height = cut_y - y_min
    if base_height < total_height * 0.03 or base_height > total_height * max_base_fraction:
        return vertices, tris, {"base_removed": False, "base_component_count": 0}

    face_centroid_y = face_centroids[:, 1]
    if scan_from_high_y:
        keep_face_mask = face_centroid_y < cut_y
    else:
        keep_face_mask = face_centroid_y > cut_y
    removed_faces = int((~keep_face_mask).sum())
    kept_faces = int(keep_face_mask.sum())
    if removed_faces == 0 or kept_faces == 0:
        return vertices, tris, {"base_removed": False, "base_component_count": 0}

    trimmed_vertices, trimmed_tris = _reindex_submesh(vertices, tris, keep_face_mask)
    return trimmed_vertices, trimmed_tris, {
        "base_removed": True,
        "base_removed_faces": removed_faces,
        "base_cut_y": cut_y,
        "base_height_fraction": base_height / total_height,
        "base_removed_from_high_y": bool(scan_from_high_y),
    }


def _trim_rectangular_base_auto(
    vertices: np.ndarray,
    tris: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool | int]]:
    search_fractions = (0.35, 0.5, 0.65)
    for max_fraction in search_fractions:
        out_vertices, out_tris, info = _trim_rectangular_base_by_y_profile(
            vertices,
            tris,
            scan_from_high_y=True,
            max_base_fraction=max_fraction,
        )
        if bool(info.get("base_removed", False)):
            info = dict(info)
            info["base_search_fraction"] = float(max_fraction)
            return out_vertices, out_tris, info

    return vertices, tris, {"base_removed": False}


def reconstruct_mesh_with_sam3d(
    sam3d_config_path,
    sam3d_image_rgb: np.ndarray,
    sam3d_mask_bool: np.ndarray,
    sam_seed: int,
    device: str,
):
    """Run SAM3D to reconstruct a 3D mesh from an RGBA crop."""
    inference = Sam3DInference(sam3d_config_path, compile_model=False, device=device)
    sam_output = inference.run(sam3d_image_rgb, sam3d_mask_bool, seed=int(sam_seed))
    mesh = sam_output.get("glb")
    if mesh is None:
        raise RuntimeError("SAM3D output missing mesh ('glb').")
    return mesh


def scale_mesh_to_target_y(mesh: Any, target_y: float) -> Any:
    """Uniformly scale the mesh so its Y extent matches the given target height."""
    if target_y <= 0:
        return mesh.copy()
    bounds = mesh.bounds
    current_y = bounds[1, 1] - bounds[0, 1]
    if current_y <= 0:
        return mesh.copy()
    factor = target_y / current_y
    mesh_scaled = mesh.copy()
    mesh_scaled.apply_scale(factor)
    return mesh_scaled


def _scale_vertices_to_target_y(vertices: np.ndarray, target_y: float) -> tuple[np.ndarray, float]:
    if target_y <= 0 or len(vertices) == 0:
        return np.array(vertices, copy=True), 1.0
    current_y = float(vertices[:, 1].max() - vertices[:, 1].min())
    if current_y <= 0:
        return np.array(vertices, copy=True), 1.0
    factor = float(target_y) / current_y
    return np.asarray(vertices, dtype=float) * factor, factor


def align_mesh_with_depth(
    mesh: Any,
    depth_u8: np.ndarray,
    alignment_cfg: dict,
    device: str,
    target_y: float | None = None,
) -> tuple[Any, float, dict[str, float], np.ndarray]:
    """Trim the base, scale the remaining mesh to the target Y height, then align around Y."""
    import trimesh
    from pose_alignment import orthographic_mask_from_obj_gpu as align_mod  # type: ignore

    vertices = np.array(mesh.vertices, dtype=float)
    tris = np.array(mesh.faces, dtype=int)
    if vertices.size == 0 or tris.size == 0:
        raise RuntimeError("Mesh has no vertices/faces for alignment.")

    vertices[:, 1] *= -1.0
    vertices, tris, base_trim_info = _trim_rectangular_base_auto(vertices, tris)
    vertices, post_base_scale_factor = _scale_vertices_to_target_y(vertices, float(target_y or 0.0))
    vertices, tris = align_mod.voxel_simplify(vertices, tris, float(alignment_cfg["voxel"]))

    normal = np.array(alignment_cfg["normal"], dtype=float)
    origin = np.array(alignment_cfg["origin"], dtype=float)
    normal[1] *= -1.0
    origin[1] *= -1.0
    n, u, v = align_mod.build_plane_basis(normal)

    vertices_t = torch.from_numpy(vertices.astype(np.float32)).to(device)
    origin_t = torch.from_numpy(origin.astype(np.float32)).to(device)
    n_t = torch.from_numpy(n.astype(np.float32)).to(device)
    u_t = torch.from_numpy(u.astype(np.float32)).to(device)
    v_t = torch.from_numpy(v.astype(np.float32)).to(device)

    best = None
    best_mask = None
    for angle in np.arange(0.0, 360.0 + 1e-6, float(alignment_cfg["scan_y_step"])):
        v_rot = align_mod.rotate_vertices(vertices_t, float(angle), float(alignment_cfg["rotate_z"]))
        proj_mask, _, _ = align_mod.project_to_mask_torch(
            v_rot,
            tris,
            origin_t,
            n_t,
            u_t,
            v_t,
            int(alignment_cfg["width"]),
            int(alignment_cfg["height"]),
            float(alignment_cfg["padding"]),
            grayscale=True,
        )
        proj_u8 = align_mod.normalize_to_uint8(proj_mask)
        if proj_u8.shape != depth_u8.shape:
            proj_u8 = align_mod.resize_nearest(proj_u8, depth_u8.shape[1], depth_u8.shape[0])

        depth_mask = depth_u8 > 0
        proj_mask_bool = proj_u8 > 0
        shape_score = align_mod.compute_iou(depth_mask, proj_mask_bool)
        overlap = np.logical_and(depth_mask, proj_mask_bool)
        depth_score = align_mod.compute_depth_l1(proj_u8, depth_u8, overlap)
        score = (
            float(alignment_cfg["weight_shape"]) * shape_score
            + float(alignment_cfg["weight_depth"]) * depth_score
        )

        if best is None or score > best[0]:
            best = (score, float(angle), shape_score, depth_score)
            best_mask = proj_u8

    if best is None or best_mask is None:
        raise RuntimeError("Failed to compute alignment score.")

    best_angle_raw = best[1]
    adjusted_angle = (best_angle_raw + 180.0) % 360.0

    v_rot_best = align_mod.rotate_vertices(vertices_t, best_angle_raw, float(alignment_cfg["rotate_z"]))
    v_rot_np = v_rot_best.detach().cpu().numpy().astype(float)
    v_rot_np[:, 1] *= -1.0

    aligned_mesh = trimesh.Trimesh(vertices=v_rot_np, faces=tris, process=False)
    metrics = {
        "score": float(best[0]),
        "shape_iou": float(best[2]),
        "depth_l1": float(best[3]),
        "base_removed": bool(base_trim_info.get("base_removed", False)),
        "base_removed_faces": int(base_trim_info.get("base_removed_faces", 0)),
        "base_height_fraction": float(base_trim_info.get("base_height_fraction", 0.0)),
        "post_base_scale_factor": float(post_base_scale_factor),
    }
    if "base_cut_y" in base_trim_info:
        metrics["base_cut_y"] = float(base_trim_info["base_cut_y"])
    if "base_removed_from_high_y" in base_trim_info:
        metrics["base_removed_from_high_y"] = bool(base_trim_info["base_removed_from_high_y"])
    if "base_search_fraction" in base_trim_info:
        metrics["base_search_fraction"] = float(base_trim_info["base_search_fraction"])
    return aligned_mesh, float(adjusted_angle), metrics, best_mask
