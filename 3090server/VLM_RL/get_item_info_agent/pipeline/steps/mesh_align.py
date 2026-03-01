from __future__ import annotations

from typing import Any

import numpy as np
import torch

from get_item_info_agent.pipeline.adapters.sam3d_adapter import Sam3DInference


def reconstruct_mesh_with_sam3d(
    sam3d_config_path,
    sam3d_image_rgb: np.ndarray,
    sam3d_mask_bool: np.ndarray,
    sam_seed: int,
):
    """Run SAM3D to reconstruct a 3D mesh from an RGBA crop."""
    inference = Sam3DInference(sam3d_config_path, compile_model=False)
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


def align_mesh_with_depth(
    mesh: Any,
    depth_u8: np.ndarray,
    alignment_cfg: dict,
    device: str,
) -> tuple[Any, float, dict[str, float], np.ndarray]:
    """Rotate the mesh around Y to best match the depth image silhouette via IoU + L1 scoring."""
    import trimesh
    from pose_alignment import orthographic_mask_from_obj_gpu as align_mod  # type: ignore

    vertices = np.array(mesh.vertices, dtype=float)
    tris = np.array(mesh.faces, dtype=int)
    if vertices.size == 0 or tris.size == 0:
        raise RuntimeError("Mesh has no vertices/faces for alignment.")

    vertices[:, 1] *= -1.0
    parts = align_mod.split_components_by_y(vertices, tris)
    if parts:
        vertices, tris, _ = parts[0]
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
    }
    return aligned_mesh, float(adjusted_angle), metrics, best_mask
