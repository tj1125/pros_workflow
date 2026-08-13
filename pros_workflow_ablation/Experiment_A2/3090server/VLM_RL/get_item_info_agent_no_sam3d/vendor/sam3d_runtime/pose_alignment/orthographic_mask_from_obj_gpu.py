#!/usr/bin/env python3
"""Orthographic projection mask for an OBJ onto a plane in Unity coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_obj(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                idxs = []
                for p in parts[1:]:
                    v = p.split("/")[0]
                    if not v:
                        continue
                    idx = int(v)
                    idxs.append(idx)
                if len(idxs) >= 3:
                    faces.append(idxs)
    if not vertices or not faces:
        raise ValueError(f"OBJ missing vertices or faces: {path}")
    return np.array(vertices, dtype=float), triangulate_faces(faces, len(vertices))


def connected_components(tris: np.ndarray, vcount: int) -> List[np.ndarray]:
    adj = [set() for _ in range(vcount)]
    for i0, i1, i2 in tris:
        adj[i0].update([i1, i2])
        adj[i1].update([i0, i2])
        adj[i2].update([i0, i1])
    seen = np.zeros(vcount, dtype=bool)
    comps: List[np.ndarray] = []
    for start in range(vcount):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for n in adj[v]:
                if not seen[n]:
                    seen[n] = True
                    stack.append(n)
        comps.append(np.array(comp, dtype=int))
    return comps


def compute_face_normals(vertices: np.ndarray, tris: np.ndarray) -> np.ndarray:
    v0 = vertices[tris[:, 0]]
    v1 = vertices[tris[:, 1]]
    v2 = vertices[tris[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return normals / norms



def split_components_by_y(
    vertices: np.ndarray, tris: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    comps = connected_components(tris, len(vertices))
    parts: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for comp in comps:
        keep_vert = np.zeros(len(vertices), dtype=bool)
        keep_vert[comp] = True
        keep_tri = keep_vert[tris].all(axis=1)
        tris_kept = tris[keep_tri]
        if tris_kept.size == 0:
            continue
        used = np.unique(tris_kept)
        new_index = -np.ones(len(vertices), dtype=int)
        new_index[used] = np.arange(len(used))
        new_vertices = vertices[used]
        new_tris = new_index[tris_kept]
        y_min = float(new_vertices[:, 1].min())
        parts.append((new_vertices, new_tris, y_min))
    parts.sort(key=lambda item: item[2])
    return parts



def save_obj(path: Path, vertices: np.ndarray, tris: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in tris:
            i0, i1, i2 = tri + 1
            f.write(f"f {i0} {i1} {i2}\n")


def voxel_simplify(
    vertices: np.ndarray, tris: np.ndarray, voxel: float
) -> Tuple[np.ndarray, np.ndarray]:
    if voxel <= 0:
        return vertices, tris
    grid = np.floor(vertices / voxel).astype(np.int64)
    key = grid[:, 0] * 73856093 + grid[:, 1] * 19349663 + grid[:, 2] * 83492791
    _, unique_idx, inv = np.unique(key, return_index=True, return_inverse=True)
    new_vertices = vertices[unique_idx]
    new_tris = inv[tris]
    # Drop degenerate triangles
    keep = (
        (new_tris[:, 0] != new_tris[:, 1])
        & (new_tris[:, 0] != new_tris[:, 2])
        & (new_tris[:, 1] != new_tris[:, 2])
    )
    new_tris = new_tris[keep]
    return new_vertices, new_tris


def triangulate_faces(faces: Iterable[List[int]], vcount: int) -> np.ndarray:
    tris: List[List[int]] = []
    for face in faces:
        # Convert to 0-based indices, handle negative indices.
        idxs = []
        for idx in face:
            if idx < 0:
                idxs.append(vcount + idx)
            else:
                idxs.append(idx - 1)
        if len(idxs) == 3:
            tris.append(idxs)
        else:
            # fan triangulation
            for i in range(1, len(idxs) - 1):
                tris.append([idxs[0], idxs[i], idxs[i + 1]])
    return np.array(tris, dtype=int)


def build_plane_basis(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = normal / np.linalg.norm(normal)
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(n, up)) > 0.9:
        up = np.array([1.0, 0.0, 0.0])
    # Left-handed basis: reverse cross-product order.
    u = np.cross(n, up)
    u = u / np.linalg.norm(u)
    v = np.cross(u, n)
    v = v / np.linalg.norm(v)
    return n, u, v


def project_points(
    pts: np.ndarray, origin: np.ndarray, n: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    rel = pts - origin
    d = np.dot(rel, n)
    proj = rel - d[:, None] * n[None, :]
    x = np.dot(proj, u)
    y = np.dot(proj, v)
    return np.stack([x, y], axis=1)


def project_points_torch(
    pts: torch.Tensor,
    origin: torch.Tensor,
    n: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    rel = pts - origin
    d = (rel * n).sum(dim=1)
    proj = rel - d[:, None] * n[None, :]
    x = (proj * u).sum(dim=1)
    y = (proj * v).sum(dim=1)
    return torch.stack([x, y], dim=1)


def fit_to_image(
    pts2d: np.ndarray, width: int, height: int, padding: float
) -> Tuple[np.ndarray, float, np.ndarray]:
    min_xy = pts2d.min(axis=0)
    max_xy = pts2d.max(axis=0)
    extent = max_xy - min_xy
    extent = np.maximum(extent, 1e-6)
    pad_px = max(1, int(min(width, height) * padding))
    usable = np.array([width - 1 - 2 * pad_px, height - 1 - 2 * pad_px], dtype=float)
    scale = float(np.min(usable / extent))
    center = (min_xy + max_xy) * 0.5
    pts_centered = (pts2d - center) * scale
    img_center = np.array([width / 2.0, height / 2.0])
    pts_img = pts_centered + img_center
    return pts_img, scale, center


def rasterize_triangles(
    verts2d: np.ndarray, tris: np.ndarray, width: int, height: int
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for i0, i1, i2 in tris:
        p0 = verts2d[i0]
        p1 = verts2d[i1]
        p2 = verts2d[i2]
        min_xy = np.floor(np.minimum(np.minimum(p0, p1), p2)).astype(int)
        max_xy = np.ceil(np.maximum(np.maximum(p0, p1), p2)).astype(int)
        min_x = max(min_xy[0], 0)
        min_y = max(min_xy[1], 0)
        max_x = min(max_xy[0], width - 1)
        max_y = min(max_xy[1], height - 1)
        if min_x > max_x or min_y > max_y:
            continue
        xs = np.arange(min_x, max_x + 1)
        ys = np.arange(min_y, max_y + 1)
        xx, yy = np.meshgrid(xs, ys)
        pts = np.stack([xx + 0.5, yy + 0.5], axis=-1)

        v0 = p1 - p0
        v1 = p2 - p0
        v2 = pts - p0
        den = v0[0] * v1[1] - v0[1] * v1[0]
        if abs(den) < 1e-12:
            continue
        a = (v2[..., 0] * v1[1] - v2[..., 1] * v1[0]) / den
        b = (v0[0] * v2[..., 1] - v0[1] * v2[..., 0]) / den
        inside = (a >= 0) & (b >= 0) & (a + b <= 1)
        if not np.any(inside):
            continue
        mask[min_y : max_y + 1, min_x : max_x + 1][inside] = 255
    return mask


def rasterize_distance(
    verts2d: np.ndarray,
    tris: np.ndarray,
    values: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    dist = np.full((height, width), np.inf, dtype=float)
    for i0, i1, i2 in tris:
        p0 = verts2d[i0]
        p1 = verts2d[i1]
        p2 = verts2d[i2]
        min_xy = np.floor(np.minimum(np.minimum(p0, p1), p2)).astype(int)
        max_xy = np.ceil(np.maximum(np.maximum(p0, p1), p2)).astype(int)
        min_x = max(min_xy[0], 0)
        min_y = max(min_xy[1], 0)
        max_x = min(max_xy[0], width - 1)
        max_y = min(max_xy[1], height - 1)
        if min_x > max_x or min_y > max_y:
            continue
        xs = np.arange(min_x, max_x + 1)
        ys = np.arange(min_y, max_y + 1)
        xx, yy = np.meshgrid(xs, ys)
        pts = np.stack([xx + 0.5, yy + 0.5], axis=-1)

        v0 = p1 - p0
        v1 = p2 - p0
        v2 = pts - p0
        den = v0[0] * v1[1] - v0[1] * v1[0]
        if abs(den) < 1e-12:
            continue
        a = (v2[..., 0] * v1[1] - v2[..., 1] * v1[0]) / den
        b = (v0[0] * v2[..., 1] - v0[1] * v2[..., 0]) / den
        inside = (a >= 0) & (b >= 0) & (a + b <= 1)
        if not np.any(inside):
            continue
        w0 = 1.0 - a - b
        tri_dist = w0 * values[i0] + a * values[i1] + b * values[i2]
        block = dist[min_y : max_y + 1, min_x : max_x + 1]
        block_inside = block[inside]
        block[inside] = np.minimum(block_inside, tri_dist[inside])
    return dist


def save_mask(path: Path, mask: np.ndarray) -> None:
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(mask).save(path)
    except Exception:
        # Fallback to PGM if Pillow is unavailable.
        pgm_path = path.with_suffix(".pgm")
        header = f"P5\n{mask.shape[1]} {mask.shape[0]}\n255\n".encode("ascii")
        with pgm_path.open("wb") as f:
            f.write(header)
            f.write(mask.tobytes())
        print(f"Pillow not available, wrote PGM: {pgm_path}")


def load_grayscale(path: Path) -> np.ndarray:
    try:
        from PIL import Image  # type: ignore

        return np.array(Image.open(path).convert("L"))
    except Exception:
        try:
            import cv2  # type: ignore

            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to read image: {path}")
            return img
        except Exception as exc:
            raise RuntimeError(f"Unable to load image {path}: {exc}") from exc


def resize_nearest(img: np.ndarray, width: int, height: int) -> np.ndarray:
    if img.shape[0] == height and img.shape[1] == width:
        return img
    try:
        from PIL import Image  # type: ignore

        return np.array(Image.fromarray(img).resize((width, height), Image.NEAREST))
    except Exception:
        try:
            import cv2  # type: ignore

            return cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        except Exception as exc:
            raise RuntimeError(f"Unable to resize image: {exc}") from exc


def normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    vals = values.astype(np.float32)
    vmin = float(vals.min())
    vmax = float(vals.max())
    denom = max(vmax - vmin, 1e-6)
    out = (vals - vmin) / denom * 255.0
    return np.clip(out, 0, 255).round().astype(np.uint8)


def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def compute_depth_l1(
    proj_u8: np.ndarray, depth_u8: np.ndarray, mask: np.ndarray
) -> float:
    if not np.any(mask):
        return 0.0
    diff = np.abs(proj_u8.astype(np.int16) - depth_u8.astype(np.int16))
    mean_diff = float(diff[mask].mean())
    return 1.0 - (mean_diff / 255.0)


def rotate_vertices(vertices: torch.Tensor, ry_deg: float, rz_deg: float) -> torch.Tensor:
    device = vertices.device
    dtype = vertices.dtype
    ry = torch.deg2rad(torch.tensor(ry_deg, device=device, dtype=dtype))
    rz = torch.deg2rad(torch.tensor(rz_deg, device=device, dtype=dtype))
    cos_y = torch.cos(ry)
    sin_y = torch.sin(ry)
    cos_z = torch.cos(rz)
    sin_z = torch.sin(rz)
    zero = torch.zeros((), device=device, dtype=dtype)
    one = torch.ones((), device=device, dtype=dtype)
    # Left-handed rotation (Unity-style): flip the sign on sin terms.
    Ry = torch.stack(
        [
            torch.stack([cos_y, zero, -sin_y]),
            torch.stack([zero, one, zero]),
            torch.stack([sin_y, zero, cos_y]),
        ]
    )
    Rz = torch.stack(
        [
            torch.stack([cos_z, sin_z, zero]),
            torch.stack([-sin_z, cos_z, zero]),
            torch.stack([zero, zero, one]),
        ]
    )
    return vertices @ (Rz @ Ry).T


def project_to_mask_torch(
    vertices: torch.Tensor,
    tris: np.ndarray,
    origin: torch.Tensor,
    n: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    width: int,
    height: int,
    padding: float,
    grayscale: bool,
) -> Tuple[np.ndarray, float, np.ndarray]:
    verts2d_t = project_points_torch(vertices, origin, n, u, v)
    verts2d = verts2d_t.detach().cpu().numpy()
    verts_img, scale, center = fit_to_image(verts2d, width, height, padding)
    if grayscale:
        distances = ((vertices - origin) @ n).detach().cpu().numpy()
        dist_img = rasterize_distance(verts_img, tris, distances, width, height)
        mask = np.zeros((height, width), dtype=np.uint8)
        finite = np.isfinite(dist_img)
        if np.any(finite):
            dmin = float(dist_img[finite].min())
            dmax = float(dist_img[finite].max())
            if dmax - dmin > 1e-12:
                shifted = dist_img - dmin
                scaled = shifted / (dmax - dmin) * 255.0
                mask[finite] = np.clip(scaled[finite], 0, 255).round().astype(np.uint8)
        return mask, scale, center
    mask = rasterize_triangles(verts_img, tris, width, height)
    return mask, scale, center



def project_to_mask(
    vertices: np.ndarray,
    tris: np.ndarray,
    origin: np.ndarray,
    n: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    width: int,
    height: int,
    padding: float,
    grayscale: bool,
) -> np.ndarray:
    verts2d = project_points(vertices, origin, n, u, v)
    verts_img, _, _ = fit_to_image(verts2d, width, height, padding)
    if grayscale:
        distances = (vertices - origin) @ n
        dist_img = rasterize_distance(verts_img, tris, distances, width, height)
        mask = np.zeros((height, width), dtype=np.uint8)
        finite = np.isfinite(dist_img)
        if np.any(finite):
            dmin = float(dist_img[finite].min())
            dmax = float(dist_img[finite].max())
            if dmax - dmin > 1e-12:
                shifted = dist_img - dmin
                scaled = shifted / (dmax - dmin) * 255.0
                mask[finite] = np.clip(scaled[finite], 0, 255).round().astype(np.uint8)
        return mask
    return rasterize_triangles(verts_img, tris, width, height)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orthographic projection mask of an OBJ onto a plane."
    )
    parser.add_argument(
        "--obj",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "object.obj",
        help="Path to OBJ file",
    )
    parser.add_argument(
        "--out-obj",
        type=Path,
        default=PROJECT_ROOT
        / "pose_alignment"
        / "orthographic_mesh.obj",
        help="Path to save processed OBJ used for projection",
    )

    parser.add_argument("--width", type=int, default=64, help="Mask width")
    parser.add_argument("--height", type=int, default=64, help="Mask height")
    parser.add_argument(
        "--normal",
        type=float,
        nargs=3,
        default=[-0.8742383830716731, -0.4241241858417162, -0.2362751035304569],
        help="Plane normal (nx ny nz)",
    )
    parser.add_argument(
        "--origin",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Point on plane (x y z)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Padding ratio around projected bounds",
    )
    parser.add_argument(
        "--rotate-z",
        type=float,
        default=0.0,
        help="Rotation around Z axis in degrees (applied before projection)",
    )
    parser.add_argument(
        "--voxel",
        type=float,
        default=0.01,
        help="Voxel size for simple mesh simplification (0 disables)",
    )
    parser.add_argument(
        "--weight-shape",
        type=float,
        default=0.7,
        help="Weight for shape (IoU) term",
    )
    parser.add_argument(
        "--weight-depth",
        type=float,
        default=0.3,
        help="Weight for depth (L1) term",
    )
    parser.add_argument(
        "--scan-y-step",
        type=float,
        default=45,
        help="Scan Y rotation in degrees (0 disables). Example: 45",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (cuda or cpu)",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"

    align_dir = PROJECT_ROOT / "Camera_3D_Localization" / "result" / "seg_crop_depth"
    align_candidates = []
    for pattern in (
        "rgb_1_1*.png",
        "Camera_Room1_1*.png",
        "Camera_Room1_1_color_image_raw_compressed*.png",
    ):
        align_candidates = sorted(align_dir.glob(pattern))
        if align_candidates:
            break
    if not align_candidates:
        raise FileNotFoundError(
            f"No alignment depth image found for camera 1 under {align_dir}"
        )
    args.align_depth = align_candidates[0]
    args.grayscale = True
    args.out = PROJECT_ROOT / "Camera_3D_Localization" / "result" / "align_depth" / "align_depth.png"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    vertices_np, tris = parse_obj(args.obj)
    # Flip Y to match Unity coordinate system before splitting.
    vertices_np[:, 1] *= -1.0
    parts = split_components_by_y(vertices_np, tris)
    if len(parts) >= 1:
        base_vertices, base_tris, _ = parts[0]
        vertices_np, tris = base_vertices, base_tris
    else:
        print("Split failed to find any components; using full mesh.")
    vertices_np, tris = voxel_simplify(vertices_np, tris, args.voxel)
    normal = np.array(args.normal, dtype=float)
    normal[1] *= -1.0
    origin = np.array(args.origin, dtype=float)
    origin[1] *= -1.0
    n, u, v = build_plane_basis(normal)
    device = torch.device(args.device)
    vertices = torch.from_numpy(vertices_np.astype(np.float32)).to(device)
    origin_t = torch.from_numpy(origin.astype(np.float32)).to(device)
    n_t = torch.from_numpy(n.astype(np.float32)).to(device)
    u_t = torch.from_numpy(u.astype(np.float32)).to(device)
    v_t = torch.from_numpy(v.astype(np.float32)).to(device)
    grayscale = args.grayscale or args.align_depth is not None

    if args.scan_y_step > 0.0 and args.align_depth is not None:
        depth_img = load_grayscale(args.align_depth)
        depth_u8 = normalize_to_uint8(depth_img)
        best = None
        best_mask = None
        for angle in np.arange(0.0, 360.0 + 1e-6, args.scan_y_step):
            v_rot = rotate_vertices(vertices, angle, args.rotate_z)
            mask, _, _ = project_to_mask_torch(
                v_rot,
                tris,
                origin_t,
                n_t,
                u_t,
                v_t,
                args.width,
                args.height,
                args.padding,
                grayscale,
            )
            proj_u8 = normalize_to_uint8(mask)
            if depth_u8.shape != proj_u8.shape:
                proj_u8 = resize_nearest(
                    proj_u8, depth_u8.shape[1], depth_u8.shape[0]
                )
            depth_mask = depth_u8 > 0
            proj_mask = proj_u8 > 0
            shape_score = compute_iou(depth_mask, proj_mask)
            overlap = np.logical_and(depth_mask, proj_mask)
            depth_score = compute_depth_l1(proj_u8, depth_u8, overlap)
            score = args.weight_shape * shape_score + args.weight_depth * depth_score
            if best is None or score > best[0]:
                best = (score, angle, shape_score, depth_score)
                best_mask = mask
        if best is None or best_mask is None:
            raise RuntimeError("Scan failed to produce a valid score.")
        save_mask(args.out, best_mask)
        v_rot_best = rotate_vertices(vertices, best[1], args.rotate_z)
        v_rot_np = v_rot_best.detach().cpu().numpy().astype(float)
        v_rot_np[:, 1] *= -1.0
        save_obj(args.out_obj, v_rot_np, tris)
        print(f"saved_obj: {args.out_obj}")
        adjusted = (best[1] + 180.0) % 360.0
        print(f"best_angle_y: {adjusted:.3f}")
        print(f"shape_iou: {best[2]:.6f}")
        print(f"depth_l1: {best[3]:.6f}")
        print(f"score: {best[0]:.6f}")
        mask = best_mask
        scale = float("nan")
        center = np.array([float("nan"), float("nan")], dtype=float)
    else:
        v_rot = rotate_vertices(vertices, 0.0, args.rotate_z)
        mask, scale, center = project_to_mask_torch(
            v_rot,
            tris,
            origin_t,
            n_t,
            u_t,
            v_t,
            args.width,
            args.height,
            args.padding,
            grayscale,
        )
        save_mask(args.out, mask)
        v_rot_np = v_rot.detach().cpu().numpy().astype(float)
        v_rot_np[:, 1] *= -1.0
        save_obj(args.out_obj, v_rot_np, tris)
        print(f"saved_obj: {args.out_obj}")

    if args.align_depth is not None and args.scan_y_step <= 0.0:
        depth_img = load_grayscale(args.align_depth)
        depth_u8 = normalize_to_uint8(depth_img)
        proj_u8 = normalize_to_uint8(mask)
        if depth_u8.shape != proj_u8.shape:
            proj_u8 = resize_nearest(proj_u8, depth_u8.shape[1], depth_u8.shape[0])
        depth_mask = depth_u8 > 0
        proj_mask = proj_u8 > 0
        shape_score = compute_iou(depth_mask, proj_mask)
        overlap = np.logical_and(depth_mask, proj_mask)
        depth_score = compute_depth_l1(proj_u8, depth_u8, overlap)
        score = args.weight_shape * shape_score + args.weight_depth * depth_score
        print(f"shape_iou: {shape_score:.6f}")
        print(f"depth_l1: {depth_score:.6f}")
        print(f"score: {score:.6f}")

    print(f"obj: {args.obj}")
    print(f"out: {args.out}")
    print(f"normal: {n.tolist()}")
    print(f"origin: {origin.tolist()}")
    print(f"scale: {scale:.6f}")
    print(f"center_xy: {center.tolist()}")


if __name__ == "__main__":
    main()
