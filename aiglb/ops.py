"""Mesh assembly helpers: transforms, merges, arrays, mirroring, unwrapping."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import trimesh
from trimesh.visual import TextureVisuals

from .primitives import _pack

__all__ = [
    "transformed", "concat", "mirror", "linear_array", "radial_array", "box_project",
    "transform_matrix",
]


def transform_matrix(position=(0, 0, 0), rotation=(0, 0, 0), scale=1.0) -> np.ndarray:
    """4x4 matrix from position (m), rotation (degrees, XYZ) and scale.

    Rotation order: X (roll) then Y (pitch) then Z (yaw) — applied as R = Rz·Ry·Rx.
    """
    def _vec3(v, default=(0.0, 0.0, 0.0)):
        if isinstance(v, (int, float)):
            return np.array([float(v)] * 3, dtype=np.float64)
        arr = np.asarray(v, dtype=np.float64).ravel()
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3)
        if arr.shape[0] != 3:
            raise ValueError(f"expected a scalar or 3 components, got {v!r}")
        return arr

    pos = _vec3(position)
    scl = _vec3(scale, (1.0, 1.0, 1.0)) if not isinstance(scale, (int, float)) else np.full(3, float(scale))
    rx, ry, rz = np.deg2rad(_vec3(rotation))
    cx, sx_, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx_], [0, sx_, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    M = np.eye(4)
    M[:3, :3] = R * scl[None, :]  # scale along local axes, then rotate
    M[:3, 3] = pos
    return M


def _normal_matrix(M: np.ndarray) -> np.ndarray:
    """Rotation part only (normals don't carry translation); renormalized."""
    A = M[:3, :3]
    # proper normal matrix: inverse-transpose; fall back to plain A for scale-laden mats
    try:
        N = np.linalg.inv(A).T
        if np.isfinite(N).all():
            return N
    except np.linalg.LinAlgError:
        pass
    return A


def transformed(mesh: trimesh.Trimesh, position=(0, 0, 0), rotation=(0, 0, 0),
                scale=1.0) -> trimesh.Trimesh:
    """Return a copy of ``mesh`` transformed by position/rotation/scale."""
    M = transform_matrix(position, rotation, scale)
    return _apply(mesh, M)


def _apply(mesh: trimesh.Trimesh, M: np.ndarray) -> trimesh.Trimesh:
    v = trimesh.transform_points(mesh.vertices, M)
    vn = _get_normals(mesh) @ _normal_matrix(M).T
    ln = np.linalg.norm(vn, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    vn = vn / ln
    uv = np.asarray(mesh.visual.uv) if isinstance(mesh.visual, TextureVisuals) else None
    new = trimesh.Trimesh(vertices=v, faces=mesh.faces.copy(), vertex_normals=vn, process=False)
    if uv is not None:
        new.visual = TextureVisuals(uv=uv.copy())
    return new


def _get_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """Analytic normals from cache when present, else area-weighted (no scipy)."""
    try:
        cache = mesh._cache.cache
        if 'vertex_normals' in cache:
            n = np.asarray(cache['vertex_normals'], dtype=np.float64)
            if len(n) == len(mesh.vertices):
                return n
    except Exception:
        pass
    return _fallback_normals(mesh)


def _mesh_parts(mesh: trimesh.Trimesh):
    """(vertices, faces, normals, uv) with safe fallbacks."""
    normals = _get_normals(mesh)
    uv = np.asarray(mesh.visual.uv) if isinstance(mesh.visual, TextureVisuals) else None
    return (np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64),
            np.asarray(normals, dtype=np.float64),
            uv)


def _fallback_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    v = mesh.vertices
    f = mesh.faces
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    normals = np.zeros((len(v), 3))
    np.add.at(normals, f[:, 0], fn)
    np.add.at(normals, f[:, 1], fn)
    np.add.at(normals, f[:, 2], fn)
    ln = np.linalg.norm(normals, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return normals / ln


def concat(meshes: Sequence[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Merge multiple meshes into one (UVs preserved; missing UVs get box-projected)."""
    if not meshes:
        raise ValueError("concat() needs at least one mesh")
    if len(meshes) == 1 and isinstance(meshes[0].visual, TextureVisuals):
        m = meshes[0]
        v, f, n, uv = _mesh_parts(m)
        return _pack(v, f, n, uv)
    V, F, N, UV = [], [], [], []
    offset = 0
    has_uv_all = True
    parts = []
    for m in meshes:
        v, f, n, uv = _mesh_parts(m)
        if uv is None:
            has_uv_all = False
            uv = np.zeros((len(v), 2))
        parts.append((v, f, n, uv))
    for v, f, n, uv in parts:
        V.append(v)
        N.append(n)
        UV.append(uv)
        F.append(f + offset)
        offset += len(v)
    out = _pack(np.vstack(V), np.vstack(F), np.vstack(N), np.vstack(UV))
    if not has_uv_all:
        out = box_project(out, uv_scale=1.0)
    return out


def mirror(mesh: trimesh.Trimesh, axis: str = "x") -> trimesh.Trimesh:
    """Mirror a mesh across the plane perpendicular to ``axis`` ('x' | 'y' | 'z')."""
    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x', 'y' or 'z'")
    i = {"x": 0, "y": 1, "z": 2}[axis]
    v, f, n, uv = _mesh_parts(mesh)
    v = v.copy(); n = n.copy()
    v[:, i] *= -1.0
    n[:, i] *= -1.0
    f = f[:, ::-1].copy()  # flip winding to keep faces outward
    if uv is None:
        uv = np.zeros((len(v), 2))
    return _pack(v, f, n, uv)


def linear_array(mesh: trimesh.Trimesh, count: int, step=(1.0, 0.0, 0.0)) -> trimesh.Trimesh:
    """``count`` copies of ``mesh`` translated by ``step * k``."""
    count = int(count)
    if count < 1:
        raise ValueError("count must be >= 1")
    copies = [transformed(mesh, position=np.asarray(step, dtype=np.float64) * k) for k in range(count)]
    return concat(copies)


def radial_array(mesh: trimesh.Trimesh, count: int, radius: float,
                 axis: str = "y", phase: float = 0.0, spin: bool = True) -> trimesh.Trimesh:
    """``count`` copies of ``mesh`` placed on a circle of ``radius`` around ``axis``.

    ``spin=True`` also rotates each copy around the axis so it faces outward
    (gear teeth, spokes, columns). ``phase`` in degrees.
    """
    count = int(count)
    if count < 1:
        raise ValueError("count must be >= 1")
    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError("axis must be 'x', 'y' or 'z'")
    copies = []
    for k in range(count):
        ang = math.radians(phase) + 2 * math.pi * k / count
        deg = math.degrees(ang)
        if axis == "y":
            pos = (radius * math.cos(ang), 0.0, radius * math.sin(ang))
            rot = (0.0, -deg, 0.0) if spin else (0.0, 0.0, 0.0)
        elif axis == "x":
            pos = (0.0, radius * math.cos(ang), radius * math.sin(ang))
            rot = (-deg, 0.0, 0.0) if spin else (0.0, 0.0, 0.0)
        else:  # z
            pos = (radius * math.cos(ang), radius * math.sin(ang), 0.0)
            rot = (0.0, 0.0, deg) if spin else (0.0, 0.0, 0.0)
        copies.append(transformed(mesh, position=pos, rotation=rot))
    return concat(copies)


def box_project(mesh: trimesh.Trimesh, uv_scale: float = 1.0,
                uv_offset=(0.0, 0.0), uv_rotation: float = 0.0) -> trimesh.Trimesh:
    """Dominant-axis planar unwrap for arbitrary meshes (per-face, hard edges).

    ``uv_scale`` tiles per meter; ``uv_offset`` shifts UVs; ``uv_rotation``
    rotates them (radians).
    """
    v, f, n, _ = _mesh_parts(mesh)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    axis = np.argmax(np.abs(fn), axis=1)
    V = np.empty((len(f) * 3, 3))
    N = np.empty((len(f) * 3, 3))
    UV = np.empty((len(f) * 3, 2))
    F = np.arange(len(f) * 3).reshape(-1, 3)
    for k, face in enumerate(f):
        for j, vi in enumerate(face):
            V[k * 3 + j] = v[vi]
            N[k * 3 + j] = n[vi]
            p = v[vi]
            if axis[k] == 0:   # +/-X : project on ZY
                UV[k * 3 + j] = (p[2], p[1])
            elif axis[k] == 1:  # +/-Y : project on XZ
                UV[k * 3 + j] = (p[0], p[2])
            else:               # +/-Z : project on XY
                UV[k * 3 + j] = (p[0], p[1])
    if uv_rotation:
        c, s = math.cos(uv_rotation), math.sin(uv_rotation)
        R = np.array([[c, -s], [s, c]])
        UV = UV @ R.T
    UV *= uv_scale
    UV[:, 0] += uv_offset[0]
    UV[:, 1] += uv_offset[1]
    return _pack(V, F, N, UV)
