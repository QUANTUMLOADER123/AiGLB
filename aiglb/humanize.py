"""Humanizer: makes CG-perfect shapes read as hand-made.

Three tools:

* :func:`rounded_box` — a box with every edge and corner rounded. Stops models
  looking like razor-edged CG cubes. Controlled by ``sharpness`` (2 = pebble,
  5 = classic rounded box, 9+ = nearly sharp).
* :func:`superellipsoid` — the underlying general form (rounded ellipsoid /
  rounded-box / pebble family).
* :func:`displace` / :func:`humanize` — low-amplitude organic noise over any
  mesh (plus optional tessellation so box-like primitives can wobble).

None of this needs external assets; everything is computed.
"""
from __future__ import annotations

import numpy as np
import trimesh
from trimesh.visual import TextureVisuals

__all__ = ["rounded_box", "superellipsoid", "displace", "humanize", "subdivide"]


def superellipsoid(radius=(0.5, 0.5, 0.5), sharpness: float = 5.0,
                   subdivisions: int = 3) -> trimesh.Trimesh:
    """Superellipsoid (Lamé surface) — the anti-cube.

    Parameters
    ----------
    radius : (rx, ry, rz) half-extents in meters
    sharpness : exponent n. 2.0 = ellipsoid, ~4-6 = rounded box,
                >= 8.0 = crisp rounded box. Higher = sharper edges.
    subdivisions : mesh density. Interior latitude bands = 12 + 8 * subdivisions,
                   longitude = 4x that (subdivisions=3 → 36 x 144).
    """
    rx, ry, rz = (float(v) for v in radius)
    n = float(sharpness)
    if n < 1.0:
        raise ValueError("sharpness must be >= 1.0 (2.0 = ellipsoid)")
    e = 2.0 / n

    def spow(p):
        return np.sign(p) * np.abs(p) ** e

    bands = max(6, 6 + 4 * int(subdivisions))
    lon = bands * 4
    us = -np.pi / 2.0 + (np.arange(1, bands + 1) * np.pi / (bands + 1))  # interior, exclude poles
    vs = np.arange(lon) * (2.0 * np.pi / lon)
    cu = spow(np.cos(us))
    su = spow(np.sin(us))
    cv = spow(np.cos(vs))
    sv = spow(np.sin(vs))

    verts = [[0.0, -ry, 0.0]]  # 0 = south pole
    for i in range(bands):
        for j in range(lon):
            verts.append([rx * cu[i] * cv[j], ry * su[i], rz * cu[i] * sv[j]])
    verts.append([0.0, ry, 0.0])  # last = north pole
    north = len(verts) - 1

    def row(i):
        return 1 + i * lon

    faces = []
    # south fan
    for j in range(lon):
        faces.append((0, row(0) + j, row(0) + (j + 1) % lon))
    # middle quads
    for i in range(bands - 1):
        for j in range(lon):
            j1 = (j + 1) % lon
            a, b = row(i) + j, row(i) + j1
            c, d = row(i + 1) + j, row(i + 1) + j1
            faces.append((a, c, b))
            faces.append((b, c, d))
    # north fan
    last = row(bands - 1)
    for j in range(lon):
        faces.append((north, last + (j + 1) % lon, last + j))

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    # outward normals from the implicit gradient  F(p) = sum |p_i/h_i|^n = 1
    h = np.array([rx, ry, rz])
    with np.errstate(divide="ignore", invalid="ignore"):
        grad = (np.sign(verts) * (np.abs(verts) / h) ** (n - 1.0)) / h
    ln = np.linalg.norm(grad, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    normals = grad / ln
    return trimesh.Trimesh(vertices=verts, faces=faces,
                           vertex_normals=normals, process=False)


def rounded_box(width: float, height: float, depth: float, *,
                sharpness: float = 5.0, subdivisions: int = 3) -> trimesh.Trimesh:
    """Box with rounded edges/corners. ``sharpness``: 5 typical, 8+ crisper."""
    return superellipsoid((width / 2.0, height / 2.0, depth / 2.0),
                          sharpness=sharpness, subdivisions=subdivisions)


def subdivide(mesh: trimesh.Trimesh, iterations: int = 1) -> trimesh.Trimesh:
    """Split every triangle into 4 per iteration."""
    out = mesh
    for _ in range(int(iterations)):
        out = out.subdivide()
    return out


def _value_noise3(grid: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.random((grid, grid, grid))


def _trilinear(lattice: np.ndarray, p: np.ndarray) -> np.ndarray:
    g = lattice.shape[0]
    p = np.mod(p, g)
    i0 = np.floor(p).astype(np.int64)
    f = p - i0
    i1 = (i0 + 1) % g
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    sx = fx * fx * (3 - 2 * fx)
    sy = fy * fy * (3 - 2 * fy)
    sz = fz * fz * (3 - 2 * fz)
    c000 = lattice[i0[..., 0], i0[..., 1], i0[..., 2]]
    c100 = lattice[i1[..., 0], i0[..., 1], i0[..., 2]]
    c010 = lattice[i0[..., 0], i1[..., 1], i0[..., 2]]
    c110 = lattice[i1[..., 0], i1[..., 1], i0[..., 2]]
    c001 = lattice[i0[..., 0], i0[..., 1], i1[..., 2]]
    c101 = lattice[i1[..., 0], i0[..., 1], i1[..., 2]]
    c011 = lattice[i0[..., 0], i1[..., 1], i1[..., 2]]
    c111 = lattice[i1[..., 0], i1[..., 1], i1[..., 2]]
    c00 = c000 + (c100 - c000) * sx
    c10 = c010 + (c110 - c010) * sx
    c01 = c001 + (c101 - c001) * sx
    c11 = c011 + (c111 - c011) * sx
    c0 = c00 + (c10 - c00) * sy
    c1 = c01 + (c11 - c01) * sy
    return c0 + (c1 - c0) * sz


def _fractal3(points_unit: np.ndarray, features: int, octaves: int,
              seed: int) -> np.ndarray:
    """Fractal value noise over points in [0, 1]^3, output ~[-1, 1]."""
    total = np.zeros(len(points_unit))
    amp, norm = 1.0, 0.0
    cells = max(1, int(features))
    for i in range(int(octaves)):
        lattice = _value_noise3(cells, seed + 101 * i)
        total += amp * (_trilinear(lattice, points_unit * cells) - 0.5) * 2.0
        norm += amp
        amp *= 0.5
        cells *= 2
    return total / norm


def _area_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    fn = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                  vertices[faces[:, 2]] - vertices[faces[:, 0]])
    normals = np.zeros((len(vertices), 3))
    np.add.at(normals, faces[:, 0], fn)
    np.add.at(normals, faces[:, 1], fn)
    np.add.at(normals, faces[:, 2], fn)
    ln = np.linalg.norm(normals, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return normals / ln


def displace(mesh: trimesh.Trimesh, amplitude: float = 0.01,
             noise_scale: float = 2.5, seed: int = 0,
             octaves: int = 3) -> trimesh.Trimesh:
    """Push vertices along their normals with fractal 3D value noise.

    ``amplitude`` — peak displacement in meters; keep tiny (0.5–2 % of the
    model's size, e.g. 0.005 on a 0.5 m part).
    ``noise_scale`` — number of noise features across the model.
    """
    amp = float(amplitude)
    if amp <= 0.0:
        return mesh.copy()
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces)
    normals = _area_normals(v, f)
    bounds = mesh.bounds
    extent = np.maximum(bounds[1] - bounds[0], 1e-9)
    u = (v - bounds[0]) / extent
    offset = _fractal3(u, features=max(1, round(noise_scale)), octaves=octaves,
                       seed=int(seed))
    v2 = v + normals * (offset[:, None] * amp)
    out = trimesh.Trimesh(vertices=v2, faces=f.copy(),
                          vertex_normals=_area_normals(v2, f), process=False)
    if isinstance(mesh.visual, TextureVisuals):
        out.visual = TextureVisuals(uv=np.asarray(mesh.visual.uv).copy())
    return out


def humanize(mesh: trimesh.Trimesh, *, noise: float = 0.0,
             noise_scale: float = 2.5, seed: int = 0,
             tessellate: int = 0) -> trimesh.Trimesh:
    """Apply tessellation + organic displacement to a mesh.

    ``tessellate`` — subdivision iterations (use 2–3 on primitives like
    box/cylinder/cone before displacing, otherwise the noise has no vertices
    to grab). ``noise`` — peak displacement in meters. Returns a mesh with
    recomputed smooth normals; UVs, if any, are preserved.
    """
    out = mesh
    if tessellate:
        out = subdivide(out, tessellate)
    if noise and noise > 0:
        out = displace(out, amplitude=noise, noise_scale=noise_scale, seed=seed)
    else:
        # still refresh smooth normals for a hand-made feel
        f = np.asarray(out.faces)
        v = np.asarray(out.vertices)
        out = trimesh.Trimesh(vertices=v, faces=f,
                              vertex_normals=_area_normals(v, f), process=False)
        if isinstance(mesh.visual, TextureVisuals):
            out.visual = TextureVisuals(uv=np.asarray(mesh.visual.uv).copy())
    return out
