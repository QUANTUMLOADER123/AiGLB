"""Part-mating helpers: stop the classic "floating parts" failure mode.

LLMs tend to place parts on exact touching planes computed by eye, leaving
hairline gaps; or worse, random offsets that read as empty holes in renders.
The cure: mate parts by *measurement*, not by eye.

* :func:`touch` — slide a part along an axis until its face sits exactly on
  another part's face (use ``gap=-0.002`` to overlap ~2 mm, which renders
  seamlessly and survives precision loss).
* :func:`place_on` — same, but stacks a mesh on the *top* of another along y
  (or any axis), aligned to center.
* :func:`fit_diameter` / :func:`fit_box` — scale a part so it matches a target
  dimension (e.g. make a grip match a hand).
* :func:`assembly_report` — floating/hidden part detection for ``aiglb check``.

Recommended assembly style::

    stock  = aiglb.rounded_box(...)          # buttstock
    receiver = aiglb.box(...)
    scene.add("receiver", receiver, "lib:...", position=(0, 0, 0))
    stock_pos = aiglb.touch_pos(receiver_bounds, stock, axis="z", side="min", gap=-0.003)
"""
from __future__ import annotations

import numpy as np
import trimesh

__all__ = [
    "touch", "place_on", "fit_diameter", "fit_box",
    "bounds_gap", "connection_report",
]


def touch(mesh: trimesh.Trimesh, other: trimesh.Trimesh, axis: str = "y",
          side: str = "min", gap: float = -0.002) -> trimesh.Trimesh:
    """Translate ``mesh`` along ``axis`` so its ``side`` face meets ``other``.

    Parameters
    ----------
    axis : 'x' | 'y' | 'z'
    side : which face of ``mesh`` to seat — 'min' seats it against the top
           (max face) of ``other`` — 'max' seats it against the bottom.
    gap : meters of clearance; **negative overlaps** the parts (recommended
          -0.001..-0.005 for invisible seams).
    """
    axis = _axis_index(axis)
    side = side.lower()
    if side not in ("min", "max"):
        raise ValueError("side must be 'min' or 'max'")
    m_lo, m_hi = _bounds(mesh)
    o_lo, o_hi = _bounds(other)
    if side == "min":
        delta = o_hi[axis] + gap - m_lo[axis]
    else:
        delta = o_lo[axis] - gap - m_hi[axis]
    offset = np.zeros(3)
    offset[axis] = delta
    out = mesh.copy()
    out.apply_translation(offset)
    return out


def place_on(mesh: trimesh.Trimesh, other: trimesh.Trimesh, axis: str = "y",
             gap: float = -0.002, align: str = "center") -> trimesh.Trimesh:
    """Like :func:`touch` but also centers the part on the other two axes.

    ``align``: 'center' (default), 'min' or 'max' — alignment along the two
    non-axis directions using bounding boxes.
    """
    axis = _axis_index(axis)
    out = touch(mesh, other, axis=axis, side="min", gap=gap)
    others = [i for i in range(3) if i != axis]
    m_lo, m_hi = _bounds(out)
    o_lo, o_hi = _bounds(other)
    offset = np.zeros(3)
    for i in others:
        if align == "center":
            offset[i] = (o_lo[i] + o_hi[i]) / 2 - (m_lo[i] + m_hi[i]) / 2
        elif align == "min":
            offset[i] = o_lo[i] - m_lo[i]
        elif align == "max":
            offset[i] = o_hi[i] - m_hi[i]
        else:
            raise ValueError("align must be 'center', 'min' or 'max'")
    out.apply_translation(offset)
    return out


def fit_diameter(mesh: trimesh.Trimesh, target: float, axis: str | None = None) -> trimesh.Trimesh:
    """Uniformly scale a mesh so its bounding-cylinder diameter matches ``target``.

    ``axis=None`` uses the two largest extent axes; otherwise the given axis is
    left out of the diameter computation (scaling stays uniform!).
    """
    if axis is not None:
        idx = _axis_index(axis)
        others = [i for i in range(3) if i != idx]
    else:
        extents = mesh.bounds[1] - mesh.bounds[0]
        order = np.argsort(extents)
        others = order[1:]
    d = mesh.bounds[1][others] - mesh.bounds[0][others]
    scale = float(target) / float(max(d[0], d[1], 1e-9))
    out = mesh.copy()
    out.apply_scale(scale)
    # keep it centered on the same center
    center_now = out.bounds.mean(axis=0)
    center_was = mesh.bounds.mean(axis=0) * scale
    out.apply_translation(center_was - center_now)
    return out


def fit_box(mesh: trimesh.Trimesh, size=(0.1, 0.1, 0.1)) -> trimesh.Trimesh:
    """Non-uniformly scale a mesh so its bounding box equals ``size`` (x, y, z)."""
    extents = mesh.bounds[1] - mesh.bounds[0]
    target = np.asarray(size, dtype=np.float64)
    scale = target / np.maximum(extents, 1e-9)
    out = mesh.copy()
    out.apply_scale(scale)
    center_now = out.bounds.mean(axis=0)
    center_was = (mesh.bounds.mean(axis=0) * scale)
    out.apply_translation(center_was - center_now)
    return out


# ---------------------------------------------------------------------------
# assembly diagnostics
# ---------------------------------------------------------------------------

_AXIS = {"x": 0, "y": 1, "z": 2}


def _axis_index(axis) -> int:
    a = str(axis).lower()
    if a not in _AXIS:
        raise ValueError("axis must be 'x', 'y' or 'z'")
    return _AXIS[a]


def _bounds(mesh: trimesh.Trimesh):
    b = mesh.bounds
    return np.asarray(b[0], dtype=np.float64), np.asarray(b[1], dtype=np.float64)


def bounds_gap(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float:
    """Distance between axis-aligned bounding boxes, 0 when overlapping (meters)."""
    a_lo, a_hi = _bounds(a)
    b_lo, b_hi = _bounds(b)
    lower = np.maximum(a_lo - b_hi, b_lo - a_hi).max()
    if lower <= 0:
        return 0.0
    # per-axis separation outside overlap
    sep = np.maximum(b_lo - a_hi, a_lo - b_hi)
    sep = np.clip(sep, 0.0, None)
    return float(np.linalg.norm(sep))


def _surface_samples(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, int(count), seed=0)
        return np.asarray(pts, dtype=np.float64)
    except Exception:
        return np.asarray(mesh.vertices[:: max(1, len(mesh.vertices) // count)], dtype=np.float64)


def _triangle_distance(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Exact point->triangle distances (Ericson, Real-Time Collision Detection 5.1.5).

    points: (N, 3), tris: (T, 3, 3) -> (N,) distances. Vectorized over both.
    """
    a = tris[:, 0]  # (T, 3)
    ab = tris[:, 1] - a
    ac = tris[:, 2] - a
    n = points.shape[0]
    best = np.full(n, np.inf)
    T = tris.shape[0]
    CH = 4000
    for t0 in range(0, T, CH):
        A = a[t0:t0 + CH]; AB = ab[t0:t0 + CH]; AC = ac[t0:t0 + CH]
        # (n, c, 3) broadcasts
        P = points[:, None, :]
        AP = P - A[None, :, :]
        d1 = (AP * AB[None, :, :]).sum(-1)
        d2 = (AP * AC[None, :, :]).sum(-1)
        # region A (vertex a)
        best_i = np.full((n, A.shape[0]), np.inf)
        # candidate points computed per region; start with all-INF, fill progressively
        closest = np.empty((n, A.shape[0], 3))
        dist2 = np.full((n, A.shape[0]), np.inf)
        # --- vertex A
        d2a = (AP * AP).sum(-1)
        mask = (d1 <= 0) & (d2 <= 0)
        dist2 = np.where(mask, d2a, dist2)
        # --- vertex B
        BP = P - (A + AB)[None, :, :]
        d3 = (BP * AB[None, :, :]).sum(-1)
        d4 = (BP * AC[None, :, :]).sum(-1)
        d2b = (BP * BP).sum(-1)
        mask = (d3 >= 0) & (d4 <= d3)
        dist2 = np.where(mask, d2b, dist2)
        # --- edge AB
        vc = d1 * d4 - d3 * d2
        mask = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        v = np.where(d1 - d3 != 0, d1 / (d1 - d3 + 1e-30), 0.0)
        q = A[None, :, :] + v[..., None] * AB[None, :, :]
        dist2 = np.where(mask, ((P - q) ** 2).sum(-1), dist2)
        # --- vertex C
        CP = P - (A + AC)[None, :, :]
        d5 = (CP * AB[None, :, :]).sum(-1)
        d6 = (CP * AC[None, :, :]).sum(-1)
        d2c = (CP * CP).sum(-1)
        mask = (d6 >= 0) & (d5 <= d6)
        dist2 = np.where(mask, d2c, dist2)
        # --- edge AC
        vb = d5 * d2 - d1 * d6
        mask = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        w = np.where(d2 - d6 != 0, d2 / (d2 - d6 + 1e-30), 0.0)
        q = A[None, :, :] + w[..., None] * AC[None, :, :]
        dist2 = np.where(mask, ((P - q) ** 2).sum(-1), dist2)
        # --- edge BC
        BC = AC - AB
        d4b = d4 - d3
        d5b = d5 - d6
        va = d3 * d6 - d5 * d4
        mask = (va <= 0) & (d4b >= 0) & (d5b >= 0)
        w = np.where(d4b + d5b != 0, d4b / (d4b + d5b + 1e-30), 0.0)
        q = (A + AB)[None, :, :] + w[..., None] * BC[None, :, :]
        dist2 = np.where(mask, ((P - q) ** 2).sum(-1), dist2)
        # --- interior
        denom = va + vb + vc
        mask = dist2 == np.inf  # only where no Voronoi region claimed it
        v = np.where(denom != 0, vb / (denom + 1e-30), 0.0)
        w = np.where(denom != 0, vc / (denom + 1e-30), 0.0)
        q = A[None, :, :] + v[..., None] * AB[None, :, :] + w[..., None] * AC[None, :, :]
        dist2 = np.where(mask, ((P - q) ** 2).sum(-1), dist2)
        best = np.minimum(best, np.sqrt(dist2.min(axis=1)))
    return best


def _point_mesh_distance(points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """Exact point->triangle-mesh distance."""
    tris = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    return _triangle_distance(np.asarray(points, dtype=np.float64),
                              np.asarray(tris, dtype=np.float64))


def _points_inside_mesh(points: np.ndarray, mesh: trimesh.Trimesh) -> np.ndarray:
    """Ray-parity point-in-mesh test (Möller–Trumbore, vectorized, no deps)."""
    tris = np.asarray(mesh.vertices)[np.asarray(mesh.faces)].astype(np.float64)
    v0 = tris[:, 0]; e1 = tris[:, 1] - v0; e2 = tris[:, 2] - v0
    d = np.array([1.0, 0.5773502691896258, 0.3333333333333333])  # +X, irrational-ish slope
    d = d / np.linalg.norm(d)
    inside = np.zeros(len(points), dtype=bool)
    CH = 500
    for t0 in range(0, len(tris), CH):
        E1 = e1[t0:t0 + CH]; E2 = e2[t0:t0 + CH]; V0 = v0[t0:t0 + CH]
        pvec = np.cross(d, E2)                    # (C, 3)
        det = np.einsum("ij,ij->i", E1, pvec) if False else (E1 * pvec).sum(-1)  # (C,)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        for i in range(len(points)):
            o = points[i]
            tvec = o - V0                         # (C, 3)
            u = (tvec * pvec).sum(-1) * inv
            qvec = np.cross(tvec, E1)             # (C, 3)
            v = (qvec * d).sum(-1) * inv
            t = (qvec * E2).sum(-1) * inv
            hit = ok & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-9)
            if hit.any():
                inside[i] ^= (int(hit.sum()) % 2 == 1)
    return inside


def _containment_fraction(points: np.ndarray, mesh: trimesh.Trimesh):
    try:
        return float(np.mean(_points_inside_mesh(np.asarray(points, dtype=np.float64), mesh)))
    except Exception:
        return None


def connection_report(parts, gap: float = 0.004, samples: int = 120) -> list[str]:
    """Detect floating / hidden parts in a Scene.

    Parameters
    ----------
    parts : iterable of objects with ``.name`` and ``.mesh`` (aiglb Part)
    gap : maximum 'touching' distance in meters (default 4 mm)
    samples : surface sample count per part

    Returns a list of human-readable warnings. Empty = all parts are connected.
    """
    parts = list(parts)
    warnings: list[str] = []
    if len(parts) < 2:
        return warnings
    sample_cache = [(p.name, _surface_samples(p.mesh, samples)) for p in parts]
    adjacency = {p.name: set() for p in parts}
    nearest = {p.name: (np.inf, None) for p in parts}
    contain_frac = {}  # (i, j) -> fraction of i's samples inside j

    for i, pi in enumerate(parts):
        for j in range(i + 1, len(parts)):
            pj = parts[j]
            bg = bounds_gap(pi.mesh, pj.mesh)
            if bg > gap * 4:
                if bg < nearest[pi.name][0]:
                    nearest[pi.name] = (bg, pj.name)
                if bg < nearest[pj.name][0]:
                    nearest[pj.name] = (bg, pi.name)
                continue
            d = float(_point_mesh_distance(sample_cache[i][1], pj.mesh).min())
            if d > gap:
                d_rev = float(_point_mesh_distance(sample_cache[j][1], pi.mesh).min())
                d = min(d, d_rev)
            connected = d <= gap
            if not connected and bg == 0.0:
                # bboxes overlap: an overlap is a connectivity proof too
                fa = _containment_fraction(sample_cache[i][1], pj.mesh)
                fb = _containment_fraction(sample_cache[j][1], pi.mesh)
                contain_frac[(i, j)] = fa if fa is not None else 0.0
                contain_frac[(j, i)] = fb if fb is not None else 0.0
                connected = contain_frac[(i, j)] > 0.02 or contain_frac[(j, i)] > 0.02
            if connected:
                adjacency[pi.name].add(pj.name)
                adjacency[pj.name].add(pi.name)
            else:
                if d < nearest[pi.name][0]:
                    nearest[pi.name] = (d, pj.name)
                if d < nearest[pj.name][0]:
                    nearest[pj.name] = (d, pi.name)

    y_floor = min(float(p.mesh.bounds[0][1]) for p in parts)
    for p in parts:
        best, best_name = nearest[p.name]
        on_floor = abs(float(p.mesh.bounds[0][1]) - y_floor) < 1e-6 and y_floor <= 1e-6
        if not adjacency[p.name] and not on_floor:
            warnings.append(
                f"part '{p.name}' does not touch any other part "
                f"(nearest: '{best_name}' at {best * 1000:.1f} mm) — "
                f"mate them with aiglb.touch(...)/place_on(...) or negative overlap"
            )
    # hidden-inside check: part whose samples are (mostly) enclosed by a bigger part
    for i, p in enumerate(parts):
        for j, q in enumerate(parts):
            if i == j:
                continue
            frac = contain_frac.get((i, j))
            if frac is None:
                continue
            if frac > 0.65 and len(q.mesh.faces) >= len(p.mesh.faces):
                warnings.append(
                    f"part '{p.name}' is hidden inside part '{q.name}' "
                    f"({frac * 100:.0f}% of its surface is enclosed) — remove it or offset it"
                )
    return warnings
