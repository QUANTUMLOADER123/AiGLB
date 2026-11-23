"""Primitive meshes with built-in analytic normals and world-space UVs.

Conventions
-----------
* 1 unit = 1 meter, +Y is up (glTF convention).
* Most primitives are centered at their local origin, EXCEPT:
    - ``plane``        lies on XZ at y = 0 (faces +Y)
    - ``extrude``      stands on y = 0, occupies y in [0, height]
    - ``wedge``        stands on y = 0
  Everything else is centered at the origin.
* ``uv_scale``: texture density. UVs are generated from real-world coordinates,
  so ``uv_scale=1`` tiles the texture once per meter. Raise it for tiny parts.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import trimesh
from trimesh.visual import TextureVisuals

__all__ = [
    "box", "plane", "cylinder", "cone", "tube", "sphere", "capsule", "torus",
    "lathe", "extrude", "wedge",
]


# ---------------------------------------------------------------------------
# low-level packing
# ---------------------------------------------------------------------------

def _pack(vertices, faces, normals, uvs) -> trimesh.Trimesh:
    """Assemble a Trimesh with analytic normals + UVs (no vertex merging)."""
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    n = n / ln
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    uv = np.asarray(uvs, dtype=np.float64).reshape(-1, 2)
    mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_normals=n, process=False)
    mesh.visual = TextureVisuals(uv=uv)
    return mesh


def _wrap_indices(count: int) -> tuple[np.ndarray, np.ndarray]:
    """(count,) -> (count + 1,) with the wrap-around index appended."""
    idx = np.arange(count + 1)
    return idx, idx % count


# ---------------------------------------------------------------------------
# box / plane
# ---------------------------------------------------------------------------

def box(width: float = 1.0, height: float = 1.0, depth: float = 1.0,
        uv_scale: float = 1.0) -> trimesh.Trimesh:
    """Axis-aligned box, centered at the origin. Per-face texture mapping."""
    w, h, d = width / 2, height / 2, depth / 2
    # (normal, 4 corners CCW seen from outside, uv axes)
    F = [
        ((1, 0, 0),  [(w, -h, -d), (w, h, -d), (w, h, d), (w, -h, d)], (2, 1)),
        ((-1, 0, 0), [(-w, -h, d), (-w, h, d), (-w, h, -d), (-w, -h, -d)], (2, 1)),
        ((0, 1, 0),  [(-w, h, -d), (-w, h, d), (w, h, d), (w, h, -d)], (0, 2)),
        ((0, -1, 0), [(-w, -h, d), (-w, -h, -d), (w, -h, -d), (w, -h, d)], (0, 2)),
        ((0, 0, 1),  [(-w, -h, d), (w, -h, d), (w, h, d), (-w, h, d)], (0, 1)),
        ((0, 0, -1), [(w, -h, -d), (-w, -h, -d), (-w, h, -d), (w, h, -d)], (0, 1)),
    ]
    verts, faces, norms, uvs = [], [], [], []
    for normal, corners, (a1, a2) in F:
        base = len(verts)
        for c in corners:
            verts.append(c)
            norms.append(normal)
            uvs.append((c[a1] * uv_scale, c[a2] * uv_scale))
        faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return _pack(verts, faces, norms, uvs)


def plane(width: float = 1.0, depth: float = 1.0,
          uv_scale: float = 1.0) -> trimesh.Trimesh:
    """Flat horizontal quad on XZ at y = 0, facing +Y."""
    w, d = width / 2, depth / 2
    verts = [(-w, 0, -d), (-w, 0, d), (w, 0, d), (w, 0, -d)]
    norms = [(0, 1, 0)] * 4
    uvs = [(x * uv_scale, z * uv_scale) for x, _, z in verts]
    faces = [(0, 1, 2), (0, 2, 3)]
    # check winding: cross((1,0)-(0,0), (2,0)-(0,0)) on XZ -> +Y? (0,0,2d)x... verify in test
    return _pack(verts, faces, norms, uvs)


# ---------------------------------------------------------------------------
# cylinder / cone / tube
# ---------------------------------------------------------------------------

def cylinder(radius: float = 0.5, height: float = 1.0, segments: int = 32,
             radius_top: float | None = None, capped: bool = True,
             uv_scale: float = 1.0) -> trimesh.Trimesh:
    """Vertical cylinder (or frustum with ``radius_top``), centered at origin.

    ``radius_top=0`` gives a cone (apex, not a cap). ``capped=False`` for an open pipe.
    """
    rb = float(radius)
    rt = float(radius if radius_top is None else radius_top)
    h = float(height)
    n = int(max(3, segments))
    if rb < 0 or rt < 0:
        raise ValueError("radii must be >= 0")
    if rb == 0 and rt == 0:
        raise ValueError("at least one radius must be > 0")

    r_avg = max((rb + rt) / 2.0, 1e-9)
    slope = (rt - rb) / h if h else 0.0  # dr/dy
    verts: list = []
    norms: list = []
    uvs: list = []
    faces: list = []

    def add_ring(radius_at, y, normal_tilt):
        start = len(verts)
        for i in range(n + 1):
            a = 2 * math.pi * i / n
            ca, sa = math.cos(a), math.sin(a)
            verts.append((radius_at * ca, y, radius_at * sa))
            norms.append((ca, normal_tilt, sa))  # renormalized in _pack
            uvs.append((i / n * 2 * math.pi * r_avg * uv_scale, y * uv_scale))
        return start

    tilt = -slope  # normal y component numerator (cos, -slope, sin)
    cone_apex = rt == 0.0

    bottom = add_ring(rb, -h / 2, tilt)
    if cone_apex:
        # one apex vertex per segment (shared position, outward-tilted normals)
        apex_start = len(verts)
        for i in range(n + 1):
            a = 2 * math.pi * i / n
            ca, sa = math.cos(a), math.sin(a)
            verts.append((0.0, h / 2, 0.0))
            norms.append((ca, -slope, sa))
            uvs.append((i / n * 2 * math.pi * r_avg * uv_scale, h / 2 * uv_scale))
        for i in range(n):
            faces.append((bottom + i, apex_start + i, bottom + i + 1))
    else:
        top = add_ring(rt, h / 2, tilt)
        for i in range(n):
            faces.append((bottom + i, top + i + 1, bottom + i + 1))
            faces.append((bottom + i, top + i, top + i + 1))

    def add_cap(radius_at, y, ny):
        start = len(verts)
        center = len(verts)
        verts.append((0.0, y, 0.0))
        norms.append((0.0, ny, 0.0))
        uvs.append((0.0, 0.0))
        for i in range(n + 1):
            a = 2 * math.pi * i / n
            ca, sa = math.cos(a), math.sin(a)
            verts.append((radius_at * ca, y, radius_at * sa))
            norms.append((0.0, ny, 0.0))
            uvs.append((radius_at * ca * uv_scale, radius_at * sa * uv_scale))
        for i in range(n):
            if ny > 0:
                faces.append((center, start + 1 + i + 1, start + 1 + i))
            else:
                faces.append((center, start + 1 + i, start + 1 + i + 1))
        return

    if capped:
        add_cap(rb, -h / 2, -1.0)
        if not cone_apex and rt > 0:
            add_cap(rt, h / 2, 1.0)

    return _pack(verts, faces, norms, uvs)


def cone(radius: float = 0.5, height: float = 1.0, segments: int = 32,
         uv_scale: float = 1.0) -> trimesh.Trimesh:
    """Cone with apex at +y/2, base cap at -y/2."""
    return cylinder(radius=radius, height=height, segments=segments,
                    radius_top=0.0, capped=True, uv_scale=uv_scale)


def tube(outer_radius: float = 0.5, inner_radius: float = 0.4, height: float = 1.0,
         segments: int = 32, uv_scale: float = 1.0) -> trimesh.Trimesh:
    """Hollow pipe (rectangular ring profile), centered at the origin."""
    if inner_radius >= outer_radius:
        raise ValueError("inner_radius must be < outer_radius")
    r0, r1, h2 = inner_radius, outer_radius, height / 2
    profile = [(r0, -h2), (r1, -h2), (r1, h2), (r0, h2), (r0, -h2)]
    return lathe(profile, segments=segments, smooth=False, uv_scale=uv_scale)


# ---------------------------------------------------------------------------
# sphere
# ---------------------------------------------------------------------------

def sphere(radius: float = 0.5, width_segments: int = 32, height_segments: int = 18,
           uv_scale: float = 1.0) -> trimesh.Trimesh:
    """UV sphere with smooth normals, centered at the origin."""
    r = float(radius)
    n = int(max(3, width_segments))
    m = int(max(2, height_segments))
    verts: list = []
    norms: list = []
    uvs: list = []
    faces: list = []

    # poles
    top = len(verts); verts.append((0.0, r, 0.0)); norms.append((0.0, 1.0, 0.0)); uvs.append((0.0, 0.5 * np.pi * r * uv_scale))
    bottom = len(verts); verts.append((0.0, -r, 0.0)); norms.append((0.0, -1.0, 0.0)); uvs.append((0.0, -0.5 * np.pi * r * uv_scale))

    ring_starts = [top]
    for j in range(1, m):
        theta = math.pi * j / m  # 0..pi from top
        y = math.cos(theta) * r
        rr = math.sin(theta) * r
        start = len(verts)
        for i in range(n + 1):
            a = 2 * math.pi * i / n
            ca, sa = math.cos(a), math.sin(a)
            verts.append((rr * ca, y, rr * sa))
            norms.append((rr * ca / r, y / r, rr * sa / r))
            uvs.append((i / n * 2 * math.pi * r * uv_scale, theta * r * uv_scale))
        ring_starts.append(start)
    ring_starts.append(bottom)

    for j in range(1, m):
        upper = ring_starts[j - 1]
        lower = ring_starts[j]
        upper_count = 1 if j == 1 else n + 1
        if j == 1:
            # fan from top pole
            for i in range(n):
                faces.append((upper, lower + i + 1, lower + i))
            continue
        for i in range(n):
            faces.append((upper + i, lower + i + 1, lower + i))
            faces.append((upper + i, upper + i + 1, lower + i + 1))
    # bottom fan
    last_ring = ring_starts[-2]
    for i in range(n):
        faces.append((bottom, last_ring + i, last_ring + i + 1))

    return _pack(verts, faces, norms, uvs)


# ---------------------------------------------------------------------------
# lathe / capsule / torus
# ---------------------------------------------------------------------------

def lathe(profile: Sequence[tuple[float, float]], segments: int = 48,
          smooth: bool = True, uv_scale: float = 1.0) -> trimesh.Trimesh:
    """Surface of revolution around the +Y axis.

    ``profile`` — list of ``(radius, y)`` points from bottom to top, e.g.
    ``[(0, 0), (0.35, 0.02), (0.45, 0.25), (0.12, 0.8), (0.18, 0.85)]`` makes a flask.

    * Start/end at radius 0 to close the ends (single pole vertex).
    * Profile where first point == last point creates a closed loop (tube, torus).
    * ``smooth=False`` gives crisp hard edges between profile segments.
    """
    pts = [(float(r), float(y)) for r, y in profile]
    if len(pts) < 2:
        raise ValueError("lathe profile needs at least 2 points")
    for r, _ in pts:
        if r < -1e-9:
            raise ValueError("profile radii must be >= 0")
    closed = np.allclose(pts[0], pts[-1], atol=1e-9) and len(pts) > 3

    n = int(max(3, segments))
    P = np.array(pts, dtype=np.float64)
    radii, ys = P[:, 0], P[:, 1]

    # signed area to auto-orient closed loops (we want outward normals)
    flip = False
    if closed:
        area = 0.5 * np.sum(radii * np.roll(ys, -1) - np.roll(radii, -1) * ys)
        flip = area < 0

    # segment tangents & normals in (r, y) space
    # open profile: closed=False -> treat indices with clamping
    m = len(pts)

    def seg_normal(i0, i1):
        d = P[i1] - P[i0]
        ln = math.hypot(d[0], d[1])
        if ln < 1e-12:
            return np.array([0.0, 1.0])
        # outward: rotate tangent by -90 deg in (r, y)
        nrm = np.array([d[1], -d[0]]) / ln
        if flip:
            nrm = -nrm
        return nrm

    seg_norms = [seg_normal(i, i + 1) for i in range(m - 1)]

    def row_normal(j):
        if j == 0:
            acc = seg_norms[0]
        elif j == m - 1:
            if closed:
                acc = seg_norms[-1] + seg_norms[0]
            else:
                acc = seg_norms[-1]
        else:
            acc = seg_norms[j - 1] + seg_norms[j]
        ln = np.linalg.norm(acc)
        return acc / ln if ln else np.array([1.0, 0.0])

    # arc length for v coordinate
    arcs = [0.0]
    for i in range(1, m):
        arcs.append(arcs[-1] + float(np.linalg.norm(P[i] - P[i - 1])))
    total_arc = arcs[-1] if arcs[-1] > 1e-12 else 1.0
    r_char = max(float(np.mean(radii)), 1e-6)

    verts: list = []
    norms: list = []
    uvs: list = []
    faces: list = []

    def emit_ring(j, normal2, row_arc):
        r, y = pts[j]
        start = len(verts)
        if r < 1e-9:
            # pole row: single vertex on the axis
            verts.append((0.0, y, 0.0))
            norms.append((0.0, 1.0 if j > len(pts) // 2 else -1.0, 0.0))
            uvs.append((0.5, row_arc * uv_scale))
            return start, True
        for i in range(n + 1):
            a = 2 * math.pi * i / n
            ca, sa = math.cos(a), math.sin(a)
            verts.append((r * ca, y, r * sa))
            norms.append((normal2[0] * ca, normal2[1], normal2[0] * sa))
            uvs.append((i / n * 2 * math.pi * r_char * uv_scale, row_arc * uv_scale))
        return start, False

    if smooth:
        nrm2 = [row_normal(j) for j in range(m)]
        ring_start, ring_is_pole = [], []
        for j in range(m):
            s, isp = emit_ring(j, nrm2[j], arcs[j])
            ring_start.append(s); ring_is_pole.append(isp)
        for j in range(m - 1):
            s0, p0 = ring_start[j], ring_is_pole[j]
            s1, p1 = ring_start[j + 1], ring_is_pole[j + 1]
            if p0 and p1:
                continue
            if p0:
                for i in range(n):
                    faces.append((s0, s1 + i, s1 + i + 1))
            elif p1:
                for i in range(n):
                    faces.append((s1, s0 + i + 1, s0 + i))
            else:
                for i in range(n):
                    faces.append((s0 + i, s1 + i, s1 + i + 1))
                    faces.append((s0 + i, s1 + i + 1, s0 + i + 1))
    else:
        # hard edges: duplicate rows at every profile joint
        prev_start, prev_pole = None, None
        for j in range(m - 1):
            seg_nrm = seg_norms[j]
            s_top, isp_top = emit_ring(j, seg_nrm, arcs[j])
            s_bot, isp_bot = emit_ring(j + 1, seg_nrm, arcs[j + 1])
            if prev_start is None:
                prev_start, prev_pole = s_top, isp_top
            a0, a1 = prev_start, s_bot
            if prev_pole and isp_bot:
                prev_start, prev_pole = None, None
                continue
            if prev_pole:
                for i in range(n):
                    faces.append((a0, a1 + i, a1 + i + 1))
            elif isp_bot:
                for i in range(n):
                    faces.append((a1, a0 + i + 1, a0 + i))
            else:
                for i in range(n):
                    faces.append((a0 + i, a1 + i, a1 + i + 1))
                    faces.append((a0 + i, a1 + i + 1, a0 + i + 1))
            prev_start, prev_pole = s_bot, isp_bot

    mesh = _pack(verts, faces, norms, uvs)
    # pole vertex normals: point along axis by profile tangent direction
    vn = mesh.vertex_normals.copy()
    for j, jstart in enumerate([0]):  # poles handled inline; axialize pole rows here
        pass
    return mesh


def capsule(radius: float = 0.25, height: float = 1.0, segments: int = 32,
            rings: int = 10, uv_scale: float = 1.0) -> trimesh.Trimesh:
    """Capsule (stadium of revolution). ``height`` = cylinder section length."""
    r = float(radius)
    h2 = float(height) / 2.0
    profile: list[tuple[float, float]] = []
    for i in range(rings + 1):
        a = -math.pi / 2 + (math.pi / 2) * i / rings  # -90..0 deg (bottom cap)
        profile.append((max(r * math.cos(a), 0.0), -h2 + r * math.sin(a)))
    for i in range(rings + 1):
        a = (math.pi / 2) * i / rings  # 0..90 deg (top cap)
        profile.append((max(r * math.cos(a), 0.0), h2 + r * math.sin(a)))
    return lathe(profile, segments=segments, smooth=True, uv_scale=uv_scale)


def torus(radius: float = 0.5, tube: float = 0.15, segments: int = 48,
          sides: int = 24, uv_scale: float = 1.0) -> trimesh.Trimesh:
    """Torus in the XZ plane (hole along +Y)."""
    R, rt = float(radius), float(tube)
    profile = []
    for i in range(sides + 1):
        a = 2 * math.pi * i / sides
        profile.append((R + rt * math.cos(a), rt * math.sin(a)))
    return lathe(profile, segments=segments, smooth=True, uv_scale=uv_scale)


# ---------------------------------------------------------------------------
# extrude / wedge
# ---------------------------------------------------------------------------

def extrude(polygon: Sequence[tuple[float, float]], height: float = 0.5,
            uv_scale: float = 1.0, center: bool = False) -> trimesh.Trimesh:
    """Extrude a 2D polygon footprint into a standing prism (+Y is height).

    Draw the footprint like a map: ``x`` is east, ``y`` is north. The shape
    stands on y = 0 and spans y in [0, height]; footprint +y maps to world -z.
    Concave polygons are supported.
    """
    pts = np.asarray(polygon, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise ValueError("polygon must be a list of (x, y) pairs, >= 3 points")
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    if abs(area) < 1e-12:
        raise ValueError("polygon has zero area")
    if area < 0:
        pts = pts[::-1].copy()  # force CCW
        area = -area
    h = float(height)
    if h <= 0:
        raise ValueError("height must be > 0")

    verts: list = []
    norms: list = []
    uvs: list = []
    faces: list = []

    # --- walls (split vertices, hard edges) ---
    perimeter = [0.0]
    for i in range(len(pts)):
        p0, p1 = pts[i], pts[(i + 1) % len(pts)]
        perimeter.append(perimeter[-1] + float(np.linalg.norm(p1 - p0)))

    for i in range(len(pts)):
        p0, p1 = pts[i], pts[(i + 1) % len(pts)]
        edge = p1 - p0
        ln = np.linalg.norm(edge)
        if ln < 1e-12:
            continue
        nrm = np.array([edge[1], -edge[0]]) / ln  # outward for CCW
        base = len(verts)
        u0 = perimeter[i] * uv_scale
        u1 = perimeter[i + 1] * uv_scale
        for (px, py), u in (((p0[0], p0[1]), u0), ((p1[0], p1[1]), u1)):
            verts.append((px, 0.0, -py))   # stand up: polygon y -> -z
            norms.append((nrm[0], 0.0, -nrm[1]))
            uvs.append((u, 0.0))
        top = len(verts)
        for (px, py), u in (((p0[0], p0[1]), u0), ((p1[0], p1[1]), u1)):
            verts.append((px, h, -py))
            norms.append((nrm[0], 0.0, -nrm[1]))
            uvs.append((u, h * uv_scale))
        b0, b1, t0, t1 = base, base + 1, top, top + 1
        faces.append((b0, b1, t1))
        faces.append((b0, t1, t0))

    # --- caps via shapely triangulation ---
    try:
        from shapely.geometry import Polygon
        poly = Polygon(pts)
        try:
            # shapely >= 2: true constrained Delaunay, clipped to the polygon
            from shapely import constrained_delaunay_triangles
            coll = constrained_delaunay_triangles(poly)
            tris = list(coll.geoms) if hasattr(coll, "geoms") else list(coll)
        except ImportError:
            from shapely.ops import triangulate
            # keep only triangles fully inside the (possibly concave) polygon
            tris = [t for t in triangulate(poly) if poly.covers(t)]
        if not tris:
            raise ValueError("shapely could not triangulate polygon")
    except ImportError:
        # ear clipping fallback: fan from centroid (convex-ish polys only)
        centroid = pts.mean(axis=0)
        class _Tri:
            def __init__(self, a, b, c):
                self.coords = [tuple(a), tuple(b), tuple(c)]
        tris = [_Tri(centroid, pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]

    def cap(z, ny):
        base = len(verts)
        for t in tris:
            coords = list(t.exterior.coords) if hasattr(t, "exterior") else list(t.coords)
            # drop duplicated closing point if present
            while len(coords) > 3 and np.allclose(coords[0], coords[-1]):
                coords = coords[:-1]
            if len(coords) != 3:
                continue
            # shapely triangulation preserves CCW for CCW input; ensure winding
            (ax, ay), (bx, by), (cx, cy) = [tuple(c) for c in coords]
            cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if (cross > 0) != (ny > 0):
                coords = [coords[0], coords[2], coords[1]]
            idx = []
            for (px, py) in coords:
                idx.append(len(verts))
                verts.append((px, z, -py))
                norms.append((0.0, ny, 0.0))
                uvs.append((px * uv_scale, py * uv_scale))
            faces.append(tuple(idx))

    cap(0.0, -1.0)   # bottom
    cap(h, 1.0)      # top

    mesh = _pack(verts, faces, norms, uvs)
    if center:
        mesh.apply_translation((0, -h / 2, 0))
    return mesh


def wedge(width: float = 1.0, height: float = 1.0, depth: float = 1.0,
          uv_scale: float = 1.0) -> trimesh.Trimesh:
    """Right triangular prism (ramp). Vertical face at z=-d/2, hypotenuse to z=+d/2.

    Stands on y = 0. Handy for roofs, ramps, doorstops.
    """
    w, h, d = width / 2, float(height), depth / 2
    # cross-section in ZY: A(y=0,z=-d) B(y=0,z=+d) C(y=h,z=-d)
    verts, faces, norms, uvs = [], [], [], []

    def quad(p0, p1, p2, p3, normal):
        nonlocal faces
        base = len(verts)
        for p in (p0, p1, p2, p3):
            verts.append(p)
            norms.append(normal)
            uvs.append(((p[0] + p[2]) * uv_scale, (p[1]) * uv_scale))
        faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]

    # bottom (faces -Y)
    quad((-w, 0, -d), (w, 0, -d), (w, 0, d), (-w, 0, d), (0, -1, 0))
    # back vertical (z=-d, faces -Z)
    quad((-w, 0, -d), (-w, h, -d), (w, h, -d), (w, 0, -d), (0, 0, -1))
    # sloped top: from (y=0,z=+d) up to (y=h,z=-d), faces +Y/+Z
    n_slope = (0.0, 2 * d, h)
    nl = math.sqrt(n_slope[1] ** 2 + n_slope[2] ** 2)
    n_slope = (0.0, n_slope[1] / nl, n_slope[2] / nl)
    quad((-w, 0, d), (w, 0, d), (w, h, -d), (-w, h, -d), n_slope)

    # end triangles
    for sx, ny in ((1, (1, 0, 0)), (-1, (-1, 0, 0))):
        base = len(verts)
        for p in ((sx * w, 0, -d), (sx * w, 0, d), (sx * w, h, -d)):
            verts.append(p)
            norms.append(ny)
            uvs.append((p[2] * uv_scale, p[1] * uv_scale))
        if sx > 0:
            faces.append((base, base + 2, base + 1))
        else:
            faces.append((base, base + 1, base + 2))
    return _pack(verts, faces, norms, uvs)
