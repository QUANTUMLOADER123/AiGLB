"""GLB inspection: quick sanity report for humans and LLMs."""
from __future__ import annotations

import os

import numpy as np
import trimesh

__all__ = ["inspect_glb", "format_report"]


def inspect_glb(path: str) -> dict:
    """Load ``path`` and return a stats dict (also used by Scene.save)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    size = os.path.getsize(path)
    loaded = trimesh.load(path, force="scene")

    warnings: list[str] = []
    total_v = total_f = 0
    part_reports = []
    materials_used: set[str] = set()

    geometries = loaded.geometry
    for name, geom in geometries.items():
        total_v += len(geom.vertices)
        total_f += len(geom.faces)
        try:
            welded = trimesh.Trimesh(vertices=geom.vertices.copy(), faces=geom.faces.copy(), process=True)
            watertight = bool(welded.is_watertight)
        except Exception:
            watertight = False
        bmin, bmax = geom.bounds
        extents = bmax - bmin
        part_reports.append({
            "name": name,
            "vertices": int(len(geom.vertices)),
            "triangles": int(len(geom.faces)),
            "watertight": watertight,
            "size": [float(x) for x in extents],
        })
        if len(geom.faces) == 0:
            warnings.append(f"part '{name}' has no faces")
        if float(extents.max()) == 0.0:
            warnings.append(f"part '{name}' is a point (zero-size)")

    # raw GLB JSON for material/texture details
    material_names: list[str] = []
    embedded_images = 0
    extensions: list[str] = []
    tangents_total = tangents_present = 0
    try:
        from pygltflib import GLTF2
        g = GLTF2.load(path)
        material_names = [m.name or f"material{i}" for i, m in enumerate(g.materials)]
        embedded_images = len(g.images)
        extensions = list(g.extensionsUsed or [])
        for mesh in g.meshes or []:
            for prim in mesh.primitives:
                tangents_total += 1
                if getattr(prim.attributes, "TANGENT", None) is not None:
                    tangents_present += 1
    except Exception:
        warnings.append("could not parse GLB JSON with pygltflib")

    bounds = np.asarray(loaded.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        bounds_size = (0.0, 0.0, 0.0)
        warnings.append("scene has no finite bounds")
    else:
        bounds_size = tuple(float(x) for x in (bounds[1] - bounds[0]))
        if max(bounds_size) > 100:
            warnings.append(f"scene is {max(bounds_size):.1f} m across — check your units (1 unit = 1 m)")
        if max(bounds_size) < 0.01:
            warnings.append("scene is smaller than 1 cm — check your units (1 unit = 1 m)")

    open_parts = [p["name"] for p in part_reports if not p["watertight"]]
    if open_parts and len(geometries):
        # open surfaces (planes) are legit; only warn when *most* parts are open
        if len(open_parts) == len(geometries) and len(geometries) > 2:
            warnings.append(f"no part is watertight ({', '.join(open_parts[:5])}) — fine for rendering, not for boolean/printing")

    if total_f > 400_000:
        warnings.append(f"{total_f} triangles — consider fewer segments for web use")

    return {
        "path": path,
        "file_size_bytes": int(size),
        "parts": int(len(geometries)),
        "total_vertices": int(total_v),
        "total_triangles": int(total_f),
        "material_names": material_names,
        "embedded_images": int(embedded_images),
        "extensions_used": extensions,
        "tangents_primitives": [tangents_present, tangents_total],
        "bounds_size": bounds_size,
        "part_reports": part_reports,
        "warnings": warnings,
    }


def format_report(path: str) -> str:
    r = inspect_glb(path)
    lines = [
        f"GLB: {r['path']}",
        f"  file:           {r['file_size_bytes'] / 1024:.0f} KB",
        f"  parts:          {r['parts']}",
        f"  triangles:      {r['total_triangles']}",
        f"  vertices:       {r['total_vertices']}",
        f"  size (m):       {r['bounds_size'][0]:.3f} × {r['bounds_size'][1]:.3f} × {r['bounds_size'][2]:.3f}",
        f"  materials:      {', '.join(r['material_names']) or '—'}",
        f"  textures:       {r['embedded_images']} embedded images",
    ]
    if r["extensions_used"]:
        lines.append(f"  extensions:     {', '.join(r['extensions_used'])}")
    t_present, t_total = r.get("tangents_primitives", (0, 0))
    if t_total:
        lines.append(f"  tangents:       {t_present}/{t_total} primitives have TANGENT")
    lines.append("  per part:")
    for p in r["part_reports"]:
        wt = "watertight" if p["watertight"] else "open"
        lines.append(f"    - {p['name']:<24s} {p['triangles']:>6d} tris  {wt}  "
                     f"({p['size'][0]:.2f}×{p['size'][1]:.2f}×{p['size'][2]:.2f} m)")
    for w in r["warnings"]:
        lines.append(f"  ⚠ {w}")
    return "\n".join(lines)
