"""Scene assembly: collect named parts and export a .glb file."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Union

import numpy as np
import trimesh
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

from . import materials as _materials
from .ops import transform_matrix, _apply

__all__ = ["Scene", "Part"]

MaterialLike = Union[str, PBRMaterial, None]


def _unique(name: str, used: set[str]) -> str:
    base = re.sub(r"[^0-9A-Za-z_\-]+", "_", str(name)).strip("_") or "part"
    candidate, k = base, 2
    while candidate in used:
        candidate = f"{base}_{k}"
        k += 1
    used.add(candidate)
    return candidate


@dataclass
class Part:
    """A named mesh + material pair inside a Scene."""

    name: str
    mesh: trimesh.Trimesh
    material: PBRMaterial

    def __repr__(self) -> str:
        return f"<Part {self.name!r}: {len(self.mesh.faces)} tris>"


class Scene:
    """Collect parts, then ``save()`` as a binary glTF (.glb).

    Example
    -------
    >>> scene = aiglb.Scene("robot")
    >>> scene.add("body", aiglb.box(0.6, 0.8, 0.35), "steel", position=(0, 1.0, 0))
    >>> scene.add("head", aiglb.sphere(0.28), "plastic_white", position=(0, 1.55, 0))
    >>> scene.save("robot.glb")
    """

    def __init__(self, name: str = "model") -> None:
        self.name = name
        self.parts: list[Part] = []
        self._used_names: set[str] = set()

    # ------------------------------------------------------------------
    def add(self, name: str, mesh: trimesh.Trimesh, material: MaterialLike = None,
            position=(0, 0, 0), rotation=(0, 0, 0), scale=1.0) -> Part:
        """Add a mesh to the scene.

        Parameters
        ----------
        name : part name (unique-ified; used as node name in the .glb)
        mesh : any aiglb primitive/op result, or a trimesh.Trimesh
        material : preset name ('oak', 'gold', ...), hex color ('#ff8800'),
                   PBRMaterial, or None for neutral gray
        position : (x, y, z) meters, applied after rotation/scale
        rotation : (rx, ry, rz) **degrees**, X then Y then Z
        scale : scalar or (sx, sy, sz)
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"mesh must be a trimesh.Trimesh, got {type(mesh)}")
        mat = _materials.resolve_material(material)
        M = transform_matrix(position, rotation, scale)
        moved = _apply(mesh, M)
        moved = self._apply_material(mat, moved)
        part_name = _unique(name, self._used_names)
        part = Part(part_name, moved, mat)
        self.parts.append(part)
        return part

    @staticmethod
    def _apply_material(mat: PBRMaterial, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Attach ``mat`` to ``mesh``.

        If the mesh already has UVs (all primitives ship world-scale UVs), the
        material's ``_uv_scale`` / ``_uv_offset`` / ``_uv_rotation`` (library
        materials) are applied on top of them; meshes without UVs get a fresh
        dominant-axis box projection with those parameters. Works on a copy.
        """
        from .ops import box_project
        image = getattr(mat, "baseColorTexture", None)
        uv_scale = float(getattr(mat, "_uv_scale", 1.0))
        uv_offset = getattr(mat, "_uv_offset", (0.0, 0.0))
        uv_rotation = float(getattr(mat, "_uv_rotation", 0.0))
        kwargs = {"material": mat}
        if image is not None:
            kwargs["image"] = image
        src_mesh = mesh
        mesh = mesh.copy()
        try:  # keep glTF's NORMAL export path intact: it checks the cache
            cached_vn = getattr(src_mesh, "_cache", None) and src_mesh._cache.cache.get("vertex_normals")
            if cached_vn is not None:
                mesh._cache["vertex_normals"] = np.array(cached_vn, dtype=np.float64, copy=True)
        except Exception:
            pass
        if isinstance(mesh.visual, TextureVisuals):
            uv = np.asarray(mesh.visual.uv, dtype=np.float64)
            if uv_rotation or uv_scale != 1.0 or tuple(uv_offset) != (0.0, 0.0):
                c_, s_ = np.cos(uv_rotation), np.sin(uv_rotation)
                rot = np.array([[c_, -s_], [s_, c_]])
                uv = (uv @ rot.T) * uv_scale + np.asarray(uv_offset, dtype=np.float64)
            kwargs["uv"] = uv
            mesh.visual = TextureVisuals(**kwargs)
            return mesh
        projected = box_project(mesh, uv_scale=uv_scale,
                                uv_offset=uv_offset, uv_rotation=uv_rotation)
        projected.visual = TextureVisuals(uv=np.asarray(projected.visual.uv), **kwargs)
        return projected

    # ------------------------------------------------------------------
    def save(self, path: str | None = None) -> dict:
        """Export the scene as .glb and print a verification report.

        Path defaults to ``$AIGLB_OUTPUT`` or ``./<scene-name>.glb``.
        """
        if path is None:
            path = os.environ.get("AIGLB_OUTPUT") or f"{self.name}.glb"
        if not self.parts:
            raise ValueError("scene is empty — add parts before save()")
        if not path.lower().endswith(".glb"):
            path += ".glb"

        t0 = time.time()
        tri_scene = trimesh.Scene()
        for part in self.parts:
            tri_scene.add_geometry(part.mesh, node_name=part.name, geom_name=part.name)
        tri_scene.export(path)

        # inject MikkTSpace-compatible tangents for normal-mapped materials
        try:
            from .tangents import inject_tangents
            report_tangents = inject_tangents(path)
        except Exception as exc:  # never let post-processing break a save
            report_tangents = -1
            print(f"  ⚠ tangent injection skipped: {exc}")

        from .validate import inspect_glb
        from .fit import connection_report
        try:
            assembly_warnings = connection_report(self.parts)
        except Exception as exc:
            assembly_warnings = [f"assembly check skipped: {exc}"]
        report = inspect_glb(path)
        report["tangents_injected"] = report_tangents
        report["assembly_warnings"] = assembly_warnings
        for w in assembly_warnings:
            report["warnings"].append(w)
        report["export_seconds"] = round(time.time() - t0, 3)

        print(self._format_report(report, path))
        return report

    # ------------------------------------------------------------------
    @staticmethod
    def _format_report(report: dict, path: str) -> str:
        size_kb = report["file_size_bytes"] / 1024
        dims = report["bounds_size"]
        lines = [
            f"✓ saved {path}  ({size_kb:.0f} KB, {report['total_triangles']} tris, "
            f"{report['total_vertices']} verts, {report['parts']} parts, "
            f"{len(report['material_names'])} materials, {report['embedded_images']} textures)",
            f"  bounds: {dims[0]:.3f} × {dims[1]:.3f} × {dims[2]:.3f} m",
        ]
        if report.get("tangents_injected", 0) > 0:
            lines.append(f"  tangents: injected into {report['tangents_injected']} primitives (Khronos-clean)")
        for warning in report["warnings"]:
            lines.append(f"  ⚠ {warning}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.parts)

    def __repr__(self) -> str:
        return f"<Scene {self.name!r}: {len(self.parts)} parts>"
