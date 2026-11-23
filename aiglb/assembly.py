"""Assembly diagnostics for exported .glb files (used by ``aiglb check``).

Reuses the same machinery as the build-time report: detects parts that float
(touch nothing) and parts fully hidden inside other parts.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .fit import connection_report

__all__ = ["assembly_report_path", "assembly_issues"]


class _Part:
    def __init__(self, name, mesh):
        self.name = name
        self.mesh = mesh


def assembly_issues(path: str, gap: float = 0.004) -> list[str]:
    obj = trimesh.load(path)
    parts: list[_Part] = []
    if isinstance(obj, trimesh.Scene):
        # one entry per graph node that carries geometry, with world transform applied
        for node_name in obj.graph.nodes_geometry:
            transform, geom_name = obj.graph[node_name]
            mesh = obj.geometry.get(geom_name)
            if mesh is None:
                continue
            mesh = mesh.copy()
            mesh.apply_transform(transform)
            parts.append(_Part(node_name, mesh))
    elif isinstance(obj, trimesh.Trimesh):
        parts.append(_Part("model", obj))
    else:
        try:
            geoms = list(obj.geometry.values())
            parts = [_Part(f"part_{i}", g) for i, g in enumerate(geoms)]
        except Exception:
            return [f"could not parse {path} for assembly analysis"]
    if len(parts) < 2:
        return ["assembly: single-part model, nothing to check"]
    return connection_report(parts, gap=gap)


def assembly_report_path(path: str) -> str:
    warnings = assembly_issues(path)
    lines = ["Assembly:"]
    if not warnings:
        lines.append("  ✓ no floating or hidden parts detected")
        return "\n".join(lines)
    for w in warnings:
        lines.append(f"  ⚠ {w}")
    return "\n".join(lines)
