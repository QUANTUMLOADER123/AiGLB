"""Bundled CC0 texture library access.

Ships 245 real, tileable 512 px PBR texture sets (base color / normal /
roughness / metalness) transcoded from CC0 collections:
ambientCG, CGBookcase, TextureCan, Poly Haven, ShareTextures and DevTextures.

Use :func:`list_library` / :func:`library_info` to discover entries and
``library_material(...)`` (in ``aiglb.materials``) to build materials.
Everything works fully offline — the files live inside the package.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Iterable

from PIL import Image

__all__ = [
    "list_library", "library_info", "library_categories", "library_path",
    "load_maps", "texture_index", "LICENSE_NOTE",
]

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
_TEXTURES = os.path.join(_ASSETS, "textures")
_GRUNGE = os.path.join(_ASSETS, "grunge")

LICENSE_NOTE = (
    "All bundled textures are CC0-1.0 (public domain). "
    "See aiglb/assets/TEXTURE_LICENSES.txt for attribution."
)


@lru_cache(maxsize=1)
def texture_index() -> dict:
    with open(os.path.join(_TEXTURES, "index.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _by_name() -> dict:
    return {t["name"]: t for t in texture_index()["textures"]}


def library_path(name: str) -> str:
    """Absolute directory of a bundled texture set (raises KeyError if unknown)."""
    entry = _by_name().get(name)
    if entry is None:
        raise KeyError(f"Unknown bundled texture '{name}'. "
                       f"{len(_by_name())} sets available — see list_library().")
    return os.path.join(_TEXTURES, *name.split("/"))


def list_library(category: str | None = None) -> list[str]:
    """Names of bundled texture sets, e.g. ``'ambientcg/PaintedMetal004'``.

    Pass ``category=`` to filter (see :func:`library_categories`).
    """
    items = texture_index()["textures"]
    if category is not None:
        cat = category.lower().strip()
        items = [t for t in items if t["category"] == cat]
    return sorted(t["name"] for t in items)


def library_categories() -> dict[str, int]:
    """Category -> number of bundled sets."""
    counts: dict[str, int] = {}
    for t in texture_index()["textures"]:
        counts[t["category"]] = counts.get(t["category"], 0) + 1
    return dict(sorted(counts.items()))


def library_info(name: str) -> dict:
    """Metadata dict for a bundled set: maps, category, license, source."""
    entry = _by_name().get(name)
    if entry is None:
        raise KeyError(f"Unknown bundled texture '{name}'.")
    return dict(entry)


@lru_cache(maxsize=64)
def _decoded_maps(name: str):
    """Shared decoded images per set (treated read-only by callers)."""
    info = library_info(name)
    d = library_path(name)
    out: dict = {}
    for kind in ("basecolor", "normal", "roughness", "metalness"):
        fname = info.get(kind)
        if not fname:
            continue
        img = Image.open(os.path.join(d, fname))
        img.load()
        fmt = img.format  # keep .format so the exporter embeds original JPEG bytes
        out[kind] = img.convert("RGB")
        out[kind].format = fmt
    return out


def load_maps(name: str, maps: Iterable[str] = ("basecolor", "normal", "roughness", "metalness")) -> dict:
    """Load maps of a bundled set as PIL images.

    Returns a dict like ``{'basecolor': Image(RGB), 'normal': Image(RGB),
    'roughness': Image(RGB), 'metalness': Image(RGB) | None,
    'metallic_constant': float | None}`` — keys absent when a set has no map.
    Images are shared cached instances; treat them as read-only.
    """
    info = library_info(name)
    shared = _decoded_maps(name)
    out = {k: v for k, v in shared.items() if k in maps}
    if "metalness" not in out and "metallic" in info:
        out["metallic_constant"] = float(info["metallic"])
    return out


def grunge_path(name: str) -> str:
    """Absolute path of a bundled grunge/splatter overlay ('splatter01'... 'stain01')."""
    path = os.path.join(_GRUNGE, name.lower() + ".png")
    if not os.path.exists(path):
        raise KeyError(f"Unknown grunge map '{name}'. "
                       f"Available: {', '.join(grunge_names())}")
    return path


def grunge_names() -> list[str]:
    return sorted(f[:-4] for f in os.listdir(_GRUNGE) if f.endswith(".png"))
