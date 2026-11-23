"""PBR material factory for aiglb glTF exports.

Two ways to get a material:
  * ``color_material("#8899aa", roughness=0.4, metalness=0.0)`` — plain PBR color
  * ``texture_material("wood", ...)``                          — procedural map
  * a preset name string, e.g. ``"oak"`` — see ``list_presets()``
"""
from __future__ import annotations

from typing import Union

import numpy as np
from PIL import Image
import trimesh
from trimesh.visual.material import PBRMaterial

from . import textures

__all__ = [
    "color_material", "texture_material", "preset_material",
    "list_presets", "PRESET_NAMES", "resolve_material",
]


ColorLike = Union[str, tuple, list, np.ndarray]


def _to_rgba_u8(color: ColorLike, alpha: float = 1.0) -> np.ndarray:
    """Normalize hex '#rrggbb' / '#rgb' / 0-255 / 0-1 float sequences to uint8 RGBA."""
    if color is None:
        color = (200, 200, 200)
    if isinstance(color, str):
        s = color.lstrip("#")
        if len(s) == 3:
            s = "".join(c * 2 for c in s)
        if len(s) == 6:
            s += "ff"
        if len(s) != 8:
            raise ValueError(f"Bad color '{color}', expected '#rgb' '#rrggbb' or '#rrggbbaa'")
        arr = np.array([int(s[i:i + 2], 16) for i in range(0, 8, 2)], dtype=np.uint8)
        arr[3] = int(round(arr[3] / 255 * alpha * 255)) if alpha != 1.0 else arr[3]
        return arr
    arr = np.asarray(color, dtype=np.float64).ravel()
    if arr.shape[0] == 3:
        arr = np.concatenate([arr, [alpha]])
    elif arr.shape[0] != 4:
        raise ValueError(f"Color sequence must have 3 or 4 components, got {arr.shape[0]}")
    if arr.max() <= 1.0 and arr.min() >= 0.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _factor01(color: ColorLike) -> list[float]:
    arr = _to_rgba_u8(color).astype(np.float64) / 255.0
    return arr.tolist()


def color_material(color: ColorLike, roughness: float = 0.45, metalness: float = 0.0,
                   emissive: ColorLike | None = None, emissive_intensity: float = 1.0,
                   opacity: float = 1.0, name: str | None = None,
                   double_sided: bool = False) -> PBRMaterial:
    """Plain glTF metallic-roughness material.

    Parameters
    ----------
    color : hex string ('#ff8800'), or RGB(A) sequence (0-255 or 0-1)
    roughness, metalness : 0..1
    emissive : optional glow color (e.g. for lamps, screens)
    emissive_intensity : multiplier on emissive color
    opacity : < 1 enables alpha blending (glass-like)
    """
    alpha = float(np.clip(opacity, 0.0, 1.0))
    key = "col:" + repr((repr(color), round(float(roughness), 4), round(float(metalness), 4),
                        repr(emissive), round(float(emissive_intensity), 4),
                        round(alpha, 4), double_sided))
    cached = _cache_get(key)
    if cached is not None:
        return cached
    mat = PBRMaterial(
        name=name or "color_material",
        baseColorFactor=_to_rgba_u8(color, alpha=alpha),
        metallicFactor=float(np.clip(metalness, 0, 1)),
        roughnessFactor=float(np.clip(roughness, 0, 1)),
        doubleSided=bool(double_sided),
    )
    if alpha < 1.0:
        mat.alphaMode = "BLEND"
        if not np.isclose(alpha, _to_rgba_u8(color, alpha)[3] / 255.0):
            pass
        mat.baseColorFactor = _to_rgba_u8(color, alpha=alpha)
    if emissive is not None:
        e = np.array(_factor01(emissive)[:3], dtype=np.float64) * float(emissive_intensity)
        e = np.clip(e, 0.0, 1.0)  # glTF requires emissiveFactor components in [0, 1]
        mat.emissiveFactor = e.tolist()
    _cache_put(key, mat)
    return mat


def texture_material(texture: str, *, size: int = 512, seed: int | None = None,
                     roughness: float = 0.75, metalness: float = 0.0,
                     normal: bool = True, normal_strength: float | None = None,
                     emissive_texture: str | None = None,
                     emissive_intensity: float = 1.0,
                     name: str | None = None,
                     double_sided: bool = False,
                     **texture_kwargs) -> PBRMaterial:
    """PBR material from a built-in procedural texture (see ``list_textures()``).

    Extra keyword arguments (e.g. ``base=(0.4, 0.2, 0.1)`` for fabric/leather,
    ``light=``/``dark=`` for wood/marble, ``tint=``) are forwarded to the
    texture generator. See aiglb/textures.py for per-texture parameters.
    """
    key = "tex:" + repr((texture, size, seed, round(float(roughness), 4),
                         round(float(metalness), 4), normal, normal_strength,
                         emissive_texture, round(float(emissive_intensity), 4),
                         double_sided, repr(sorted(texture_kwargs.items()))))
    cached = _cache_get(key)
    if cached is not None:
        return cached
    if texture_kwargs:
        fn = textures._lookup(texture)
        import functools
        import inspect
        # validate kwargs early with a clear error
        params = inspect.signature(fn).parameters
        unknown = set(texture_kwargs) - set(params)
        if unknown:
            raise TypeError(f"texture '{texture}' got unexpected options: {sorted(unknown)}. "
                            f"Available: {sorted(params)}")
        img, _ = fn(size, seed if seed is not None else 0, **texture_kwargs)
        albedo = img
    else:
        albedo = textures.make_texture(texture, size=size, seed=seed)
    mat = PBRMaterial(
        name=name or texture,
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        baseColorTexture=albedo,
        metallicFactor=float(np.clip(metalness, 0, 1)),
        roughnessFactor=float(np.clip(roughness, 0, 1)),
        doubleSided=bool(double_sided),
    )
    if normal:
        if texture_kwargs:
            fn = textures._lookup(texture)
            _, height = fn(size, seed if seed is not None else 0, **texture_kwargs)
            s = normal_strength if normal_strength is not None else textures._NORMAL_STRENGTH.get(texture, 1.0)
            mat.normalTexture = textures.normal_from_height(height, strength=s)
        else:
            mat.normalTexture = textures.make_normal_map(texture, size=size, seed=seed,
                                                        strength=normal_strength)
    if emissive_texture is not None:
        mat.emissiveTexture = textures.make_emissive(emissive_texture, size=size, seed=seed)
        e = float(np.clip(emissive_intensity, 0.0, 1.0))
        mat.emissiveFactor = [e, e, e]
    _cache_put(key, mat)
    return mat


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------

def _presets() -> dict[str, PBRMaterial]:
    """Fresh copies of every preset (materials are cheap to build)."""
    tex = lambda texname, **kw: texture_material(texname, **kw)  # noqa: E731
    col = color_material
    presets = {
        # --- metals ---
        "chrome":        lambda: col("#f2f4f6", roughness=0.06, metalness=1.0, name="chrome"),
        "steel":         lambda: tex("brushed_metal", roughness=0.32, metalness=1.0,
                                     normal=True, normal_strength=0.35, name="steel"),
        "brushed_steel": lambda: tex("brushed_metal", roughness=0.42, metalness=1.0,
                                     normal=True, normal_strength=0.5, name="brushed_steel"),
        "aluminum":      lambda: col("#c9ccd1", roughness=0.30, metalness=1.0, name="aluminum"),
        "gunmetal":      lambda: col("#4a4d52", roughness=0.38, metalness=1.0, name="gunmetal"),
        "black_metal":   lambda: col("#1c1d1f", roughness=0.45, metalness=0.9, name="black_metal"),
        "gold":          lambda: col("#d4af37", roughness=0.16, metalness=1.0, name="gold"),
        "brass":         lambda: col("#b08d3e", roughness=0.28, metalness=1.0, name="brass"),
        "copper":        lambda: col("#b46a41", roughness=0.26, metalness=1.0, name="copper"),
        # --- woods ---
        "oak":           lambda: tex("wood", roughness=0.62, metalness=0.0, name="oak"),
        "walnut":        lambda: tex("wood", seed=7, roughness=0.58, name="walnut"),
        "pine":          lambda: col("#c8a878", roughness=0.68, name="pine"),
        "dark_wood":     lambda: tex("wood_planks", roughness=0.6, name="dark_wood"),
        "planks":        lambda: tex("wood_planks", roughness=0.64, name="planks"),
        # --- stone / masonry ---
        "marble":        lambda: tex("marble", roughness=0.32, name="marble"),
        "granite":       lambda: tex("granite", roughness=0.55, name="granite"),
        "brick":         lambda: tex("brick", roughness=0.9, name="brick"),
        "concrete":      lambda: tex("concrete", roughness=0.88, name="concrete"),
        "terracotta":    lambda: tex("terracotta", roughness=0.8, name="terracotta"),
        # --- ceramic & plastic ---
        "ceramic_white": lambda: tex("ceramic", roughness=0.22, name="ceramic_white"),
        "ceramic_black": lambda: col("#141414", roughness=0.25, name="ceramic_black"),
        "porcelain":     lambda: col("#f5f2ec", roughness=0.18, name="porcelain"),
        "plastic_red":    lambda: col("#c92a2a", roughness=0.32, name="plastic_red"),
        "plastic_orange": lambda: col("#e8590c", roughness=0.32, name="plastic_orange"),
        "plastic_yellow": lambda: col("#f2b705", roughness=0.32, name="plastic_yellow"),
        "plastic_green":  lambda: col("#2b8a3e", roughness=0.34, name="plastic_green"),
        "plastic_blue":   lambda: col("#1971c2", roughness=0.32, name="plastic_blue"),
        "plastic_white":  lambda: col("#e9ecef", roughness=0.38, name="plastic_white"),
        "plastic_black":  lambda: col("#151517", roughness=0.36, name="plastic_black"),
        "rubber_black":   lambda: col("#0f0f10", roughness=0.92, name="rubber_black"),
        "rubber_gray":    lambda: col("#3a3b3d", roughness=0.9, name="rubber_gray"),
        # --- fabric & leather ---
        "fabric_blue":   lambda: tex("fabric", roughness=0.95, name="fabric_blue"),
        "fabric_red":    lambda: tex("fabric", seed=4, base=(0.55, 0.13, 0.13), roughness=0.95,
                                     name="fabric_red"),
        "fabric_gray":   lambda: tex("fabric", seed=5, base=(0.38, 0.38, 0.40), roughness=0.95,
                                     name="fabric_gray"),
        "leather_brown": lambda: tex("leather", roughness=0.72, name="leather_brown"),
        "leather_black": lambda: tex("leather", seed=6, base=(0.09, 0.08, 0.08), roughness=0.7,
                                     name="leather_black"),
        # --- organic ---
        "grass":         lambda: tex("grass", roughness=1.0, name="grass"),
        "leaf_green":    lambda: col("#2f6b2f", roughness=0.75, name="leaf_green"),
        "bark":          lambda: tex("wood", seed=9, roughness=0.95, name="bark"),
        # --- special ---
        "glass":         lambda: col("#cfe8ef", roughness=0.05, metalness=0.0, opacity=0.32,
                                     name="glass"),
        "water":         lambda: col("#2d6a8f", roughness=0.08, metalness=0.0, opacity=0.68,
                                     name="water"),
        "checker_floor": lambda: tex("checker", roughness=0.4, name="checker_floor"),
        "grid_tiles":    lambda: tex("grid", roughness=0.5, name="grid_tiles"),
        "lava":          lambda: tex("lava", roughness=0.85, normal_strength=1.6,
                                     emissive_texture="lava", emissive_intensity=1.6,
                                     name="lava"),
        "ember":         lambda: col("#2a1408", roughness=0.9, emissive="#ff7b1c",
                                     emissive_intensity=1.8, name="ember"),
        "lamp_warm":     lambda: col("#fff3d6", roughness=0.5, emissive="#ffd98a",
                                     emissive_intensity=1.35, name="lamp_warm"),
        "neon_pink":     lambda: col("#ff4fd8", roughness=0.4, emissive="#ff2fd0",
                                     emissive_intensity=2.0, name="neon_pink"),
        "neon_cyan":     lambda: col("#37e6ff", roughness=0.4, emissive="#19d8ff",
                                     emissive_intensity=2.0, name="neon_cyan"),
        "screen":        lambda: col("#0b1020", roughness=0.15, emissive="#4d6bff",
                                     emissive_intensity=0.9, name="screen"),
        "mirror":        lambda: col("#ffffff", roughness=0.0, metalness=1.0, name="mirror"),
    }
    return presets


PRESET_NAMES = sorted(_presets())
_material_cache: dict = {}


def _cache_get(key):
    return _material_cache.get(key)


def _cache_put(key, mat):
    _material_cache[key] = mat
    return mat


def list_presets() -> list[str]:
    return list(PRESET_NAMES)


def clear_material_cache() -> None:
    """Drop cached shared material instances (rarely needed)."""
    _material_cache.clear()


def _cached(key: str, factory):
    mat = _material_cache.get(key)
    if mat is None:
        mat = factory()
        _material_cache[key] = mat
    return mat


def preset_material(name: str) -> PBRMaterial:
    presets = _presets()
    key = name.lower().strip()
    if key not in presets:
        raise KeyError(f"Unknown material preset '{name}'. Available: {', '.join(sorted(presets))}")
    # shared instances: identical presets embedded once in the .glb
    return _cached(f"preset:{key}", presets[key])


def resolve_material(material: Union[str, PBRMaterial, None]) -> PBRMaterial:
    """Accepts a preset name, a PBRMaterial, or None (neutral gray)."""
    if material is None:
        return color_material("#9a9a9c", roughness=0.6, name="default_gray")
    if isinstance(material, PBRMaterial):
        return material
    if isinstance(material, str):
        low = material.lower().strip()
        if low in PRESET_NAMES:
            return preset_material(low)
        # treat unknown strings as hex colors if they look like one
        if low.startswith("#") or all(c in "0123456789abcdef" for c in low):
            return color_material(low, name=low)
        raise KeyError(f"Unknown material preset '{material}'. "
                       f"Available presets: {', '.join(PRESET_NAMES)} "
                       f"(or pass a hex color like '#88aaff')")
    raise TypeError(f"material must be a preset name, hex color, or PBRMaterial — got {type(material)}")
