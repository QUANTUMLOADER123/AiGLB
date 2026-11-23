"""Procedural aging / wear pipeline for PBR map sets.

Every effect is deterministic given ``seed`` and works on float32 HxWxC arrays
(0..1). The goal: one-line arguments that make surfaces look used — chipped
paint, rust, dirt, dust, scuffs, fingerprints — instead of fresh-from-the-oven
CG surfaces.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from . import textures as _tex

__all__ = ["age_maps", "default_chip_color", "default_rust_color"]

DEFAULT_CHIP = (0.42, 0.43, 0.46)   # bare steel under chipped paint
DEFAULT_RUST = (0.40, 0.18, 0.075)  # oxidized iron blotches
_DUST = (0.72, 0.70, 0.66)
_DIRT = (0.30, 0.25, 0.18)


def default_chip_color():
    return DEFAULT_CHIP


def default_rust_color():
    return DEFAULT_RUST


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _fbm(size, octaves, cells, seed):
    return _tex.fbm(size, octaves=octaves, seed=seed, base_cells=cells)


def _smoothstep(edge0, edge1, x):
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _as_f32(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0


def _to_u8(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGB")


def _lum(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _anisotropic_smudge(size, seed, aspect=3.0, octaves=4, cells=6):
    """Directional smear mask: stretches noise along one axis (fingerprints/scuffs)."""
    noise = _fbm(size, octaves=octaves, cells=cells, seed=seed)
    # take a slice of 2x-wide noise to fake elongated smudges
    wide = _fbm(size, octaves=octaves, cells=max(cells // 2, 2), seed=seed + 7)
    return _np_norm(noise * 0.65 + wide * 0.35)


def _np_norm(a):
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _scratch_overlay(size, seed, count, width_px, alpha_lo, alpha_hi,
                     min_len, max_len, angle_center=None, angle_spread=math.pi):
    """Draw scratch lines -> float mask (0..1) where 1 = strongest scratch."""
    if count <= 0:
        return np.zeros((size, size), dtype=np.float32)
    rng = np.random.default_rng(seed)
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    for _ in range(count):
        x0, y0 = rng.uniform(0, size, 2)
        if angle_center is None:
            ang = rng.uniform(0, 2 * math.pi)
        else:
            ang = angle_center + rng.uniform(-angle_spread, angle_spread)
        length = rng.uniform(min_len, max_len)
        x1, y1 = x0 + math.cos(ang) * length, y0 + math.sin(ang) * length
        alpha = int(rng.uniform(alpha_lo, alpha_hi) * 255)
        w = max(1, int(round(rng.uniform(width_px * 0.6, width_px * 1.5))))
        draw.line([(x0, y0), (x1, y1)], fill=alpha, width=w)
    return np.asarray(mask, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def age_maps(base: Image.Image | None, roughness: Image.Image | None,
             metalness: Image.Image | None, *,
             seed: int = 0, size: int | None = None,
             brightness: float = 1.0, contrast: float = 1.0,
             saturation: float = 1.0,
             tint: tuple | None = None,
             roughness_offset: float = 0.0, roughness_contrast: float = 1.0,
             wear: float = 0.0,
             scratches: float = 0.0, scuffs: float = 0.0,
             paint_chip: float = 0.0, chip_color=None,
             rust: float = 0.0, rust_color=None,
             dirt: float = 0.0, dust: float = 0.0,
             grunge: float = 0.0, fingerprints: float = 0.0):
    """Age a PBR map set. Pass PIL images (or None to synthesize placeholders).

    Returns ``(base_img, roughness_img, metalness_img_or_None)`` as PIL images.
    All knobs are 0..1 unless noted. ``wear`` is the master control: it raises
    the effective level of chipping/scuffing/dirt combined.
    """
    if base is None:
        raise ValueError("age_maps requires a basecolor image")
    size = size or base.size[0]
    rng = np.random.default_rng(seed)

    rgb = _as_f32(base)
    ruf = _as_f32(roughness)[..., 0] if roughness is not None else np.full((size, size), 0.75, np.float32)
    mel = _as_f32(metalness)[..., 0] if metalness is not None else None

    # ---- master wear: raises chip/scuff/dirt floors -----------------------
    chip_eff = min(1.0, paint_chip + wear * 0.60)
    scuff_eff = min(1.0, scuffs + wear * 0.55)
    scratch_eff = min(1.0, scratches + wear * 0.45)
    dirt_eff = min(1.0, dirt + wear * 0.45)

    # ---- global color adjustments ------------------------------------------
    if tint is not None:
        t = np.asarray(tint, dtype=np.float32).reshape(1, 1, 3)
        rgb = rgb * t
    if brightness != 1.0:
        rgb = rgb * np.float32(brightness)
    if contrast != 1.0:
        rgb = (rgb - 0.5) * np.float32(contrast) + 0.5
    if saturation != 1.0:
        luma = _lum(rgb)[..., None]
        rgb = luma + (rgb - luma) * np.float32(saturation)
    rgb = np.clip(rgb, 0.0, 1.0)

    if roughness_contrast != 1.0:
        ruf = np.clip((ruf - 0.5) * roughness_contrast + 0.5, 0.0, 1.0)

    def _blend(dst, color, mask):
        color = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
        m = mask[..., None].astype(np.float32)
        return dst * (1.0 - m) + color * m

    # ---- dirt: dark low-frequency grime -----------------------------------
    if dirt_eff > 0.001:
        m = _np_norm(_fbm(size, 4, 5, seed + 11))
        m = _smoothstep(1.0 - dirt_eff * 0.85, 1.0, m) * 0.75
        rgb = _blend(rgb, _DIRT, m)
        ruf = np.clip(ruf + m * 0.28, 0.0, 1.0)

    # ---- dust: bright high-frequency powder --------------------------------
    if dust > 0.001:
        m = _fbm(size, 5, 28, seed + 13)
        m = _np_norm(m) ** 1.6 * dust
        rgb = _blend(rgb, _DUST, m * 0.55)
        ruf = np.clip(ruf + m * 0.30, 0.0, 1.0)

    # ---- grunge: sooty splatter overlay (bundled DevTextures masks) --------
    if grunge > 0.001:
        from . import library as _lib
        names = _lib.grunge_names()
        pick = names[int(rng.integers(0, len(names)))] if names else None
        if pick:
            mask_img = Image.open(_lib.grunge_path(pick)).convert("L").resize((size, size), Image.LANCZOS)
            m = np.asarray(mask_img, dtype=np.float32) / 255.0
            m = _np_norm(m) ** 1.35 * grunge
            rgb = rgb * (1.0 - m[..., None] * 0.62)
            ruf = np.clip(ruf + m * 0.22, 0.0, 1.0)

    # ---- fingerprints: oily elongated smudges ------------------------------
    if fingerprints > 0.001:
        m = _anisotropic_smudge(size, seed + 17, octaves=4, cells=7)
        m = _smoothstep(0.55, 0.95, m) * fingerprints
        rgb = rgb * (1.0 - m[..., None] * 0.22)
        ruf = ruf * (1.0 - m * 0.38)

    # ---- scuffs: long, aligned, soft drag streaks ---------------------------
    if scuff_eff > 0.001:
        cnt = int(scuff_eff * 46) + 2
        ang = float(rng.uniform(0, math.pi))
        m = _scratch_overlay(size, seed + 19, cnt, width_px=2.2,
                             alpha_lo=0.25, alpha_hi=0.6,
                             min_len=size * 0.35, max_len=size * 1.1,
                             angle_center=ang, angle_spread=0.30)
        m = m * scuff_eff
        rgb = _blend(rgb, (0.80, 0.79, 0.76), m * 0.55)
        ruf = ruf * (1.0 - m * 0.35)

    # ---- scratches: chaotic bright scratches, polished floor ---------------
    if scratch_eff > 0.001:
        cnt = int(scratch_eff * 80) + 2
        m = _scratch_overlay(size, seed + 23, cnt, width_px=1.3,
                             alpha_lo=0.35, alpha_hi=0.9,
                             min_len=size * 0.06, max_len=size * 0.6)
        m = m * scratch_eff
        rgb = _blend(rgb, (0.78, 0.77, 0.74), m * 0.7)
        ruf = ruf * (1.0 - m * 0.5)

    # ---- paint chip: masked erosion revealing bare substrate ----------------
    if chip_eff > 0.001:
        chip = np.asarray(chip_color if chip_color is not None else DEFAULT_CHIP, dtype=np.float32)
        m = _np_norm(_fbm(size, 5, 7, seed + 29))
        m = _smoothstep(1.0 - chip_eff * 0.72, 1.0, m)
        # chip edges get a thin lighter halo (worn lip)
        halo = _smoothstep(1.0 - chip_eff * 0.72 - 0.09, 1.0 - chip_eff * 0.72, m) * (1.0 - m)
        rgb = _blend(rgb, chip, m * 0.92)
        rgb = np.clip(rgb + halo[..., None] * 0.10, 0.0, 1.0)
        ruf = np.clip(ruf * (1.0 - m * 0.45) + halo * 0.05, 0.0, 1.0)
        if mel is not None:
            mel = np.clip(mel * (1.0 - m) + m * 0.95, 0.0, 1.0)
        else:
            mel = m * 0.95

    # ---- rust: oxidized bloom, kills metalness ------------------------------
    if rust > 0.001:
        rustc = np.asarray(rust_color if rust_color is not None else DEFAULT_RUST, dtype=np.float32)
        m = _np_norm(_fbm(size, 5, 9, seed + 31))
        m = _smoothstep(1.0 - rust * 0.62, 1.0, m)
        variation = _fbm(size, 3, 24, seed + 37)[..., None]
        rust_rgb = rustc.reshape(1, 1, 3) * (0.65 + 0.7 * variation)
        rgb = rgb * (1.0 - m[..., None]) + rust_rgb * m[..., None]
        ruf = np.clip(ruf + m * 0.35, 0.0, 1.0)
        if mel is not None:
            mel = mel * (1.0 - m)
        else:
            mel = np.zeros((size, size), dtype=np.float32)

    if roughness_offset != 0.0:
        ruf = np.clip(ruf + np.float32(roughness_offset), 0.0, 1.0)

    metal_img = _to_u8(np.stack([mel, mel, mel], axis=-1)) if mel is not None else None
    return _to_u8(rgb), _to_u8(np.stack([ruf, ruf, ruf], axis=-1)), metal_img


def pack_metallic_roughness(roughness_img: Image.Image, metalness_img: Image.Image | None,
                            size: int, metal_constant: float | None = None) -> Image.Image:
    """glTF metallicRoughnessTexture: G = roughness, B = metalness.

    Precedence for B: explicit metalness image > ``metal_constant`` fill > 1.0.
    """
    ruf = np.asarray(roughness_img.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)
    if metalness_img is not None:
        mel = np.asarray(metalness_img.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)
    elif metal_constant is not None:
        mel = np.full((size, size), float(np.clip(metal_constant, 0.0, 1.0)) * 255.0, dtype=np.float32)
    else:
        mel = np.full((size, size), 255.0, dtype=np.float32)
    out = np.stack([np.full_like(ruf, 255.0), ruf, mel], axis=-1)
    return Image.fromarray(out.astype(np.uint8), "RGB")


def scale_normal_image(img: Image.Image, strength: float) -> Image.Image:
    """Multiply the tangent-space XY component of a normal map (renormalized)."""
    if strength == 1.0:
        return img
    arr = _as_f32(img) * 2.0 - 1.0
    n = np.empty_like(arr)
    n[..., 0] = arr[..., 0] * strength
    n[..., 1] = arr[..., 1] * strength
    n[..., 2] = arr[..., 2]
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    ln[ln == 0] = 1.0
    n = n / ln
    return _to_u8(n * 0.5 + 0.5)
