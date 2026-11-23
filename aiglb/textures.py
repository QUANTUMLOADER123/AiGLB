"""Procedural PBR texture synthesis for aiglb.

All textures are generated with numpy (tileable where it matters) and returned
as PIL images, so no external asset packs are ever needed. Every generator can
also produce a height field, which we convert into a tangent-space normal map.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

__all__ = [
    "list_textures", "make_texture", "make_normal_map", "make_emissive",
    "normal_from_height", "TEXTURE_NAMES",
]


# --------------------------------------------------------------------------
# noise helpers (tileable value noise + fBm)
# --------------------------------------------------------------------------

def _value_noise(size: int, cells: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = rng.random((cells, cells))
    lin = np.linspace(0, cells, size, endpoint=False)
    X, Y = np.meshgrid(lin, lin)
    xi = np.floor(X).astype(np.int64)
    yi = np.floor(Y).astype(np.int64)
    fx = X - xi
    fy = Y - yi
    sx = fx * fx * (3.0 - 2.0 * fx)
    sy = fy * fy * (3.0 - 2.0 * fy)
    x0, x1 = xi % cells, (xi + 1) % cells
    y0, y1 = yi % cells, (yi + 1) % cells
    n = (g[y0, x0] * (1 - sx) * (1 - sy) + g[y0, x1] * sx * (1 - sy)
         + g[y1, x0] * (1 - sx) * sy + g[y1, x1] * sx * sy)
    return n


def fbm(size: int, octaves: int = 4, seed: int = 0, base_cells: int = 4,
        gain: float = 0.5) -> np.ndarray:
    total = np.zeros((size, size), dtype=np.float64)
    amp = 1.0
    cells = base_cells
    norm = 0.0
    s = seed
    for _ in range(octaves):
        total += amp * _value_noise(size, cells, s)
        norm += amp
        amp *= gain
        cells *= 2
        s += 101
    return total / norm


def _hash2(a: np.ndarray, b: np.ndarray, seed: int = 0) -> np.ndarray:
    """Deterministic pseudo-random per (a, b) pair, in [0, 1)."""
    h = np.sin(a * 127.1 + b * 311.7 + seed * 74.7) * 43758.5453
    return h - np.floor(h)


def _to_img(rgb: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8), "RGB")


def _grid(size: int):
    lin = (np.arange(size) + 0.5) / size
    return np.meshgrid(lin, lin)  # X, Y in [0, 1)


def _mix(c0, c1, t):
    c0 = np.asarray(c0, dtype=np.float64)
    c1 = np.asarray(c1, dtype=np.float64)
    return c0 + (c1 - c0) * t[..., None]


def _normalize01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-9:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


# --------------------------------------------------------------------------
# height-field -> normal map
# --------------------------------------------------------------------------

def normal_from_height(height: np.ndarray, strength: float = 2.0) -> Image.Image:
    h = height.astype(np.float64)
    gy, gx = np.gradient(h)
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(h)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    n = np.stack([nx / length, ny / length, nz / length], axis=-1)
    return Image.fromarray(((n * 0.5 + 0.5) * 255.0 + 0.5).astype(np.uint8), "RGB")


# --------------------------------------------------------------------------
# individual generators:  name(size, seed) -> (color_img, height_arr)
# --------------------------------------------------------------------------

def _wood(size, seed=0, light=(0.72, 0.54, 0.33), dark=(0.36, 0.22, 0.12)):
    X, Y = _grid(size)
    warp = fbm(size, octaves=4, seed=seed + 1, base_cells=4)
    fine = fbm(size, octaves=5, seed=seed + 2, base_cells=8)
    v = X[0] * 9.0 + warp * 3.2 + fine * 0.6
    t = 0.5 + 0.5 * np.sin(v * 2 * np.pi)
    t = t ** 1.6
    rgb = _mix(dark, light, t)
    height = t * 0.6 + fine * 0.4
    return _to_img(rgb), height


def _wood_planks(size, seed=0):
    X, Y = _grid(size)
    warp = fbm(size, octaves=4, seed=seed + 3, base_cells=6)
    planks = 6
    row = np.floor(Y * planks).astype(np.int64)
    shade = _hash2(row, np.zeros_like(row), seed) * 0.35 + 0.75
    v = X * 14.0 + warp * 2.5
    t = 0.5 + 0.5 * np.sin(v * 2 * np.pi)
    base = _mix((0.30, 0.18, 0.09), (0.62, 0.42, 0.22), t ** 1.4)
    seam_y = np.abs((Y * planks) % 1.0 - 0.5)
    seam = (seam_y > 0.46)[..., None]
    rgb = base * shade[..., None]
    rgb = np.where(seam, np.array([0.08, 0.05, 0.03]), rgb)
    height = np.where(seam[..., 0], 0.0, t)
    return _to_img(rgb), height


def _marble(size, seed=0, light=(0.93, 0.92, 0.90), dark=(0.32, 0.33, 0.36)):
    X, Y = _grid(size)
    warp = fbm(size, octaves=5, seed=seed + 5, base_cells=4)
    v = X * 3.0 + Y * 1.5 + warp * 4.5
    s = np.abs(np.sin(v * np.pi))
    vein = s ** 10.0  # thin sharp veins
    vein2 = (np.abs(np.sin((v * 0.6 + warp * 2.0) * np.pi))) ** 24.0
    rgb = _mix(light, dark, np.clip(vein * 0.75 + vein2, 0, 1))
    height = 1.0 - np.clip(vein * 0.8 + vein2, 0, 1)
    return _to_img(rgb), height


def _granite(size, seed=0):
    base = fbm(size, octaves=5, seed=seed + 7, base_cells=8)
    speck1 = (fbm(size, octaves=2, seed=seed + 8, base_cells=48) > 0.62)
    speck2 = (fbm(size, octaves=2, seed=seed + 9, base_cells=64) < 0.35)
    rgb = _mix((0.42, 0.42, 0.44), (0.72, 0.72, 0.74), base)
    rgb = np.where(speck1[..., None], np.array([0.18, 0.18, 0.20]), rgb)
    rgb = np.where(speck2[..., None], np.array([0.88, 0.88, 0.90]), rgb)
    return _to_img(rgb), base


def _brick(size, seed=0):
    X, Y = _grid(size)
    rows, cols = 8, 4
    rowm = np.floor(Y * rows).astype(np.int64)
    offset = (rowm % 2) * 0.5
    Xm = X + offset / cols * 2
    col = np.floor(Xm * cols)
    fx = (Xm * cols) % 1.0
    fy = (Y * rows) % 1.0
    mortar_w, mortar_h = 0.055, 0.075
    is_mortar = (fx < mortar_w) | (fy < mortar_h)
    h = _hash2(col, rowm, seed)
    brick_base = _mix((0.42, 0.16, 0.10), (0.72, 0.33, 0.20), h)
    jitter = fbm(size, octaves=3, seed=seed + 11, base_cells=16)
    brick_rgb = brick_base * (0.82 + 0.36 * jitter)[..., None]
    mortar_rgb = np.array([0.78, 0.75, 0.70]) * (0.9 + 0.2 * jitter)[..., None]
    rgb = np.where(is_mortar[..., None], mortar_rgb, brick_rgb)
    height = np.where(is_mortar, 0.0, 1.0) * (0.7 + 0.3 * jitter)
    return _to_img(rgb), height


def _concrete(size, seed=0):
    base = fbm(size, octaves=5, seed=seed + 13, base_cells=6)
    fine = fbm(size, octaves=6, seed=seed + 14, base_cells=32)
    rgb = _mix((0.52, 0.52, 0.51), (0.74, 0.74, 0.73), base * 0.75 + fine * 0.25)
    return _to_img(rgb), base * 0.6 + fine * 0.4


def _ceramic(size, seed=0, tint=(0.94, 0.93, 0.91)):
    n = fbm(size, octaves=3, seed=seed + 15, base_cells=5)
    rgb = np.array(tint) * (0.92 + 0.12 * n[..., None])
    return _to_img(rgb), n * 0.15


def _terracotta(size, seed=0):
    n = fbm(size, octaves=4, seed=seed + 16, base_cells=6)
    rgb = _mix((0.55, 0.26, 0.14), (0.78, 0.42, 0.24), n)
    return _to_img(rgb), n


def _fabric(size, seed=0, base=(0.23, 0.33, 0.55)):
    X, Y = _grid(size)
    warp = np.sin(Y * np.pi * 96) * 0.5 + np.sin(X * np.pi * 96) * 0.5
    n = fbm(size, octaves=4, seed=seed + 17, base_cells=10)
    shade = 0.75 + 0.25 * (0.5 + 0.5 * warp) * (0.7 + 0.6 * n)
    rgb = np.array(base) * shade[..., None]
    return _to_img(rgb), 0.5 + 0.3 * warp + 0.2 * n


def _leather(size, seed=0, base=(0.38, 0.21, 0.11)):
    n = fbm(size, octaves=6, seed=seed + 18, base_cells=9)
    cracks = _normalize01(fbm(size, octaves=5, seed=seed + 19, base_cells=22))
    crack_lines = (cracks > 0.68) & (cracks < 0.74)
    rgb = _mix(np.array(base) * 0.55, np.array(base) * 1.35, n)
    rgb = np.where(crack_lines[..., None], np.array(base) * 0.4, rgb)
    return _to_img(rgb), n * 0.7 + (~crack_lines) * 0.3


def _brushed_metal(size, seed=0, base=(0.72, 0.74, 0.78)):
    rng = np.random.default_rng(seed + 20)
    streaks = rng.random((size, max(size // 24, 1)))
    streaks_img = Image.fromarray((streaks * 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    s = np.asarray(streaks_img, dtype=np.float64) / 255.0
    broad = fbm(size, octaves=3, seed=seed + 21, base_cells=3)
    shade = 0.72 + 0.35 * s * 0.6 + 0.15 * broad
    rgb = np.array(base) * shade[..., None]
    return _to_img(rgb), s * 0.4


def _grass(size, seed=0):
    base = fbm(size, octaves=5, seed=seed + 22, base_cells=6)
    blades = fbm(size, octaves=3, seed=seed + 23, base_cells=48)
    dark = np.array([0.10, 0.24, 0.07])
    light = np.array([0.30, 0.50, 0.14])
    rgb = _mix(dark, light, _normalize01(base * 0.65 + blades * 0.35))
    return _to_img(rgb), blades


def _checker(size, seed=0, c1=(0.90, 0.89, 0.86), c2=(0.12, 0.12, 0.13), n=8):
    X, Y = _grid(size)
    m = ((np.floor(X * n) + np.floor(Y * n)) % 2 == 0)
    rgb = np.where(m[..., None], np.array(c1), np.array(c2))
    return _to_img(rgb), np.where(m, 1.0, 0.35)


def _stripes(size, seed=0, c1=(0.85, 0.83, 0.80), c2=(0.15, 0.16, 0.20), n=10):
    X, Y = _grid(size)
    m = (np.floor(X * n) % 2 == 0)
    rgb = np.where(m[..., None], np.array(c1), np.array(c2))
    return _to_img(rgb), m.astype(np.float64)


def _dots(size, seed=0, bg=(0.88, 0.86, 0.82), dot=(0.20, 0.22, 0.30), n=8, rad=0.16):
    X, Y = _grid(size)
    cx = (X * n) % 1.0 - 0.5
    cy = (Y * n) % 1.0 - 0.5
    m = np.sqrt(cx * cx + cy * cy) < rad
    rgb = np.where(m[..., None], np.array(dot), np.array(bg))
    return _to_img(rgb), np.where(m, 1.0, 0.0)


def _grid_tiles(size, seed=0, bg=(0.80, 0.78, 0.74), line=(0.45, 0.44, 0.42), n=8, w=0.03):
    X, Y = _grid(size)
    fx = (X * n) % 1.0
    fy = (Y * n) % 1.0
    m = (fx < w) | (fy < w)
    rgb = np.where(m[..., None], np.array(line), np.array(bg))
    return _to_img(rgb), np.where(m, 0.0, 1.0)


def _lava(size, seed=0):
    warp = fbm(size, octaves=5, seed=seed + 24, base_cells=5)
    ridge = 1.0 - np.abs(2.0 * warp - 1.0)  # ridge lines -> cracks
    crack = _normalize01(ridge) ** 6.0
    rock = fbm(size, octaves=4, seed=seed + 25, base_cells=9)
    rock_rgb = _mix((0.05, 0.04, 0.04), (0.16, 0.13, 0.12), rock)
    glow = np.zeros((size, size, 3))
    glow[..., 0] = crack                      # hot orange cracks
    glow[..., 1] = crack * 0.32
    glow[..., 2] = crack * 0.02
    rgb = rock_rgb + glow * 0.9
    return _to_img(rgb), crack


def _gradient_sky(size, seed=0, top=(0.35, 0.55, 0.85), bottom=(0.85, 0.88, 0.92)):
    X, Y = _grid(size)
    t = Y  # vertical gradient
    rgb = _mix(top, bottom, t)
    return _to_img(rgb), t


_GENERATORS = {
    "wood": _wood,
    "wood_planks": _wood_planks,
    "marble": _marble,
    "granite": _granite,
    "brick": _brick,
    "concrete": _concrete,
    "ceramic": _ceramic,
    "terracotta": _terracotta,
    "fabric": _fabric,
    "leather": _leather,
    "brushed_metal": _brushed_metal,
    "grass": _grass,
    "checker": _checker,
    "stripes": _stripes,
    "dots": _dots,
    "grid": _grid_tiles,
    "lava": _lava,
    "gradient_sky": _gradient_sky,
}

TEXTURE_NAMES = sorted(_GENERATORS)

# textures that ship with a matching normal map by default
_NORMAL_STRENGTH = {
    "wood": 1.4, "wood_planks": 2.2, "marble": 0.9, "granite": 1.2,
    "brick": 2.6, "concrete": 1.1, "ceramic": 0.35, "terracotta": 1.6,
    "fabric": 1.6, "leather": 1.8, "brushed_metal": 0.5, "grass": 1.4,
    "checker": 0.8, "lava": 2.0,
}


def list_textures() -> list[str]:
    """Names of all built-in procedural textures."""
    return list(TEXTURE_NAMES)


def _lookup(name: str):
    fn = _GENERATORS.get(name)
    if fn is None:
        raise KeyError(f"Unknown texture '{name}'. Available: {', '.join(TEXTURE_NAMES)}")
    return fn


def make_texture(name: str, size: int = 512, seed: int | None = None) -> Image.Image:
    """Generate a tileable-ish RGB texture image by name."""
    fn = _lookup(name)
    img, _ = fn(size, seed if seed is not None else 0)
    return img


def make_height(name: str, size: int = 512, seed: int | None = None) -> np.ndarray:
    fn = _lookup(name)
    _, height = fn(size, seed if seed is not None else 0)
    return height


def make_normal_map(name: str, size: int = 512, seed: int | None = None,
                    strength: float | None = None) -> Image.Image:
    """Normal map for a texture, derived from its height field."""
    height = make_height(name, size=size, seed=seed)
    s = strength if strength is not None else _NORMAL_STRENGTH.get(name, 1.0)
    return normal_from_height(height, strength=s)


def make_emissive(name: str, size: int = 512, seed: int | None = None) -> Image.Image:
    """Emissive (glow) map for emissive textures like 'lava'."""
    if name == "lava":
        height = make_height("lava", size=size, seed=seed)
        glow = _normalize01(height) ** 3.0
        rgb = np.stack([glow, glow * 0.35, glow * 0.03], axis=-1)
        return _to_img(rgb)
    raise KeyError(f"Texture '{name}' has no emissive map. Try 'lava'.")
