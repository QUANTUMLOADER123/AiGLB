"""Software preview renderer for .glb files (numpy + Pillow only, no GPU).

Correct perspective projection with a per-pixel z-buffer, perspective-correct
texture filtering, two-sided lambert shading, emissive maps and a faint world
grid on the ground plane for scale. Good enough to judge a model, light enough
to run anywhere.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

__all__ = ["render_preview"]

_VIEWS = {
    "hero": (21, 37), "iso": (26, 40), "front": (4, 90), "back": (4, -90),
    "side": (4, 0), "top": (85, 90),
}


def _factor01(value, default):
    if value is None:
        return np.asarray(default, dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).ravel()[:3]
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), constant_values=1.0)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr


def _camera(view: str, center: np.ndarray, span: float):
    elev, azim = _VIEWS.get(view, _VIEWS["hero"])
    e, a = math.radians(elev), math.radians(azim)
    # direction from target to camera (world space)
    d = np.array([math.cos(e) * math.cos(a), math.sin(e), math.cos(e) * math.sin(a)])
    dist = span * 1.12 + 1e-6
    eye = center + d * dist
    fwd = center - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    if np.linalg.norm(right) < 1e-6:  # looking straight down
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return eye, np.stack([right, up, fwd])  # rows: camera axes


def render_preview(path: str, out: Optional[str] = None, view: str = "hero",
                   size: int = 1100, grid: bool = True) -> str:
    import trimesh

    scene = trimesh.load(path, force="scene")

    # ---------------- gather triangles ----------------
    raw_tris = []
    total = sum(len(g.faces) for g in scene.geometry.values())
    stride = max(1, int(math.ceil(total / 250000)))

    for name, geom in scene.geometry.items():
        verts = np.asarray(geom.vertices, dtype=np.float64)
        faces = geom.faces
        fnorms = np.asarray(geom.face_normals, dtype=np.float64)
        vis = geom.visual
        uv = getattr(vis, "uv", None)
        mat = getattr(vis, "material", None)

        base_tex = None
        emis_tex = None
        factor = np.array([0.62, 0.63, 0.66])
        emissive = np.array([0.0, 0.0, 0.0])
        alpha = 1.0
        if mat is not None:
            bct = getattr(mat, "baseColorTexture", None)
            if bct is not None:
                try:
                    base_tex = np.asarray(bct.convert("RGB"), dtype=np.float64) / 255.0
                except Exception:
                    base_tex = None
            factor = _factor01(getattr(mat, "baseColorFactor", None), [0.62, 0.63, 0.66])
            bcf = getattr(mat, "baseColorFactor", None)
            if bcf is not None:
                b = np.asarray(bcf, dtype=np.float64).ravel()
                if b.size >= 4:
                    alpha = float(b[3] / 255.0) if b.max() > 1.0 else float(b[3])
            emt = getattr(mat, "emissiveTexture", None)
            if emt is not None:
                try:
                    emis_tex = np.asarray(emt.convert("RGB"), dtype=np.float64) / 255.0
                except Exception:
                    emis_tex = None
            emissive = _factor01(getattr(mat, "emissiveFactor", None), [0.0, 0.0, 0.0])

        for i in range(0, len(faces), stride):
            f = faces[i]
            raw_tris.append({
                "p": verts[f],
                "n": fnorms[i],
                "uv": uv[f] if uv is not None else None,
                "base_tex": base_tex,
                "factor": factor,
                "emis_tex": emis_tex,
                "emissive": emissive,
                "alpha": alpha,
            })

    if not raw_tris:
        raise ValueError("scene has no triangles to render")

    allv = np.vstack([t["p"] for t in raw_tris])
    center = (allv.max(axis=0) + allv.min(axis=0)) / 2.0
    span = float(np.ptp(allv, axis=0).max()) or 1.0

    H = W = int(size)
    eye, cam = _camera(view, center, span)
    f = 0.5 * H / math.tan(math.radians(30))  # ~60° vertical fov
    cx = cy = H / 2.0

    # light (world space, from upper-left-front) + cool fill from the back-right
    L = np.array([-0.42, 0.78, 0.46])
    L = L / np.linalg.norm(L)
    L2 = np.array([0.62, 0.35, -0.58])
    L2 = L2 / np.linalg.norm(L2)

    # background: subtle vertical gradient + vignette
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    g = yy / H
    canvas = np.empty((H, W, 3))
    top = np.array([0.105, 0.115, 0.145])
    bottom = np.array([0.045, 0.05, 0.065])
    canvas[:] = bottom[None, None, :] + (top - bottom)[None, None, :] * (1.0 - g[..., None])
    r2 = ((xx - W / 2) / (W * 0.72)) ** 2 + ((yy - H / 2) / (H * 0.72)) ** 2
    canvas *= (1.0 - 0.38 * np.clip(r2, 0, 1))[..., None]

    zbuf = np.full((H, W), -np.inf)  # stores 1/z (larger = closer)

    def to_camera(p):
        return (p - eye) @ cam.T  # camera looks along +z

    # ---------------- draw faint ground grid ----------------
    if grid and abs(span) > 1e-6:
        grid_extent = math.ceil(span * 0.85)
        gpx = []
        for gx in range(-grid_extent, grid_extent + 1):
            gpx.append(((gx, 0.0, -grid_extent), (gx, 0.0, grid_extent)))
            gpx.append(((-grid_extent, 0.0, gx), (grid_extent, 0.0, gx)))
        grid_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        gd = ImageDraw.Draw(grid_img)
        for a, b in gpx:
            ca, cb = to_camera(np.asarray(a, float)), to_camera(np.asarray(b, float))
            if ca[2] <= 0.01 or cb[2] <= 0.01:
                continue
            pa = (cx + f * ca[0] / ca[2], cy - f * ca[1] / ca[2])
            pb = (cx + f * cb[0] / cb[2], cy - f * cb[1] / cb[2])
            gd.line([pa, pb], fill=(140, 160, 190, 26), width=1)
        canvas[:] = np.asarray(Image.alpha_composite(
            Image.fromarray((canvas * 255).astype(np.uint8)).convert("RGBA"), grid_img
        ).convert("RGB"), dtype=np.float64) / 255.0

    # ---------------- rasterize triangles ----------------
    light_cam = cam @ L  # light direction in camera space (not strictly needed)

    for t in raw_tris:
        p_cam = (t["p"] - eye) @ cam.T
        if (p_cam[:, 2] <= 0.01).any():
            continue
        inv_z = 1.0 / p_cam[:, 2]
        sx = cx + f * p_cam[:, 0] * inv_z
        sy = cy - f * p_cam[:, 1] * inv_z
        x0 = max(int(np.floor(sx.min())) - 1, 0)
        x1 = min(int(np.ceil(sx.max())) + 1, W - 1)
        y0 = max(int(np.floor(sy.min())) - 1, 0)
        y1 = min(int(np.ceil(sy.max())) + 1, H - 1)
        if x1 <= x0 or y1 <= y0:
            continue

        # edge functions (2D)
        ax_, ay_ = sx[0], sy[0]
        bx_, by_ = sx[1], sy[1]
        cx_, cy_ = sx[2], sy[2]
        denom = (by_ - cy_) * (ax_ - cx_) + (cx_ - bx_) * (ay_ - cy_)
        if abs(denom) < 1e-9:
            continue

        gx, gy = np.meshgrid(np.arange(x0, x1 + 1, dtype=np.float64),
                             np.arange(y0, y1 + 1, dtype=np.float64))
        w0 = ((by_ - cy_) * (gx - cx_) + (cx_ - bx_) * (gy - cy_)) / denom
        w1 = ((cy_ - ay_) * (gx - cx_) + (ax_ - cx_) * (gy - cy_)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue

        pix_inv_z = w0 * inv_z[0] + w1 * inv_z[1] + w2 * inv_z[2]
        zb_slice = zbuf[y0:y1 + 1, x0:x1 + 1]
        fn = fn_slice = None
        closer = inside & (pix_inv_z > zb_slice)
        if not closer.any():
            continue

        # shade: key light (two-sided) + weaker fill, lifted ambient
        n = t["n"]
        nl = float(np.dot(n, L))
        nl2 = float(np.dot(n, L2))
        shade = np.float64(np.clip(0.51 + 0.52 * abs(nl) + 0.26 * max(nl2, 0.0), 0.0, 1.3))

        # color
        if t["base_tex"] is not None and t["uv"] is not None:
            tex = t["base_tex"]
            th, tw = tex.shape[:2]
            u = w0 * t["uv"][0, 0] + w1 * t["uv"][1, 0] + w2 * t["uv"][2, 0]
            v = w0 * t["uv"][0, 1] + w1 * t["uv"][1, 1] + w2 * t["uv"][2, 1]
            uu = (u % 1.0) * (tw - 1)
            vv = (1.0 - (v % 1.0)) * (th - 1)
            u0 = np.floor(uu).astype(np.int64)
            v0 = np.floor(vv).astype(np.int64)
            du = uu - u0
            dv = vv - v0
            u1 = (u0 + 1) % tw
            v1 = (v0 + 1) % th
            u0 = u0 % tw
            v0 = v0 % th
            c00 = tex[v0, u0]; c01 = tex[v0, u1]; c10 = tex[v1, u0]; c11 = tex[v1, u1]
            col = (c00 * (1 - du[..., None]) * (1 - dv[..., None]) +
                   c01 * du[..., None] * (1 - dv[..., None]) +
                   c10 * (1 - du[..., None]) * dv[..., None] +
                   c11 * du[..., None] * dv[..., None])
        else:
            col = np.broadcast_to(t["factor"], (gy.shape[0], gy.shape[1], 3)).copy()

        col = col * shade
        if t["emis_tex"] is not None:
            tex = t["emis_tex"]
            th, tw = tex.shape[:2]
            if t["uv"] is not None:
                u = w0 * t["uv"][0, 0] + w1 * t["uv"][1, 0] + w2 * t["uv"][2, 0]
                v = w0 * t["uv"][0, 1] + w1 * t["uv"][1, 1] + w2 * t["uv"][2, 1]
                uu = (u % 1.0) * (tw - 1)
                vv = (1.0 - (v % 1.0)) * (th - 1)
                ui = np.clip(uu.astype(np.int64), 0, tw - 1)
                vi = np.clip(vv.astype(np.int64), 0, th - 1)
                col = col + tex[vi, ui] * t["emissive"][None, None, :] * 1.6
        elif t["emissive"].max() > 0:
            col = col + t["emissive"][None, None, :] * 0.85

        sl = canvas[y0:y1 + 1, x0:x1 + 1]
        if t["alpha"] < 0.99:
            a = t["alpha"]
            sl[closer] = sl[closer] * (1 - a) + col[closer] * a
        else:
            sl[closer] = col[closer]
        zb_slice[closer] = pix_inv_z[closer]

    canvas = np.clip(canvas, 0, 1) ** (1.0 / 1.42)  # lift midtones
    if out is None:
        out = path.rsplit(".", 1)[0] + "_preview.png"
    Image.fromarray((canvas * 255).astype(np.uint8)).save(out)
    return out
