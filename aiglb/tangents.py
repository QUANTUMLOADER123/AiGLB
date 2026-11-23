"""Inject glTF TANGENT attributes (vec4, MikkTSpace-compatible layout) into a .glb.

trimesh does not export tangents, but glTF normal maps need them for portable
rendering. This post-processes a saved .glb in place: for every primitive whose
material uses a normal map, tangents are computed from positions+UVs
(Lengyel), orthogonalized against the normals, and appended to the binary
chunk. Everything else stays byte-identical.
"""
from __future__ import annotations

import numpy as np
from pygltflib import GLTF2, Accessor, Attributes, BufferView

__all__ = ["inject_tangents"]

_COMPONENT = {
    5120: np.int8, 5121: np.uint8, 5122: np.int16,
    5123: np.uint16, 5125: np.uint32, 5126: np.float32,
}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


def _read_accessor(g, blob: bytes, index: int) -> np.ndarray:
    acc = g.accessors[index]
    dtype = _COMPONENT[acc.componentType]
    ncomp = _NCOMP[acc.type]
    view = g.bufferViews[acc.bufferView]
    item = np.dtype(dtype).itemsize
    base = (view.byteOffset or 0) + (acc.byteOffset or 0)
    stride = view.byteStride or (item * ncomp)
    if stride == item * ncomp:
        arr = np.frombuffer(blob, dtype=dtype, count=acc.count * ncomp, offset=base)
        return arr.reshape(acc.count, ncomp)
    mv = memoryview(blob)[base:base + stride * acc.count]
    return np.ndarray((acc.count, ncomp), dtype=dtype, buffer=mv, strides=(stride, item))


def _tangents_for(pos: np.ndarray, uv: np.ndarray, nrm: np.ndarray,
                  faces: np.ndarray) -> np.ndarray:
    v_count = len(pos)
    T = np.zeros((v_count, 3), dtype=np.float64)
    B = np.zeros((v_count, 3), dtype=np.float64)

    p = pos[faces]                       # (F, 3, 3)
    t = uv[faces]                        # (F, 3, 2)
    e1 = p[:, 1] - p[:, 0]
    e2 = p[:, 2] - p[:, 0]
    duv1 = t[:, 1] - t[:, 0]
    duv2 = t[:, 2] - t[:, 0]
    det = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
    valid = np.abs(det) > 1e-12
    r = np.where(valid, 1.0 / np.where(valid, det, 1.0), 0.0)
    sdir = (e1 * duv2[:, 1][:, None] - e2 * duv1[:, 1][:, None]) * r[:, None]
    tdir = (e2 * duv1[:, 0][:, None] - e1 * duv2[:, 0][:, None]) * r[:, None]
    np.add.at(T, faces, sdir[:, None, :])
    np.add.at(B, faces, tdir[:, None, :])

    N = nrm.astype(np.float64)
    T = T - N * (T * N).sum(axis=1, keepdims=True)
    t_len = np.linalg.norm(T, axis=1)
    # degenerate vertices (all-UV-singular fans): pick any stable perpendicular
    missing = t_len < 1e-10
    if missing.any():
        fallback = np.zeros((missing.sum(), 3))
        fallback[:, 0] = 1.0
        bad_n = N[missing]
        alt = np.tile(np.array([0.0, 1.0, 0.0]), (missing.sum(), 1))
        use_z = np.abs(bad_n[:, 1]) > 0.9
        alt[use_z] = np.array([1.0, 0.0, 0.0])
        fb = np.cross(bad_n, alt)
        fb_len = np.linalg.norm(fb, axis=1, keepdims=True)
        fb_len[fb_len == 0] = 1.0
        T[missing] = fb / fb_len
        t_len = np.linalg.norm(T, axis=1)
    T = T / np.maximum(t_len[:, None], 1e-12)

    w = np.sign(np.einsum("ij,ij->i", np.cross(N, T), B))
    w[w == 0] = 1.0
    return np.concatenate([T, w[:, None]], axis=1).astype(np.float32)


def inject_tangents(path: str) -> int:
    """Add TANGENT accessors to a .glb in place. Returns #primitives patched."""
    g = GLTF2().load(path)
    blob = bytearray(g.binary_blob() or b"")

    # which materials use a normal map?
    normal_mats = set()
    for i, m in enumerate(g.materials or []):
        if getattr(m, "normalTexture", None) is not None:
            normal_mats.add(i)
    if not normal_mats:
        return 0

    patches = []  # (primitive, tangents float32)
    for mesh in g.meshes or []:
        for prim in mesh.primitives:
            if prim.material is None or prim.material not in normal_mats:
                continue
            if getattr(prim.attributes, "TANGENT", None) is not None:
                continue
            if getattr(prim.attributes, "TEXCOORD_0", None) is None:
                continue
            pos = _read_accessor(g, blob, prim.attributes.POSITION).astype(np.float64)
            uv = _read_accessor(g, blob, prim.attributes.TEXCOORD_0).astype(np.float64)
            if getattr(prim.attributes, "NORMAL", None) is not None:
                nrm = _read_accessor(g, blob, prim.attributes.NORMAL).astype(np.float64)
            else:  # derive flat normals as a safety net
                nrm = np.zeros_like(pos)
                nrm[:, 1] = 1.0
            if prim.indices is not None:
                faces = _read_accessor(g, blob, prim.indices).reshape(-1, 3).astype(np.int64)
            else:
                faces = np.arange(len(pos), dtype=np.int64).reshape(-1, 3)
            patches.append((prim, _tangents_for(pos, uv, nrm, faces)))

    if not patches:
        return 0

    # pad to 4-byte alignment, then append each tangent array
    if len(blob) % 4:
        blob += b"\0" * (4 - len(blob) % 4)
    for prim, tangents in patches:
        start = len(blob)
        blob += tangents.tobytes()
        view = BufferView(buffer=0, byteOffset=start, byteLength=tangents.nbytes,
                          target=ARRAY_BUFFER)
        g.bufferViews.append(view)
        acc = Accessor(bufferView=len(g.bufferViews) - 1, componentType=5126,
                       count=len(tangents), type="VEC4",
                       min=tangents.min(axis=0).tolist(), max=tangents.max(axis=0).tolist())
        g.accessors.append(acc)
        prim.attributes.TANGENT = len(g.accessors) - 1

    g.buffers[0].byteLength = len(blob)
    g.set_binary_blob(bytes(blob))
    g.save_binary(path)
    return len(patches)
