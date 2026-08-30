#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# GKDT PyTorch checkpoint -> GGUF converter for the cpp_ggml C++ runtime.
#
# Mirrors the integration layout of ultralytics-ggml / free-splatter-ggml:
#   models/pytorch/*.best  (input, untouched)
#   models/gguf/*.gguf     (generated runtime models)
#
# The converter folds every Python-side detail into the GGUF file so that the
# C++ runtime needs no PyTorch, no dinov3 repo and no BPE vocab on disk:
#   - qkv.bias_mask is folded into qkv.bias (LinearKMaskedBias)
#   - dinov3 RoPE periods buffer is exported as a tensor
#   - the full SimpleTokenizer BPE vocab + merges are embedded as KV arrays
#   - architectural hyper-parameters are exported as `gkd.*` KVs
#
# Usage:
#   python3 scripts/convert_gkd_to_gguf.py --checkpoint models/pytorch/gkd_fullset.best \
#       [--dtype f32|f16|q8_0] [--output models/gguf/gkdt-l-f32.gguf]
# ------------------------------------------------------------------------------
import argparse
import gzip
import html
import os
import re
import struct
import sys
from functools import lru_cache

import numpy as np

try:
    import torch
except ImportError:
    sys.exit("PyTorch is required for conversion (inference never needs it).")

# ------------------------------------------------------------------------------
# GGUF v3 constants (must match ggml/src/gguf.h)
# ------------------------------------------------------------------------------
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# gguf value types
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32 = 0, 1, 2, 3, 4, 5
T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 6, 7, 8, 9, 10, 11, 12

# ggml tensor types
GGML_F32, GGML_F16, GGML_Q4_0, GGML_Q8_0, GGML_Q4_K = 0, 1, 2, 8, 12


# ------------------------------------------------------------------------------
# Minimal GGUF v3 writer
# ------------------------------------------------------------------------------
class GGUFWriter:
    def __init__(self, path, alignment=GGUF_DEFAULT_ALIGNMENT):
        self.path = path
        self.alignment = alignment
        self.kvs = []        # (key, type, value)
        self.tensors = []    # (name, ne[list], ggml_type, raw_bytes)

    # ---- scalar helpers ------------------------------------------------------
    @staticmethod
    def _pack_str(s):
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def add_kv(self, key, vtype, value):
        self.kvs.append((key, vtype, value))

    def add_u32(self, key, v):
        self.add_kv(key, T_UINT32, int(v))

    def add_i32(self, key, v):
        self.add_kv(key, T_INT32, int(v))

    def add_f32(self, key, v):
        self.add_kv(key, T_FLOAT32, float(v))

    def add_bool(self, key, v):
        self.add_kv(key, T_BOOL, bool(v))

    def add_str(self, key, v):
        self.add_kv(key, T_STRING, str(v))

    def add_arr_str(self, key, values):
        self.add_kv(key, T_ARRAY, (T_STRING, [str(x) for x in values]))

    def add_arr_f32(self, key, values):
        self.add_kv(key, T_ARRAY, (T_FLOAT32, [float(x) for x in values]))

    def add_tensor(self, name, ne, ggml_type, data):
        assert isinstance(data, (bytes, bytearray, np.ndarray))
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        self.tensors.append((name, [int(x) for x in ne], ggml_type, data))

    # ---- serialization -------------------------------------------------------
    def write(self):
        header = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(self.tensors), len(self.kvs))
        body = bytearray()
        for key, vtype, value in self.kvs:
            body += self._pack_str(key)
            body += struct.pack("<I", vtype)
            body += self._pack_value(vtype, value)

        infos = bytearray()
        offset = 0
        for name, ne, gtype, data in self.tensors:
            infos += self._pack_str(name)
            infos += struct.pack("<I", len(ne))
            infos += struct.pack("<" + "Q" * len(ne), *ne)
            infos += struct.pack("<I", gtype)
            infos += struct.pack("<Q", offset)
            offset += len(data)
            offset = (offset + self.alignment - 1) // self.alignment * self.alignment

        cur = len(header) + len(body) + len(infos)
        pad = (cur + self.alignment - 1) // self.alignment * self.alignment - cur

        with open(self.path, "wb") as f:
            f.write(header)
            f.write(body)
            f.write(infos)
            f.write(b"\x00" * pad)
            for name, ne, gtype, data in self.tensors:
                f.write(data)
                aligned = (len(data) + self.alignment - 1) // self.alignment * self.alignment
                f.write(b"\x00" * (aligned - len(data)))
        size_mb = os.path.getsize(self.path) / 1024 / 1024
        print(f"==> wrote {self.path}  ({len(self.tensors)} tensors, {size_mb:.1f} MiB)")

    def _pack_value(self, vtype, value):
        if vtype == T_STRING:
            return self._pack_str(value)
        if vtype == T_BOOL:
            return struct.pack("<B", 1 if value else 0)
        if vtype == T_UINT8:
            return struct.pack("<B", value)
        if vtype == T_INT8:
            return struct.pack("<b", value)
        if vtype == T_UINT16:
            return struct.pack("<H", value)
        if vtype == T_INT16:
            return struct.pack("<h", value)
        if vtype == T_UINT32:
            return struct.pack("<I", value)
        if vtype == T_INT32:
            return struct.pack("<i", value)
        if vtype == T_UINT64:
            return struct.pack("<Q", value)
        if vtype == T_INT64:
            return struct.pack("<q", value)
        if vtype == T_FLOAT32:
            return struct.pack("<f", value)
        if vtype == T_FLOAT64:
            return struct.pack("<d", value)
        if vtype == T_ARRAY:
            etype, elems = value
            out = struct.pack("<IQ", etype, len(elems))
            for e in elems:
                out += self._pack_value(etype, e)
            return out
        raise ValueError(f"unsupported kv type {vtype}")


# ------------------------------------------------------------------------------
# quantization
# ------------------------------------------------------------------------------
def quantize_q8_0(x: np.ndarray) -> bytes:
    """Block-wise Q8_0: 32 elements per block, f16 scale + int8 payload."""
    x = x.astype(np.float32).reshape(-1)
    n = x.size
    assert n % 32 == 0, "Q8_0 requires the element count to be a multiple of 32"
    blocks = x.reshape(-1, 32)
    amax = np.max(np.abs(blocks), axis=1)
    d = (amax / 127.0).astype(np.float32)
    inv = np.where(d > 0, 1.0 / d, 0.0).astype(np.float32)
    q = np.clip(np.rint(blocks * inv[:, None]), -127, 127).astype(np.int8)
    out = np.empty((blocks.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8).reshape(-1, 32)
    return out.tobytes()


def quantize_q4_0(x: np.ndarray) -> bytes:
    """Block-wise Q4_0 (port of quantize_row_q4_0_ref, ggml-quants.c:113).

    32 elements per block: f16 scale d (signed, d = signed_max / -8) + 16 bytes
    of nibbles, value = (q - 8) * d."""
    x = x.astype(np.float32).reshape(-1)
    assert x.size % 32 == 0, "Q4_0 requires the element count to be a multiple of 32"
    blocks = x.reshape(-1, 32)
    amax_idx = np.argmax(np.abs(blocks), axis=1)          # C picks the first |v| max
    maxv = blocks[np.arange(len(blocks)), amax_idx]        # signed max
    d = (maxv / np.float32(-8.0)).astype(np.float32)
    inv = np.where(d != 0, np.float32(1.0) / np.where(d != 0, d, np.float32(1.0)),
                   np.float32(0.0)).astype(np.float32)
    # (int8_t)(x*id + 8.5f): x*id is within [-8, 8] so the value is >= 0.5
    q = (blocks * inv[:, None] + np.float32(8.5)).astype(np.int8)
    q = np.minimum(q, 15).astype(np.uint8)                 # MIN(15, ...)
    out = np.empty((len(blocks), 18), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q[:, :16] | (q[:, 16:] << 4)
    return out.tobytes()


def _make_qkx2_quants(x, weights, nmax=15, rmin=-1.0, rdelta=0.1, nstep=20):
    """Vectorized port of make_qkx2_quants (ggml-quants.c:799).

    x, weights: (..., 32) float32. Returns (L uint8, scale float32, the_min
    float32) elementwise over the leading dims. All arithmetic in float32 to
    mirror the C reference."""
    mn = x.min(axis=-1)
    mx = x.max(axis=-1)
    sum_w = weights.sum(axis=-1)
    sum_x = (weights * x).sum(axis=-1)
    mn = np.minimum(mn, np.float32(0.0))                   # if (min > 0) min = 0
    eq = mx == mn                                          # (max == min) early-out
    denom = np.where(eq, np.float32(1.0), mx - mn)
    iscale = np.where(eq, np.float32(0.0), np.float32(nmax) / denom)
    scale = np.where(eq, np.float32(0.0), np.float32(1.0) / np.where(eq, np.float32(1.0), iscale))
    L = np.clip(np.rint(iscale[..., None] * (x - mn[..., None])), 0, nmax).astype(np.int32)
    diff = scale[..., None] * L + mn[..., None] - x
    best = (weights * diff * diff).sum(axis=-1)            # use_mad = false
    L = np.where(eq[..., None], 0, L)
    for is_ in range(nstep + 1):
        isc = np.where(eq, np.float32(0.0),
                       (np.float32(rmin) + np.float32(rdelta) * np.float32(is_)
                        + np.float32(nmax)) / denom)
        Laux = np.clip(np.rint(isc[..., None] * (x - mn[..., None])), 0, nmax).astype(np.int32)
        sl = (weights * Laux).sum(axis=-1)
        sl2 = (weights * Laux * Laux).sum(axis=-1)
        sxl = (weights * Laux * x).sum(axis=-1)
        D = sum_w * sl2 - sl * sl
        ok = (D > 0) & ~eq
        with np.errstate(divide="ignore", invalid="ignore"):
            t_scale = (sum_w * sxl - sum_x * sl) / D
            t_min = (sl2 * sum_x - sl * sxl) / D
        fix = t_min > 0
        t_min = np.where(fix, np.float32(0.0), t_min)
        t_scale = np.where(fix, sxl / np.where(sl2 == 0, np.float32(1.0), sl2), t_scale)
        cur = (weights * (t_scale[..., None] * Laux + t_min[..., None] - x) ** 2).sum(axis=-1)
        better = ok & (cur < best)
        best = np.where(better, cur, best)
        scale = np.where(better, t_scale, scale).astype(np.float32)
        mn = np.where(better, t_min, mn).astype(np.float32)   # returned as -min
    return L.astype(np.uint8), scale, -mn


def quantize_q4_K(x: np.ndarray) -> bytes:
    """Block-wise Q4_K (port of quantize_row_q4_K_ref, ggml-quants.c:1457).

    256 elements per super-block: f16 d + f16 dmin + 12 bytes of packed 6-bit
    per-32 scales/mins + 128 bytes of nibbles. x = d*(q - 8)*sc - dmin*m."""
    x = x.astype(np.float32).reshape(-1)
    assert x.size % 256 == 0, "Q4_K requires the element count to be a multiple of 256"
    nb = x.size // 256
    xb = x.reshape(nb, 8, 32)
    sum_x2 = (xb * xb).sum(axis=-1)
    weights = np.sqrt(sum_x2 / np.float32(32.0))[..., None] + np.abs(xb)
    L, scales, mins = _make_qkx2_quants(xb, weights, 15, -1.0, 0.1, 20)

    max_scale = scales.max(axis=-1)                        # (nb,)
    max_min = mins.max(axis=-1)
    inv_scale = np.where(max_scale > 0, np.float32(63.0) / np.where(max_scale > 0, max_scale, np.float32(1)),
                         np.float32(0.0)).astype(np.float32)
    inv_min = np.where(max_min > 0, np.float32(63.0) / np.where(max_min > 0, max_min, np.float32(1)),
                       np.float32(0.0)).astype(np.float32)
    ls = np.minimum(np.rint(inv_scale[:, None] * scales), 63).astype(np.uint8)
    lm = np.minimum(np.rint(inv_min[:, None] * mins), 63).astype(np.uint8)

    # 6-bit scales/mins packed into 12 bytes (exact C index mapping)
    sc = np.zeros((nb, 12), dtype=np.uint8)
    for j in range(4):
        sc[:, j] = ls[:, j]
        sc[:, j + 4] = lm[:, j]
    for j in range(4, 8):
        sc[:, j + 4] = (ls[:, j] & 0xF) | ((lm[:, j] & 0xF) << 4)
        sc[:, j - 4] |= (ls[:, j] >> 4) << 6
        sc[:, j] |= (lm[:, j] >> 4) << 6

    # d / dmin round-trip through f16 (the runtime dequantizes with these)
    d16 = (max_scale / np.float32(63.0)).astype(np.float32).astype(np.float16)
    dmin16 = (max_min / np.float32(63.0)).astype(np.float32).astype(np.float16)
    df = d16.astype(np.float32)[:, None]
    dmf = dmin16.astype(np.float32)[:, None]

    # unpack the effective per-sub-block scale (get_scale_min_k4)
    scj = np.empty((nb, 8), dtype=np.int32)
    mmj = np.empty((nb, 8), dtype=np.int32)
    for j in range(4):
        scj[:, j] = sc[:, j] & 63
        mmj[:, j] = sc[:, j + 4] & 63
    for j in range(4, 8):
        scj[:, j] = (sc[:, j + 4] & 0xF).astype(np.int32) | ((sc[:, j - 4].astype(np.int32) >> 6) << 4)
        mmj[:, j] = (sc[:, j + 4].astype(np.int32) >> 4) | ((sc[:, j].astype(np.int32) >> 6) << 4)

    dj = df * scj                                          # (nb, 8) effective scales
    dmj = dmf * mmj
    ok = dj > 0
    # blocks with d == 0 keep the search-stage L (exact C `if (!d) continue;`)
    L2 = np.where(ok[..., None],
                  np.clip(np.rint((xb + dmj[..., None]) / np.where(ok, dj, np.float32(1.0))[..., None]),
                          0, 15), L).astype(np.uint8)

    q = np.empty((nb, 128), dtype=np.uint8)
    Lflat = L2.reshape(nb, 256)
    for c in range(4):
        base = c * 64
        q[:, c * 32:(c + 1) * 32] = Lflat[:, base:base + 32] | (Lflat[:, base + 32:base + 64] << 4)

    out = np.empty((nb, 144), dtype=np.uint8)
    out[:, 0:2] = d16.view(np.uint8).reshape(-1, 2)
    out[:, 2:4] = dmin16.view(np.uint8).reshape(-1, 2)
    out[:, 4:16] = sc
    out[:, 16:] = q
    return out.tobytes()


def to_dtype(x: np.ndarray, dtype: str) -> tuple:
    """Return (ggml_type, raw bytes) for the requested storage dtype.

    Constraints:
    - quantized types need the fastest dim (ne0) to be a multiple of the block
      size (32 for q4_0/q8_0, 256 for q4_K); anything else stays F32.
    - only 2-D weight matrices are quantized; vectors (biases, norms, scales)
      stay F32 so element-wise ops never need to dequantize.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    if dtype == "f32":
        return GGML_F32, x.tobytes()
    if dtype == "f16":
        # F16 only for 2-D weight matrices (mul_mat handles them natively);
        # vectors stay F32 because the CUDA binbcast kernels reject mixed
        # F16 broadcast sources.
        if x.ndim == 2:
            return GGML_F16, x.astype(np.float16).tobytes()
        return GGML_F32, x.tobytes()
    if dtype == "q8_0":
        if x.ndim == 2 and x.shape[-1] % 32 == 0:
            return GGML_Q8_0, quantize_q8_0(x)
        return GGML_F32, x.tobytes()
    if dtype == "q4_0":
        if x.ndim == 2 and x.shape[-1] % 32 == 0:
            return GGML_Q4_0, quantize_q4_0(x)
        return GGML_F32, x.tobytes()
    if dtype == "q4_K":
        if x.ndim == 2 and x.shape[-1] % 256 == 0:
            return GGML_Q4_K, quantize_q4_K(x)
        return GGML_F32, x.tobytes()
    raise ValueError(f"unknown dtype {dtype}")


# ------------------------------------------------------------------------------
# weight helpers.
#
# ggml memory layout: element (i0, i1, ..., in) of ne {d0, d1, ..., dn} lives at
# i0 + i1*d0 + ... — i.e. EXACTLY the numpy C-order of an array whose shape is
# the REVERSED (dn, ..., d1, d0). Therefore:
#   torch [out, in] Linear weight (C-order) == ggml ne {in, out}  (no transpose!)
#   torch [OC, IC, KH, KW] conv weight      == ggml ne {KW, KH, IC, OC}
# and exporting is a plain contiguous copy of the raw tensor.
# ------------------------------------------------------------------------------
def lin(t):
    """nn.Linear weight [out,in] -> ggml ne {in,out} float32 (raw C-order)."""
    return np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())


def vec(t):
    return np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy().reshape(-1))


def conv2d_w(t):
    """nn.Conv2d weight [OC,IC,KH,KW] -> ggml ne {KW,KH,IC,OC} (raw C-order)."""
    return np.ascontiguousarray(t.detach().to(torch.float32).cpu().numpy())


# ------------------------------------------------------------------------------
# SimpleTokenizer vocab export (same BPE as CLIP / dinov3 dinotxt)
# ------------------------------------------------------------------------------
@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
         list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def load_bpe_vocab(bpe_path):
    with gzip.open(bpe_path, "rb") as f:
        data = f.read().decode("utf-8")
    merges = data.split("\n")[1:49152 - 256 - 2 + 1]
    merges = [tuple(m.split()) for m in merges if m.strip()]
    vocab = list(bytes_to_unicode().values())
    vocab = vocab + [v + "</w>" for v in vocab]
    for m in merges:
        vocab.append("".join(m))
    vocab.extend(["<|startoftext|>", "<|endoftext|>"])
    return vocab, [f"{a} {b}" for a, b in merges]


# ------------------------------------------------------------------------------
# main conversion
# ------------------------------------------------------------------------------
def convert(checkpoint, bpe_path, dtype, output, app_cfg):
    print(f"==> loading checkpoint {checkpoint} (this can take a minute)")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model"]

    w = GGUFWriter(output)
    w.add_str("general.architecture", "gkdt")
    w.add_str("general.name", "GKDT-L (for-app release)" if "fullset" in os.path.basename(checkpoint) else "GKDT")
    w.add_str("general.license", "academic")
    w.add_u32("general.alignment", GGUF_DEFAULT_ALIGNMENT)

    w.add_str("gkd.task", "general_keypoint_detection")
    w.add_str("gkd.trunk", "DINOv3")

    # ---- visual encoder (DINOv3 ViT) ----------------------------------------
    vit = {k[len("dinov3_visual_encoder."):]: v for k, v in sd.items() if k.startswith("dinov3_visual_encoder.")}
    D = vit["patch_embed.proj.weight"].shape[0]          # embed dim
    patch = vit["patch_embed.proj.weight"].shape[-1]     # patch size (16)
    n_blocks = 0
    while f"blocks.{n_blocks}.norm1.weight" in vit:
        n_blocks += 1
    heads = {384: 6, 768: 12, 1024: 16, 1280: 20, 4096: 32}[D]
    n_storage = vit["storage_tokens"].shape[1] if "storage_tokens" in vit else 0
    img_size = app_cfg.get("SQUARE_IMAGE_LENGTH", 384)
    assert img_size % patch == 0
    feat_w = img_size // patch

    w.add_str("gkd.vis.arch", "dinov3_vitl16" if D == 1024 else f"dinov3_vit_D{D}")
    w.add_u32("gkd.vis.embed_dim", D)
    w.add_u32("gkd.vis.depth", n_blocks)
    w.add_u32("gkd.vis.num_heads", heads)
    w.add_u32("gkd.vis.patch_size", patch)
    w.add_u32("gkd.vis.n_storage_tokens", n_storage)
    w.add_f32("gkd.vis.norm_eps", 1e-5)
    w.add_f32("gkd.vis.rope_base", 100.0)
    w.add_bool("gkd.vis.has_layerscale", "blocks.0.ls1.gamma" in vit)
    w.add_bool("gkd.vis.mask_k_bias", "blocks.0.attn.qkv.bias_mask" in vit)
    w.add_str("gkd.vis.ffn_act", "gelu")
    w.add_u32("gkd.vis.img_size", img_size)
    w.add_u32("gkd.vis.feat_width", feat_w)
    head_dim = D // heads
    assert head_dim % 4 == 0, "rope requires head_dim % (4*1) == 0"

    print(f"==> visual encoder: D={D} blocks={n_blocks} heads={heads} storage={n_storage}")

    pw32 = conv2d_w(vit["patch_embed.proj.weight"])
    # numpy C-order (1024,3,16,16) == ggml ne {16,16,3,1024} (reversed shape)
    w.add_tensor("vis.patch_w", list(pw32.shape)[::-1], *to_dtype(pw32, dtype))
    w.add_tensor("vis.patch_b", [D], *to_dtype(vec(vit["patch_embed.proj.bias"]), dtype))
    # concat type-match: cls/storage join the F32 token stream, keep them F32
    w.add_tensor("vis.cls_token", [D], GGML_F32, vec(vit["cls_token"]).tobytes())
    if n_storage > 0:
        w.add_tensor("vis.storage_tokens", [D, n_storage], GGML_F32, vec(vit["storage_tokens"][0]).tobytes())
    w.add_tensor("vis.rope_periods", [D // heads // 4], *to_dtype(vec(vit["rope_embed.periods"]), "f32"))
    w.add_tensor("vis.norm_w", [D], *to_dtype(vec(vit["norm.weight"]), dtype))
    w.add_tensor("vis.norm_b", [D], *to_dtype(vec(vit["norm.bias"]), dtype))

    for i in range(n_blocks):
        p = f"blocks.{i}."
        w.add_tensor(f"vis.b{i}.norm1_w", [D], *to_dtype(vec(vit[p + "norm1.weight"]), dtype))
        w.add_tensor(f"vis.b{i}.norm1_b", [D], *to_dtype(vec(vit[p + "norm1.bias"]), dtype))
        qkv_w = np.ascontiguousarray(vit[p + "attn.qkv.weight"].numpy().astype(np.float32))
        qkv_b = vec(vit[p + "attn.qkv.bias"])
        if p + "attn.qkv.bias_mask" in vit:
            qkv_b = qkv_b * vec(vit[p + "attn.qkv.bias_mask"])  # fold LinearKMaskedBias
        w.add_tensor(f"vis.b{i}.qkv_w", [D, 3 * D], *to_dtype(qkv_w, dtype))
        w.add_tensor(f"vis.b{i}.qkv_b", [3 * D], *to_dtype(qkv_b, dtype))
        w.add_tensor(f"vis.b{i}.proj_w", [D, D], *to_dtype(lin(vit[p + "attn.proj.weight"]), dtype))
        w.add_tensor(f"vis.b{i}.proj_b", [D], *to_dtype(vec(vit[p + "attn.proj.bias"]), dtype))
        w.add_tensor(f"vis.b{i}.ls1", [D], *to_dtype(vec(vit[p + "ls1.gamma"]), dtype))
        w.add_tensor(f"vis.b{i}.ls2", [D], *to_dtype(vec(vit[p + "ls2.gamma"]), dtype))
        w.add_tensor(f"vis.b{i}.norm2_w", [D], *to_dtype(vec(vit[p + "norm2.weight"]), dtype))
        w.add_tensor(f"vis.b{i}.norm2_b", [D], *to_dtype(vec(vit[p + "norm2.bias"]), dtype))
        w.add_tensor(f"vis.b{i}.fc1_w", [D, 4 * D], *to_dtype(lin(vit[p + "mlp.fc1.weight"]), dtype))
        w.add_tensor(f"vis.b{i}.fc1_b", [4 * D], *to_dtype(vec(vit[p + "mlp.fc1.bias"]), dtype))
        w.add_tensor(f"vis.b{i}.fc2_w", [4 * D, D], *to_dtype(lin(vit[p + "mlp.fc2.weight"]), dtype))
        w.add_tensor(f"vis.b{i}.fc2_b", [D], *to_dtype(vec(vit[p + "mlp.fc2.bias"]), dtype))

    # ---- text encoder (dinotxt TextTransformer + TextHead) -------------------
    txt = {k[len("dinov3_text_encoder."):]: v for k, v in sd.items() if k.startswith("dinov3_text_encoder.")}
    TD = txt["backbone.token_embedding.weight"].shape[1]
    T_layers = 0
    while f"backbone.blocks.{T_layers}.attention_norm.weight" in txt:
        T_layers += 1
    vocab_size, ctx_len = txt["backbone.token_embedding.weight"].shape[0], txt["backbone.positional_embedding"].shape[0]
    T_heads = {1280: 20, 768: 12, 512: 8}[TD]
    ffn_h = txt["backbone.blocks.0.feed_forward.fc1.weight"].shape[0]
    proj_out = txt["head.linear_projection.weight"].shape[0]

    print(f"==> text encoder: D={TD} blocks={T_layers} heads={T_heads} ffn={ffn_h} proj={proj_out}")

    w.add_str("gkd.txt.arch", "dinotxt")
    w.add_u32("gkd.txt.dim", TD)
    w.add_u32("gkd.txt.layers", T_layers)
    w.add_u32("gkd.txt.num_heads", T_heads)
    w.add_u32("gkd.txt.context_length", ctx_len)
    w.add_u32("gkd.txt.vocab_size", vocab_size)
    w.add_f32("gkd.txt.norm_eps", 1e-5)
    w.add_bool("gkd.txt.causal", True)
    w.add_str("gkd.txt.act", "gelu")
    w.add_u32("gkd.txt.proj_dim", proj_out)
    w.add_bool("gkd.txt.head_linear", True)

    w.add_tensor("txt.tok_emb", [TD, vocab_size], *to_dtype(lin(txt["backbone.token_embedding.weight"]), dtype))
    w.add_tensor("txt.pos_emb", [TD, ctx_len], *to_dtype(vec(txt["backbone.positional_embedding"]), dtype))
    for i in range(T_layers):
        p = f"backbone.blocks.{i}."
        w.add_tensor(f"txt.b{i}.attn_norm_w", [TD], *to_dtype(vec(txt[p + "attention_norm.weight"]), dtype))
        w.add_tensor(f"txt.b{i}.attn_norm_b", [TD], *to_dtype(vec(txt[p + "attention_norm.bias"]), dtype))
        w.add_tensor(f"txt.b{i}.qkv_w", [TD, 3 * TD], *to_dtype(lin(txt[p + "attention.qkv.weight"]), dtype))
        w.add_tensor(f"txt.b{i}.proj_w", [TD, TD], *to_dtype(lin(txt[p + "attention.proj.weight"]), dtype))
        w.add_tensor(f"txt.b{i}.proj_b", [TD], *to_dtype(vec(txt[p + "attention.proj.bias"]), dtype))
        w.add_tensor(f"txt.b{i}.ffn_norm_w", [TD], *to_dtype(vec(txt[p + "ffn_norm.weight"]), dtype))
        w.add_tensor(f"txt.b{i}.ffn_norm_b", [TD], *to_dtype(vec(txt[p + "ffn_norm.bias"]), dtype))
        w.add_tensor(f"txt.b{i}.fc1_w", [TD, ffn_h], *to_dtype(lin(txt[p + "feed_forward.fc1.weight"]), dtype))
        w.add_tensor(f"txt.b{i}.fc1_b", [ffn_h], *to_dtype(vec(txt[p + "feed_forward.fc1.bias"]), dtype))
        w.add_tensor(f"txt.b{i}.fc2_w", [ffn_h, TD], *to_dtype(lin(txt[p + "feed_forward.fc2.weight"]), dtype))
        w.add_tensor(f"txt.b{i}.fc2_b", [TD], *to_dtype(vec(txt[p + "feed_forward.fc2.bias"]), dtype))
    w.add_tensor("txt.ln_final_w", [TD], *to_dtype(vec(txt["backbone.ln_final.weight"]), dtype))
    w.add_tensor("txt.ln_final_b", [TD], *to_dtype(vec(txt["backbone.ln_final.bias"]), dtype))
    w.add_tensor("txt.head_proj_w", [TD, proj_out], *to_dtype(lin(txt["head.linear_projection.weight"]), dtype))

    # ---- text adaptation net (CLIP-style transformer) ------------------------
    anet = {k[len("text_anet."):]: v for k, v in sd.items() if k.startswith("text_anet.")}
    a_in = anet["proj_in.projector.0.weight"].shape[0]
    a_model = anet["net.resblocks.0.attn.in_proj_weight"].shape[1]
    a_heads = a_model // 64
    a_blocks = 0
    while f"net.resblocks.{a_blocks}.attn.in_proj_weight" in anet:
        a_blocks += 1
    a_out = anet["proj"].shape[1]

    print(f"==> text anet: in={a_in} model={a_model} blocks={a_blocks} heads={a_heads} out={a_out}")

    w.add_str("gkd.anet.arch", "clip_transformer")
    w.add_u32("gkd.anet.dim_in", a_in)
    w.add_u32("gkd.anet.dim_model", a_model)
    w.add_u32("gkd.anet.blocks", a_blocks)
    w.add_u32("gkd.anet.num_heads", a_heads)
    w.add_u32("gkd.anet.dim_out", a_out)
    w.add_f32("gkd.anet.norm_eps", 1e-5)
    w.add_u32("gkd.text_feature_dim_half", a_out // 2)  # DINOv3: take lower half of text features

    w.add_tensor("anet.norm_in_w", [a_in], *to_dtype(vec(anet["proj_in.projector.0.weight"]), dtype))
    w.add_tensor("anet.norm_in_b", [a_in], *to_dtype(vec(anet["proj_in.projector.0.bias"]), dtype))
    w.add_tensor("anet.proj_in_w", [a_in, a_model], *to_dtype(lin(anet["proj_in.projector.1.weight"]), dtype))
    for i in range(a_blocks):
        p = f"net.resblocks.{i}."
        w.add_tensor(f"anet.b{i}.ln1_w", [a_model], *to_dtype(vec(anet[p + "ln_1.weight"]), dtype))
        w.add_tensor(f"anet.b{i}.ln1_b", [a_model], *to_dtype(vec(anet[p + "ln_1.bias"]), dtype))
        w.add_tensor(f"anet.b{i}.in_w", [a_model, 3 * a_model], *to_dtype(lin(anet[p + "attn.in_proj_weight"]), dtype))
        w.add_tensor(f"anet.b{i}.in_b", [3 * a_model], *to_dtype(vec(anet[p + "attn.in_proj_bias"]), dtype))
        w.add_tensor(f"anet.b{i}.out_w", [a_model, a_model], *to_dtype(lin(anet[p + "attn.out_proj.weight"]), dtype))
        w.add_tensor(f"anet.b{i}.out_b", [a_model], *to_dtype(vec(anet[p + "attn.out_proj.bias"]), dtype))
        w.add_tensor(f"anet.b{i}.ln2_w", [a_model], *to_dtype(vec(anet[p + "ln_2.weight"]), dtype))
        w.add_tensor(f"anet.b{i}.ln2_b", [a_model], *to_dtype(vec(anet[p + "ln_2.bias"]), dtype))
        a_ff = anet[p + "mlp.c_fc.weight"].shape[0]
        w.add_tensor(f"anet.b{i}.fc_w", [a_model, a_ff], *to_dtype(lin(anet[p + "mlp.c_fc.weight"]), dtype))
        w.add_tensor(f"anet.b{i}.fc_b", [a_ff], *to_dtype(vec(anet[p + "mlp.c_fc.bias"]), dtype))
        w.add_tensor(f"anet.b{i}.proj_w", [a_ff, a_model], *to_dtype(lin(anet[p + "mlp.c_proj.weight"]), dtype))
        w.add_tensor(f"anet.b{i}.proj_b", [a_model], *to_dtype(vec(anet[p + "mlp.c_proj.bias"]), dtype))
    w.add_tensor("anet.ln_w", [a_model], *to_dtype(vec(anet["ln.weight"]), dtype))
    w.add_tensor("anet.ln_b", [a_model], *to_dtype(vec(anet["ln.bias"]), dtype))
    # anet.proj is used as x @ proj (not F.linear). Store proj.t() raw so that
    # ggml mul_mat(proj {a_model, a_out}, x {a_model, ...}) yields x @ proj.
    w.add_tensor("anet.proj", [a_model, a_out], *to_dtype(lin(anet["proj"].t().contiguous()), dtype))
    w.add_tensor("anet.last_norm_w", [a_out], *to_dtype(vec(anet["last_norm.weight"]), dtype))
    w.add_tensor("anet.last_norm_b", [a_out], *to_dtype(vec(anet["last_norm.bias"]), dtype))

    # ---- kernel generation transformer ---------------------------------------
    kg = {k[len("kg_transformer."):]: v for k, v in sd.items() if k.startswith("kg_transformer.")}
    K = kg["blocks.0.self_attention.in_proj_weight"].shape[1]
    K_heads = K // 64
    K_blocks = 0
    while f"blocks.{K_blocks}.self_attention.in_proj_weight" in kg:
        K_blocks += 1
    K_ff = kg["blocks.0.feed_forward.linear1.weight"].shape[0]
    use_mask_token = "mask_token" in kg

    print(f"==> kg transformer: D={K} blocks={K_blocks} heads={K_heads} d_ff={K_ff} mask_token={use_mask_token}")

    w.add_str("gkd.kg.arch", "kg_transformer")
    w.add_u32("gkd.kg.dim", K)
    w.add_u32("gkd.kg.blocks", K_blocks)
    w.add_u32("gkd.kg.num_heads", K_heads)
    w.add_u32("gkd.kg.d_ff", K_ff)
    w.add_bool("gkd.kg.use_mask_token", use_mask_token)
    w.add_bool("gkd.kg.use_sa", True)
    w.add_bool("gkd.kg.use_ca", True)
    w.add_f32("gkd.kg.norm_eps", 1e-5)

    assert K == D, "kg dim must equal the visual feature dim for the identity t2i/k2i projectors"
    if use_mask_token:
        w.add_tensor("kg.mask_token", [K], *to_dtype(vec(kg["mask_token"]), dtype))
    for i in range(K_blocks):
        p = f"blocks.{i}."
        w.add_tensor(f"kg.b{i}.norm1_w", [K], *to_dtype(vec(kg[p + "norm1.weight"]), dtype))
        w.add_tensor(f"kg.b{i}.norm1_b", [K], *to_dtype(vec(kg[p + "norm1.bias"]), dtype))
        w.add_tensor(f"kg.b{i}.sa_in_w", [K, 3 * K], *to_dtype(lin(kg[p + "self_attention.in_proj_weight"]), dtype))
        w.add_tensor(f"kg.b{i}.sa_in_b", [3 * K], *to_dtype(vec(kg[p + "self_attention.in_proj_bias"]), dtype))
        w.add_tensor(f"kg.b{i}.sa_out_w", [K, K], *to_dtype(lin(kg[p + "self_attention.out_proj.weight"]), dtype))
        w.add_tensor(f"kg.b{i}.sa_out_b", [K], *to_dtype(vec(kg[p + "self_attention.out_proj.bias"]), dtype))
        w.add_tensor(f"kg.b{i}.nca1_w", [K], *to_dtype(vec(kg[p + "norm_ca1.weight"]), dtype))
        w.add_tensor(f"kg.b{i}.nca1_b", [K], *to_dtype(vec(kg[p + "norm_ca1.bias"]), dtype))
        w.add_tensor(f"kg.b{i}.nca2_w", [K], *to_dtype(vec(kg[p + "norm_ca2.weight"]), dtype))
        w.add_tensor(f"kg.b{i}.nca2_b", [K], *to_dtype(vec(kg[p + "norm_ca2.bias"]), dtype))
        # cross-attn in_proj split into q/k/v: the engine must not mul_mat
        # through offset views of a packed weight (silently mis-addresses).
        wq = lin(kg[p + "cross_attention.in_proj_weight"][0:K])
        wk = lin(kg[p + "cross_attention.in_proj_weight"][K:2 * K])
        wv = lin(kg[p + "cross_attention.in_proj_weight"][2 * K:3 * K])
        w.add_tensor(f"kg.b{i}.ca_q_w", [K, K], *to_dtype(wq, dtype))
        w.add_tensor(f"kg.b{i}.ca_k_w", [K, K], *to_dtype(wk, dtype))
        w.add_tensor(f"kg.b{i}.ca_v_w", [K, K], *to_dtype(wv, dtype))
        w.add_tensor(f"kg.b{i}.ca_q_b", [K], *to_dtype(vec(kg[p + "cross_attention.in_proj_bias"][0:K]), dtype))
        w.add_tensor(f"kg.b{i}.ca_k_b", [K], *to_dtype(vec(kg[p + "cross_attention.in_proj_bias"][K:2 * K]), dtype))
        w.add_tensor(f"kg.b{i}.ca_v_b", [K], *to_dtype(vec(kg[p + "cross_attention.in_proj_bias"][2 * K:3 * K]), dtype))
        w.add_tensor(f"kg.b{i}.ca_out_w", [K, K], *to_dtype(lin(kg[p + "cross_attention.out_proj.weight"]), dtype))
        w.add_tensor(f"kg.b{i}.ca_out_b", [K], *to_dtype(vec(kg[p + "cross_attention.out_proj.bias"]), dtype))
        w.add_tensor(f"kg.b{i}.norm3_w", [K], *to_dtype(vec(kg[p + "norm3.weight"]), dtype))
        w.add_tensor(f"kg.b{i}.norm3_b", [K], *to_dtype(vec(kg[p + "norm3.bias"]), dtype))
        w.add_tensor(f"kg.b{i}.ff1_w", [K, K_ff], *to_dtype(lin(kg[p + "feed_forward.linear1.weight"]), dtype))
        w.add_tensor(f"kg.b{i}.ff1_b", [K_ff], *to_dtype(vec(kg[p + "feed_forward.linear1.bias"]), dtype))
        w.add_tensor(f"kg.b{i}.ff2_w", [K_ff, K], *to_dtype(lin(kg[p + "feed_forward.linear2.weight"]), dtype))
        w.add_tensor(f"kg.b{i}.ff2_b", [K], *to_dtype(vec(kg[p + "feed_forward.linear2.bias"]), dtype))
    w.add_tensor("kg.final_norm_w", [K], *to_dtype(vec(kg["final_norm.weight"]), dtype))
    w.add_tensor("kg.final_norm_b", [K], *to_dtype(vec(kg["final_norm.bias"]), dtype))

    # ---- detection head: parameter-free for the released configs -------------
    # bilinear up-scale 4, kernel expander reso 1, k2i identity (dims match),
    # kernel_norm = True. Assert the assumptions the C++ graph bakes in.
    w.add_str("gkd.det.up_type", "bilinear")
    w.add_u32("gkd.det.up_scale", 4)
    w.add_u32("gkd.det.kernel_expander_reso", 1)
    w.add_bool("gkd.det.kernel_norm", True)

    # ---- preprocessing / prompting -------------------------------------------
    w.add_arr_f32("gkd.img_mean", [0.485, 0.456, 0.406])
    w.add_arr_f32("gkd.img_std", [0.229, 0.224, 0.225])
    w.add_f32("gkd.sigma", app_cfg.get("SIGMA", 14.0))
    w.add_u32("gkd.pad_kps", app_cfg.get("PAD_KPS", 80))
    w.add_u32("gkd.heatmap_scale", 4)  # heatmaps are up_scale * feat_width

    # ---- tokenizer ------------------------------------------------------------
    vocab, merges = load_bpe_vocab(bpe_path)
    assert len(vocab) == vocab_size, f"bpe vocab {len(vocab)} != model vocab {vocab_size}"
    w.add_arr_str("gkd.tokenizer.vocab", vocab)
    w.add_arr_str("gkd.tokenizer.merges", merges)
    print(f"==> tokenizer: {len(vocab)} tokens, {len(merges)} merges")

    w.write()


def main():
    ap = argparse.ArgumentParser(description="GKDT checkpoint -> GGUF converter")
    ap.add_argument("--checkpoint", default="models/pytorch/gkd_fullset.best")
    ap.add_argument("--bpe", default=None, help="bpe_simple_vocab_16e6.txt.gz (default: repo dinov3_kd copy)")
    ap.add_argument("--dtype", default="f32", choices=["f32", "f16", "q8_0", "q4_0", "q4_K"])
    ap.add_argument("--output", default=None)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--sigma", type=float, default=14.0)
    ap.add_argument("--pad-kps", type=int, default=80)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # cpp_ggml/
    repo = os.path.dirname(root)                                        # repository root
    if args.bpe is None:
        args.bpe = os.path.join(repo, "network", "dinov3_kd", "bpe_simple_vocab_16e6.txt.gz")
    if args.output is None:
        name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        os.makedirs(os.path.join(root, "models", "gguf"), exist_ok=True)
        args.output = os.path.join(root, "models", "gguf", f"{name}-{args.dtype}.gguf")

    cfg = {"SQUARE_IMAGE_LENGTH": args.img_size, "SIGMA": args.sigma, "PAD_KPS": args.pad_kps}
    convert(args.checkpoint, args.bpe, args.dtype, args.output, cfg)


if __name__ == "__main__":
    main()
