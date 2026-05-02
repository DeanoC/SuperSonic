#!/usr/bin/env python3
"""
GPTQ-style INT4 calibration bake for SuperSonic.

Reads a HuggingFace Qwen3.5 checkpoint, runs GPTQ calibration against
WikiText-2, and writes a SuperSonic baked package at
  {model-dir}/.supersonic/v{FORMAT_VERSION}-int4-gptq/

Runtime format (unchanged from the previous min/max baker, so the HIP kernels
don't change):
  - Packed INT4, 2 nibbles/byte: low = even col, high = odd col
  - Per-tile BF16 scale + zero_point, group_size=128 (2D tile: rows/gs x cols/gs)
  - Asymmetric quant: q = clamp(round(w/scale + zero_f), 0, 15)
  - Dequant: w = q*scale - zero_f*scale

GPTQ details:
  - Hessian H = 2/N * sum(x x^T) collected via nn.Linear forward-pre hooks
  - Diagonal dampened by damp * mean(diag(H))
  - Upper-triangular Cholesky(H^-1); column-wise error propagation
  - Scales for a column-group are chosen from the (error-updated) weight values
    at the start of each group, per gs-row tile
  - Sequential calibration: each layer's quantized outputs feed the next
    layer's activation collection (standard GPTQ pattern)
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import gc
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn


def _malloc_trim() -> None:
    """Return free()'d host pages to the OS — glibc's allocator otherwise
    holds them in its arena cache, which made the per-layer GPTQ loop on
    35B-A3B grow RSS by ~10 GiB across 40 layers and OOM at the lm_head
    step. Best-effort: silently no-op on non-glibc platforms."""
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _release_host_memory() -> None:
    """Run between layers / heavy steps to reclaim host RAM. The pattern is:
    drop unreferenced Python objects → empty CUDA caching pool → ask glibc
    to release arena pages back to the OS. Without the malloc_trim call the
    OS still sees the pages as resident even after PyTorch frees them."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _malloc_trim()

# -- constants mirrored from crates/model-store/src/manifest.rs --
FORMAT_VERSION = 2
CONVERTER_VERSION = 1

# LayoutTag strings must match the Rust enum variants exactly (serde default).
LAYOUT_RAW = "Raw"
LAYOUT_CONV_SQ = "DepthwiseConvSqueezed"
LAYOUT_HEAD_BIAS = "HeadBiasReshaped"
LAYOUT_HEAD_EXP = "HeadExpReshaped"
LAYOUT_FP8_DEQ = "Fp8Dequantized"
LAYOUT_FP8_NATIVE = "Fp8Native"
LAYOUT_INT4 = "Int4Quantized"


def log(msg: str) -> None:
    print(msg, flush=True)


def copy_weight_in_row_chunks(dst: torch.Tensor, src: torch.Tensor, rows_per_chunk: int = 4096) -> None:
    """Copy a large CPU weight into an existing module parameter without a full-device clone."""
    with torch.no_grad():
        for start in range(0, src.shape[0], rows_per_chunk):
            end = min(start + rows_per_chunk, src.shape[0])
            dst[start:end].copy_(
                src[start:end].to(device=dst.device, dtype=dst.dtype, non_blocking=True)
            )


# ---------------------------------------------------------------------------
# Target-tensor selection (mirrors crates/model-store/src/baker.rs::is_int4_target)
# ---------------------------------------------------------------------------
# Fused 3D MoE expert tensors in Qwen3.6-MoE store all experts in one slab and
# do NOT use the `.weight` suffix:
#     mlp.experts.gate_up_proj    [E, 2*moe_int, hidden]
#     mlp.experts.down_proj       [E, hidden,    moe_int]
# They quantize via a separate fused-MoE driver (see fused_expert_minmax_int4),
# but `is_int4_target` still classifies them as INT4 targets so the planner /
# manifest accounting line up.
FUSED_EXPERT_SUFFIXES = (
    ".mlp.experts.gate_up_proj",
    ".mlp.experts.down_proj",
)


def is_fused_expert_target(name: str) -> bool:
    return any(name.endswith(s) for s in FUSED_EXPERT_SUFFIXES)


def is_int4_target(name: str) -> bool:
    if is_fused_expert_target(name):
        return True
    if not name.endswith(".weight"):
        return False
    if ("layernorm" in name
            or "norm.weight" in name
            or "embed_tokens" in name
            or "conv1d" in name):
        return False
    if "in_proj_b.weight" in name or "in_proj_a.weight" in name:
        return False
    return ("_proj" in name) or (name == "lm_head.weight")


# ---------------------------------------------------------------------------
# Byte encoding helpers
# ---------------------------------------------------------------------------
def align_up(x: int, a: int) -> int:
    return (x + a - 1) & ~(a - 1)


def bf16_to_bytes(t: torch.Tensor) -> bytes:
    """Serialise a bfloat16 tensor as little-endian bytes (2 bytes/element)."""
    t = t.to(torch.bfloat16).contiguous().cpu()
    # view-as-int16 preserves the raw 2-byte pattern; numpy writes LE on LE hosts.
    arr = t.view(torch.int16).numpy()
    if sys.byteorder != "little":
        arr = arr.byteswap()
    return arr.tobytes()


def tensor_to_bytes(t: torch.Tensor, dtype_str: str) -> bytes:
    t = t.contiguous().cpu()
    if dtype_str == "bf16":
        return bf16_to_bytes(t)
    if dtype_str == "f32":
        return t.to(torch.float32).numpy().tobytes()
    if dtype_str == "f16":
        return t.to(torch.float16).numpy().tobytes()
    if dtype_str == "u8":
        return t.to(torch.uint8).numpy().tobytes()
    if dtype_str == "u32":
        return t.to(torch.uint32).numpy().tobytes()
    if dtype_str == "i64":
        return t.to(torch.int64).numpy().tobytes()
    raise ValueError(f"unsupported dtype_str {dtype_str}")


def torch_dtype_to_str(dt: torch.dtype) -> str:
    return {
        torch.bfloat16: "bf16",
        torch.float32: "f32",
        torch.float16: "f16",
        torch.uint8: "u8",
        torch.int64: "i64",
    }.get(dt, "bf16")


# ---------------------------------------------------------------------------
# Rust-side tensor transforms (mirror crates/model-store/src/transforms.rs)
# ---------------------------------------------------------------------------
def classify_tensor(name: str, shape: list[int], weight_prefix: str,
                    layer_types: list[str]) -> str:
    prefix = f"{weight_prefix}.layers."
    if name.startswith(prefix):
        rest = name[len(prefix):]
        dot = rest.find(".")
        if dot > 0:
            try:
                idx = int(rest[:dot])
                if idx < len(layer_types) and layer_types[idx] == "linear_attention":
                    if name.endswith(".conv1d.weight") and len(shape) == 3 and shape[1] == 1:
                        return LAYOUT_CONV_SQ
                    if name.endswith(".dt_bias") and len(shape) == 1:
                        return LAYOUT_HEAD_BIAS
                    if name.endswith(".A_log") and len(shape) == 1:
                        return LAYOUT_HEAD_EXP
            except ValueError:
                pass
    return LAYOUT_RAW


def apply_layout(t: torch.Tensor, shape: list[int], layout: str,
                 dtype_str: str) -> tuple[bytes, list[int], str]:
    if layout == LAYOUT_RAW:
        return tensor_to_bytes(t, dtype_str), shape, dtype_str
    if layout == LAYOUT_CONV_SQ:
        return tensor_to_bytes(t, dtype_str), [shape[0], shape[2]], dtype_str
    if layout == LAYOUT_HEAD_BIAS:
        return tensor_to_bytes(t, dtype_str), [1, 1, shape[0]], dtype_str
    if layout == LAYOUT_HEAD_EXP:
        t_exp = torch.exp(t.to(torch.float32))
        return bf16_to_bytes(t_exp), [1, 1, shape[0]], "bf16"
    raise ValueError(f"unknown layout {layout}")


# ---------------------------------------------------------------------------
# GPTQ core
# ---------------------------------------------------------------------------
@torch.no_grad()
def gptq_quantize(
    W: torch.Tensor,     # [out, in] float32 (in-place allowed)
    H: torch.Tensor,     # [in, in] float32
    group_size: int,
    damp: float,
    blocksize: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run GPTQ column-wise quantization with 2D tile scales matching the
    SuperSonic runtime format (one (scale, zero) pair per gs x gs tile).

    Returns
    -------
    Q_dq        [out, in]         float32  dequantized quantized weights
    nibbles     [out, in]         uint8    4-bit values, 0..15
    scale_tile  [out/gs, in/gs]   float32  per-tile scale
    zero_tile   [out/gs, in/gs]   float32  per-tile zero_f
    """
    out_f, in_f = W.shape
    device = W.device
    gs = group_size
    if in_f % gs != 0:
        raise ValueError(f"in_features {in_f} must be a multiple of group_size {gs}")
    if out_f % gs != 0:
        raise ValueError(f"out_features {out_f} must be a multiple of group_size {gs}")

    scale_rows = out_f // gs
    scale_cols = in_f // gs
    row_to_gr = torch.arange(out_f, device=device) // gs  # [out_f] -> tile index

    # --- Hessian preparation ---
    H = H.clone().to(torch.float32)
    diag_mean = torch.mean(torch.diagonal(H)).item()
    if not math.isfinite(diag_mean) or diag_mean <= 0.0:
        # No activations → isotropic fallback (equivalent to plain MSE).
        H = torch.eye(in_f, device=device, dtype=torch.float32)
        diag_mean = 1.0
    d_idx = torch.arange(in_f, device=device)
    H[d_idx, d_idx] += damp * max(diag_mean, 1e-8)

    # Dead columns: no activation energy → zero those columns and clamp diag.
    dead = torch.diag(H) < 1e-10
    if dead.any():
        W[:, dead] = 0.0
        H[dead, dead] = 1.0

    # Upper-triangular Cholesky of H^{-1}, same convention as AutoGPTQ.
    try:
        L = torch.linalg.cholesky(H)
    except Exception:
        H[d_idx, d_idx] += 10.0 * damp * max(diag_mean, 1e-8)
        L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    # --- output buffers ---
    W = W.clone().to(torch.float32)
    nibbles = torch.zeros((out_f, in_f), dtype=torch.uint8, device=device)
    Q_dq = torch.zeros_like(W)
    scale_tile = torch.zeros((scale_rows, scale_cols), dtype=torch.float32, device=device)
    zero_tile = torch.zeros((scale_rows, scale_cols), dtype=torch.float32, device=device)

    def set_tile_scales(gc: int, group_view: torch.Tensor) -> None:
        # group_view: [out_f, W] where W <= gs
        # Reshape to [scale_rows, gs, W] and compute per-row-tile min/max in one go.
        # PR4d quality fix: AutoGPTQ-style scale search. Plain min/max gave
        # per-tensor cos_sim 0.88-0.97 vs safetensors on the 35B-A3B bake;
        # see docs/qwen36-moe-bake-quality-audit.md. Search over a small grid
        # of "shrunk" scale factors (p < 1) and pick the per-row-tile setting
        # that minimises post-BF16-round MSE on the dequantised tile. p < 1
        # trades clipping at the distribution tails for tighter quantisation
        # across the body — typically net positive for trained weights
        # which concentrate near zero.
        width = group_view.shape[1]
        gv = group_view.reshape(scale_rows, gs, width)
        tmax_orig = gv.amax(dim=(1, 2))  # [scale_rows]
        tmin_orig = gv.amin(dim=(1, 2))

        def derive(tmin_v: torch.Tensor, tmax_v: torch.Tensor):
            rng_v = tmax_v - tmin_v
            sc_v = torch.where(rng_v > 0, rng_v / 15.0, torch.ones_like(rng_v))
            zf_v = torch.where(rng_v > 0, -tmin_v / sc_v, torch.zeros_like(rng_v))
            # Round through BF16 so what we evaluate matches what the runtime
            # kernel reads (scale/zero are BF16 sidecars).
            return (sc_v.to(torch.bfloat16).to(torch.float32),
                    zf_v.to(torch.bfloat16).to(torch.float32))

        def tile_mse(sc_v: torch.Tensor, zf_v: torch.Tensor) -> torch.Tensor:
            # Quantise + dequantise this column-group with the candidate
            # (sc, zf) per row-tile, return per-row-tile MSE vs `gv`.
            sc_b = sc_v.unsqueeze(1).unsqueeze(2)  # [scale_rows, 1, 1]
            zf_b = zf_v.unsqueeze(1).unsqueeze(2)
            # Guard against the rng==0 fallback yielding sc_b=1 → sc_b is
            # always > 0 from `derive`; this is just defensive.
            safe_sc = torch.where(sc_b == 0, torch.ones_like(sc_b), sc_b)
            q = torch.clamp(torch.round(gv / safe_sc + zf_b), 0.0, 15.0)
            recon = (q * sc_b - zf_b * sc_b).to(torch.bfloat16).to(torch.float32)
            return ((gv - recon) ** 2).mean(dim=(1, 2))

        # Baseline: full min/max (p=1.0). Always kept as the fallback so a
        # numerically odd shrink can never regress us below the original.
        best_sc, best_zf = derive(tmin_orig, tmax_orig)
        best_mse = tile_mse(best_sc, best_zf)

        # Shrunk-range candidates. p < 1 multiplies both tmin and tmax by p
        # (symmetric shrink toward zero — same convention as AutoGPTQ's
        # `Quantizer.find_params` mse=True path). Grid kept small (~7
        # candidates) so the per-block scale-search cost stays under a
        # millisecond per call; AutoGPTQ defaults to maxshrink=0.8 grid=100,
        # but the marginal MSE gain after the first ~5 candidates is
        # negligible for trained weights.
        for p in (0.95, 0.90, 0.85, 0.80, 0.75, 0.70):
            sc_p, zf_p = derive(p * tmin_orig, p * tmax_orig)
            mse_p = tile_mse(sc_p, zf_p)
            better = mse_p < best_mse
            best_sc = torch.where(better, sc_p, best_sc)
            best_zf = torch.where(better, zf_p, best_zf)
            best_mse = torch.where(better, mse_p, best_mse)

        scale_tile[:, gc] = best_sc
        zero_tile[:, gc] = best_zf

    # Iterate columns in blocks; use blocksize == group_size so block boundaries
    # line up with column-group boundaries.
    blocksize = min(blocksize, gs)
    for b in range(0, in_f, blocksize):
        e = min(b + blocksize, in_f)
        count = e - b
        W1 = W[:, b:e].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[b:e, b:e]
        Hinv1_diag = torch.diagonal(Hinv1).clamp_min(1e-12)

        for i in range(count):
            col = b + i
            w_col = W1[:, i]
            d_ii = Hinv1_diag[i]

            if col % gs == 0:
                gc = col // gs
                tile_end_global = min(col + gs, in_f)
                in_block_end = min(i + gs, count)
                if tile_end_global <= e:
                    group_view = W1[:, i:in_block_end]
                else:
                    ext = W[:, e:tile_end_global]
                    group_view = torch.cat([W1[:, i:count], ext], dim=1)
                set_tile_scales(gc, group_view)

            gc = col // gs
            sc_col = scale_tile[row_to_gr, gc]  # [out_f]
            zf_col = zero_tile[row_to_gr, gc]
            q = torch.clamp(torch.round(w_col / sc_col + zf_col), 0.0, 15.0)
            q_int = q.to(torch.uint8)
            # Compute the dequantised value as the kernel does (`q*s - zf*s`)
            # and round through BF16 so the residual error passed to later
            # columns matches what the runtime kernel will actually see.
            q_dq = (q * sc_col - zf_col * sc_col)
            q_dq = q_dq.to(torch.bfloat16).to(torch.float32)

            nibbles[:, col] = q_int
            Q_dq[:, col] = q_dq
            Q1[:, i] = q_dq
            err = (w_col - q_dq) / d_ii
            if i + 1 < count:
                W1[:, i + 1:] -= err.unsqueeze(1) * Hinv1[i, i + 1:]
            Err1[:, i] = err

        # Propagate accumulated block error to remaining columns.
        if e < in_f:
            W[:, e:] -= Err1 @ Hinv[b:e, e:]
        W[:, b:e] = Q1

    return Q_dq, nibbles, scale_tile, zero_tile


def pack_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit nibbles 2/byte along the last axis.

    Accepts 2D `[rows, cols]` (dense projections) or 3D `[E, rows, cols]`
    (fused MoE experts). Output keeps all leading axes and halves the last:
    `[..., cols]` -> `[..., cols/2]`.
    """
    if nibbles.dim() < 2:
        raise ValueError(f"expected >=2D nibble tensor, got shape {tuple(nibbles.shape)}")
    cols = nibbles.shape[-1]
    if cols % 2 != 0:
        raise ValueError(f"last dim must be even, got {cols}")
    leading = nibbles.shape[:-1]
    r = nibbles.reshape(*leading, cols // 2, 2).to(torch.uint8)
    return (r[..., 0] | (r[..., 1] << 4)).contiguous()


@torch.no_grad()
def _minmax_int4_search_2d(
    tiles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """AutoGPTQ-style scale search for 2D-tiled INT4 quantisation.

    `tiles` is a [scale_rows, gs, scale_cols, gs] float32 tensor — i.e. the
    weights laid out so that axes (1, 3) span one quantisation tile. Returns
    `(scale, zero)` of shape `[scale_rows, scale_cols]` (BF16-rounded F32),
    chosen per-tile from a small grid of "shrink" factors that minimise
    post-BF16-round MSE on the dequantised tile.

    Used by `fused_expert_minmax_int4*`. The dense-projection scale search
    in `gptq_quantize::set_tile_scales` runs the same algorithm specialised
    for one column-group at a time (1D scale shape) so the two paths agree
    on what "best scale" means.
    """
    tmax_orig = tiles.amax(dim=(1, 3))  # [scale_rows, scale_cols]
    tmin_orig = tiles.amin(dim=(1, 3))

    def derive(tmin_v: torch.Tensor, tmax_v: torch.Tensor):
        rng_v = tmax_v - tmin_v
        sc_v = torch.where(rng_v > 0, rng_v / 15.0, torch.ones_like(rng_v))
        zf_v = torch.where(rng_v > 0, -tmin_v / sc_v, torch.zeros_like(rng_v))
        # Round through BF16 — what the runtime kernel reads.
        return (sc_v.to(torch.bfloat16).to(torch.float32),
                zf_v.to(torch.bfloat16).to(torch.float32))

    def tile_mse(sc_v: torch.Tensor, zf_v: torch.Tensor) -> torch.Tensor:
        sc_b = sc_v.unsqueeze(1).unsqueeze(3)  # [sr, 1, sc, 1]
        zf_b = zf_v.unsqueeze(1).unsqueeze(3)
        safe_sc = torch.where(sc_b == 0, torch.ones_like(sc_b), sc_b)
        q = torch.clamp(torch.round(tiles / safe_sc + zf_b), 0.0, 15.0)
        recon = (q * sc_b - zf_b * sc_b).to(torch.bfloat16).to(torch.float32)
        return ((tiles - recon) ** 2).mean(dim=(1, 3))  # [sr, sc]

    best_sc, best_zf = derive(tmin_orig, tmax_orig)
    best_mse = tile_mse(best_sc, best_zf)
    for p in (0.95, 0.90, 0.85, 0.80, 0.75, 0.70):
        sc_p, zf_p = derive(p * tmin_orig, p * tmax_orig)
        mse_p = tile_mse(sc_p, zf_p)
        better = mse_p < best_mse
        best_sc = torch.where(better, sc_p, best_sc)
        best_zf = torch.where(better, zf_p, best_zf)
        best_mse = torch.where(better, mse_p, best_mse)
    return best_sc, best_zf


# ---------------------------------------------------------------------------
# Fused MoE expert quantization (no GPTQ — min/max group-quant per expert).
#
# Qwen3.6-MoE stores each layer's experts as fused 3D tensors:
#     mlp.experts.gate_up_proj    [E, 2*moe_int, hidden]   bf16
#     mlp.experts.down_proj       [E, hidden,    moe_int]  bf16
# These are nn.Parameters under a custom MoE module, NOT a list of nn.Linear,
# so the GPTQ driver above (which walks nn.Linear) skips them. Running them
# through full Hessian-aware GPTQ on the producer host is also impractical for
# 256 experts × 40 layers × 2 projections.
#
# Plan §15 option (c): keep GPTQ for the dense projections, give the experts a
# straight min/max INT4 group-quant per expert. Each expert gets its own
# `[out/gs, in/gs]` BF16 scale+zero tile, fused along axis 0 with the other
# experts in the layer. The runtime accounting in
# `crates/qwen36_moe/src/weights.rs` (search `int4_bytes`) already sizes for
# this layout.
# ---------------------------------------------------------------------------
@torch.no_grad()
def fused_expert_minmax_int4(
    W: torch.Tensor,
    group_size: int,
    work_device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-expert min/max INT4 group-quant for a fused MoE weight slab.

    Parameters
    ----------
    W : [E, out, in] tensor (any float dtype). Treated as `E` parallel
        `[out, in]` matrices; each is independently quantized.
    group_size : tile dim (must divide both `out` and `in`; `in` must be even).
    work_device : optional device for the per-expert reductions; defaults to
        `W.device` (avoids an unnecessary copy when the model is already on GPU).

    Returns
    -------
    nibbles : [E, out, in]            uint8, values 0..15  (CPU)
    scales  : [E, out/gs, in/gs]      float32 (BF16-rounded, CPU)
    zeros   : [E, out/gs, in/gs]      float32 (BF16-rounded, CPU)
    """
    if W.dim() != 3:
        raise ValueError(f"fused expert tensor must be 3D, got shape {tuple(W.shape)}")
    E, out_f, in_f = W.shape
    gs = group_size
    if in_f % gs != 0 or in_f % 2 != 0:
        raise ValueError(
            f"in_features {in_f} must be divisible by group_size={gs} and even"
        )
    if out_f % gs != 0:
        raise ValueError(f"out_features {out_f} must be divisible by group_size={gs}")
    scale_rows = out_f // gs
    scale_cols = in_f // gs
    nibbles_out = torch.empty((E, out_f, in_f), dtype=torch.uint8)
    scale_out = torch.empty((E, scale_rows, scale_cols), dtype=torch.float32)
    zero_out = torch.empty((E, scale_rows, scale_cols), dtype=torch.float32)
    dev = work_device if work_device is not None else W.device

    for e in range(E):
        slab = W[e].to(device=dev, dtype=torch.float32)
        # Tile shape [sr, gs, sc, gs] so axes (1, 3) span one tile.
        tiles = slab.reshape(scale_rows, gs, scale_cols, gs)
        # Per-tile scale search (same algorithm as gptq_quantize).
        sc, zf = _minmax_int4_search_2d(tiles)
        sc_full = sc.repeat_interleave(gs, dim=0).repeat_interleave(gs, dim=1)
        zf_full = zf.repeat_interleave(gs, dim=0).repeat_interleave(gs, dim=1)
        q = torch.clamp(torch.round(slab / sc_full + zf_full), 0.0, 15.0).to(torch.uint8)
        nibbles_out[e] = q.cpu()
        scale_out[e] = sc.cpu()
        zero_out[e] = zf.cpu()
        del slab, tiles, sc_full, zf_full, q

    return nibbles_out, scale_out, zero_out


@torch.no_grad()
def fused_expert_minmax_int4_packed(
    W: torch.Tensor,
    group_size: int,
    work_device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """As `fused_expert_minmax_int4`, but pack nibbles per-expert as we go.

    The 35B-A3B fused-expert nibble buffer is ~32 GiB unpacked vs ~16 GiB
    packed. Packing each expert's slab into the output buffer immediately
    (rather than calling the unpacked variant and packing afterward) keeps
    peak host RAM at the packed size — the unpacked builder allocates the
    full `[E, out, in]` uint8 tensor, which on 35B-A3B alone is ~32 GiB and
    won't co-exist with an Accelerate-offloaded BF16 model on a 64 GiB host.

    Returns
    -------
    packed_nibbles : [E, out, in/2]      uint8 (CPU)
    scales         : [E, out/gs, in/gs]  float32 (BF16-rounded, CPU)
    zeros          : [E, out/gs, in/gs]  float32 (BF16-rounded, CPU)
    """
    if W.dim() != 3:
        raise ValueError(f"fused expert tensor must be 3D, got shape {tuple(W.shape)}")
    E, out_f, in_f = W.shape
    gs = group_size
    if in_f % gs != 0 or in_f % 2 != 0:
        raise ValueError(
            f"in_features {in_f} must be divisible by group_size={gs} and even"
        )
    if out_f % gs != 0:
        raise ValueError(f"out_features {out_f} must be divisible by group_size={gs}")
    scale_rows = out_f // gs
    scale_cols = in_f // gs
    packed_out = torch.empty((E, out_f, in_f // 2), dtype=torch.uint8)
    scale_out = torch.empty((E, scale_rows, scale_cols), dtype=torch.float32)
    zero_out = torch.empty((E, scale_rows, scale_cols), dtype=torch.float32)
    dev = work_device if work_device is not None else W.device

    for e in range(E):
        slab = W[e].to(device=dev, dtype=torch.float32)
        tiles = slab.reshape(scale_rows, gs, scale_cols, gs)
        # Per-tile scale search (same algorithm as gptq_quantize).
        sc, zf = _minmax_int4_search_2d(tiles)
        sc_full = sc.repeat_interleave(gs, dim=0).repeat_interleave(gs, dim=1)
        zf_full = zf.repeat_interleave(gs, dim=0).repeat_interleave(gs, dim=1)
        q = torch.clamp(torch.round(slab / sc_full + zf_full), 0.0, 15.0).to(torch.uint8)
        # Pack 2 nibbles/byte before leaving the device — this is the inner-most
        # cost on the host RAM budget.
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous().cpu()
        packed_out[e] = packed
        scale_out[e] = sc.cpu()
        zero_out[e] = zf.cpu()
        del slab, tiles, sc_full, zf_full, q, packed

    return packed_out, scale_out, zero_out


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------
class HessianHook:
    """Accumulate H = 2/N * sum_t x_t x_t^T on the inputs to a linear."""

    def __init__(self, linear: nn.Linear):
        self.linear = linear
        self.H: torch.Tensor | None = None
        self.N: int = 0
        self._handle = linear.register_forward_pre_hook(self._pre)

    def _pre(self, module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        x = inputs[0]
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        x = x.to(torch.float32)
        n = x.shape[0]
        xx = (x.T @ x) * 2.0
        if self.H is None:
            self.H = xx / n
            self.N = n
        else:
            new_N = self.N + n
            self.H = self.H * (self.N / new_N) + xx / new_N
            self.N = new_N

    def close(self) -> None:
        self._handle.remove()


class _CatcherStop(Exception):
    pass


class _Catcher(nn.Module):
    """Wraps the first transformer layer so we can snapshot (hidden, kwargs)
    without running the rest of the model."""

    def __init__(self, orig: nn.Module):
        super().__init__()
        self.orig = orig
        self.hiddens: list[torch.Tensor] = []
        self.kwargs: dict[str, Any] | None = None

    def forward(self, hidden_states, *args, **kwargs):
        self.hiddens.append(hidden_states.detach().cpu())
        if self.kwargs is None:
            self.kwargs = {k: _detach_tree(v) for k, v in kwargs.items()}
        raise _CatcherStop()


def _detach_tree(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach()
    if isinstance(x, tuple):
        return tuple(_detach_tree(v) for v in x)
    if isinstance(x, list):
        return [_detach_tree(v) for v in x]
    if isinstance(x, dict):
        return {k: _detach_tree(v) for k, v in x.items()}
    return x


def _move_tree(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, tuple):
        return tuple(_move_tree(v, device) for v in x)
    if isinstance(x, list):
        return [_move_tree(v, device) for v in x]
    if isinstance(x, dict):
        return {k: _move_tree(v, device) for k, v in x.items()}
    return x


def capture_layer0_inputs(
    model: nn.Module,
    layers: nn.ModuleList,
    calib_ids: torch.Tensor,
    device: torch.device,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    nsamples = calib_ids.shape[0]
    catcher = _Catcher(layers[0])
    layers[0] = catcher
    try:
        with torch.no_grad():
            for s in range(nsamples):
                ids = calib_ids[s:s + 1].to(device)
                try:
                    model(ids, use_cache=False)
                except _CatcherStop:
                    pass
                del ids
    finally:
        layers[0] = catcher.orig
    assert len(catcher.hiddens) == nsamples, "catcher missed samples"
    assert catcher.kwargs is not None, "catcher missed kwargs"
    return catcher.hiddens, catcher.kwargs


def move_kwargs_to(kwargs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: _move_tree(v, device) for k, v in kwargs.items()}


# ---------------------------------------------------------------------------
# Sequential per-layer GPTQ driver
# ---------------------------------------------------------------------------
def _materialize_param(
    param: torch.Tensor,
    hf_name: str,
    safetensors_loader: Callable[[str], torch.Tensor | None] | None,
) -> torch.Tensor:
    """Return the parameter's actual data, even if Accelerate offloaded it.

    Under `device_map="auto"`, offloaded parameters appear as `meta`-device
    placeholders after a forward pass — `param.data` then yields a meta
    tensor and any subsequent op on it crashes mid-GPTQ with a device
    mismatch. When that happens, fall back to reading the raw bytes
    straight from the source safetensors files (the loader callback knows
    the prefix-mapping logic that resolves HF flattened names to raw
    safetensors keys).
    """
    if param.device.type != "meta":
        return param.detach()
    if safetensors_loader is None:
        raise RuntimeError(
            f"{hf_name} is on meta device and no safetensors fallback was given"
        )
    t = safetensors_loader(hf_name)
    if t is None:
        raise RuntimeError(
            f"{hf_name} is on meta device and could not be loaded from safetensors"
        )
    return t.detach()


def _writeback_param(
    mod: nn.Module,
    Q_dq: torch.Tensor,
) -> bool:
    """Try to copy the dequantised quantised weight back into `mod.weight`.

    Returns True when the write succeeded. Under Accelerate offloading the
    parameter's `.data` may be on `meta`, in which case `.copy_()` would
    silently no-op into a non-existent tensor; rather than masking that
    failure, return False so the caller can adjust expectations (the
    per-layer re-run will then see the BF16 original instead of Q_dq —
    weakens GPTQ error propagation but does not corrupt anything).
    """
    p = mod.weight
    if p.device.type == "meta":
        return False
    p.data.copy_(Q_dq.to(p.dtype))
    return True


def _stream_quantized_tensor(
    writer: "StreamingPackageWriter",
    raw_name: str,
    nibbles: torch.Tensor,           # [out, in] u8 (unpacked) or [E, out, in/2] u8 (already packed)
    scale_t: torch.Tensor,           # f32, BF16-rounded
    zero_t: torch.Tensor,            # f32, BF16-rounded
) -> None:
    """Pack (if needed) and stream a quantized tensor + sidecars to `writer`.
    Mirrors the per-tensor block from the old in-RAM `tensors_out`-building
    loop in `main`, kept in one place so dense GPTQ / lm_head / fused
    experts all share the same on-disk layout."""
    if nibbles.dim() == 3:
        # Fused MoE experts arrive already packed.
        packed = nibbles
    else:
        packed = pack_nibbles(nibbles)
    writer.write_tensor(
        raw_name, packed.numpy().tobytes(),
        list(packed.shape), "u8", LAYOUT_INT4,
    )
    writer.write_tensor(
        f"{raw_name}_int4_scale", bf16_to_bytes(scale_t),
        list(scale_t.shape), "bf16", LAYOUT_RAW,
    )
    writer.write_tensor(
        f"{raw_name}_int4_zero", bf16_to_bytes(zero_t),
        list(zero_t.shape), "bf16", LAYOUT_RAW,
    )


def _selfcheck_dense(
    nibbles: torch.Tensor,
    scale_t: torch.Tensor,
    zero_t: torch.Tensor,
    live: torch.Tensor,
    group_size: int,
) -> tuple[bool, float]:
    """Reconstruct a dense GPTQ tensor from (nibbles, scale, zero) and
    compare against the live `mod.weight.data`. Returns (matched, linf).
    A mismatch indicates a bake-vs-runtime format bug (not quantization
    quality), so we want this loud and immediate."""
    rows, cols = nibbles.shape
    row_gr = torch.arange(rows) // group_size
    col_gc = torch.arange(cols) // group_size
    sc_full = scale_t[row_gr][:, col_gc]
    zf_full = zero_t[row_gr][:, col_gc]
    recon = nibbles.float() * sc_full - zf_full * sc_full
    recon = recon.to(torch.bfloat16).to(torch.float32)
    matched = torch.equal(recon, live)
    linf = (recon - live).abs().max().item() if not matched else 0.0
    return matched, linf


def _selfcheck_fused(
    packed: torch.Tensor,             # [E, out, in/2] u8
    scale_t: torch.Tensor,            # [E, out/gs, in/gs] f32
    zero_t: torch.Tensor,
    orig_bf16: torch.Tensor | None,   # [E, out, in] BF16 (may be None if unavailable)
    group_size: int,
) -> tuple[bool, float]:
    """Reconstruct a fused MoE expert tensor from packed nibbles + sidecars
    and compare against the BF16 original. Returns (nibble_range_ok, linf).
    Linf is reported as a quality signal — fused experts use min/max
    quant, so non-zero Linf is expected; we only fail loudly on the
    nibble-range invariant."""
    E, rows, packed_cols = packed.shape
    cols = packed_cols * 2
    lo = (packed & 0x0F).to(torch.uint8)
    hi = (packed >> 4).to(torch.uint8)
    unpacked = torch.empty((E, rows, cols), dtype=torch.uint8)
    unpacked[:, :, 0::2] = lo
    unpacked[:, :, 1::2] = hi
    nibble_range_ok = int(unpacked.max().item()) <= 15
    linf = 0.0
    if orig_bf16 is not None:
        row_gr = torch.arange(rows) // group_size
        col_gc = torch.arange(cols) // group_size
        sc_full = scale_t.index_select(1, row_gr).index_select(2, col_gc)
        zf_full = zero_t.index_select(1, row_gr).index_select(2, col_gc)
        recon = unpacked.float() * sc_full - zf_full * sc_full
        recon = recon.to(torch.bfloat16).to(torch.float32)
        orig_f32 = orig_bf16.detach().to(device="cpu", dtype=torch.float32)
        linf = (recon - orig_f32).abs().max().item()
        del recon, sc_full, zf_full, orig_f32
    del lo, hi, unpacked
    return nibble_range_ok, linf


def quantize_model(
    model: nn.Module,
    calib_ids: torch.Tensor,
    device: torch.device,
    group_size: int,
    damp: float,
    writer: "StreamingPackageWriter",
    hf_to_raw: dict[str, str],
    safetensors_loader: Callable[[str], torch.Tensor | None] | None = None,
) -> dict[str, Any]:
    """
    Run sequential GPTQ, **streaming** each quantized tensor to `writer` as
    it's produced (rather than accumulating in a host-RAM dict). Returns a
    self-check stats dict:

        {
          "quantized_names": set[str],         # HF state-dict names that were quantized
          "dense_total":     int,
          "dense_mismatch":  int,
          "dense_worst":     (name, linf),
          "fused_total":     int,
          "fused_worst":     (name, linf),
          "nibble_range_ok": bool,
        }

    `hf_to_raw` maps HF state-dict names to safetensors keys so the writer
    can use raw names (the runtime-visible layout). `safetensors_loader(hf_name)`
    is consulted only when a parameter is on the `meta` device (Accelerate
    offloading); the dense paths still prefer the live `mod.weight` when
    it's on a real device.
    """
    model.eval()
    # Locate the transformer-decoder block list. Qwen3.5 (non-MM) stores it at
    # `model.model.layers`; Qwen3.6-MoE uses the multimodal class
    # `Qwen3_5MoeForConditionalGeneration` and nests under
    # `model.model.language_model.layers` alongside `.visual.*`.
    inner = model.model
    if hasattr(inner, "layers"):
        text_root = inner
    elif hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
        text_root = inner.language_model
    else:
        raise SystemExit(
            "could not locate transformer layers under model.model[.language_model]"
        )
    layers = text_root.layers
    num_layers = len(layers)

    # Map nn.Linear -> state-dict weight name (for naming output tensors)
    module_to_name: dict[int, str] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            module_to_name[id(mod)] = name + ".weight"

    log(f"[gptq] capturing layer-0 inputs from {calib_ids.shape[0]} samples...")
    hidden_cpu, layer_kwargs = capture_layer0_inputs(model, layers, calib_ids, device)
    layer_kwargs_dev = move_kwargs_to(layer_kwargs, device)

    stats: dict[str, Any] = {
        "quantized_names": set(),     # HF state-dict names successfully streamed
        "dense_total": 0,
        "dense_mismatch": 0,
        "dense_worst": ("", 0.0),
        "fused_total": 0,
        "fused_worst": ("", 0.0),
        "nibble_range_ok": True,
    }
    nsamples = len(hidden_cpu)

    for layer_idx in range(num_layers):
        layer = layers[layer_idx]
        targets: list[tuple[str, nn.Linear]] = []
        for mod in layer.modules():
            if isinstance(mod, nn.Linear):
                name = module_to_name[id(mod)]
                if is_int4_target(name):
                    targets.append((name, mod))

        log(f"[gptq] layer {layer_idx + 1}/{num_layers}: "
            f"{len(targets)} targets, collecting Hessians over {nsamples} samples")

        hooks = {name: HessianHook(mod) for name, mod in targets}
        with torch.no_grad():
            for s in range(nsamples):
                hs = hidden_cpu[s].to(device)
                out = layer(hs, **layer_kwargs_dev)
                if isinstance(out, tuple):
                    out = out[0]
                del hs, out
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Quantize each target module.
        for name, mod in targets:
            hook = hooks[name]
            H = hook.H
            hook.close()
            if H is None:
                log(f"[gptq]   {name}: WARNING no activations captured, skipping")
                continue
            t0 = time.perf_counter()
            # Bypass Accelerate's `meta`-device placeholder by reading from
            # safetensors when needed. The fallback is silent on layers whose
            # weights happen to be on a real device after forward.
            raw = _materialize_param(mod.weight, name, safetensors_loader)
            W = raw.to(device=device, dtype=torch.float32)
            Q_dq, nibbles, scale_t, zero_t = gptq_quantize(W, H, group_size, damp)
            elapsed = time.perf_counter() - t0
            wrote = _writeback_param(mod, Q_dq)

            raw_name = hf_to_raw.get(name)
            if raw_name is None:
                log(f"[gptq]   {name}: WARN no safetensors raw name; skipping write")
            else:
                nibbles_cpu = nibbles.cpu()
                scale_cpu   = scale_t.cpu()
                zero_cpu    = zero_t.cpu()
                _stream_quantized_tensor(writer, raw_name, nibbles_cpu, scale_cpu, zero_cpu)
                stats["quantized_names"].add(name)
                # Inline self-check: recon should byte-match `mod.weight.data`
                # after the writeback above. Only meaningful when the writeback
                # actually landed (meta-device params skip writeback).
                if wrote:
                    stats["dense_total"] += 1
                    live = mod.weight.data.to(torch.float32).cpu()
                    matched, linf = _selfcheck_dense(
                        nibbles_cpu, scale_cpu, zero_cpu, live, group_size,
                    )
                    if not matched:
                        stats["dense_mismatch"] += 1
                        if linf > stats["dense_worst"][1]:
                            stats["dense_worst"] = (name, linf)
                    del live
                del nibbles_cpu, scale_cpu, zero_cpu

            log(f"[gptq]   {name}: shape={tuple(W.shape)} "
                f"H_N={hook.N} took {elapsed:.1f}s"
                + ("" if wrote else " [meta param: re-run will use BF16]"))
            del raw, W, Q_dq, nibbles, scale_t, zero_t, H

        # Re-run the (now partially-quantized) layer so the next layer sees
        # post-quantization activations.
        with torch.no_grad():
            for s in range(nsamples):
                hs = hidden_cpu[s].to(device)
                out = layer(hs, **layer_kwargs_dev)
                if isinstance(out, tuple):
                    out = out[0]
                hidden_cpu[s] = out.detach().cpu()
                del hs, out
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Hand free()d arena pages back to the OS. Without this the C
        # runtime accumulates ~10 GiB of fragmentation across 40 layers
        # of 35B-A3B and OOMs at the lm_head step.
        _release_host_memory()

    # --- lm_head GPTQ pass ---
    # The transformer layer loop above hooks every nn.Linear inside the text
    # decoder, but lm_head sits outside. Capture its Hessian here (post-final-
    # norm hidden state) and run the same column-wise GPTQ. This lets the
    # runtime skip the 250k×hidden BF16 read on the dominant decode-side
    # matmul. `final_norm` lives on the same submodule as the layer list
    # (text_root), which is `model.model` for Qwen3.5 and
    # `model.model.language_model` for Qwen3.6-MoE.
    final_norm = getattr(text_root, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if (
        isinstance(lm_head, nn.Linear)
        and final_norm is not None
        and is_int4_target("lm_head.weight")
    ):
        log(f"[gptq] lm_head: collecting Hessian over {nsamples} samples")
        # Compute the Hessian directly from `final_norm(hidden)` instead of
        # invoking `lm_head(normed)` to fire a forward-pre hook. The hook
        # version produced a [1, seqlen, vocab] BF16 output (~1 GiB at
        # vocab=248320) per iteration; on a 40 GiB-cpu host that already
        # holds 30+ GiB of offloaded weights, the transient pushed RSS over
        # the OOM threshold near the end of the 128-sample loop.
        H: torch.Tensor | None = None
        H_N = 0
        with torch.no_grad():
            for s in range(nsamples):
                hs = hidden_cpu[s].to(device)
                normed = final_norm(hs)
                x = normed
                if x.dim() > 2:
                    x = x.reshape(-1, x.shape[-1])
                x = x.to(torch.float32)
                n = x.shape[0]
                xx = (x.T @ x) * 2.0
                if H is None:
                    H = xx / n
                    H_N = n
                else:
                    new_N = H_N + n
                    H = H * (H_N / new_N) + xx / new_N
                    H_N = new_N
                del hs, normed, x, xx
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if H is None:
            log("[gptq]   lm_head: WARNING no activations captured, skipping")
        else:
            t0 = time.perf_counter()
            # The output head is huge on 9B-class checkpoints. Quantize it on
            # CPU to avoid requiring an extra multi-GiB clone on an already-full
            # producer GPU after the layer sweep. Fall back to safetensors if
            # Accelerate has the lm_head weight on `meta`.
            raw = _materialize_param(lm_head.weight, "lm_head.weight", safetensors_loader)
            W = raw.to(device="cpu", dtype=torch.float32)
            H = H.detach().to(device="cpu", dtype=torch.float32)
            Q_dq, nibbles, scale_t, zero_t = gptq_quantize(W, H, group_size, damp)
            elapsed = time.perf_counter() - t0
            wrote = lm_head.weight.device.type != "meta"
            if wrote:
                copy_weight_in_row_chunks(lm_head.weight.data, Q_dq)

            # lm_head.weight is always `lm_head.weight` raw — outside the
            # `model.language_model.*` prefix so `hf_to_raw` may not have it.
            raw_name = hf_to_raw.get("lm_head.weight", "lm_head.weight")
            nibbles_cpu = nibbles.cpu()
            scale_cpu   = scale_t.cpu()
            zero_cpu    = zero_t.cpu()
            _stream_quantized_tensor(writer, raw_name, nibbles_cpu, scale_cpu, zero_cpu)
            stats["quantized_names"].add("lm_head.weight")
            if wrote:
                stats["dense_total"] += 1
                live = lm_head.weight.data.to(torch.float32).cpu()
                matched, linf = _selfcheck_dense(
                    nibbles_cpu, scale_cpu, zero_cpu, live, group_size,
                )
                if not matched:
                    stats["dense_mismatch"] += 1
                    if linf > stats["dense_worst"][1]:
                        stats["dense_worst"] = ("lm_head.weight", linf)
                del live
            del nibbles_cpu, scale_cpu, zero_cpu

            log(f"[gptq]   lm_head: shape={tuple(W.shape)} "
                f"H_N={H_N} took {elapsed:.1f}s"
                + ("" if wrote else " [meta param: write-back skipped]"))
            del raw, W, Q_dq, nibbles, scale_t, zero_t, H
            _release_host_memory()

    # --- Fused MoE expert pass ---
    # Qwen3.6-MoE stores all experts in two 3D nn.Parameters per layer rather
    # than as a list of nn.Linear, so the per-layer GPTQ loop above never sees
    # them. Also: a Hessian-aware GPTQ over the fused layout doesn't really fit
    # (see docs/qwen36-moe-plan.md §15). Plan §15 option (c) — plain min/max
    # INT4 group-quant per expert, gs=128, BF16 scale/zero — covers the runtime
    # VRAM gap (~60 GiB BF16 → ~15 GiB INT4 at 35B) cheaply.
    #
    # Memory model: under `device_map="auto"` the fused experts are mostly on
    # CPU (or partially disk-offloaded). We iterate `named_parameters()` to
    # avoid materialising a whole state_dict, materialise one fused tensor at
    # a time on `device`, pack nibbles immediately to halve RAM use vs the
    # unpacked layout, then free the BF16 source. Writing path detects packed
    # 3D shapes via `dim() == 3` (dense GPTQ stays 2D + unpacked).
    fused_param_names = sorted(
        n for n, _ in model.named_parameters() if is_fused_expert_target(n)
    )
    if fused_param_names:
        log(f"[gptq] fused MoE experts: quantizing {len(fused_param_names)} "
            f"3D tensors (min/max group-quant, streaming packed nibbles to disk)")
        named_params = dict(model.named_parameters())
        for name in fused_param_names:
            t0 = time.perf_counter()
            param = named_params[name]
            # Same Accelerate `meta` problem: fall back to safetensors when
            # the fused tensor is offloaded.
            raw = _materialize_param(param, name, safetensors_loader)
            W = raw.to(device="cpu", dtype=torch.bfloat16)
            packed, scale_t, zero_t = fused_expert_minmax_int4_packed(
                W, group_size=group_size, work_device=device,
            )
            elapsed = time.perf_counter() - t0
            log(f"[gptq]   {name}: shape={tuple(W.shape)} -> "
                f"packed={tuple(packed.shape)} took {elapsed:.1f}s")

            raw_name = hf_to_raw.get(name, name)
            _stream_quantized_tensor(writer, raw_name, packed, scale_t, zero_t)
            stats["quantized_names"].add(name)
            stats["fused_total"] += 1
            # Inline self-check while `W` is still on CPU. Linf is purely a
            # quality signal (min/max quant always shows non-zero error);
            # the nibble-range invariant is the structural pin.
            range_ok, linf = _selfcheck_fused(packed, scale_t, zero_t, W, group_size)
            if not range_ok:
                stats["nibble_range_ok"] = False
            if linf > stats["fused_worst"][1]:
                stats["fused_worst"] = (name, linf)

            del raw, W, packed, scale_t, zero_t
            if device.type == "cuda":
                torch.cuda.empty_cache()
            # Reclaim host pages between fused experts. Each tensor is a few
            # hundred MiB on 35B-A3B (256 experts × MoE_intermediate); without
            # malloc_trim, glibc keeps them in its arena cache and we drift
            # ~10 GiB upward over the 80-tensor pass.
            _release_host_memory()

    return stats


# ---------------------------------------------------------------------------
# Perplexity sanity check
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_ppl(model: nn.Module, tokenizer, device: torch.device,
                seqlen: int, n_chunks: int) -> float:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[0]
    total_tokens = ids.numel()
    n_chunks = min(n_chunks, total_tokens // seqlen)
    if n_chunks < 1:
        return float("nan")
    nll_sum = 0.0
    tok_sum = 0
    for c in range(n_chunks):
        chunk = ids[c * seqlen:(c + 1) * seqlen].to(device).unsqueeze(0)
        out = model(chunk, labels=chunk)
        # HF shifts labels internally: loss is averaged over (seqlen-1) tokens.
        n = chunk.numel() - 1
        nll_sum += out.loss.float().item() * n
        tok_sum += n
    return math.exp(nll_sum / tok_sum)


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------
def _load_raw_tensor_names(model_dir: Path) -> set[str]:
    """Read the raw safetensors key list (pre-HF-flattening)."""
    from safetensors import safe_open
    index = model_dir / "model.safetensors.index.json"
    keys: set[str] = set()
    if index.exists():
        idx = json.loads(index.read_text())
        keys.update(idx["weight_map"].keys())
        return keys
    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            keys.update(f.keys())
        return keys
    for p in sorted(model_dir.glob("model*.safetensors")):
        with safe_open(str(p), framework="pt") as f:
            keys.update(f.keys())
    if not keys:
        raise SystemExit(f"no safetensors files found in {model_dir}")
    return keys


def load_raw_tensor(model_dir: Path, name: str) -> torch.Tensor | None:
    """Load a tensor directly from safetensors, preserving its original dtype.

    Used for tensors HF downcasts at load time (e.g. A_log is stored F32 but
    HF loads as BF16, which loses precision before our exp() transform).
    """
    from safetensors import safe_open
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        idx = json.loads(index.read_text())
        shard = idx["weight_map"].get(name)
        if shard is None:
            return None
        with safe_open(str(model_dir / shard), framework="pt") as f:
            return f.get_tensor(name)
    for p in [model_dir / "model.safetensors", *model_dir.glob("model*.safetensors")]:
        if not p.exists():
            continue
        with safe_open(str(p), framework="pt") as f:
            if name in f.keys():
                return f.get_tensor(name)
    return None


class RawTensorLoader:
    def __init__(self, model_dir: Path):
        from safetensors import safe_open

        self.model_dir = model_dir
        self.safe_open = safe_open
        index = model_dir / "model.safetensors.index.json"
        if index.exists():
            idx = json.loads(index.read_text())
            self.weight_map: dict[str, str] = dict(idx["weight_map"])
        else:
            self.weight_map = {}
            for p in [model_dir / "model.safetensors", *model_dir.glob("model*.safetensors")]:
                if not p.exists():
                    continue
                with safe_open(str(p), framework="pt", device="cpu") as f:
                    for k in f.keys():
                        self.weight_map[k] = p.name
        self._handles: dict[str, Any] = {}

    def get(self, name: str) -> torch.Tensor | None:
        shard = self.weight_map.get(name)
        if shard is None:
            return None
        handle = self._handles.get(shard)
        if handle is None:
            handle = self.safe_open(str(self.model_dir / shard), framework="pt", device="cpu")
            self._handles[shard] = handle
        return handle.get_tensor(name)


def dequant_fp8_blocks(
    w: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    if w.dim() != 2 or scale_inv.dim() != 2:
        raise ValueError(
            f"FP8 dequant expects 2D tensors, got weight={tuple(w.shape)} "
            f"scale={tuple(scale_inv.shape)}"
        )
    rows, cols = w.shape
    scale_rows, scale_cols = scale_inv.shape
    out = torch.empty((rows, cols), dtype=torch.bfloat16)
    w_f = w.to(torch.float32)
    s_f = scale_inv.to(torch.float32)
    for sr in range(scale_rows):
        r0 = sr * block_size
        r1 = min(r0 + block_size, rows)
        for sc in range(scale_cols):
            c0 = sc * block_size
            c1 = min(c0 + block_size, cols)
            out[r0:r1, c0:c1] = (w_f[r0:r1, c0:c1] * s_f[sr, sc]).to(torch.bfloat16)
    return out


def dequant_fp8_tensor(w: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    if w.dim() == 2:
        return dequant_fp8_blocks(w.detach().cpu(), scale_inv.detach().cpu())
    if w.dim() == 3 and scale_inv.dim() == 3:
        return torch.stack(
            [
                dequant_fp8_blocks(w[e].detach().cpu(), scale_inv[e].detach().cpu())
                for e in range(w.shape[0])
            ],
            dim=0,
        )
    raise ValueError(
        f"unsupported FP8 dequant shape: weight={tuple(w.shape)} "
        f"scale={tuple(scale_inv.shape)}"
    )


def _device_map_target(model: nn.Module, name: str) -> str | int | torch.device | None:
    """Return the Accelerate device-map target for a parameter, if present."""
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return None
    parts = name.split(".")
    for i in range(len(parts), -1, -1):
        prefix = ".".join(parts[:i])
        if prefix in device_map:
            return device_map[prefix]
    return None


def _torch_device_from_target(target: str | int | torch.device) -> torch.device | None:
    if isinstance(target, torch.device):
        return target
    if isinstance(target, int):
        return torch.device("cuda", target)
    if target == "disk":
        return None
    return torch.device(target)


def set_module_parameter(model: nn.Module, name: str, value: torch.Tensor) -> bool:
    parts = name.split(".")
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    old = module._parameters[parts[-1]]
    if old is not None and old.device.type != "meta":
        target_device = old.device
    else:
        mapped = _device_map_target(model, name)
        target_device = _torch_device_from_target(mapped) if mapped is not None else None
    if target_device is None:
        log(
            f"[bake-int4] WARNING: {name} is offloaded/meta; leaving FP8 "
            "parameter in place to preserve Accelerate placement"
        )
        return False
    module._parameters[parts[-1]] = nn.Parameter(
        value.to(device=target_device, dtype=torch.bfloat16),
        requires_grad=old.requires_grad if old is not None else False,
    )
    return True


def load_fused_expert_bf16(
    model_dir: Path,
    raw_keys: set[str],
    raw_base: str,
    kind: str,
    loader: RawTensorLoader | None = None,
) -> torch.Tensor:
    expert_re = re.compile(rf"^{re.escape(raw_base)}\.(\d+)\.")
    expert_ids = sorted({
        int(m.group(1))
        for k in raw_keys
        if (m := expert_re.match(k)) is not None
    })
    if not expert_ids:
        raise KeyError(f"no raw experts found under {raw_base}")
    chunks: list[torch.Tensor] = []
    for expert_id in expert_ids:
        base = f"{raw_base}.{expert_id}"
        if kind == "gate_up_proj":
            gate = load_raw_tensor_bf16(model_dir, f"{base}.gate_proj.weight", loader)
            up = load_raw_tensor_bf16(model_dir, f"{base}.up_proj.weight", loader)
            chunks.append(torch.cat([gate, up], dim=0).unsqueeze(0))
            del gate, up
        elif kind == "down_proj":
            down = load_raw_tensor_bf16(model_dir, f"{base}.down_proj.weight", loader)
            chunks.append(down.unsqueeze(0))
            del down
        else:
            raise ValueError(f"unknown fused expert kind {kind}")
    out = torch.cat(chunks, dim=0)
    del chunks
    return out


def dequantize_remaining_fp8_parameters(
    model: nn.Module,
    model_dir: Path,
    raw_keys: set[str],
) -> int:
    named_params = dict(model.named_parameters())
    loader = RawTensorLoader(model_dir)
    fp8_names = [name for name, param in named_params.items() if param.dtype == torch.float8_e4m3fn]
    if fp8_names:
        log(f"[bake-int4] dequantizing {len(fp8_names)} remaining FP8 parameter(s)")
    converted = 0
    processed = 0
    for name, param in list(named_params.items()):
        if param.dtype != torch.float8_e4m3fn:
            continue
        processed += 1
        scale = named_params.get(f"{name}_scale_inv")
        if scale is not None:
            bf16 = dequant_fp8_tensor(param, scale).to(torch.bfloat16)
        elif name.endswith(".mlp.experts.gate_up_proj") or name.endswith(".mlp.experts.down_proj"):
            raw_name = name
            if raw_name.startswith("model.layers."):
                raw_name = raw_name.replace("model.layers.", "model.language_model.layers.", 1)
            kind = raw_name.rsplit(".", 1)[1]
            raw_base = raw_name.rsplit(".", 1)[0]
            bf16 = load_fused_expert_bf16(model_dir, raw_keys, raw_base, kind, loader)
        else:
            log(f"[bake-int4] WARNING: FP8 parameter {name} has no scale_inv; leaving as-is")
            continue
        if set_module_parameter(model, name, bf16):
            converted += 1
        if converted and (converted % 10 == 0 or processed == len(fp8_names)):
            log(f"[bake-int4]   dequantized {converted}/{len(fp8_names)} FP8 parameter(s)")
        del bf16
    if converted:
        _release_host_memory()
    return converted


def load_raw_tensor_bf16(
    model_dir: Path,
    name: str,
    loader: RawTensorLoader | None = None,
) -> torch.Tensor:
    t = loader.get(name) if loader is not None else load_raw_tensor(model_dir, name)
    if t is None:
        raise KeyError(f"raw tensor not found: {name}")
    scale_name = f"{name}_scale_inv"
    scale = loader.get(scale_name) if loader is not None else load_raw_tensor(model_dir, scale_name)
    if scale is not None:
        return dequant_fp8_tensor(t, scale)
    return t.to(torch.bfloat16)


def stream_mtp_tensors(
    writer: "StreamingPackageWriter",
    model_dir: Path,
    raw_keys: set[str],
) -> int:
    """Stream Qwen3.6 MoE MTP tensors in runtime layout.

    HF stores the MTP MoE experts as per-expert `gate_proj`/`up_proj`/
    `down_proj` FP8 tensors. Runtime code uses the same fused expert layout as
    the main decoder layers: one `[E, 2*moe_int, hidden]` gate/up slab and one
    `[E, hidden, moe_int]` down slab.
    """
    expert_re = re.compile(r"^mtp\.layers\.0\.mlp\.experts\.(\d+)\.")
    passthrough = sorted(
        k for k in raw_keys
        if k.startswith("mtp.")
        and not k.endswith("_scale_inv")
        and expert_re.match(k) is None
    )
    loader = RawTensorLoader(model_dir)
    written = 0
    for raw_name in passthrough:
        if writer.has(raw_name):
            continue
        t = load_raw_tensor_bf16(model_dir, raw_name, loader).detach().to(device="cpu")
        b = bf16_to_bytes(t)
        writer.write_tensor(raw_name, b, list(t.shape), "bf16", LAYOUT_RAW)
        written += 1
        del t, b

    expert_ids = sorted({
        int(m.group(1))
        for k in raw_keys
        if (m := expert_re.match(k)) is not None
    })
    if expert_ids:
        gate_up_chunks: list[torch.Tensor] = []
        down_chunks: list[torch.Tensor] = []
        for expert_id in expert_ids:
            base = f"mtp.layers.0.mlp.experts.{expert_id}"
            gate = load_raw_tensor_bf16(model_dir, f"{base}.gate_proj.weight", loader)
            up = load_raw_tensor_bf16(model_dir, f"{base}.up_proj.weight", loader)
            down = load_raw_tensor_bf16(model_dir, f"{base}.down_proj.weight", loader)
            gate_up_chunks.append(torch.cat([gate, up], dim=0).unsqueeze(0))
            down_chunks.append(down.unsqueeze(0))
            del gate, up, down

        fused_specs = [
            ("mtp.layers.0.mlp.experts.gate_up_proj", torch.cat(gate_up_chunks, dim=0)),
            ("mtp.layers.0.mlp.experts.down_proj", torch.cat(down_chunks, dim=0)),
        ]
        for raw_name, fused in fused_specs:
            if writer.has(raw_name):
                continue
            b = bf16_to_bytes(fused)
            writer.write_tensor(raw_name, b, list(fused.shape), "bf16", LAYOUT_RAW)
            written += 1
            del fused, b
        del gate_up_chunks, down_chunks

    _release_host_memory()
    return written


def _build_name_map(hf_keys: list[str], raw_keys: set[str]) -> dict[str, str]:
    """
    Map each HF state-dict name to its safetensors name. HF flattens some
    nested modules (e.g. Qwen3.5 drops ".language_model."), so we try the
    known substitutions and take the first one that resolves.
    """
    candidates = [
        lambda n: n,
        lambda n: n.replace("model.", "model.language_model.", 1)
                  if n.startswith("model.") and not n.startswith("model.language_model.")
                  else n,
        lambda n: n.replace("model.", "model.text_model.", 1)
                  if n.startswith("model.") and not n.startswith("model.text_model.")
                  else n,
    ]
    out: dict[str, str] = {}
    for k in hf_keys:
        for fn in candidates:
            cand = fn(k)
            if cand in raw_keys:
                out[k] = cand
                break
    return out


class StreamingPackageWriter:
    """Streaming sink for the bake's `weights.bin` + `manifest.json`.

    The original in-RAM `tensors_out` list held every quantized tensor's
    bytes (~17 GiB on 35B-A3B: ~15 GiB packed expert nibbles + ~1 GiB BF16
    embeds + ~1 GiB scale/zero sidecars) until the final `write_package`
    flushed them out. On a 64 GiB host, that list plus the still-loaded
    BF16 model plus the GPTQ working set blew through host RAM and OOM'd
    near the end of the run.
    Mirrors `BakePackageWriter` in `bake_q4km.py`. The on-disk format
    (4096-byte alignment, `manifest.json` schema) is identical to the old
    `write_package`; only the producer side becomes streaming.
    """

    def __init__(self, out_dir: Path, model_family: str = "qwen35"):
        self.out_dir = out_dir
        self.model_family = model_family
        out_dir.mkdir(parents=True, exist_ok=True)
        self.weights_path = out_dir / "weights.bin"
        self._fh = open(self.weights_path, "wb")
        self._cursor = 0
        self._entries: list[dict[str, Any]] = []
        self._names: set[str] = set()

    def write_tensor(
        self,
        name: str,
        data: bytes,
        shape: list[int],
        dtype_str: str,
        layout: str,
    ) -> int:
        """Append a tensor; return its 4096-aligned byte offset."""
        if name in self._names:
            raise ValueError(f"duplicate tensor name {name!r} written to bake")
        offset = align_up(self._cursor, 4096)
        if offset > self._cursor:
            self._fh.write(b"\x00" * (offset - self._cursor))
        self._fh.write(data)
        byte_len = len(data)
        self._entries.append({
            "name": name,
            "shape": list(shape),
            "dtype": dtype_str,
            "layout": layout,
            "offset": offset,
            "byte_len": byte_len,
        })
        self._names.add(name)
        self._cursor = offset + byte_len
        return offset

    def has(self, name: str) -> bool:
        return name in self._names

    def finalize(self) -> None:
        """Flush weights.bin and emit manifest.json. Sorts manifest entries
        alphabetically for stable diffs — runtime keys by name into a
        HashMap so on-disk ordering doesn't matter, but a sorted manifest
        makes diffing two bake outputs sane."""
        self._fh.flush()
        self._fh.close()
        sorted_entries = sorted(self._entries, key=lambda e: e["name"])
        manifest = {
            "format_version": FORMAT_VERSION,
            "converter_version": CONVERTER_VERSION,
            "model_family": self.model_family,
            "tensors": sorted_entries,
        }
        with open(self.out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        log(f"[bake-int4] wrote {self._cursor / (1024 * 1024):.1f} MiB to {self.weights_path}")
        log(f"[bake-int4] manifest: {self.out_dir / 'manifest.json'}")

    def __enter__(self) -> "StreamingPackageWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.finalize()
        else:
            # On error: close the file handle but skip manifest emission so
            # downstream code can't accidentally consume a partial bake.
            self._fh.close()


def write_package(
    out_dir: Path,
    tensors: list[tuple[str, bytes, list[int], str, str]],
) -> None:
    """Compat wrapper for callers that still want the in-RAM list API.
    All new code should construct a `StreamingPackageWriter` directly so
    peak host RAM stays bounded by the largest single tensor."""
    with StreamingPackageWriter(out_dir) as writer:
        for (name, data, shape, dtype_str, layout) in tensors:
            writer.write_tensor(name, data, shape, dtype_str, layout)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPTQ INT4 calibration bake for SuperSonic")
    p.add_argument("--model-dir", required=True, type=Path,
                   help="Path to the HuggingFace model directory")
    p.add_argument("--num-samples", type=int, default=128,
                   help="Number of calibration sequences (default 128)")
    p.add_argument("--seqlen", type=int, default=2048,
                   help="Calibration sequence length (default 2048)")
    p.add_argument("--group-size", type=int, default=128,
                   help="INT4 group size (must be 128 to match runtime)")
    p.add_argument("--damp", type=float, default=0.01,
                   help="Hessian diagonal dampening factor (default 0.01)")
    p.add_argument("--device", default=None,
                   help="Torch device (default: cuda if available else cpu)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-ppl", action="store_true",
                   help="Skip the perplexity sanity check after baking")
    p.add_argument("--ppl-chunks", type=int, default=16,
                   help="Number of 2048-token chunks for the PPL check (default 16)")
    p.add_argument("--out-dir", default=None, type=Path,
                   help="Override output directory (default: "
                        "{model-dir}/.supersonic/v{FORMAT_VERSION}-int4-gptq)")
    # Streaming offload — needed when BF16 weights don't fit a single GPU
    # (35B-A3B is ~67 GiB, never fits 24 GiB VRAM). Same pattern as
    # oracle/q4km_stream_gptq_bake.py: pass through to from_pretrained and
    # skip the unconditional .to(device) so HF Accelerate manages placement.
    p.add_argument("--device-map", default=None,
                   help="Optional Transformers device_map, e.g. 'auto'. When set, "
                        "the model is NOT explicitly moved with .to(device); "
                        "HF Accelerate spreads layers across GPU/CPU/disk.")
    p.add_argument("--max-memory", default=None,
                   help="Optional JSON max_memory for HF Accelerate, e.g. "
                        "'{\"0\":\"20GiB\",\"cpu\":\"50GiB\"}' (24 GiB GPU + 64 GiB host).")
    p.add_argument("--offload-folder", default=None, type=Path,
                   help="Disk-offload folder for params that don't fit GPU+CPU "
                        "(needed for ~3 GiB shortfall on 35B-A3B with 24+50 GiB).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_dir: Path = args.model_dir
    if not model_dir.exists():
        raise SystemExit(f"model dir does not exist: {model_dir}")

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[bake-int4] device={device}")
    if device.type == "cpu":
        log("[bake-int4] WARNING: running GPTQ on CPU will be very slow for a 4B "
            "model. Use a CUDA/ROCm GPU if available.")

    # --- Load model + tokenizer ---
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    log(f"[bake-int4] loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    log(f"[bake-int4] loading model (bf16) from {model_dir}")
    raw_keys = _load_raw_tensor_names(model_dir)

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    hf_config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    quant_cfg = getattr(hf_config, "quantization_config", None)
    quant_method = (
        quant_cfg.get("quant_method")
        if isinstance(quant_cfg, dict)
        else getattr(quant_cfg, "quant_method", None)
    )
    if quant_method == "fp8":
        from transformers import FineGrainedFP8Config

        load_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        log("[bake-int4] source checkpoint is FP8; dequantizing to BF16 at load")
    if args.device_map:
        load_kwargs["device_map"] = args.device_map
        log(f"[bake-int4] device_map={args.device_map!r}")
    if args.max_memory:
        raw_max_memory = json.loads(args.max_memory)
        # Accelerate accepts ints (GPU index) or "cpu"/"disk" string keys.
        load_kwargs["max_memory"] = {
            (int(k) if isinstance(k, str) and k.isdigit() else k): v
            for k, v in raw_max_memory.items()
        }
        log(f"[bake-int4] max_memory={load_kwargs['max_memory']}")
    if args.offload_folder:
        args.offload_folder.mkdir(parents=True, exist_ok=True)
        load_kwargs["offload_folder"] = str(args.offload_folder)
        log(f"[bake-int4] offload_folder={args.offload_folder}")

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
    converted_fp8 = dequantize_remaining_fp8_parameters(model, model_dir, raw_keys)
    if converted_fp8:
        log(f"[bake-int4] dequantized {converted_fp8} remaining FP8 parameter(s) to BF16")
    if args.device_map is None:
        model = model.to(device)
    model.eval()

    # --- Determine the canonical tensor-name prefix by reading the raw
    # safetensors key set. HuggingFace's AutoModelForCausalLM may flatten
    # nested modules (e.g. drop ".language_model." for Qwen3.5), so the
    # state_dict names don't always match what the Rust loader expects.
    # Anchor the prefix on ".embed_tokens.weight" — some checkpoints also
    # include an "mtp.layers.*" multi-token-prediction module, so matching
    # on ".layers.0." alone picks the wrong one.
    weight_prefix = None
    for k in raw_keys:
        if k.endswith(".embed_tokens.weight"):
            weight_prefix = k[: -len(".embed_tokens.weight")]
            break
    if weight_prefix is None:
        # Fallback: first key with ".layers.0." — sorted so it's stable.
        for k in sorted(raw_keys):
            if ".layers.0." in k:
                weight_prefix = k.split(".layers.")[0]
                break
    if weight_prefix is None:
        raise SystemExit("could not infer weight prefix from safetensors keys")
    log(f"[bake-int4] canonical weight prefix (from safetensors): {weight_prefix!r}")
    model_family = (
        "qwen36-moe"
        if any(".mlp.experts." in k for k in raw_keys)
        else "qwen35"
    )
    log(f"[bake-int4] model_family={model_family!r}")

    # Avoid `model.state_dict()` here — under `device_map="auto"` it gathers
    # every offloaded param onto CPU at once, which on a 35B model easily
    # exceeds host RAM. `named_parameters()` + `named_buffers()` give the same
    # name set without materialising data; we materialise per-tensor on demand
    # in the writing loop below.
    named_lookup: dict[str, torch.Tensor] = {}
    for n, p in model.named_parameters():
        named_lookup[n] = p
    for n, b in model.named_buffers():
        named_lookup.setdefault(n, b)
    sd_keys = list(named_lookup.keys())
    hf_to_raw = _build_name_map(sd_keys, raw_keys)
    missing_in_raw = [k for k in sd_keys if k not in hf_to_raw]
    if missing_in_raw:
        log(f"[bake-int4] {len(missing_in_raw)} state-dict tensors have no "
            f"safetensors counterpart (e.g. tied weights) — skipping them.")

    config = model.config
    text_cfg = getattr(config, "text_config", config)
    num_layers = int(text_cfg.num_hidden_layers)
    layer_types = list(getattr(text_cfg, "layer_types", None) or [])
    if not layer_types:
        layer_types = [
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(num_layers)
        ]
    log(f"[bake-int4] num_layers={num_layers} "
        f"full_layers={sum(1 for t in layer_types if t == 'full_attention')}")

    # `safetensors_loader` is the meta-device escape hatch for `quantize_model`.
    # When Accelerate offloads a parameter, its `.data` becomes a meta tensor
    # after the per-layer forward and any subsequent op crashes with a
    # device-mismatch. The loader resolves the HF flattened name to a raw
    # safetensors key (using `hf_to_raw`) and reads the bytes directly.
    def safetensors_loader(hf_name: str) -> torch.Tensor | None:
        raw_name = hf_to_raw.get(hf_name)
        if raw_name is None:
            # Try the live mapping candidates as a last resort — covers
            # parameters that aren't in the state-dict-derived sd_keys (e.g.
            # untied lm_head when the safetensors prefix differs).
            for cand in (
                hf_name,
                hf_name.replace("model.", "model.language_model.", 1)
                if hf_name.startswith("model.")
                and not hf_name.startswith("model.language_model.")
                else hf_name,
            ):
                if cand in raw_keys:
                    raw_name = cand
                    break
        if raw_name is None:
            return None
        return load_raw_tensor(model_dir, raw_name)

    # --- Load calibration data ---
    log("[bake-int4] loading WikiText-2 train split via `datasets`...")
    from datasets import load_dataset
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(r["text"] for r in train if r["text"].strip())
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[0]
    log(f"[bake-int4] tokenized train: {ids.numel()} tokens")
    if ids.numel() < args.seqlen * 2:
        raise SystemExit(
            f"not enough calibration tokens ({ids.numel()}) for "
            f"seqlen={args.seqlen}"
        )

    torch.manual_seed(args.seed)
    max_start = ids.numel() - args.seqlen - 1
    starts = torch.randint(0, max_start, (args.num_samples,))
    calib = torch.stack([ids[s:s + args.seqlen] for s in starts])
    log(f"[bake-int4] calibration batch: {tuple(calib.shape)}")

    # --- Open the streaming writer up front so quantized tensors flow to
    # disk as soon as they're produced. The producer side never holds more
    # than one (nibbles, scale, zero) triple in RAM at a time; on 35B-A3B
    # that's ~1 GiB peak (largest fused expert slab) instead of the ~17 GiB
    # the in-RAM `tensors_out` list reached before this refactor.
    out_dir = args.out_dir or (
        model_dir / ".supersonic" / f"v{FORMAT_VERSION}-int4-gptq"
    )
    # Context-manager form: on a partial run (exception propagating out),
    # `__exit__` closes the file handle but skips manifest emission so a
    # downstream bake-consumer can't accidentally read a half-written package
    # as if it were complete.
    with StreamingPackageWriter(out_dir, model_family=model_family) as writer:
        # --- GPTQ ---
        t0 = time.perf_counter()
        stats = quantize_model(
            model, calib, device,
            group_size=args.group_size,
            damp=args.damp,
            writer=writer,
            hf_to_raw=hf_to_raw,
            safetensors_loader=safetensors_loader,
        )
        gptq_elapsed = time.perf_counter() - t0
        log(f"[bake-int4] GPTQ done in {gptq_elapsed / 60.0:.1f} min "
            f"({len(stats['quantized_names'])} tensors quantized + streamed)")

        # --- Sample generation sanity check (quantized weights live in the model) ---
        try:
            sample_ids = tokenizer("The quick brown fox", return_tensors="pt"
                                   ).input_ids.to(device)
            with torch.no_grad():
                gen = model.generate(sample_ids, max_new_tokens=12,
                                     do_sample=False, use_cache=True)
            log(f"[bake-int4] sample gen (post-quant, python-side): "
                f"{tokenizer.decode(gen[0], skip_special_tokens=True)!r}")
        except Exception as ex:
            log(f"[bake-int4] sample gen failed: {ex}")

        # Self-check stats — already collected inline during quantize_model
        # so no second pass is needed (the old standalone pass had to
        # re-walk the entire `quantized` dict and held every tensor's bytes
        # in RAM until it finished).
        log(f"[self-check] dense INT4: "
            f"{stats['dense_mismatch']}/{stats['dense_total']} mismatch"
            + (f" (worst: {stats['dense_worst'][0]} "
               f"Linf={stats['dense_worst'][1]:.2e})"
               if stats['dense_mismatch'] else ""))
        if stats['fused_total']:
            log(f"[self-check] fused MoE INT4: {stats['fused_total']} tensors, "
                f"nibble_range_ok={stats['nibble_range_ok']}, "
                f"worst-Linf-vs-bf16={stats['fused_worst'][0]} "
                f"({stats['fused_worst'][1]:.2e})")

        # --- Perplexity sanity check (with quantized weights live in the model) ---
        if not args.skip_ppl:
            log("[bake-int4] running perplexity sanity check on WikiText-2 test...")
            try:
                ppl = compute_ppl(model, tokenizer, device,
                                  seqlen=args.seqlen, n_chunks=args.ppl_chunks)
                log(f"[bake-int4] PPL (WikiText-2 test, {args.ppl_chunks} chunks): "
                    f"{ppl:.2f}")
            except Exception as ex:
                log(f"[bake-int4] PPL check failed: {ex}")

        # --- Stream non-quantized tensors ---
        log("[bake-int4] streaming non-quantized tensors...")
        # Walk the HF state dict but emit under raw-safetensors names so the
        # Rust loader (which expects the raw layout, e.g.
        # "model.language_model.X") finds every tensor.
        eligible = [
            hf_name for hf_name in sd_keys
            if hf_name in hf_to_raw
            and not hf_to_raw[hf_name].endswith("_scale_inv")
            and (
                hf_to_raw[hf_name].startswith(f"{weight_prefix}.")
                or hf_to_raw[hf_name] == "lm_head.weight"
            )
        ]
        eligible.sort()

        # ------------------------------------------------------------------
        # Phase 6.1: pass-through `mtp.*` tensors (multi-token-prediction
        # head) directly from safetensors. HF's `Qwen3_5MoeForCausalLM`
        # strips them via `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`,
        # so `named_parameters()` doesn't see them, but the bake reader
        # (`crates/model-store/src/baker.rs`) does include them in the
        # raw_keys filter (PR shipped alongside this change). Stored as
        # raw BF16 — no INT4 calibration this round; the MTP block is
        # one layer's worth of compute and BF16 vs INT4 is a wash for
        # speculative-decode draft pass throughput.
        mtp_written = stream_mtp_tensors(writer, model_dir, raw_keys)
        if mtp_written:
            log(f"[bake-int4] passed through {mtp_written} mtp.* "
                f"tensor(s) (BF16 raw)")

        # When `lm_head.weight` is tied to `embed_tokens.weight`, the HF state
        # dict entry usually has no safetensors counterpart and the eligible
        # loop above skips it. But if quantize_model just ran GPTQ on
        # `model.lm_head` and streamed an INT4 tensor for it, that tensor is
        # already in the writer — `writer.has(...)` skips it and the runtime
        # loads the INT4 version instead of falling back to the BF16 embed alias.
        if ("lm_head.weight" in stats["quantized_names"]
                and "lm_head.weight" not in eligible):
            eligible.append("lm_head.weight")
            if "lm_head.weight" not in hf_to_raw:
                hf_to_raw["lm_head.weight"] = "lm_head.weight"
            if "lm_head.weight" not in named_lookup and hasattr(model, "lm_head"):
                named_lookup["lm_head.weight"] = model.lm_head.weight

        for hf_name in eligible:
            raw_name = hf_to_raw[hf_name]
            # Quantized tensors were streamed during quantize_model already;
            # the writer's name set is the source of truth.
            if writer.has(raw_name):
                continue
            # Materialise one tensor at a time onto host RAM — under
            # device_map="auto" this is what triggers Accelerate to fetch the
            # offloaded slice. If the param is on `meta` (post-forward
            # placeholder), fall back to safetensors via the loader so we
            # never write a meta tensor's bytes into the bake.
            param_ref = named_lookup.get(hf_name)
            if param_ref is None:
                continue
            if param_ref.device.type == "meta":
                t = safetensors_loader(hf_name)
                if t is None:
                    log(f"[bake-int4] WARNING {hf_name}: meta param + no "
                        f"safetensors fallback — skipping")
                    continue
                t = t.detach().to(device="cpu")
            else:
                t = param_ref.detach().to(device="cpu")
            shape = list(t.shape)
            layout = classify_tensor(raw_name, shape, weight_prefix, layer_types)
            # For A_log (HeadExpReshaped): the HF model stores bf16(raw_A_log)
            # and computes exp() at runtime as exp(bf16(raw)). Keep that —
            # reading the raw F32 from safetensors here diverges from what the
            # live HF model uses, which breaks Python-vs-Rust equivalence.
            dtype_str = torch_dtype_to_str(t.dtype)
            b, final_shape, final_dtype = apply_layout(t, shape, layout, dtype_str)
            writer.write_tensor(raw_name, b, final_shape, final_dtype, layout)
            del t, b
    log(f"[bake-int4] done. Output: {out_dir}")


if __name__ == "__main__":
    main()
