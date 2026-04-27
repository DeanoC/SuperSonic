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
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

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
def is_int4_target(name: str) -> bool:
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
        width = group_view.shape[1]
        gv = group_view.reshape(scale_rows, gs, width)
        tmax = gv.amax(dim=(1, 2))  # [scale_rows]
        tmin = gv.amin(dim=(1, 2))
        rng = tmax - tmin
        sc = torch.where(rng > 0, rng / 15.0, torch.ones_like(rng))
        zf = torch.where(rng > 0, -tmin / sc, torch.zeros_like(rng))
        # Round through BF16 so the values we use to reconstruct weights match
        # exactly what the runtime kernel reads (scale/zero are stored BF16).
        sc = sc.to(torch.bfloat16).to(torch.float32)
        zf = zf.to(torch.bfloat16).to(torch.float32)
        scale_tile[:, gc] = sc
        zero_tile[:, gc] = zf

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
    """[rows, cols] uint8 0..15 -> [rows, cols/2] uint8 packed 2/byte."""
    rows, cols = nibbles.shape
    if cols % 2 != 0:
        raise ValueError(f"cols must be even, got {cols}")
    r = nibbles.reshape(rows, cols // 2, 2).to(torch.uint8)
    return (r[..., 0] | (r[..., 1] << 4)).contiguous()


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
def quantize_model(
    model: nn.Module,
    calib_ids: torch.Tensor,
    device: torch.device,
    group_size: int,
    damp: float,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Run sequential GPTQ. Returns {tensor_name: (nibbles, scale_f32, zero_f32)}
    where nibbles is [out, in] uint8 and scale/zero are [out/gs, in/gs] f32.
    """
    model.eval()
    inner = model.model
    layers = inner.layers
    num_layers = len(layers)

    # Map nn.Linear -> state-dict weight name (for naming output tensors)
    module_to_name: dict[int, str] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            module_to_name[id(mod)] = name + ".weight"

    log(f"[gptq] capturing layer-0 inputs from {calib_ids.shape[0]} samples...")
    hidden_cpu, layer_kwargs = capture_layer0_inputs(model, layers, calib_ids, device)
    layer_kwargs_dev = move_kwargs_to(layer_kwargs, device)

    quantized: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
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
            W = mod.weight.data.to(torch.float32)
            Q_dq, nibbles, scale_t, zero_t = gptq_quantize(W, H, group_size, damp)
            elapsed = time.perf_counter() - t0
            log(f"[gptq]   {name}: shape={tuple(W.shape)} "
                f"H_N={hook.N} took {elapsed:.1f}s")
            # Swap in dequantized weights so subsequent forwards see the quantized
            # version of this module.
            mod.weight.data.copy_(Q_dq.to(mod.weight.dtype))
            quantized[name] = (nibbles.cpu(), scale_t.cpu(), zero_t.cpu())
            del W, Q_dq, nibbles, scale_t, zero_t, H

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

    # --- lm_head GPTQ pass ---
    # The transformer layer loop above hooks every nn.Linear inside `inner.layers`,
    # but lm_head sits outside. Capture its Hessian here (post-final-norm hidden
    # state) and run the same column-wise GPTQ. This lets the runtime skip the
    # 250k×hidden BF16 read on the dominant decode-side matmul.
    final_norm = getattr(inner, "norm", None)
    lm_head = getattr(model, "lm_head", None)
    if (
        isinstance(lm_head, nn.Linear)
        and final_norm is not None
        and is_int4_target("lm_head.weight")
    ):
        log(f"[gptq] lm_head: collecting Hessian over {nsamples} samples")
        hook = HessianHook(lm_head)
        with torch.no_grad():
            for s in range(nsamples):
                hs = hidden_cpu[s].to(device)
                normed = final_norm(hs)
                _ = lm_head(normed)
                del hs, normed
            if device.type == "cuda":
                torch.cuda.empty_cache()
        H = hook.H
        hook.close()
        if H is None:
            log("[gptq]   lm_head: WARNING no activations captured, skipping")
        else:
            t0 = time.perf_counter()
            # The output head is huge on 9B-class checkpoints. Quantize it on
            # CPU to avoid requiring an extra multi-GiB clone on an already-full
            # producer GPU after the layer sweep.
            W = lm_head.weight.data.detach().to(device="cpu", dtype=torch.float32)
            H = H.detach().to(device="cpu", dtype=torch.float32)
            Q_dq, nibbles, scale_t, zero_t = gptq_quantize(W, H, group_size, damp)
            elapsed = time.perf_counter() - t0
            log(f"[gptq]   lm_head: shape={tuple(W.shape)} "
                f"H_N={hook.N} took {elapsed:.1f}s")
            copy_weight_in_row_chunks(lm_head.weight.data, Q_dq)
            quantized["lm_head.weight"] = (nibbles.cpu(), scale_t.cpu(), zero_t.cpu())
            del W, Q_dq, nibbles, scale_t, zero_t, H

    return quantized


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


def write_package(
    out_dir: Path,
    tensors: list[tuple[str, bytes, list[int], str, str]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    cursor = 0
    weights_path = out_dir / "weights.bin"
    with open(weights_path, "wb") as f:
        for (name, data, shape, dtype_str, layout) in tensors:
            offset = align_up(cursor, 4096)
            if offset > cursor:
                f.write(b"\x00" * (offset - cursor))
            f.write(data)
            byte_len = len(data)
            entries.append({
                "name": name,
                "shape": shape,
                "dtype": dtype_str,
                "layout": layout,
                "offset": offset,
                "byte_len": byte_len,
            })
            cursor = offset + byte_len
    manifest = {
        "format_version": FORMAT_VERSION,
        "converter_version": CONVERTER_VERSION,
        "model_family": "qwen35",
        "tensors": entries,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"[bake-int4] wrote {cursor / (1024 * 1024):.1f} MiB to {weights_path}")
    log(f"[bake-int4] manifest: {out_dir / 'manifest.json'}")


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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"[bake-int4] loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    log(f"[bake-int4] loading model (bf16) from {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # --- Determine the canonical tensor-name prefix by reading the raw
    # safetensors key set. HuggingFace's AutoModelForCausalLM may flatten
    # nested modules (e.g. drop ".language_model." for Qwen3.5), so the
    # state_dict names don't always match what the Rust loader expects.
    raw_keys = _load_raw_tensor_names(model_dir)
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

    sd_keys = list(model.state_dict().keys())
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

    # --- GPTQ ---
    t0 = time.perf_counter()
    quantized = quantize_model(
        model, calib, device,
        group_size=args.group_size,
        damp=args.damp,
    )
    gptq_elapsed = time.perf_counter() - t0
    log(f"[bake-int4] GPTQ done in {gptq_elapsed / 60.0:.1f} min "
        f"({len(quantized)} tensors quantized)")

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

    # --- Self-consistency check: reconstruct EVERY INT4 tensor from the
    # just-computed (nibbles, scale, zero) and verify it matches the value
    # copied into the live model's Linear weight. Any mismatch indicates a
    # bake-vs-runtime format bug (not a quantization quality issue).
    try:
        gs = args.group_size
        hf_modules = dict(model.named_modules())
        mismatch = 0
        worst: tuple[str, float] = ("", 0.0)
        for hf_name, (nibbles_s, scale_s, zero_s) in quantized.items():
            rows, cols = nibbles_s.shape
            row_gr = torch.arange(rows) // gs
            col_gc = torch.arange(cols) // gs
            sc_full = scale_s[row_gr][:, col_gc]
            zf_full = zero_s[row_gr][:, col_gc]
            recon = nibbles_s.float() * sc_full - zf_full * sc_full
            recon = recon.to(torch.bfloat16).to(torch.float32)
            mod = hf_modules.get(hf_name[: -len(".weight")])
            if mod is None:
                continue
            live = mod.weight.data.to(torch.float32).cpu()
            if not torch.equal(recon, live):
                mismatch += 1
                linf = (recon - live).abs().max().item()
                if linf > worst[1]:
                    worst = (hf_name, linf)
        log(f"[self-check] INT4 tensors: {mismatch}/{len(quantized)} mismatch"
            + (f" (worst: {worst[0]} Linf={worst[1]:.2e})" if mismatch else ""))
    except Exception as ex:
        log(f"[self-check] failed: {ex}")

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

    # --- Assemble tensors for writing ---
    log("[bake-int4] serialising tensors...")
    # Walk the HF state dict but emit under raw-safetensors names so the Rust
    # loader (which expects the raw layout, e.g. "model.language_model.X")
    # finds every tensor.
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

    sd = model.state_dict()
    # When `lm_head.weight` is tied to `embed_tokens.weight`, the HF state dict
    # entry usually has no safetensors counterpart and the eligible loop above
    # skips it. But if we just ran GPTQ on `model.lm_head` and produced a
    # quantized tensor for it, force-include `lm_head.weight` so the runtime
    # loads the INT4 version instead of falling back to the BF16 embed alias.
    if "lm_head.weight" in quantized and "lm_head.weight" not in eligible:
        eligible.append("lm_head.weight")
        if "lm_head.weight" not in hf_to_raw:
            hf_to_raw["lm_head.weight"] = "lm_head.weight"
        if "lm_head.weight" not in sd and hasattr(model, "lm_head"):
            sd["lm_head.weight"] = model.lm_head.weight.data
    tensors_out: list[tuple[str, bytes, list[int], str, str]] = []
    for hf_name in eligible:
        raw_name = hf_to_raw[hf_name]
        t = sd[hf_name]
        shape = list(t.shape)
        if hf_name in quantized:
            nibbles, scale_t, zero_t = quantized[hf_name]
            packed = pack_nibbles(nibbles)
            packed_bytes = packed.numpy().tobytes()
            tensors_out.append((
                raw_name, packed_bytes,
                [packed.shape[0], packed.shape[1]],
                "u8", LAYOUT_INT4,
            ))
            tensors_out.append((
                f"{raw_name}_int4_scale",
                bf16_to_bytes(scale_t),
                list(scale_t.shape), "bf16", LAYOUT_RAW,
            ))
            tensors_out.append((
                f"{raw_name}_int4_zero",
                bf16_to_bytes(zero_t),
                list(zero_t.shape), "bf16", LAYOUT_RAW,
            ))
        else:
            layout = classify_tensor(raw_name, shape, weight_prefix, layer_types)
            # For A_log (HeadExpReshaped): the HF model stores bf16(raw_A_log)
            # and computes exp() at runtime as exp(bf16(raw)). Keep that —
            # reading the raw F32 from safetensors here diverges from what the
            # live HF model uses, which breaks Python-vs-Rust equivalence.
            dtype_str = torch_dtype_to_str(t.dtype)
            b, final_shape, final_dtype = apply_layout(t, shape, layout, dtype_str)
            tensors_out.append((raw_name, b, final_shape, final_dtype, layout))

    tensors_out.sort(key=lambda x: x[0])

    out_dir = args.out_dir or (
        model_dir / ".supersonic" / f"v{FORMAT_VERSION}-int4-gptq"
    )
    write_package(out_dir, tensors_out)
    log(f"[bake-int4] done. Output: {out_dir}")


if __name__ == "__main__":
    main()
