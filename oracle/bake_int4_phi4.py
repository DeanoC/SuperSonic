#!/usr/bin/env python3
"""
GPTQ-style INT4 calibration bake for SuperSonic — Phi-4-mini.

Reads a HuggingFace Phi-4-mini checkpoint, runs GPTQ calibration against
WikiText-2, and writes a SuperSonic baked package at
  {model-dir}/.supersonic/v{FORMAT_VERSION}-int4-gptq/

Mirrors `oracle/bake_int4.py` (Qwen3.5) but adapted for Phi-4's two
fused-projection tensors:
  - `model.layers.{i}.self_attn.qkv_proj.weight` -> split into q/k/v
  - `model.layers.{i}.mlp.gate_up_proj.weight`   -> split into gate/up

The fused tensors are quantized as a single Linear (so the Hessian comes from
the same input that all sub-projections see — equivalent to GPTQ-ing each
sub-projection with an identical H), then the resulting (nibbles, scale, zero)
tensors are sliced row-wise into per-projection shards. The split lines up
cleanly with the group_size=128 tile boundary because Phi-4-mini's
projection sizes (q=3072, kv=1024, intermediate=8192) are all multiples of
128.

This produces output names matching `Phi4Weights::load_baked`'s expectations
(`*.self_attn.q_proj.weight`, etc.), so the runtime `--int4` path picks the
bake up automatically.

Usage:
    python3 bake_int4_phi4.py --model-dir /path/to/Phi-4-mini-instruct
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# -- constants mirrored from crates/model-store/src/manifest.rs --
FORMAT_VERSION = 2
CONVERTER_VERSION = 1

# LayoutTag strings must match the Rust enum variants exactly.
LAYOUT_RAW = "Raw"
LAYOUT_INT4 = "Int4Quantized"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Target-tensor selection. Phi-4 has fused qkv_proj + gate_up_proj plus
# non-fused o_proj + down_proj. All four are INT4-quantized; the two fused
# ones are sliced into sub-projections after quantization.
# ---------------------------------------------------------------------------
def is_int4_target(name: str) -> bool:
    if not name.endswith(".weight"):
        return False
    if "layernorm" in name or "norm.weight" in name or "embed_tokens" in name:
        return False
    if "lm_head" in name:
        # Phi4 ties lm_head to embed_tokens at the runtime; the Phi4 weights
        # crate has no INT4 lm_head dispatch yet, so leave it BF16 here.
        return False
    return "_proj" in name


# ---------------------------------------------------------------------------
# Byte encoding helpers
# ---------------------------------------------------------------------------
def align_up(x: int, a: int) -> int:
    return (x + a - 1) & ~(a - 1)


def bf16_to_bytes(t: torch.Tensor) -> bytes:
    t = t.to(torch.bfloat16).contiguous().cpu()
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
    raise ValueError(f"unsupported dtype_str {dtype_str}")


def torch_dtype_to_str(dt: torch.dtype) -> str:
    return {
        torch.bfloat16: "bf16",
        torch.float32: "f32",
        torch.float16: "f16",
        torch.uint8: "u8",
    }.get(dt, "bf16")


# ---------------------------------------------------------------------------
# GPTQ core (copied verbatim from bake_int4.py — algorithm is family-agnostic).
# ---------------------------------------------------------------------------
@torch.no_grad()
def gptq_quantize(
    W: torch.Tensor,
    H: torch.Tensor,
    group_size: int,
    damp: float,
    blocksize: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out_f, in_f = W.shape
    device = W.device
    gs = group_size
    if in_f % gs != 0:
        raise ValueError(f"in_features {in_f} must be a multiple of group_size {gs}")
    if out_f % gs != 0:
        raise ValueError(f"out_features {out_f} must be a multiple of group_size {gs}")

    scale_rows = out_f // gs
    scale_cols = in_f // gs
    row_to_gr = torch.arange(out_f, device=device) // gs

    H = H.clone().to(torch.float32)
    diag_mean = torch.mean(torch.diagonal(H)).item()
    if not math.isfinite(diag_mean) or diag_mean <= 0.0:
        H = torch.eye(in_f, device=device, dtype=torch.float32)
        diag_mean = 1.0
    d_idx = torch.arange(in_f, device=device)
    H[d_idx, d_idx] += damp * max(diag_mean, 1e-8)

    dead = torch.diag(H) < 1e-10
    if dead.any():
        W[:, dead] = 0.0
        H[dead, dead] = 1.0

    try:
        L = torch.linalg.cholesky(H)
    except Exception:
        H[d_idx, d_idx] += 10.0 * damp * max(diag_mean, 1e-8)
        L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    W = W.clone().to(torch.float32)
    nibbles = torch.zeros((out_f, in_f), dtype=torch.uint8, device=device)
    Q_dq = torch.zeros_like(W)
    scale_tile = torch.zeros((scale_rows, scale_cols), dtype=torch.float32, device=device)
    zero_tile = torch.zeros((scale_rows, scale_cols), dtype=torch.float32, device=device)

    def set_tile_scales(gc: int, group_view: torch.Tensor) -> None:
        width = group_view.shape[1]
        gv = group_view.reshape(scale_rows, gs, width)
        tmax = gv.amax(dim=(1, 2))
        tmin = gv.amin(dim=(1, 2))
        rng = tmax - tmin
        sc = torch.where(rng > 0, rng / 15.0, torch.ones_like(rng))
        zf = torch.where(rng > 0, -tmin / sc, torch.zeros_like(rng))
        sc = sc.to(torch.bfloat16).to(torch.float32)
        zf = zf.to(torch.bfloat16).to(torch.float32)
        scale_tile[:, gc] = sc
        zero_tile[:, gc] = zf

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
            sc_col = scale_tile[row_to_gr, gc]
            zf_col = zero_tile[row_to_gr, gc]
            q = torch.clamp(torch.round(w_col / sc_col + zf_col), 0.0, 15.0)
            q_int = q.to(torch.uint8)
            q_dq = (q * sc_col - zf_col * sc_col)
            q_dq = q_dq.to(torch.bfloat16).to(torch.float32)

            nibbles[:, col] = q_int
            Q_dq[:, col] = q_dq
            Q1[:, i] = q_dq
            err = (w_col - q_dq) / d_ii
            if i + 1 < count:
                W1[:, i + 1:] -= err.unsqueeze(1) * Hinv1[i, i + 1:]
            Err1[:, i] = err

        if e < in_f:
            W[:, e:] -= Err1 @ Hinv[b:e, e:]
        W[:, b:e] = Q1

    return Q_dq, nibbles, scale_tile, zero_tile


def pack_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    rows, cols = nibbles.shape
    if cols % 2 != 0:
        raise ValueError(f"cols must be even, got {cols}")
    r = nibbles.reshape(rows, cols // 2, 2).to(torch.uint8)
    return (r[..., 0] | (r[..., 1] << 4)).contiguous()


# ---------------------------------------------------------------------------
# Activation capture — identical to Qwen baker.
# ---------------------------------------------------------------------------
class HessianHook:
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


def capture_layer0_inputs(model, layers, calib_ids, device):
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
    assert len(catcher.hiddens) == nsamples
    assert catcher.kwargs is not None
    return catcher.hiddens, catcher.kwargs


def quantize_model(model, calib_ids, device, group_size, damp):
    model.eval()
    inner = model.model
    layers = inner.layers
    num_layers = len(layers)

    module_to_name: dict[int, str] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            module_to_name[id(mod)] = name + ".weight"

    log(f"[gptq] capturing layer-0 inputs from {calib_ids.shape[0]} samples...")
    hidden_cpu, layer_kwargs = capture_layer0_inputs(model, layers, calib_ids, device)
    layer_kwargs_dev = {k: _move_tree(v, device) for k, v in layer_kwargs.items()}

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
            mod.weight.data.copy_(Q_dq.to(mod.weight.dtype))
            quantized[name] = (nibbles.cpu(), scale_t.cpu(), zero_t.cpu())
            del W, Q_dq, nibbles, scale_t, zero_t, H

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

    return quantized


# ---------------------------------------------------------------------------
# Phi-4 fused-tensor splitting. Operates on the post-GPTQ
# (nibbles, scale, zero) triples and fans them out into per-projection shards
# whose names match `Phi4Weights::load_baked`'s expectations.
# ---------------------------------------------------------------------------
def split_phi4_quantized(
    quantized: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    q_rows: int,
    k_rows: int,
    intermediate_size: int,
    group_size: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Replace fused entries with their per-projection slices. Original keys
    pointing to `qkv_proj.weight` and `gate_up_proj.weight` are removed; new
    keys pointing to `q_proj.weight`, `k_proj.weight`, `v_proj.weight`,
    `gate_proj.weight`, `up_proj.weight` are added.

    Slicing is row-wise on the OUTPUT-feature axis, which is the leading
    dim of all three tensors:
      - nibbles: [out, in/2]  -> [out_slice, in/2]
      - scale:   [out/gs, in/gs] -> [out_slice/gs, in/gs]
      - zero:    same shape as scale.
    """
    gs = group_size
    out: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    for name, (nib, sc, zr) in list(quantized.items()):
        if name.endswith(".self_attn.qkv_proj.weight"):
            prefix = name[: -len(".self_attn.qkv_proj.weight")]
            v_rows = k_rows
            total = q_rows + k_rows + v_rows
            assert nib.shape[0] == total, (
                f"qkv_proj rows {nib.shape[0]} != q+k+v {total}"
            )
            assert q_rows % gs == 0 and k_rows % gs == 0, (
                "q_rows and k_rows must be multiples of group_size for clean split"
            )
            slices = [
                ("q_proj", 0, q_rows),
                ("k_proj", q_rows, q_rows + k_rows),
                ("v_proj", q_rows + k_rows, total),
            ]
            for sub, lo, hi in slices:
                sub_name = f"{prefix}.self_attn.{sub}.weight"
                out[sub_name] = (
                    nib[lo:hi, :].contiguous(),
                    sc[lo // gs: hi // gs, :].contiguous(),
                    zr[lo // gs: hi // gs, :].contiguous(),
                )
        elif name.endswith(".mlp.gate_up_proj.weight"):
            prefix = name[: -len(".mlp.gate_up_proj.weight")]
            total = 2 * intermediate_size
            assert nib.shape[0] == total, (
                f"gate_up_proj rows {nib.shape[0]} != 2*intermediate {total}"
            )
            assert intermediate_size % gs == 0, (
                "intermediate_size must be a multiple of group_size for clean split"
            )
            slices = [
                ("gate_proj", 0, intermediate_size),
                ("up_proj", intermediate_size, total),
            ]
            for sub, lo, hi in slices:
                sub_name = f"{prefix}.mlp.{sub}.weight"
                out[sub_name] = (
                    nib[lo:hi, :].contiguous(),
                    sc[lo // gs: hi // gs, :].contiguous(),
                    zr[lo // gs: hi // gs, :].contiguous(),
                )
        else:
            out[name] = (nib, sc, zr)
    return out


# ---------------------------------------------------------------------------
# Manifest writer
# ---------------------------------------------------------------------------
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
        "model_family": "phi4",
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
    p = argparse.ArgumentParser(description="Phi-4 GPTQ INT4 calibration bake")
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--damp", type=float, default=0.01)
    p.add_argument("--device", default=None,
                   help="Torch device (default: cuda if available else cpu)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--ppl-chunks", type=int, default=8)
    p.add_argument("--out-dir", default=None, type=Path,
                   help="Override output dir (default: "
                        "{model-dir}/.supersonic/v{FORMAT_VERSION}-int4-gptq)")
    return p.parse_args()


@torch.no_grad()
def compute_ppl(model, tokenizer, device, seqlen, n_chunks):
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
        n = chunk.numel() - 1
        nll_sum += out.loss.float().item() * n
        tok_sum += n
    return math.exp(nll_sum / tok_sum)


def main() -> None:
    args = parse_args()
    model_dir: Path = args.model_dir
    if not model_dir.exists():
        raise SystemExit(f"model dir does not exist: {model_dir}")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[bake-int4] device={device}")
    if device.type == "cpu":
        log("[bake-int4] WARNING: GPTQ on CPU for a 3.8B model will be slow.")

    # Don't trust_remote_code — Phi-4-mini's bundled `modeling_phi3.py` imports
    # `LossKwargs` which was removed in transformers 5.x. Built-in Phi3ForCausalLM
    # is functionally equivalent for inference.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"[bake-int4] loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    log(f"[bake-int4] loading model (bf16) from {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    config = model.config
    text_cfg = getattr(config, "text_config", config)
    num_layers = int(text_cfg.num_hidden_layers)
    num_attention_heads = int(text_cfg.num_attention_heads)
    num_key_value_heads = int(text_cfg.num_key_value_heads)
    head_dim = int(text_cfg.hidden_size // num_attention_heads)
    intermediate_size = int(text_cfg.intermediate_size)
    q_rows = num_attention_heads * head_dim
    k_rows = num_key_value_heads * head_dim

    log(f"[bake-int4] phi4: layers={num_layers} q_rows={q_rows} "
        f"k_rows={k_rows} intermediate={intermediate_size}")

    # Calibration data
    log("[bake-int4] loading WikiText-2 train split via `datasets`...")
    from datasets import load_dataset
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(r["text"] for r in train if r["text"].strip())
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[0]
    log(f"[bake-int4] tokenized train: {ids.numel()} tokens")
    if ids.numel() < args.seqlen * 2:
        raise SystemExit(
            f"not enough calibration tokens ({ids.numel()}) for seqlen={args.seqlen}"
        )
    torch.manual_seed(args.seed)
    max_start = ids.numel() - args.seqlen - 1
    starts = torch.randint(0, max_start, (args.num_samples,))
    calib = torch.stack([ids[s:s + args.seqlen] for s in starts])
    log(f"[bake-int4] calibration batch: {tuple(calib.shape)}")

    # GPTQ
    t0 = time.perf_counter()
    quantized = quantize_model(model, calib, device, args.group_size, args.damp)
    gptq_elapsed = time.perf_counter() - t0
    log(f"[bake-int4] GPTQ done in {gptq_elapsed / 60.0:.1f} min "
        f"({len(quantized)} fused tensors quantized)")

    # Sample generation sanity check
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

    # Perplexity sanity check
    if not args.skip_ppl:
        log("[bake-int4] running perplexity sanity check on WikiText-2 test...")
        try:
            ppl = compute_ppl(model, tokenizer, device,
                              seqlen=args.seqlen, n_chunks=args.ppl_chunks)
            log(f"[bake-int4] PPL (WikiText-2 test, {args.ppl_chunks} chunks): "
                f"{ppl:.2f}")
        except Exception as ex:
            log(f"[bake-int4] PPL check failed: {ex}")

    # Split fused tensors -> per-projection shards.
    quantized_split = split_phi4_quantized(
        quantized, q_rows, k_rows, intermediate_size, args.group_size,
    )
    log(f"[bake-int4] split {len(quantized)} fused → {len(quantized_split)} shards")

    # Assemble output tensors. Walk the model state dict for non-quantized
    # tensors (norms, embed_tokens) and emit them raw.
    log("[bake-int4] serialising tensors...")
    sd = model.state_dict()
    tensors_out: list[tuple[str, bytes, list[int], str, str]] = []

    # 1. Quantized projections (split).
    for shard_name, (nibbles, scale_t, zero_t) in quantized_split.items():
        packed = pack_nibbles(nibbles)
        tensors_out.append((
            shard_name,
            packed.numpy().tobytes(),
            [packed.shape[0], packed.shape[1]],
            "u8", LAYOUT_INT4,
        ))
        tensors_out.append((
            f"{shard_name}_int4_scale",
            bf16_to_bytes(scale_t),
            list(scale_t.shape), "bf16", LAYOUT_RAW,
        ))
        tensors_out.append((
            f"{shard_name}_int4_zero",
            bf16_to_bytes(zero_t),
            list(zero_t.shape), "bf16", LAYOUT_RAW,
        ))

    # 2. Non-quantized tensors (norms, embed) — emit raw BF16.
    skip_substrings = (
        ".qkv_proj.weight",
        ".gate_up_proj.weight",
        ".q_proj.weight", ".k_proj.weight", ".v_proj.weight",
        ".o_proj.weight",
        ".gate_proj.weight", ".up_proj.weight", ".down_proj.weight",
    )
    for name, t in sd.items():
        if name == "lm_head.weight":
            # tied to embed_tokens; the runtime aliases.
            continue
        if any(name.endswith(s) for s in skip_substrings):
            continue
        if not (name.startswith("model.") or name == "lm_head.weight"):
            continue
        shape = list(t.shape)
        dtype_str = torch_dtype_to_str(t.dtype)
        tensors_out.append((
            name,
            tensor_to_bytes(t, dtype_str),
            shape, dtype_str, LAYOUT_RAW,
        ))

    tensors_out.sort(key=lambda x: x[0])

    # Sanity: each layer must have q/k/v/o/gate/up/down + their scales/zeros.
    needed_per_layer = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    names_set = {t[0] for t in tensors_out}
    for layer_idx in range(num_layers):
        for needed in needed_per_layer:
            full = f"model.layers.{layer_idx}.{needed}"
            for suffix in ("", "_int4_scale", "_int4_zero"):
                key = full + suffix
                if key not in names_set:
                    raise SystemExit(f"[bake-int4] missing output tensor: {key}")

    out_dir = args.out_dir or (
        model_dir / ".supersonic" / f"v{FORMAT_VERSION}-int4-gptq"
    )
    write_package(out_dir, tensors_out)
    log(f"[bake-int4] done. Output: {out_dir}")


if __name__ == "__main__":
    main()
