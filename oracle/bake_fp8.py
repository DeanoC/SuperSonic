#!/usr/bin/env python3
"""
Per-block FP8-E4M3 calibration bake for SuperSonic — Qwen3.5.

Quantizes BF16 projection weights to FP8-E4M3-FN with per-block (128×128)
absmax scales, writing a SuperSonic baked package at
  {model-dir}/.supersonic/v{FORMAT_VERSION}-fp8/

Mirrors the Qwen 3.6 native FP8 layout: each `*_proj.weight` becomes
  - `name`            : u8  [rows, cols]            FP8-E4M3 bytes (Fp8Native)
  - `name_scale_inv`  : bf16 [rows/128, cols/128]   per-block scale (Raw)

so the existing Rust runtime FP8-dequant path (used today for Qwen 3.6
FP8-native checkpoints) reads it without any new code on the C++ / kernel
side. Non-projection tensors (norms, embeddings, conv1d, dt_bias, A_log)
get the same layout transforms the Rust BF16 baker applies — squeezed
conv1d, reshaped dt_bias, A_log → exp(A_log) → BF16.

Usage:
    python3 oracle/bake_fp8.py --model-dir /path/to/Qwen3.5-0.8B
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

# -- constants mirrored from crates/model-store/src/manifest.rs --
FORMAT_VERSION = 2
CONVERTER_VERSION = 1

LAYOUT_RAW = "Raw"
LAYOUT_FP8_NATIVE = "Fp8Native"
LAYOUT_DEPTHWISE_CONV_SQUEEZED = "DepthwiseConvSqueezed"
LAYOUT_HEAD_BIAS_RESHAPED = "HeadBiasReshaped"
LAYOUT_HEAD_EXP_RESHAPED = "HeadExpReshaped"

ALIGN = 4096
BLOCK_SIZE = 128
MAX_E4M3 = 448.0  # finite max of float8_e4m3fn

LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")


def log(msg: str) -> None:
    print(msg, flush=True)


def align_up(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


# ---------------------------------------------------------------------------
# Tensor classification
# ---------------------------------------------------------------------------
def is_fp8_quant_target(name: str, shape: tuple[int, ...]) -> bool:
    """True if this 2D weight should be quantized to FP8.

    Match every per-layer projection by looking for `_proj` anywhere in the
    leaf module name. Covers:

      * `gate_proj` / `up_proj` / `down_proj`     (MLP)
      * `q_proj` / `k_proj` / `v_proj` / `o_proj` (full attention)
      * `out_proj`                                (linear-attn output)
      * `in_proj_qkv` / `in_proj_z`               (linear-attn input)
      * `in_proj_a` / `in_proj_b`                 (linear-attn input;
                                                   filtered by the shape
                                                   divisibility check below
                                                   — Qwen 3.5 stores those
                                                   at `[16, hidden]`, 16 < 128)

    The previous check `"_proj.weight" in name` matched only names where
    `_proj` was the immediate parent of `.weight`, so every linear-attention
    `in_proj_qkv.weight` / `in_proj_z.weight` was silently left BF16,
    leaving the bake only partially quantized for the linear layers (about
    a third of the tensors in a Qwen 3.5 hybrid model).

    Skip everything else: norms, embeddings, conv1d / dt_bias / A_log
    transforms, layer scalars, rotary buffers. **Also skip `lm_head.weight`**
    — the runtime does not currently consume `lm_head_scale` (decode lm_head
    matmul is BF16-only across `decode_engine.rs` and `compute_logits_for_range`),
    so quantizing it would feed FP8 bytes into a BF16 matmul and either
    corrupt logits or page-fault. Tied-embedding checkpoints (most Qwen3.5
    sizes) don't have a standalone `lm_head.weight` so this never mattered;
    Qwen3.5-9B does, and was producing GPU memory faults until this exclusion
    landed.
    """
    if len(shape) != 2:
        return False
    if not name.endswith(".weight"):
        return False
    # Embedding / lm_head tables — skip until runtime lm_head FP8 support
    # is wired (runtime currently only knows BF16 lm_head).
    if "embed_tokens" in name or name == "lm_head.weight":
        return False
    # Norms (caught by 1D check above mostly, but be explicit).
    if "layernorm" in name or name.endswith(".norm.weight"):
        return False
    if "_proj" not in name:
        return False
    return shape[0] % BLOCK_SIZE == 0 and shape[1] % BLOCK_SIZE == 0


def linear_attn_layer_indices(config_path: Path) -> set[int]:
    """Return the set of linear-attention layer indices for Qwen3.5.

    Qwen3.5 alternates: full attention every 4th layer starting at index 3.
    Reads the model's config.json to compute this; falls back to the
    standard `(idx + 1) % 4 != 0` rule if the config doesn't carry the
    attention pattern explicitly.
    """
    cfg = json.load(open(config_path))
    text = cfg.get("text_config", cfg)
    n = int(text["num_hidden_layers"])
    pattern = text.get("attention_pattern") or text.get("layer_types")
    if isinstance(pattern, list) and len(pattern) == n:
        return {
            i for i, t in enumerate(pattern)
            if "linear" in str(t).lower() or "ssm" in str(t).lower()
        }
    full_attn_period = int(text.get("full_attn_period", 4))
    return {i for i in range(n) if (i + 1) % full_attn_period != 0}


def classify_layout(name: str, shape: tuple[int, ...], linear_layer_idx: set[int]) -> str:
    """Return the layout tag a tensor should have in the bake.

    Mirrors `classify_tensor` in crates/model-store/src/baker.rs. Only
    linear-attention layers' conv1d/dt_bias/A_log get reshape transforms;
    everything else is `Raw`.
    """
    m = LAYER_RE.match(name)
    if not m:
        return LAYOUT_RAW
    idx = int(m.group(1))
    if idx not in linear_layer_idx:
        return LAYOUT_RAW
    if name.endswith(".conv1d.weight") and len(shape) == 3 and shape[1] == 1:
        return LAYOUT_DEPTHWISE_CONV_SQUEEZED
    if name.endswith(".dt_bias") and len(shape) == 1:
        return LAYOUT_HEAD_BIAS_RESHAPED
    if name.endswith(".A_log") and len(shape) == 1:
        return LAYOUT_HEAD_EXP_RESHAPED
    return LAYOUT_RAW


# ---------------------------------------------------------------------------
# FP8 quantization
# ---------------------------------------------------------------------------
def quantize_bf16_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [rows, cols] → (FP8-E4M3 [rows, cols] uint8, scale [rows/128, cols/128] BF16).

    Per-block (128×128) absmax: scale = absmax / MAX_E4M3, then
    quantized_byte = round_to_e4m3(value / scale). The kernel reads back as
    `value ≈ fp8_to_f32(byte) * bf16_to_f32(scale)`.
    """
    assert weight.dim() == 2, f"expected 2D, got {tuple(weight.shape)}"
    rows, cols = weight.shape
    assert rows % BLOCK_SIZE == 0 and cols % BLOCK_SIZE == 0, \
        f"shape {tuple(weight.shape)} not divisible by {BLOCK_SIZE}"

    w_f32 = weight.float()
    # Reshape into [rows/128, 128, cols/128, 128] and absmax over the inner
    # (1, 3) dims to get [rows/128, cols/128].
    sr = rows // BLOCK_SIZE
    sc = cols // BLOCK_SIZE
    blocks = w_f32.view(sr, BLOCK_SIZE, sc, BLOCK_SIZE).permute(0, 2, 1, 3).contiguous()
    # blocks: [sr, sc, 128, 128]
    absmax = blocks.abs().amax(dim=(2, 3))  # [sr, sc]
    # Avoid /0 for all-zero blocks.
    scale = (absmax / MAX_E4M3).clamp_min(torch.finfo(torch.float32).tiny)
    # Per-element scale matrix [rows, cols] via repeat-interleave.
    scale_full = (
        scale.repeat_interleave(BLOCK_SIZE, dim=0)
              .repeat_interleave(BLOCK_SIZE, dim=1)
    )
    # Divide by scale, cast to E4M3 (rounds + clamps to representable range),
    # reinterpret bytes as uint8.
    q_e4m3 = (w_f32 / scale_full).to(torch.float8_e4m3fn)
    q_bytes = q_e4m3.view(torch.uint8)
    return q_bytes.contiguous(), scale.to(torch.bfloat16).contiguous()


# ---------------------------------------------------------------------------
# Layout transforms (1:1 with crates/model-store/src/transforms.rs)
# ---------------------------------------------------------------------------
def apply_squeeze_dim1(t: torch.Tensor) -> torch.Tensor:
    assert t.dim() == 3 and t.shape[1] == 1, \
        f"squeeze_dim1: expected [C, 1, K], got {tuple(t.shape)}"
    return t.squeeze(1).contiguous()


def apply_head_bias_reshape(t: torch.Tensor) -> torch.Tensor:
    assert t.dim() == 1, f"head_bias_reshape: expected [H], got {tuple(t.shape)}"
    return t.view(1, 1, -1).contiguous()


def apply_a_log_to_exp_bf16(t: torch.Tensor) -> torch.Tensor:
    assert t.dim() == 1, f"a_log_to_exp_bf16: expected [H], got {tuple(t.shape)}"
    return t.float().exp().to(torch.bfloat16).view(1, 1, -1).contiguous()


# ---------------------------------------------------------------------------
# Manifest dtype names (mirror dtype_name in baker.rs)
# ---------------------------------------------------------------------------
def torch_dtype_to_str(dt: torch.dtype) -> str:
    if dt == torch.bfloat16:
        return "bf16"
    if dt == torch.float32:
        return "f32"
    if dt == torch.float16:
        return "f16"
    if dt == torch.uint8:
        return "u8"
    if dt == torch.float8_e4m3fn:
        return "f8_e4m3"
    raise ValueError(f"unsupported dtype: {dt}")


def tensor_to_bytes(t: torch.Tensor, dtype_str: str) -> bytes:
    """Return the LE byte representation of `t`, validating the requested dtype.

    f8_e4m3 accepts either a `torch.uint8` reinterpret (what `quantize_bf16_to_fp8`
    produces) or a native `torch.float8_e4m3fn` tensor — viewed as uint8 before
    `numpy.tobytes()` because numpy doesn't know about float8.
    """
    if dtype_str == "f8_e4m3":
        if t.dtype == torch.float8_e4m3fn:
            t = t.view(torch.uint8)
        elif t.dtype != torch.uint8:
            raise AssertionError(
                f"f8_e4m3 must be uint8 or float8_e4m3fn, got {t.dtype}"
            )
        return t.contiguous().cpu().numpy().tobytes()
    expected = {"bf16": torch.bfloat16, "f32": torch.float32, "f16": torch.float16, "u8": torch.uint8}[dtype_str]
    if t.dtype != expected:
        t = t.to(expected)
    return t.contiguous().cpu().view(torch.uint8).numpy().tobytes()


# ---------------------------------------------------------------------------
# Streaming writer
# ---------------------------------------------------------------------------
class StreamingTensorWriter:
    def __init__(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.weights_path = out_dir / "weights.bin"
        self.f = open(self.weights_path, "wb")
        self.entries: list[dict] = []
        self.cursor = 0

    def add(self, name: str, data: bytes, shape: list[int], dtype_str: str, layout: str) -> None:
        offset = align_up(self.cursor)
        if offset > self.cursor:
            self.f.write(b"\x00" * (offset - self.cursor))
        self.f.write(data)
        self.entries.append({
            "name": name,
            "shape": shape,
            "dtype": dtype_str,
            "layout": layout,
            "offset": offset,
            "byte_len": len(data),
        })
        self.cursor = offset + len(data)

    def close(self) -> None:
        self.f.close()
        sorted_entries = sorted(self.entries, key=lambda e: e["name"])
        manifest = {
            "format_version": FORMAT_VERSION,
            "converter_version": CONVERTER_VERSION,
            "model_family": "qwen35",
            "tensors": sorted_entries,
        }
        with open(self.out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        log(f"[bake-fp8] wrote {self.cursor / (1024 * 1024):.1f} MiB to {self.weights_path}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def collect_tensor_index(model_dir: Path) -> tuple[list[Path], dict[str, int]]:
    """Return (shards, name → shard_idx) over every .safetensors in model_dir."""
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no .safetensors found in {model_dir}")
    index: dict[str, int] = {}
    for i, sf in enumerate(shards):
        with safe_open(str(sf), framework="pt") as f:
            for key in f.keys():
                index[key] = i
    return shards, index


def main() -> None:
    ap = argparse.ArgumentParser(description="Per-block FP8 bake for Qwen3.5")
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default=None, type=Path,
                    help="Override output dir (default: {model-dir}/.supersonic/v{FORMAT_VERSION}-fp8)")
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    args = ap.parse_args()

    if args.block_size != BLOCK_SIZE:
        log(f"[bake-fp8] WARNING: --block-size={args.block_size} is non-default; "
            "the Rust runtime currently assumes 128.")

    model_dir = args.model_dir.resolve()
    out_dir = args.out_dir or (model_dir / ".supersonic" / f"v{FORMAT_VERSION}-fp8")

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"missing config.json under {model_dir}")
    linear_idx = linear_attn_layer_indices(config_path)
    log(f"[bake-fp8] linear-attn layer indices ({len(linear_idx)}): "
        f"{sorted(linear_idx)[:8]}{' ...' if len(linear_idx) > 8 else ''}")

    shards, index = collect_tensor_index(model_dir)
    log(f"[bake-fp8] {len(shards)} shard(s), {len(index)} tensors")

    # Filter to language-model tensors + lm_head; sort for deterministic output.
    weight_prefix = "model.language_model"
    eligible = sorted(
        n for n in index
        if n.startswith(f"{weight_prefix}.") or n == "lm_head.weight"
    )
    log(f"[bake-fp8] {len(eligible)} eligible tensors")

    writer = StreamingTensorWriter(out_dir)
    open_shards: dict[int, Any] = {}

    def get_handle(idx: int):
        h = open_shards.get(idx)
        if h is None:
            h = safe_open(str(shards[idx]), framework="pt")
            h.__enter__()
            open_shards[idx] = h
        return h

    t0 = time.perf_counter()
    n_quant = 0
    n_skip = 0
    try:
        for name in eligible:
            shard_idx = index[name]
            handle = get_handle(shard_idx)
            t = handle.get_tensor(name)
            shape = tuple(t.shape)

            # FP8 quantization / pass-through candidate.
            if is_fp8_quant_target(name, shape):
                if t.dtype == torch.bfloat16:
                    # BF16 source → quantize to E4M3 with per-block scales.
                    packed, scale_bf16 = quantize_bf16_to_fp8(t)
                    writer.add(
                        name,
                        tensor_to_bytes(packed, "f8_e4m3"),
                        list(shape),
                        "f8_e4m3",
                        LAYOUT_FP8_NATIVE,
                    )
                    writer.add(
                        f"{name}_scale_inv",
                        tensor_to_bytes(scale_bf16, "bf16"),
                        list(scale_bf16.shape),
                        "bf16",
                        LAYOUT_RAW,
                    )
                    n_quant += 1
                    continue
                if t.dtype == torch.float8_e4m3fn:
                    # FP8-native source (e.g. Qwen 3.6 *-FP8 checkpoints) —
                    # store the bytes as-is with the Fp8Native layout tag so
                    # `upload_bake.py --fp8-native` accepts the bake. The
                    # companion `_scale_inv` tensor is in source safetensors
                    # already and falls through to the raw write path below.
                    fp8_bytes = t.contiguous().cpu().view(torch.uint8).numpy().tobytes()
                    writer.add(
                        name,
                        fp8_bytes,
                        list(shape),
                        "f8_e4m3",
                        LAYOUT_FP8_NATIVE,
                    )
                    n_quant += 1
                    continue
                # Other dtype on a quant-target name: warn loudly and fall
                # through to the raw path so we don't silently drop tensors.
                log(f"[bake-fp8] WARNING: quant target {name} has unexpected "
                    f"dtype {t.dtype}; storing raw")

            # Layout-transform path.
            layout = classify_layout(name, shape, linear_idx)
            if layout == LAYOUT_DEPTHWISE_CONV_SQUEEZED:
                t2 = apply_squeeze_dim1(t)
                writer.add(name, tensor_to_bytes(t2, torch_dtype_to_str(t2.dtype)),
                           list(t2.shape), torch_dtype_to_str(t2.dtype), layout)
            elif layout == LAYOUT_HEAD_BIAS_RESHAPED:
                t2 = apply_head_bias_reshape(t)
                writer.add(name, tensor_to_bytes(t2, torch_dtype_to_str(t2.dtype)),
                           list(t2.shape), torch_dtype_to_str(t2.dtype), layout)
            elif layout == LAYOUT_HEAD_EXP_RESHAPED:
                t2 = apply_a_log_to_exp_bf16(t)
                writer.add(name, tensor_to_bytes(t2, torch_dtype_to_str(t2.dtype)),
                           list(t2.shape), torch_dtype_to_str(t2.dtype), layout)
            else:
                writer.add(name, tensor_to_bytes(t, torch_dtype_to_str(t.dtype)),
                           list(shape), torch_dtype_to_str(t.dtype), LAYOUT_RAW)
            n_skip += 1
    finally:
        for h in open_shards.values():
            try:
                h.__exit__(None, None, None)
            except Exception:
                pass

    writer.close()
    elapsed = time.perf_counter() - t0
    log(f"[bake-fp8] quantized {n_quant} projection tensors, "
        f"passed through {n_skip} layout/raw tensors in {elapsed:.1f}s")
    log(f"[bake-fp8] done. Output: {out_dir}")


if __name__ == "__main__":
    main()
