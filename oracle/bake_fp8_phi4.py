#!/usr/bin/env python3
"""
Per-block FP8-E4M3 calibration bake for SuperSonic — Phi-4-mini.

Sister script to `oracle/bake_fp8.py` (Qwen3.5). Differences from the Qwen
baker:

  * Phi-4 ships **fused** projections `self_attn.qkv_proj.weight` and
    `mlp.gate_up_proj.weight`. The runtime expects them split into
    q_proj / k_proj / v_proj and gate_proj / up_proj (matching the BF16
    Rust baker `bake_phi4`). This script splits along dim 0 first, then
    quantizes each shard independently. Phi-4-mini's projection sizes
    (q=3072, kv=1024, intermediate=8192) are all multiples of 128 so the
    splits land on block boundaries.

  * No linear-attention layers, so no `conv1d` / `dt_bias` / `A_log`
    layout transforms. Everything that isn't a quantization target is
    stored Raw.

Output mirrors `bake_fp8.py`: `name` (FP8-E4M3, Fp8Native) +
`name_scale_inv` (BF16, Raw, [rows/128, cols/128]) per quantized projection.
The runtime FP8-dequant path lights up automatically from the existing
`Phi4FP8ScaleDesc` plumbing (kernel side already wired).

Usage:
    python3 oracle/bake_fp8_phi4.py --model-dir /path/to/Phi-4-mini-instruct
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

# Mirrored from oracle/bake_fp8.py / crates/model-store/src/manifest.rs.
FORMAT_VERSION = 2
CONVERTER_VERSION = 1

LAYOUT_RAW = "Raw"
LAYOUT_FP8_NATIVE = "Fp8Native"

ALIGN = 4096
BLOCK_SIZE = 128
MAX_E4M3 = 448.0


def log(msg: str) -> None:
    print(msg, flush=True)


def align_up(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


# ---------------------------------------------------------------------------
# FP8 quantization (identical to bake_fp8.py — kept inline so this script
# stays a single-file producer)
# ---------------------------------------------------------------------------
def quantize_bf16_to_fp8(
    weight: torch.Tensor, block_size: int = BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    assert weight.dim() == 2, f"expected 2D, got {tuple(weight.shape)}"
    rows, cols = weight.shape
    assert rows % block_size == 0 and cols % block_size == 0, \
        f"shape {tuple(weight.shape)} not divisible by {block_size}"

    w_f32 = weight.float()
    sr = rows // block_size
    sc = cols // block_size
    blocks = w_f32.view(sr, block_size, sc, block_size).permute(0, 2, 1, 3).contiguous()
    absmax = blocks.abs().amax(dim=(2, 3))
    scale = (absmax / MAX_E4M3).clamp_min(torch.finfo(torch.float32).tiny)
    scale_full = (
        scale.repeat_interleave(block_size, dim=0)
              .repeat_interleave(block_size, dim=1)
    )
    q_e4m3 = (w_f32 / scale_full).to(torch.float8_e4m3fn)
    q_bytes = q_e4m3.view(torch.uint8)
    return q_bytes.contiguous(), scale.to(torch.bfloat16).contiguous()


# ---------------------------------------------------------------------------
# Tensor classification (Phi-4 specific)
# ---------------------------------------------------------------------------
def is_fp8_quant_target(
    name: str, shape: tuple[int, ...], block_size: int = BLOCK_SIZE
) -> bool:
    """True if a 2D weight should be quantized to FP8.

    Phi-4 has no linear-attention layers, so the matcher is simpler than the
    Qwen3.5 sister. Quantize every leaf module name containing `_proj`:

      * `gate_proj` / `up_proj` / `down_proj`     (MLP, post-split)
      * `q_proj` / `k_proj` / `v_proj` / `o_proj` (attention, post-split)

    The fused source tensors `qkv_proj` and `gate_up_proj` are NOT matched
    here — they're split into shards before this function ever sees them.

    Skip lm_head: the runtime's lm_head matmul is BF16-only (same as the
    Qwen3.5 case in bake_fp8.py). On Phi-4-mini this is also moot because
    embeddings are tied — there's no standalone `lm_head.weight`.
    """
    if len(shape) != 2:
        return False
    if not name.endswith(".weight"):
        return False
    if "embed_tokens" in name or name == "lm_head.weight":
        return False
    if "layernorm" in name or name.endswith(".norm.weight"):
        return False
    if "_proj" not in name:
        return False
    return shape[0] % block_size == 0 and shape[1] % block_size == 0


# ---------------------------------------------------------------------------
# Manifest dtype names
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

    def close(self, model_family: str = "phi4") -> None:
        self.f.close()
        sorted_entries = sorted(self.entries, key=lambda e: e["name"])
        manifest = {
            "format_version": FORMAT_VERSION,
            "converter_version": CONVERTER_VERSION,
            "model_family": model_family,
            "tensors": sorted_entries,
        }
        with open(self.out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        log(f"[bake-fp8-phi4] wrote {self.cursor / (1024 * 1024):.1f} MiB to {self.weights_path}")


# ---------------------------------------------------------------------------
# Phi-4 fused-tensor splitting
# ---------------------------------------------------------------------------
def parse_phi4_qkv_name(name: str) -> str | None:
    """If `name` is a Phi-4 fused qkv_proj, return its layer-prefix; else None."""
    if name.endswith(".self_attn.qkv_proj.weight"):
        return name[: -len(".qkv_proj.weight")]
    return None


def parse_phi4_gate_up_name(name: str) -> str | None:
    if name.endswith(".mlp.gate_up_proj.weight"):
        return name[: -len(".gate_up_proj.weight")]
    return None


def split_qkv(
    t: torch.Tensor, q_rows: int, k_rows: int, v_rows: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert t.dim() == 2, f"qkv split: expected 2D, got {tuple(t.shape)}"
    assert t.shape[0] == q_rows + k_rows + v_rows, (
        f"qkv split: dim0 {t.shape[0]} != q {q_rows} + k {k_rows} + v {v_rows}"
    )
    q = t[:q_rows].contiguous()
    k = t[q_rows : q_rows + k_rows].contiguous()
    v = t[q_rows + k_rows :].contiguous()
    return q, k, v


def split_gate_up(
    t: torch.Tensor, intermediate: int
) -> tuple[torch.Tensor, torch.Tensor]:
    assert t.dim() == 2, f"gate_up split: expected 2D, got {tuple(t.shape)}"
    assert t.shape[0] == 2 * intermediate, (
        f"gate_up split: dim0 {t.shape[0]} != 2 * intermediate ({2 * intermediate})"
    )
    return t[:intermediate].contiguous(), t[intermediate:].contiguous()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def collect_tensor_index(model_dir: Path) -> tuple[list[Path], dict[str, int]]:
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
    ap = argparse.ArgumentParser(description="Per-block FP8 bake for Phi-4-mini")
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default=None, type=Path,
                    help="Override output dir (default: {model-dir}/.supersonic/v{FORMAT_VERSION}-fp8)")
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE,
                    help=f"Per-tile FP8 block size (default {BLOCK_SIZE}; "
                         f"the Rust runtime currently only supports 128).")
    ap.add_argument("--model-family", default="phi4",
                    help="Override the manifest model_family field (default: phi4).")
    args = ap.parse_args()

    block_size: int = args.block_size
    if block_size != BLOCK_SIZE:
        log(f"[bake-fp8-phi4] WARNING: --block-size={block_size} is non-default; "
            "the Rust runtime currently assumes 128 — use only for "
            "calibration experiments, not production bakes.")

    model_dir: Path = args.model_dir.resolve()
    out_dir: Path = args.out_dir or (model_dir / ".supersonic" / f"v{FORMAT_VERSION}-fp8")

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"missing config.json under {model_dir}")
    cfg = json.load(open(config_path))
    num_attention_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg["num_key_value_heads"])
    head_dim = int(cfg.get("head_dim", cfg["hidden_size"] // num_attention_heads))
    intermediate_size = int(cfg["intermediate_size"])
    q_rows = num_attention_heads * head_dim
    k_rows = num_kv_heads * head_dim
    v_rows = k_rows
    log(f"[bake-fp8-phi4] q_rows={q_rows} k_rows={k_rows} v_rows={v_rows} "
        f"intermediate={intermediate_size}")

    shards, index = collect_tensor_index(model_dir)
    log(f"[bake-fp8-phi4] {len(shards)} shard(s), {len(index)} tensors")

    # Phi-4 weights live under `model.*` (no `language_model` indirection).
    eligible = sorted(
        n for n in index
        if n.startswith("model.") or n == "lm_head.weight"
    )
    log(f"[bake-fp8-phi4] {len(eligible)} eligible tensors")

    writer = StreamingTensorWriter(out_dir)
    open_shards: dict[int, Any] = {}

    def get_handle(idx: int):
        h = open_shards.get(idx)
        if h is None:
            h = safe_open(str(shards[idx]), framework="pt")
            h.__enter__()
            open_shards[idx] = h
        return h

    def emit_quantized(name: str, t: torch.Tensor) -> None:
        """Quantize a 2D BF16 tensor → emit (FP8 weight, scale_inv) pair."""
        packed, scale_bf16 = quantize_bf16_to_fp8(t, block_size)
        writer.add(
            name,
            tensor_to_bytes(packed, "f8_e4m3"),
            list(t.shape),
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

    def emit_raw(name: str, t: torch.Tensor) -> None:
        writer.add(
            name,
            tensor_to_bytes(t, torch_dtype_to_str(t.dtype)),
            list(t.shape),
            torch_dtype_to_str(t.dtype),
            LAYOUT_RAW,
        )

    t0 = time.perf_counter()
    n_quant = 0
    n_split_quant = 0
    n_raw = 0
    try:
        for name in eligible:
            shard_idx = index[name]
            t = get_handle(shard_idx).get_tensor(name)
            shape = tuple(t.shape)

            # Fused QKV → split into q/k/v shards, quantize each.
            qkv_prefix = parse_phi4_qkv_name(name)
            if qkv_prefix is not None:
                q, k, v = split_qkv(t, q_rows, k_rows, v_rows)
                if t.dtype != torch.bfloat16:
                    raise SystemExit(
                        f"Phi-4 FP8 baker only supports BF16 source for "
                        f"projections; {name} is {t.dtype}"
                    )
                emit_quantized(f"{qkv_prefix}.q_proj.weight", q)
                emit_quantized(f"{qkv_prefix}.k_proj.weight", k)
                emit_quantized(f"{qkv_prefix}.v_proj.weight", v)
                n_split_quant += 3
                continue

            # Fused gate_up → split into gate/up, quantize each.
            gu_prefix = parse_phi4_gate_up_name(name)
            if gu_prefix is not None:
                gate, up = split_gate_up(t, intermediate_size)
                if t.dtype != torch.bfloat16:
                    raise SystemExit(
                        f"Phi-4 FP8 baker only supports BF16 source for "
                        f"projections; {name} is {t.dtype}"
                    )
                emit_quantized(f"{gu_prefix}.gate_proj.weight", gate)
                emit_quantized(f"{gu_prefix}.up_proj.weight", up)
                n_split_quant += 2
                continue

            # Plain projection (o_proj, down_proj) — BF16 → quantize directly.
            if is_fp8_quant_target(name, shape, block_size):
                if t.dtype == torch.bfloat16:
                    emit_quantized(name, t)
                    n_quant += 1
                    continue
                if t.dtype == torch.float8_e4m3fn:
                    # Phi-4-mini upstream is BF16 today; if a future checkpoint
                    # ships pre-quantized FP8 projections we'd also need their
                    # companion `*_scale_inv` tensors so the runtime FP8 path
                    # has scales to dequantize against. Without them the
                    # runtime would either silently fall back to BF16 matmul
                    # against FP8-packed bytes (corrupt results / OOB reads)
                    # or skip FP8 dispatch entirely. Fail fast until that
                    # plumbing exists.
                    raise SystemExit(
                        f"Phi-4 FP8 baker: source {name} is already "
                        f"float8_e4m3fn but no companion scale_inv tensor "
                        f"is available — pre-quantized FP8 sources are not "
                        f"yet supported. Re-export from a BF16 checkpoint."
                    )
                log(f"[bake-fp8-phi4] WARNING: quant target {name} has unexpected "
                    f"dtype {t.dtype}; storing raw")

            # Norms / embeddings / lm_head / anything else — pass through.
            emit_raw(name, t)
            n_raw += 1
    finally:
        for h in open_shards.values():
            try:
                h.__exit__(None, None, None)
            except Exception:
                pass

    writer.close(model_family=args.model_family)
    elapsed = time.perf_counter() - t0
    log(f"[bake-fp8-phi4] split+quantized {n_split_quant} shards, "
        f"directly-quantized {n_quant}, passed through {n_raw} tensors "
        f"in {elapsed:.1f}s")
    log(f"[bake-fp8-phi4] done. Output: {out_dir}")


if __name__ == "__main__":
    main()
