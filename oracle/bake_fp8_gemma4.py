#!/usr/bin/env python3
"""
Per-block FP8-E4M3 calibration bake for SuperSonic — Gemma 4.

Sister script to `oracle/bake_fp8.py` (Qwen 3.5) and
`oracle/bake_fp8_phi4.py` (Phi-4). Differences from those:

  * Gemma 4 weights live under `model.language_model.*` (not `model.*`).
  * No fused projections — q/k/v_proj, gate/up_proj, o_proj, down_proj
    are all distinct tensors in the safetensors. We can quantize them
    in-place without splitting.
  * Adds two PLE projections per layer (`per_layer_input_gate.weight`,
    `per_layer_projection.weight`) that the runtime kernel also reads
    via FP8 dequant. These are simple 2D Linear weights — same baker.
  * Shared-KV layers (the runtime aliases earlier layers' caches) do
    NOT contain k_proj / v_proj / k_norm in the safetensors at all,
    so the matcher naturally skips them.

Output mirrors the Phi-4 baker: `name` (FP8-E4M3, Fp8Native) +
`name_scale_inv` (BF16, Raw, [rows/128, cols/128]) per quantized
projection. The runtime FP8-dequant path lights up automatically from
the existing `Gemma4FP8ScaleDesc` plumbing (kernel side already wired).

Usage:
    python3 oracle/bake_fp8_gemma4.py --model-dir /path/to/gemma-4-E2B
"""

from __future__ import annotations

import argparse
import json
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
# FP8 quantization (identical to bake_fp8_phi4.py)
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
# Tensor classification (Gemma 4 specific)
# ---------------------------------------------------------------------------
QUANT_PROJ_SUFFIXES = (
    ".q_proj.weight",
    ".k_proj.weight",
    ".v_proj.weight",
    ".o_proj.weight",
    ".gate_proj.weight",
    ".up_proj.weight",
    ".down_proj.weight",
    ".per_layer_input_gate.weight",
    ".per_layer_projection.weight",
)


def is_fp8_quant_target(
    name: str, shape: tuple[int, ...], block_size: int = BLOCK_SIZE
) -> bool:
    """True if a 2D weight should be FP8-quantized.

    Gemma 4 splits projections out of the box (no fused qkv / gate_up
    tensors), so we just match by suffix. PLE projections are included.

    Skip:
      * Embeddings (BF16, also tied to lm_head).
      * Norms (`*_layernorm.weight`, `.norm.weight`, `q_norm`, `k_norm`,
        `post_per_layer_input_norm`, `per_layer_projection_norm`).
      * `per_layer_model_projection.weight` — different shape, not a
        per-layer Linear; the runtime treats it as raw BF16.
      * lm_head (tied to embeddings on Gemma 4 dense variants — no
        standalone tensor exists).
    """
    if len(shape) != 2:
        return False
    if not any(name.endswith(suf) for suf in QUANT_PROJ_SUFFIXES):
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
    expected = {
        "bf16": torch.bfloat16,
        "f32": torch.float32,
        "f16": torch.float16,
        "u8": torch.uint8,
    }[dtype_str]
    if t.dtype != expected:
        t = t.to(expected)
    # `.view(torch.uint8)` requires a non-scalar tensor; some Gemma 4
    # safetensors include 0-d scalars (e.g. `model.dummy_token_index`).
    # `.reshape((-1,))` materializes a 1-d view that can be re-viewed.
    flat = t.contiguous().cpu().reshape((-1,))
    return flat.view(torch.uint8).numpy().tobytes()


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

    def add(
        self, name: str, data: bytes, shape: list[int], dtype_str: str, layout: str
    ) -> None:
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

    def close(self, model_family: str = "gemma4") -> None:
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
        log(
            f"[bake-fp8-gemma4] wrote {self.cursor / (1024 * 1024):.1f} MiB "
            f"to {self.weights_path}"
        )


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
    ap = argparse.ArgumentParser(description="Per-block FP8 bake for Gemma 4")
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument(
        "--out-dir",
        default=None,
        type=Path,
        help="Override output dir (default: {model-dir}/.supersonic/v{FORMAT_VERSION}-fp8)",
    )
    ap.add_argument(
        "--block-size",
        type=int,
        default=BLOCK_SIZE,
        help=f"Per-tile FP8 block size (default {BLOCK_SIZE}; the Rust "
             f"runtime currently only supports 128).",
    )
    ap.add_argument(
        "--model-family",
        default="gemma4",
        help="Override the manifest model_family field (default: gemma4).",
    )
    args = ap.parse_args()

    block_size: int = args.block_size
    if block_size != BLOCK_SIZE:
        log(
            f"[bake-fp8-gemma4] WARNING: --block-size={block_size} is non-default; "
            "the Rust runtime currently assumes 128 — use only for "
            "calibration experiments, not production bakes."
        )

    model_dir: Path = args.model_dir.resolve()
    out_dir: Path = args.out_dir or (model_dir / ".supersonic" / f"v{FORMAT_VERSION}-fp8")

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"missing config.json under {model_dir}")
    cfg = json.load(open(config_path))
    # Gemma 4 dense models nest the language model config under
    # `text_config`; multimodal wrappers add an extra `language_model`
    # name level under `model.`. Either way, the safetensor keys reflect
    # the on-disk layout.
    log(f"[bake-fp8-gemma4] model_dir={model_dir}")

    shards, index = collect_tensor_index(model_dir)
    log(f"[bake-fp8-gemma4] {len(shards)} shard(s), {len(index)} tensors")

    eligible = sorted(index.keys())
    log(f"[bake-fp8-gemma4] {len(eligible)} eligible tensors")

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
    n_raw = 0
    try:
        for name in eligible:
            shard_idx = index[name]
            t = get_handle(shard_idx).get_tensor(name)
            shape = tuple(t.shape)

            if is_fp8_quant_target(name, shape, block_size):
                if t.dtype == torch.bfloat16:
                    emit_quantized(name, t)
                    n_quant += 1
                    continue
                if t.dtype == torch.float8_e4m3fn:
                    # Pre-quantized FP8 source — Gemma 4 upstream is BF16
                    # today, so this path is only exercised by future
                    # checkpoints. Fail fast here rather than emit FP8 bytes
                    # without a companion scale_inv tensor (mirrors the
                    # Phi-4 baker fix from PR #47).
                    raise SystemExit(
                        f"Gemma 4 FP8 baker: source {name} is already "
                        f"float8_e4m3fn but no companion scale_inv tensor "
                        f"is available — pre-quantized FP8 sources are not "
                        f"yet supported. Re-export from a BF16 checkpoint."
                    )
                log(
                    f"[bake-fp8-gemma4] WARNING: quant target {name} has "
                    f"unexpected dtype {t.dtype}; storing raw"
                )

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
    log(
        f"[bake-fp8-gemma4] quantized {n_quant}, passed through {n_raw} "
        f"tensors in {elapsed:.1f}s"
    )
    log(f"[bake-fp8-gemma4] done. Output: {out_dir}")


if __name__ == "__main__":
    main()
