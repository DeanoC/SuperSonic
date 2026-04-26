#!/usr/bin/env python3
"""
Oracle validation for GGUF -> SuperSonic Q4KM bakes.

This validates the bake itself, not full model quality:
  * every mapped GGUF tensor has a native-bake entry;
  * layout/shape/dtype metadata matches the expected SuperSonic transform;
  * selected raw tensors match the GGUF source after inverse llama.cpp transforms;
  * selected GGML K-block tensors preserve the exact source block bytes;
  * selected native INT4 tensors, if present, reconstruct close to the source.

The INT4 reconstruction reuses `int4_corpus_compare.dequant_int4_tensor`, which
matches the runtime packed-nibble + BF16 scale/zero semantics used by kernels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bake_q4km
from int4_corpus_compare import bf16_bytes_to_f32, dequant_int4_tensor, load_bake


@dataclass
class TensorCheck:
    name: str
    source_name: str
    layout: str
    max_abs: float
    mean_abs: float
    rel_rmse: float
    samples: int


def load_baked_raw(by_name: dict, weights: np.memmap, name: str) -> np.ndarray:
    meta = by_name[name]
    raw = weights[meta["offset"]: meta["offset"] + meta["byte_len"]]
    dtype = meta["dtype"]
    if dtype == "bf16":
        arr = bf16_bytes_to_f32(np.frombuffer(raw, dtype=np.int16))
    elif dtype == "f32":
        arr = np.frombuffer(raw, dtype=np.float32).copy()
    elif dtype == "f16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dtype == "u8":
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    else:
        raise SystemExit(f"{name}: unsupported baked dtype {dtype}")
    return arr.reshape(meta["shape"])


def apply_expected_layout(t: torch.Tensor, name: str, layout: str, a_log_precomputed: bool) -> np.ndarray:
    shape = list(t.shape)
    if a_log_precomputed:
        if len(shape) != 1:
            raise SystemExit(f"{name}: expected rank-1 precomputed A_log, got {shape}")
        return t.to(torch.float32).reshape(1, 1, shape[0]).cpu().numpy()
    if layout == bake_q4km.LAYOUT_CONV_SQ and len(shape) == 3:
        t = t.squeeze(1)
    elif layout == bake_q4km.LAYOUT_HEAD_BIAS:
        t = t.reshape(1, 1, shape[0])
    elif layout == bake_q4km.LAYOUT_HEAD_EXP:
        t = torch.exp(t.to(torch.float32)).reshape(1, 1, shape[0])
    return t.to(torch.float32).cpu().numpy()


def compare_arrays(actual: np.ndarray, expected: np.ndarray, max_items: int) -> tuple[float, float, float, int]:
    if actual.shape != expected.shape:
        raise SystemExit(f"shape mismatch: actual {actual.shape} expected {expected.shape}")
    a = actual.reshape(-1).astype(np.float32, copy=False)
    e = expected.reshape(-1).astype(np.float32, copy=False)
    n = a.size
    if n > max_items:
        # Deterministic, spread over the full tensor without allocating random state.
        idx = np.linspace(0, n - 1, num=max_items, dtype=np.int64)
        a = a[idx]
        e = e[idx]
        n = max_items
    diff = np.abs(a - e)
    rmse = math.sqrt(float(np.mean((a - e) ** 2)))
    denom = math.sqrt(float(np.mean(e ** 2))) + 1e-12
    return float(diff.max(initial=0.0)), float(diff.mean() if n else 0.0), rmse / denom, int(n)


def choose_validation_tensors(mapped_infos: list[tuple[bake_q4km.GgufTensorInfo, str]], by_name: dict, limit: int) -> list[tuple[bake_q4km.GgufTensorInfo, str]]:
    preferred_suffixes = [
        ".linear_attn.A_log",
        ".linear_attn.dt_bias",
        ".linear_attn.conv1d.weight",
        ".linear_attn.out_proj.weight",
        ".linear_attn.in_proj_qkv.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
        ".self_attn.q_proj.weight",
        ".self_attn.o_proj.weight",
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
    ]
    picked: list[tuple[bake_q4km.GgufTensorInfo, str]] = []
    seen = set()
    for suffix in preferred_suffixes:
        for info, mapped in mapped_infos:
            if mapped in seen or mapped not in by_name:
                continue
            if mapped.endswith(suffix):
                picked.append((info, mapped))
                seen.add(mapped)
                break
            if len(picked) >= limit:
                return picked
    for info, mapped in mapped_infos:
        if len(picked) >= limit:
            break
        if mapped not in seen and mapped in by_name:
            picked.append((info, mapped))
            seen.add(mapped)
    return picked


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a SuperSonic Q4KM bake against its GGUF source")
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--gguf-file", required=True, type=Path)
    ap.add_argument("--bake-dir", type=Path, default=None)
    ap.add_argument("--weight-prefix", default="model.language_model")
    ap.add_argument("--max-items", type=int, default=262144)
    ap.add_argument("--max-tensors", type=int, default=16)
    ap.add_argument(
        "--int4-rel-rmse-max",
        type=float,
        default=0.75,
        help="Gate for calibration-free native INT4 re-quantization error vs dequantized GGUF source.",
    )
    ap.add_argument("--raw-max-abs", type=float, default=0.02)
    args = ap.parse_args()

    bake_dir = args.bake_dir or (args.model_dir / ".supersonic" / f"v{bake_q4km.FORMAT_VERSION}-q4km")
    manifest, weights = load_bake(bake_dir)
    by_name = {t["name"]: t for t in manifest["tensors"]}
    if manifest.get("quant_profile") != "q4km-ggml-v1":
        raise SystemExit(f"{bake_dir}: expected quant_profile=q4km-ggml-v1")

    _, layer_types = bake_q4km.load_config_context(args.model_dir)
    gguf = bake_q4km.parse_gguf(args.gguf_file)
    checks: list[TensorCheck] = []
    try:
        mapped_infos = []
        missing = []
        for info in gguf.tensors:
            mapped = bake_q4km.map_gguf_name(info.name, args.weight_prefix)
            if mapped is None or mapped.startswith(f"{args.weight_prefix}.visual.") or ".mtp." in mapped:
                continue
            mapped_infos.append((info, mapped))
            if mapped not in by_name:
                missing.append((info.name, mapped))
        if missing:
            for src, mapped in missing[:20]:
                print(f"[missing] {src} -> {mapped}")
            raise SystemExit(f"{len(missing)} mapped GGUF tensors missing from bake")

        print(f"[oracle] mapped={len(mapped_infos)} manifest_tensors={len(by_name)}")
        selected = choose_validation_tensors(mapped_infos, by_name, args.max_tensors)
        for info, mapped in selected:
            meta = by_name[mapped]
            raw_layout = bake_q4km.ggml_k_layout(info.ggml_type)
            if raw_layout is not None and meta["layout"] in (
                bake_q4km.LAYOUT_GGML_Q4K,
                bake_q4km.LAYOUT_GGML_Q5K,
                bake_q4km.LAYOUT_GGML_Q6K,
            ):
                if meta["layout"] != raw_layout:
                    raise SystemExit(f"{mapped}: layout {meta['layout']} != expected {raw_layout}")
                cols = info.dims[0]
                rows = bake_q4km.prod(info.dims[1:]) if len(info.dims) > 1 else 1
                row_bytes = bake_q4km.ggml_row_size(info.ggml_type, cols)
                if meta["dtype"] != "u8" or meta["shape"] != [rows, row_bytes]:
                    raise SystemExit(
                        f"{mapped}: raw GGML metadata dtype/shape {meta['dtype']} {meta['shape']} "
                        f"!= u8 {[rows, row_bytes]}"
                    )
                raw = weights[meta["offset"]: meta["offset"] + meta["byte_len"]]
                actual = np.frombuffer(raw, dtype=np.uint8)
                expected = np.frombuffer(bake_q4km.raw_gguf_tensor_bytes(gguf, info), dtype=np.uint8)
                max_abs, mean_abs, rel_rmse, samples = compare_arrays(actual, expected, args.max_items)
                if max_abs != 0.0:
                    raise SystemExit(f"{mapped}: raw GGML block bytes differ from source")
                checks.append(TensorCheck(mapped, info.name, meta["layout"], max_abs, mean_abs, rel_rmse, samples))
                continue

            source = bake_q4km.load_gguf_tensor(gguf, info)
            source, a_log_precomputed = bake_q4km.undo_gguf_tensor_transform(mapped, source)
            expected_layout = bake_q4km.classify_tensor(
                mapped, list(source.shape), args.weight_prefix, layer_types
            )
            if a_log_precomputed:
                expected_layout = bake_q4km.LAYOUT_HEAD_EXP
            if meta["layout"] != bake_q4km.LAYOUT_INT4 and meta["layout"] != expected_layout:
                raise SystemExit(f"{mapped}: layout {meta['layout']} != expected {expected_layout}")

            expected = apply_expected_layout(source, mapped, meta["layout"], a_log_precomputed)
            if meta["layout"] == bake_q4km.LAYOUT_INT4:
                actual = dequant_int4_tensor(by_name, weights, mapped)
            else:
                actual = load_baked_raw(by_name, weights, mapped)
            max_abs, mean_abs, rel_rmse, samples = compare_arrays(actual, expected, args.max_items)
            checks.append(TensorCheck(mapped, info.name, meta["layout"], max_abs, mean_abs, rel_rmse, samples))

            if meta["layout"] == bake_q4km.LAYOUT_INT4:
                if rel_rmse > args.int4_rel_rmse_max:
                    raise SystemExit(
                        f"{mapped}: INT4 rel_rmse {rel_rmse:.4f} exceeds {args.int4_rel_rmse_max}"
                    )
            else:
                if max_abs > args.raw_max_abs:
                    raise SystemExit(
                        f"{mapped}: raw max_abs {max_abs:.6f} exceeds {args.raw_max_abs}"
                    )

        for c in checks:
            print(
                f"[check] {c.layout:>17} {c.name} <- {c.source_name} "
                f"samples={c.samples} max_abs={c.max_abs:.6f} "
                f"mean_abs={c.mean_abs:.6f} rel_rmse={c.rel_rmse:.6f}"
            )
        print(f"[summary] ok checks={len(checks)}")
        return 0
    finally:
        gguf.close()


if __name__ == "__main__":
    raise SystemExit(main())
