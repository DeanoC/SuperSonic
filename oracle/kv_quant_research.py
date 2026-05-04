#!/usr/bin/env python3
"""Lightweight Qwen KV-cache quantization research harness.

Consumes an NPZ with arrays named `q`, `k`, and `v`, or generates a
deterministic Qwen-shaped synthetic fixture. Expected shapes:

  q: [layers, q_heads, head_dim] or [q_heads, head_dim]
  k: [layers, kv_heads, tokens, head_dim] or [kv_heads, tokens, head_dim]
  v: same shape as k

The harness simulates storage formats on CPU and reports per-layer/per-head
attention-output error plus estimated resident bytes. It is intentionally
kernel-free; HIP implementations should be added only after this identifies a
promising format.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Scheme:
    name: str
    bits: int
    group_size: int
    scale_bytes: int
    per_channel_key: bool = False
    quantize_key: bool = True
    quantize_value: bool = True


SCHEMES = [
    Scheme("bf16", 16, 0, 0, quantize_key=False, quantize_value=False),
    Scheme("fp8_e4m3_token", 8, 0, 4),
    Scheme("int4_token_group64", 4, 64, 4),
    Scheme("int4_k_only_token_group64", 4, 64, 4, quantize_value=False),
    Scheme("int4_v_only_token_group64", 4, 64, 4, quantize_key=False),
    Scheme("fp8_k_int4_v_group64", 4, 64, 4),
    Scheme("int2_token_group64", 2, 64, 4),
    Scheme("kivi_like_int2_k_channel_v_token", 2, 64, 4, per_channel_key=True),
]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_heads, head_dim = q.shape
    kv_heads, _, _ = k.shape
    gqa = q_heads // kv_heads
    out = np.empty((q_heads, head_dim), dtype=np.float32)
    scale = 1.0 / math.sqrt(head_dim)
    for qh in range(q_heads):
        kvh = qh // gqa
        scores = (k[kvh].astype(np.float32) @ q[qh].astype(np.float32)) * scale
        probs = _softmax(scores)
        out[qh] = probs @ v[kvh].astype(np.float32)
    return out


def _uniform_quant_dequant(x: np.ndarray, bits: int, axis: int | tuple[int, ...]) -> np.ndarray:
    qmax = (1 << bits) - 1
    xmin = np.min(x, axis=axis, keepdims=True)
    xmax = np.max(x, axis=axis, keepdims=True)
    scale = np.maximum((xmax - xmin) / qmax, 1e-8)
    q = np.rint((x - xmin) / scale).clip(0, qmax)
    return (q * scale + xmin).astype(np.float32)


def _simulate_scheme(k: np.ndarray, v: np.ndarray, scheme: Scheme) -> tuple[np.ndarray, np.ndarray]:
    if scheme.name == "bf16":
        return k.astype(np.float32), v.astype(np.float32)
    if scheme.name == "fp8_e4m3_token":
        # E4M3 is approximated as symmetric per-token 8-bit here. This is a
        # storage/error baseline, not a bit-exact FP8 emulator.
        return (
            _uniform_quant_dequant(k, 8, axis=-1) if scheme.quantize_key else k.astype(np.float32),
            _uniform_quant_dequant(v, 8, axis=-1) if scheme.quantize_value else v.astype(np.float32),
        )
    if scheme.name == "fp8_k_int4_v_group64":
        return (
            _uniform_quant_dequant(k, 8, axis=-1),
            _uniform_quant_dequant(v, 4, axis=-1),
        )
    if not scheme.quantize_key:
        kq = k.astype(np.float32)
    elif scheme.per_channel_key:
        kq = _uniform_quant_dequant(k, scheme.bits, axis=1)
    else:
        kq = _uniform_quant_dequant(k, scheme.bits, axis=-1)
    vq = (
        _uniform_quant_dequant(v, scheme.bits, axis=-1)
        if scheme.quantize_value
        else v.astype(np.float32)
    )
    return kq, vq


def _estimate_bytes(shape: tuple[int, int, int], scheme: Scheme) -> int:
    kv_heads, tokens, head_dim = shape
    side_elems = kv_heads * tokens * head_dim
    if scheme.name == "bf16":
        return side_elems * 2 * 2
    key_bits = scheme.bits if scheme.quantize_key else 16
    value_bits = scheme.bits if scheme.quantize_value else 16
    data_bytes = (side_elems * key_bits + 7) // 8
    data_bytes += (side_elems * value_bits + 7) // 8
    if scheme.name == "fp8_e4m3_token":
        quantized_sides = int(scheme.quantize_key) + int(scheme.quantize_value)
        return data_bytes + kv_heads * tokens * quantized_sides * scheme.scale_bytes
    if scheme.name == "fp8_k_int4_v_group64":
        key_data = side_elems
        key_meta = kv_heads * tokens * scheme.scale_bytes
        value_data = (side_elems * 4 + 7) // 8
        value_groups = max(1, math.ceil(head_dim / max(1, scheme.group_size)))
        value_meta = kv_heads * tokens * value_groups * scheme.scale_bytes * 2
        return key_data + key_meta + value_data + value_meta
    groups = max(1, math.ceil(head_dim / max(1, scheme.group_size)))
    meta = 0
    if scheme.quantize_key:
        meta += kv_heads * tokens * groups * scheme.scale_bytes * 2
    if scheme.quantize_value:
        meta += kv_heads * tokens * groups * scheme.scale_bytes * 2
    if scheme.per_channel_key and scheme.quantize_key:
        meta += kv_heads * head_dim * 2 * scheme.scale_bytes
    return data_bytes + meta


def _load_or_generate(
    path: Path | None, args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    metadata = {}
    if path is not None:
        data = np.load(path)
        q, k, v = data["q"], data["k"], data["v"]
        for name in ["layer_ids", "prompt_position", "prompt_tokens"]:
            if name in data:
                value = data[name]
                metadata[name] = value.tolist() if value.ndim > 0 else int(value)
    else:
        rng = np.random.default_rng(args.seed)
        q = rng.normal(size=(args.layers, args.q_heads, args.head_dim)).astype(np.float32)
        k = rng.normal(size=(args.layers, args.kv_heads, args.tokens, args.head_dim)).astype(np.float32)
        v = rng.normal(size=k.shape).astype(np.float32)
    if q.ndim == 2:
        q = q[None, ...]
    if k.ndim == 3:
        k = k[None, ...]
    if v.ndim == 3:
        v = v[None, ...]
    if q.shape[0] != k.shape[0] or k.shape != v.shape:
        raise ValueError(f"shape mismatch q={q.shape} k={k.shape} v={v.shape}")
    return q.astype(np.float32), k.astype(np.float32), v.astype(np.float32), metadata


def run(args: argparse.Namespace) -> dict:
    q, k, v, metadata = _load_or_generate(args.input, args)
    results = []
    for scheme in SCHEMES:
        layer_rows = []
        for layer in range(k.shape[0]):
            dense = _attention(q[layer], k[layer], v[layer])
            kq, vq = _simulate_scheme(k[layer], v[layer], scheme)
            approx = _attention(q[layer], kq, vq)
            diff = approx - dense
            dense_norm = np.linalg.norm(dense, axis=1)
            err = np.linalg.norm(diff, axis=1)
            rel = err / np.maximum(dense_norm, 1e-8)
            layer_rows.append(
                {
                    "layer": layer,
                    "max_abs": float(np.max(np.abs(diff))),
                    "mean_l2": float(np.mean(err)),
                    "max_rel_l2": float(np.max(rel)),
                    "head_rel_l2": [float(x) for x in rel],
                }
            )
        bytes_est = _estimate_bytes(tuple(k.shape[1:]), scheme) * k.shape[0]
        results.append(
            {
                "scheme": scheme.name,
                "estimated_vram_bytes": bytes_est,
                "estimated_vram_mib": bytes_est / (1024 * 1024),
                "layers": layer_rows,
                "max_layer_rel_l2": max(row["max_rel_l2"] for row in layer_rows),
            }
        )
    candidates = [r for r in results if r["scheme"] not in {"bf16"}]
    best = min(candidates, key=lambda r: (r["max_layer_rel_l2"], r["estimated_vram_bytes"]))
    threshold = args.max_rel_l2_threshold
    passing = [
        r["scheme"]
        for r in candidates
        if threshold is not None and r["max_layer_rel_l2"] <= threshold
    ]
    return {
        "input": str(args.input) if args.input else "synthetic",
        "shape": {"q": list(q.shape), "k": list(k.shape), "v": list(v.shape)},
        "metadata": metadata,
        "results": results,
        "max_rel_l2_threshold": threshold,
        "passing_schemes": passing,
        "recommended_first_hip_candidate": best["scheme"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulate Qwen KV-cache quantization candidates")
    ap.add_argument("--input", type=Path, help="NPZ containing q/k/v arrays")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--q-heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument(
        "--max-rel-l2-threshold",
        type=float,
        help="Optional pass/fail threshold applied to each scheme's max layer relative L2.",
    )
    ap.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit non-zero if no compressed scheme satisfies --max-rel-l2-threshold.",
    )
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.fail_on_threshold and args.max_rel_l2_threshold is None:
        ap.error("--fail-on-threshold requires --max-rel-l2-threshold")
    payload = run(args)
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    if args.fail_on_threshold and not payload["passing_schemes"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
