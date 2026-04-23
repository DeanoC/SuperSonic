#!/usr/bin/env python3
"""PyTorch reference for certified tiered KV attention on Llama 3.1.

This module is intentionally small and deterministic. It is the oracle
contract that the Rust/CUDA implementation should match before optimization:

* complete KV blocks use per-channel INT8 keys;
* values use per-token, per-group INT4 packing;
* original BF16/FP16 K/V remain available for fallback;
* adaptive top-K promotes high-mass blocks to FP16 keys;
* emitted telemetry exposes the key/value error bounds and fallback decisions.

The functions are written for synthetic unit tests and layer-level diagnostics.
Full HF model generation should call these primitives from a thin harness rather
than re-implementing the math in benchmark scripts.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class CertifiedKvConfig:
    block_size: int = 16
    value_group_size: int = 16
    tau_cov: float = 0.995
    k_min: int = 2
    k_max: int = 128
    v_tol: float = 0.05
    ranking_r: int = 1
    rung1_threshold: float = 0.02
    rung1_multiplier: float = 2.0
    eps_guard: float = 0.01


@dataclass
class TieredKvCache:
    keys_int8: torch.Tensor
    key_scales: torch.Tensor
    values_int4_packed: torch.Tensor
    value_scales: torch.Tensor
    value_zeros: torch.Tensor
    value_errors: torch.Tensor
    keys_original: torch.Tensor
    values_original: torch.Tensor
    aligned_tokens: int
    total_tokens: int
    config: CertifiedKvConfig

    @property
    def device(self) -> torch.device:
        return self.keys_original.device

    @property
    def kv_heads(self) -> int:
        return int(self.keys_original.shape[0])

    @property
    def head_dim(self) -> int:
        return int(self.keys_original.shape[2])

    @property
    def num_blocks(self) -> int:
        return int(self.key_scales.shape[1])

    @property
    def has_tail(self) -> bool:
        return self.total_tokens > self.aligned_tokens


def _require_kv_shape(keys: torch.Tensor, values: torch.Tensor) -> None:
    if keys.ndim != 3 or values.ndim != 3:
        raise ValueError("keys and values must be [kv_heads, tokens, head_dim]")
    if keys.shape != values.shape:
        raise ValueError(f"key/value shape mismatch: {tuple(keys.shape)} vs {tuple(values.shape)}")


def _pack_int4(q: torch.Tensor) -> torch.Tensor:
    if q.shape[-1] % 2 != 0:
        raise ValueError("INT4 packing requires an even innermost dimension")
    q = q.to(torch.uint8)
    return (q[..., 0::2] & 0x0F) | ((q[..., 1::2] & 0x0F) << 4)


def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    return torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def quantize_values_int4(values: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token, per-group INT4 quantization.

    Returns packed values, scales, zeros, and per-token L2 reconstruction error.
    """
    if values.shape[-1] % group_size != 0:
        raise ValueError("head_dim must be divisible by value_group_size")
    if values.shape[-1] % 2 != 0:
        raise ValueError("head_dim must be even for INT4 packing")

    original_shape = values.shape
    num_groups = values.shape[-1] // group_size
    grouped = values.float().reshape(*values.shape[:-1], num_groups, group_size)
    mins = grouped.amin(dim=-1)
    maxs = grouped.amax(dim=-1)
    scales = ((maxs - mins).clamp(min=1e-8) / 15.0).to(torch.float16)
    zeros = mins.to(torch.float16)
    q = ((grouped - mins.unsqueeze(-1)) / scales.float().unsqueeze(-1)).round()
    q = q.clamp(0, 15).to(torch.uint8)
    packed = _pack_int4(q.reshape(*values.shape[:-1], values.shape[-1]))
    deq = (q.float() * scales.float().unsqueeze(-1) + zeros.float().unsqueeze(-1)).reshape(original_shape)
    err = (values.float() - deq).norm(dim=-1)
    return packed.contiguous(), scales.contiguous(), zeros.contiguous(), err.contiguous()


def dequantize_values_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    head_dim = packed.shape[-1] * 2
    num_groups = head_dim // group_size
    q = _unpack_int4(packed).reshape(*packed.shape[:-1], num_groups, group_size).float()
    deq = q * scales.float().unsqueeze(-1) + zeros.float().unsqueeze(-1)
    return deq.reshape(*packed.shape[:-1], head_dim)


def build_tiered_kv_cache(
    keys: torch.Tensor,
    values: torch.Tensor,
    config: CertifiedKvConfig | None = None,
) -> TieredKvCache:
    """Build a tiered cache from post-RoPE keys and values.

    Trailing partial blocks remain in the original tensors and are attended
    with FP16/BF16 keys/values. Only complete blocks are quantized.
    """
    _require_kv_shape(keys, values)
    cfg = config or CertifiedKvConfig()
    if cfg.block_size <= 0:
        raise ValueError("block_size must be positive")
    if cfg.value_group_size <= 0:
        raise ValueError("value_group_size must be positive")

    kv_heads, total_tokens, head_dim = keys.shape
    aligned_tokens = (total_tokens // cfg.block_size) * cfg.block_size
    num_blocks = aligned_tokens // cfg.block_size
    device = keys.device

    keys_original = keys.detach().clone()
    values_original = values.detach().clone()

    if num_blocks == 0:
        empty_i8 = torch.empty(kv_heads, 0, head_dim, dtype=torch.int8, device=device)
        empty_scales = torch.empty(kv_heads, 0, head_dim, dtype=torch.float32, device=device)
        empty_packed = torch.empty(kv_heads, 0, head_dim // 2, dtype=torch.uint8, device=device)
        empty_v_meta = torch.empty(kv_heads, 0, head_dim // cfg.value_group_size, dtype=torch.float16, device=device)
        empty_err = torch.empty(kv_heads, 0, dtype=torch.float32, device=device)
        return TieredKvCache(
            empty_i8,
            empty_scales,
            empty_packed,
            empty_v_meta,
            empty_v_meta.clone(),
            empty_err,
            keys_original,
            values_original,
            aligned_tokens,
            total_tokens,
            cfg,
        )

    k_blocks = keys[:, :aligned_tokens, :].float().reshape(kv_heads, num_blocks, cfg.block_size, head_dim)
    k_absmax = k_blocks.abs().amax(dim=2).clamp(min=1e-8)
    key_scales = k_absmax / 127.0
    keys_int8 = (k_blocks / key_scales[:, :, None, :]).round().clamp(-127, 127).to(torch.int8)
    keys_int8 = keys_int8.reshape(kv_heads, aligned_tokens, head_dim).contiguous()

    packed_v, v_scales, v_zeros, token_err = quantize_values_int4(
        values[:, :aligned_tokens, :], cfg.value_group_size
    )
    value_errors = token_err.reshape(kv_heads, num_blocks, cfg.block_size).amax(dim=-1)

    return TieredKvCache(
        keys_int8=keys_int8,
        key_scales=key_scales.contiguous(),
        values_int4_packed=packed_v,
        value_scales=v_scales,
        value_zeros=v_zeros,
        value_errors=value_errors,
        keys_original=keys_original,
        values_original=values_original,
        aligned_tokens=aligned_tokens,
        total_tokens=total_tokens,
        config=cfg,
    )


def dequantize_keys(cache: TieredKvCache) -> torch.Tensor:
    if cache.num_blocks == 0:
        return cache.keys_original[:, :0, :].float()
    kv_heads, _, head_dim = cache.keys_int8.shape
    blocks = cache.keys_int8.float().reshape(kv_heads, cache.num_blocks, cache.config.block_size, head_dim)
    return (blocks * cache.key_scales[:, :, None, :]).reshape(kv_heads, cache.aligned_tokens, head_dim)


def score_blocks_int8(
    q: torch.Tensor,
    cache: TieredKvCache,
    gqa_group: int,
    q_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-Q-head block maxima and exp sums from INT8 keys."""
    if q.ndim != 2:
        raise ValueError("q must be [q_heads, head_dim]")
    q_heads, head_dim = q.shape
    if head_dim != cache.head_dim:
        raise ValueError("q head_dim does not match cache")
    if q_heads != cache.kv_heads * gqa_group:
        raise ValueError("q_heads must equal kv_heads * gqa_group")
    if q_scale is None:
        q_scale = 1.0 / math.sqrt(head_dim)

    m_b = torch.empty(q_heads, cache.num_blocks, dtype=torch.float32, device=q.device)
    s_b = torch.empty_like(m_b)
    for qh in range(q_heads):
        kvh = qh // gqa_group
        qv = q[qh].float()
        for bid in range(cache.num_blocks):
            start = bid * cache.config.block_size
            end = start + cache.config.block_size
            k = cache.keys_int8[kvh, start:end, :].float() * cache.key_scales[kvh, bid, :].float()
            scores = (k * qv).sum(dim=-1) * q_scale
            mb = scores.max()
            m_b[qh, bid] = mb
            s_b[qh, bid] = torch.exp(scores - mb).sum()
    return m_b, s_b


def block_mass_fraction(m_b: torch.Tensor, s_b: torch.Tensor) -> torch.Tensor:
    if m_b.numel() == 0:
        return torch.empty_like(m_b)
    m_global = m_b.amax(dim=1, keepdim=True)
    mass = s_b.clamp(min=1e-30) * torch.exp(m_b - m_global)
    return mass / mass.sum(dim=1, keepdim=True).clamp(min=1e-30)


def adaptive_topk_mask(
    m_b: torch.Tensor,
    s_b: torch.Tensor,
    config: CertifiedKvConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paper adaptive top-K selector.

    Returns topk_mask, k_star, tail_mass, mass_frac.
    """
    mass_frac = block_mass_fraction(m_b, s_b)
    q_heads, num_blocks = mass_frac.shape
    if num_blocks == 0:
        return (
            torch.zeros_like(mass_frac, dtype=torch.bool),
            torch.zeros(q_heads, dtype=torch.int32, device=m_b.device),
            torch.zeros(q_heads, dtype=torch.float32, device=m_b.device),
            mass_frac,
        )

    sorted_mass, sorted_idx = mass_frac.sort(dim=1, descending=True)
    cumsum = sorted_mass.cumsum(dim=1)
    k_star = torch.searchsorted(cumsum, torch.full((q_heads, 1), config.tau_cov, device=m_b.device)).squeeze(1) + 1
    hi = min(config.k_max, num_blocks)
    lo = min(config.k_min, hi)
    k_star = k_star.clamp(min=lo, max=hi).to(torch.int32)
    keep_sorted = torch.arange(num_blocks, device=m_b.device).unsqueeze(0) < k_star.long().unsqueeze(1)
    mask = torch.zeros_like(mass_frac, dtype=torch.bool)
    mask.scatter_(1, sorted_idx, keep_sorted)
    captured = (mass_frac * mask.float()).sum(dim=1)
    tail = (1.0 - captured).clamp(min=0.0)
    return mask, k_star, tail, mass_frac


def score_delta_bound(
    q: torch.Tensor,
    key_scales: torch.Tensor,
    gqa_group: int,
    q_scale: float | None = None,
) -> torch.Tensor:
    """Runtime per-head, per-block score-error bound Delta."""
    q_heads, head_dim = q.shape
    if q_scale is None:
        q_scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty(q_heads, key_scales.shape[1], dtype=torch.float32, device=q.device)
    for qh in range(q_heads):
        kvh = qh // gqa_group
        out[qh] = 0.5 * q_scale * (q[qh].float().abs().unsqueeze(0) * key_scales[kvh].float()).sum(dim=-1)
    return out


def value_error_bound(
    mass_frac: torch.Tensor,
    value_errors: torch.Tensor,
    gqa_group: int,
) -> torch.Tensor:
    """Per-head value error bound for using INT4 values on quantized blocks."""
    q_heads, num_blocks = mass_frac.shape
    e_val = torch.empty(q_heads, dtype=torch.float32, device=mass_frac.device)
    for qh in range(q_heads):
        kvh = qh // gqa_group
        e_val[qh] = (mass_frac[qh, :num_blocks] * value_errors[kvh, :num_blocks]).sum()
    return e_val


def dense_attention(q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, gqa_group: int) -> torch.Tensor:
    q_heads, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty(q_heads, values.shape[-1], dtype=torch.float32, device=q.device)
    for qh in range(q_heads):
        kvh = qh // gqa_group
        scores = (keys[kvh].float() * q[qh].float()).sum(dim=-1) * scale
        weights = torch.softmax(scores, dim=-1)
        out[qh] = weights @ values[kvh].float()
    return out


def certified_attention_step(
    q: torch.Tensor,
    cache: TieredKvCache,
    gqa_group: int,
    force_dense_fallback: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run one oracle certified attention step and return output + telemetry."""
    cfg = cache.config
    dense_out = dense_attention(q, cache.keys_original, cache.values_original, gqa_group)
    if force_dense_fallback or cache.num_blocks == 0:
        return dense_out, {
            "mode": "dense_fallback",
            "config": asdict(cfg),
            "rung4_fired": bool(force_dense_fallback),
            "num_blocks": cache.num_blocks,
        }

    m_b, s_b = score_blocks_int8(q, cache, gqa_group)
    topk_mask, k_star, tail_mass, mass_frac = adaptive_topk_mask(m_b, s_b, cfg)
    delta_blocks = score_delta_bound(q, cache.key_scales, gqa_group)
    tail_delta = torch.where(topk_mask, torch.zeros_like(delta_blocks), delta_blocks).amax(dim=1)
    e_val = value_error_bound(mass_frac, cache.value_errors, gqa_group)
    rung2 = e_val > cfg.v_tol

    keys_deq = dequantize_keys(cache)
    values_deq = dequantize_values_int4(
        cache.values_int4_packed,
        cache.value_scales,
        cache.value_zeros,
        cfg.value_group_size,
    )
    q_heads, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(dense_out)
    for qh in range(q_heads):
        kvh = qh // gqa_group
        scores_parts = []
        value_parts = []
        for bid in range(cache.num_blocks):
            start = bid * cfg.block_size
            end = start + cfg.block_size
            k_src = cache.keys_original[kvh, start:end, :] if topk_mask[qh, bid] else keys_deq[kvh, start:end, :]
            v_src = cache.values_original[kvh, start:end, :] if rung2[qh] else values_deq[kvh, start:end, :]
            scores_parts.append((k_src.float() * q[qh].float()).sum(dim=-1) * scale)
            value_parts.append(v_src.float())
        if cache.has_tail:
            scores_parts.append((cache.keys_original[kvh, cache.aligned_tokens :, :].float() * q[qh].float()).sum(dim=-1) * scale)
            value_parts.append(cache.values_original[kvh, cache.aligned_tokens :, :].float())
        scores = torch.cat(scores_parts, dim=0)
        vals = torch.cat(value_parts, dim=0)
        out[qh] = torch.softmax(scores, dim=-1) @ vals

    e_key = 2.0 * cache.values_original.float().norm(dim=-1).amax() * torch.exp(3.0 * tail_delta) * tail_mass * (torch.exp(2.0 * tail_delta) - 1.0)
    actual_err = (out - dense_out).norm(dim=-1)
    telemetry = {
        "mode": "certified",
        "config": asdict(cfg),
        "num_blocks": cache.num_blocks,
        "tail_tokens": cache.total_tokens - cache.aligned_tokens,
        "k_star": [int(x) for x in k_star.cpu().tolist()],
        "tail_mass_int8_est": [float(x) for x in tail_mass.cpu().tolist()],
        "delta_tail_max": [float(x) for x in tail_delta.cpu().tolist()],
        "e_key": [float(x) for x in e_key.cpu().tolist()],
        "e_val": [float(x) for x in e_val.cpu().tolist()],
        "actual_l2_error": [float(x) for x in actual_err.cpu().tolist()],
        "bound_total": [float(x) for x in (e_key + e_val).cpu().tolist()],
        "rung1_fired": bool((tail_mass > cfg.rung1_threshold).any().item()),
        "rung2_fired": bool(rung2.any().item()),
        "rung3_fired": False,
        "rung4_fired": False,
    }
    return out, telemetry


def _self_test_json(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    cfg = CertifiedKvConfig(
        block_size=args.block_size,
        value_group_size=args.value_group_size,
        tau_cov=args.tau_cov,
        k_min=args.k_min,
        k_max=args.k_max,
        v_tol=args.v_tol,
    )
    keys = torch.randn(args.kv_heads, args.tokens, args.head_dim, dtype=torch.float32, device=args.device)
    values = torch.randn_like(keys)
    q = torch.randn(args.kv_heads * args.gqa_group, args.head_dim, dtype=torch.float32, device=args.device)
    cache = build_tiered_kv_cache(keys, values, cfg)
    _, telemetry = certified_attention_step(q, cache, args.gqa_group)
    return telemetry


def main() -> None:
    ap = argparse.ArgumentParser(description="Certified tiered KV oracle primitives for Llama3.1")
    ap.add_argument("--self-test", action="store_true", help="run a deterministic synthetic attention step")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=33)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--gqa-group", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--value-group-size", type=int, default=16)
    ap.add_argument("--tau-cov", type=float, default=0.995)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=128)
    ap.add_argument("--v-tol", type=float, default=0.05)
    args = ap.parse_args()
    if not args.self_test:
        raise SystemExit("pass --self-test for the current synthetic oracle harness")
    print(json.dumps(_self_test_json(args), indent=2))


if __name__ == "__main__":
    main()
