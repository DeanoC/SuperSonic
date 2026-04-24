#!/usr/bin/env python3
"""Paper-exact PyTorch oracle for certified tiered KV attention on Llama 3.1.

This module is the executable specification for the Rust/CUDA implementation.
It prioritizes exact storage semantics, runtime certificates, and fallback
decisions over speed.
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
    rung1_threshold: float = 0.005
    rung1_multiplier: float = 2.0
    eps_guard: float = 0.0001
    delta_guard_factor: float = 3.0
    score_exploration_rate: float = 0.01
    require_certified_tail_bound: bool = True


@dataclass
class TieredKvCache:
    keys_int8: torch.Tensor
    key_scales: torch.Tensor
    key_zeros: torch.Tensor
    values_int4_packed: torch.Tensor
    value_scales: torch.Tensor
    value_zeros: torch.Tensor
    value_errors: torch.Tensor
    value_norms: torch.Tensor
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


def quantize_keys_int8_asymmetric(
    keys: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-block, per-channel asymmetric signed-INT8 key quantization."""
    if keys.ndim != 3:
        raise ValueError("keys must be [kv_heads, aligned_tokens, head_dim]")
    kv_heads, aligned_tokens, head_dim = keys.shape
    if aligned_tokens % block_size != 0:
        raise ValueError("key token count must be block-aligned")
    num_blocks = aligned_tokens // block_size
    if num_blocks == 0:
        empty_i8 = torch.empty(kv_heads, 0, head_dim, dtype=torch.int8, device=keys.device)
        empty_meta = torch.empty(kv_heads, 0, head_dim, dtype=torch.float32, device=keys.device)
        return empty_i8, empty_meta, empty_meta.clone()

    blocks = keys.float().reshape(kv_heads, num_blocks, block_size, head_dim)
    mins = blocks.amin(dim=2)
    maxs = blocks.amax(dim=2)
    spans = maxs - mins
    scales = (spans / 255.0).clamp(min=1.0e-8)
    zeros = mins + 128.0 * scales
    q = ((blocks - zeros[:, :, None, :]) / scales[:, :, None, :]).round()
    q = q.clamp(-128, 127).to(torch.int8)
    return q.reshape(kv_heads, aligned_tokens, head_dim).contiguous(), scales.contiguous(), zeros.contiguous()


def dequantize_keys(cache: TieredKvCache) -> torch.Tensor:
    if cache.num_blocks == 0:
        return cache.keys_original[:, :0, :].float()
    kv_heads, _, head_dim = cache.keys_int8.shape
    blocks = cache.keys_int8.float().reshape(kv_heads, cache.num_blocks, cache.config.block_size, head_dim)
    deq = blocks * cache.key_scales[:, :, None, :] + cache.key_zeros[:, :, None, :]
    return deq.reshape(kv_heads, cache.aligned_tokens, head_dim)


def quantize_values_int4(values: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token, per-group INT4 quantization."""
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
    """Build Tier-1 compressed KV plus Tier-2 originals.

    Only complete blocks are quantized. A trailing partial block remains in the
    original tensors and is modeled as BF16/FP16 scratch data.
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
        empty_key_meta = torch.empty(kv_heads, 0, head_dim, dtype=torch.float32, device=device)
        empty_packed = torch.empty(kv_heads, 0, head_dim // 2, dtype=torch.uint8, device=device)
        empty_v_meta = torch.empty(kv_heads, 0, head_dim // cfg.value_group_size, dtype=torch.float16, device=device)
        empty_block = torch.empty(kv_heads, 0, dtype=torch.float32, device=device)
        return TieredKvCache(
            empty_i8,
            empty_key_meta,
            empty_key_meta.clone(),
            empty_packed,
            empty_v_meta,
            empty_v_meta.clone(),
            empty_block,
            empty_block.clone(),
            keys_original,
            values_original,
            aligned_tokens,
            total_tokens,
            cfg,
        )

    keys_i8, key_scales, key_zeros = quantize_keys_int8_asymmetric(keys[:, :aligned_tokens, :], cfg.block_size)
    packed_v, v_scales, v_zeros, token_err = quantize_values_int4(
        values[:, :aligned_tokens, :], cfg.value_group_size
    )
    value_errors = token_err.reshape(kv_heads, num_blocks, cfg.block_size).amax(dim=-1)
    value_norms = values[:, :aligned_tokens, :].float().norm(dim=-1).reshape(
        kv_heads, num_blocks, cfg.block_size
    ).amax(dim=-1)

    return TieredKvCache(
        keys_int8=keys_i8,
        key_scales=key_scales,
        key_zeros=key_zeros,
        values_int4_packed=packed_v,
        value_scales=v_scales,
        value_zeros=v_zeros,
        value_errors=value_errors.contiguous(),
        value_norms=value_norms.contiguous(),
        keys_original=keys_original,
        values_original=values_original,
        aligned_tokens=aligned_tokens,
        total_tokens=total_tokens,
        config=cfg,
    )


def score_blocks_int8(
    q: torch.Tensor,
    cache: TieredKvCache,
    gqa_group: int,
    q_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-Q-head block maxima and exp sums from INT8-dequantized keys."""
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
            k = (
                cache.keys_int8[kvh, start:end, :].float()
                * cache.key_scales[kvh, bid, :].float()
                + cache.key_zeros[kvh, bid, :].float()
            )
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


def adaptive_topk_mask(
    m_b: torch.Tensor,
    s_b: torch.Tensor,
    config: CertifiedKvConfig,
    delta_blocks: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adaptive top-K selector plus Rung-1 expansion."""
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
    target = torch.full((q_heads, 1), config.tau_cov, device=m_b.device)
    k_star = torch.searchsorted(cumsum.contiguous(), target).squeeze(1) + 1
    hi = min(config.k_max, num_blocks)
    lo = min(config.k_min, hi)
    k_star = k_star.clamp(min=lo, max=hi).to(torch.int64)

    if delta_blocks is not None:
        k_star = _expand_for_tail_bound(sorted_idx, mass_frac, delta_blocks, k_star, config)

    keep_sorted = torch.arange(num_blocks, device=m_b.device).unsqueeze(0) < k_star.unsqueeze(1)
    mask = torch.zeros_like(mass_frac, dtype=torch.bool)
    mask.scatter_(1, sorted_idx, keep_sorted)
    captured = (mass_frac * mask.float()).sum(dim=1)
    tail = (1.0 - captured).clamp(min=0.0)
    return mask, k_star.to(torch.int32), tail, mass_frac


def _expand_for_tail_bound(
    sorted_idx: torch.Tensor,
    mass_frac: torch.Tensor,
    delta_blocks: torch.Tensor,
    k_star: torch.Tensor,
    config: CertifiedKvConfig,
) -> torch.Tensor:
    q_heads, num_blocks = mass_frac.shape
    expanded = k_star.clone()
    for qh in range(q_heads):
        while int(expanded[qh]) < num_blocks:
            mask = torch.zeros(num_blocks, dtype=torch.bool, device=mass_frac.device)
            mask[sorted_idx[qh, : int(expanded[qh])]] = True
            tail = (mass_frac[qh] * (~mask).float()).sum().clamp(min=0.0)
            tail_delta = torch.where(mask, torch.zeros_like(delta_blocks[qh]), delta_blocks[qh]).max()
            true_tail = torch.exp(config.delta_guard_factor * tail_delta) * tail
            if true_tail <= config.rung1_threshold:
                break
            next_k = max(int(expanded[qh]) + 1, int(math.ceil(float(expanded[qh]) * config.rung1_multiplier)))
            expanded[qh] = min(next_k, num_blocks)
    return expanded


def _tail_delta(delta_blocks: torch.Tensor, topk_mask: torch.Tensor) -> torch.Tensor:
    if delta_blocks.numel() == 0:
        return torch.zeros(delta_blocks.shape[0], dtype=torch.float32, device=delta_blocks.device)
    tail_delta = torch.where(topk_mask, torch.zeros_like(delta_blocks), delta_blocks).amax(dim=1)
    return tail_delta


def value_error_bound(
    mass_frac: torch.Tensor,
    value_errors: torch.Tensor,
    gqa_group: int,
    promote_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-head achieved value bound. Promoted blocks contribute zero error."""
    q_heads, num_blocks = mass_frac.shape
    e_val = torch.empty(q_heads, dtype=torch.float32, device=mass_frac.device)
    for qh in range(q_heads):
        kvh = qh // gqa_group
        eta = value_errors[kvh, :num_blocks]
        if promote_mask is not None:
            eta = torch.where(promote_mask[qh, :num_blocks], torch.zeros_like(eta), eta)
        e_val[qh] = (mass_frac[qh, :num_blocks] * eta).sum()
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


def _fp16_block_log_masses(q: torch.Tensor, keys: torch.Tensor, gqa_group: int, block_size: int) -> torch.Tensor:
    q_heads, head_dim = q.shape
    num_blocks = keys.shape[1] // block_size
    out = torch.empty(q_heads, num_blocks, dtype=torch.float32, device=q.device)
    scale = 1.0 / math.sqrt(head_dim)
    for qh in range(q_heads):
        kvh = qh // gqa_group
        qv = q[qh].float()
        for bid in range(num_blocks):
            start = bid * block_size
            end = start + block_size
            scores = (keys[kvh, start:end, :].float() * qv).sum(dim=-1) * scale
            m = scores.max()
            out[qh, bid] = m + torch.log(torch.exp(scores - m).sum().clamp(min=1e-30))
    return out


def ranking_consistency_fallback_heads(
    int8_m_b: torch.Tensor,
    int8_s_b: torch.Tensor,
    fp16_log_mass: torch.Tensor,
    topk_mask: torch.Tensor,
    delta_blocks: torch.Tensor,
    ranking_r: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_heads, num_blocks = int8_m_b.shape
    int8_log_mass = int8_m_b + torch.log(int8_s_b.clamp(min=1e-30))
    order_fail = torch.zeros(q_heads, dtype=torch.bool, device=int8_m_b.device)
    boundary_fail = torch.zeros_like(order_fail)
    if ranking_r <= 0 or num_blocks == 0:
        return order_fail, boundary_fail, int8_log_mass

    for qh in range(q_heads):
        selected = torch.nonzero(topk_mask[qh], as_tuple=False).flatten()
        r = min(ranking_r, int(selected.numel()))
        if r == 0:
            continue
        int8_sel = int8_log_mass[qh, selected]
        fp16_sel = fp16_log_mass[qh, selected]
        int8_top = selected[torch.topk(int8_sel, r).indices]
        fp16_top = selected[torch.topk(fp16_sel, r).indices]
        order_fail[qh] = not torch.equal(int8_top, fp16_top)

        boundary = torch.topk(fp16_sel, r).values[-1]
        tail = ~topk_mask[qh]
        if tail.any():
            boundary_fail[qh] = bool(((int8_log_mass[qh] + delta_blocks[qh]) > boundary)[tail].any().item())
    return order_fail, boundary_fail, int8_log_mass


def _score_consistency_violations(
    q: torch.Tensor,
    cache: TieredKvCache,
    gqa_group: int,
    topk_mask: torch.Tensor,
    delta_blocks: torch.Tensor,
) -> torch.Tensor:
    q_heads, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    keys_deq = dequantize_keys(cache)
    violations = torch.zeros(q_heads, dtype=torch.bool, device=q.device)
    for qh in range(q_heads):
        kvh = qh // gqa_group
        for bid in torch.nonzero(topk_mask[qh], as_tuple=False).flatten().tolist():
            start = bid * cache.config.block_size
            end = start + cache.config.block_size
            fp16_scores = (cache.keys_original[kvh, start:end, :].float() * q[qh].float()).sum(dim=-1) * scale
            int8_scores = (keys_deq[kvh, start:end, :] * q[qh].float()).sum(dim=-1) * scale
            if (fp16_scores - int8_scores).abs().max() > delta_blocks[qh, bid] + cache.config.eps_guard:
                violations[qh] = True
    return violations


def _mixed_attention(
    q: torch.Tensor,
    cache: TieredKvCache,
    gqa_group: int,
    key_promote_mask: torch.Tensor,
    value_promote_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    keys_deq = dequantize_keys(cache)
    values_deq = dequantize_values_int4(
        cache.values_int4_packed,
        cache.value_scales,
        cache.value_zeros,
        cache.config.value_group_size,
    )
    q_heads, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty(q_heads, head_dim, dtype=torch.float32, device=q.device)
    block_mass = torch.zeros(q_heads, cache.num_blocks, dtype=torch.float32, device=q.device)
    for qh in range(q_heads):
        kvh = qh // gqa_group
        scores_parts = []
        value_parts = []
        block_slices: list[slice] = []
        pos = 0
        for bid in range(cache.num_blocks):
            start = bid * cache.config.block_size
            end = start + cache.config.block_size
            k_src = cache.keys_original[kvh, start:end, :] if key_promote_mask[qh, bid] else keys_deq[kvh, start:end, :]
            v_src = cache.values_original[kvh, start:end, :] if value_promote_mask[qh, bid] else values_deq[kvh, start:end, :]
            scores_parts.append((k_src.float() * q[qh].float()).sum(dim=-1) * scale)
            value_parts.append(v_src.float())
            block_slices.append(slice(pos, pos + cache.config.block_size))
            pos += cache.config.block_size
        if cache.has_tail:
            scores_parts.append((cache.keys_original[kvh, cache.aligned_tokens :, :].float() * q[qh].float()).sum(dim=-1) * scale)
            value_parts.append(cache.values_original[kvh, cache.aligned_tokens :, :].float())
        scores = torch.cat(scores_parts, dim=0)
        vals = torch.cat(value_parts, dim=0)
        weights = torch.softmax(scores, dim=-1)
        for bid, block_slice in enumerate(block_slices):
            block_mass[qh, bid] = weights[block_slice].sum()
        out[qh] = weights @ vals
    return out, block_mass


def certified_attention_step(
    q: torch.Tensor,
    cache: TieredKvCache,
    gqa_group: int,
    force_dense_fallback: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run one certified attention step and return output plus certificate telemetry."""
    cfg = cache.config
    dense_out = dense_attention(q, cache.keys_original, cache.values_original, gqa_group)
    base_telemetry: dict[str, Any] = {
        "config": asdict(cfg),
        "num_blocks": cache.num_blocks,
        "tail_tokens": cache.total_tokens - cache.aligned_tokens,
    }
    if force_dense_fallback or cache.num_blocks == 0:
        base_telemetry.update(
            {
                "mode": "dense_fallback",
                "rung1_fired": False,
                "rung2_fired": False,
                "rung3_fired": False,
                "rung4_fired": bool(force_dense_fallback),
            }
        )
        return dense_out, base_telemetry

    m_b, s_b = score_blocks_int8(q, cache, gqa_group)
    delta_blocks = score_delta_bound(q, cache.key_scales, gqa_group)
    initial_mask, initial_k, initial_tail, mass_frac = adaptive_topk_mask(m_b, s_b, cfg)
    topk_mask, k_star, tail_mass, _ = adaptive_topk_mask(m_b, s_b, cfg, delta_blocks)
    tail_delta = _tail_delta(delta_blocks, topk_mask)
    true_tail_bound = torch.exp(cfg.delta_guard_factor * tail_delta) * tail_mass
    rung1_fired = bool((k_star.long() > initial_k.long()).any().item())
    tail_uncertified = bool((true_tail_bound > cfg.rung1_threshold).any().item())
    if cfg.require_certified_tail_bound and tail_uncertified and bool((~topk_mask).any().item()):
        telem = dict(base_telemetry)
        telem.update(
            {
                "mode": "dense_fallback",
                "k_star": [int(x) for x in k_star.cpu().tolist()],
                "tail_mass_int8_est": [float(x) for x in tail_mass.cpu().tolist()],
                "true_tail_bound": [float(x) for x in true_tail_bound.cpu().tolist()],
                "delta_tail_max": [float(x) for x in tail_delta.cpu().tolist()],
                "rung1_fired": rung1_fired,
                "rung2_fired": False,
                "rung3_fired": False,
                "rung4_fired": True,
                "rung4_reason": "tail_bound",
            }
        )
        return dense_out, telem

    value_promote_mask = torch.zeros_like(topk_mask, dtype=torch.bool)
    for qh in range(q.shape[0]):
        kvh = qh // gqa_group
        value_promote_mask[qh] = mass_frac[qh] * cache.value_errors[kvh] > cfg.v_tol

    fp16_log_mass = _fp16_block_log_masses(q, cache.keys_original[:, : cache.aligned_tokens, :], gqa_group, cfg.block_size)
    order_fail, boundary_fail, int8_log_mass = ranking_consistency_fallback_heads(
        m_b, s_b, fp16_log_mass, topk_mask, delta_blocks, cfg.ranking_r
    )
    score_fail = _score_consistency_violations(q, cache, gqa_group, topk_mask, delta_blocks)
    if bool(score_fail.any().item()):
        telem = dict(base_telemetry)
        telem.update(
            {
                "mode": "dense_fallback",
                "k_star": [int(x) for x in k_star.cpu().tolist()],
                "score_consistency_violations": int(score_fail.sum().item()),
                "rung1_fired": rung1_fired,
                "rung2_fired": bool(value_promote_mask.any().item()),
                "rung3_fired": False,
                "rung4_fired": True,
                "rung4_reason": "score_consistency",
            }
        )
        return dense_out, telem

    out, actual_mass = _mixed_attention(q, cache, gqa_group, topk_mask, value_promote_mask)
    rung3_heads = order_fail | boundary_fail
    if bool(rung3_heads.any().item()):
        out[rung3_heads] = dense_out[rung3_heads]
        value_promote_mask[rung3_heads] = True

    achieved_e_val = value_error_bound(actual_mass, cache.value_errors, gqa_group, value_promote_mask)
    vmax = torch.empty(q.shape[0], dtype=torch.float32, device=q.device)
    for qh in range(q.shape[0]):
        kvh = qh // gqa_group
        vmax[qh] = cache.value_norms[kvh].max()
    e_key = 2.0 * vmax * true_tail_bound * (torch.exp(2.0 * tail_delta) - 1.0)
    e_key = torch.where(rung3_heads, torch.zeros_like(e_key), e_key)
    achieved_e_val = torch.where(rung3_heads, torch.zeros_like(achieved_e_val), achieved_e_val)
    actual_err = (out - dense_out).norm(dim=-1)

    telemetry = dict(base_telemetry)
    telemetry.update(
        {
            "mode": "certified",
            "k_star": [int(x) for x in k_star.cpu().tolist()],
            "initial_k_star": [int(x) for x in initial_k.cpu().tolist()],
            "tail_mass_initial": [float(x) for x in initial_tail.cpu().tolist()],
            "tail_mass_int8_est": [float(x) for x in tail_mass.cpu().tolist()],
            "true_tail_bound": [float(x) for x in true_tail_bound.cpu().tolist()],
            "delta_tail_max": [float(x) for x in tail_delta.cpu().tolist()],
            "vmax": [float(x) for x in vmax.cpu().tolist()],
            "e_key": [float(x) for x in e_key.cpu().tolist()],
            "e_val": [float(x) for x in achieved_e_val.cpu().tolist()],
            "actual_l2_error": [float(x) for x in actual_err.cpu().tolist()],
            "bound_total": [float(x) for x in (e_key + achieved_e_val).cpu().tolist()],
            "rung1_fired": rung1_fired,
            "rung2_fired": bool(value_promote_mask.any().item()),
            "rung3_fired": bool(rung3_heads.any().item()),
            "rung4_fired": False,
            "ranking_order_fallback_heads": int(order_fail.sum().item()),
            "ranking_boundary_fallback_heads": int(boundary_fail.sum().item()),
            "score_consistency_violations": 0,
            "int8_log_mass_max": float(int8_log_mass.max().item()),
        }
    )
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
        require_certified_tail_bound=not args.allow_uncertified_tail,
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
    ap.add_argument("--allow-uncertified-tail", action="store_true")
    args = ap.parse_args()
    if not args.self_test:
        raise SystemExit("pass --self-test for the current synthetic oracle harness")
    print(json.dumps(_self_test_json(args), indent=2))


if __name__ == "__main__":
    main()
