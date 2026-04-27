#!/usr/bin/env python3
"""
GPTQ-style INT4 calibration bake for Gemma 4 (E2B / E4B dense).

Sister script to `bake_int4.py` — identical runtime format, different model
shape. Gemma 4 differs from Qwen3.5 in ways that matter for the calibration
sweep:

  * Each decoder layer takes a per-layer `per_layer_input` kwarg sliced out
    of a `[batch, seq, num_layers, ple_hidden]` tensor computed once per
    forward pass. The GPTQ loop has to feed each layer its own slice.
  * The attention mask + RoPE position embeddings are per-layer-TYPE
    (sliding vs. full), so the loop switches between two prebuilt versions
    of each based on `config.layer_types[layer_idx]`.
  * Shared-KV layers (the last `num_kv_shared_layers`) read K/V from a
    mutable `shared_kv_states` dict populated by the owning layers earlier
    in the same forward. We carry one dict per calibration sample so the
    sequential GPTQ sweep preserves that state.
  * Target modules differ per layer type: non-shared layers have
    q/k/v/o_proj; shared layers only q_proj/o_proj (HF drops k/v_proj
    weights on load for shared layers via `_keys_to_ignore_on_load_unexpected`).

Runtime bake format is unchanged (same LayoutTag values, same `(packed, scale,
zero)` trio per quantized tensor, same manifest schema) so the Rust-side INT4
reader can be shared with the Qwen3.5 path once it lands.

Usage:
    python3 oracle/bake_int4_gemma4.py \\
        --model-dir /path/to/gemma-4-E2B \\
        [--num-samples 128 --seqlen 2048 --device cuda]
"""

from __future__ import annotations

import argparse
import json
import math
import os
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

LAYOUT_RAW = "Raw"
LAYOUT_INT4 = "Int4Quantized"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Target-tensor selection
# ---------------------------------------------------------------------------
def is_int4_target_gemma4(name: str) -> bool:
    """True if this tensor should be quantized to INT4.

    Quantize: per-layer projection matrices (q/k/v/o_proj, gate/up/down_proj,
    per_layer_input_gate, per_layer_projection).

    Skip: all norms (including q_norm/k_norm/v_norm with RMSNorm scale), main
    embedding table, per-layer embedding table (PLE, huge, mmap-only), the
    global per_layer_model_projection (only ~27-55 MB — not worth the extra
    Hessian-tracking complexity of a non-decoder-layer target), lm_head
    (tied to embed_tokens anyway), layer_scalar, rotary inv_freq buffers.
    """
    if not name.endswith(".weight"):
        return False
    # Norm weights: `*_layernorm.weight`, `*_norm.weight`, `norm.weight`.
    if "layernorm" in name:
        return False
    if "_norm.weight" in name or name.endswith(".norm.weight"):
        return False
    # Embedding / lm_head tables.
    if "embed_tokens" in name or "lm_head" in name:
        return False
    # Global per_layer_model_projection is a matmul but lives at the TextModel
    # level, not inside a decoder layer, so it's outside the GPTQ sweep.
    if "per_layer_model_projection" in name:
        return False
    # Rotary inv_freq / layer_scalar / other buffers — already filtered by
    # the `.weight` suffix but be explicit.
    if "layer_scalar" in name or "inv_freq" in name:
        return False
    # Per-layer targets: q/k/v/o_proj, mlp.{gate,up,down}_proj,
    # per_layer_input_gate, per_layer_projection.
    return (
        name.endswith("_proj.weight")
        or name.endswith("_projection.weight")
        or name.endswith("_gate.weight")
    )


# ---------------------------------------------------------------------------
# Byte encoding helpers (shared with bake_int4.py)
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
# GPTQ core (mirror of bake_int4.py — identical math, same 2D tile scales)
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
        raise ValueError(f"in_features {in_f} must be multiple of {gs}")
    if out_f % gs != 0:
        raise ValueError(f"out_features {out_f} must be multiple of {gs}")

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
# Hessian hook
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


# ---------------------------------------------------------------------------
# Pre-layer-loop state capture
# ---------------------------------------------------------------------------
def compute_pre_layer_state(
    language_model: nn.Module,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict, dict, torch.Tensor]:
    """Reproduce the prefix of `Gemma4TextModel.forward` up to the layer loop.

    Returns (inputs_embeds, per_layer_inputs, position_embeddings_dict,
    causal_mask_mapping, position_ids) — exactly the set of values each
    decoder layer consumes, but not yet sliced / indexed by layer.
    """
    from transformers.masking_utils import (
        create_causal_mask,
        create_sliding_window_causal_mask,
    )

    config = language_model.config
    inputs_embeds = language_model.embed_tokens(input_ids)

    per_layer_inputs = None
    if language_model.hidden_size_per_layer_input:
        pli_raw = language_model.get_per_layer_inputs(input_ids, inputs_embeds)
        per_layer_inputs = language_model.project_per_layer_inputs(
            inputs_embeds, pli_raw
        )

    seq_len = inputs_embeds.shape[1]
    position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)

    mask_kwargs = {
        "config": config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": position_ids,
    }
    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }

    position_embeddings: dict = {}
    for layer_type in language_model.unique_layer_types:
        position_embeddings[layer_type] = language_model.rotary_emb(
            inputs_embeds, position_ids, layer_type
        )

    return (
        inputs_embeds,
        per_layer_inputs,
        position_embeddings,
        causal_mask_mapping,
        position_ids,
    )


# ---------------------------------------------------------------------------
# Sequential per-layer GPTQ driver (Gemma 4 variant)
# ---------------------------------------------------------------------------
def quantize_gemma4(
    language_model: nn.Module,
    calib_ids: torch.Tensor,
    device: torch.device,
    group_size: int,
    damp: float,
    ckpt_path: Path | None = None,
    fresh: bool = False,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Sequential GPTQ over Gemma 4 text layers. Returns
    {tensor_name: (nibbles, scale_f32, zero_f32)}.

    When `ckpt_path` is set, saves a full-state checkpoint (quantized tensors
    so far, per-sample hiddens, per-sample shared_kv_states) atomically after
    every layer. If the checkpoint file already exists and `fresh` is False,
    loads it and resumes from the last completed layer — the already-quantized
    layers' weights are reconstructed (dequant) and copied back into the live
    model before the GPTQ loop restarts. Skips all layers < ckpt.layer_idx.
    """
    language_model.eval()
    config = language_model.config
    layers = language_model.layers
    num_layers = len(layers)
    num_samples = calib_ids.shape[0]

    # Map nn.Linear -> state-dict weight name (scoped to the language_model subtree).
    module_to_name: dict[int, str] = {}
    module_by_name: dict[str, nn.Module] = {}
    for name, mod in language_model.named_modules():
        if isinstance(mod, nn.Linear):
            module_to_name[id(mod)] = name + ".weight"
            module_by_name[name] = mod

    # --- Capture pre-layer state per sample ---
    # Per-sample caches hold only the token-dependent tensors: `hiddens[s]` and
    # `per_layer_inputs_list[s]`. Attention masks, RoPE position embeddings, and
    # `position_ids` are functions of `seq_len` only (all samples share it),
    # so capture them once from sample 0 and reuse. On a shared-memory APU this
    # avoids a multi-GB host-side duplication that previously OOM-killed the
    # bake at 128×2048.
    # `per_layer_inputs_list` is the biggest host-side cache at large
    # calibrations: [1, S, num_layers, ple_hidden] BF16 per sample. At 128×2048
    # on E2B that's ~4.5 GiB. On a shared-memory APU we can't afford to keep
    # it resident, so spill it to disk and reload the current layer's slice on
    # demand. Keep the spill under the bake output tree so cleanup is local to
    # the run rather than leaking large `/tmp` directories across retries.
    import atexit
    import shutil
    import tempfile
    pli_cache_root = ckpt_path.parent if ckpt_path is not None else Path(tempfile.gettempdir())
    pli_cache_root.mkdir(parents=True, exist_ok=True)
    pli_cache_dir = Path(tempfile.mkdtemp(prefix="gemma4_int4_pli_", dir=pli_cache_root))
    atexit.register(shutil.rmtree, pli_cache_dir, ignore_errors=True)
    log(f"[gptq] spilling per_layer_inputs cache to {pli_cache_dir}")
    log(f"[gptq] capturing pre-layer state for {num_samples} samples...")
    hiddens: list[torch.Tensor] = []
    pli_paths: list[Path | None] = []
    shared_pos_emb: dict = {}
    shared_mask: dict = {}
    shared_pos_ids: torch.Tensor | None = None

    def save_large_torch_obj(obj, path: Path) -> None:
        torch.save(obj, path, _use_new_zipfile_serialization=False)

    for s in range(num_samples):
        ids = calib_ids[s:s + 1].to(device)
        with torch.no_grad():
            h, pli, pe, am, pid = compute_pre_layer_state(language_model, ids)
        hiddens.append(h.detach().cpu())
        if pli is not None:
            pli_path = pli_cache_dir / f"pli_{s:05d}.pt"
            save_large_torch_obj(pli.detach().cpu(), pli_path)
            pli_paths.append(pli_path)
            del pli
        else:
            pli_paths.append(None)
        if s == 0:
            shared_pos_emb = {k: (v[0].detach().cpu(), v[1].detach().cpu()) for k, v in pe.items()}
            shared_mask = {k: (v.detach().cpu() if v is not None else None) for k, v in am.items()}
            shared_pos_ids = pid.detach().cpu()

    # shared_kv_states carries across layers within one sample, so allocate one
    # dict per sample. Python dicts are live references — layers mutate them.
    kv_dicts: list[dict] = [{} for _ in range(num_samples)]

    quantized: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    # --- Resume from checkpoint if available ---
    start_layer = 0
    if ckpt_path is not None and ckpt_path.exists() and not fresh:
        log(f"[resume] loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        start_layer = int(ckpt["layer_idx"])
        quantized = ckpt["quantized"]
        hiddens = ckpt["hiddens"]
        kv_dicts = ckpt["kv_dicts"]
        # Re-apply quantized weights to the live model so subsequent layer
        # forwards see post-quant activations (matches the sequential invariant
        # that a fresh run maintains via the `mod.weight.data.copy_(...)` step).
        for tensor_name, (nibbles_s, scale_s, zero_s) in quantized.items():
            mod_name = tensor_name[: -len(".weight")]
            mod = module_by_name.get(mod_name)
            if mod is None:
                continue
            rows, cols = nibbles_s.shape
            row_gr = torch.arange(rows) // group_size
            col_gc = torch.arange(cols) // group_size
            sc_full = scale_s[row_gr][:, col_gc]
            zf_full = zero_s[row_gr][:, col_gc]
            recon = nibbles_s.float() * sc_full - zf_full * sc_full
            recon = recon.to(torch.bfloat16)
            mod.weight.data.copy_(recon.to(mod.weight.dtype).to(mod.weight.device))
        log(
            f"[resume] restored layer_idx={start_layer}, "
            f"{len(quantized)} quantized tensors re-applied, {len(hiddens)} hiddens, "
            f"{sum(len(d) for d in kv_dicts)} total kv entries"
        )

    def save_checkpoint(next_layer: int) -> None:
        if ckpt_path is None:
            return
        tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".new")
        save_large_torch_obj(
            {
                "layer_idx": next_layer,
                "quantized": quantized,
                "hiddens": hiddens,
                "kv_dicts": kv_dicts,
            },
            tmp,
        )
        tmp.replace(ckpt_path)
        log(f"[ckpt] saved at layer {next_layer}/{num_layers} -> {ckpt_path}")

    for layer_idx in range(start_layer, num_layers):
        layer = layers[layer_idx]
        layer_type = config.layer_types[layer_idx]

        targets: list[tuple[str, nn.Linear]] = []
        for mod in layer.modules():
            if isinstance(mod, nn.Linear):
                name = module_to_name.get(id(mod))
                if name is None:
                    continue
                if is_int4_target_gemma4(name):
                    targets.append((name, mod))

        log(
            f"[gptq] layer {layer_idx + 1}/{num_layers} ({layer_type}): "
            f"{len(targets)} targets, collecting Hessians over {num_samples} samples"
        )

        hooks = {name: HessianHook(mod) for name, mod in targets}

        # Move the (shared) pos_emb + attention_mask + position_ids for this
        # layer_type to device once — they don't change across samples.
        pe_shared = shared_pos_emb[layer_type]
        pe_dev = (pe_shared[0].to(device), pe_shared[1].to(device))
        am_shared = shared_mask[layer_type]
        am_dev = am_shared.to(device) if am_shared is not None else None
        pid_dev = shared_pos_ids.to(device)

        with torch.no_grad():
            for s in range(num_samples):
                h = hiddens[s].to(device)
                pli_slice = None
                if pli_paths[s] is not None:
                    pli_full = torch.load(pli_paths[s], weights_only=True, map_location="cpu")
                    pli_slice = pli_full[:, :, layer_idx, :].to(device)
                    del pli_full
                _ = layer(
                    h,
                    pli_slice,
                    shared_kv_states=kv_dicts[s],
                    position_embeddings=pe_dev,
                    attention_mask=am_dev,
                    position_ids=pid_dev,
                    past_key_values=None,
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Quantize each target module using its captured Hessian.
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
            log(
                f"[gptq]   {name}: shape={tuple(W.shape)} "
                f"H_N={hook.N} took {elapsed:.1f}s"
            )
            mod.weight.data.copy_(Q_dq.to(mod.weight.dtype))
            quantized[name] = (nibbles.cpu(), scale_t.cpu(), zero_t.cpu())
            del W, Q_dq, nibbles, scale_t, zero_t, H

        # Re-run with quantized weights so layer N+1 sees post-quant activations.
        # pe_dev / am_dev / pid_dev are already staged on device above and are
        # reused here without re-uploading.
        new_hiddens: list[torch.Tensor] = []
        with torch.no_grad():
            for s in range(num_samples):
                h = hiddens[s].to(device)
                pli_slice = None
                if pli_paths[s] is not None:
                    pli_full = torch.load(pli_paths[s], weights_only=True, map_location="cpu")
                    pli_slice = pli_full[:, :, layer_idx, :].to(device)
                    del pli_full
                out = layer(
                    h,
                    pli_slice,
                    shared_kv_states=kv_dicts[s],
                    position_embeddings=pe_dev,
                    attention_mask=am_dev,
                    position_ids=pid_dev,
                    past_key_values=None,
                )
                if isinstance(out, tuple):
                    out = out[0]
                new_hiddens.append(out.detach().cpu())
            if device.type == "cuda":
                torch.cuda.empty_cache()
        hiddens = new_hiddens

        # Atomic-rename checkpoint after each completed layer. On the next run,
        # passing the same ckpt_path will resume from layer_idx+1.
        save_checkpoint(layer_idx + 1)

        # Force Python GC + CUDA caching-allocator release at layer boundary.
        # With double-wide layers (15+ on E2B) each Hessian alone is 604 MB
        # F32, and Python's allocator fragments across layers. On a shared-
        # memory APU we need the freed blocks back in the common pool.
        import gc
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Clean up the per_layer_inputs spill directory + checkpoint now that the
    # bake has completed successfully.
    import shutil
    shutil.rmtree(pli_cache_dir, ignore_errors=True)
    if ckpt_path is not None and ckpt_path.exists():
        ckpt_path.unlink()

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
        n = chunk.numel() - 1
        nll_sum += out.loss.float().item() * n
        tok_sum += n
    return math.exp(nll_sum / tok_sum)


# ---------------------------------------------------------------------------
# Name mapping / manifest writer (shared shape with bake_int4.py)
# ---------------------------------------------------------------------------
def _load_raw_tensor_names(model_dir: Path) -> set[str]:
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
    # Gemma 4's multimodal wrapper keeps the `.language_model.` infix in the
    # state dict for `AutoModelForImageTextToText`, so the identity map works.
    # Defensive fallbacks covered in case a future version collapses the hierarchy.
    candidates = [
        lambda n: n,
        lambda n: n.replace("model.", "model.language_model.", 1)
                  if n.startswith("model.") and not n.startswith("model.language_model.")
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


class StreamingTensorWriter:
    """Write tensors to `weights.bin` one at a time so we never hold more than
    one tensor's bytes buffer in host memory at once. Entries accumulate into
    a manifest that's written on `.close()`.

    The prior implementation buffered every tensor's bytes in a list before
    opening the output file — for a ~6 GiB Gemma 4 bake that spike collided
    with the live model on GPU and OOM-killed the process on 15.2 GiB iGPUs.
    Streaming keeps peak host memory tied to a single tensor.

    Entries must be sorted by name externally (to match the earlier package
    layout); this writer accepts them in whatever order the caller provides.
    """

    def __init__(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.weights_path = out_dir / "weights.bin"
        self.f = open(self.weights_path, "wb")
        self.entries: list[dict] = []
        self.cursor = 0

    def add(self, name: str, data: bytes, shape: list[int],
            dtype_str: str, layout: str) -> None:
        offset = align_up(self.cursor, 4096)
        if offset > self.cursor:
            self.f.write(b"\x00" * (offset - self.cursor))
        self.f.write(data)
        byte_len = len(data)
        self.entries.append({
            "name": name,
            "shape": shape,
            "dtype": dtype_str,
            "layout": layout,
            "offset": offset,
            "byte_len": byte_len,
        })
        self.cursor = offset + byte_len

    def close(self, model_family: str = "gemma4") -> None:
        self.f.close()
        # Emit the manifest with tensors in sorted order so the bake on disk
        # matches the pre-streaming convention — makes byte-comparing two
        # bakes easier.
        sorted_entries = sorted(self.entries, key=lambda e: e["name"])
        manifest = {
            "format_version": FORMAT_VERSION,
            "converter_version": CONVERTER_VERSION,
            "model_family": model_family,
            "tensors": sorted_entries,
        }
        with open(self.out_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        log(f"[bake-int4] wrote {self.cursor / (1024 * 1024):.1f} MiB "
            f"to {self.weights_path}")
        log(f"[bake-int4] manifest: {self.out_dir / 'manifest.json'}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPTQ INT4 bake for Gemma 4")
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--damp", type=float, default=0.01)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-ppl", action="store_true",
                   help="Skip the WikiText-2 perplexity check "
                        "(recommended on small-VRAM GPUs to avoid OOM).")
    p.add_argument("--sanity-generate", action="store_true",
                   help="Run a tiny `The quick brown fox` generation on the "
                        "quantized model before serialising. Off by default; "
                        "enabling it needs the model to stay on GPU until "
                        "after serialisation on small-VRAM machines, which "
                        "can OOM.")
    p.add_argument("--ppl-chunks", type=int, default=16)
    p.add_argument("--out-dir", default=None, type=Path,
                   help="Default: {model-dir}/.supersonic/v{FORMAT_VERSION}-int4-gptq")
    p.add_argument("--ckpt-path", default=None, type=Path,
                   help="Path to a layer-level checkpoint file. If the file "
                        "exists, the bake resumes from the last completed "
                        "layer (skipping earlier GPTQ work) unless --fresh is "
                        "set. Default: {out-dir}/bake_ckpt.pt")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore any existing checkpoint at --ckpt-path and "
                        "start the GPTQ sweep from layer 0.")
    return p.parse_args()


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
        log("[bake-int4] WARNING: running GPTQ on CPU will be slow. "
            "Consider --device cuda if you have a GPU.")

    # --- Load model + tokenizer via the multimodal wrapper ---
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    log(f"[bake-int4] loading tokenizer from {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    log(f"[bake-int4] loading model (bf16) from {model_dir}")
    # On shared-memory APUs we can't afford the CPU→GPU staging spike that
    # `from_pretrained(...).to(device)` causes (peak = 2× model size). Use
    # `device_map` so HF loads each tensor straight to the target device via
    # meta-tensor init + per-shard materialization. For CPU bakes the
    # `device_map="cpu"` path is equivalent to the legacy load.
    device_map = str(device) if device.type != "cpu" else "cpu"
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    model.eval()

    # Navigate to the text language model — that's what our decoder layers live in.
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        language_model = model.model.language_model
    else:
        raise SystemExit("could not locate `.model.language_model` on loaded model")

    raw_keys = _load_raw_tensor_names(model_dir)
    weight_prefix = None
    for k in raw_keys:
        if k.endswith(".embed_tokens.weight") and "language_model" in k:
            weight_prefix = k[: -len(".embed_tokens.weight")]
            break
    if weight_prefix is None:
        for k in raw_keys:
            if k.endswith(".embed_tokens.weight"):
                weight_prefix = k[: -len(".embed_tokens.weight")]
                break
    if weight_prefix is None:
        raise SystemExit("could not infer weight prefix from safetensors keys")
    log(f"[bake-int4] weight prefix (safetensors): {weight_prefix!r}")

    sd_keys = list(model.state_dict().keys())
    hf_to_raw = _build_name_map(sd_keys, raw_keys)
    missing = [k for k in sd_keys if k not in hf_to_raw]
    if missing:
        log(f"[bake-int4] {len(missing)} state-dict tensors have no safetensors "
            "counterpart (e.g. tied weights) — skipping.")

    text_cfg = language_model.config
    num_layers = text_cfg.num_hidden_layers
    layer_types = list(text_cfg.layer_types)
    log(f"[bake-int4] num_layers={num_layers} "
        f"full_layers={sum(1 for t in layer_types if t == 'full_attention')} "
        f"num_kv_shared_layers={text_cfg.num_kv_shared_layers}")

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
            f"not enough calibration tokens ({ids.numel()}) for seqlen={args.seqlen}"
        )
    torch.manual_seed(args.seed)
    max_start = ids.numel() - args.seqlen - 1
    starts = torch.randint(0, max_start, (args.num_samples,))
    calib = torch.stack([ids[s:s + args.seqlen] for s in starts])
    log(f"[bake-int4] calibration batch: {tuple(calib.shape)}")

    # --- GPTQ ---
    # Resolve the checkpoint path. Default: next to the output bake dir so
    # reruns with the same `--out-dir` automatically share one checkpoint.
    out_dir = args.out_dir or (
        model_dir / ".supersonic" / f"v{FORMAT_VERSION}-int4-gptq"
    )
    ckpt_path: Path = args.ckpt_path or (out_dir / "bake_ckpt.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and ckpt_path.exists():
        log(f"[bake-int4] --fresh: removing existing checkpoint {ckpt_path}")
        ckpt_path.unlink()
    elif ckpt_path.exists():
        log(f"[bake-int4] resuming from checkpoint {ckpt_path} "
            "(pass --fresh to restart from layer 0)")

    t0 = time.perf_counter()
    quantized = quantize_gemma4(
        language_model, calib, device,
        group_size=args.group_size,
        damp=args.damp,
        ckpt_path=ckpt_path,
        fresh=args.fresh,
    )
    elapsed = time.perf_counter() - t0
    log(f"[bake-int4] GPTQ done in {elapsed / 60.0:.1f} min "
        f"({len(quantized)} tensors quantized)")

    # --- Sample generation sanity check ---
    if args.sanity_generate:
        try:
            sample_ids = tokenizer("The quick brown fox", return_tensors="pt"
                                   ).input_ids.to(device)
            with torch.no_grad():
                gen = model.generate(sample_ids, max_new_tokens=12,
                                     do_sample=False, use_cache=True)
            log(f"[bake-int4] sample gen (post-quant): "
                f"{tokenizer.decode(gen[0], skip_special_tokens=True)!r}")
        except Exception as ex:
            log(f"[bake-int4] sample gen failed: {ex}")

    # Move the live model to CPU before serialisation. On constrained iGPUs the
    # BF16 model plus Python's per-tensor host buffers during serialisation can
    # OOM-kill the process — moving to CPU frees ~model_size of VRAM and shifts
    # the serialisation reads to host memory. The self-check below and the
    # serialisation loop both only need CPU tensors.
    if device.type != "cpu":
        log("[bake-int4] moving model to CPU before serialisation")
        model = model.to("cpu")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        # language_model is an attribute of model; it also moves with it.

    # --- Self-consistency check ---
    try:
        gs = args.group_size
        lm_modules = dict(language_model.named_modules())
        mismatch = 0
        worst: tuple[str, float] = ("", 0.0)
        for tensor_name, (nibbles_s, scale_s, zero_s) in quantized.items():
            rows, cols = nibbles_s.shape
            row_gr = torch.arange(rows) // gs
            col_gc = torch.arange(cols) // gs
            sc_full = scale_s[row_gr][:, col_gc]
            zf_full = zero_s[row_gr][:, col_gc]
            recon = nibbles_s.float() * sc_full - zf_full * sc_full
            recon = recon.to(torch.bfloat16).to(torch.float32)
            # tensor_name is like "layers.0.mlp.gate_proj.weight" relative to
            # language_model. Strip ".weight" to find the module.
            mod = lm_modules.get(tensor_name[: -len(".weight")])
            if mod is None:
                continue
            live = mod.weight.data.to(torch.float32).cpu()
            if not torch.equal(recon, live):
                mismatch += 1
                linf = (recon - live).abs().max().item()
                if linf > worst[1]:
                    worst = (tensor_name, linf)
        log(f"[self-check] INT4 tensors: {mismatch}/{len(quantized)} mismatch"
            + (f" (worst: {worst[0]} Linf={worst[1]:.2e})" if mismatch else ""))
    except Exception as ex:
        log(f"[self-check] failed: {ex}")

    # --- Perplexity ---
    if not args.skip_ppl:
        # PPL needs the model back on GPU. Only run if the caller explicitly
        # opts in — it's the single biggest OOM risk on small-VRAM machines
        # since model has to go back to device alongside long-sequence
        # activations for the test corpus.
        try:
            if device.type != "cpu":
                log("[bake-int4] moving model back to GPU for PPL check")
                model = model.to(device)
            log("[bake-int4] running perplexity sanity check on WikiText-2 test...")
            ppl = compute_ppl(model, tokenizer, device,
                              seqlen=args.seqlen, n_chunks=args.ppl_chunks)
            log(f"[bake-int4] PPL: {ppl:.2f}")
            if device.type != "cpu":
                model = model.to("cpu")
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception as ex:
            log(f"[bake-int4] PPL check failed: {ex}")

    # --- Assemble tensors for writing ---
    log("[bake-int4] serialising tensors...")
    # The calibration loop prefixes target names relative to `language_model`
    # (e.g. "layers.0.mlp.gate_proj.weight"); to resolve against the full raw
    # safetensors namespace we prepend the weight_prefix + "." (e.g.
    # "model.language_model.layers.0.mlp.gate_proj.weight"). Build this map once.
    lm_prefix_map: dict[str, str] = {}
    for lm_name in quantized:
        raw_name = f"{weight_prefix}.{lm_name}"
        if raw_name in raw_keys:
            lm_prefix_map[lm_name] = raw_name
        else:
            log(f"[bake-int4] WARNING: quantized tensor {lm_name!r} has no "
                f"safetensors counterpart at {raw_name!r}, skipping")

    # Walk the top-level model's state dict and emit under raw names.
    eligible = [
        hf_name for hf_name in sd_keys
        if hf_name in hf_to_raw
        and not hf_to_raw[hf_name].endswith("_scale_inv")
        and hf_to_raw[hf_name].startswith(f"{weight_prefix}.")
    ]
    eligible.sort()

    # Build a reverse lookup from raw-name -> language_model-relative name.
    raw_to_lm: dict[str, str] = {v: k for k, v in lm_prefix_map.items()}

    out_dir = args.out_dir or (
        model_dir / ".supersonic" / f"v{FORMAT_VERSION}-int4-gptq"
    )
    writer = StreamingTensorWriter(out_dir)
    sd = model.state_dict()

    # Serialise in alphabetical order by raw name — matches the previous
    # buffered-write convention, keeps manifest entries stable across runs.
    for hf_name in eligible:
        raw_name = hf_to_raw[hf_name]
        t = sd[hf_name]
        shape = list(t.shape)
        lm_name = raw_to_lm.get(raw_name)
        if lm_name is not None:
            # INT4-quantized tensor: emit (packed, scale, zero) trio. Convert
            # + write one tensor at a time and drop references so the bytes
            # buffer is freed before the next tensor is materialised.
            nibbles, scale_t, zero_t = quantized[lm_name]
            packed = pack_nibbles(nibbles)
            packed_bytes = packed.numpy().tobytes()
            writer.add(
                raw_name, packed_bytes,
                [packed.shape[0], packed.shape[1]],
                "u8", LAYOUT_INT4,
            )
            del packed, packed_bytes
            scale_bytes = bf16_to_bytes(scale_t)
            writer.add(
                f"{raw_name}_int4_scale", scale_bytes,
                list(scale_t.shape), "bf16", LAYOUT_RAW,
            )
            del scale_bytes
            zero_bytes = bf16_to_bytes(zero_t)
            writer.add(
                f"{raw_name}_int4_zero", zero_bytes,
                list(zero_t.shape), "bf16", LAYOUT_RAW,
            )
            del zero_bytes
            # Drop the GPTQ-side tensors now — they were cloned to CPU but
            # they add up to hundreds of MB across the full sweep, and we
            # still have all non-quantized tensors (norms/embeds) to dump.
            quantized[lm_name] = None  # release references
        else:
            # Non-quantized tensor: store as-is (BF16 / F32 / scalar).
            dtype_str = torch_dtype_to_str(t.dtype)
            data = tensor_to_bytes(t, dtype_str)
            writer.add(raw_name, data, shape, dtype_str, LAYOUT_RAW)
            del data

    writer.close(model_family="gemma4")
    log(f"[bake-int4] done. Output: {out_dir}")


if __name__ == "__main__":
    main()
