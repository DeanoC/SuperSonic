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
FORMAT_VERSION = 1
CONVERTER_VERSION = 2

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
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Sequential GPTQ over Gemma 4 text layers. Returns
    {tensor_name: (nibbles, scale_f32, zero_f32)}.
    """
    language_model.eval()
    config = language_model.config
    layers = language_model.layers
    num_layers = len(layers)
    num_samples = calib_ids.shape[0]

    # Map nn.Linear -> state-dict weight name (scoped to the language_model subtree).
    module_to_name: dict[int, str] = {}
    for name, mod in language_model.named_modules():
        if isinstance(mod, nn.Linear):
            module_to_name[id(mod)] = name + ".weight"

    # --- Capture pre-layer state per sample ---
    log(f"[gptq] capturing pre-layer state for {num_samples} samples...")
    hiddens: list[torch.Tensor] = []
    per_layer_inputs_list: list[torch.Tensor] = []
    pos_emb_list: list[dict] = []
    mask_list: list[dict] = []
    pos_ids_list: list[torch.Tensor] = []

    for s in range(num_samples):
        ids = calib_ids[s:s + 1].to(device)
        with torch.no_grad():
            h, pli, pe, am, pid = compute_pre_layer_state(language_model, ids)
        hiddens.append(h.detach().cpu())
        per_layer_inputs_list.append(
            pli.detach().cpu() if pli is not None else None
        )
        # Keep masks/pos_embs/pos_ids on CPU to avoid OOM; move to device per-layer.
        pos_emb_list.append({k: (v[0].detach().cpu(), v[1].detach().cpu()) for k, v in pe.items()})
        mask_list.append({k: (v.detach().cpu() if v is not None else None) for k, v in am.items()})
        pos_ids_list.append(pid.detach().cpu())

    # shared_kv_states carries across layers within one sample, so allocate one
    # dict per sample. Python dicts are live references — layers mutate them.
    kv_dicts: list[dict] = [{} for _ in range(num_samples)]

    quantized: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    for layer_idx in range(num_layers):
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

        with torch.no_grad():
            for s in range(num_samples):
                h = hiddens[s].to(device)
                pli_slice = (
                    per_layer_inputs_list[s][:, :, layer_idx, :].to(device)
                    if per_layer_inputs_list[s] is not None
                    else None
                )
                pe = pos_emb_list[s][layer_type]
                pe_dev = (pe[0].to(device), pe[1].to(device))
                am = mask_list[s][layer_type]
                am_dev = am.to(device) if am is not None else None
                pid = pos_ids_list[s].to(device)
                _ = layer(
                    h,
                    pli_slice,
                    shared_kv_states=kv_dicts[s],
                    position_embeddings=pe_dev,
                    attention_mask=am_dev,
                    position_ids=pid,
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
        new_hiddens: list[torch.Tensor] = []
        with torch.no_grad():
            for s in range(num_samples):
                h = hiddens[s].to(device)
                pli_slice = (
                    per_layer_inputs_list[s][:, :, layer_idx, :].to(device)
                    if per_layer_inputs_list[s] is not None
                    else None
                )
                pe = pos_emb_list[s][layer_type]
                pe_dev = (pe[0].to(device), pe[1].to(device))
                am = mask_list[s][layer_type]
                am_dev = am.to(device) if am is not None else None
                pid = pos_ids_list[s].to(device)
                out = layer(
                    h,
                    pli_slice,
                    shared_kv_states=kv_dicts[s],
                    position_embeddings=pe_dev,
                    attention_mask=am_dev,
                    position_ids=pid,
                    past_key_values=None,
                )
                if isinstance(out, tuple):
                    out = out[0]
                new_hiddens.append(out.detach().cpu())
            if device.type == "cuda":
                torch.cuda.empty_cache()
        hiddens = new_hiddens

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


def write_package(
    out_dir: Path,
    tensors: list[tuple[str, bytes, list[int], str, str]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
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
        "model_family": "gemma4",
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
    p = argparse.ArgumentParser(description="GPTQ INT4 bake for Gemma 4")
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--damp", type=float, default=0.01)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--ppl-chunks", type=int, default=16)
    p.add_argument("--out-dir", default=None, type=Path,
                   help="Default: {model-dir}/.supersonic/v{FORMAT_VERSION}-int4-gptq")
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
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir), torch_dtype=torch.bfloat16
    )
    model.eval()
    if device.type != "cpu":
        model = model.to(device)

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
    t0 = time.perf_counter()
    quantized = quantize_gemma4(
        language_model, calib, device,
        group_size=args.group_size,
        damp=args.damp,
    )
    elapsed = time.perf_counter() - t0
    log(f"[bake-int4] GPTQ done in {elapsed / 60.0:.1f} min "
        f"({len(quantized)} tensors quantized)")

    # --- Sample generation sanity check ---
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
        log("[bake-int4] running perplexity sanity check on WikiText-2 test...")
        try:
            ppl = compute_ppl(model, tokenizer, device,
                              seqlen=args.seqlen, n_chunks=args.ppl_chunks)
            log(f"[bake-int4] PPL: {ppl:.2f}")
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

    sd = model.state_dict()
    tensors_out: list[tuple[str, bytes, list[int], str, str]] = []

    for hf_name in eligible:
        raw_name = hf_to_raw[hf_name]
        t = sd[hf_name]
        shape = list(t.shape)

        lm_name = raw_to_lm.get(raw_name)
        if lm_name is not None:
            # INT4-quantized tensor: emit (packed, scale, zero) trio.
            nibbles, scale_t, zero_t = quantized[lm_name]
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
            # Non-quantized tensor: store as-is (BF16 / F32 / scalar).
            dtype_str = torch_dtype_to_str(t.dtype)
            data = tensor_to_bytes(t, dtype_str)
            tensors_out.append((raw_name, data, shape, dtype_str, LAYOUT_RAW))

    tensors_out.sort(key=lambda x: x[0])

    out_dir = args.out_dir or (
        model_dir / ".supersonic" / f"v{FORMAT_VERSION}-int4-gptq"
    )
    write_package(out_dir, tensors_out)
    log(f"[bake-int4] done. Output: {out_dir}")


if __name__ == "__main__":
    main()
