#!/usr/bin/env python3
"""
Corpus-level Python-oracle ↔ Rust comparison for GPTQ INT4 bakes.

For each prompt:
  1. Run greedy decode through Python, using weights reconstructed from the
     INT4 bake exactly as the Rust kernel would (packed nibbles + bf16
     scale + bf16 zero → bf16(q*s - zf*s)). No live HF quantisation.
  2. Run greedy decode through `cargo run --release --bin supersonic --int4`.
  3. Report per-prompt first divergence token + whole-string match.

Exits 0 if all prompts match to max-new-tokens, non-zero otherwise.

Usage:
    python oracle/int4_corpus_compare.py --model-dir <DIR> \\
        [--prompts oracle/int4_corpus_prompts.json] \\
        [--max-new-tokens 32]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PROMPTS = [
    "The quick brown fox",
    "Once upon a time, in a land far away,",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The capital of France is",
    "In 1969, humans first walked on the",
    "To make a grilled cheese sandwich, you will need",
    "The Pythagorean theorem states that",
    "A short poem about the sea:\n",
    "Translate to French: Good morning, how are you?\nAnswer:",
    "The three primary colors are",
    "Water boils at 100 degrees Celsius at sea level because",
    "Write a haiku about winter.\n",
]
HYBRID_ATTENTION_LOOP_THRESHOLD = 10


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Load baked INT4 package and reconstruct f32 weights matching the kernel.
# ---------------------------------------------------------------------------
def load_bake(bake_dir: Path) -> tuple[dict, np.memmap]:
    manifest = json.loads((bake_dir / "manifest.json").read_text())
    weights = np.memmap(bake_dir / "weights.bin", dtype=np.uint8, mode="r")
    return manifest, weights


def bf16_bytes_to_f32(arr_i16: np.ndarray) -> np.ndarray:
    u32 = (arr_i16.astype(np.uint32) & 0xFFFF) << 16
    return u32.view(np.float32).astype(np.float32)


def f32_to_bf16_rounded(arr_f32: np.ndarray) -> np.ndarray:
    """Round f32 array through bf16 (round-to-nearest-even)."""
    u32 = arr_f32.view(np.uint32).copy()
    # RNE: add 0x7FFF + lsb-of-target to bits, then mask
    bias = ((u32 >> 16) & 1) + 0x7FFF
    u32 = (u32 + bias) & 0xFFFF0000
    return u32.view(np.float32)


def dequant_int4_tensor(manifest_by_name: dict, weights: np.memmap,
                        name: str) -> np.ndarray:
    """Reconstruct the f32 weight matrix for an INT4 tensor, matching the kernel."""
    t = manifest_by_name[name]
    assert t["layout"] == "Int4Quantized"
    rows, packed_cols = t["shape"]
    cols = packed_cols * 2
    raw = np.asarray(weights[t["offset"]: t["offset"] + t["byte_len"]]).reshape(rows, packed_cols)
    nib = np.empty((rows, cols), dtype=np.uint8)
    nib[:, 0::2] = raw & 0xF
    nib[:, 1::2] = (raw >> 4) & 0xF

    sc_meta = manifest_by_name[name + "_int4_scale"]
    zr_meta = manifest_by_name[name + "_int4_zero"]
    sc_i16 = np.frombuffer(
        weights[sc_meta["offset"]: sc_meta["offset"] + sc_meta["byte_len"]],
        dtype=np.int16,
    ).reshape(sc_meta["shape"])
    zr_i16 = np.frombuffer(
        weights[zr_meta["offset"]: zr_meta["offset"] + zr_meta["byte_len"]],
        dtype=np.int16,
    ).reshape(zr_meta["shape"])
    sc = bf16_bytes_to_f32(sc_i16)
    zr = bf16_bytes_to_f32(zr_i16)

    # Kernel group size — infer from shapes
    gs_row = rows // sc.shape[0]
    gs_col = cols // sc.shape[1]
    assert gs_row == gs_col, f"non-square group? {gs_row} vs {gs_col}"
    gs = gs_row
    row_gr = np.arange(rows) // gs
    col_gc = np.arange(cols) // gs
    sc_full = sc[row_gr][:, col_gc]
    zr_full = zr[row_gr][:, col_gc]
    # Match kernel: (nibble * s) - (zf * s) then round through bf16
    recon = nib.astype(np.float32) * sc_full - zr_full * sc_full
    return f32_to_bf16_rounded(recon)


def load_raw_tensor(manifest_by_name: dict, weights: np.memmap,
                    name: str) -> np.ndarray:
    """Load a non-INT4 tensor from the bake as a numpy array."""
    t = manifest_by_name[name]
    b = weights[t["offset"]: t["offset"] + t["byte_len"]]
    if t["dtype"] == "bf16":
        arr = bf16_bytes_to_f32(np.frombuffer(b, dtype=np.int16))
    elif t["dtype"] == "f32":
        arr = np.frombuffer(b, dtype=np.float32).copy()
    elif t["dtype"] == "f16":
        arr = np.frombuffer(b, dtype=np.float16).astype(np.float32)
    elif t["dtype"] == "u8":
        arr = np.frombuffer(b, dtype=np.uint8).copy()
    elif t["dtype"] == "i64":
        arr = np.frombuffer(b, dtype=np.int64).astype(np.float32)
    else:
        raise ValueError(f"unsupported dtype {t['dtype']}")
    return arr.reshape(t["shape"])


# ---------------------------------------------------------------------------
# Build a HF model with weights taken from the bake (not from safetensors).
# ---------------------------------------------------------------------------
def build_oracle_model(model_dir: Path, bake_dir: Path, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    log(f"[oracle] loading HF model (bf16) to {device}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception as exc:
        log(f"[oracle] tokenizer remote code failed ({exc}); retrying with built-in tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=False)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
    except Exception as exc:
        log(f"[oracle] remote model code failed ({exc}); retrying with built-in architecture")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
        ).to(device)
    model.eval()

    manifest, weights = load_bake(bake_dir)
    by_name = {t["name"]: t for t in manifest["tensors"]}
    log(f"[oracle] bake has {len(by_name)} tensors")

    def split_raw_name_for_fused(hf_wname: str) -> list[str] | None:
        if hf_wname.endswith(".self_attn.qkv_proj.weight"):
            prefix = hf_wname[: -len(".self_attn.qkv_proj.weight")]
            return [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
        if hf_wname.endswith(".mlp.gate_up_proj.weight"):
            prefix = hf_wname[: -len(".mlp.gate_up_proj.weight")]
            return [
                f"{prefix}.mlp.gate_proj.weight",
                f"{prefix}.mlp.up_proj.weight",
            ]
        return None

    # Build translation from HF state-dict name to bake raw name.
    sd_keys = list(model.state_dict().keys())
    hf_to_raw: dict[str, str] = {}
    for k in sd_keys:
        if k in by_name:
            hf_to_raw[k] = k
            continue
        # Try inserting ".language_model."
        if k.startswith("model.") and not k.startswith("model.language_model."):
            cand = k.replace("model.", "model.language_model.", 1)
            if cand in by_name:
                hf_to_raw[k] = cand

    # Patch every INT4 weight in the HF model with the bake's dequantised values.
    # Phi-4's HF module keeps qkv_proj and gate_up_proj fused, while the
    # SuperSonic bake splits them into per-projection shards; stitch those back
    # together along the output-row axis before copying into the module.
    int4_count = 0
    int4_fused_count = 0
    for mod_name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        hf_wname = mod_name + ".weight"
        raw_wname = hf_to_raw.get(hf_wname)
        if raw_wname is not None:
            meta = by_name.get(raw_wname)
            if meta is None or meta["layout"] != "Int4Quantized":
                continue
            recon = dequant_int4_tensor(by_name, weights, raw_wname)
        else:
            shard_names = split_raw_name_for_fused(hf_wname)
            if not shard_names or not all(
                by_name.get(n, {}).get("layout") == "Int4Quantized"
                for n in shard_names
            ):
                continue
            recon = np.concatenate(
                [dequant_int4_tensor(by_name, weights, n) for n in shard_names],
                axis=0,
            )
            int4_fused_count += 1
        assert recon.shape == tuple(mod.weight.shape), \
            f"shape mismatch for {raw_wname}: {recon.shape} vs {tuple(mod.weight.shape)}"
        with torch.no_grad():
            # Already bf16-rounded values in f32; .to(bfloat16) is exact.
            mod.weight.data.copy_(torch.from_numpy(recon).to(torch.bfloat16).to(device))
        int4_count += 1
    log(
        f"[oracle] patched {int4_count} INT4 linear layers with "
        f"kernel-accurate weights ({int4_fused_count} fused)"
    )

    # Patch A_log (was F32, HF downcasts to bf16; bake stores exp(F32)->bf16).
    alog_count = 0
    for sd_name, t in by_name.items():
        if not sd_name.endswith(".A_log") or t["layout"] != "HeadExpReshaped":
            continue
        # Find matching HF tensor
        hf_name = None
        for k in sd_keys:
            if hf_to_raw.get(k) == sd_name:
                hf_name = k
                break
        if hf_name is None:
            continue
        # bake stored exp(F32) in bf16 with shape [1, 1, H] — the model stores
        # log-A (F32) in shape [H]. Invert: log(exp(F32)) = the stored value's
        # ln; we want the model's A_log to be ln(bake's stored bf16 value).
        # Simpler: don't patch A_log — it's tiny and the bf16 cast matters
        # less after exp() consumption in the kernel. Leave as HF-loaded.
        alog_count += 1
    log(f"[oracle] {alog_count} A_log tensors present (left as HF-loaded)")

    return tokenizer, model


def install_deterministic_projection_oracle(
    model,
    include_lm_head: bool = False,
    attention_mode: str = "hybrid",
) -> dict:
    """Patch decode math to use kernel-style F32 ops plus BF16 boundaries.

    Phi-4 CUDA trace replay validates projections with deterministic F32 dot
    products and BF16 output boundaries. This opt-in oracle mode also patches
    Phi-style RMSNorm and SwiGLU boundaries so the HF loop does not keep using
    live PyTorch BF16 choices around those projections.
    """

    patched_linear = 0
    patched_norm = 0
    patched_mlp = 0
    patched_attn = 0
    patched_names: list[str] = []

    def deterministic_linear_forward(module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(
            x.float(),
            module.weight.float(),
            module.bias.float() if module.bias is not None else None,
        )
        return out.to(torch.bfloat16)

    def deterministic_rms_norm_forward(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        src = hidden_states.float()
        variance = src.pow(2).mean(-1, keepdim=True)
        out = src * torch.rsqrt(variance + module.variance_epsilon) * module.weight.float()
        return out.to(torch.bfloat16)

    def deterministic_mlp_forward(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = module.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        silu = gate.float() / (1.0 + torch.exp(-gate.float()))
        up_states = (silu * up.float()).to(torch.bfloat16)
        return module.down_proj(up_states)

    def deterministic_attention_forward(
        module: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb, repeat_kv

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)
        qkv = module.qkv_proj(hidden_states)
        query_pos = module.config.num_attention_heads * module.head_dim
        kv_pos = module.num_key_value_heads * module.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + kv_pos]
        value_states = qkv[..., query_pos + kv_pos :]

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        query_states = query_states.to(torch.bfloat16)
        key_states = key_states.to(torch.bfloat16)
        value_states = value_states.to(torch.bfloat16)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                module.layer_idx,
            )

        key_states = repeat_kv(key_states, module.num_key_value_groups)
        value_states = repeat_kv(value_states, module.num_key_value_groups)
        q_f32 = query_states.float()
        k_f32 = key_states.float()
        v_f32 = value_states.float()
        use_loop_attention = (
            q_f32.shape[2] == 1
            and getattr(module, "_supersonic_loop_attention", attention_mode == "loop")
        )
        if use_loop_attention:
            score_rows = []
            for pos in range(k_f32.shape[2]):
                score = torch.zeros_like(q_f32[:, :, 0, 0])
                for dim in range(module.head_dim):
                    score = score + q_f32[:, :, 0, dim] * k_f32[:, :, pos, dim]
                score_rows.append(score * module.scaling)
            attn_weights = torch.stack(score_rows, dim=-1).unsqueeze(2)
        else:
            attn_weights = torch.matmul(q_f32, k_f32.transpose(2, 3)) * module.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_probs = torch.softmax(attn_weights, dim=-1, dtype=torch.float32)
        if use_loop_attention:
            acc = torch.zeros_like(v_f32[:, :, 0:1, :])
            for pos in range(v_f32.shape[2]):
                acc = acc + attn_probs[:, :, :, pos : pos + 1] * v_f32[:, :, pos : pos + 1, :]
            attn_output = acc.to(torch.bfloat16)
        else:
            attn_output = torch.matmul(attn_probs, v_f32).to(torch.bfloat16)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        return module.o_proj(attn_output), attn_probs.to(torch.bfloat16)

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name == "lm_head" and not include_lm_head:
            continue
        module.forward = types.MethodType(deterministic_linear_forward, module)
        patched_linear += 1
        patched_names.append(name)

    for name, module in model.named_modules():
        if hasattr(module, "variance_epsilon") and hasattr(module, "weight"):
            module.forward = types.MethodType(deterministic_rms_norm_forward, module)
            patched_norm += 1
            patched_names.append(name)
        if hasattr(module, "gate_up_proj") and hasattr(module, "down_proj"):
            module.forward = types.MethodType(deterministic_mlp_forward, module)
            patched_mlp += 1
            patched_names.append(name)
        if hasattr(module, "qkv_proj") and hasattr(module, "o_proj"):
            module._supersonic_attention_mode = attention_mode
            module._supersonic_loop_attention = attention_mode == "loop"
            module.forward = types.MethodType(deterministic_attention_forward, module)
            patched_attn += 1
            patched_names.append(name)

    return {
        "enabled": True,
        "patched_linear_modules": patched_linear,
        "patched_rms_norm_modules": patched_norm,
        "patched_mlp_modules": patched_mlp,
        "patched_attention_modules": patched_attn,
        "patched_lm_head": include_lm_head,
        "attention_mode": attention_mode,
        "patched_module_names": patched_names,
        "semantics": (
            "Linear/RMSNorm/attention/SwiGLU use F32 arithmetic with explicit "
            "BF16 output rounding at kernel-style boundaries"
        ),
    }


# ---------------------------------------------------------------------------
# Greedy decode for comparison
# ---------------------------------------------------------------------------
def collect_eos_token_ids(tokenizer, model) -> set[int]:
    eos_ids: set[int] = set()

    def add(value) -> None:
        if value is None:
            return
        if isinstance(value, int):
            eos_ids.add(value)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add(item)

    add(getattr(tokenizer, "eos_token_id", None))
    add(getattr(getattr(model, "config", None), "eos_token_id", None))
    add(getattr(getattr(model, "generation_config", None), "eos_token_id", None))
    return eos_ids


@torch.no_grad()
def topk_payload(logits: torch.Tensor, tokenizer, top_k: int) -> list[dict]:
    if top_k <= 0:
        return []
    logits_cpu = logits.float().detach().cpu()
    k = min(top_k, logits_cpu.numel())
    vals, idx = torch.topk(logits_cpu, k)
    return [
        {
            "id": int(i),
            "logit": float(v),
            "text": tokenizer.decode([int(i)], skip_special_tokens=False),
        }
        for v, i in zip(vals, idx)
    ]


def set_hybrid_attention_mode(model, kv_len_after_call: int, force_loop: bool) -> None:
    use_loop = force_loop or kv_len_after_call >= HYBRID_ATTENTION_LOOP_THRESHOLD
    for module in model.modules():
        if getattr(module, "_supersonic_attention_mode", None) == "hybrid":
            module._supersonic_loop_attention = use_loop


@torch.no_grad()
def python_greedy(tokenizer, model, prompt: str, max_new_tokens: int,
                  device: torch.device, top_k: int = 16) -> tuple[list[int], str, list[list[dict]]]:
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    force_loop_attention = ids.shape[1] >= HYBRID_ATTENTION_LOOP_THRESHOLD
    eos_token_ids = collect_eos_token_ids(tokenizer, model)
    past = None
    next_id = None
    next_top: list[dict] = []
    for pos in range(ids.shape[1]):
        set_hybrid_attention_mode(model, pos + 1, force_loop_attention)
        out = model(
            input_ids=ids[:, pos:pos + 1],
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        logits = out.logits[0, -1]
        next_id = int(torch.argmax(logits).item())
        next_top = topk_payload(logits, tokenizer, top_k) if top_k > 0 else []

    new_ids: list[int] = []
    top_by_token: list[list[dict]] = []
    for _ in range(max_new_tokens):
        assert next_id is not None
        new_ids.append(next_id)
        top_by_token.append(next_top)
        if next_id in eos_token_ids:
            break
        set_hybrid_attention_mode(model, ids.shape[1] + len(new_ids), force_loop_attention)
        out = model(
            input_ids=torch.tensor([[next_id]], device=device),
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        logits = out.logits[0, -1]
        next_id = int(torch.argmax(logits).item())
        next_top = topk_payload(logits, tokenizer, top_k) if top_k > 0 else []

    text = tokenizer.decode(ids[0].tolist() + new_ids, skip_special_tokens=True)
    return new_ids, text, top_by_token


def rust_greedy(binary: Path, model_variant: str, model_dir: Path,
                prompt: str, max_new_tokens: int) -> tuple[list[int], str]:
    """Run supersonic once, parse generated tokens from its output."""
    cmd = [
        str(binary), "--model", model_variant,
        "--model-dir", str(model_dir),
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--int4",
    ]
    # The corpus script commonly runs under HSA_OVERRIDE_GFX_VERSION=11.0.0
    # so torch-rocm loads its gfx1100 kernels on the iGPU. That override must
    # NOT leak into supersonic — Rust reads the real gfx1150 arch and matches
    # against its registry.
    env = {k: v for k, v in os.environ.items() if k != "HSA_OVERRIDE_GFX_VERSION"}
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"supersonic failed:\n{proc.stderr}\n{proc.stdout}")
    # Parse "[tokens] <id> <id> ..." line
    tokens: list[int] = []
    text_lines: list[str] = []
    for line in proc.stdout.splitlines():
        if line.startswith("[tokens] "):
            tokens = [int(x) for x in line[len("[tokens] "):].split()]
        else:
            text_lines.append(line)
    return tokens, "\n".join(text_lines)


def configure_torch_backend(disable_bf16_reduced_precision_reduction: bool) -> dict:
    if disable_bf16_reduced_precision_reduction and hasattr(
        torch.backends.cuda.matmul,
        "allow_bf16_reduced_precision_reduction",
    ):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "allow_tf32": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
        "allow_bf16_reduced_precision_reduction": getattr(
            torch.backends.cuda.matmul,
            "allow_bf16_reduced_precision_reduction",
            None,
        ),
    }


def rust_materialized_diagnostics(binary: Path, model_variant: str, model_dir: Path,
                                  prompt: str, max_new_tokens: int) -> dict:
    with tempfile.NamedTemporaryFile(prefix="supersonic-rust-logits-", suffix=".json", delete=False) as f:
        dump_path = Path(f.name)
    cmd = [
        str(binary), "--model", model_variant,
        "--model-dir", str(model_dir),
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--int4",
    ]
    env = {k: v for k, v in os.environ.items() if k != "HSA_OVERRIDE_GFX_VERSION"}
    env["SUPERSONIC_PHI4_DUMP_LOGITS"] = str(dump_path)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"supersonic diagnostic failed:\n{proc.stderr}\n{proc.stdout}")
        tokens: list[int] = []
        text_lines: list[str] = []
        for line in proc.stdout.splitlines():
            if line.startswith("[tokens] "):
                tokens = [int(x) for x in line[len("[tokens] "):].split()]
            else:
                text_lines.append(line)
        payload = json.loads(dump_path.read_text())
        return {
            "tokens": tokens,
            "text": "\n".join(text_lines),
            "samples": payload.get("samples", []),
        }
    finally:
        try:
            dump_path.unlink()
        except FileNotFoundError:
            pass


def _logit_for(top: list[dict], token_id: int) -> float | None:
    for item in top:
        if item.get("id") == token_id:
            return float(item["logit"])
    return None


def mismatch_diagnostic(py_ids: list[int], rs_ids: list[int], first_diverge: int,
                        py_top_by_token: list[list[dict]], rust_diag: dict,
                        near_tie_threshold: float) -> dict:
    py_token = py_ids[first_diverge] if first_diverge < len(py_ids) else None
    rust_token = rs_ids[first_diverge] if first_diverge < len(rs_ids) else None
    py_top = py_top_by_token[first_diverge] if first_diverge < len(py_top_by_token) else []
    rust_diag_tokens = rust_diag.get("tokens", [])
    prefix_aligned = rust_diag_tokens[:first_diverge] == rs_ids[:first_diverge]

    if not prefix_aligned:
        return {
            "python_token": py_token,
            "rust_token": rust_token,
            "python_top": py_top,
            "rust_materialized_tokens": rust_diag_tokens,
            "rust_materialized_top": [],
            "logit_gaps": [],
            "near_tie": False,
            "prefix_aligned": False,
        }

    rust_samples = rust_diag.get("samples", [])
    rust_sample = rust_samples[first_diverge] if first_diverge < len(rust_samples) else {}
    rust_top = rust_sample.get("top", [])

    py_chosen_logit = _logit_for(py_top, py_token) if py_token is not None else None
    py_rust_logit = _logit_for(py_top, rust_token) if rust_token is not None else None
    rust_chosen_logit = _logit_for(rust_top, rust_token) if rust_token is not None else None
    rust_py_logit = _logit_for(rust_top, py_token) if py_token is not None else None

    gaps = []
    if py_chosen_logit is not None and py_rust_logit is not None:
        gaps.append(abs(py_chosen_logit - py_rust_logit))
    if rust_chosen_logit is not None and rust_py_logit is not None:
        gaps.append(abs(rust_chosen_logit - rust_py_logit))
    near_tie = bool(gaps) and min(gaps) <= near_tie_threshold
    return {
        "python_token": py_token,
        "rust_token": rust_token,
        "python_top": py_top,
        "rust_materialized_tokens": rust_diag.get("tokens", []),
        "rust_materialized_top": rust_top,
        "logit_gaps": gaps,
        "near_tie": near_tie,
        "prefix_aligned": True,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--model-variant", default="qwen3.5-4b",
                   help="Registry key: qwen3.5-{0.8b,2b,4b,9b}")
    p.add_argument("--bake-subdir", default=".supersonic/v1-int4-gptq")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--prompts", type=Path, default=None,
                   help="JSON file containing a list of prompts. "
                        "Falls back to a built-in default set.")
    p.add_argument("--binary", type=Path,
                   default=Path("target/release/supersonic"),
                   help="Path to the Rust supersonic binary")
    p.add_argument("--device", default=None)
    p.add_argument("--report", type=Path, default=None,
                   help="Optional JSON output path for detailed results")
    p.add_argument("--diagnose-mismatches", action="store_true",
                   help="On token mismatch, rerun Rust with materialized logits and include top-k diagnostics.")
    p.add_argument("--diagnostic-top-k", type=int, default=16)
    p.add_argument("--near-tie-threshold", type=float, default=0.5)
    p.add_argument("--disable-bf16-reduced-precision-reduction", action="store_true",
                   help="Force torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False "
                        "before loading the oracle model. Useful for Phi-4 CUDA accumulation diagnostics.")
    p.add_argument("--deterministic-projection-oracle", action="store_true",
                   help="Patch Python nn.Linear projections to use F32 matmul plus BF16 output rounding. "
                        "Intended for Phi-4 CUDA INT4 parity diagnostics.")
    p.add_argument("--deterministic-lm-head", action="store_true",
                   help="With --deterministic-projection-oracle, also patch lm_head. "
                        "By default only decoder math is patched.")
    p.add_argument("--deterministic-attention-mode",
                   choices=["matmul", "loop", "hybrid"],
                   default="hybrid",
                   help="Attention reduction mode for --deterministic-projection-oracle. "
                        "hybrid uses explicit decode-order loops for longer KV contexts.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch_backend = configure_torch_backend(args.disable_bf16_reduced_precision_reduction)
    log(f"[oracle] torch_backend={torch_backend}")
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Prompts
    if args.prompts and args.prompts.exists():
        prompts = json.loads(args.prompts.read_text())
    else:
        prompts = DEFAULT_PROMPTS
    log(f"[corpus] {len(prompts)} prompts; max_new_tokens={args.max_new_tokens}")

    # Build oracle once
    bake_dir = args.model_dir / args.bake_subdir
    if not bake_dir.exists():
        log(f"ERROR: bake dir not found: {bake_dir}")
        return 2
    tokenizer, model = build_oracle_model(args.model_dir, bake_dir, device)
    projection_oracle = {"enabled": False}
    if args.deterministic_projection_oracle:
        projection_oracle = install_deterministic_projection_oracle(
            model,
            include_lm_head=args.deterministic_lm_head,
            attention_mode=args.deterministic_attention_mode,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log(
            "[oracle] deterministic_projection_oracle="
            f"{projection_oracle['patched_linear_modules']} linear, "
            f"{projection_oracle['patched_rms_norm_modules']} rms_norm, "
            f"{projection_oracle['patched_attention_modules']} attention, "
            f"{projection_oracle['patched_mlp_modules']} mlp modules patched"
        )

    results: list[dict] = []
    matches = 0
    for i, prompt in enumerate(prompts):
        log(f"\n[corpus] ({i+1}/{len(prompts)}) prompt={prompt!r}")
        t0 = time.perf_counter()
        diagnostic_top_k = args.diagnostic_top_k if args.diagnose_mismatches else 0
        py_ids, py_text, py_top_by_token = python_greedy(
            tokenizer, model, prompt, args.max_new_tokens, device, diagnostic_top_k,
        )
        py_ms = (time.perf_counter() - t0) * 1000
        log(f"  python  ({py_ms:.0f}ms): {py_text!r}")
        try:
            rs_ids, rs_text = rust_greedy(
                args.binary, args.model_variant, args.model_dir,
                prompt, args.max_new_tokens,
            )
        except Exception as e:
            log(f"  rust FAILED: {e}")
            results.append({"prompt": prompt, "error": str(e)})
            continue
        log(f"  rust:    {rs_text!r}")

        # Token-level comparison
        n = min(len(py_ids), len(rs_ids))
        first_diverge = next((j for j in range(n) if py_ids[j] != rs_ids[j]), n)
        ok = py_ids == rs_ids
        if ok:
            matches += 1
        log(f"  agree_for={first_diverge}/{max(len(py_ids), len(rs_ids))} "
            f"{'MATCH' if ok else 'DIVERGE'}")
        result = {
            "prompt": prompt,
            "python_tokens": py_ids,
            "rust_tokens": rs_ids,
            "first_divergence": first_diverge,
            "matched": ok,
        }
        if args.diagnose_mismatches and not ok:
            rust_diag = rust_materialized_diagnostics(
                args.binary, args.model_variant, args.model_dir,
                prompt, args.max_new_tokens,
            )
            diag = mismatch_diagnostic(
                py_ids, rs_ids, first_diverge, py_top_by_token, rust_diag,
                args.near_tie_threshold,
            )
            result["diagnostic"] = diag
            log(
                "  diagnostic: "
                f"near_tie={diag['near_tie']} "
                f"prefix_aligned={diag['prefix_aligned']} "
                f"python_token={diag['python_token']} rust_token={diag['rust_token']} "
                f"gaps={diag['logit_gaps']}"
            )
        results.append(result)

    log(f"\n[corpus] {matches}/{len(prompts)} prompts matched in full")
    if args.report:
        args.report.write_text(json.dumps({
            "matches": matches,
            "total": len(prompts),
            "torch_backend": torch_backend,
            "deterministic_projection_oracle": projection_oracle,
            "results": results,
        }, indent=2))
        log(f"[corpus] detailed report: {args.report}")

    return 0 if matches == len(prompts) else 1


if __name__ == "__main__":
    sys.exit(main())
