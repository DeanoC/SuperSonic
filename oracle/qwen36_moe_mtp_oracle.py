"""
Qwen3.6-MoE multi-token-prediction (MTP) head reference oracle.

Reproduces vLLM's `Qwen3NextMultiTokenPredictor.forward` byte-for-byte in
pure PyTorch, then dumps every intermediate as base64-BF16 to a JSON file
for Rust-side parity testing. The 35B-A3B MTP head is one full-attention
MoE block fused into the base hidden state via an `mtp.fc` linear, three
RMSNorms (`mtp.pre_fc_norm_{hidden,embedding}`, `mtp.norm`), and shared
`embed_tokens` / `lm_head` with the base model.

End-to-end equation per draft step `k`, recurrent through `k = 0..K-1`:

    e        = embed_tokens(next_token_id)
    e_norm   = rmsnorm(e, mtp.pre_fc_norm_embedding.weight, eps)
    h_norm   = rmsnorm(h_base, mtp.pre_fc_norm_hidden.weight, eps)
    fused    = mtp.fc @ cat([e_norm, h_norm], dim=-1)        # cat order: e first
    out      = layer(fused, residual=None, pos=base_len+k, kv=mtp_kv_buffer)
    h_post   = rmsnorm(out, mtp.norm.weight, eps)
    logits   = lm_head @ h_post
    next     = argmax(logits)                                # vLLM drafts greedy
    h_base   = h_post                                        # recurrent

vLLM source: `vllm/model_executor/models/qwen3_next_mtp.py` —
`Qwen3NextMultiTokenPredictor.forward` (cat order, fc, layer call,
norm-with-residual fold) and `vllm/v1/spec_decode/llm_base_proposer.py`
(K>1 loop: input_ids = argmax of step k's logits, hidden_states = step k's
post-norm output).

## Synthetic mode (default)

The 35B-A3B base model is ~64 GiB BF16 and does not fit on a 24 GiB GPU
or a 64 GiB host. For parity testing we don't need real-prefill output —
the SuperSonic-side kernel transforms `(h_base, next_token_id, position)`
deterministically, so a seeded synthetic `h_base` + a chosen
`next_token_id` are sufficient inputs. The oracle defaults to that mode
and only loads what it actually needs:
  - 19 `mtp.*` tensors (~1.6 GiB BF16)
  - shared `embed_tokens.weight` (vocab × hidden = 248320 × 2048 × 2 ≈ 970 MiB)
  - shared `lm_head.weight` (same shape, different tensor; tied=False)
  - one `Qwen3_5MoeDecoderLayer` instantiated from the config (≈ structural
    only; we replace its random init with the MTP weights)

`--mode prefill` runs the live HF base model on a prompt, takes the final
post-norm hidden state and greedy next token, and then drafts through the
standalone MTP head. This is a big-box validation path for machines that can
hold the 35B-A3B BF16-dequantized checkpoint. The default synthetic mode stays
cheap for kernel parity tests that only need deterministic MTP inputs.

Used by `crates/runner/tests/qwen36_moe_mtp_parity.rs` (Phase 6.2c+) to
gate the SuperSonic-side MTP kernel against this reference.

Usage:
    .venv-bake/bin/python oracle/qwen36_moe_mtp_oracle.py \
      --model-dir /path/to/Qwen3.6-35B-A3B \
      --num-speculative-tokens 3 \
      --base-seq-len 12 \
      --base-next-token 71093 \
      --seed 42 \
      --out /tmp/qwen36_mtp.json

    .venv-bake/bin/python oracle/qwen36_moe_mtp_oracle.py \
      --model-dir /path/to/Qwen3.6-35B-A3B-FP8 \
      --mode prefill \
      --prompt "The quick brown fox jumps over" \
      --num-speculative-tokens 3 \
      --device cuda:0 \
      --out /tmp/qwen36_mtp_real.json
"""
import argparse
import base64
import gc
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeTextRotaryEmbedding,
)


def b64_bf16(t: torch.Tensor) -> str:
    """Pack a tensor as BF16 little-endian and base64-encode."""
    arr = t.detach().to(torch.bfloat16).cpu().contiguous()
    # PyTorch BF16 doesn't go through numpy directly; reinterpret as u16.
    u16 = arr.view(torch.uint16).numpy()
    return base64.b64encode(u16.tobytes()).decode()


def b64_i32(ids: list[int]) -> str:
    return base64.b64encode(np.asarray(ids, dtype=np.int32).tobytes()).decode()


def load_named_tensors(model_dir: Path, names: list[str]) -> dict[str, torch.Tensor]:
    """Load specific tensor names from the safetensors index."""
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    by_shard: dict[str, list[str]] = {}
    for n in names:
        shard = index["weight_map"].get(n)
        if shard is None:
            raise SystemExit(f"tensor '{n}' not in safetensors index")
        by_shard.setdefault(shard, []).append(n)
    out: dict[str, torch.Tensor] = {}
    for shard, keys in by_shard.items():
        with safe_open(str(model_dir / shard), framework="pt") as f:
            for k in keys:
                out[k] = f.get_tensor(k)
    return out


def _load_raw_tensor_names(model_dir: Path) -> set[str]:
    index = model_dir / "model.safetensors.index.json"
    if index.exists():
        return set(json.loads(index.read_text())["weight_map"].keys())
    keys: set[str] = set()
    for p in [model_dir / "model.safetensors", *model_dir.glob("model*.safetensors")]:
        if not p.exists():
            continue
        with safe_open(str(p), framework="pt", device="cpu") as f:
            keys.update(f.keys())
    if not keys:
        raise SystemExit(f"no safetensors found in {model_dir}")
    return keys


class RawTensorLoader:
    """Small indexed safetensors loader used by the FP8 checkpoint path."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        index = model_dir / "model.safetensors.index.json"
        if index.exists():
            self.weight_map: dict[str, str] = dict(json.loads(index.read_text())["weight_map"])
        else:
            self.weight_map = {}
            for p in [model_dir / "model.safetensors", *model_dir.glob("model*.safetensors")]:
                if not p.exists():
                    continue
                with safe_open(str(p), framework="pt", device="cpu") as f:
                    for k in f.keys():
                        self.weight_map[k] = p.name
        self._handles: dict[str, Any] = {}

    def get(self, name: str) -> torch.Tensor | None:
        shard = self.weight_map.get(name)
        if shard is None:
            return None
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.model_dir / shard), framework="pt", device="cpu")
            self._handles[shard] = handle
        return handle.get_tensor(name)


def dequant_fp8_blocks(
    w: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    if w.dim() != 2 or scale_inv.dim() != 2:
        raise ValueError(
            f"FP8 dequant expects 2D tensors, got weight={tuple(w.shape)} "
            f"scale={tuple(scale_inv.shape)}"
        )
    rows, cols = w.shape
    scale_rows, scale_cols = scale_inv.shape
    out = torch.empty((rows, cols), dtype=torch.bfloat16)
    w_f = w.to(torch.float32)
    s_f = scale_inv.to(torch.float32)
    for sr in range(scale_rows):
        r0 = sr * block_size
        r1 = min(r0 + block_size, rows)
        for sc in range(scale_cols):
            c0 = sc * block_size
            c1 = min(c0 + block_size, cols)
            out[r0:r1, c0:c1] = (w_f[r0:r1, c0:c1] * s_f[sr, sc]).to(torch.bfloat16)
    return out


def dequant_fp8_tensor(w: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    if w.dim() == 2:
        return dequant_fp8_blocks(w.detach().cpu(), scale_inv.detach().cpu())
    if w.dim() == 3 and scale_inv.dim() == 3:
        return torch.stack(
            [dequant_fp8_blocks(w[e].detach().cpu(), scale_inv[e].detach().cpu())
             for e in range(w.shape[0])],
            dim=0,
        )
    raise ValueError(
        f"unsupported FP8 dequant shape: weight={tuple(w.shape)} "
        f"scale={tuple(scale_inv.shape)}"
    )


def load_raw_tensor_bf16(
    model_dir: Path,
    name: str,
    loader: RawTensorLoader | None = None,
) -> torch.Tensor:
    t = loader.get(name) if loader is not None else None
    if t is None:
        loaded = load_named_tensors(model_dir, [name])
        t = loaded.get(name)
    if t is None:
        raise KeyError(f"raw tensor not found: {name}")
    scale_name = f"{name}_scale_inv"
    scale = loader.get(scale_name) if loader is not None else None
    if scale is not None and t.dim() == 2:
        return dequant_fp8_blocks(t, scale)
    return t.to(torch.bfloat16)


def load_fused_expert_bf16(
    model_dir: Path,
    raw_keys: set[str],
    raw_base: str,
    kind: str,
    loader: RawTensorLoader | None = None,
) -> torch.Tensor:
    expert_re = re.compile(rf"^{re.escape(raw_base)}\.(\d+)\.")
    expert_ids = sorted({
        int(m.group(1))
        for k in raw_keys
        if (m := expert_re.match(k)) is not None
    })
    if not expert_ids:
        raise KeyError(f"no raw experts found under {raw_base}")
    chunks: list[torch.Tensor] = []
    for expert_id in expert_ids:
        base = f"{raw_base}.{expert_id}"
        if kind == "gate_up_proj":
            gate = load_raw_tensor_bf16(model_dir, f"{base}.gate_proj.weight", loader)
            up = load_raw_tensor_bf16(model_dir, f"{base}.up_proj.weight", loader)
            chunks.append(torch.cat([gate, up], dim=0).unsqueeze(0))
            del gate, up
        elif kind == "down_proj":
            down = load_raw_tensor_bf16(model_dir, f"{base}.down_proj.weight", loader)
            chunks.append(down.unsqueeze(0))
            del down
        else:
            raise ValueError(f"unknown fused expert kind {kind}")
    out = torch.cat(chunks, dim=0)
    del chunks
    return out


def set_module_parameter(model: nn.Module, name: str, value: torch.Tensor) -> None:
    parts = name.split(".")
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    old = module._parameters[parts[-1]]
    module._parameters[parts[-1]] = nn.Parameter(
        value,
        requires_grad=old.requires_grad if old is not None else False,
    )


def dequantize_remaining_fp8_parameters(
    model: nn.Module,
    model_dir: Path,
    raw_keys: set[str],
) -> int:
    """Transformers dequantizes most FP8 tensors, but fused experts can remain
    as FP8 parameters without registered scale tensors. Rebuild those from raw
    per-expert safetensors so real-prefill matches the BF16 reference model."""
    named_params = dict(model.named_parameters())
    fp8_names = [n for n, p in named_params.items() if p.dtype == torch.float8_e4m3fn]
    if fp8_names:
        print(f"[mtp-oracle] dequantizing {len(fp8_names)} remaining FP8 parameter(s)...")
    loader = RawTensorLoader(model_dir)
    converted = 0
    for name, param in list(named_params.items()):
        if param.dtype != torch.float8_e4m3fn:
            continue
        scale = named_params.get(f"{name}_scale_inv")
        if scale is not None:
            bf16 = dequant_fp8_tensor(param, scale).to(torch.bfloat16)
        elif name.endswith(".mlp.experts.gate_up_proj") or name.endswith(".mlp.experts.down_proj"):
            raw_name = name
            if raw_name.startswith("model.layers."):
                raw_name = raw_name.replace("model.layers.", "model.language_model.layers.", 1)
            kind = raw_name.rsplit(".", 1)[1]
            raw_base = raw_name.rsplit(".", 1)[0]
            bf16 = load_fused_expert_bf16(model_dir, raw_keys, raw_base, kind, loader)
        else:
            print(f"[mtp-oracle] WARNING: FP8 parameter {name} has no scale_inv; leaving as-is")
            continue
        set_module_parameter(model, name, bf16.to(param.device))
        converted += 1
        if converted % 10 == 0 or converted == len(fp8_names):
            print(f"[mtp-oracle]   dequantized {converted}/{len(fp8_names)}")
        del bf16
    if converted:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return converted


def load_mtp_runtime_tensors(model_dir: Path, raw_keys: set[str]) -> dict[str, torch.Tensor]:
    """Load the 19 runtime MTP tensors, fusing raw per-expert FP8 weights."""
    loader = RawTensorLoader(model_dir)
    expert_re = re.compile(r"^mtp\.layers\.0\.mlp\.experts\.(\d+)\.")
    passthrough = sorted(
        k for k in raw_keys
        if k.startswith("mtp.")
        and not k.endswith("_scale_inv")
        and expert_re.match(k) is None
    )
    out = {k: load_raw_tensor_bf16(model_dir, k, loader) for k in passthrough}
    if any(expert_re.match(k) for k in raw_keys):
        raw_base = "mtp.layers.0.mlp.experts"
        out["mtp.layers.0.mlp.experts.gate_up_proj"] = load_fused_expert_bf16(
            model_dir, raw_keys, raw_base, "gate_up_proj", loader
        )
        out["mtp.layers.0.mlp.experts.down_proj"] = load_fused_expert_bf16(
            model_dir, raw_keys, raw_base, "down_proj", loader
        )
    return out


def rmsnorm(x: torch.Tensor, gain: torch.Tensor, eps: float) -> torch.Tensor:
    """Standard RMSNorm with HF Qwen3_5MoeRMSNorm `(1.0 + gain)` unit offset."""
    x_f = x.to(torch.float32)
    rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    g_f = gain.to(torch.float32)
    return (x_f * rms * (1.0 + g_f)).to(x.dtype)


def find_full_attention_layer_idx(text_cfg) -> int:
    for i, t in enumerate(text_cfg.layer_types):
        if t == "full_attention":
            return i
    raise SystemExit("no full_attention layer in config; can't instantiate MTP layer")


def build_mtp_layer(text_cfg, mtp_state: dict[str, torch.Tensor], device: str):
    """Instantiate a `Qwen3_5MoeDecoderLayer` (full-attention variant) and
    load `mtp.layers.0.*` into it. Returns `(layer, rotary)` — the rotary
    module produces (cos, sin) for any (x, position_ids).

    HF instantiates the layer with random init; we immediately overwrite
    via `load_state_dict`. This avoids loading the full 35B base model.

    The MTP block is conceptually layer 0 of its own single-layer "model"
    — vLLM's KV cache for the MTP head is keyed by layer_idx=0. Force the
    layer to use layer_idx=0 by spoofing `layer_types[0] = "full_attention"`
    in a deep-copy of the config; without this, instantiating with the
    base model's first full-attn index (e.g. 3) would force the
    `DynamicCache.update(..., layer_idx=3)` call to pad indices 0..2
    with empty placeholders, which is wasteful but also bumps cache
    `seen_tokens` indexing in subtle ways across HF transformers
    versions.
    """
    import copy
    modified_cfg = copy.deepcopy(text_cfg)
    # Force the first slot to "full_attention" so layer_idx=0 selects
    # the full-attn variant (Qwen3_5MoeAttention).
    modified_cfg.layer_types = ["full_attention"] + list(
        modified_cfg.layer_types[1:]
    )

    layer = Qwen3_5MoeDecoderLayer(modified_cfg, layer_idx=0)
    layer = layer.to(torch.bfloat16).to(device).eval()
    rotary = Qwen3_5MoeTextRotaryEmbedding(modified_cfg).to(device).eval()

    # Drop the `mtp.layers.0.` prefix and check expected names.
    prefix = "mtp.layers.0."
    layer_state: dict[str, torch.Tensor] = {}
    for k, v in mtp_state.items():
        if not k.startswith(prefix):
            continue
        sub = k[len(prefix):]
        layer_state[sub] = v.to(torch.bfloat16)

    expected = set(name for name, _ in layer.named_parameters())
    missing = expected - set(layer_state.keys())
    if missing:
        raise SystemExit(
            f"missing MTP layer weights for: {sorted(missing)[:8]} "
            f"(layer expects {len(expected)} params, got {len(layer_state)})"
        )

    # `strict=False` because HF may register non-parameter buffers
    # (rotary inv_freq lives inside the rotary module, not the layer,
    # so the layer state_dict has only the parameter names we provide).
    layer.load_state_dict(layer_state, strict=False)
    return layer, rotary


@torch.no_grad()
def mtp_step(
    h_base: torch.Tensor,           # [1, hidden] BF16
    next_token_id: int,
    position: int,
    embed_weight: torch.Tensor,     # [vocab, hidden] BF16, base
    mtp_layer: torch.nn.Module,
    pre_fc_h: torch.Tensor,
    pre_fc_e: torch.Tensor,
    fc_w: torch.Tensor,             # [hidden, 2*hidden]
    norm_w: torch.Tensor,
    lm_head_weight: torch.Tensor,   # [vocab, hidden] BF16, base
    eps: float,
    past_kv,
    device: str,
):
    """One vLLM-faithful MTP draft step. Returns intermediates as a dict."""
    # Embed.
    e = embed_weight[next_token_id : next_token_id + 1].to(torch.bfloat16)
    e = e.to(device)  # [1, hidden]

    e_norm = rmsnorm(e, pre_fc_e, eps)
    h_norm = rmsnorm(h_base.to(torch.bfloat16), pre_fc_h, eps)

    # vLLM cat order: [e_norm, h_norm] (embedding first, hidden second).
    cat = torch.cat([e_norm, h_norm], dim=-1)  # [1, 2*hidden]
    fused = torch.nn.functional.linear(cat, fc_w.to(torch.bfloat16))  # [1, hidden]

    # The decoder layer expects [batch, seq, hidden].
    fused_3d = fused.unsqueeze(1)  # [1, 1, hidden]
    pos_ids = torch.tensor([[position]], device=device)

    # HF's `Qwen3_5MoeDecoderLayer` requires precomputed RoPE
    # (cos, sin). The rotary module returned by `build_mtp_layer` is
    # passed in via `mtp_layer._mtp_oracle_rope`.
    cos_sin = mtp_layer._mtp_oracle_rope(fused_3d, pos_ids)
    layer_out = mtp_layer(
        hidden_states=fused_3d,
        position_embeddings=cos_sin,
        position_ids=pos_ids,
        past_key_values=past_kv,
        use_cache=True,
        cache_position=pos_ids[0],
    )

    if isinstance(layer_out, tuple):
        attn_out_3d = layer_out[0]
    else:
        attn_out_3d = layer_out
    attn_out = attn_out_3d.squeeze(1)  # [1, hidden]
    # The HF layer mutates `past_kv` in place (cache.update inside
    # Qwen3_5MoeAttention) — there's no "new" cache to thread through.
    # The caller's persistent `DynamicCache` instance has just grown by
    # one position.

    h_post = rmsnorm(attn_out, norm_w, eps)
    logits = torch.nn.functional.linear(h_post.to(torch.bfloat16), lm_head_weight)
    next_tok = int(logits.argmax(dim=-1).item())

    # Keep all returned tensors 2D `[1, *]` so the caller's recurrence
    # (`h_base = out["h_post"]`) stays shape-stable. The JSON dump
    # squeezes to 1D at write time.
    return dict(
        e_norm=e_norm,
        h_norm=h_norm,
        fused=fused,
        attn_out=attn_out,
        h_post=h_post,
        logits=logits,
        next_tok=next_tok,
    )


def _text_model(causal_lm: nn.Module) -> nn.Module:
    root = getattr(causal_lm, "model", causal_lm)
    if hasattr(root, "layers") and hasattr(root, "norm"):
        return root
    language_model = getattr(root, "language_model", None)
    if language_model is not None and hasattr(language_model, "layers"):
        return language_model
    raise SystemExit("could not locate Qwen text model under AutoModelForCausalLM")


@torch.no_grad()
def run_real_prefill(
    model_dir: Path,
    prompt: str,
    device: str,
    raw_keys: set[str],
) -> tuple[torch.Tensor, int, int, list[int], torch.Tensor, torch.Tensor, str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[mtp-oracle] loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    hf_config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    quant_cfg = getattr(hf_config, "quantization_config", None)
    quant_method = (
        quant_cfg.get("quant_method")
        if isinstance(quant_cfg, dict)
        else getattr(quant_cfg, "quant_method", None)
    )
    if quant_method == "fp8":
        from transformers import FineGrainedFP8Config

        load_kwargs["quantization_config"] = FineGrainedFP8Config(dequantize=True)
        print("[mtp-oracle] source checkpoint is FP8; dequantizing base model to BF16")
    if device.startswith("cuda"):
        load_kwargs["device_map"] = {"": device}

    print(f"[mtp-oracle] loading base model from {model_dir}...")
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
    converted = dequantize_remaining_fp8_parameters(model, model_dir, raw_keys)
    if converted:
        print(f"[mtp-oracle] dequantized {converted} remaining FP8 parameter(s) to BF16")
    if "device_map" not in load_kwargs:
        model = model.to(device)
    model.eval()

    enc = tokenizer(prompt, return_tensors="pt")
    prompt_token_ids = [int(x) for x in enc["input_ids"][0].tolist()]
    enc = {k: v.to(device) for k, v in enc.items()}
    base_seq_len = int(enc["input_ids"].shape[1])
    print(f"[mtp-oracle] running real prefill for {len(prompt_token_ids)} prompt token(s)...")
    base = _text_model(model)(
        **enc,
        use_cache=False,
        return_dict=True,
    )
    h_base = base.last_hidden_state[:, -1, :].to(torch.bfloat16)
    logits = model.lm_head(h_base)
    base_next_tok = int(logits.argmax(dim=-1).item())
    continuation_text = tokenizer.decode([base_next_tok], skip_special_tokens=False)
    # Keep only the two shared matrices the MTP head needs; release the full
    # base model before building/running the standalone MTP decoder layer.
    embed_w = model.get_input_embeddings().weight.detach().to(torch.bfloat16).clone()
    lm_head_w = model.lm_head.weight.detach().to(torch.bfloat16).clone()
    h_base = h_base.detach().clone()
    del model, base, logits, enc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        h_base,
        base_seq_len,
        base_next_tok,
        prompt_token_ids,
        embed_w,
        lm_head_w,
        continuation_text,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["synthetic", "prefill"], default="synthetic",
                    help="synthetic uses seeded h_base; prefill runs the base model "
                         "on --prompt and drafts from its greedy next token.")
    ap.add_argument("--prompt", default=None,
                    help="Prompt text for --mode prefill.")
    ap.add_argument("--num-speculative-tokens", type=int, default=3,
                    help="K — vLLM default is 2 per the model card; we use 3 "
                         "for richer parity coverage of the recurrent path.")
    ap.add_argument("--base-seq-len", type=int, default=12,
                    help="Position the base model would have been at when "
                         "feeding h_base into MTP (synthetic mode; real "
                         "prefill mode would derive from the prompt).")
    ap.add_argument("--base-next-token", type=int, default=None,
                    help="Token id sampled by the base model's lm_head; fed "
                         "into the first MTP step. Defaults to a random "
                         "in-range id seeded by --seed.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu",
                    help="cpu | cuda:0. CPU is the reproducible default.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print(f"[mtp-oracle] reading config from {args.model_dir}...")
    cfg = AutoConfig.from_pretrained(str(args.model_dir))
    text_cfg = cfg.text_config
    hidden = int(text_cfg.hidden_size)
    vocab = int(text_cfg.vocab_size)
    eps = float(text_cfg.rms_norm_eps)

    raw_keys = _load_raw_tensor_names(args.model_dir)
    print(f"[mtp-oracle] loading MTP runtime tensors...")
    tensors = load_mtp_runtime_tensors(args.model_dir, raw_keys)

    embed_w = None
    lm_head_w = None
    prompt_token_ids = None
    prompt_next_text = None
    if args.mode == "prefill":
        if not args.prompt:
            raise SystemExit("--prompt is required with --mode prefill")
        (
            h_base,
            base_seq_len,
            base_next_tok,
            prompt_token_ids,
            embed_w,
            lm_head_w,
            prompt_next_text,
        ) = run_real_prefill(args.model_dir, args.prompt, args.device, raw_keys)
        h_base = h_base.to(args.device)
    else:
        print(f"[mtp-oracle] loading tied embed/lm_head for synthetic mode...")
        loader = RawTensorLoader(args.model_dir)
        embed_w = load_raw_tensor_bf16(
            args.model_dir, "model.language_model.embed_tokens.weight", loader
        )
        lm_head_w = load_raw_tensor_bf16(args.model_dir, "lm_head.weight", loader)
        # Synthetic h_base + base next token.
        h_base = torch.from_numpy(
            rng.standard_normal((1, hidden)).astype(np.float32) * 0.5
        ).to(torch.bfloat16).to(args.device)
        base_next_tok = (
            args.base_next_token
            if args.base_next_token is not None
            else int(rng.integers(0, vocab))
        )
        base_seq_len = args.base_seq_len
    h_base_step0 = h_base.detach().clone().squeeze(0).cpu()

    assert embed_w is not None
    assert lm_head_w is not None
    embed_w = embed_w.to(args.device)
    lm_head_w = lm_head_w.to(args.device)
    pre_fc_h = tensors["mtp.pre_fc_norm_hidden.weight"].to(args.device)
    pre_fc_e = tensors["mtp.pre_fc_norm_embedding.weight"].to(args.device)
    fc_w = tensors["mtp.fc.weight"].to(args.device)
    norm_w = tensors["mtp.norm.weight"].to(args.device)

    mtp_state = {k: v for k, v in tensors.items() if k.startswith("mtp.layers.0.")}
    print(f"[mtp-oracle] building MTP decoder layer (full-attention variant)...")
    mtp_layer, rotary = build_mtp_layer(text_cfg, mtp_state, args.device)
    # Attach the rotary so `mtp_step` can call it without an extra arg.
    mtp_layer._mtp_oracle_rope = rotary

    print(f"[mtp-oracle] running {args.num_speculative_tokens} draft step(s) "
          f"from base_seq_len={base_seq_len}, base_next_token={base_next_tok}...")
    # Persistent KV cache for the MTP layer — HF's attention mutates it in
    # place via `cache.update(...)` so all K draft steps share one
    # `DynamicCache` instance. Step k+1's attention reads K/V that
    # accumulated through steps 0..k (matching vLLM's MTP behaviour: each
    # draft step appends to the same single-layer MTP KV buffer).
    mtp_kv = DynamicCache()
    next_tok = base_next_tok
    steps = []
    for k in range(args.num_speculative_tokens):
        out = mtp_step(
            h_base=h_base,
            next_token_id=next_tok,
            position=base_seq_len + k,
            embed_weight=embed_w,
            mtp_layer=mtp_layer,
            pre_fc_h=pre_fc_h,
            pre_fc_e=pre_fc_e,
            fc_w=fc_w,
            norm_w=norm_w,
            lm_head_weight=lm_head_w,
            eps=eps,
            past_kv=mtp_kv,
            device=args.device,
        )
        # Per-step embed row for the input token. Lets the SuperSonic-side
        # parity test feed the same `e_in` the oracle saw (avoids re-loading
        # 970 MiB of `embed_tokens.weight` on the Rust side just to gather
        # one row).
        input_embed_row = embed_w[next_tok : next_tok + 1].squeeze(0).to(torch.bfloat16)
        steps.append(dict(
            step=k,
            position=base_seq_len + k,
            input_token_id=next_tok,
            draft_token_id=out["next_tok"],
            input_token_embed_bf16=b64_bf16(input_embed_row),
            e_norm_bf16=b64_bf16(out["e_norm"].squeeze(0)),
            h_norm_bf16=b64_bf16(out["h_norm"].squeeze(0)),
            fused_bf16=b64_bf16(out["fused"].squeeze(0)),
            attn_out_bf16=b64_bf16(out["attn_out"].squeeze(0)),
            h_post_bf16=b64_bf16(out["h_post"].squeeze(0)),
            logits_bf16=b64_bf16(out["logits"].squeeze(0)),
        ))
        # `mtp_kv` has been mutated in place by mtp_layer.forward —
        # nothing to thread through, just the recurrent h_base/next_tok.
        h_base = out["h_post"]
        next_tok = out["next_tok"]

    # Pre-fusion weights — the SuperSonic kernel under test (Phase 6.2c.1)
    # reads exactly these three tensors, so dumping them into the JSON keeps
    # the parity test self-contained (no need to re-open safetensors on the
    # Rust side just to load 16 MiB of fc_w + 8 KiB of norm gains).
    prefusion_weights = dict(
        fc_w_bf16=b64_bf16(fc_w),                       # [hidden, 2*hidden]
        pre_fc_norm_embedding_w_bf16=b64_bf16(pre_fc_e),  # [hidden]
        pre_fc_norm_hidden_w_bf16=b64_bf16(pre_fc_h),     # [hidden]
    )

    fixture = dict(
        schema="qwen36-moe-mtp-oracle-v1",
        mode=args.mode,
        config=dict(
            hidden=hidden,
            vocab=vocab,
            num_attention_heads=int(text_cfg.num_attention_heads),
            num_kv_heads=int(text_cfg.num_key_value_heads),
            head_dim=int(text_cfg.head_dim),
            rms_norm_eps=eps,
            rope_theta=float(text_cfg.rope_parameters["rope_theta"]),
            partial_rotary_factor=float(text_cfg.partial_rotary_factor),
            num_experts=int(text_cfg.num_experts),
            moe_intermediate_size=int(text_cfg.moe_intermediate_size),
            shared_expert_intermediate_size=int(text_cfg.shared_expert_intermediate_size),
            top_k=int(text_cfg.num_experts_per_tok),
        ),
        seed=args.seed,
        prompt=args.prompt if args.mode == "prefill" else None,
        prompt_token_ids=prompt_token_ids,
        base_next_token_text=prompt_next_text,
        base_seq_len=base_seq_len,
        base_next_token_id=base_next_tok,
        h_base_step0_bf16=b64_bf16(h_base_step0),
        prefusion_weights=prefusion_weights,
        draft_token_ids=[s["draft_token_id"] for s in steps],
        steps=steps,
    )

    args.out.write_text(json.dumps(fixture))
    print(f"[mtp-oracle] wrote {len(steps)} step(s) to {args.out}")
    print(f"[mtp-oracle] base→draft: [{base_next_tok}, "
          f"{', '.join(str(s['draft_token_id']) for s in steps)}]")


if __name__ == "__main__":
    main()
