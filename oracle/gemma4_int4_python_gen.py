#!/usr/bin/env python3
"""
Diagnostic: load the Gemma 4 INT4 bake, reconstruct each quantized tensor
from (nibbles, scale, zero), overwrite the live BF16 model's weights, and
either (a) run greedy generation on a test prompt (default) or (b) dump
per-layer post-block hidden states + final logits as JSON via
``--emit-hiddens PATH``.

Purpose: reveals whether Rust-INT4 output divergence from BF16 is a Rust-side
pipeline bug (if Python output matches BF16 but Rust doesn't) or a fundamental
GPTQ-quality artefact (if Python output matches Rust's INT4 output). The
``--emit-hiddens`` mode pairs with ``gemma4_int4_layer_diag`` on the Rust side
to locate the exact layer where two pipelines diverge.

Not a bake tool — does not write any files except the optional JSON dump
produced by ``--emit-hiddens``. Safe to run with --device cpu.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import torch
import torch.nn as nn


def tensor_to_b64(t: torch.Tensor) -> str:
    """Serialize a CPU tensor as base64 of its raw byte storage.

    NumPy has no native BFloat16 dtype, so we route through the raw torch
    storage bytes rather than `.numpy().tobytes()` (which fails on bf16).
    `.clone()` forces fresh contiguous storage sized to the logical tensor —
    without it, a sliced view's `untyped_storage()` returns the parent's
    bytes instead of the slice's.
    """
    flat = t.detach().to("cpu").contiguous().clone()
    return base64.b64encode(bytes(flat.untyped_storage())).decode("ascii")


def dequantize_int4(nibbles: torch.Tensor, scale: torch.Tensor,
                    zero: torch.Tensor, group_size: int) -> torch.Tensor:
    """Rebuild the BF16 tensor from the (nibbles, scale, zero) triplet.

    Mirrors the Python self-check in `bake_int4_gemma4.py` and the Rust INT4
    dequant in `kernels/gemma4.hip::g4_int4_dequant_8`.
    """
    rows, cols = nibbles.shape
    row_gr = torch.arange(rows) // group_size
    col_gc = torch.arange(cols) // group_size
    sc_full = scale[row_gr][:, col_gc]
    zf_full = zero[row_gr][:, col_gc]
    recon = nibbles.float() * sc_full - zf_full * sc_full
    return recon.to(torch.bfloat16)


def load_bake(bake_dir: Path) -> tuple[dict, bytes]:
    manifest = json.loads((bake_dir / "manifest.json").read_text())
    data = (bake_dir / "weights.bin").read_bytes()
    return manifest, data


def bytes_to_tensor(buf: bytes, dtype: str, shape: list[int]) -> torch.Tensor:
    if dtype == "bf16":
        t = torch.frombuffer(bytearray(buf), dtype=torch.bfloat16)
    elif dtype == "u8":
        t = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
    elif dtype == "f32":
        t = torch.frombuffer(bytearray(buf), dtype=torch.float32)
    else:
        raise ValueError(f"unsupported dtype {dtype}")
    return t.reshape(shape)


def unpack_nibbles(packed: torch.Tensor, shape: list[int]) -> torch.Tensor:
    """Inverse of `pack_nibbles` in bake_int4_gemma4.py: u8[rows, cols/2] -> u8[rows, cols]."""
    # Pack convention: lo = nibbles[:, 2k], hi = nibbles[:, 2k+1].
    out_shape = (shape[0], shape[1] * 2)
    lo = (packed & 0x0F)
    hi = ((packed >> 4) & 0x0F)
    out = torch.empty(out_shape, dtype=torch.uint8)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out


def apply_bake_to_model(model: nn.Module, bake_dir: Path, group_size: int,
                       weight_prefix: str) -> tuple[int, int]:
    manifest, data = load_bake(bake_dir)

    entries = {e["name"]: e for e in manifest["tensors"]}
    # Group INT4 triplets by base weight name.
    int4_targets: dict[str, dict[str, dict]] = {}
    for name, e in entries.items():
        if e["layout"] == "Int4Quantized":
            int4_targets.setdefault(name, {})["packed"] = e
        elif name.endswith("_int4_scale"):
            base = name[: -len("_int4_scale")]
            int4_targets.setdefault(base, {})["scale"] = e
        elif name.endswith("_int4_zero"):
            base = name[: -len("_int4_zero")]
            int4_targets.setdefault(base, {})["zero"] = e

    applied = 0
    missing = 0
    sd = dict(model.named_parameters())
    for base_name, parts in int4_targets.items():
        if "packed" not in parts or "scale" not in parts or "zero" not in parts:
            print(f"[warn] incomplete INT4 trio for {base_name}")
            continue
        # The bake uses raw names (`model.language_model.layers.0.self_attn.q_proj.weight`).
        # The model's parameter names may be `layers.0.self_attn.q_proj.weight`
        # (rooted at language_model). Strip the prefix.
        lm_name = base_name
        if lm_name.startswith(weight_prefix + "."):
            lm_name = lm_name[len(weight_prefix) + 1:]
        if lm_name not in sd:
            # Try fully-qualified variants.
            alt = None
            for k in sd:
                if k.endswith(lm_name) or k == base_name:
                    alt = k
                    break
            if alt is None:
                missing += 1
                continue
            lm_name = alt

        p = parts["packed"]
        s = parts["scale"]
        z = parts["zero"]
        packed_bytes = data[p["offset"]:p["offset"] + p["byte_len"]]
        scale_bytes = data[s["offset"]:s["offset"] + s["byte_len"]]
        zero_bytes = data[z["offset"]:z["offset"] + z["byte_len"]]
        packed = bytes_to_tensor(packed_bytes, "u8", p["shape"])
        # The packed shape is [out, in/2]; reconstruct nibbles as [out, in].
        out_rows = p["shape"][0]
        in_cols = p["shape"][1] * 2
        nibbles = unpack_nibbles(packed, [out_rows, p["shape"][1]])
        scale = bytes_to_tensor(scale_bytes, "bf16", s["shape"]).float()
        zero = bytes_to_tensor(zero_bytes, "bf16", z["shape"]).float()
        recon = dequantize_int4(nibbles, scale, zero, group_size)
        target_p = sd[lm_name]
        if tuple(target_p.shape) != (out_rows, in_cols):
            print(f"[warn] shape mismatch {lm_name}: "
                  f"model={tuple(target_p.shape)} bake={(out_rows, in_cols)}")
            missing += 1
            continue
        target_p.data.copy_(recon.to(target_p.dtype).to(target_p.device))
        applied += 1
    return applied, missing


def resolve_language_model(model: nn.Module) -> nn.Module:
    """Descend through the multimodal wrapper to reach ``Gemma4TextModel``."""
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "model"):
        return model.model
    return model


def emit_hiddens_mode(
    model: nn.Module,
    tokenizer,
    prompt: str,
    device: torch.device,
    out_path: Path,
) -> None:
    """Run a single text-stack forward pass and dump per-layer hidden states.

    Writes a JSON with:
      prompt_token_ids         : list[int]  — tokenizer(add_special_tokens=True)
      prefill_per_layer_hidden : list[str]  — base64 BF16, one entry per layer
      prefill_per_layer_hidden_shape : [1,1,hidden]
      final_norm_hidden        : base64 BF16 [1,1,hidden], post-final-norm
      logits                   : list[float] — softcapped logits at last token
      vocab_size, hidden_size, num_layers
    """
    language_model = resolve_language_model(model)
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    prompt_ids = input_ids[0].cpu().tolist()

    # HuggingFace Gemma 4 returns `hidden_states` as a tuple of length
    # `num_layers + 1`: index 0 is the input embedding, indices 1..=N are the
    # post-decoder-block outputs for layers 0..N-1. But the LAST entry
    # (`hidden_states[N]`, i.e. layer N-1's post-block slot) gets OVERWRITTEN
    # with the post-final-norm `last_hidden_state`. That makes the default
    # layer-N-1 comparison an apples-to-oranges pre-vs-post-norm check
    # — cos_sim drops catastrophically even with identical weights. Work
    # around by registering a forward pre-hook on `language_model.norm` that
    # captures the true pre-norm input right before the final RMSNorm runs.
    pre_norm_snapshot: dict[str, torch.Tensor] = {}

    def pre_norm_hook(_module, inputs):
        pre_norm_snapshot["h"] = inputs[0].detach().clone()

    final_norm_mod = getattr(language_model, "norm", None)
    if final_norm_mod is None:
        raise RuntimeError("language_model has no `.norm` attribute — cannot capture layer N-1 pre-norm")
    hook_handle = final_norm_mod.register_forward_pre_hook(pre_norm_hook)
    try:
        with torch.no_grad():
            inner = language_model(
                input_ids=input_ids, use_cache=False, output_hidden_states=True,
            )
    finally:
        hook_handle.remove()

    hidden_tuple = inner.hidden_states
    if hidden_tuple is None:
        raise RuntimeError("output_hidden_states=True did not populate hidden_states")
    if "h" not in pre_norm_snapshot:
        raise RuntimeError("forward pre-hook on language_model.norm did not fire")

    # Build per-layer dump: layers 0..N-2 come from hidden_states[1..-1]
    # (all reliably pre-final-norm), layer N-1 comes from the pre-hook
    # capture on the final norm module (true pre-norm).
    per_layer: list[str] = []
    per_layer_shape: list[int] | None = None
    for layer_h in hidden_tuple[1:-1]:
        last = layer_h[:, -1:, :].to(torch.bfloat16)
        per_layer.append(tensor_to_b64(last))
        if per_layer_shape is None:
            per_layer_shape = list(last.shape)
    last_layer_pre_norm = pre_norm_snapshot["h"][:, -1:, :].to(torch.bfloat16)
    per_layer.append(tensor_to_b64(last_layer_pre_norm))

    final_hidden = inner.last_hidden_state[:, -1:, :].to(torch.bfloat16)

    # Softcapped logits via the full multimodal wrapper (it applies
    # `final_logit_softcapping` inside forward for us).
    with torch.no_grad():
        full_out = model(input_ids=input_ids, use_cache=False)
    logits_last = full_out.logits[0, -1, :].float().cpu().tolist()

    hidden_size = int(final_hidden.shape[-1])
    vocab_size = int(full_out.logits.shape[-1])

    out = {
        "prompt_token_ids": prompt_ids,
        "hidden_size": hidden_size,
        "num_layers": len(per_layer),
        "vocab_size": vocab_size,
        "prefill_per_layer_hidden": per_layer,
        "prefill_per_layer_hidden_shape": per_layer_shape,
        "final_norm_hidden": tensor_to_b64(final_hidden),
        "final_norm_hidden_shape": list(final_hidden.shape),
        "logits": logits_last,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    print(
        f"[python-int4] wrote {out_path} "
        f"(layers={len(per_layer)}, hidden={hidden_size}, vocab={vocab_size}, "
        f"prompt_tokens={len(prompt_ids)})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--bake-dir", default=None, type=Path)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--weight-prefix", default="model.language_model")
    ap.add_argument(
        "--emit-hiddens",
        type=Path,
        default=None,
        help="If set, skip generation and dump per-layer hidden states + "
             "softcapped logits as JSON to this path (for pairing with "
             "`gemma4_int4_layer_diag` on the Rust side).",
    )
    ap.add_argument(
        "--no-apply-bake",
        action="store_true",
        help="Skip applying the INT4 bake — run the BF16 model as-is. Pairs "
             "with `gemma4_int4_layer_diag --bf16` to isolate pipeline-wide "
             "drift from INT4-specific drift.",
    )
    args = ap.parse_args()

    bake_dir = args.bake_dir or (args.model_dir / ".supersonic" / "v1-int4-gptq")
    if not (bake_dir / "manifest.json").exists():
        raise SystemExit(f"no manifest at {bake_dir}")

    device = torch.device(args.device)
    print(f"[python-int4] device={device} bake={bake_dir}")

    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir), torch_dtype=torch.bfloat16
    )
    model.eval()
    if device.type != "cpu":
        model = model.to(device)

    if args.no_apply_bake:
        print("[python-int4] --no-apply-bake: running BF16 model without overlaying the bake")
    else:
        applied, missing = apply_bake_to_model(
            model, bake_dir, args.group_size, args.weight_prefix
        )
        print(f"[python-int4] applied {applied} INT4 tensors ({missing} missing)")

    if args.emit_hiddens is not None:
        emit_hiddens_mode(model, tokenizer, args.prompt, device, args.emit_hiddens)
        return

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    print(f"[python-int4] prompt_tokens={input_ids.shape[1]} ids={input_ids[0].tolist()}")
    with torch.no_grad():
        gen = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    new_ids = gen[0, input_ids.shape[1]:].tolist()
    print(f"[python-int4] generated_ids={new_ids}")
    print(f"[python-int4] text={tokenizer.decode(gen[0], skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
