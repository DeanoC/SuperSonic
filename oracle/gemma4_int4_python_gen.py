#!/usr/bin/env python3
"""
Diagnostic: load the Gemma 4 INT4 bake, reconstruct each quantized tensor
from (nibbles, scale, zero), overwrite the live BF16 model's weights, and
run greedy generation on a test prompt.

Purpose: reveals whether Rust-INT4 output divergence from BF16 is a Rust-side
pipeline bug (if Python output matches BF16 but Rust doesn't) or a fundamental
GPTQ-quality artefact (if Python output matches Rust's INT4 output).

Not a bake tool — does not write any files. Safe to run with --device cpu.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--bake-dir", default=None, type=Path)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--max-new-tokens", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--weight-prefix", default="model.language_model")
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

    applied, missing = apply_bake_to_model(
        model, bake_dir, args.group_size, args.weight_prefix
    )
    print(f"[python-int4] applied {applied} INT4 tensors ({missing} missing)")

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
