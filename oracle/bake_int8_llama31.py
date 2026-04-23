#!/usr/bin/env python3
"""
Bake Meta-Llama-3.1-8B into a local SuperSonic INT8 package using the
BitsAndBytes `load_in_8bit=True` path as the quantization source of truth.

Output layout:
  {model_dir}/.supersonic/v1-int8-bnb/
    manifest.json
    weights.bin

Quantized linear weights are stored as raw int8 bytes under their original
`.weight` tensor names with layout `Int8Quantized` and manifest dtype `u8`.
Their companion per-row scales are stored as sibling `.SCB` tensors in F32.
Dense tensors that BnB leaves unquantized (for example `embed_tokens` and
`lm_head`) are copied verbatim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


FORMAT_VERSION = 1
CONVERTER_VERSION = 2
LAYOUT_RAW = "Raw"
LAYOUT_INT8 = "Int8Quantized"
ALIGN = 4096


def align_up(x: int, align: int) -> int:
    return (x + align - 1) & ~(align - 1)


def default_out_dir(model_dir: Path) -> Path:
    return model_dir / ".supersonic" / f"v{FORMAT_VERSION}-int8-bnb"


def tensor_bytes_and_meta(name: str, tensor: torch.Tensor) -> tuple[bytes, str, str]:
    t = tensor.detach().contiguous().cpu()
    if name.endswith(".weight") and t.dtype == torch.int8:
        return t.numpy().view(np.uint8).tobytes(), "u8", LAYOUT_INT8
    if t.dtype == torch.bfloat16:
        return bytes(t.untyped_storage()), "bf16", LAYOUT_RAW
    if t.dtype == torch.float32:
        return t.numpy().tobytes(), "f32", LAYOUT_RAW
    if t.dtype == torch.float16:
        return t.numpy().tobytes(), "f16", LAYOUT_RAW
    if t.dtype == torch.uint8:
        return t.numpy().tobytes(), "u8", LAYOUT_RAW
    if t.dtype == torch.int64:
        return t.numpy().tobytes(), "i64", LAYOUT_RAW
    raise TypeError(f"unsupported tensor dtype for {name}: {t.dtype}")


def iter_linear_tensors(prefix: str, linear: torch.nn.Module):
    yield f"{prefix}.weight", linear.weight
    scb = getattr(linear.weight, "SCB", None)
    if scb is not None:
        yield f"{prefix}.SCB", scb


def iter_llama_tensors(model):
    for idx, layer in enumerate(model.model.layers):
        lp = f"model.layers.{idx}"

        sa = layer.self_attn
        yield from iter_linear_tensors(f"{lp}.self_attn.q_proj", sa.q_proj)
        yield from iter_linear_tensors(f"{lp}.self_attn.k_proj", sa.k_proj)
        yield from iter_linear_tensors(f"{lp}.self_attn.v_proj", sa.v_proj)
        yield from iter_linear_tensors(f"{lp}.self_attn.o_proj", sa.o_proj)

        mlp = layer.mlp
        yield from iter_linear_tensors(f"{lp}.mlp.gate_proj", mlp.gate_proj)
        yield from iter_linear_tensors(f"{lp}.mlp.up_proj", mlp.up_proj)
        yield from iter_linear_tensors(f"{lp}.mlp.down_proj", mlp.down_proj)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bake Llama 3.1 8B INT8 weights for SuperSonic")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    model_dir = args.model_dir
    out_dir = args.out_dir or default_out_dir(model_dir)
    if args.fresh and out_dir.exists():
        for child in out_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                raise SystemExit(f"refusing to remove unexpected directory inside bake dir: {child}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (model_dir / "config.json").is_file():
        raise SystemExit(f"missing config.json in {model_dir}")

    print(f"[bake-int8] model_dir={model_dir}")
    print(f"[bake-int8] out_dir={out_dir}")
    print(f"[bake-int8] device={args.device}")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        quantization_config=quantization_config,
        device_map={"": args.device},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    tensors_to_write = list(iter_llama_tensors(model))

    weights_path = out_dir / "weights.bin"
    manifest_path = out_dir / "manifest.json"
    cursor = 0
    tensors = []
    quantized_count = 0

    with open(weights_path, "wb") as f:
        total = len(tensors_to_write)
        for idx, (name, tensor) in enumerate(tensors_to_write, start=1):
            if idx <= 8 or idx % 32 == 0 or idx >= total - 2:
                print(
                    f"[bake-int8] tensor {idx}/{total}: {name} shape={list(tensor.shape)} dtype={tensor.dtype}",
                    flush=True,
                )
            blob, dtype_name, layout = tensor_bytes_and_meta(name, tensor)
            offset = align_up(cursor, ALIGN)
            if offset > cursor:
                f.write(b"\x00" * (offset - cursor))
            f.write(blob)
            f.flush()
            tensors.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": dtype_name,
                    "layout": layout,
                    "offset": offset,
                    "byte_len": len(blob),
                }
            )
            if layout == LAYOUT_INT8:
                quantized_count += 1
            cursor = offset + len(blob)

    manifest = {
        "format_version": FORMAT_VERSION,
        "converter_version": CONVERTER_VERSION,
        "model_family": "llama31",
        "tensors": tensors,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[bake-int8] tensors={len(tensors)} int8_weights={quantized_count}")
    print(f"[bake-int8] wrote {cursor / (1024 * 1024):.1f} MiB to {weights_path}")
    print(f"[bake-int8] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
