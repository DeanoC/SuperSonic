#!/usr/bin/env python3
"""
Collect per-tensor activation Hessians for Q4KM GPTQ requantization.

This is the producer for:
    oracle/bake_q4km.py --quantizer gptq --hessian-dir <out>

It intentionally reuses the Qwen GPTQ oracle machinery from bake_int4.py:
HessianHook, layer-0 catcher, calibration corpus sampling, and raw
safetensors-name mapping.  The collected files are named by the raw
SuperSonic tensor name, so GGUF imports and safetensors imports can consume the
same hessian directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

from bake_int4 import (
    HessianHook,
    _build_name_map,
    _load_raw_tensor_names,
    capture_layer0_inputs,
    is_int4_target,
    move_kwargs_to,
)
from bake_q4km import sanitize_tensor_name


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect Qwen Hessians for Q4KM GPTQ baking")
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return p.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


@torch.no_grad()
def collect_hessians(
    model: nn.Module,
    calib_ids: torch.Tensor,
    raw_name_for_hf: dict[str, str],
    out_dir: Path,
    device: torch.device,
) -> dict[str, str]:
    model.eval()
    inner = model.model
    layers = inner.layers
    module_to_name: dict[int, str] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            module_to_name[id(mod)] = name + ".weight"

    log(f"[hessian] capturing layer-0 inputs from {calib_ids.shape[0]} samples")
    hidden_cpu, layer_kwargs = capture_layer0_inputs(model, layers, calib_ids, device)
    layer_kwargs_dev = move_kwargs_to(layer_kwargs, device)
    nsamples = len(hidden_cpu)
    index: dict[str, str] = {}

    for layer_idx, layer in enumerate(layers):
        targets: list[tuple[str, str, nn.Linear]] = []
        for mod in layer.modules():
            if not isinstance(mod, nn.Linear):
                continue
            hf_name = module_to_name[id(mod)]
            raw_name = raw_name_for_hf.get(hf_name)
            if raw_name is not None and is_int4_target(hf_name):
                targets.append((hf_name, raw_name, mod))

        log(f"[hessian] layer {layer_idx + 1}/{len(layers)}: {len(targets)} targets")
        hooks = {hf_name: HessianHook(mod) for hf_name, _, mod in targets}
        for s in range(nsamples):
            hs = hidden_cpu[s].to(device)
            out = layer(hs, **layer_kwargs_dev)
            if isinstance(out, tuple):
                out = out[0]
            hidden_cpu[s] = out.detach().cpu()
            del hs, out

        for hf_name, raw_name, _ in targets:
            hook = hooks[hf_name]
            try:
                if hook.H is None:
                    log(f"[hessian]   {raw_name}: WARNING no activations captured")
                    continue
                rel = f"{sanitize_tensor_name(raw_name)}.pt"
                torch.save({"H": hook.H.cpu(), "N": hook.N, "name": raw_name}, out_dir / rel)
                index[raw_name] = rel
                log(f"[hessian]   {raw_name}: shape={tuple(hook.H.shape)} N={hook.N}")
            finally:
                hook.close()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return index


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    log(f"[hessian] device={device}")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch_dtype(args.dtype),
        trust_remote_code=True,
    ).to(device)

    raw_keys = _load_raw_tensor_names(args.model_dir)
    hf_to_raw = _build_name_map(list(model.state_dict().keys()), raw_keys)
    log(f"[hessian] mapped {len(hf_to_raw)} HF state tensors to raw names")

    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(r["text"] for r in train if r["text"].strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if ids.numel() < args.seqlen * 2:
        raise SystemExit(f"not enough calibration tokens for seqlen={args.seqlen}: {ids.numel()}")
    torch.manual_seed(args.seed)
    starts = torch.randint(0, ids.numel() - args.seqlen - 1, (args.num_samples,))
    calib = torch.stack([ids[s:s + args.seqlen] for s in starts])
    log(f"[hessian] calibration batch={tuple(calib.shape)}")

    index = collect_hessians(model, calib, hf_to_raw, args.out_dir, device)
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2))
    log(f"[hessian] wrote {len(index)} Hessians to {args.out_dir}")


if __name__ == "__main__":
    main()
