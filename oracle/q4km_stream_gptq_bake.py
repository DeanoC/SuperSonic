#!/usr/bin/env python3
"""
Streaming GPTQ-style Q4KM bake from a GGUF source.

This script avoids materializing all Hessians or all baked tensor bytes at
once.  It uses a loadable Transformers model only to collect activation
Hessians layer-by-layer, but it quantizes weights loaded from the GGUF source
and immediately appends each tensor to the SuperSonic bake package.

Default memory behavior is conservative for 24 GB cards:
  * one target Hessian at a time (--target-batch-size 1);
  * one GGUF tensor dequantized at a time;
  * one baked tensor triplet written immediately.

The source GGUF must correspond to the same architecture/checkpoint as the
Transformers model used for activation collection.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn

from bake_int4 import (
    capture_layer0_inputs,
    gptq_quantize,
    is_int4_target,
    move_kwargs_to,
)
import bake_q4km


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stream a GGUF -> Q4KM-GPTQ SuperSonic bake")
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--activation-model-dir", type=Path, default=None,
                   help="Optional HF/Transformers model directory used only for activation collection")
    p.add_argument("--gguf-file", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--model", default="qwen3.6-27b")
    p.add_argument("--weight-prefix", default="model.language_model")
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--damp", type=float, default=0.01)
    p.add_argument("--device", default=None)
    p.add_argument("--gptq-device", default=None,
                   help="Device for GPTQ Cholesky/quantization. Defaults to --device; use cpu for large tensors on 24 GB GPUs.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-batch-size", type=int, default=1)
    p.add_argument("--use-gguf-loader", action="store_true",
                   help="Pass gguf_file=... to Transformers from_pretrained for activation collection")
    p.add_argument("--device-map", default=None,
                   help="Optional Transformers device_map, e.g. auto. If set, the script will not call .to(device).")
    p.add_argument("--max-memory", default=None,
                   help="Optional JSON max_memory for Transformers, e.g. '{\"0\":\"22GiB\",\"cpu\":\"48GiB\"}'")
    p.add_argument("--offload-folder", type=Path, default=None)
    p.add_argument("--limit-layers", type=int, default=None,
                   help="Debug: stop after this many layers")
    p.add_argument("--smoke-only", action="store_true",
                   help="Load model, tokenize calibration text, capture layer-0 inputs, then exit without writing a bake")
    return p.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


class CpuHessianHook:
    """Accumulate H on CPU to leave VRAM for the offloaded activation model."""

    def __init__(self, linear: nn.Linear):
        self.H: torch.Tensor | None = None
        self.N = 0
        self._handle = linear.register_forward_pre_hook(self._pre)

    def _pre(self, module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        x = inputs[0]
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        x = x.detach().to(device="cpu", dtype=torch.float32)
        n = x.shape[0]
        xx = (x.T @ x) * 2.0
        if self.H is None:
            self.H = xx / n
            self.N = n
        else:
            new_n = self.N + n
            self.H.mul_(self.N / new_n)
            self.H.add_(xx / new_n)
            self.N = new_n

    def close(self) -> None:
        self._handle.remove()


def hf_to_raw_name(hf_name: str, weight_prefix: str) -> str:
    if hf_name == "lm_head.weight":
        return hf_name
    if hf_name.startswith("model.language_model."):
        return hf_name
    if hf_name.startswith("model."):
        return hf_name.replace("model.", f"{weight_prefix}.", 1)
    return hf_name


def layer_idx_from_name(name: str, weight_prefix: str) -> int | None:
    m = re.match(rf"^{re.escape(weight_prefix)}\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def gguf_source_index(gguf: bake_q4km.GgufFile, weight_prefix: str) -> dict[str, bake_q4km.GgufTensorInfo]:
    out: dict[str, bake_q4km.GgufTensorInfo] = {}
    for info in gguf.tensors:
        mapped = bake_q4km.map_gguf_name(info.name, weight_prefix)
        if mapped is not None:
            out[mapped] = info
    return out


def load_transformed_gguf_tensor(
    gguf: bake_q4km.GgufFile,
    info: bake_q4km.GgufTensorInfo,
    mapped: str,
) -> tuple[torch.Tensor, bool, str | None]:
    t = bake_q4km.load_gguf_tensor(gguf, info)
    t, a_log_precomputed = bake_q4km.undo_gguf_tensor_transform(mapped, t)
    raw_dtype = None
    if info.ggml_type not in (
        bake_q4km.GGML_TYPE_F32,
        bake_q4km.GGML_TYPE_F16,
        bake_q4km.GGML_TYPE_BF16,
    ):
        raw_dtype = "bf16"
    return t, a_log_precomputed, raw_dtype


def encode_raw_or_minmax(
    gguf: bake_q4km.GgufFile,
    info: bake_q4km.GgufTensorInfo,
    mapped: str,
    weight_prefix: str,
    layer_types: list[str],
    group_size: int,
) -> list[tuple[str, bytes, list[int], str, str]]:
    t, a_log_precomputed, raw_dtype = load_transformed_gguf_tensor(gguf, info, mapped)
    entries, _ = bake_q4km.encode_tensor_entries(
        mapped,
        t,
        weight_prefix,
        layer_types,
        group_size,
        bake_q4km.QUANT_MINMAX,
        None,
        0.01,
        "cpu",
        a_log_precomputed,
        raw_dtype,
    )
    return entries


@torch.no_grad()
def quantize_target_from_gguf(
    gguf: bake_q4km.GgufFile,
    info: bake_q4km.GgufTensorInfo,
    raw_name: str,
    H: torch.Tensor,
    group_size: int,
    damp: float,
    device: torch.device,
) -> tuple[list[tuple[str, bytes, list[int], str, str]], torch.Tensor]:
    W, _, _ = load_transformed_gguf_tensor(gguf, info, raw_name)
    if W.ndim != 2:
        raise SystemExit(f"{raw_name}: GPTQ target must be 2D, got {tuple(W.shape)}")
    rows, cols = W.shape
    if tuple(H.shape) != (cols, cols):
        raise SystemExit(f"{raw_name}: Hessian shape {tuple(H.shape)} does not match {cols} input columns")
    Q_dq, nibbles, scales, zeros = gptq_quantize(
        W.to(device=device, dtype=torch.float32),
        H.to(device=device, dtype=torch.float32),
        group_size,
        damp,
    )
    packed = bake_q4km.pack_nibbles(nibbles.cpu())
    entries = [
        (raw_name, packed.numpy().tobytes(), list(packed.shape), "u8", bake_q4km.LAYOUT_INT4),
        (f"{raw_name}_int4_scale", bake_q4km.bf16_to_bytes(scales.cpu()), list(scales.shape), "bf16", bake_q4km.LAYOUT_RAW),
        (f"{raw_name}_int4_zero", bake_q4km.bf16_to_bytes(zeros.cpu()), list(zeros.shape), "bf16", bake_q4km.LAYOUT_RAW),
    ]
    return entries, Q_dq.detach().cpu()


def load_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    from transformers import AutoModelForCausalLM

    kwargs = {
        "torch_dtype": torch_dtype(args.dtype),
        "trust_remote_code": True,
    }
    if args.use_gguf_loader:
        kwargs["gguf_file"] = str(args.gguf_file)
    if args.device_map:
        kwargs["device_map"] = args.device_map
    if args.max_memory:
        raw_max_memory = json.loads(args.max_memory)
        kwargs["max_memory"] = {
            (int(k) if isinstance(k, str) and k.isdigit() else k): v
            for k, v in raw_max_memory.items()
        }
    if args.offload_folder:
        kwargs["offload_folder"] = str(args.offload_folder)

    model_dir = args.activation_model_dir or args.model_dir
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    if not args.device_map:
        model = model.to(device)
    model.eval()
    return model


def main() -> int:
    args = parse_args()
    if args.target_batch_size < 1:
        raise SystemExit("--target-batch-size must be >= 1")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    gptq_device = torch.device(args.gptq_device or str(device))
    log(f"[stream-gptq] device={device}")
    log(f"[stream-gptq] gptq_device={gptq_device}")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    model = load_model(args, device)
    inner = model.model
    layers = inner.layers
    _, layer_types = bake_q4km.load_config_context(args.model_dir)

    log("[stream-gptq] loading calibration text")
    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(r["text"] for r in train if r["text"].strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if ids.numel() < args.seqlen * 2:
        raise SystemExit(f"not enough calibration tokens for seqlen={args.seqlen}: {ids.numel()}")
    torch.manual_seed(args.seed)
    starts = torch.randint(0, ids.numel() - args.seqlen - 1, (args.num_samples,))
    calib = torch.stack([ids[s:s + args.seqlen] for s in starts])
    log(f"[stream-gptq] calibration batch={tuple(calib.shape)}")

    log("[stream-gptq] capturing layer-0 inputs")
    hidden_cpu, layer_kwargs = capture_layer0_inputs(model, layers, calib, device)
    layer_kwargs_dev = move_kwargs_to(layer_kwargs, device)
    if args.smoke_only:
        log(f"[stream-gptq] smoke ok: captured {len(hidden_cpu)} samples; kwargs={sorted(layer_kwargs.keys())}")
        return 0

    module_to_hf: dict[int, str] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            module_to_hf[id(mod)] = name + ".weight"

    out_dir = args.out_dir or (args.model_dir / ".supersonic" / f"v{bake_q4km.FORMAT_VERSION}-q4km-gptq")
    gguf = bake_q4km.parse_gguf(args.gguf_file)
    try:
        source_by_raw = gguf_source_index(gguf, args.weight_prefix)
        emitted: set[str] = set()
        family = "qwen36-moe" if "35b-a3b" in args.model.lower() else "qwen35"
        with bake_q4km.BakePackageWriter(
            out_dir,
            family,
            "gguf",
            "ggml-q4-k-family+stream-gptq",
            "q4km-gptq-v1",
        ) as writer:
            # Emit top-level tensors first.
            for raw_name in sorted(source_by_raw):
                if layer_idx_from_name(raw_name, args.weight_prefix) is not None:
                    continue
                writer.write_entries(encode_raw_or_minmax(
                    gguf, source_by_raw[raw_name], raw_name,
                    args.weight_prefix, layer_types, args.group_size,
                ))
                emitted.add(raw_name)

            num_layers = len(layers)
            if args.limit_layers is not None:
                num_layers = min(num_layers, args.limit_layers)
            for layer_idx in range(num_layers):
                layer = layers[layer_idx]
                layer_prefix = f"{args.weight_prefix}.layers.{layer_idx}."
                layer_sources = {
                    name: info for name, info in source_by_raw.items()
                    if name.startswith(layer_prefix)
                }
                targets: list[tuple[str, str, nn.Linear]] = []
                for mod in layer.modules():
                    if not isinstance(mod, nn.Linear):
                        continue
                    hf_name = module_to_hf[id(mod)]
                    raw_name = hf_to_raw_name(hf_name, args.weight_prefix)
                    if raw_name in layer_sources and is_int4_target(hf_name):
                        targets.append((hf_name, raw_name, mod))

                log(f"[stream-gptq] layer {layer_idx + 1}/{len(layers)}: {len(targets)} GPTQ targets")
                target_raw_names = {raw for _, raw, _ in targets}
                for start in range(0, len(targets), args.target_batch_size):
                    batch = targets[start:start + args.target_batch_size]
                    hooks = {raw_name: CpuHessianHook(mod) for _, raw_name, mod in batch}
                    for s in range(len(hidden_cpu)):
                        hs = hidden_cpu[s].to(device)
                        out = layer(hs, **layer_kwargs_dev)
                        if isinstance(out, tuple):
                            out = out[0]
                        del hs, out

                    for _, raw_name, mod in batch:
                        hook = hooks[raw_name]
                        try:
                            if hook.H is None:
                                raise SystemExit(f"{raw_name}: no activations captured")
                            entries, q_dq_cpu = quantize_target_from_gguf(
                                gguf,
                                layer_sources[raw_name],
                                raw_name,
                                hook.H,
                                args.group_size,
                                args.damp,
                                gptq_device,
                            )
                            writer.write_entries(entries)
                            emitted.add(raw_name)
                            if tuple(mod.weight.shape) == tuple(q_dq_cpu.shape):
                                mod.weight.data.copy_(q_dq_cpu.to(device=mod.weight.device, dtype=mod.weight.dtype))
                            log(f"[stream-gptq]   {raw_name}: H_N={hook.N} emitted")
                        finally:
                            hook.close()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                # Emit non-GPTQ tensors for the layer.
                for raw_name in sorted(layer_sources):
                    if raw_name in emitted or raw_name in target_raw_names:
                        continue
                    writer.write_entries(encode_raw_or_minmax(
                        gguf, layer_sources[raw_name], raw_name,
                        args.weight_prefix, layer_types, args.group_size,
                    ))
                    emitted.add(raw_name)

                # Advance sequential calibration state with the quantized layer.
                for s in range(len(hidden_cpu)):
                    hs = hidden_cpu[s].to(device)
                    out = layer(hs, **layer_kwargs_dev)
                    if isinstance(out, tuple):
                        out = out[0]
                    hidden_cpu[s] = out.detach().cpu()
                    del hs, out
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            missing = sorted(set(source_by_raw) - emitted)
            if missing and args.limit_layers is not None:
                log(
                    f"[stream-gptq] debug partial bake: leaving {len(missing)} tensors "
                    "unemitted because --limit-layers was set"
                )
            elif missing:
                log(f"[stream-gptq] emitting {len(missing)} remaining tensors without GPTQ")
                for raw_name in missing:
                    writer.write_entries(encode_raw_or_minmax(
                        gguf, source_by_raw[raw_name], raw_name,
                        args.weight_prefix, layer_types, args.group_size,
                    ))
        log(f"[stream-gptq] done: {out_dir}")
        return 0
    finally:
        gguf.close()


if __name__ == "__main__":
    raise SystemExit(main())
