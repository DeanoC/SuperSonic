#!/usr/bin/env python3
"""
Compare Phi-4 INT4 Rust hidden states against the Python INT4-patched oracle
after a configurable number of decoder layers.

This is intended for near-tie CUDA parity work. It uses the same GPTQ bake
reconstruction as int4_corpus_compare.py, then runs Rust with
SUPERSONIC_PHI4_LIMIT_LAYERS and SUPERSONIC_PHI4_DUMP_HIDDEN. Rust dumps the
pre-final-norm hidden vector, so this script applies the baked final norm
weight before comparing to the Python model output.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from int4_corpus_compare import build_oracle_model, load_bake, load_raw_tensor


def f32_to_bf16_rounded(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=True)
    bits = arr.view(np.uint32)
    bias = ((bits >> 16) & 1) + 0x7FFF
    bits = ((bits + bias) & 0xFFFF0000).astype(np.uint32)
    return bits.view(np.float32)


def rust_hidden_for_layers(
    binary: Path,
    model_dir: Path,
    model_variant: str,
    prompt: str,
    layers: int,
) -> np.ndarray:
    with tempfile.NamedTemporaryFile(
        prefix=f"supersonic-phi4-hidden-l{layers}-",
        suffix=".json",
        delete=False,
    ) as dump_file:
        dump_path = Path(dump_file.name)

    env = os.environ.copy()
    env["SUPERSONIC_PHI4_LIMIT_LAYERS"] = str(layers)
    env["SUPERSONIC_PHI4_DUMP_HIDDEN"] = str(dump_path)
    proc = subprocess.run(
        [
            str(binary),
            "--model",
            model_variant,
            "--model-dir",
            str(model_dir),
            "--prompt",
            prompt,
            "--max-new-tokens",
            "0",
            "--int4",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Rust run failed for layers={layers}:\n{proc.stderr}\n{proc.stdout}")
    payload = json.loads(dump_path.read_text())
    dump_path.unlink(missing_ok=True)
    return np.asarray(payload["hidden"], dtype=np.float32)


def rust_component_dump_for_layers(
    binary: Path,
    model_dir: Path,
    model_variant: str,
    prompt: str,
    layers: int,
) -> dict:
    with tempfile.NamedTemporaryFile(
        prefix=f"supersonic-phi4-components-l{layers}-",
        suffix=".json",
        delete=False,
    ) as dump_file:
        dump_path = Path(dump_file.name)

    env = os.environ.copy()
    env["SUPERSONIC_PHI4_LIMIT_LAYERS"] = str(layers)
    env["SUPERSONIC_PHI4_DUMP_LAYER_COMPONENTS"] = str(dump_path)
    proc = subprocess.run(
        [
            str(binary),
            "--model",
            model_variant,
            "--model-dir",
            str(model_dir),
            "--prompt",
            prompt,
            "--max-new-tokens",
            "0",
            "--int4",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Rust component dump failed for layers={layers}:\n{proc.stderr}\n{proc.stdout}")
    payload = json.loads(dump_path.read_text())
    dump_path.unlink(missing_ok=True)
    return payload


def rust_layer_trace(
    binary: Path,
    model_dir: Path,
    model_variant: str,
    prompt: str,
) -> dict:
    with tempfile.NamedTemporaryFile(
        prefix="supersonic-phi4-layer-trace-",
        suffix=".json",
        delete=False,
    ) as dump_file:
        dump_path = Path(dump_file.name)

    env = os.environ.copy()
    env["SUPERSONIC_PHI4_DUMP_LAYER_TRACE"] = str(dump_path)
    proc = subprocess.run(
        [
            str(binary),
            "--model",
            model_variant,
            "--model-dir",
            str(model_dir),
            "--prompt",
            prompt,
            "--max-new-tokens",
            "0",
            "--int4",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Rust layer trace failed:\n{proc.stderr}\n{proc.stdout}")
    payload = json.loads(dump_path.read_text())
    dump_path.unlink(missing_ok=True)
    return payload


def parse_layers(value: str) -> list[int]:
    layers = [int(x) for x in value.split(",") if x.strip()]
    if not layers or any(n <= 0 for n in layers):
        raise argparse.ArgumentTypeError("layers must be a comma-separated list of positive ints")
    return layers


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.float().detach().cpu().numpy().astype(np.float32)


def delta_row(name: str, py: np.ndarray, rust: np.ndarray) -> dict:
    delta = np.abs(py.astype(np.float32) - rust.astype(np.float32))
    cosine = float(np.dot(py, rust) / (np.linalg.norm(py) * np.linalg.norm(rust)))
    return {
        "name": name,
        "max_delta": float(delta.max()),
        "mean_delta": float(delta.mean()),
        "p99_delta": float(np.quantile(delta, 0.99)),
        "cosine": cosine,
    }


def silu_np(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32, copy=False)
    return arr / (1.0 + np.exp(-arr))


def gate_up_replay_from_normed(
    model,
    layer_num: int,
    normed: np.ndarray,
) -> dict[str, np.ndarray]:
    layer = model.model.layers[layer_num - 1]
    weight = layer.mlp.gate_up_proj.weight.detach().float().cpu().numpy()
    raw = weight @ normed.astype(np.float32)
    gate_raw, up_raw = np.split(raw.astype(np.float32), 2)
    gate_bf16 = f32_to_bf16_rounded(gate_raw)
    up_bf16 = f32_to_bf16_rounded(up_raw)
    gate_up = f32_to_bf16_rounded(silu_np(gate_bf16) * up_bf16)
    return {
        "gate_raw": gate_raw,
        "up_raw": up_raw,
        "gate_up": gate_up,
    }


def attn_replay_from_flat(
    model,
    layer_num: int,
    layer_input: np.ndarray,
    attn_flat: np.ndarray,
) -> dict[str, np.ndarray]:
    layer = model.model.layers[layer_num - 1]
    weight = layer.self_attn.o_proj.weight.detach().float().cpu().numpy()
    o_raw = weight @ attn_flat.astype(np.float32)
    o_out = f32_to_bf16_rounded(o_raw.astype(np.float32))
    post = f32_to_bf16_rounded(layer_input.astype(np.float32) + o_out)
    return {
        "o_proj_raw": o_raw.astype(np.float32),
        "attn_out": o_out,
        "post_attn_hidden": post,
    }


def qkv_replay_from_input(
    model,
    layer_num: int,
    layer_input: np.ndarray,
) -> dict[str, np.ndarray]:
    layer = model.model.layers[layer_num - 1]
    norm_weight = layer.input_layernorm.weight.detach().float().cpu().numpy()
    eps = float(model.config.rms_norm_eps)
    src = layer_input.astype(np.float32)
    inv_rms = 1.0 / np.sqrt(np.mean(src.astype(np.float64) ** 2) + eps)
    normed = f32_to_bf16_rounded((src * inv_rms * norm_weight).astype(np.float32))
    weight = layer.self_attn.qkv_proj.weight.detach().float().cpu().numpy()
    raw = weight @ normed.astype(np.float32)
    q_dim = model.config.num_attention_heads * model.config.hidden_size // model.config.num_attention_heads
    kv_dim = model.config.num_key_value_heads * model.config.hidden_size // model.config.num_attention_heads
    q_raw, k_raw, v_raw = np.split(raw.astype(np.float32), [q_dim, q_dim + kv_dim])
    return {
        "q_raw": f32_to_bf16_rounded(q_raw),
        "k_raw": f32_to_bf16_rounded(k_raw),
        "v_raw": f32_to_bf16_rounded(v_raw),
    }


def top_logits(logits: torch.Tensor, tokenizer, k: int = 8) -> list[dict]:
    values, indices = torch.topk(logits.float().detach().cpu(), min(k, logits.numel()))
    return [
        {
            "id": int(idx),
            "logit": float(value),
            "text": tokenizer.decode([int(idx)], skip_special_tokens=False),
        }
        for value, idx in zip(values, indices)
    ]


def parse_candidate_ids(value: str | None) -> list[int]:
    if value is None or not value.strip():
        return []
    ids = [int(x) for x in value.split(",") if x.strip()]
    if any(n < 0 for n in ids):
        raise argparse.ArgumentTypeError("candidate ids must be non-negative token ids")
    return ids


def candidate_logits(logits: torch.Tensor, tokenizer, ids: list[int]) -> list[dict]:
    logits_cpu = logits.float().detach().cpu()
    rows = []
    for idx in ids:
        rows.append({
            "id": idx,
            "logit": float(logits_cpu[idx].item()),
            "text": tokenizer.decode([idx], skip_special_tokens=False),
        })
    return rows


def component_logits(
    model,
    component: np.ndarray,
    tokenizer,
    device: torch.device,
    candidate_ids: list[int],
) -> dict:
    dtype = next(model.parameters()).dtype
    hidden = torch.tensor(component, dtype=dtype, device=device).view(1, 1, -1)
    with torch.no_grad():
        logits = model.lm_head(model.model.norm(hidden))[0, 0]
    top = top_logits(logits, tokenizer)
    row = {
        "argmax": int(torch.argmax(logits).item()),
        "top": top,
    }
    if candidate_ids:
        row["candidates"] = candidate_logits(logits, tokenizer, candidate_ids)
        if len(candidate_ids) >= 2:
            logits_cpu = logits.float().detach().cpu()
            row["candidate_margin_0_minus_1"] = float(
                logits_cpu[candidate_ids[0]].item() - logits_cpu[candidate_ids[1]].item()
            )
    if len(top) >= 2:
        row["top_margin"] = float(top[0]["logit"] - top[1]["logit"])
    return row


def python_component_trace(
    model,
    input_ids: torch.Tensor,
    layers: int,
    incremental: bool,
) -> dict[str, np.ndarray]:
    target = model.model.layers[layers - 1]
    captured: dict[str, np.ndarray] = {}

    def save_last(name: str, tensor: torch.Tensor) -> None:
        captured[name] = tensor_to_numpy(tensor[0, -1])

    def save_attn_out(_module, _inputs, output) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        save_last("attn_out", tensor)

    def save_attn_flat(_module, inputs) -> None:
        save_last("attn_flat", inputs[0])

    def save_gate_up(_module, _inputs, output) -> None:
        gate, up = output.chunk(2, dim=-1)
        save_last("gate_raw", gate)
        save_last("up_raw", up)
        save_last("gate_up", F.silu(gate) * up)

    def save_qkv(_module, _inputs, output) -> None:
        q_dim = model.config.num_attention_heads * model.config.hidden_size // model.config.num_attention_heads
        kv_dim = model.config.num_key_value_heads * model.config.hidden_size // model.config.num_attention_heads
        q, k, v = torch.split(output, [q_dim, kv_dim, kv_dim], dim=-1)
        save_last("q_raw", q)
        save_last("k_raw", k)
        save_last("v_raw", v)

    hooks = [
        target.register_forward_pre_hook(
            lambda _module, inputs: save_last("layer_input", inputs[0])
        ),
        target.self_attn.register_forward_hook(save_attn_out),
        target.self_attn.o_proj.register_forward_pre_hook(save_attn_flat),
        target.post_attention_layernorm.register_forward_pre_hook(
            lambda _module, inputs: save_last("post_attn_hidden", inputs[0])
        ),
        target.post_attention_layernorm.register_forward_hook(
            lambda _module, _inputs, output: save_last("post_attn_normed", output)
        ),
        target.mlp.register_forward_hook(
            lambda _module, _inputs, output: save_last("mlp_out", output)
        ),
        target.register_forward_hook(
            lambda _module, _inputs, output: save_last("final_hidden", output)
        ),
    ]
    if hasattr(target.self_attn, "qkv_proj"):
        hooks.append(target.self_attn.qkv_proj.register_forward_hook(save_qkv))
    if hasattr(target.mlp, "gate_up_proj"):
        hooks.append(target.mlp.gate_up_proj.register_forward_hook(save_gate_up))
    original_layers = int(model.config.num_hidden_layers)
    model.config.num_hidden_layers = layers
    try:
        with torch.no_grad():
            if not incremental:
                model.model(input_ids=input_ids, use_cache=False)
            else:
                past = None
                for pos in range(input_ids.shape[1]):
                    out = model.model(
                        input_ids=input_ids[:, pos:pos + 1],
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    past = out.past_key_values
    finally:
        model.config.num_hidden_layers = original_layers
        for hook in hooks:
            hook.remove()
    if "gate_up" in captured and hasattr(target.mlp, "down_proj"):
        weight = target.mlp.down_proj.weight.detach().float().cpu().numpy()
        captured["down_raw"] = weight @ captured["gate_up"].astype(np.float32)
    return captured


def python_all_component_traces(
    model,
    input_ids: torch.Tensor,
    incremental: bool,
) -> dict[int, dict[str, np.ndarray]]:
    captured: dict[int, dict[str, np.ndarray]] = {
        idx + 1: {} for idx in range(len(model.model.layers))
    }

    def save_last(layer_num: int, name: str, tensor: torch.Tensor) -> None:
        captured[layer_num][name] = tensor_to_numpy(tensor[0, -1])

    hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        layer_num = layer_idx + 1

        def save_attn_out(_module, _inputs, output, layer_num=layer_num) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            save_last(layer_num, "attn_out", tensor)

        def save_attn_flat(_module, inputs, layer_num=layer_num) -> None:
            save_last(layer_num, "attn_flat", inputs[0])

        def save_gate_up(_module, _inputs, output, layer_num=layer_num) -> None:
            gate, up = output.chunk(2, dim=-1)
            save_last(layer_num, "gate_raw", gate)
            save_last(layer_num, "up_raw", up)
            save_last(layer_num, "gate_up", F.silu(gate) * up)

        def save_qkv(_module, _inputs, output, layer_num=layer_num) -> None:
            q_dim = model.config.num_attention_heads * model.config.hidden_size // model.config.num_attention_heads
            kv_dim = model.config.num_key_value_heads * model.config.hidden_size // model.config.num_attention_heads
            q, k, v = torch.split(output, [q_dim, kv_dim, kv_dim], dim=-1)
            save_last(layer_num, "q_raw", q)
            save_last(layer_num, "k_raw", k)
            save_last(layer_num, "v_raw", v)

        hooks.extend([
            layer.register_forward_pre_hook(
                lambda _module, inputs, layer_num=layer_num:
                    save_last(layer_num, "layer_input", inputs[0])
            ),
            layer.self_attn.register_forward_hook(save_attn_out),
            layer.self_attn.o_proj.register_forward_pre_hook(save_attn_flat),
            layer.post_attention_layernorm.register_forward_pre_hook(
                lambda _module, inputs, layer_num=layer_num:
                    save_last(layer_num, "post_attn_hidden", inputs[0])
            ),
            layer.post_attention_layernorm.register_forward_hook(
                lambda _module, _inputs, output, layer_num=layer_num:
                    save_last(layer_num, "post_attn_normed", output)
            ),
            layer.mlp.register_forward_hook(
                lambda _module, _inputs, output, layer_num=layer_num:
                    save_last(layer_num, "mlp_out", output)
            ),
            layer.register_forward_hook(
                lambda _module, _inputs, output, layer_num=layer_num:
                    save_last(layer_num, "final_hidden", output)
            ),
        ])
        if hasattr(layer.self_attn, "qkv_proj"):
            hooks.append(layer.self_attn.qkv_proj.register_forward_hook(save_qkv))
        if hasattr(layer.mlp, "gate_up_proj"):
            hooks.append(layer.mlp.gate_up_proj.register_forward_hook(save_gate_up))
    try:
        with torch.no_grad():
            if not incremental:
                model.model(input_ids=input_ids, use_cache=False)
            else:
                past = None
                for pos in range(input_ids.shape[1]):
                    out = model.model(
                        input_ids=input_ids[:, pos:pos + 1],
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    past = out.past_key_values
    finally:
        for hook in hooks:
            hook.remove()
    for layer_idx, layer in enumerate(model.model.layers):
        layer_num = layer_idx + 1
        if "gate_up" in captured[layer_num] and hasattr(layer.mlp, "down_proj"):
            weight = layer.mlp.down_proj.weight.detach().float().cpu().numpy()
            captured[layer_num]["down_raw"] = (
                weight @ captured[layer_num]["gate_up"].astype(np.float32)
            )
    return captured


def python_hidden_for_layers(
    model,
    input_ids: torch.Tensor,
    layers: int,
    incremental: bool,
) -> torch.Tensor:
    original_layers = int(model.config.num_hidden_layers)
    model.config.num_hidden_layers = layers
    try:
        with torch.no_grad():
            if not incremental:
                return model.model(input_ids=input_ids, use_cache=False).last_hidden_state[:, -1, :].detach()

            past = None
            last_hidden = None
            for pos in range(input_ids.shape[1]):
                out = model.model(
                    input_ids=input_ids[:, pos:pos + 1],
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = out.past_key_values
                last_hidden = out.last_hidden_state[:, -1, :].detach()
            assert last_hidden is not None
            return last_hidden
    finally:
        model.config.num_hidden_layers = original_layers


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-variant", default="phi4-mini")
    parser.add_argument("--bake-subdir", default=".supersonic/v2-int4-gptq")
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("1,2,4,8,16,24,32"))
    parser.add_argument("--component-dump-layers", default=None,
                        help="Also compare dumped Rust components. Use an integer, comma list, or 'all' for all --layers entries.")
    parser.add_argument("--candidate-ids", type=parse_candidate_ids, default=[],
                        help="Comma-separated token ids to include when reporting component logits.")
    parser.add_argument("--logit-dump-layers", default=None,
                        help="Also compare Python lm_head logits from Python vs Rust hidden. "
                             "Use an integer, comma list, or 'all' for all --layers entries.")
    parser.add_argument("--python-incremental", action="store_true",
                        help="Run the Python oracle one token at a time with KV cache, matching corpus greedy mode.")
    parser.add_argument("--single-launch-trace", action="store_true",
                        help="Use SUPERSONIC_PHI4_DUMP_LAYER_TRACE to compare layer components from one Rust megakernel launch.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--disable-bf16-reduced-precision-reduction", action="store_true",
                        help="Force torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False "
                             "before loading the oracle model.")
    args = parser.parse_args()

    torch_backend = configure_torch_backend(args.disable_bf16_reduced_precision_reduction)
    print(f"[phi4-drift] torch_backend={torch_backend}")
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    bake_dir = args.model_dir / args.bake_subdir
    if not bake_dir.exists():
        print(f"ERROR: bake dir not found: {bake_dir}", file=sys.stderr)
        return 2

    tokenizer, model = build_oracle_model(args.model_dir, bake_dir, device)
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    manifest, weights = load_bake(bake_dir)
    by_name = {t["name"]: t for t in manifest["tensors"]}
    norm_name = "model.norm.weight"
    if norm_name not in by_name:
        print(f"ERROR: {norm_name} not found in bake", file=sys.stderr)
        return 2
    norm_weight = load_raw_tensor(by_name, weights, norm_name).astype(np.float32)
    eps = float(model.config.rms_norm_eps)

    original_layers = int(model.config.num_hidden_layers)
    results: list[dict] = []
    py_hidden_by_layer: dict[int, torch.Tensor] = {}
    rust_hidden_by_layer: dict[int, np.ndarray] = {}
    print(f"[phi4-drift] prompt={args.prompt!r} tokens={token_ids}")
    single_trace = None
    single_trace_by_layer = {}
    if args.single_launch_trace:
        single_trace = rust_layer_trace(
            args.binary,
            args.model_dir,
            args.model_variant,
            args.prompt,
        )
        single_trace_by_layer = {
            int(row["layer"]): row
            for row in single_trace.get("layer_trace", [])
        }
        print(
            f"[phi4-trace] loaded single-launch Rust trace with "
            f"{len(single_trace_by_layer)} layers"
        )
    py_component_traces = None
    if single_trace_by_layer:
        py_component_traces = python_all_component_traces(
            model,
            input_ids,
            args.python_incremental,
        )
    for layers in args.layers:
        if layers > original_layers:
            print(f"ERROR: requested {layers} layers, model has {original_layers}", file=sys.stderr)
            return 2

        if py_component_traces is not None:
            dtype = next(model.parameters()).dtype
            py_pre_norm = torch.tensor(
                py_component_traces[layers]["final_hidden"],
                dtype=dtype,
                device=device,
            ).view(1, 1, -1)
            with torch.no_grad():
                py_hidden_t = model.model.norm(py_pre_norm)[:, 0, :].detach()
        else:
            py_hidden_t = python_hidden_for_layers(model, input_ids, layers, args.python_incremental)
        py_hidden = py_hidden_t[0].float().cpu().numpy()

        if single_trace_by_layer:
            rust_pre_norm = np.asarray(
                single_trace_by_layer[layers]["final_hidden"],
                dtype=np.float32,
            )
        else:
            rust_pre_norm = rust_hidden_for_layers(
                args.binary,
                args.model_dir,
                args.model_variant,
                args.prompt,
                layers,
            )
        inv_rms = 1.0 / np.sqrt(np.mean(rust_pre_norm.astype(np.float64) ** 2) + eps)
        rust_hidden = f32_to_bf16_rounded((rust_pre_norm * inv_rms * norm_weight).astype(np.float32))
        py_hidden_by_layer[layers] = py_hidden_t
        rust_hidden_by_layer[layers] = rust_hidden

        delta = np.abs(py_hidden - rust_hidden)
        cosine = float(np.dot(py_hidden, rust_hidden) / (np.linalg.norm(py_hidden) * np.linalg.norm(rust_hidden)))
        row = {
            "layers": layers,
            "max_delta": float(delta.max()),
            "mean_delta": float(delta.mean()),
            "p99_delta": float(np.quantile(delta, 0.99)),
            "cosine": cosine,
        }
        results.append(row)
        print(
            f"[phi4-drift] layers={layers:2d} "
            f"max={row['max_delta']:.6f} mean={row['mean_delta']:.6f} "
            f"p99={row['p99_delta']:.6f} cos={row['cosine']:.8f}"
            )

    logit_result = None
    if args.logit_dump_layers is not None:
        if args.logit_dump_layers == "all":
            logit_layers = args.layers
        else:
            logit_layers = parse_layers(args.logit_dump_layers)
        logit_rows = []
        for layers in logit_layers:
            if layers not in py_hidden_by_layer:
                print(
                    f"ERROR: --logit-dump-layers includes {layers}, which must also be present in --layers",
                    file=sys.stderr,
                )
                return 2
            py_hidden_t = py_hidden_by_layer[layers]
            rust_hidden_t = torch.tensor(
                rust_hidden_by_layer[layers],
                dtype=py_hidden_t.dtype,
                device=device,
            ).unsqueeze(0)
            with torch.no_grad():
                py_logits_t = model.lm_head(py_hidden_t)[0]
                rust_logits_t = model.lm_head(rust_hidden_t)[0]
            py_logits = tensor_to_numpy(py_logits_t)
            rust_logits = tensor_to_numpy(rust_logits_t)
            row = delta_row("lm_head_logits", py_logits, rust_logits)
            logit_row = {
                **row,
                "layers": layers,
                "python_argmax": int(torch.argmax(py_logits_t).item()),
                "rust_hidden_argmax": int(torch.argmax(rust_logits_t).item()),
                "python_top": top_logits(py_logits_t, tokenizer),
                "rust_hidden_top": top_logits(rust_logits_t, tokenizer),
            }
            logit_rows.append(logit_row)
            print(
                f"[phi4-logits] layers={layers:2d} "
                f"max={row['max_delta']:.6f} mean={row['mean_delta']:.6f} "
                f"p99={row['p99_delta']:.6f} cos={row['cosine']:.8f} "
                f"py_argmax={logit_row['python_argmax']} "
                f"rust_hidden_argmax={logit_row['rust_hidden_argmax']}"
            )
        logit_result = logit_rows

    component_result = None
    if args.component_dump_layers is not None:
        if args.component_dump_layers == "all":
            component_layers = args.layers
        else:
            component_layers = parse_layers(args.component_dump_layers)
        component_results = []
        for layers in component_layers:
            if layers <= 0 or layers > original_layers:
                print(f"ERROR: requested component layers={layers}, model has {original_layers}", file=sys.stderr)
                return 2
            if layers not in py_hidden_by_layer:
                print(
                    f"ERROR: --component-dump-layers includes {layers}, which must also be present in --layers",
                    file=sys.stderr,
                )
                return 2
        for layers in component_layers:
            rust_dump = None
            if single_trace_by_layer:
                if layers not in single_trace_by_layer:
                    print(f"ERROR: single trace missing layer {layers}", file=sys.stderr)
                    return 2
                trace_row = single_trace_by_layer[layers]
            else:
                rust_dump = rust_component_dump_for_layers(
                    args.binary,
                    args.model_dir,
                    args.model_variant,
                    args.prompt,
                    layers,
                )
            if py_component_traces is not None:
                py_components = py_component_traces[layers]
            else:
                py_components = python_component_trace(model, input_ids, layers, args.python_incremental)
            if single_trace_by_layer:
                rust_components = {
                    "layer_input": np.asarray(trace_row["layer_input"], dtype=np.float32),
                    "post_attn_hidden": np.asarray(trace_row["post_attn_hidden"], dtype=np.float32),
                    "post_attn_normed": np.asarray(trace_row["post_attn_normed"], dtype=np.float32),
                    "attn_flat": np.asarray(trace_row["attn_flat"], dtype=np.float32),
                    "gate_raw": np.asarray(trace_row["gate_raw"], dtype=np.float32),
                    "up_raw": np.asarray(trace_row["up_raw"], dtype=np.float32),
                    "gate_up": np.asarray(trace_row["gate_up"], dtype=np.float32),
                    "down_raw": np.asarray(trace_row["down_raw"], dtype=np.float32),
                    "mlp_out": np.asarray(trace_row["mlp_out"], dtype=np.float32),
                    "final_hidden": np.asarray(trace_row["final_hidden"], dtype=np.float32),
                }
                for name in ["q_raw", "k_raw", "v_raw"]:
                    if name in trace_row:
                        rust_components[name] = np.asarray(trace_row[name], dtype=np.float32)
                rust_components["attn_out"] = (
                    rust_components["post_attn_hidden"] - rust_components["layer_input"]
                )
            else:
                rust_components = {
                    "post_attn_hidden": np.asarray(rust_dump["post_attn_hidden"], dtype=np.float32),
                    "post_attn_normed": np.asarray(rust_dump["post_attn_normed"], dtype=np.float32),
                    "mlp_out": np.asarray(rust_dump["mlp_out"], dtype=np.float32),
                    "final_hidden": np.asarray(rust_dump["final_hidden"], dtype=np.float32),
                }
                if layers > 1:
                    rust_components["layer_input"] = rust_hidden_for_layers(
                        args.binary,
                        args.model_dir,
                        args.model_variant,
                        args.prompt,
                        layers - 1,
                    )
                    rust_components["attn_out"] = (
                        rust_components["post_attn_hidden"] - rust_components["layer_input"]
                    )
            component_rows = []
            component_names = [
                "post_attn_hidden",
                "post_attn_normed",
                "mlp_out",
                "final_hidden",
            ]
            if "gate_up" in py_components and "gate_up" in rust_components:
                component_names.insert(2, "gate_up")
            if "up_raw" in py_components and "up_raw" in rust_components:
                component_names.insert(2, "up_raw")
            if "gate_raw" in py_components and "gate_raw" in rust_components:
                component_names.insert(2, "gate_raw")
            if "down_raw" in py_components and "down_raw" in rust_components:
                component_names.insert(5, "down_raw")
            if "attn_flat" in py_components and "attn_flat" in rust_components:
                component_names.insert(0, "attn_flat")
            for name in ["v_raw", "k_raw", "q_raw"]:
                if name in py_components and name in rust_components:
                    component_names.insert(0, name)
            if layers > 1 or single_trace_by_layer:
                component_names = ["layer_input", "attn_out"] + component_names
            for name in component_names:
                component_rows.append(
                    delta_row(name, py_components[name], rust_components[name])
                )
            replay_rows = []
            rust_qkv_replay = None
            py_qkv_replay = None
            if all(
                name in rust_components
                for name in ["layer_input", "q_raw", "k_raw", "v_raw"]
            ):
                rust_qkv_replay = qkv_replay_from_input(
                    model,
                    layers,
                    rust_components["layer_input"],
                )
                for name in ["q_raw", "k_raw", "v_raw"]:
                    replay_rows.append(
                        delta_row(
                            f"rust_qkv_replay_{name}",
                            rust_qkv_replay[name],
                            rust_components[name],
                        )
                    )
            if all(
                name in py_components
                for name in ["layer_input", "q_raw", "k_raw", "v_raw"]
            ):
                py_qkv_replay = qkv_replay_from_input(
                    model,
                    layers,
                    py_components["layer_input"],
                )
                for name in ["q_raw", "k_raw", "v_raw"]:
                    replay_rows.append(
                        delta_row(
                            f"py_qkv_replay_{name}",
                            py_qkv_replay[name],
                            py_components[name],
                        )
                    )
            if rust_qkv_replay is not None and py_qkv_replay is not None:
                for name in ["q_raw", "k_raw", "v_raw"]:
                    replay_rows.append(
                        delta_row(
                            f"py_vs_rust_qkv_replay_{name}",
                            py_qkv_replay[name],
                            rust_qkv_replay[name],
                        )
                    )
            rust_attn_replay = None
            py_attn_replay = None
            if all(
                name in rust_components
                for name in ["layer_input", "attn_flat", "post_attn_hidden"]
            ):
                rust_attn_replay = attn_replay_from_flat(
                    model,
                    layers,
                    rust_components["layer_input"],
                    rust_components["attn_flat"],
                )
                for name in ["attn_out", "post_attn_hidden"]:
                    replay_rows.append(
                        delta_row(
                            f"rust_attn_replay_{name}",
                            rust_attn_replay[name],
                            rust_components[name],
                        )
                    )
            if all(
                name in py_components
                for name in ["layer_input", "attn_flat", "post_attn_hidden"]
            ):
                py_attn_replay = attn_replay_from_flat(
                    model,
                    layers,
                    py_components["layer_input"],
                    py_components["attn_flat"],
                )
                for name in ["attn_out", "post_attn_hidden"]:
                    replay_rows.append(
                        delta_row(
                            f"py_attn_replay_{name}",
                            py_attn_replay[name],
                            py_components[name],
                        )
                    )
            if rust_attn_replay is not None and py_attn_replay is not None:
                for name in ["attn_out", "post_attn_hidden"]:
                    replay_rows.append(
                        delta_row(
                            f"py_vs_rust_attn_replay_{name}",
                            py_attn_replay[name],
                            rust_attn_replay[name],
                        )
                    )
            rust_input_replay = None
            py_input_replay = None
            if all(
                name in rust_components
                for name in ["post_attn_normed", "gate_raw", "up_raw", "gate_up"]
            ):
                rust_input_replay = gate_up_replay_from_normed(
                    model,
                    layers,
                    rust_components["post_attn_normed"],
                )
                for name in ["gate_raw", "up_raw", "gate_up"]:
                    replay_rows.append(
                        delta_row(
                            f"rust_input_replay_{name}",
                            rust_input_replay[name],
                            rust_components[name],
                        )
                    )
            if all(
                name in py_components
                for name in ["post_attn_normed", "gate_raw", "up_raw", "gate_up"]
            ):
                py_input_replay = gate_up_replay_from_normed(
                    model,
                    layers,
                    py_components["post_attn_normed"],
                )
                for name in ["gate_raw", "up_raw", "gate_up"]:
                    replay_rows.append(
                        delta_row(
                            f"py_input_replay_{name}",
                            py_input_replay[name],
                            py_components[name],
                        )
                    )
            if rust_input_replay is not None and py_input_replay is not None:
                for name in ["gate_raw", "up_raw", "gate_up"]:
                    replay_rows.append(
                        delta_row(
                            f"py_vs_rust_input_replay_{name}",
                            py_input_replay[name],
                            rust_input_replay[name],
                        )
                    )
            component_logit_rows = {}
            for name in ["post_attn_hidden", "final_hidden"]:
                py_logits = component_logits(
                    model,
                    py_components[name],
                    tokenizer,
                    device,
                    args.candidate_ids,
                )
                rust_logits = component_logits(
                    model,
                    rust_components[name],
                    tokenizer,
                    device,
                    args.candidate_ids,
                )
                component_logit_rows[name] = {
                    "python": py_logits,
                    "rust": rust_logits,
                }
            py_post = py_components["post_attn_hidden"]
            py_mlp = py_components["mlp_out"]
            rust_post = rust_components["post_attn_hidden"]
            rust_mlp = rust_components["mlp_out"]
            residual_mix_rows = {
                "py_post_plus_py_mlp": component_logits(
                    model,
                    py_post + py_mlp,
                    tokenizer,
                    device,
                    args.candidate_ids,
                ),
                "py_post_plus_rust_mlp": component_logits(
                    model,
                    py_post + rust_mlp,
                    tokenizer,
                    device,
                    args.candidate_ids,
                ),
                "rust_post_plus_py_mlp": component_logits(
                    model,
                    rust_post + py_mlp,
                    tokenizer,
                    device,
                    args.candidate_ids,
                ),
                "rust_post_plus_rust_mlp": component_logits(
                    model,
                    rust_post + rust_mlp,
                    tokenizer,
                    device,
                    args.candidate_ids,
                ),
            }
            component_results.append({
                "layers": layers,
                "rows": component_rows,
                "replay_rows": replay_rows,
                "logits": component_logit_rows,
                "residual_mixes": residual_mix_rows,
            })
            for row in component_rows:
                print(
                    f"[phi4-components] layers={layers:2d} {row['name']:<17} "
                    f"max={row['max_delta']:.6f} mean={row['mean_delta']:.6f} "
                    f"p99={row['p99_delta']:.6f} cos={row['cosine']:.8f}"
                )
            for row in replay_rows:
                print(
                    f"[phi4-replay] layers={layers:2d} {row['name']:<33} "
                    f"max={row['max_delta']:.6f} mean={row['mean_delta']:.6f} "
                    f"p99={row['p99_delta']:.6f} cos={row['cosine']:.8f}"
                )
            for name, logits in component_logit_rows.items():
                print(
                    f"[phi4-component-logits] layers={layers:2d} {name:<17} "
                    f"py_arg={logits['python']['argmax']} rust_arg={logits['rust']['argmax']} "
                    f"py_margin={logits['python'].get('top_margin', 0.0):.6f} "
                    f"rust_margin={logits['rust'].get('top_margin', 0.0):.6f}"
                )
            for name, logits in residual_mix_rows.items():
                print(
                    f"[phi4-residual-mix] layers={layers:2d} {name:<23} "
                    f"arg={logits['argmax']} "
                    f"top_margin={logits.get('top_margin', 0.0):.6f} "
                    f"cand_margin={logits.get('candidate_margin_0_minus_1', 0.0):.6f}"
                )
        component_result = component_results

    model.config.num_hidden_layers = original_layers
    if args.report:
        args.report.write_text(json.dumps({
            "prompt": args.prompt,
            "prompt_tokens": token_ids,
            "torch_backend": torch_backend,
            "results": results,
            "logit_result": logit_result,
            "component_result": component_result,
        }, indent=2))
        print(f"[phi4-drift] report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
