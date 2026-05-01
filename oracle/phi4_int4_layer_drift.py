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


def parse_layers(value: str) -> list[int]:
    layers = [int(x) for x in value.split(",") if x.strip()]
    if not layers or any(n <= 0 for n in layers):
        raise argparse.ArgumentTypeError("layers must be a comma-separated list of positive ints")
    return layers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-variant", default="phi4-mini")
    parser.add_argument("--bake-subdir", default=".supersonic/v2-int4-gptq")
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--prompt", default="The quick brown fox")
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("1,2,4,8,16,24,32"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

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
    print(f"[phi4-drift] prompt={args.prompt!r} tokens={token_ids}")
    for layers in args.layers:
        if layers > original_layers:
            print(f"ERROR: requested {layers} layers, model has {original_layers}", file=sys.stderr)
            return 2

        model.config.num_hidden_layers = layers
        with torch.no_grad():
            py_out = model.model(input_ids=input_ids, use_cache=False)
        py_hidden = py_out.last_hidden_state[0, -1].float().cpu().numpy()

        rust_pre_norm = rust_hidden_for_layers(
            args.binary,
            args.model_dir,
            args.model_variant,
            args.prompt,
            layers,
        )
        inv_rms = 1.0 / np.sqrt(np.mean(rust_pre_norm.astype(np.float64) ** 2) + eps)
        rust_hidden = f32_to_bf16_rounded((rust_pre_norm * inv_rms * norm_weight).astype(np.float32))

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

    model.config.num_hidden_layers = original_layers
    if args.report:
        args.report.write_text(json.dumps({
            "prompt": args.prompt,
            "prompt_tokens": token_ids,
            "results": results,
        }, indent=2))
        print(f"[phi4-drift] report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
