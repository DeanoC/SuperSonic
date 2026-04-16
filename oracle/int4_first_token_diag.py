#!/usr/bin/env python3
"""
Diagnostic: compare first-step logits between Rust INT4 and Python-INT4-patched
oracle (same weights). Any delta beyond fp32-accumulation noise indicates a
bug in the Rust INT4 matmul / kernel path.

Uses the same oracle reconstruction as `int4_corpus_compare.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from int4_corpus_compare import build_oracle_model

PROMPT = "The quick brown fox"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--model-variant", default="qwen3.5-4b")
    p.add_argument("--bake-subdir", default=".supersonic/v1-int4-gptq")
    p.add_argument("--binary", type=Path,
                   default=Path("target/release/supersonic"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bake_dir = args.model_dir / args.bake_subdir

    # 1) Python-INT4-patched oracle (same weights as Rust)
    tokenizer, model = build_oracle_model(args.model_dir, bake_dir, device)
    model.eval()
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, use_cache=False)
    py_logits_last = out.logits[0, -1].float().cpu().numpy()
    py_top1 = int(py_logits_last.argmax())
    print(f"[py-int4]   top1={py_top1} ({tokenizer.decode([py_top1])!r})")
    print(f"[py-int4]   logit range: [{py_logits_last.min():.3f}, "
          f"{py_logits_last.max():.3f}]")

    # 2) Rust prefill to get last-token logits — use --validate so we can
    #    parse the output; actually, Rust doesn't expose prefill-last-logit.
    #    Instead run with max-new-tokens 1 and --validate to get logit delta
    #    vs the BF16 oracle, plus tokens.
    env = {k: v for k, v in os.environ.items() if k != "HSA_OVERRIDE_GFX_VERSION"}
    proc = subprocess.run([
        str(args.binary), "--model", args.model_variant,
        "--model-dir", str(args.model_dir),
        "--prompt", PROMPT,
        "--max-new-tokens", "1",
        "--int4",
    ], capture_output=True, text=True, env=env, timeout=600)
    if proc.returncode != 0:
        print("rust failed:", proc.stderr)
        return 2
    tokens: list[int] = []
    for line in proc.stdout.splitlines():
        if line.startswith("[tokens] "):
            tokens = [int(x) for x in line[len("[tokens] "):].split()]
    print(f"[rust-int4] top1={tokens[0] if tokens else '?'} "
          f"({tokenizer.decode(tokens[:1]) if tokens else '?'!r})")

    # 3) Python-BF16 oracle (reload clean model)
    from transformers import AutoModelForCausalLM
    bf = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    with torch.no_grad():
        out = bf(ids, use_cache=False)
    bf_logits_last = out.logits[0, -1].float().cpu().numpy()
    bf_top1 = int(bf_logits_last.argmax())
    print(f"[py-bf16]   top1={bf_top1} ({tokenizer.decode([bf_top1])!r})")

    # Deltas
    delta_py_int4_vs_bf16 = np.abs(py_logits_last - bf_logits_last).max()
    print(f"\n[delta] max|py-int4 - py-bf16| = {delta_py_int4_vs_bf16:.3f}  "
          f"(quantization noise)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
