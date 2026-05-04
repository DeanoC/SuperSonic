#!/usr/bin/env python3
"""Run the Qwen KV quantization capture + harness matrix.

This is intentionally an opt-in research smoke, not a unit test. It expects
local HuggingFace Qwen3.5 model directories and a Python environment with
PyTorch/Transformers available.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_PROMPT = (
    "KV cache quantization should preserve attention score ordering for keys and "
    "weighted value sums for values. This capture prompt gives the Qwen "
    "full-attention layers a longer prefix with repeated technical vocabulary, "
    "mixed punctuation, and several clauses so the research harness can estimate "
    "per-layer and per-head attention-output error on real tensors. "
) * 3


def model_label(model_dir: Path) -> str:
    name = model_dir.name.lower().replace("qwen3.5-", "").replace(".", "_")
    return name.replace("-", "_")


def prompt_ids(model_dir: Path, tokens: int) -> str:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    ids = tokenizer.encode(DEFAULT_PROMPT, add_special_tokens=False)[:tokens]
    if len(ids) < tokens:
        raise RuntimeError(
            f"prompt only produced {len(ids)} tokens for {model_dir}; requested {tokens}"
        )
    return ",".join(str(tok) for tok in ids)


def run_checked(cmd: list[str], cwd: Path) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def summarize(report_path: Path) -> dict:
    payload = json.loads(report_path.read_text())
    rows = {row["scheme"]: row for row in payload["results"]}
    return {
        "shape": payload["shape"],
        "passing_schemes": payload["passing_schemes"],
        "recommended": payload["recommended_first_hip_candidate"],
        "fp8_max_rel_l2": rows["fp8_e4m3_token"]["max_layer_rel_l2"],
        "int4_max_rel_l2": rows["int4_token_group64"]["max_layer_rel_l2"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture real Qwen3.5 KV tensors and run quantization thresholds"
    )
    parser.add_argument(
        "--model-dir",
        action="append",
        type=Path,
        required=True,
        help="Local Qwen3.5 model directory. Repeat for multiple sizes.",
    )
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.035)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/supersonic-kv-quant"))
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="SuperSonic repository root.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    capture_script = args.repo_root / "oracle" / "qwen35_oracle.py"
    harness_script = args.repo_root / "oracle" / "kv_quant_research.py"
    summary = {}

    for model_dir in args.model_dir:
        label = model_label(model_dir)
        ids = prompt_ids(model_dir, args.tokens)
        capture_path = args.out_dir / f"qwen35-{label}-kv-capture-{args.tokens}.npz"
        report_path = args.out_dir / f"qwen35-{label}-kv-quant-report-{args.tokens}.json"
        run_checked(
            [
                sys.executable,
                str(capture_script),
                "--model-id",
                str(model_dir),
                "--prompt-ids",
                ids,
                "--max-new-tokens",
                "0",
                "--dtype",
                args.dtype,
                "--device",
                args.device,
                "--kv-quant-capture-npz",
                str(capture_path),
                "--kv-quant-capture-only",
            ],
            args.repo_root,
        )
        run_checked(
            [
                sys.executable,
                str(harness_script),
                "--input",
                str(capture_path),
                "--max-rel-l2-threshold",
                str(args.threshold),
                "--fail-on-threshold",
                "--output",
                str(report_path),
            ],
            args.repo_root,
        )
        summary[label] = summarize(report_path)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
