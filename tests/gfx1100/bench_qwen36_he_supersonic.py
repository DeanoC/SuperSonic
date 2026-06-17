#!/usr/bin/env python3
"""Run SuperSonic Qwen3.6 over the Lucebox HumanEval DFlash prompts.

This is a comparison harness, not a correctness gate: Lucebox benchmarks a
Qwen3.6-27B GGUF target plus DFlash draft. SuperSonic can run the same
tokenizer/config/source GGUF through a Q4KM-sourced native INT4 package on HIP.
The prompt set and generation length are kept identical.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_LUCEBOX_BENCH = Path("/home/deano/projects/lucebox-hub/server/scripts/bench_he.py")
RESULT_RE = re.compile(
    r"\[result\]\s+prompt_tokens=(?P<prompt>\d+)\s+"
    r"generated_tokens=(?P<generated>\d+)\s+"
    r"decode_ms=(?P<decode_ms>[0-9.]+)\s+"
    r"ms_per_(?:step|tok)=(?P<ms_per_step>[0-9.]+)"
)


def load_lucebox_prompts(path: Path) -> list[tuple[str, str]]:
    script_dir = str(path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("lucebox_bench_he", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import Lucebox bench script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompts = getattr(module, "PROMPTS", None)
    if not prompts:
        raise RuntimeError(f"no PROMPTS found in {path}")
    return [(str(name), str(prompt)) for name, prompt in prompts]


def run_one(args: argparse.Namespace, name: str, prompt: str, warmup: bool = False) -> dict:
    cmd = [
        str(args.binary),
        "--backend",
        args.backend,
        "--model",
        args.model,
        "--model-dir",
        str(args.model_dir),
        "--prompt",
        prompt,
        "--prompt-no-special-tokens",
        "--context-size",
        str(args.context_size),
        "--max-new-tokens",
        str(args.warmup_new_tokens if warmup else args.n_gen),
        "--temperature",
        "0",
        "--top-k",
        "1",
        "--sampling-seed",
        str(args.seed),
        "--no-download",
    ]
    if args.quant != "none":
        cmd.append(f"--{args.quant}")
    if args.ignore_eos:
        cmd.append("--ignore-eos")
    if args.emit_stage_timings:
        cmd.append("--emit-stage-timings")
    if args.kv_fp8:
        cmd.append("--kv-fp8")
    if args.dflash:
        cmd.append("--dflash")
        cmd.extend(["--dflash-draft-dir", str(args.dflash_draft_dir)])
        if args.dflash_block:
            cmd.extend(["--dflash-block", str(args.dflash_block)])

    env = os.environ.copy()
    env["SUPERSONIC_BACKENDS"] = args.backend

    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=args.timeout,
        env=env,
    )
    elapsed = time.monotonic() - start
    combined = proc.stdout + "\n" + proc.stderr
    match = RESULT_RE.search(combined)
    row = {
        "name": name,
        "returncode": proc.returncode,
        "wall_s": elapsed,
        "stdout_tail": proc.stdout[-args.tail_chars :],
        "stderr_tail": proc.stderr[-args.tail_chars :],
    }
    if match:
        row.update(
            {
                "prompt_tokens": int(match.group("prompt")),
                "generated_tokens": int(match.group("generated")),
                "decode_ms": float(match.group("decode_ms")),
                "ms_per_step": float(match.group("ms_per_step")),
                "tok_s": 1000.0 / float(match.group("ms_per_step")),
            }
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/mnt/data/tmp/supersonic-qwen36-27b-lucebox"),
    )
    parser.add_argument(
        "--quant",
        choices=["q4km-gptq", "q4km", "int4", "none"],
        default="q4km-gptq",
    )
    parser.add_argument("--lucebox-bench", type=Path, default=DEFAULT_LUCEBOX_BENCH)
    parser.add_argument("--backend", default="hip")
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--n-gen", type=int, default=256)
    parser.add_argument("--warmup-new-tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--tail-chars", type=int, default=4000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--emit-stage-timings", action="store_true")
    parser.add_argument("--kv-fp8", action="store_true")
    parser.add_argument("--dflash", action="store_true")
    parser.add_argument(
        "--dflash-draft-dir",
        type=Path,
        default=Path("/mnt/data/tmp/qwen36-27b-dflash-q8-bf16"),
    )
    parser.add_argument("--dflash-block", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=Path("target/qwen36_he_supersonic.json"))
    args = parser.parse_args()

    if not args.binary.exists():
        raise FileNotFoundError(args.binary)
    prompts = load_lucebox_prompts(args.lucebox_bench)
    if args.start_index < 0 or args.start_index > len(prompts):
        raise ValueError(f"--start-index must be in 0..{len(prompts)}")
    prompts = prompts[args.start_index :]
    if args.limit > 0:
        prompts = prompts[: args.limit]

    rows = []
    if args.warmup and prompts:
        print(f"[warmup] {prompts[0][0]} {args.warmup_new_tokens} tokens", flush=True)
        warmup = run_one(args, prompts[0][0], prompts[0][1], warmup=True)
        if warmup["returncode"] != 0:
            print(warmup["stderr_tail"] or warmup["stdout_tail"], file=sys.stderr)

    print(f"{'prompt':28s} {'ptok':>5s} {'gen':>5s} {'ms/tok':>8s} {'tok/s':>8s} {'wall_s':>8s}")
    print("-" * 70)
    for name, prompt in prompts:
        row = run_one(args, name, prompt)
        rows.append(row)
        if row["returncode"] != 0 or "tok_s" not in row:
            print(f"{name:28s} FAILED rc={row['returncode']}")
            print(row["stderr_tail"] or row["stdout_tail"], file=sys.stderr)
            continue
        print(
            f"{name:28s} {row['prompt_tokens']:5d} {row['generated_tokens']:5d} "
            f"{row['ms_per_step']:8.2f} {row['tok_s']:8.2f} {row['wall_s']:8.1f}",
            flush=True,
        )

    ok = [r for r in rows if r.get("returncode") == 0 and "tok_s" in r]
    if not ok:
        return 1
    summary = {
        "count": len(ok),
        "mean_tok_s": sum(r["tok_s"] for r in ok) / len(ok),
        "mean_ms_per_step": sum(r["ms_per_step"] for r in ok) / len(ok),
        "min_tok_s": min(r["tok_s"] for r in ok),
        "max_tok_s": max(r["tok_s"] for r in ok),
    }
    payload = {
        "schema": "supersonic-qwen36-he-comparison-v1",
        "model": args.model,
        "model_dir": str(args.model_dir),
        "quant": args.quant,
        "dflash": args.dflash,
        "dflash_draft_dir": str(args.dflash_draft_dir) if args.dflash else None,
        "dflash_block": args.dflash_block if args.dflash_block else None,
        "backend": args.backend,
        "context_size": args.context_size,
        "n_gen": args.n_gen,
        "lucebox_bench": str(args.lucebox_bench),
        "summary": summary,
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    print("-" * 70)
    print(f"{'MEAN':28s} {'':5s} {'':5s} {summary['mean_ms_per_step']:8.2f} {summary['mean_tok_s']:8.2f}")
    print(f"[wrote] {args.out_json}")
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
