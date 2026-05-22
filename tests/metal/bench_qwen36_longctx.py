#!/usr/bin/env python3
"""Benchmark Qwen3.6-MoE long-context decode on Apple Metal.

This is the Apple M5 Max companion to ``tests/gfx1100/bench_qwen36_longctx.py``.
It keeps the same row/report shape, but constrains the lane to the supported
Metal v1 path: Qwen3.6-35B-A3B INT4, chained decode, BF16 KV, no sparse/VMM
or speculative modes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any


MODEL = "qwen3.6-35b-a3b"
SCHEMA = "qwen36-moe-metal-longctx-bench-v1"

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "contexts": "512",
        "max_new_tokens": 1,
        "warmup": False,
        "timeout": 1800,
    },
    "comparison": {
        "contexts": "512,2048,8192",
        "max_new_tokens": 4,
        "warmup": True,
        "timeout": 21600,
    },
    "full": {
        "contexts": "512,2048,8192",
        "max_new_tokens": 16,
        "warmup": True,
        "timeout": 28800,
    },
}


def load_base() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "gfx1100" / "bench_qwen36_longctx.py"
    spec = importlib.util.spec_from_file_location("qwen36_longctx_base", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base()


def apply_preset_defaults(args: argparse.Namespace) -> argparse.Namespace:
    preset = PRESETS.get(args.preset or "comparison", {})
    args.contexts = args.contexts or preset.get("contexts") or "512,2048,8192"
    args.max_new_tokens = args.max_new_tokens or preset.get("max_new_tokens") or 4
    args.timeout = args.timeout or preset.get("timeout") or 1800
    if args.warmup is None:
        args.warmup = bool(preset.get("warmup", True))
    return args


def resolve_model_dir(raw_model_dir: Path | None, env: dict[str, str]) -> Path:
    if raw_model_dir is not None:
        return raw_model_dir
    if env.get("SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR"):
        return Path(env["SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR"])
    if env.get("SUPERSONIC_TEST_MODEL_ROOT"):
        return Path(env["SUPERSONIC_TEST_MODEL_ROOT"]) / MODEL
    return Path.home() / ".cache" / "supersonic-metal-models" / MODEL


def build_metal_env(base_env: dict[str, str]) -> dict[str, str]:
    env = base_env.copy()
    env["SUPERSONIC_BACKENDS"] = "metal"

    for key in (
        "SUPERSONIC_VMM_KV",
        "SUPERSONIC_VMM_MOE_ISLANDS",
        "SUPERSONIC_MOE_ISLAND_CAP_EXPERTS",
        "SUPERSONIC_MOE_ISLAND_TELEMETRY_JSON",
        "SUPERSONIC_MOE_ISLAND_PREFETCH",
        "SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS",
    ):
        env.pop(key, None)

    env["SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL"] = "0"
    env["SUPERSONIC_QWEN36_MOE_BATCHED_ATTN"] = "0"
    env["SUPERSONIC_QWEN36_MOE_GROUPED_FFN"] = "0"
    env["SUPERSONIC_QWEN36_DENSE_PREFILL_TOKEN_LOOP"] = "1"
    return env


def run_one(
    args: argparse.Namespace,
    context_tokens: int,
    prompt: str,
    expected_answer: str,
    warmup: bool,
) -> dict[str, Any]:
    env = build_metal_env(os.environ)
    cmd = [
        str(args.binary),
        "--backend",
        "metal",
        "--model",
        MODEL,
        "--model-dir",
        str(args.model_dir),
        "--int4",
        "--prompt",
        prompt,
        "--context-size",
        str(context_tokens),
        "--max-new-tokens",
        str(args.warmup_new_tokens if warmup else args.max_new_tokens),
        "--temperature",
        "0",
        "--top-k",
        "1",
        "--sampling-seed",
        str(args.seed),
        "--no-download",
        "--emit-stage-timings",
        "--emit-generated-json",
    ]

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        output = stdout + stderr
        return {
            "context_tokens_requested": context_tokens,
            "mode": "int4",
            "returncode": -1,
            "timeout_seconds": args.timeout,
            "wall_seconds": elapsed,
            "command": cmd,
            "prompt_chars": len(prompt),
            "expected_answer": expected_answer,
            "generated_text": BASE.parse_generated_json(output),
            "niah_contains_expected": False,
            "generated_ids": BASE.parse_tokens(output),
            "stage": BASE.parse_stage_timings(output),
            "chain_breakdown": BASE.parse_chain_breakdown(output),
            "lifecycle": BASE.parse_lifecycle_timings(output),
            "result": BASE.parse_result(output),
            "vmm_residency": BASE.dense_vmm_residency(output),
            "stdout_tail": stdout[-1600:],
            "stderr_tail": stderr[-3200:],
            "error": f"timed out after {args.timeout} seconds",
        }

    output = proc.stdout + proc.stderr
    elapsed = time.monotonic() - start
    generated_text = BASE.parse_generated_json(output)
    row: dict[str, Any] = {
        "context_tokens_requested": context_tokens,
        "mode": "int4",
        "returncode": proc.returncode,
        "wall_seconds": elapsed,
        "command": cmd,
        "prompt_chars": len(prompt),
        "expected_answer": expected_answer,
        "generated_text": generated_text,
        "niah_contains_expected": (
            expected_answer in generated_text if generated_text is not None else False
        ),
        "generated_ids": BASE.parse_tokens(output),
        "stage": BASE.parse_stage_timings(output),
        "chain_breakdown": BASE.parse_chain_breakdown(output),
        "lifecycle": BASE.parse_lifecycle_timings(output),
        "result": BASE.parse_result(output),
        "vmm_residency": BASE.dense_vmm_residency(output),
        "stdout_tail": proc.stdout[-1600:],
        "stderr_tail": proc.stderr[-3200:],
    }
    if proc.returncode != 0:
        row["error"] = output[-5000:]
        return row
    total_ms = row["stage"].get("total_ms_avg") or row["result"].get("ms_per_tok")
    row["tok_per_s"] = 1000.0 / total_ms if total_ms else None
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="comparison",
        help="named sweep defaults; explicit CLI flags override preset values",
    )
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--contexts")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--warmup-new-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--timeout", type=int)
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="run one short warmup before each measured context",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("target/qwen36_metal_longctx.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("target/qwen36_metal_longctx.md"),
    )
    args = apply_preset_defaults(parser.parse_args())
    args.model_dir = resolve_model_dir(args.model_dir, os.environ)

    try:
        contexts = BASE.parse_int_list(args.contexts)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be > 0")
    if args.warmup_new_tokens <= 0:
        parser.error("--warmup-new-tokens must be > 0")
    if not args.binary.exists():
        raise FileNotFoundError(args.binary)
    if not args.model_dir.exists():
        raise FileNotFoundError(
            f"{args.model_dir}; set SUPERSONIC_TEST_MODEL_ROOT or pass --model-dir"
        )

    prompts = {
        context: BASE.make_niah_prompt(context, args.seed + context)
        for context in contexts
    }
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="qwen36-metal-longctx-"):
        for context in contexts:
            prompt, expected = prompts[context]
            if args.warmup:
                print(f"[warmup] context={context} mode=int4", flush=True)
                run_one(args, context, prompt, expected, warmup=True)
            print(f"[bench] context={context} mode=int4", flush=True)
            row = run_one(args, context, prompt, expected, warmup=False)
            rows.append(row)
            if row.get("returncode") != 0:
                print(row.get("error") or row.get("stderr_tail") or "run failed", file=sys.stderr)
                break
            stage = row.get("stage") or {}
            lifecycle = row.get("lifecycle") or {}
            print(
                f"  total_ms={stage.get('total_ms_avg')} tok/s={row.get('tok_per_s')} "
                f"prefill_ms={lifecycle.get('prefill_total_ms')} "
                f"niah={row.get('niah_contains_expected')} ids={row.get('generated_ids')}",
                flush=True,
            )

    summary = BASE.summarize(rows)
    md = BASE.markdown(rows, summary)
    payload = {
        "schema": SCHEMA,
        "model": MODEL,
        "model_dir": str(args.model_dir),
        "backend": "metal",
        "preset": args.preset,
        "contexts": contexts,
        "modes": ["int4"],
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "summary": summary,
        "recommendation": BASE.recommendation(summary),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md)
    print(f"[wrote] {args.out_json}")
    print(f"[wrote] {args.out_md}")
    print(md, end="")
    return 1 if any(row.get("returncode") != 0 for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
