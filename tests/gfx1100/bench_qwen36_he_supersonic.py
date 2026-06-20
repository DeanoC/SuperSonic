#!/usr/bin/env python3
"""Run SuperSonic Qwen3.6 over the Lucebox HumanEval DFlash prompts.

This is a comparison harness, not a correctness gate. It can run either the
Lucebox-style Qwen3.6-27B GGUF target plus DFlash draft, or the native
Qwen3.6-35B-A3B MoE path over the same HumanEval prompt set.
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
DEFAULT_LUCEBOX_HE_JSONL = Path(
    "/home/deano/projects/lucebox-hub/harness/benchmarks/prompts/bench_he.jsonl"
)
DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR = Path("/mnt/data/tmp/qwen36-27b-dflash-q8-bf16")
DEFAULT_LUCEBOX_DRAFT_DIR = Path("/mnt/data/lucebox-hub/models/draft")
DEFAULT_27B_MODEL = "qwen3.6-27b"
DEFAULT_27B_MODEL_DIR = Path("/mnt/data/tmp/supersonic-qwen36-27b-lucebox")
DEFAULT_27B_QUANT = "q4km-gptq"
DEFAULT_35B_A3B_MODEL = "qwen3.6-35b-a3b"
DEFAULT_35B_A3B_MODEL_DIR = Path("/mnt/data/models/Qwen3.6-35B-A3B")
DEFAULT_35B_A3B_QUANT = "int4"
DEFAULT_OUT_JSON = Path("target/qwen36_he_supersonic.json")
DEFAULT_35B_A3B_OUT_JSON = Path("target/qwen36_35b_a3b_he_supersonic.json")
DEFAULT_CONTEXT_SIZE = 512
LUCEBOX_SERVING_CONTEXT_SIZE = 1024
LUCEBOX_DRAFT_ALIASES = {
    "supersonic-q8-bf16": {
        "label": "supersonic-q8-bf16",
        "config_dir": DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR,
        "gguf": None,
    },
    "lucebox-q4-k-m": {
        "label": "lucebox-q4-k-m",
        "config_dir": DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR,
        "gguf": DEFAULT_LUCEBOX_DRAFT_DIR / "dflash-draft-3.6-q4_k_m.gguf",
    },
    "lucebox-q8-0": {
        "label": "lucebox-q8-0",
        "config_dir": DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR,
        "gguf": DEFAULT_LUCEBOX_DRAFT_DIR / "dflash-draft-3.6-q8_0.gguf",
    },
}
RESULT_RE = re.compile(
    r"\[result\]\s+prompt_tokens=(?P<prompt>\d+)\s+"
    r"generated_tokens=(?P<generated>\d+)\s+"
    r"decode_ms=(?P<decode_ms>[0-9.]+)\s+"
    r"ms_per_(?:step|tok)=(?P<ms_per_step>[0-9.]+)"
)


TARGET_PROFILES = {
    "qwen36-27b-lucebox": {
        "model": DEFAULT_27B_MODEL,
        "model_dir": DEFAULT_27B_MODEL_DIR,
        "quant": DEFAULT_27B_QUANT,
        "out_json": DEFAULT_OUT_JSON,
    },
    "qwen36-35b-a3b": {
        "model": DEFAULT_35B_A3B_MODEL,
        "model_dir": DEFAULT_35B_A3B_MODEL_DIR,
        "quant": DEFAULT_35B_A3B_QUANT,
        "out_json": DEFAULT_35B_A3B_OUT_JSON,
    },
}


def load_lucebox_script_prompts(path: Path) -> list[tuple[str, str]]:
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


QWEN3_NO_THINKING_PREFILL = "<think>\n\n</think>\n\n"


def render_chatml(messages: list[dict], *, no_thinking: bool = False) -> str:
    chunks = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    chunks.append("<|im_start|>assistant\n")
    if no_thinking:
        chunks.append(QWEN3_NO_THINKING_PREFILL)
    return "".join(chunks)


def load_lucebox_jsonl_prompts(path: Path, prompt_format: str) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            name = str(case.get("id") or case.get("name") or f"case_{idx:03d}")
            messages = case.get("messages")
            if isinstance(messages, list) and prompt_format in {
                "chatml",
                "chatml-no-thinking",
            }:
                prompt = render_chatml(
                    messages, no_thinking=prompt_format == "chatml-no-thinking"
                )
            elif isinstance(messages, list) and messages:
                prompt = "\n\n".join(str(m.get("content", "")) for m in messages)
            else:
                prompt = str(case.get("prompt") or case.get("content") or "")
            if not prompt:
                raise RuntimeError(f"empty prompt in {path}:{idx}")
            prompts.append((name, prompt))
    if not prompts:
        raise RuntimeError(f"no prompts found in {path}")
    return prompts


def load_prompts(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.prompt_source == "script":
        return load_lucebox_script_prompts(args.lucebox_bench)
    if args.prompt_source == "jsonl":
        return load_lucebox_jsonl_prompts(args.lucebox_jsonl, args.prompt_format)
    raise ValueError(f"unknown prompt source: {args.prompt_source}")


def resolve_dflash_draft(args: argparse.Namespace) -> tuple[Path, Path | None, str]:
    config_dir = args.dflash_draft_dir
    gguf = args.dflash_draft_gguf
    label = "custom"
    if args.dflash_draft_variant:
        alias = LUCEBOX_DRAFT_ALIASES[args.dflash_draft_variant]
        label = alias["label"]
        config_dir = alias["config_dir"]
        gguf = alias["gguf"]
    if config_dir is None:
        config_dir = DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR
    return Path(config_dir), Path(gguf) if gguf else None, label


def apply_target_profile(args: argparse.Namespace) -> None:
    profile = TARGET_PROFILES[args.target_profile]
    if args.model is None:
        args.model = profile["model"]
    if args.model_dir is None:
        args.model_dir = profile["model_dir"]
    if args.quant is None:
        args.quant = profile["quant"]
    if args.out_json is None:
        args.out_json = profile["out_json"]


def apply_lucebox_serving_mode(args: argparse.Namespace) -> None:
    args.prompt_source = "jsonl"
    args.prompt_format = "chatml-no-thinking"
    args.ignore_eos = False
    if args.target_profile == "qwen36-27b-lucebox":
        args.dflash = True
        args.dflash_draft_variant = args.dflash_draft_variant or "lucebox-q4-k-m"
    if args.context_size == DEFAULT_CONTEXT_SIZE:
        args.context_size = LUCEBOX_SERVING_CONTEXT_SIZE


def run_one(args: argparse.Namespace, name: str, prompt: str, warmup: bool = False) -> dict:
    dflash_draft_dir, dflash_draft_gguf, dflash_draft_label = resolve_dflash_draft(args)
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
    if args.prompt_no_special_tokens:
        cmd.append("--prompt-no-special-tokens")
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
        cmd.extend(["--dflash-draft-dir", str(dflash_draft_dir)])
        if args.dflash_block:
            cmd.extend(["--dflash-block", str(args.dflash_block)])

    env = os.environ.copy()
    env["SUPERSONIC_BACKENDS"] = args.backend
    if dflash_draft_gguf is not None:
        env["SUPERSONIC_DFLASH_DRAFT_GGUF"] = str(dflash_draft_gguf)

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
                "requested_tokens": args.warmup_new_tokens if warmup else args.n_gen,
                "stopped_early": int(match.group("generated"))
                < (args.warmup_new_tokens if warmup else args.n_gen),
                "decode_ms": float(match.group("decode_ms")),
                "ms_per_step": float(match.group("ms_per_step")),
                "tok_s": 1000.0 / float(match.group("ms_per_step")),
            }
        )
    if args.dflash:
        row["dflash_draft_label"] = dflash_draft_label
        row["dflash_draft_dir"] = str(dflash_draft_dir)
        row["dflash_draft_gguf"] = str(dflash_draft_gguf) if dflash_draft_gguf else None
    return row


def build_summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("returncode") == 0 and "tok_s" in r]
    total_generated = sum(r["generated_tokens"] for r in ok)
    total_decode_ms = sum(r["decode_ms"] for r in ok)
    return {
        "count": len(ok),
        "mean_tok_s": sum(r["tok_s"] for r in ok) / len(ok),
        "weighted_tok_s": (1000.0 * total_generated / total_decode_ms)
        if total_decode_ms
        else 0.0,
        "mean_ms_per_step": sum(r["ms_per_step"] for r in ok) / len(ok),
        "min_tok_s": min(r["tok_s"] for r in ok),
        "max_tok_s": max(r["tok_s"] for r in ok),
        "total_generated_tokens": total_generated,
        "total_decode_ms": total_decode_ms,
        "stopped_early_count": sum(1 for r in ok if r.get("stopped_early")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument(
        "--target-profile",
        choices=sorted(TARGET_PROFILES),
        default="qwen36-27b-lucebox",
        help=(
            "Default model/model-dir/quant/output bundle. Explicit individual "
            "arguments still take precedence."
        ),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--quant",
        choices=["q4km-gptq", "q4km", "int4", "none"],
        default=None,
    )
    parser.add_argument("--lucebox-bench", type=Path, default=DEFAULT_LUCEBOX_BENCH)
    parser.add_argument("--lucebox-jsonl", type=Path, default=DEFAULT_LUCEBOX_HE_JSONL)
    parser.add_argument("--prompt-source", choices=["script", "jsonl"], default="script")
    parser.add_argument(
        "--prompt-format",
        choices=["raw", "chatml", "chatml-no-thinking"],
        default="raw",
    )
    parser.add_argument(
        "--lucebox-serving-mode",
        action="store_true",
        help=(
            "Preset for Lucebox HTTP-style comparison: JSONL HE prompts, "
            "Qwen3 no-thinking ChatML prompt text, stop on EOS, and "
            "Lucebox Q4_K_M draft GGUF."
        ),
    )
    parser.add_argument("--backend", default="hip")
    parser.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE)
    parser.add_argument("--n-gen", type=int, default=256)
    parser.add_argument("--warmup-new-tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--tail-chars", type=int, default=4000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-eos", action="store_true")
    parser.add_argument(
        "--prompt-no-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--emit-stage-timings", action="store_true")
    parser.add_argument("--kv-fp8", action="store_true")
    parser.add_argument("--dflash", action="store_true")
    parser.add_argument(
        "--dflash-draft-variant",
        choices=sorted(LUCEBOX_DRAFT_ALIASES),
        default=None,
        help=(
            "Convenience draft selection. Lucebox variants use their GGUF draft "
            "weights with the local SuperSonic DFlash config directory."
        ),
    )
    parser.add_argument(
        "--dflash-draft-dir",
        type=Path,
        default=DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR,
    )
    parser.add_argument(
        "--dflash-draft-gguf",
        type=Path,
        default=None,
        help=(
            "Optional Lucebox-style GGUF draft weights. The config still comes "
            "from --dflash-draft-dir."
        ),
    )
    parser.add_argument("--dflash-block", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    apply_target_profile(args)
    if args.lucebox_serving_mode:
        apply_lucebox_serving_mode(args)
    if args.stop_on_eos:
        args.ignore_eos = False

    if not args.binary.exists():
        raise FileNotFoundError(args.binary)
    if args.dflash:
        dflash_draft_dir, dflash_draft_gguf, _draft_label = resolve_dflash_draft(args)
        if not dflash_draft_dir.exists():
            raise FileNotFoundError(dflash_draft_dir)
        if dflash_draft_gguf is not None and not dflash_draft_gguf.exists():
            raise FileNotFoundError(dflash_draft_gguf)
    prompts = load_prompts(args)
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
    summary = build_summary(rows)
    dflash_draft_dir, dflash_draft_gguf, dflash_draft_label = resolve_dflash_draft(args)
    payload = {
        "schema": "supersonic-qwen36-he-comparison-v2",
        "model": args.model,
        "model_dir": str(args.model_dir),
        "quant": args.quant,
        "dflash": args.dflash,
        "dflash_draft_label": dflash_draft_label if args.dflash else None,
        "dflash_draft_dir": str(dflash_draft_dir) if args.dflash else None,
        "dflash_draft_gguf": str(dflash_draft_gguf) if args.dflash and dflash_draft_gguf else None,
        "dflash_block": args.dflash_block if args.dflash_block else None,
        "backend": args.backend,
        "context_size": args.context_size,
        "n_gen": args.n_gen,
        "eos_policy": "ignore" if args.ignore_eos else "stop",
        "prompt_source": args.prompt_source,
        "prompt_format": args.prompt_format,
        "lucebox_bench": str(args.lucebox_bench),
        "lucebox_jsonl": str(args.lucebox_jsonl),
        "lucebox_serving_mode": args.lucebox_serving_mode,
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
