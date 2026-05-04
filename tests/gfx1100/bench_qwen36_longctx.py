#!/usr/bin/env python3
"""Benchmark Qwen3.6-MoE long-context decode on HIP/gfx1100.

The harness generates deterministic long prompts, runs the existing
SuperSonic CLI, and records stage timing plus VMM residency telemetry for
baseline INT4, KV-FP8, and optional sparse-MoE island cap variants.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MIB = 1024 * 1024
GIB = 1024 * MIB
TOKEN_CHARS = 2


@dataclass(frozen=True)
class BenchMode:
    label: str
    kv_fp8: bool
    sparse_cap: int | None = None
    prefetch_mode: str | None = None
    prefetch_ranks: str | None = None


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("list must contain at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError("values must be positive integers")
    seen: set[int] = set()
    deduped: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def parse_modes(raw: str, sparse_caps: list[int]) -> list[BenchMode]:
    modes: list[BenchMode] = []
    for part in raw.split(","):
        name = part.strip().lower().replace("_", "-")
        if not name:
            continue
        if name in {"int4", "vmm", "baseline", "int4-vmm"}:
            modes.append(BenchMode("int4-vmm", kv_fp8=False))
        elif name in {"kv-fp8", "int4-kv-fp8"}:
            modes.append(BenchMode("int4-kv-fp8", kv_fp8=True))
        elif name in {"sparse", "cap"}:
            for cap in sparse_caps:
                modes.append(BenchMode(f"cap{cap}", kv_fp8=False, sparse_cap=cap))
        elif name in {"sparse-kv-fp8", "cap-kv-fp8"}:
            for cap in sparse_caps:
                modes.append(BenchMode(f"cap{cap}-kv-fp8", kv_fp8=True, sparse_cap=cap))
        else:
            raise ValueError(f"unknown mode {part!r}")
    if not modes:
        raise ValueError("mode list is empty")
    seen: set[str] = set()
    deduped: list[BenchMode] = []
    for mode in modes:
        if mode.label in seen:
            continue
        seen.add(mode.label)
        deduped.append(mode)
    return deduped


def make_niah_prompt(context_tokens: int, seed: int) -> tuple[str, str]:
    """Return a deterministic NIAH-style prompt and expected answer string."""
    if context_tokens < 32:
        raise ValueError("context_tokens must be at least 32")
    needle = f"SSB-NEEDLE-{seed % 100000:05d}"
    intro = (
        "You are reading a long diagnostic log. Remember the secret value if "
        "one appears. "
    )
    question = (
        "\nQuestion: What is the secret value? Answer with only the secret value.\nAnswer:"
    )
    target_chars = max(256, context_tokens * TOKEN_CHARS)
    filler_words = [
        "cache",
        "route",
        "kernel",
        "tensor",
        "context",
        "expert",
        "prefetch",
        "latency",
        "resident",
        "decode",
        "attention",
        "wave",
    ]
    chunks: list[str] = [intro]
    i = 0
    halfway = target_chars // 2
    inserted = False
    while sum(len(chunk) for chunk in chunks) < target_chars - len(question):
        if not inserted and sum(len(chunk) for chunk in chunks) >= halfway:
            chunks.append(f" The secret value is {needle}. ")
            inserted = True
        word = filler_words[(i + seed) % len(filler_words)]
        chunks.append(f"{word}-{i % 997:03d} ")
        i += 1
    if not inserted:
        chunks.append(f" The secret value is {needle}. ")
    return "".join(chunks) + question, needle


def parse_stage_timings(output: str) -> dict[str, float]:
    match = re.search(r"\[qwen36-moe stage-timings\]\s+(.+)", output)
    if not match:
        return {}
    timings: dict[str, float] = {}
    for part in match.group(1).split():
        if "=" not in part:
            continue
        key, value = part.strip("()").split("=", 1)
        try:
            timings[key] = float(value)
        except ValueError:
            pass
    return timings


def parse_tokens(output: str) -> list[int]:
    match = re.search(r"\[tokens\]\s*\[([^\]]*)\]", output)
    if not match:
        match = re.search(r"Generated ids:\s*\[([^\]]*)\]", output)
    if not match:
        return []
    raw = match.group(1).strip()
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_generated_json(output: str) -> str | None:
    match = re.search(r"\[generated_json\]\s+(.+)", output)
    if not match:
        return None
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, str) else None


def parse_result(output: str) -> dict[str, float]:
    match = re.search(r"\[result\]\s+(.+)", output)
    if not match:
        summary = re.search(
            r"Generated\s+([0-9]+)\s+tokens?\s+\(([0-9]+)\s+prompt\s+\+\s+([0-9]+)\s+new\)",
            output,
        )
        if not summary:
            return {}
        generated_tokens, prompt_tokens, new_tokens = summary.groups()
        return {
            "generated_tokens": float(generated_tokens),
            "prompt_tokens": float(prompt_tokens),
            "new_tokens": float(new_tokens),
        }
    result: dict[str, float] = {}
    for part in match.group(1).split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        value = value.rstrip(",")
        try:
            result[key] = float(value)
        except ValueError:
            pass
    return result


def parse_mib_field(output: str, prefix: str, field: str) -> int | None:
    line = next((line for line in output.splitlines() if prefix in line), "")
    if not line:
        return None
    match = re.search(rf"(?:^|\s){re.escape(field)}=([0-9.]+)MiB(?:\s|$)", line)
    if not match:
        return None
    return int(round(float(match.group(1)) * MIB))


def load_sparse_telemetry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return payload.get("summary") or {}


def dense_vmm_residency(output: str) -> dict[str, Any]:
    moe_resident = parse_mib_field(output, "routed expert slabs active", "resident")
    moe_reserved = parse_mib_field(output, "routed expert slabs active", "reserved")
    kv_resident = parse_mib_field(output, "KV active", "resident") or 0
    kv_reserved = parse_mib_field(output, "KV active", "reserved") or 0
    return {
        "moe_resident_bytes": moe_resident,
        "moe_reserved_bytes": moe_reserved,
        "kv_resident_bytes": kv_resident,
        "kv_reserved_bytes": kv_reserved,
        "total_vmm_resident_bytes": (
            moe_resident + kv_resident if moe_resident is not None else None
        ),
        "total_vmm_reserved_bytes": (
            moe_reserved + kv_reserved if moe_reserved is not None else None
        ),
    }


def fmt_num(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def fmt_gib(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value) / GIB:.2f}"


def build_run_env(
    base_env: dict[str, str],
    args: argparse.Namespace,
    mode: BenchMode,
    telemetry_path: Path,
) -> dict[str, str]:
    env = base_env.copy()
    env["SUPERSONIC_BACKENDS"] = args.backend
    env.pop("SUPERSONIC_MOE_ISLAND_CAP_EXPERTS", None)
    env.pop("SUPERSONIC_MOE_ISLAND_TELEMETRY_JSON", None)
    env.pop("SUPERSONIC_MOE_ISLAND_PREFETCH", None)
    env.pop("SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS", None)
    env.pop("SUPERSONIC_VMM_MOE_ISLANDS", None)
    if args.force_moe_vmm or mode.sparse_cap is not None:
        env["SUPERSONIC_VMM_MOE_ISLANDS"] = "1"
    if mode.sparse_cap is not None:
        env["SUPERSONIC_MOE_ISLAND_CAP_EXPERTS"] = str(mode.sparse_cap)
        env["SUPERSONIC_MOE_ISLAND_TELEMETRY_JSON"] = str(telemetry_path)
        if mode.prefetch_mode:
            env["SUPERSONIC_MOE_ISLAND_PREFETCH"] = mode.prefetch_mode
        if mode.prefetch_ranks:
            env["SUPERSONIC_MOE_ISLAND_PREFETCH_RANKS"] = mode.prefetch_ranks
    return env


def run_one(
    args: argparse.Namespace,
    mode: BenchMode,
    context_tokens: int,
    prompt: str,
    expected_answer: str,
    tmp: Path,
    warmup: bool,
) -> dict[str, Any]:
    telemetry_path = tmp / f"{mode.label}-{context_tokens}.json"
    env = build_run_env(os.environ, args, mode, telemetry_path)

    cmd = [
        str(args.binary),
        "--backend",
        args.backend,
        "--model",
        "qwen3.6-35b-a3b",
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
    if mode.kv_fp8:
        cmd.append("--kv-fp8")
    if args.no_persistent_decode:
        cmd.append("--no-persistent-decode")

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
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        output = stdout + stderr
        return {
            "context_tokens_requested": context_tokens,
            "mode": mode.label,
            "kv_fp8": mode.kv_fp8,
            "sparse_cap": mode.sparse_cap,
            "returncode": -1,
            "timeout_seconds": args.timeout,
            "wall_seconds": elapsed,
            "command": cmd,
            "prompt_chars": len(prompt),
            "expected_answer": expected_answer,
            "generated_text": parse_generated_json(output),
            "niah_contains_expected": False,
            "generated_ids": parse_tokens(output),
            "stage": parse_stage_timings(output),
            "result": parse_result(output),
            "vmm_residency": dense_vmm_residency(output),
            "stdout_tail": stdout[-1600:],
            "stderr_tail": stderr[-3200:],
            "error": f"timed out after {args.timeout} seconds",
        }
    output = proc.stdout + proc.stderr
    elapsed = time.monotonic() - start
    generated_text = parse_generated_json(output)
    row: dict[str, Any] = {
        "context_tokens_requested": context_tokens,
        "mode": mode.label,
        "kv_fp8": mode.kv_fp8,
        "sparse_cap": mode.sparse_cap,
        "returncode": proc.returncode,
        "wall_seconds": elapsed,
        "command": cmd,
        "prompt_chars": len(prompt),
        "expected_answer": expected_answer,
        "generated_text": generated_text,
        "niah_contains_expected": (
            expected_answer in generated_text if generated_text is not None else False
        ),
        "generated_ids": parse_tokens(output),
        "stage": parse_stage_timings(output),
        "result": parse_result(output),
        "stdout_tail": proc.stdout[-1600:],
        "stderr_tail": proc.stderr[-3200:],
    }
    if proc.returncode != 0:
        row["error"] = output[-5000:]
        return row
    if mode.sparse_cap is not None and telemetry_path.exists():
        row["vmm_residency"] = load_sparse_telemetry(telemetry_path)
    else:
        row["vmm_residency"] = dense_vmm_residency(output)
    total_ms = row["stage"].get("total_ms_avg") or row["result"].get("ms_per_tok")
    row["tok_per_s"] = 1000.0 / total_ms if total_ms else None
    return row


def markdown(rows: list[dict[str, Any]]) -> str:
    out = [
        "| Context | Mode | wall s | total ms/tok | tok/s | prompt tokens | total resident GiB | MoE resident GiB | KV resident GiB | generated ids match | NIAH hit |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    baseline_ids_by_context: dict[int, list[int]] = {}
    for row in rows:
        if row["mode"] == "int4-vmm":
            baseline_ids_by_context[row["context_tokens_requested"]] = row.get("generated_ids") or []
    for row in rows:
        residency = row.get("vmm_residency") or {}
        stage = row.get("stage") or {}
        result = row.get("result") or {}
        context = row["context_tokens_requested"]
        generated_ids = row.get("generated_ids") or []
        baseline_ids = baseline_ids_by_context.get(context)
        ids_match = baseline_ids == generated_ids if baseline_ids is not None else None
        row["generated_ids_match_baseline"] = ids_match
        out.append(
            "| {context} | {mode} | {wall} | {ms} | {tok} | {prompt_tokens} | {total_gib} | {moe_gib} | {kv_gib} | {ids_match} | {niah} |".format(
                context=context,
                mode=row["mode"],
                wall=fmt_num(row.get("wall_seconds")),
                ms=fmt_num(stage.get("total_ms_avg") or result.get("ms_per_tok")),
                tok=fmt_num(row.get("tok_per_s")),
                prompt_tokens=fmt_num(result.get("prompt_tokens"), digits=0),
                total_gib=fmt_gib(residency.get("total_vmm_resident_bytes")),
                moe_gib=fmt_gib(residency.get("moe_resident_bytes")),
                kv_gib=fmt_gib(residency.get("kv_resident_bytes")),
                ids_match="yes" if ids_match else ("NO" if ids_match is False else "-"),
                niah="yes" if row.get("niah_contains_expected") else "NO",
            )
        )
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--model-dir", type=Path, default=Path("/mnt/data/models/Qwen3.6-35B-A3B"))
    parser.add_argument("--backend", default="hip")
    parser.add_argument("--contexts", default="8192,16384,32768")
    parser.add_argument("--modes", default="int4-vmm,int4-kv-fp8")
    parser.add_argument("--sparse-caps", default="320")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup-new-tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--no-persistent-decode", action="store_true")
    parser.add_argument(
        "--force-moe-vmm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="set SUPERSONIC_VMM_MOE_ISLANDS=1 for non-sparse modes too",
    )
    parser.add_argument("--out-json", type=Path, default=Path("target/qwen36_longctx.json"))
    parser.add_argument("--out-md", type=Path, default=Path("target/qwen36_longctx.md"))
    args = parser.parse_args()

    try:
        contexts = parse_int_list(args.contexts)
        sparse_caps = parse_int_list(args.sparse_caps)
        modes = parse_modes(args.modes, sparse_caps)
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be > 0")
    if args.warmup_new_tokens <= 0:
        parser.error("--warmup-new-tokens must be > 0")
    if not args.binary.exists():
        raise FileNotFoundError(args.binary)
    if not args.model_dir.exists():
        raise FileNotFoundError(args.model_dir)

    prompts: dict[int, tuple[str, str]] = {
        context: make_niah_prompt(context, args.seed + context) for context in contexts
    }
    rows: list[dict[str, Any]] = []
    tmp_owner = tempfile.TemporaryDirectory(prefix="qwen36-longctx-")
    tmp = Path(tmp_owner.name)
    for context in contexts:
        prompt, expected = prompts[context]
        for mode in modes:
            if not args.no_warmup:
                print(f"[warmup] context={context} mode={mode.label}", flush=True)
                run_one(args, mode, context, prompt, expected, tmp, warmup=True)
            print(f"[bench] context={context} mode={mode.label}", flush=True)
            row = run_one(args, mode, context, prompt, expected, tmp, warmup=False)
            rows.append(row)
            if row.get("returncode") != 0:
                print(row.get("error") or row.get("stderr_tail") or "run failed", file=sys.stderr)
                break
            stage = row.get("stage") or {}
            residency = row.get("vmm_residency") or {}
            print(
                f"  total_ms={stage.get('total_ms_avg')} tok/s={row.get('tok_per_s')} "
                f"resident_gib={fmt_gib(residency.get('total_vmm_resident_bytes'))} "
                f"niah={row.get('niah_contains_expected')} ids={row.get('generated_ids')}",
                flush=True,
            )

    md = markdown(rows)
    payload = {
        "schema": "qwen36-moe-longctx-bench-v1",
        "model": "qwen3.6-35b-a3b",
        "model_dir": str(args.model_dir),
        "backend": args.backend,
        "contexts": contexts,
        "modes": [mode.label for mode in modes],
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
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
