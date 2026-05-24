#!/usr/bin/env python3
"""Sweep Qwen3.6 Metal static top-N residency modes over warm decode tokens."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MODEL = "qwen3.6-35b-a3b"
SCHEMA = "qwen36-static-topn-runtime-sweep-v1"

PROMPT_SETS: dict[str, list[tuple[str, str]]] = {
    "smoke": [
        (
            "hello",
            "Hello",
        ),
    ],
    "comparison": [
        (
            "profiling",
            "Inspect a local Apple Metal inference profile and identify the next optimization target from route locality, FFN time, and command-buffer waits.",
        ),
        (
            "coding",
            "Write a compact Rust helper that parses space-delimited key=value telemetry rows and returns a typed summary with numeric fields.",
        ),
    ],
}

MODE_ALIASES: dict[str, str] = {
    "baseline": "default",
    "default": "default",
    "packed": "packed",
    "hotset": "hotset",
    "static": "static",
    "static-hotset": "static-hotset",
}
DEFAULT_MODES = "default,static,static-hotset"


def parse_key_values(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for part in line.split():
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        values[key] = raw.rstrip(",)")
    return values


def parse_number(raw: str) -> int | float | str:
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_metric_line(output: str, prefix: str) -> dict[str, Any]:
    lines = [line for line in output.splitlines() if line.startswith(prefix)]
    if not lines:
        return {}
    return {key: parse_number(value) for key, value in parse_key_values(lines[-1]).items()}


def parse_result(output: str) -> dict[str, Any]:
    return parse_metric_line(output, "[result]")


def parse_stage_timings(output: str) -> dict[str, Any]:
    return parse_metric_line(output, "[qwen36-moe stage-timings]")


def parse_chain_breakdown(output: str) -> dict[str, Any]:
    return parse_metric_line(output, "[qwen36-moe chain-breakdown]")


def parse_lifecycle_timings(output: str) -> dict[str, Any]:
    return parse_metric_line(output, "[qwen36-moe lifecycle-timings]")


def parse_expert_residency(output: str) -> dict[str, Any] | None:
    parsed = parse_metric_line(output, "[qwen36-expert-residency]")
    return parsed or None


def parse_expert_residency_policies(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith("[qwen36-expert-residency-policy]"):
            continue
        rows.append(
            {key: parse_number(value) for key, value in parse_key_values(line).items()}
        )
    return rows


def parse_generated_ids(output: str) -> list[int]:
    match = re.search(r"Generated ids:\s*\[([^\]]*)\]", output)
    if match is None:
        return []
    raw = match.group(1).strip()
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_modes(raw: str) -> list[str]:
    modes: list[str] = []
    for part in raw.split(","):
        mode = MODE_ALIASES.get(part.strip())
        if mode is None:
            raise ValueError(f"unknown mode {part!r}; expected one of {sorted(MODE_ALIASES)}")
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("at least one mode is required")
    return modes


def select_prompts(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.prompt:
        return [(f"custom_{idx + 1}", prompt) for idx, prompt in enumerate(args.prompt)]
    return PROMPT_SETS[args.prompt_set]


def resolve_model_dir(raw_model_dir: Path | None, env: dict[str, str]) -> Path:
    if raw_model_dir is not None:
        return raw_model_dir
    if env.get("SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR"):
        return Path(env["SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR"])
    if env.get("SUPERSONIC_TEST_MODEL_ROOT"):
        return Path(env["SUPERSONIC_TEST_MODEL_ROOT"]) / MODEL
    return Path.home() / ".cache" / "supersonic-metal-models" / MODEL


def build_env_overrides(args: argparse.Namespace, mode: str) -> dict[str, str]:
    overrides = {
        "SUPERSONIC_BACKENDS": "metal",
        "SUPERSONIC_QWEN36_EXPERT_RESIDENCY_PROFILE": "1",
    }
    if args.metal_profile:
        overrides["SUPERSONIC_METAL_PROFILE"] = "1"
    if mode in {"packed", "hotset", "static", "static-hotset"}:
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACKED_STAGE5"] = "1"
    if mode in {"hotset", "static-hotset"}:
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACK_HOTSET"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_FFN_EXPERT_HOTSET_CAPACITY"] = str(
            args.hotset_capacity
        )
    if mode in {"static", "static-hotset"}:
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_STATIC_TOPN"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_FFN_EXPERT_STATIC_TOPN_FILE"] = str(
            args.static_table_json
        )
        if args.static_capacity is not None:
            overrides["SUPERSONIC_METAL_QWEN36_FFN_EXPERT_STATIC_TOPN_CAPACITY"] = str(
                args.static_capacity
            )
    return overrides


def build_command(args: argparse.Namespace, prompt: str) -> list[str]:
    return [
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
        str(args.context_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
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


def output_tail(output: str, limit: int = 5000) -> str:
    return output[-limit:]


def run_row(args: argparse.Namespace, prompt_id: str, prompt: str, mode: str) -> dict[str, Any]:
    env = os.environ.copy()
    env_overrides = build_env_overrides(args, mode)
    env.update(env_overrides)
    command = build_command(args, prompt)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            env=env,
        )
        wall_seconds = time.monotonic() - started
        output = proc.stdout + proc.stderr
        status = "ok" if proc.returncode == 0 else "failed"
        row: dict[str, Any] = {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "mode": mode,
            "status": status,
            "returncode": proc.returncode,
            "wall_seconds": wall_seconds,
            "env_overrides": env_overrides,
            "command": command,
            "generated_ids": parse_generated_ids(output),
            "result": parse_result(output),
            "stage_timings": parse_stage_timings(output),
            "chain_breakdown": parse_chain_breakdown(output),
            "lifecycle_timings": parse_lifecycle_timings(output),
            "expert_residency": parse_expert_residency(output),
            "expert_residency_policies": parse_expert_residency_policies(output),
            "output_tail": output_tail(output),
        }
        return row
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        return {
            "prompt_id": prompt_id,
            "prompt": prompt,
            "mode": mode,
            "status": "timeout",
            "returncode": None,
            "wall_seconds": time.monotonic() - started,
            "env_overrides": env_overrides,
            "command": command,
            "generated_ids": [],
            "result": {},
            "stage_timings": {},
            "chain_breakdown": {},
            "lifecycle_timings": {},
            "expert_residency": None,
            "expert_residency_policies": [],
            "output_tail": output_tail(str(output)),
        }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    reference_ids = ok_rows[0].get("generated_ids", []) if ok_rows else []
    mismatches = [
        {
            "prompt_id": row.get("prompt_id"),
            "mode": row.get("mode"),
            "generated_ids": row.get("generated_ids", []),
        }
        for row in ok_rows
        if row.get("generated_ids", []) != reference_ids
    ]
    return {
        "rows": len(rows),
        "ok_rows": len(ok_rows),
        "status_counts": {
            status: sum(1 for row in rows if row.get("status") == status)
            for status in sorted({str(row.get("status")) for row in rows})
        },
        "reference_generated_ids": reference_ids,
        "generated_ids_match": not mismatches,
        "generated_id_mismatches": mismatches,
    }


def build_report(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    modes: list[str],
    prompt_set: str,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "model": MODEL,
        "prompt_set": prompt_set,
        "modes": modes,
        "max_new_tokens": args.max_new_tokens,
        "context_size": args.context_size,
        "static_table_json": str(args.static_table_json),
        "static_capacity": args.static_capacity,
        "hotset_capacity": args.hotset_capacity,
        "summary": summarize(rows),
        "rows": rows,
    }


def render_float(value: Any, precision: int = 3) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Qwen3.6 Static Top-N Runtime Sweep",
        "",
        f"- prompt_set: `{report['prompt_set']}`",
        f"- modes: `{','.join(report['modes'])}`",
        f"- max_new_tokens: `{report['max_new_tokens']}`",
        f"- generated_ids_match: `{summary['generated_ids_match']}`",
        "",
        "| Prompt | Mode | Status | IDs | Decode ms | FFN ms avg | Exact hit rate | Slot hit rate | Copied GiB | Wall s |",
        "|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        residency = row.get("expert_residency") or {}
        result = row.get("result") or {}
        chain = row.get("chain_breakdown") or {}
        copied_gib = float(residency.get("copied_bytes", 0) or 0) / (1024.0**3)
        lines.append(
            "| {prompt} | {mode} | {status} | {ids} | {decode} | {ffn} | {exact} | {slot} | {copied} | {wall} |".format(
                prompt=row.get("prompt_id", ""),
                mode=row.get("mode", ""),
                status=row.get("status", ""),
                ids=",".join(str(item) for item in row.get("generated_ids", [])),
                decode=render_float(result.get("decode_ms")),
                ffn=render_float(chain.get("ffn_ms_avg")),
                exact=render_float(residency.get("exact_hit_rate"), 6),
                slot=render_float(residency.get("slot_hit_rate"), 6),
                copied=render_float(copied_gib, 3),
                wall=render_float(row.get("wall_seconds"), 1),
            )
        )
    lines.extend(
        [
            "",
            "Rows are separate process runs. Static modes measure within-run warm reuse across generated tokens; the first token still pays resident-table allocation on full-hit layers.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--prompt-set", choices=sorted(PROMPT_SETS), default="smoke")
    parser.add_argument("--prompt", action="append", help="custom prompt; repeat for a suite")
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--static-table-json", type=Path, default=Path("target/qwen36_static_topn_mps_probe.json"))
    parser.add_argument("--static-capacity", type=int)
    parser.add_argument("--hotset-capacity", type=int, default=64)
    parser.add_argument("--metal-profile", action="store_true")
    parser.add_argument("--out-json", type=Path, default=Path("target/qwen36_static_topn_runtime_sweep.json"))
    parser.add_argument("--out-md", type=Path, default=Path("target/qwen36_static_topn_runtime_sweep.md"))
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.model_dir = resolve_model_dir(args.model_dir, os.environ)
    modes = parse_modes(args.modes)
    if any(mode in {"static", "static-hotset"} for mode in modes) and not args.static_table_json.exists():
        print(
            f"[qwen36-static-topn-runtime-sweep] error=missing_static_table_json path={args.static_table_json}",
            file=sys.stderr,
        )
        return 2
    prompts = select_prompts(args)
    prompt_set = "custom" if args.prompt else args.prompt_set
    rows: list[dict[str, Any]] = []
    for prompt_id, prompt in prompts:
        for mode in modes:
            rows.append(run_row(args, prompt_id, prompt, mode))
    report = build_report(rows, args, modes, prompt_set)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report))

    summary = report["summary"]
    print(
        "[qwen36-static-topn-runtime-sweep] rows={} ok={} generated_ids_match={}".format(
            summary["rows"],
            summary["ok_rows"],
            str(summary["generated_ids_match"]).lower(),
        )
    )
    print(f"[wrote] {args.out_json}")
    print(f"[wrote] {args.out_md}")
    return 0 if summary["ok_rows"] == summary["rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
