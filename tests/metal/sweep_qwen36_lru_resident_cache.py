#!/usr/bin/env python3
"""Sweep Qwen3.6 Metal LRU resident-cache capacities."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    import sweep_qwen36_static_topn_runtime as runtime_sweep
except ModuleNotFoundError:
    script_path = Path(__file__).with_name("sweep_qwen36_static_topn_runtime.py")
    spec = importlib.util.spec_from_file_location(
        "sweep_qwen36_static_topn_runtime",
        script_path,
    )
    runtime_sweep = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = runtime_sweep
    spec.loader.exec_module(runtime_sweep)


SCHEMA = "qwen36-lru-resident-cache-sweep-v1"
MODEL = runtime_sweep.MODEL
DEFAULT_CAPACITIES = (32, 64)


def parse_capacities(raw: str) -> list[int]:
    capacities: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        try:
            capacity = int(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid capacity {text!r}") from exc
        if capacity <= 0:
            raise argparse.ArgumentTypeError("capacities must be positive")
        if capacity not in capacities:
            capacities.append(capacity)
    if not capacities:
        raise argparse.ArgumentTypeError("at least one capacity is required")
    return capacities


def hotset_mode(capacity: int) -> str:
    return f"lru-hotset-{capacity}"


def run_capacity_row(
    args: argparse.Namespace,
    prompt_id: str,
    prompt: str,
    capacity: int,
) -> dict[str, Any]:
    capacity_args = copy.copy(args)
    capacity_args.hotset_capacity = capacity
    row = runtime_sweep.run_row(capacity_args, prompt_id, prompt, "hotset")
    row["mode"] = hotset_mode(capacity)
    row["hotset_capacity"] = capacity
    return row


def build_rows(args: argparse.Namespace, capacities: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt_id, prompt in runtime_sweep.select_prompts(args):
        rows.append(runtime_sweep.run_row(args, prompt_id, prompt, "default"))
        for capacity in capacities:
            rows.append(run_capacity_row(args, prompt_id, prompt, capacity))
    return rows


def build_report(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    capacities: list[int],
    prompt_set: str,
) -> dict[str, Any]:
    modes = ["default", *(hotset_mode(capacity) for capacity in capacities)]
    return {
        "schema": SCHEMA,
        "model": MODEL,
        "prompt_set": prompt_set,
        "modes": modes,
        "capacities": capacities,
        "max_new_tokens": args.max_new_tokens,
        "context_size": args.context_size,
        "promotion_thresholds": {
            "max_headline_ratio": args.promotion_max_headline_ratio,
            "max_ffn_ratio": args.promotion_max_ffn_ratio,
            "max_component_regression_ratio": args.promotion_max_component_regression_ratio,
            "max_command_buffer_wait_ratio": args.promotion_max_command_buffer_wait_ratio,
            "require_profile": args.promotion_require_profile,
        },
        "summary": runtime_sweep.summarize_with_gate(
            rows,
            modes,
            args.promotion_max_headline_ratio,
            args.promotion_max_ffn_ratio,
            args.promotion_max_component_regression_ratio,
            args.promotion_max_command_buffer_wait_ratio,
            args.promotion_require_profile,
        ),
        "rows": rows,
    }


def render_float(value: Any, digits: int = 3) -> str:
    return runtime_sweep.render_float(value, digits)


def top_profile_op(profile: dict[str, Any] | None) -> dict[str, Any]:
    return runtime_sweep.top_profile_op(profile)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    promotion_gate = summary.get("promotion_gate") or {}
    lines = [
        "# Qwen3.6 LRU Resident Cache Sweep",
        "",
        f"- prompt_set: `{report['prompt_set']}`",
        f"- capacities: `{','.join(str(capacity) for capacity in report['capacities'])}`",
        f"- max_new_tokens: `{report['max_new_tokens']}`",
        f"- generated_ids_match: `{summary['generated_ids_match']}`",
        f"- promotion_gate_passed: `{promotion_gate.get('passed', False)}`",
        f"- promotion_gate_passed_modes: `{','.join(promotion_gate.get('passed_modes') or []) or '-'}`",
        "",
        "| Prompt | Mode | Status | IDs | Decode ms | FFN ms avg | Slot hit rate | Evictions | Copied GiB | Top Metal op | Top Metal ms | HAL ms | Wall s |",
        "|:---|:---|:---|:---|---:|---:|---:|---:|---:|:---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        residency = row.get("expert_residency") or {}
        result = row.get("result") or {}
        chain = row.get("chain_breakdown") or {}
        top_metal = top_profile_op(row.get("metal_profile"))
        hal_summary = (row.get("hal_profile") or {}).get("summary") or {}
        copied_gib = float(residency.get("copied_bytes", 0) or 0) / (1024.0**3)
        lines.append(
            "| {prompt} | {mode} | {status} | {ids} | {decode} | {ffn} | {slot} | {evictions} | {copied} | {top_op} | {top_ms} | {hal_ms} | {wall} |".format(
                prompt=row.get("prompt_id", ""),
                mode=row.get("mode", ""),
                status=row.get("status", ""),
                ids=",".join(str(item) for item in row.get("generated_ids", [])),
                decode=render_float(result.get("decode_ms")),
                ffn=render_float(chain.get("ffn_ms_avg")),
                slot=render_float(residency.get("slot_hit_rate"), 6),
                evictions=residency.get("evictions", "-"),
                copied=render_float(copied_gib, 3),
                top_op=top_metal.get("op") or "-",
                top_ms=render_float(top_metal.get("total_ms")),
                hal_ms=render_float(hal_summary.get("total_ms")),
                wall=render_float(row.get("wall_seconds"), 1),
            )
        )
    candidates = promotion_gate.get("candidates") or []
    if candidates:
        lines.extend(
            [
                "",
                "## Promotion Gate",
                "",
                "| Mode | Passed | Failures |",
                "|:---|:---:|:---|",
            ]
        )
        for candidate in candidates:
            failures = candidate.get("failures") or []
            lines.append(
                "| {mode} | {passed} | {failures} |".format(
                    mode=candidate.get("mode"),
                    passed=str(candidate.get("passed", False)).lower(),
                    failures=", ".join(str(item) for item in failures) or "-",
                )
            )
    lines.extend(
        [
            "",
            "This gate is nonfatal. A larger LRU resident cache is promotable only if generated IDs match default, headline decode and FFN time improve, component timings stay within threshold, and command-buffer-wait does not regress when profile evidence is required.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--prompt-set",
        choices=sorted(runtime_sweep.PROMPT_SETS),
        default="smoke",
    )
    parser.add_argument("--prompt", action="append", help="custom prompt; repeat for a suite")
    parser.add_argument(
        "--capacities",
        type=parse_capacities,
        default=list(DEFAULT_CAPACITIES),
        help="comma-separated LRU resident-cache capacities to sweep",
    )
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--metal-profile", action="store_true")
    parser.add_argument(
        "--promotion-max-headline-ratio",
        type=float,
        default=0.999,
        help="maximum candidate/default headline ms/token ratio for promotion",
    )
    parser.add_argument(
        "--promotion-max-ffn-ratio",
        type=float,
        default=0.999,
        help="maximum candidate/default ffn_ms_avg ratio for promotion",
    )
    parser.add_argument(
        "--promotion-max-component-regression-ratio",
        type=float,
        default=1.10,
        help="maximum allowed ratio for full-attn, linear-attn, and lm-head buckets",
    )
    parser.add_argument(
        "--promotion-max-command-buffer-wait-ratio",
        type=float,
        default=1.05,
        help="maximum candidate/default command_buffer_wait profile ratio",
    )
    parser.add_argument(
        "--promotion-require-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require command_buffer_wait profile evidence for promotion",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("target/qwen36_lru_resident_cache_sweep.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("target/qwen36_lru_resident_cache_sweep.md"),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.model_dir = runtime_sweep.resolve_model_dir(args.model_dir, os.environ)
    capacities = args.capacities
    prompt_set = "custom" if args.prompt else args.prompt_set
    rows = build_rows(args, capacities)
    report = build_report(rows, args, capacities, prompt_set)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report))

    summary = report["summary"]
    promotion_gate = summary.get("promotion_gate") or {}
    print(
        "[qwen36-lru-resident-cache-sweep] prompt_set={} rows={} ok={} generated_ids_match={} promotion_gate_passed={}".format(
            prompt_set,
            len(rows),
            sum(1 for row in rows if row.get("status") == "ok"),
            str(summary.get("generated_ids_match", False)).lower(),
            str(promotion_gate.get("passed", False)).lower(),
        )
    )
    print(f"[wrote] {args.out_json}")
    print(f"[wrote] {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
