#!/usr/bin/env python3
"""Sweep Qwen3.6 Metal batched-prefill prototype variants."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bench_qwen36_longctx as bench  # noqa: E402


SCHEMA = "qwen36-metal-batched-prefill-variant-sweep-v1"
MODE_ALIASES: dict[str, str] = {
    "baseline": "baseline",
    "supported": "baseline",
    "prototype": "prototype-default",
    "prototype-default": "prototype-default",
    "default": "prototype-default",
    **{
        name: name
        for name in bench.BATCHED_PREFILL_VARIANTS
        if name != "default"
    },
}
DEFAULT_MODES = (
    "baseline,prototype-default,linear-direct-off,full-attn-tmajor,"
    "split-qgate,router-topk,fused-residual"
)


def parse_modes(raw: str) -> list[str]:
    modes: list[str] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        mode = MODE_ALIASES.get(stripped)
        if mode is None:
            raise ValueError(
                f"unknown mode {stripped!r}; expected one of {sorted(MODE_ALIASES)}"
            )
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("at least one mode is required")
    return modes


def variant_for_mode(mode: str) -> str:
    if mode == "baseline":
        return "default"
    if mode == "prototype-default":
        return "default"
    return mode


def prototype_enabled(mode: str) -> bool:
    return mode != "baseline"


def args_for_mode(args: argparse.Namespace, mode: str) -> argparse.Namespace:
    run_args = copy.copy(args)
    run_args.batched_prefill_prototype = prototype_enabled(mode)
    run_args.batched_prefill_variant = variant_for_mode(mode)
    return run_args


def status_for_row(row: dict[str, Any]) -> str:
    if row.get("returncode") == 0:
        return "ok"
    if row.get("returncode") == -1 and row.get("timeout_seconds"):
        return "timeout"
    return "failed"


def metric(row: dict[str, Any], key: str) -> float | None:
    lifecycle = row.get("lifecycle") or {}
    stage = row.get("stage") or {}
    result = row.get("result") or {}
    value = lifecycle.get(key) or stage.get(key) or result.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def prefill_ms(row: dict[str, Any]) -> float | None:
    return metric(row, "prefill_total_ms")


def total_ms(row: dict[str, Any]) -> float | None:
    return metric(row, "total_ms_avg")


def top_profile_op(profile: dict[str, Any] | None) -> dict[str, Any]:
    if not profile:
        return {}
    entries = profile.get("entries") or []
    return max(entries, key=lambda item: item.get("total_ms") or 0.0) if entries else {}


def compare_against_baseline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        row.get("context_tokens_requested"): row
        for row in rows
        if row.get("sweep_mode") == "baseline" and row.get("status") == "ok"
    }
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        context = row.get("context_tokens_requested")
        baseline = baselines.get(context)
        if baseline is None or row.get("status") != "ok":
            continue
        base_prefill = prefill_ms(baseline)
        row_prefill = prefill_ms(row)
        base_total = total_ms(baseline)
        row_total = total_ms(row)
        comparison: dict[str, Any] = {
            "context_tokens_requested": context,
            "mode": row.get("sweep_mode"),
            "baseline_mode": "baseline",
        }
        if base_prefill and row_prefill is not None:
            comparison["prefill_ms"] = row_prefill
            comparison["baseline_prefill_ms"] = base_prefill
            comparison["prefill_delta_ms"] = row_prefill - base_prefill
            comparison["prefill_ratio"] = row_prefill / base_prefill
        if base_total and row_total is not None:
            comparison["decode_total_ms_avg"] = row_total
            comparison["baseline_decode_total_ms_avg"] = base_total
            comparison["decode_total_ratio"] = row_total / base_total
        comparisons.append(comparison)
    return comparisons


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    reference_by_context: dict[str, list[int]] = {}
    mismatches: list[dict[str, Any]] = []
    for row in ok_rows:
        context = str(row.get("context_tokens_requested"))
        ids = row.get("generated_ids") or []
        if context not in reference_by_context:
            reference_by_context[context] = ids
        elif ids != reference_by_context[context]:
            mismatches.append(
                {
                    "context_tokens_requested": row.get("context_tokens_requested"),
                    "mode": row.get("sweep_mode"),
                    "reference_generated_ids": reference_by_context[context],
                    "generated_ids": ids,
                }
            )

    comparisons = compare_against_baseline(rows)
    best_by_context: dict[str, dict[str, Any]] = {}
    for row in ok_rows:
        value = prefill_ms(row)
        if value is None:
            continue
        context = str(row.get("context_tokens_requested"))
        best = best_by_context.get(context)
        if best is None or value < float(best["prefill_ms"]):
            best_by_context[context] = {
                "mode": row.get("sweep_mode"),
                "prefill_ms": value,
            }

    return {
        "rows": len(rows),
        "ok_rows": len(ok_rows),
        "status_counts": {
            status: sum(1 for row in rows if row.get("status") == status)
            for status in sorted({str(row.get("status")) for row in rows})
        },
        "reference_generated_ids_by_context": reference_by_context,
        "generated_ids_match": not mismatches,
        "generated_id_mismatches": mismatches,
        "comparisons": comparisons,
        "best_prefill_by_context": best_by_context,
    }


def render_float(value: Any, precision: int = 3) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def render_ratio(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}x"
    except (TypeError, ValueError):
        return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    comparisons = {
        (item.get("context_tokens_requested"), item.get("mode")): item
        for item in summary.get("comparisons") or []
    }
    lines = [
        "# Qwen3.6 Metal Batched-Prefill Variant Sweep",
        "",
        f"- contexts: `{','.join(str(item) for item in report['contexts'])}`",
        f"- modes: `{','.join(report['modes'])}`",
        f"- max_new_tokens: `{report['max_new_tokens']}`",
        f"- metal_profile: `{report['metal_profile']}`",
        f"- generated_ids_match: `{summary['generated_ids_match']}`",
        "",
        "| Context | Mode | Status | IDs | NIAH | Prefill ms | Prefill vs baseline | Total ms | Top Metal op | Top Metal ms | HAL ms | Wall s |",
        "|---:|:---|:---|:---|:---:|---:|---:|---:|:---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        context = row.get("context_tokens_requested")
        mode = row.get("sweep_mode")
        comparison = comparisons.get((context, mode), {})
        top_metal = top_profile_op(row.get("metal_profile"))
        hal_total = ((row.get("hal_profile") or {}).get("summary") or {}).get("total_ms")
        lines.append(
            "| {context} | {mode} | {status} | {ids} | {niah} | {prefill} | {ratio} | {total} | {top_op} | {top_ms} | {hal_ms} | {wall} |".format(
                context=context,
                mode=mode,
                status=row.get("status"),
                ids=",".join(str(item) for item in row.get("generated_ids") or []),
                niah=str(row.get("niah_contains_expected", False)).lower(),
                prefill=render_float(prefill_ms(row)),
                ratio=render_ratio(comparison.get("prefill_ratio")),
                total=render_float(total_ms(row)),
                top_op=top_metal.get("op") or "-",
                top_ms=render_float(top_metal.get("total_ms")),
                hal_ms=render_float(hal_total),
                wall=render_float(row.get("wall_seconds"), 1),
            )
        )
    lines.extend(
        [
            "",
            "Rows use the same deterministic NIAH prompt per context. `baseline` is the supported Metal per-token prefill path; all other modes enable the experimental batched-prefill prototype and one named env-gated variant.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_report(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    contexts: list[int],
    modes: list[str],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "model": bench.MODEL,
        "model_dir": str(args.model_dir),
        "backend": "metal",
        "contexts": contexts,
        "modes": modes,
        "max_new_tokens": args.max_new_tokens,
        "metal_profile": args.metal_profile,
        "batched_prefill_feasibility": args.batched_prefill_feasibility,
        "seed": args.seed,
        "summary": summarize(rows),
        "rows": rows,
    }


def run_row(
    args: argparse.Namespace,
    context: int,
    prompt: str,
    expected: str,
    mode: str,
) -> dict[str, Any]:
    run_args = args_for_mode(args, mode)
    if args.warmup:
        bench.run_one(run_args, context, prompt, expected, warmup=True)
    row = bench.run_one(run_args, context, prompt, expected, warmup=False)
    row["sweep_mode"] = mode
    row["status"] = status_for_row(row)
    row["prototype_enabled"] = prototype_enabled(mode)
    row["variant"] = variant_for_mode(mode)
    return row


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--contexts", default="512")
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--warmup-new-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--metal-profile", action="store_true")
    parser.add_argument(
        "--batched-prefill-feasibility",
        action="store_true",
        help="also emit grouped-MoE router/permutation occupancy rows",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("target/qwen36_metal_batched_prefill_variant_sweep.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("target/qwen36_metal_batched_prefill_variant_sweep.md"),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.model_dir = bench.resolve_model_dir(args.model_dir, os.environ)
    try:
        contexts = bench.BASE.parse_int_list(args.contexts)
        modes = parse_modes(args.modes)
    except ValueError as exc:
        print(f"[qwen36-batched-prefill-variant-sweep] error={exc}", file=sys.stderr)
        return 2
    if args.max_new_tokens <= 0:
        print("[qwen36-batched-prefill-variant-sweep] error=max_new_tokens_must_be_positive", file=sys.stderr)
        return 2
    if not args.binary.exists():
        raise FileNotFoundError(args.binary)
    if not args.model_dir.exists():
        raise FileNotFoundError(
            f"{args.model_dir}; set SUPERSONIC_TEST_MODEL_ROOT or pass --model-dir"
        )

    rows: list[dict[str, Any]] = []
    for context in contexts:
        prompt, expected = bench.BASE.make_niah_prompt(context, args.seed + context)
        for mode in modes:
            print(f"[bench] context={context} mode={mode}", flush=True)
            row = run_row(args, context, prompt, expected, mode)
            rows.append(row)
            lifecycle = row.get("lifecycle") or {}
            stage = row.get("stage") or {}
            print(
                "  status={} prefill_ms={} total_ms={} ids={}".format(
                    row.get("status"),
                    lifecycle.get("prefill_total_ms"),
                    stage.get("total_ms_avg"),
                    row.get("generated_ids"),
                ),
                flush=True,
            )

    report = build_report(rows, args, contexts, modes)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report))

    summary = report["summary"]
    print(
        "[qwen36-batched-prefill-variant-sweep] rows={} ok={} generated_ids_match={}".format(
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
