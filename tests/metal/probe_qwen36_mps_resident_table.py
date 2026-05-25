#!/usr/bin/env python3
"""Estimate Qwen3.6 Metal resident FP16 MPS expert-table viability.

This harness consumes the static top-N report from
``probe_qwen36_static_topn.py`` and combines it with the resident-shape MPS
expert pilot row. It does not enable a runtime path. Its job is to make the
next implementation decision explicit: full-hit-only resident tables, partial
resident-hit tables, or back to fused INT4.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MODEL = "qwen3.6-35b-a3b"
SCHEMA = "qwen36-mps-resident-table-probe-v2"
PILOT_PREFIX = "[qwen36-moe mps-expert-pilot]"

DEFAULT_LAYERS = 40
DEFAULT_HIDDEN = 2048
DEFAULT_MOE_INTERMEDIATE = 512
DEFAULT_TOP_K = 8
DEFAULT_BASELINE_FFN_MS = 98.761
DEFAULT_BASELINE_SOURCE = "2026-05-23 static-topn warm default ffn_ms_avg"
DEFAULT_GATE_MAX_RHS_GIB = 16.0
DEFAULT_GATE_MAX_RATIO = 0.90
DEFAULT_GATE_MIN_PARTIAL_COVERAGE = 0.50
DEFAULT_GATE_MIN_FULL_HIT_CALL_RATE = 0.50


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


def parse_mps_expert_pilot(output: str, source: str = "log") -> dict[str, Any] | None:
    lines = [line for line in output.splitlines() if line.startswith(PILOT_PREFIX)]
    if not lines:
        return None
    fields = parse_key_values(lines[-1])
    parsed: dict[str, Any] = {"source": source}
    for key, value in fields.items():
        parsed[key] = parse_number(value)
    return parsed


def manual_pilot_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.gate_up_ms is None and args.down_ms is None:
        return None
    if args.gate_up_ms is None or args.down_ms is None:
        raise ValueError("--gate-up-ms and --down-ms must be provided together")
    return {
        "source": "manual",
        "status": "ok",
        "hidden": args.hidden,
        "moe_intermediate": args.moe_intermediate,
        "top_k": args.top_k,
        "iterations": args.pilot_iters,
        "gate_up_ms": args.gate_up_ms,
        "down_ms": args.down_ms,
        "gate_up_tflops": args.gate_up_tflops or 0.0,
        "down_tflops": args.down_tflops or 0.0,
    }


def pilot_is_usable(pilot: dict[str, Any] | None) -> bool:
    if not pilot:
        return False
    return (
        pilot.get("status") == "ok"
        and float(pilot.get("gate_up_ms", 0.0)) > 0.0
        and float(pilot.get("down_ms", 0.0)) > 0.0
    )


def resolve_model_dir(raw_model_dir: Path | None, env: dict[str, str]) -> Path:
    if raw_model_dir is not None:
        return raw_model_dir
    if env.get("SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR"):
        return Path(env["SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR"])
    if env.get("SUPERSONIC_TEST_MODEL_ROOT"):
        return Path(env["SUPERSONIC_TEST_MODEL_ROOT"]) / MODEL
    return Path.home() / ".cache" / "supersonic-metal-models" / MODEL


def build_pilot_env(args: argparse.Namespace, base_env: dict[str, str]) -> dict[str, str]:
    env = base_env.copy()
    env["SUPERSONIC_BACKENDS"] = "metal"
    env["SUPERSONIC_METAL_QWEN36_MPS_EXPERT_PILOT"] = "1"
    env["SUPERSONIC_METAL_QWEN36_MPS_EXPERT_HIDDEN"] = str(args.hidden)
    env["SUPERSONIC_METAL_QWEN36_MPS_EXPERT_MOE_INTERMEDIATE"] = str(
        args.moe_intermediate
    )
    env["SUPERSONIC_METAL_QWEN36_MPS_EXPERT_TOP_K"] = str(args.top_k)
    env["SUPERSONIC_METAL_QWEN36_MPS_EXPERT_ITERS"] = str(args.pilot_iters)
    return env


def build_pilot_command(args: argparse.Namespace) -> list[str]:
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
        args.prompt,
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


def run_supersonic_mps_pilot(args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    env = build_pilot_env(args, os.environ)
    command = build_pilot_command(args)
    started = time.monotonic()
    proc = subprocess.run(
        command,
        text=True,
        capture_output=True,
        timeout=args.timeout,
        env=env,
    )
    elapsed = time.monotonic() - started
    output = proc.stdout + proc.stderr
    run_meta = {
        "command": command,
        "returncode": proc.returncode,
        "wall_seconds": elapsed,
    }
    pilot = parse_mps_expert_pilot(output, source="run")
    if proc.returncode != 0:
        run_meta["output_tail"] = output[-5000:]
    return pilot, run_meta


def load_static_report(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        report = json.load(handle)
    if "rows" not in report:
        raise ValueError(f"{path} does not look like a static top-N probe report")
    return report


def estimate_resident_mps_rhs_bytes(
    layers: int,
    capacity: int,
    hidden: int = DEFAULT_HIDDEN,
    moe_intermediate: int = DEFAULT_MOE_INTERMEDIATE,
) -> dict[str, Any]:
    gate_up_rhs_bytes_per_expert = hidden * (2 * moe_intermediate) * 2
    down_rhs_bytes_per_expert = moe_intermediate * hidden * 2
    bytes_per_expert = gate_up_rhs_bytes_per_expert + down_rhs_bytes_per_expert
    total = layers * capacity * bytes_per_expert
    return {
        "layers": layers,
        "capacity": capacity,
        "bytes_per_expert": bytes_per_expert,
        "total_bytes": total,
        "total_gib": total / (1024.0**3),
    }


def ratio(value: float | None, baseline: float) -> float | None:
    if value is None or baseline <= 0.0:
        return None
    return value / baseline


def coverage_for_gain(
    all_resident_mps_ms_per_token: float,
    baseline_ffn_ms: float,
    gain: float,
) -> float | None:
    target = baseline_ffn_ms * (1.0 - gain)
    denominator = baseline_ffn_ms - all_resident_mps_ms_per_token
    if denominator <= 0.0:
        return None
    required = (baseline_ffn_ms - target) / denominator
    return max(0.0, min(1.0, required))


def build_cost_model(
    evaluation: dict[str, Any],
    pilot: dict[str, Any] | None,
    layers: int,
    top_k: int,
    baseline_ffn_ms: float,
) -> dict[str, Any]:
    calls = int(evaluation.get("calls", 0))
    assignments = int(evaluation.get("assignments", 0))
    covered = int(evaluation.get("covered", 0))
    misses = int(evaluation.get("misses", max(0, assignments - covered)))
    full_hit_calls = int(evaluation.get("full_hit_calls", 0))
    fallback_calls = int(evaluation.get("fallback_calls", max(0, calls - full_hit_calls)))
    routed_tokens = calls / layers if layers > 0 else 0.0
    baseline_per_layer_call_ms = baseline_ffn_ms / layers if layers > 0 else 0.0

    base = {
        "status": "missing_pilot",
        "calls": calls,
        "routed_tokens_est": routed_tokens,
        "baseline_ffn_ms_per_token": baseline_ffn_ms,
        "baseline_per_layer_call_ms": baseline_per_layer_call_ms,
    }
    if not pilot_is_usable(pilot) or routed_tokens <= 0.0 or top_k <= 0:
        return base

    mps_gate_up_ms = float(pilot["gate_up_ms"])
    mps_down_ms = float(pilot["down_ms"])
    mps_call_ms = mps_gate_up_ms + mps_down_ms
    all_resident_total = calls * mps_call_ms
    all_resident_ms_per_token = all_resident_total / routed_tokens

    full_hit_only_total = (
        full_hit_calls * mps_call_ms
        + fallback_calls * baseline_per_layer_call_ms
    )
    full_hit_only_ms_per_token = full_hit_only_total / routed_tokens

    partial_total = (
        (covered / top_k) * mps_call_ms
        + (misses / top_k) * baseline_per_layer_call_ms
    )
    partial_ms_per_token = partial_total / routed_tokens

    return {
        **base,
        "status": "ok",
        "mps_gate_up_ms": mps_gate_up_ms,
        "mps_down_ms": mps_down_ms,
        "mps_call_ms": mps_call_ms,
        "all_resident_mps_ms_per_token": all_resident_ms_per_token,
        "full_hit_only_ms_per_token_est": full_hit_only_ms_per_token,
        "partial_hit_optimistic_ms_per_token_est": partial_ms_per_token,
        "all_resident_ratio": ratio(all_resident_ms_per_token, baseline_ffn_ms),
        "full_hit_only_ratio": ratio(full_hit_only_ms_per_token, baseline_ffn_ms),
        "partial_hit_optimistic_ratio": ratio(partial_ms_per_token, baseline_ffn_ms),
        "coverage_for_10pct_gain": coverage_for_gain(
            all_resident_ms_per_token,
            baseline_ffn_ms,
            0.10,
        ),
        "coverage_for_20pct_gain": coverage_for_gain(
            all_resident_ms_per_token,
            baseline_ffn_ms,
            0.20,
        ),
    }


def choose_recommendation(evaluation: dict[str, Any], cost: dict[str, Any]) -> str:
    if cost.get("status") != "ok":
        return "measure_mps_pilot"
    all_resident_ratio = float(cost.get("all_resident_ratio", 999.0))
    full_hit_ratio = float(cost.get("full_hit_only_ratio", 999.0))
    partial_ratio = float(cost.get("partial_hit_optimistic_ratio", 999.0))
    coverage = float(evaluation.get("coverage", 0.0))
    full_hit_rate = float(evaluation.get("full_hit_call_rate", 0.0))

    if all_resident_ratio >= 0.95:
        return "reject_mps_roofline"
    if full_hit_ratio <= 0.90:
        return "prototype_full_hit_resident_mps"
    if partial_ratio <= 0.90 and coverage >= 0.50 and full_hit_rate < 0.25:
        return "prototype_partial_hit_resident_mps"
    if coverage >= 0.50 and full_hit_rate < 0.25:
        return "partial_hit_required"
    return "raise_capacity_or_use_fused_int4"


def append_gate_failure(
    failures: list[str],
    condition: bool,
    failure: str,
) -> None:
    if not condition:
        failures.append(failure)


def build_candidate_gate(
    row: dict[str, Any],
    kind: str,
    max_rhs_gib: float,
    max_ratio: float,
    min_partial_coverage: float,
    min_full_hit_call_rate: float,
) -> dict[str, Any]:
    evaluation = row["evaluation_static_topn"]
    rhs = row["resident_mps_rhs"]
    cost = row["cost_model"]
    failures: list[str] = []
    status_ok = cost.get("status") == "ok"
    rhs_gib = float(rhs.get("total_gib", 0.0))
    coverage = float(evaluation.get("coverage", 0.0))
    full_hit_rate = float(evaluation.get("full_hit_call_rate", 0.0))
    append_gate_failure(failures, status_ok, "missing_usable_mps_pilot")
    append_gate_failure(failures, rhs_gib <= max_rhs_gib, "resident_rhs_too_large")

    if kind == "full_hit_only":
        estimate = cost.get("full_hit_only_ms_per_token_est")
        estimate_ratio = cost.get("full_hit_only_ratio")
        append_gate_failure(
            failures,
            full_hit_rate >= min_full_hit_call_rate,
            "full_hit_rate_below_threshold",
        )
    elif kind == "partial_hit_optimistic":
        estimate = cost.get("partial_hit_optimistic_ms_per_token_est")
        estimate_ratio = cost.get("partial_hit_optimistic_ratio")
        append_gate_failure(
            failures,
            coverage >= min_partial_coverage,
            "coverage_below_threshold",
        )
    else:
        raise ValueError(f"unknown gate candidate kind {kind!r}")

    append_gate_failure(failures, estimate_ratio is not None, "missing_estimate")
    if estimate_ratio is not None:
        append_gate_failure(
            failures,
            float(estimate_ratio) <= max_ratio,
            "estimate_not_fast_enough",
        )

    return {
        "kind": kind,
        "capacity": row["capacity"],
        "passed": not failures,
        "failures": failures,
        "estimated_ms_per_token": estimate,
        "estimated_ratio": estimate_ratio,
        "resident_mps_rhs_gib": rhs_gib,
        "coverage": coverage,
        "full_hit_call_rate": full_hit_rate,
        "requires_no_per_token_rebuild": kind == "partial_hit_optimistic",
    }


def build_viability_gate(
    rows: list[dict[str, Any]],
    max_rhs_gib: float = DEFAULT_GATE_MAX_RHS_GIB,
    max_ratio: float = DEFAULT_GATE_MAX_RATIO,
    min_partial_coverage: float = DEFAULT_GATE_MIN_PARTIAL_COVERAGE,
    min_full_hit_call_rate: float = DEFAULT_GATE_MIN_FULL_HIT_CALL_RATE,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        candidates.append(
            build_candidate_gate(
                row,
                "full_hit_only",
                max_rhs_gib,
                max_ratio,
                min_partial_coverage,
                min_full_hit_call_rate,
            )
        )
        candidates.append(
            build_candidate_gate(
                row,
                "partial_hit_optimistic",
                max_rhs_gib,
                max_ratio,
                min_partial_coverage,
                min_full_hit_call_rate,
            )
        )
    passed = [candidate for candidate in candidates if candidate["passed"]]
    full_hit_passes = [
        candidate for candidate in passed if candidate["kind"] == "full_hit_only"
    ]
    partial_passes = [
        candidate for candidate in passed if candidate["kind"] == "partial_hit_optimistic"
    ]
    if full_hit_passes:
        recommendation = "prototype_full_hit_resident_mps"
    elif partial_passes:
        recommendation = "prototype_partial_hit_resident_mps"
    elif any(row["cost_model"].get("status") != "ok" for row in rows):
        recommendation = "measure_mps_pilot"
    else:
        recommendation = "reject_resident_mps_for_now"
    best = min(
        passed,
        key=lambda candidate: candidate.get("estimated_ratio")
        if candidate.get("estimated_ratio") is not None
        else float("inf"),
        default=None,
    )
    return {
        "passed": bool(passed),
        "recommendation": recommendation,
        "best_candidate": best,
        "thresholds": {
            "max_rhs_gib": max_rhs_gib,
            "max_ratio": max_ratio,
            "min_partial_coverage": min_partial_coverage,
            "min_full_hit_call_rate": min_full_hit_call_rate,
        },
        "candidates": candidates,
    }


def build_report(
    static_report: dict[str, Any],
    pilot: dict[str, Any] | None,
    layers: int,
    hidden: int,
    moe_intermediate: int,
    top_k: int,
    baseline_ffn_ms: float,
    baseline_source: str,
    gate_max_rhs_gib: float = DEFAULT_GATE_MAX_RHS_GIB,
    gate_max_ratio: float = DEFAULT_GATE_MAX_RATIO,
    gate_min_partial_coverage: float = DEFAULT_GATE_MIN_PARTIAL_COVERAGE,
    gate_min_full_hit_call_rate: float = DEFAULT_GATE_MIN_FULL_HIT_CALL_RATE,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for static_row in static_report.get("rows", []):
        capacity = int(static_row["capacity"])
        evaluation = dict(static_row.get("evaluation_static_topn", {}))
        resident_rhs = static_row.get("resident_mps_rhs") or estimate_resident_mps_rhs_bytes(
            layers,
            capacity,
            hidden,
            moe_intermediate,
        )
        cost = build_cost_model(evaluation, pilot, layers, top_k, baseline_ffn_ms)
        rows.append(
            {
                "capacity": capacity,
                "evaluation_static_topn": evaluation,
                "resident_mps_rhs": resident_rhs,
                "cost_model": cost,
                "recommendation": choose_recommendation(evaluation, cost),
            }
        )

    usable_rows = [
        row
        for row in rows
        if row["cost_model"].get("status") == "ok"
    ]
    best_partial = None
    if usable_rows:
        best_partial = min(
            usable_rows,
            key=lambda row: row["cost_model"].get(
                "partial_hit_optimistic_ms_per_token_est",
                float("inf"),
            ),
        )

    viability_gate = build_viability_gate(
        rows,
        gate_max_rhs_gib,
        gate_max_ratio,
        gate_min_partial_coverage,
        gate_min_full_hit_call_rate,
    )
    return {
        "schema": SCHEMA,
        "model": MODEL,
        "source_static_schema": static_report.get("schema"),
        "layers": layers,
        "hidden": hidden,
        "moe_intermediate": moe_intermediate,
        "top_k": top_k,
        "baseline": {
            "ffn_ms_per_token": baseline_ffn_ms,
            "source": baseline_source,
        },
        "viability_thresholds": {
            "max_rhs_gib": gate_max_rhs_gib,
            "max_ratio": gate_max_ratio,
            "min_partial_coverage": gate_min_partial_coverage,
            "min_full_hit_call_rate": gate_min_full_hit_call_rate,
        },
        "pilot": pilot or {"source": "none", "status": "missing"},
        "rows": rows,
        "summary": {
            "best_partial_capacity": best_partial["capacity"] if best_partial else None,
            "best_partial_recommendation": best_partial["recommendation"]
            if best_partial
            else None,
            "best_partial_ms_per_token_est": best_partial["cost_model"].get(
                "partial_hit_optimistic_ms_per_token_est"
            )
            if best_partial
            else None,
            "viability_gate": viability_gate,
        },
    }


def fmt_ms(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "-"


def fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "-"


def render_markdown(report: dict[str, Any]) -> str:
    viability_gate = (report.get("summary") or {}).get("viability_gate") or {}
    lines = [
        "# Qwen3.6 MPS Resident Table Probe",
        "",
        f"- viability_gate_passed: `{viability_gate.get('passed', False)}`",
        f"- viability_gate_recommendation: `{viability_gate.get('recommendation', '-')}`",
        "",
        "| Capacity | Eval coverage | Full-hit calls | Fallback calls | MPS RHS GiB | All-resident MPS ms/tok | Full-hit-only est | Partial-hit optimistic est | Recommendation |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in report["rows"]:
        evaluation = row["evaluation_static_topn"]
        rhs = row["resident_mps_rhs"]
        cost = row["cost_model"]
        lines.append(
            "| {capacity} | {coverage} | {full_hit} | {fallback} | {rhs_gib:.2f} | {all_mps} | {full_est} | {partial_est} | {rec} |".format(
                capacity=row["capacity"],
                coverage=fmt_pct(evaluation.get("coverage")),
                full_hit=fmt_pct(evaluation.get("full_hit_call_rate")),
                fallback=evaluation.get("fallback_calls", 0),
                rhs_gib=float(rhs.get("total_gib", 0.0)),
                all_mps=fmt_ms(cost.get("all_resident_mps_ms_per_token")),
                full_est=fmt_ms(cost.get("full_hit_only_ms_per_token_est")),
                partial_est=fmt_ms(cost.get("partial_hit_optimistic_ms_per_token_est")),
                rec=row["recommendation"],
            )
        )
    lines.extend(
        [
            "",
            "The all-resident column is the FP16 MPS pilot cost if every routed expert for every layer were resident. The full-hit-only estimate falls back to the default FFN lane whenever a layer has any miss. The partial-hit estimate is deliberately optimistic: it assumes resident assignments and miss assignments can be split without an extra per-token FP16 table rebuild.",
        ]
    )
    candidates = viability_gate.get("candidates") or []
    if candidates:
        lines.extend(
            [
                "",
                "## Viability Gate",
                "",
                "| Candidate | Capacity | Passed | Ratio | RHS GiB | Coverage | Full-hit rate | Failures |",
                "|:---|---:|:---:|---:|---:|---:|---:|:---|",
            ]
        )
        for candidate in candidates:
            failures = candidate.get("failures") or []
            lines.append(
                "| {kind} | {capacity} | {passed} | {ratio} | {rhs} | {coverage} | {full_hit} | {failures} |".format(
                    kind=candidate.get("kind"),
                    capacity=candidate.get("capacity"),
                    passed=str(candidate.get("passed", False)).lower(),
                    ratio=fmt_pct(candidate.get("estimated_ratio")),
                    rhs=fmt_ms(candidate.get("resident_mps_rhs_gib")),
                    coverage=fmt_pct(candidate.get("coverage")),
                    full_hit=fmt_pct(candidate.get("full_hit_call_rate")),
                    failures=", ".join(str(item) for item in failures) or "-",
                )
            )
        lines.append("")
        lines.append(
            "The gate is nonfatal and estimates implementation viability, not runtime promotion. A partial-hit candidate is only a reason to prototype if it can be implemented without rebuilding FP16 RHS data per token; the runtime sweep remains the authority for promotion."
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--static-table-json",
        type=Path,
        default=Path("target/qwen36_static_topn_mps_probe.json"),
    )
    parser.add_argument("--pilot-log", type=Path)
    parser.add_argument("--run-pilot", action="store_true")
    parser.add_argument("--require-pilot", action="store_true")
    parser.add_argument("--gate-up-ms", type=float)
    parser.add_argument("--down-ms", type=float)
    parser.add_argument("--gate-up-tflops", type=float)
    parser.add_argument("--down-tflops", type=float)
    parser.add_argument("--pilot-iters", type=int, default=100)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--moe-intermediate", type=int, default=DEFAULT_MOE_INTERMEDIATE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--baseline-ffn-ms", type=float, default=DEFAULT_BASELINE_FFN_MS)
    parser.add_argument("--baseline-source", default=DEFAULT_BASELINE_SOURCE)
    parser.add_argument(
        "--gate-max-rhs-gib",
        type=float,
        default=DEFAULT_GATE_MAX_RHS_GIB,
        help="maximum resident FP16 RHS footprint for an MPS table candidate",
    )
    parser.add_argument(
        "--gate-max-ratio",
        type=float,
        default=DEFAULT_GATE_MAX_RATIO,
        help="maximum estimate/default FFN ratio for viability",
    )
    parser.add_argument(
        "--gate-min-partial-coverage",
        type=float,
        default=DEFAULT_GATE_MIN_PARTIAL_COVERAGE,
        help="minimum assignment coverage for a partial-hit candidate",
    )
    parser.add_argument(
        "--gate-min-full-hit-call-rate",
        type=float,
        default=DEFAULT_GATE_MIN_FULL_HIT_CALL_RATE,
        help="minimum full-hit layer-call rate for a full-hit-only candidate",
    )
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("target/qwen36_mps_resident_table_probe.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("target/qwen36_mps_resident_table_probe.md"),
    )
    return parser.parse_args(argv)


def select_pilot(args: argparse.Namespace, run_meta: dict[str, Any]) -> dict[str, Any] | None:
    manual = manual_pilot_from_args(args)
    if manual is not None:
        return manual
    if args.pilot_log:
        pilot = parse_mps_expert_pilot(args.pilot_log.read_text(), source=str(args.pilot_log))
        run_meta["pilot_log"] = str(args.pilot_log)
        return pilot
    if args.run_pilot:
        pilot, meta = run_supersonic_mps_pilot(args)
        run_meta["pilot_run"] = meta
        return pilot
    return None


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.model_dir = resolve_model_dir(args.model_dir, os.environ)
    try:
        static_report = load_static_report(args.static_table_json)
        run_meta: dict[str, Any] = {"static_table_json": str(args.static_table_json)}
        pilot = select_pilot(args, run_meta)
    except Exception as err:
        print(f"[qwen36-mps-resident-table-probe] error={err}", file=sys.stderr)
        return 2

    if args.require_pilot and not pilot_is_usable(pilot):
        print(
            "[qwen36-mps-resident-table-probe] error=missing_usable_mps_pilot",
            file=sys.stderr,
        )
        return 2

    report = build_report(
        static_report,
        pilot,
        args.layers,
        args.hidden,
        args.moe_intermediate,
        args.top_k,
        args.baseline_ffn_ms,
        args.baseline_source,
        args.gate_max_rhs_gib,
        args.gate_max_ratio,
        args.gate_min_partial_coverage,
        args.gate_min_full_hit_call_rate,
    )
    report["run_meta"] = run_meta

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report))

    best_capacity = report["summary"]["best_partial_capacity"]
    best_ms = report["summary"]["best_partial_ms_per_token_est"]
    best_rec = report["summary"]["best_partial_recommendation"]
    print(
        "[qwen36-mps-resident-table-probe] best_capacity={} best_partial_ms_per_token={} recommendation={}".format(
            best_capacity if best_capacity is not None else "none",
            f"{best_ms:.3f}" if best_ms is not None else "none",
            best_rec or "measure_mps_pilot",
        )
    )
    print(f"[wrote] {args.out_json}")
    print(f"[wrote] {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
