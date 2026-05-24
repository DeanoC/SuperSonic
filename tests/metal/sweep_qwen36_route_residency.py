#!/usr/bin/env python3
"""Sweep Qwen3.6 route locality to choose the next residency fork."""

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
SCHEMA = "qwen36-route-residency-sweep-v1"
DEFAULT_CAPACITIES = "2,4,8,16,32,64"

PROMPT_SETS: dict[str, list[tuple[str, str]]] = {
    "smoke": [
        (
            "profiling",
            "Inspect a local Apple Metal inference profile and identify the next optimization target from route locality, FFN time, and command-buffer waits.",
        ),
        (
            "coding",
            "Write a compact Rust helper that parses space-delimited key=value telemetry rows and returns a typed summary with numeric fields.",
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
        (
            "reasoning",
            "A sparse MoE decode path sees high oracle top-N coverage but poor adjacent-token LRU reuse. Explain which resident layout should be prototyped next and why.",
        ),
        (
            "summary",
            "Summarize why per-token expert slab rebuilding is a poor fit for Qwen3.6 Metal decode on a unified-memory Apple GPU.",
        ),
    ],
}


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


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value <= 0:
            raise ValueError(f"capacity must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("at least one capacity is required")
    return sorted(set(values))


def parse_metric_line(output: str, prefix: str) -> dict[str, Any]:
    lines = [line for line in output.splitlines() if line.startswith(prefix)]
    if not lines:
        return {}
    return {key: parse_number(value) for key, value in parse_key_values(lines[-1]).items()}


def parse_route_profile(output: str) -> dict[str, Any]:
    return parse_metric_line(output, "[qwen36-route-profile]")


def parse_route_cache_sims(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith("[qwen36-route-cache-sim]"):
            continue
        fields = parse_key_values(line)
        rows.append(
            {
                "scope": fields.get("scope"),
                "capacity": int(fields.get("capacity", "0")),
                "hits": int(fields.get("hits", "0")),
                "misses": int(fields.get("misses", "0")),
                "hit_rate": float(fields.get("hit_rate", "0")),
            }
        )
    return rows


def parse_route_topn_sims(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith("[qwen36-route-topn]"):
            continue
        fields = parse_key_values(line)
        rows.append(
            {
                "scope": fields.get("scope"),
                "capacity": int(fields.get("capacity", "0")),
                "covered": int(fields.get("covered", "0")),
                "total": int(fields.get("total", "0")),
                "coverage": float(fields.get("coverage", "0")),
            }
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


def parse_profile(output: str, summary_prefix: str, op_prefix: str) -> dict[str, Any] | None:
    summary_lines = [line for line in output.splitlines() if line.startswith(summary_prefix)]
    if not summary_lines:
        return None
    summary = {
        key: parse_number(value)
        for key, value in parse_key_values(summary_lines[-1]).items()
        if key not in {"op", "path"}
    }
    entries: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith(op_prefix):
            continue
        fields = parse_key_values(line)
        entry: dict[str, Any] = {
            "op": fields.get("op"),
            "calls": int(fields.get("calls", "0")),
            "mean_ms": float(fields.get("mean_ms", "0")),
            "total_ms": float(fields.get("total_ms", "0")),
            "max_ms": float(fields.get("max_ms", "0")),
        }
        if "path" in fields:
            entry["path"] = fields["path"]
        if "total_bytes" in fields:
            entry["total_bytes"] = int(fields["total_bytes"])
        entries.append(entry)
    return {"summary": summary, "entries": entries}


def aggregate_route_profiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = [row.get("route_profile") or {} for row in rows if row.get("status") == "ok"]
    calls = sum(int(profile.get("calls", 0)) for profile in profiles)
    assignments = sum(int(profile.get("assignments", 0)) for profile in profiles)
    adjacent_hits = sum(int(profile.get("adjacent_hits", 0)) for profile in profiles)
    adjacent_total = sum(int(profile.get("adjacent_total", 0)) for profile in profiles)
    dropped_calls = sum(int(profile.get("dropped_calls", 0)) for profile in profiles)
    unique_values = [
        int(profile.get("unique_layer_experts", 0))
        for profile in profiles
        if profile.get("unique_layer_experts") is not None
    ]
    return {
        "calls": calls,
        "assignments": assignments,
        "adjacent_hits": adjacent_hits,
        "adjacent_total": adjacent_total,
        "adjacent_hit_rate": adjacent_hits / adjacent_total if adjacent_total else 0.0,
        "dropped_calls": dropped_calls,
        "avg_unique_layer_experts": (
            sum(unique_values) / len(unique_values) if unique_values else 0.0
        ),
    }


def aggregate_cache_sims(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_capacity: dict[int, dict[str, int]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        for sim in row.get("cache_sims") or []:
            capacity = int(sim["capacity"])
            agg = by_capacity.setdefault(capacity, {"hits": 0, "misses": 0})
            agg["hits"] += int(sim.get("hits", 0))
            agg["misses"] += int(sim.get("misses", 0))
    out = []
    for capacity in sorted(by_capacity):
        hits = by_capacity[capacity]["hits"]
        misses = by_capacity[capacity]["misses"]
        total = hits + misses
        out.append(
            {
                "capacity": capacity,
                "hits": hits,
                "misses": misses,
                "hit_rate": hits / total if total else 0.0,
            }
        )
    return out


def aggregate_topn_sims(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_capacity: dict[int, dict[str, int]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        for sim in row.get("topn_sims") or []:
            capacity = int(sim["capacity"])
            agg = by_capacity.setdefault(capacity, {"covered": 0, "total": 0})
            agg["covered"] += int(sim.get("covered", 0))
            agg["total"] += int(sim.get("total", 0))
    out = []
    for capacity in sorted(by_capacity):
        covered = by_capacity[capacity]["covered"]
        total = by_capacity[capacity]["total"]
        out.append(
            {
                "capacity": capacity,
                "covered": covered,
                "total": total,
                "coverage": covered / total if total else 0.0,
            }
        )
    return out


def best_by(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get(key, 0.0)))


def build_decision_gate(
    rows: list[dict[str, Any]],
    aggregate_cache: list[dict[str, Any]],
    aggregate_topn: list[dict[str, Any]],
    min_lru_hit_rate: float = 0.50,
    min_static_topn_coverage: float = 0.80,
    max_candidate_capacity: int = 64,
) -> dict[str, Any]:
    failures: list[str] = []
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        failures.append("no_ok_route_profile_rows")
    if len(ok_rows) != len(rows):
        failures.append("not_all_prompts_measured")

    candidates: list[dict[str, Any]] = []
    for sim in aggregate_cache:
        capacity = int(sim["capacity"])
        if capacity > max_candidate_capacity:
            continue
        row_failures: list[str] = []
        if float(sim.get("hit_rate", 0.0)) < min_lru_hit_rate:
            row_failures.append("lru_hit_rate_below_threshold")
        candidates.append(
            {
                "kind": "lru_hotset",
                "capacity": capacity,
                "passed": not row_failures,
                "hit_rate": sim.get("hit_rate", 0.0),
                "failures": row_failures,
            }
        )

    for sim in aggregate_topn:
        capacity = int(sim["capacity"])
        if capacity > max_candidate_capacity:
            continue
        row_failures = []
        if float(sim.get("coverage", 0.0)) < min_static_topn_coverage:
            row_failures.append("static_topn_coverage_below_threshold")
        candidates.append(
            {
                "kind": "static_topn",
                "capacity": capacity,
                "passed": not row_failures,
                "coverage": sim.get("coverage", 0.0),
                "failures": row_failures,
            }
        )

    passed = [candidate for candidate in candidates if candidate["passed"]]
    if passed:
        lru_passes = [candidate for candidate in passed if candidate["kind"] == "lru_hotset"]
        static_passes = [candidate for candidate in passed if candidate["kind"] == "static_topn"]
        if lru_passes:
            recommendation = "prototype_larger_lru_resident_cache"
            best = max(lru_passes, key=lambda item: float(item.get("hit_rate", 0.0)))
        else:
            recommendation = "prototype_static_resident_table"
            best = max(static_passes, key=lambda item: float(item.get("coverage", 0.0)))
    elif ok_rows:
        recommendation = "prefer_fused_routed_int4"
        best = None
    else:
        recommendation = "run_route_residency_sweep"
        best = None

    return {
        "passed": bool(passed) and not failures,
        "recommendation": recommendation,
        "best_candidate": best,
        "failures": failures,
        "thresholds": {
            "min_lru_hit_rate": min_lru_hit_rate,
            "min_static_topn_coverage": min_static_topn_coverage,
            "max_candidate_capacity": max_candidate_capacity,
        },
        "candidates": candidates,
    }


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_summary(
    rows: list[dict[str, Any]],
    min_lru_hit_rate: float = 0.50,
    min_static_topn_coverage: float = 0.80,
    max_candidate_capacity: int = 64,
) -> dict[str, Any]:
    aggregate_cache = aggregate_cache_sims(rows)
    aggregate_topn = aggregate_topn_sims(rows)
    summary = {
        "prompt_count": len(rows),
        "status_counts": status_counts(rows),
        "measured_count": sum(1 for row in rows if row.get("status") == "ok"),
        "route_profile": aggregate_route_profiles(rows),
        "cache_sims": aggregate_cache,
        "topn_sims": aggregate_topn,
        "best_lru": best_by(aggregate_cache, "hit_rate"),
        "best_static_topn": best_by(aggregate_topn, "coverage"),
    }
    summary["decision_gate"] = build_decision_gate(
        rows,
        aggregate_cache,
        aggregate_topn,
        min_lru_hit_rate,
        min_static_topn_coverage,
        max_candidate_capacity,
    )
    return summary


def build_report(
    rows: list[dict[str, Any]],
    prompt_set: str,
    capacities: list[int],
    metal_profile: bool,
    min_lru_hit_rate: float = 0.50,
    min_static_topn_coverage: float = 0.80,
    max_candidate_capacity: int = 64,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "model": MODEL,
        "backend": "metal",
        "prompt_set": prompt_set,
        "capacities": capacities,
        "metal_profile": metal_profile,
        "summary": build_summary(
            rows,
            min_lru_hit_rate,
            min_static_topn_coverage,
            max_candidate_capacity,
        ),
        "rows": rows,
    }


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


def build_env_overrides(args: argparse.Namespace, capacities: list[int]) -> dict[str, str]:
    overrides = {
        "SUPERSONIC_BACKENDS": "metal",
        "SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL": "0",
        "SUPERSONIC_QWEN36_DENSE_PREFILL_TOKEN_LOOP": "1",
        "SUPERSONIC_QWEN36_ROUTE_PROFILE": "1",
        "SUPERSONIC_QWEN36_ROUTE_PROFILE_CAPACITIES": ",".join(
            str(capacity) for capacity in capacities
        ),
        "SUPERSONIC_QWEN36_ROUTE_PROFILE_MAX_CALLS": str(args.route_profile_max_calls),
    }
    if args.metal_profile:
        overrides["SUPERSONIC_METAL_PROFILE"] = "1"
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


def timeout_output(exc: subprocess.TimeoutExpired) -> str:
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
    return stdout + stderr


def row_from_output(
    output: str,
    prompt_id: str,
    prompt: str,
    status: str,
    returncode: int | None,
    wall_seconds: float,
    command: list[str],
    env_overrides: dict[str, str],
) -> dict[str, Any]:
    route_profile = parse_route_profile(output)
    if status == "ok" and not route_profile:
        status = "missing_route_profile"
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "status": status,
        "returncode": returncode,
        "wall_seconds": wall_seconds,
        "command": command,
        "env_overrides": env_overrides,
        "generated_ids": parse_generated_ids(output),
        "route_profile": route_profile,
        "cache_sims": parse_route_cache_sims(output),
        "topn_sims": parse_route_topn_sims(output),
        "metal_profile": parse_profile(output, "[metal-profile]", "[metal-profile-op]"),
        "hal_profile": parse_profile(output, "[hal-profile]", "[hal-profile-op]"),
        "output_tail": output_tail(output) if status != "ok" else "",
    }


def run_prompt(
    args: argparse.Namespace,
    prompt_id: str,
    prompt: str,
    capacities: list[int],
) -> dict[str, Any]:
    env = os.environ.copy()
    env_overrides = build_env_overrides(args, capacities)
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
        return row_from_output(
            output,
            prompt_id,
            prompt,
            status,
            proc.returncode,
            wall_seconds,
            command,
            env_overrides,
        )
    except subprocess.TimeoutExpired as exc:
        output = timeout_output(exc)
        return row_from_output(
            output,
            prompt_id,
            prompt,
            "timeout",
            None,
            time.monotonic() - started,
            command,
            env_overrides,
        )


def fmt_float(value: Any, precision: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "-"


def fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "-"


def top_profile_op(profile: dict[str, Any] | None) -> dict[str, Any]:
    if not profile:
        return {}
    entries = profile.get("entries") or []
    return max(entries, key=lambda item: item.get("total_ms") or 0.0) if entries else {}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    gate = summary.get("decision_gate") or {}
    lines = [
        "# Qwen3.6 Route Residency Sweep",
        "",
        f"- prompt_set: `{report['prompt_set']}`",
        f"- measured: `{summary['measured_count']}/{summary['prompt_count']}`",
        f"- aggregate_adjacent_hit_rate: `{fmt_pct(summary['route_profile'].get('adjacent_hit_rate'))}`",
        f"- decision_gate_passed: `{gate.get('passed', False)}`",
        f"- decision_gate_recommendation: `{gate.get('recommendation', '-')}`",
        "",
        "| Prompt | Status | IDs | Adjacent hit | Best LRU | Best top-N | Top Metal op | Top Metal ms | Wall s |",
        "|:---|:---|:---|---:|:---|:---|:---|---:|---:|",
    ]
    for row in report["rows"]:
        best_lru = best_by(row.get("cache_sims") or [], "hit_rate")
        best_topn = best_by(row.get("topn_sims") or [], "coverage")
        top_metal = top_profile_op(row.get("metal_profile"))
        lines.append(
            "| {prompt} | {status} | {ids} | {adjacent} | {lru} | {topn} | {top_op} | {top_ms} | {wall} |".format(
                prompt=row["prompt_id"],
                status=row["status"],
                ids=",".join(str(item) for item in row.get("generated_ids") or []),
                adjacent=fmt_pct((row.get("route_profile") or {}).get("adjacent_hit_rate")),
                lru=(
                    f"{best_lru['capacity']} @ {fmt_pct(best_lru.get('hit_rate'))}"
                    if best_lru
                    else "-"
                ),
                topn=(
                    f"{best_topn['capacity']} @ {fmt_pct(best_topn.get('coverage'))}"
                    if best_topn
                    else "-"
                ),
                top_op=top_metal.get("op") or "-",
                top_ms=fmt_float(top_metal.get("total_ms")),
                wall=fmt_float(row.get("wall_seconds"), 1),
            )
        )
    lines.extend(
        [
            "",
            "## Aggregate Capacity Rows",
            "",
            "| Capacity | LRU hit rate | LRU hits | LRU misses | Oracle top-N coverage | Covered | Total |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    cache_by_capacity = {row["capacity"]: row for row in summary.get("cache_sims") or []}
    topn_by_capacity = {row["capacity"]: row for row in summary.get("topn_sims") or []}
    for capacity in sorted(set(cache_by_capacity) | set(topn_by_capacity)):
        cache = cache_by_capacity.get(capacity, {})
        topn = topn_by_capacity.get(capacity, {})
        lines.append(
            "| {capacity} | {hit_rate} | {hits} | {misses} | {coverage} | {covered} | {total} |".format(
                capacity=capacity,
                hit_rate=fmt_pct(cache.get("hit_rate")),
                hits=cache.get("hits", "-"),
                misses=cache.get("misses", "-"),
                coverage=fmt_pct(topn.get("coverage")),
                covered=topn.get("covered", "-"),
                total=topn.get("total", "-"),
            )
        )
    candidates = gate.get("candidates") or []
    if candidates:
        lines.extend(
            [
                "",
                "## Decision Gate",
                "",
                "| Candidate | Capacity | Passed | Metric | Failures |",
                "|:---|---:|:---:|---:|:---|",
            ]
        )
        for candidate in candidates:
            metric = candidate.get("hit_rate")
            if metric is None:
                metric = candidate.get("coverage")
            lines.append(
                "| {kind} | {capacity} | {passed} | {metric} | {failures} |".format(
                    kind=candidate.get("kind"),
                    capacity=candidate.get("capacity"),
                    passed=str(candidate.get("passed", False)).lower(),
                    metric=fmt_pct(metric),
                    failures=", ".join(candidate.get("failures") or []) or "-",
                )
            )
    lines.extend(
        [
            "",
            "This harness is a decision aid, not a runtime promotion gate. LRU hit-rate points at hot-set residency; oracle top-N coverage points at static resident tables; if neither clears the configured thresholds, the next branch should favor a fused routed INT4 path over more slab residency.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--prompt-set", choices=sorted(PROMPT_SETS), default="smoke")
    parser.add_argument(
        "--prompt",
        action="append",
        help="custom prompt; repeat to run a custom prompt suite instead of --prompt-set",
    )
    parser.add_argument("--capacities", default=DEFAULT_CAPACITIES)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--route-profile-max-calls", type=int, default=65536)
    parser.add_argument("--metal-profile", action="store_true")
    parser.add_argument("--min-lru-hit-rate", type=float, default=0.50)
    parser.add_argument("--min-static-topn-coverage", type=float, default=0.80)
    parser.add_argument("--max-candidate-capacity", type=int, default=64)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("target/qwen36_route_residency_sweep.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("target/qwen36_route_residency_sweep.md"),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.model_dir = resolve_model_dir(args.model_dir, os.environ)
    capacities = parse_int_list(args.capacities)
    prompts = select_prompts(args)
    prompt_set = "custom" if args.prompt else args.prompt_set
    rows = [
        run_prompt(args, prompt_id, prompt, capacities)
        for prompt_id, prompt in prompts
    ]
    report = build_report(
        rows,
        prompt_set,
        capacities,
        args.metal_profile,
        args.min_lru_hit_rate,
        args.min_static_topn_coverage,
        args.max_candidate_capacity,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report))

    summary = report["summary"]
    gate = summary.get("decision_gate") or {}
    print(
        "[qwen36-route-residency-sweep] prompt_set={} measured={}/{} adjacent_hit_rate={:.6f} recommendation={} gate_passed={}".format(
            report["prompt_set"],
            summary["measured_count"],
            summary["prompt_count"],
            summary["route_profile"].get("adjacent_hit_rate", 0.0),
            gate.get("recommendation", "-"),
            str(gate.get("passed", False)).lower(),
        )
    )
    print(f"[wrote] {args.out_json}")
    print(f"[wrote] {args.out_md}")

    bad_statuses = {"failed", "timeout", "missing_route_profile"}
    if any(row["status"] in bad_statuses for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
