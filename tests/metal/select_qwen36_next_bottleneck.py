#!/usr/bin/env python3
"""Select the next Qwen3.6 Metal bottleneck after SOTA gates are exhausted."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "qwen36-next-bottleneck-v4"
MODEL = "qwen3.6-35b-a3b"
BACKEND = "metal"
FALLBACK_ACTION = "keep_default_lane_and_select_next_measured_bottleneck"
FFN_GATE_IDS = {
    "static_topn_runtime",
    "fused_routed_int4",
    "lru_resident_cache",
}
FFN_SUPERSEDED_IDS = {"mps_resident_table", "route_residency"}
LINEAR_GATE_IDS = {"linear_decode_variants"}
BUCKET_ACTIONS = {
    "ffn_ms_avg": "prototype_new_ffn_residency_or_compute_path",
    "linear_attn_ms_avg": "prototype_linear_attention_orchestration",
    "full_attn_ms_avg": "prototype_full_attention_orchestration",
    "lm_head_ms_avg": "prototype_lm_head_tail_path",
}


def load_report(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing report: {path}"
    try:
        loaded = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return None, f"malformed json: {exc.msg} at line {exc.lineno} column {exc.colno}"
    if not isinstance(loaded, dict):
        return None, "malformed report: top-level JSON value must be an object"
    return loaded, None


def run_cmd(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.decode("utf-8", errors="replace")


def parse_git_dirty_paths(status: str) -> list[str]:
    paths: list[str] = []
    for line in status.splitlines():
        if len(line) < 4:
            continue
        path = line[3:].strip()
        if not path:
            continue
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path.strip('"'))
    return paths


def git_fingerprint(repo_root: Path) -> dict[str, Any] | None:
    git_sha = run_cmd(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root)
    if git_sha is None:
        return None
    status = run_cmd(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
    )
    diff = run_cmd(["git", "diff", "--binary", "HEAD"], cwd=repo_root)
    if status is None or diff is None:
        return None
    digest = hashlib.sha256()
    digest.update(status.encode())
    digest.update(b"\0")
    digest.update(diff.encode())
    dirty_paths = parse_git_dirty_paths(status)
    return {
        "git_sha": git_sha.strip(),
        "git_dirty": bool(dirty_paths),
        "git_dirty_paths": dirty_paths,
        "git_diff_hash": digest.hexdigest(),
    }


def bench_perf_meta_path(perf_path: Path) -> Path:
    return perf_path.parent.parent / "meta.json"


def load_bench_perf_meta(perf_path: Path | None) -> dict[str, Any] | None:
    if perf_path is None:
        return None
    meta_path = bench_perf_meta_path(perf_path)
    if not meta_path.exists():
        return None
    try:
        loaded = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def fingerprint_matches(meta: dict[str, Any] | None, current: dict[str, Any] | None) -> bool:
    if not meta or not current:
        return False
    return (
        meta.get("git_sha") == current.get("git_sha")
        and meta.get("git_diff_hash") is not None
        and meta.get("git_diff_hash") == current.get("git_diff_hash")
    )


def bench_perf_candidate_score(
    path: Path,
    current_fingerprint: dict[str, Any] | None,
) -> tuple[int, float]:
    meta = load_bench_perf_meta(path)
    if fingerprint_matches(meta, current_fingerprint):
        score = 3
    elif (
        meta
        and current_fingerprint
        and meta.get("git_sha") == current_fingerprint.get("git_sha")
    ):
        score = 2
    elif meta:
        score = 1
    else:
        score = 0
    return (score, path.stat().st_mtime)


def latest_bench_perf_json(
    run_root: Path,
    current_fingerprint: dict[str, Any] | None = None,
) -> Path | None:
    candidates = [
        path
        for path in run_root.glob("*/perf/qwen3.6-35b-a3b_int4.json")
        if path.is_file()
    ]
    if not candidates:
        return None
    if current_fingerprint is not None:
        matching = [
            path
            for path in candidates
            if fingerprint_matches(load_bench_perf_meta(path), current_fingerprint)
        ]
        if not matching:
            return None
        return max(matching, key=lambda path: path.stat().st_mtime)
    return max(candidates, key=lambda path: bench_perf_candidate_score(path, current_fingerprint))


def report_schema(report: dict[str, Any] | None) -> Any:
    if not report:
        return None
    return report.get("schema") or report.get("schema_version")


def is_default_row(row: dict[str, Any]) -> bool:
    mode = row.get("mode")
    sweep_mode = row.get("sweep_mode")
    return mode == "default" or sweep_mode == "baseline"


def row_number(row: dict[str, Any], section: str, key: str) -> float | None:
    values = row.get(section) or {}
    if not isinstance(values, dict):
        return None
    value = values.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bucket_value(row: dict[str, Any], bucket: str) -> float | None:
    if bucket == "lm_head_ms_avg":
        return row_number(row, "stage_timings", bucket)
    return row_number(row, "chain_breakdown", bucket)


def row_label(row: dict[str, Any]) -> str:
    if row.get("prompt_id") is not None:
        return str(row["prompt_id"])
    if row.get("context_tokens_requested") is not None:
        return f"context_{row['context_tokens_requested']}"
    return "row"


def collect_bucket_samples(
    reports: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    samples: dict[str, list[dict[str, Any]]] = {
        "ffn_ms_avg": [],
        "linear_attn_ms_avg": [],
        "full_attn_ms_avg": [],
        "lm_head_ms_avg": [],
    }
    for report_name, report in reports.items():
        for row in report.get("rows") or []:
            if not isinstance(row, dict):
                continue
            if row.get("status") != "ok" or not is_default_row(row):
                continue
            for bucket in samples:
                value = bucket_value(row, bucket)
                if value is None:
                    continue
                samples[bucket].append(
                    {
                        "source": report_name,
                        "row": row_label(row),
                        "mode": row.get("mode") or row.get("sweep_mode"),
                        "value_ms": value,
                    }
                )
    return samples


def bench_perf_default_row(report: dict[str, Any]) -> dict[str, Any] | None:
    if report.get("status") != "ok":
        return None
    return {
        "prompt_id": "bench_perf",
        "mode": "default",
        "status": "ok",
        "stage_timings": report.get("stage_timings"),
        "chain_breakdown": report.get("chain_breakdown"),
        "lifecycle_timings": report.get("lifecycle_timings"),
        "profile_stage_timings": report.get("profile_stage_timings"),
        "profile_chain_breakdown": report.get("profile_chain_breakdown"),
        "profile_lifecycle_timings": report.get("profile_lifecycle_timings"),
        "metal_profile": report.get("metal_profile"),
        "hal_profile": report.get("hal_profile"),
    }


def bench_perf_runtime_report(report: dict[str, Any]) -> dict[str, Any] | None:
    row = bench_perf_default_row(report)
    if row is None:
        return None
    return {
        "schema": f"bench-perf-v{report.get('schema_version', 'unknown')}",
        "rows": [row],
    }


def summarize_bench_perf(
    report: dict[str, Any] | None,
    path: Path | None,
    meta: dict[str, Any] | None = None,
    current_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not report or report.get("status") != "ok":
        return None
    chain = report.get("chain_breakdown") or {}
    profile_chain = report.get("profile_chain_breakdown") or {}
    stage = report.get("stage_timings") or {}
    profile_stage = report.get("profile_stage_timings") or {}
    return {
        "path": str(path) if path is not None else None,
        "schema_version": report.get("schema_version"),
        "ms_per_step": report.get("ms_per_step"),
        "samples": report.get("samples"),
        "ffn_ms_avg": chain.get("ffn_ms_avg"),
        "linear_attn_ms_avg": chain.get("linear_attn_ms_avg"),
        "full_attn_ms_avg": chain.get("full_attn_ms_avg"),
        "lm_head_ms_avg": stage.get("lm_head_ms_avg"),
        "profile_ffn_ms_avg": profile_chain.get("ffn_ms_avg"),
        "profile_linear_attn_ms_avg": profile_chain.get("linear_attn_ms_avg"),
        "profile_full_attn_ms_avg": profile_chain.get("full_attn_ms_avg"),
        "profile_lm_head_ms_avg": profile_stage.get("lm_head_ms_avg"),
        "git_sha": (meta or {}).get("git_sha"),
        "git_dirty": (meta or {}).get("git_dirty"),
        "git_diff_hash": (meta or {}).get("git_diff_hash"),
        "fingerprint_match": fingerprint_matches(meta, current_fingerprint),
    }


def summarize_buckets(
    samples: dict[str, list[dict[str, Any]]],
    exhausted_buckets: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket, bucket_samples in samples.items():
        if not bucket_samples:
            continue
        values = [float(sample["value_ms"]) for sample in bucket_samples]
        rows.append(
            {
                "bucket": bucket,
                "median_ms": statistics.median(values),
                "mean_ms": sum(values) / len(values),
                "max_ms": max(values),
                "sample_count": len(values),
                "exhausted": bucket in exhausted_buckets,
                "samples": bucket_samples,
            }
        )
    rows.sort(key=lambda item: item["median_ms"], reverse=True)
    return rows


def top_profile_entries(
    reports: dict[str, dict[str, Any]],
    profile_key: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for report_name, report in reports.items():
        for row in report.get("rows") or []:
            if not isinstance(row, dict):
                continue
            if row.get("status") != "ok" or not is_default_row(row):
                continue
            profile = row.get(profile_key) or {}
            for entry in profile.get("entries") or []:
                if not isinstance(entry, dict):
                    continue
                entries.append(
                    {
                        "source": report_name,
                        "row": row_label(row),
                        "mode": row.get("mode") or row.get("sweep_mode"),
                        "op": entry.get("op"),
                        "path": entry.get("path"),
                        "total_ms": float(entry.get("total_ms") or 0.0),
                        "mean_ms": entry.get("mean_ms"),
                        "calls": entry.get("calls"),
                    }
                )
    entries.sort(key=lambda item: item["total_ms"], reverse=True)
    return entries[:limit]


def summarize_prefill(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    rows = [
        row
        for row in report.get("rows") or []
        if isinstance(row, dict) and row.get("status") == "ok"
    ]
    if not rows:
        return None
    prefill_rows = []
    for row in rows:
        lifecycle = row.get("lifecycle") or {}
        prefill_total_ms = lifecycle.get("prefill_total_ms")
        if prefill_total_ms is None:
            continue
        prefill_rows.append(
            {
                "mode": row.get("sweep_mode") or row.get("mode"),
                "context_tokens_requested": row.get("context_tokens_requested"),
                "prefill_total_ms": float(prefill_total_ms),
            }
        )
    if not prefill_rows:
        return None
    baseline = next((row for row in prefill_rows if row["mode"] == "baseline"), None)
    best = min(prefill_rows, key=lambda row: row["prefill_total_ms"])
    baseline_ms = baseline["prefill_total_ms"] if baseline else None
    return {
        "baseline_ms": baseline_ms,
        "best_mode": best["mode"],
        "best_ms": best["prefill_total_ms"],
        "best_vs_baseline": (
            None if not baseline_ms else best["prefill_total_ms"] / baseline_ms
        ),
        "promotion_gate_passed": bool(
            ((report.get("summary") or {}).get("promotion_gate") or {}).get("passed", False)
        ),
        "rows": prefill_rows,
    }


def ffn_gate_family_exhausted(sota_summary: dict[str, Any]) -> bool:
    failed = set(str(item) for item in sota_summary.get("failed_gate_ids") or [])
    superseded = set(str(item) for item in sota_summary.get("superseded_gate_ids") or [])
    return FFN_GATE_IDS.issubset(failed) and FFN_SUPERSEDED_IDS.issubset(superseded)


def linear_gate_family_exhausted(sota_summary: dict[str, Any]) -> bool:
    failed = set(str(item) for item in sota_summary.get("failed_gate_ids") or [])
    return LINEAR_GATE_IDS.issubset(failed)


def choose_recommendation(
    sota_report: dict[str, Any] | None,
    bucket_rows: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    if errors:
        return {
            "status": "input_failure",
            "action": "refresh_or_repair_input_reports",
            "reason": "one or more required selector inputs were missing or malformed",
        }
    assert sota_report is not None
    sota_summary = sota_report.get("summary") or {}
    next_action = (sota_summary.get("next_action") or {}).get("action")
    if next_action != FALLBACK_ACTION:
        return {
            "status": "defer_to_sota_gate_summary",
            "action": next_action,
            "reason": "SOTA gate summary still points at a specific gate action",
        }
    if not bucket_rows:
        return {
            "status": "input_failure",
            "action": "refresh_profiled_runtime_reports",
            "reason": "no default runtime rows with chain bucket timings were available",
        }

    dominant = bucket_rows[0]
    ffn_exhausted = ffn_gate_family_exhausted(sota_summary)
    actionable = [row for row in bucket_rows if not row["exhausted"]]
    target = actionable[0] if actionable else dominant
    if dominant["bucket"] == "ffn_ms_avg" and ffn_exhausted and actionable:
        return {
            "status": "selected",
            "action": BUCKET_ACTIONS[target["bucket"]],
            "target_bucket": target["bucket"],
            "dominant_bucket": dominant["bucket"],
            "reason": (
                "FFN remains the largest measured bucket, but current resident, "
                "static, fused, MPS, and LRU FFN forks have negative runtime gates; "
                "select the largest non-exhausted bucket next"
            ),
        }
    return {
        "status": "selected",
        "action": BUCKET_ACTIONS.get(target["bucket"], "inspect_measured_bucket"),
        "target_bucket": target["bucket"],
        "dominant_bucket": dominant["bucket"],
        "reason": "selected the largest measured default-lane bucket",
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    input_specs = {
        "sota_summary": args.sota_json,
        "static_topn_runtime": args.static_runtime_json,
        "fused_routed_int4": args.fused_json,
        "lru_resident_cache": args.lru_json,
        "linear_decode_variants": args.linear_json,
        "batched_prefill_variants": args.prefill_json,
    }
    loaded: dict[str, dict[str, Any]] = {}
    input_reports: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name, path in input_specs.items():
        report, error = load_report(path)
        input_reports[name] = {
            "path": str(path),
            "status": "ok" if error is None else "error",
            "error": error,
            "schema": report_schema(report),
        }
        if error is not None:
            errors.append(f"{name}: {error}")
        elif report is not None:
            loaded[name] = report

    current_fingerprint = git_fingerprint(getattr(args, "repo_root", Path(".")))
    bench_perf_path = getattr(args, "bench_perf_json", None)
    bench_perf_explicit = bench_perf_path is not None
    if bench_perf_path is None:
        bench_perf_path = latest_bench_perf_json(
            getattr(args, "bench_run_root", Path("target/bench-runs")),
            current_fingerprint,
        )
    bench_perf_report = None
    bench_perf_meta = None
    if bench_perf_path is None:
        input_reports["bench_perf"] = {
            "path": None,
            "status": "missing_optional",
            "error": None,
            "schema": None,
        }
    else:
        report, error = load_report(bench_perf_path)
        bench_perf_meta = load_bench_perf_meta(bench_perf_path)
        input_reports["bench_perf"] = {
            "path": str(bench_perf_path),
            "status": "ok" if error is None else "error",
            "error": error,
            "schema": report_schema(report),
            "git_sha": (bench_perf_meta or {}).get("git_sha"),
            "git_dirty": (bench_perf_meta or {}).get("git_dirty"),
            "git_diff_hash": (bench_perf_meta or {}).get("git_diff_hash"),
            "fingerprint_match": fingerprint_matches(bench_perf_meta, current_fingerprint),
        }
        if error is not None and bench_perf_explicit:
            errors.append(f"bench_perf: {error}")
        elif report is not None:
            bench_perf_report = report

    runtime_reports = {
        name: loaded[name]
        for name in (
            "static_topn_runtime",
            "fused_routed_int4",
            "lru_resident_cache",
            "linear_decode_variants",
        )
        if name in loaded
    }
    if bench_perf_report is not None:
        normalized = bench_perf_runtime_report(bench_perf_report)
        if normalized is not None:
            runtime_reports["bench_perf"] = normalized
    samples = collect_bucket_samples(runtime_reports)
    sota_report = loaded.get("sota_summary")
    exhausted_buckets: set[str] = set()
    if sota_report:
        sota_summary = sota_report.get("summary") or {}
        if ffn_gate_family_exhausted(sota_summary):
            exhausted_buckets.add("ffn_ms_avg")
        if linear_gate_family_exhausted(sota_summary):
            exhausted_buckets.add("linear_attn_ms_avg")
    bucket_rows = summarize_buckets(samples, exhausted_buckets)
    recommendation = choose_recommendation(sota_report, bucket_rows, errors)
    return {
        "schema": SCHEMA,
        "model": MODEL,
        "backend": BACKEND,
        "input_reports": input_reports,
        "errors": errors,
        "sota_next_action": (
            (((sota_report or {}).get("summary") or {}).get("next_action") or {}).get("action")
        ),
        "recommendation": recommendation,
        "decode_bucket_ranking": bucket_rows,
        "current_fingerprint": current_fingerprint,
        "bench_perf": summarize_bench_perf(
            bench_perf_report,
            bench_perf_path,
            bench_perf_meta,
            current_fingerprint,
        ),
        "prefill": summarize_prefill(loaded.get("batched_prefill_variants")),
        "top_metal_profile_ops": top_profile_entries(runtime_reports, "metal_profile"),
        "top_hal_profile_ops": top_profile_entries(runtime_reports, "hal_profile"),
    }


def fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def fmt_list(values: Any) -> str:
    if not values:
        return "-"
    return ", ".join(str(value) for value in values)


def render_markdown(report: dict[str, Any]) -> str:
    rec = report["recommendation"]
    prefill = report.get("prefill") or {}
    lines = [
        "# Qwen3.6 Metal Next Bottleneck",
        "",
        f"- status: `{rec['status']}`",
        f"- action: `{rec['action']}`",
        f"- target_bucket: `{rec.get('target_bucket') or '-'}`",
        f"- dominant_bucket: `{rec.get('dominant_bucket') or '-'}`",
        f"- sota_next_action: `{report.get('sota_next_action') or '-'}`",
        f"- reason: {rec['reason']}",
        "",
        "## Decode Buckets",
        "",
        "| Bucket | Median ms | Mean ms | Max ms | Samples | Exhausted |",
        "|:---|---:|---:|---:|---:|:---:|",
    ]
    for row in report["decode_bucket_ranking"]:
        lines.append(
            "| {bucket} | {median} | {mean} | {max_ms} | {samples} | {exhausted} |".format(
                bucket=row["bucket"],
                median=fmt_float(row.get("median_ms")),
                mean=fmt_float(row.get("mean_ms")),
                max_ms=fmt_float(row.get("max_ms")),
                samples=row.get("sample_count"),
                exhausted=str(bool(row.get("exhausted"))).lower(),
            )
        )
    if prefill:
        lines.extend(
            [
                "",
                "## Prefill",
                "",
                f"- baseline_ms: `{fmt_float(prefill.get('baseline_ms'))}`",
                f"- best_mode: `{prefill.get('best_mode') or '-'}`",
                f"- best_ms: `{fmt_float(prefill.get('best_ms'))}`",
                f"- best_vs_baseline: `{fmt_float(prefill.get('best_vs_baseline'))}`",
                f"- promotion_gate_passed: `{str(bool(prefill.get('promotion_gate_passed'))).lower()}`",
            ]
        )
    bench = report.get("bench_perf") or {}
    if bench:
        lines.extend(
            [
                "",
                "## Bench Perf",
                "",
                f"- path: `{bench.get('path') or '-'}`",
                f"- schema_version: `{bench.get('schema_version') or '-'}`",
                f"- git_sha: `{bench.get('git_sha') or '-'}`",
                f"- git_dirty: `{str(bench.get('git_dirty')).lower() if bench.get('git_dirty') is not None else '-'}`",
                f"- fingerprint_match: `{str(bool(bench.get('fingerprint_match'))).lower()}`",
                f"- ms_per_step: `{fmt_float(bench.get('ms_per_step'))}`",
                f"- linear_attn_ms_avg: `{fmt_float(bench.get('linear_attn_ms_avg'))}`",
                f"- profile_linear_attn_ms_avg: `{fmt_float(bench.get('profile_linear_attn_ms_avg'))}`",
                f"- ffn_ms_avg: `{fmt_float(bench.get('ffn_ms_avg'))}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Top Metal Ops",
            "",
            "| Source | Row | Op | Path | Total ms | Calls |",
            "|:---|:---|:---|:---|---:|---:|",
        ]
    )
    for entry in report["top_metal_profile_ops"][:5]:
        lines.append(
            "| {source} | {row} | {op} | {path} | {total} | {calls} |".format(
                source=entry.get("source"),
                row=entry.get("row"),
                op=entry.get("op") or "-",
                path=entry.get("path") or "-",
                total=fmt_float(entry.get("total_ms")),
                calls=entry.get("calls") or "-",
            )
        )
    if report.get("errors"):
        lines.extend(["", "## Errors", "", fmt_list(report["errors"])])
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sota-json",
        type=Path,
        default=Path("target/qwen36_sota_gate_summary.json"),
    )
    parser.add_argument(
        "--static-runtime-json",
        type=Path,
        default=Path("target/qwen36_static_topn_runtime_sweep.json"),
    )
    parser.add_argument(
        "--fused-json",
        type=Path,
        default=Path("target/qwen36_fused_routed_int4_sweep.json"),
    )
    parser.add_argument(
        "--lru-json",
        type=Path,
        default=Path("target/qwen36_lru_resident_cache_sweep.json"),
    )
    parser.add_argument(
        "--prefill-json",
        type=Path,
        default=Path("target/qwen36_metal_batched_prefill_variant_sweep.json"),
    )
    parser.add_argument(
        "--linear-json",
        type=Path,
        default=Path("target/qwen36_linear_decode_sweep.json"),
    )
    parser.add_argument(
        "--bench-perf-json",
        type=Path,
        help="optional bench-perf JSON; defaults to the newest target/bench-runs/*/perf/qwen3.6-35b-a3b_int4.json",
    )
    parser.add_argument(
        "--bench-run-root",
        type=Path,
        default=Path("target/bench-runs"),
        help="run root used to auto-discover the latest bench-perf JSON",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="repo root used to fingerprint the current checkout for bench-perf auto-discovery",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("target/qwen36_next_bottleneck.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("target/qwen36_next_bottleneck.md"),
    )
    parser.add_argument(
        "--require-selected",
        action="store_true",
        help="exit non-zero unless the selector reaches a concrete selected action",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    report = build_report(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report))

    rec = report["recommendation"]
    print(
        "[qwen36-next-bottleneck] status={} action={} target={} dominant={}".format(
            rec["status"],
            rec["action"],
            rec.get("target_bucket") or "-",
            rec.get("dominant_bucket") or "-",
        )
    )
    print(f"[wrote] {args.out_json}")
    print(f"[wrote] {args.out_md}")
    if args.require_selected and rec["status"] != "selected":
        return 1
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
