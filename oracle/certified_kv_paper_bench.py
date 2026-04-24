#!/usr/bin/env python3
"""One-context paper-metric benchmark for Llama 3.1 certified KV.

This orchestrates the two existing benchmark lanes at a single context length:

* PG-19 teacher-forced perplexity, one chunk by default.
* arxiv_v1/RULER-style generated retrieval tasks, one sample per subtask by
  default.

The output is a single JSON file containing quality deltas, decode timing,
certificate telemetry, memory accounting, H2D bytes, and fallback/cache stats.
It is intentionally a QA benchmark, not a replacement for the full paper sweep.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_SUBTASKS = [
    "niah_single",
    "niah_multikey",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
]


def run_cmd(cmd: list[str], timeout: int) -> None:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    proc = subprocess.run(cmd, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def first_certified_telemetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        telemetry = row.get("certified_kv_telemetry")
        if isinstance(telemetry, dict) and telemetry:
            return telemetry
    return {}


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_pg19(payload: dict[str, Any]) -> dict[str, Any]:
    by_cfg = {row["config"]: row for row in payload.get("summary", [])}
    dense = by_cfg.get("dense", {})
    cert = by_cfg.get("certified", {})
    dense_ppl = dense.get("perplexity")
    cert_ppl = cert.get("perplexity")
    cert_rows = [r for r in payload.get("results", []) if r.get("config") == "certified"]
    return {
        "benchmark": "pg19",
        "context_length": cert.get("context_length") or dense.get("context_length"),
        "chunks": payload.get("num_chunks"),
        "dense_perplexity": dense_ppl,
        "certified_perplexity": cert_ppl,
        "delta_perplexity": (
            float(cert_ppl) - float(dense_ppl)
            if cert_ppl is not None and dense_ppl is not None
            else None
        ),
        "dense_ms_per_token": dense.get("ms_per_token"),
        "certified_ms_per_token": cert.get("ms_per_token"),
        "gate_failures": payload.get("gate_failures", []),
        "certified_kv_telemetry": first_certified_telemetry(cert_rows),
    }


def summarize_arxiv(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    subtasks: dict[str, dict[str, Any]] = {}
    for key, row in summary.items():
        subtask = row["subtask"]
        bucket = subtasks.setdefault(subtask, {"subtask": subtask})
        bucket[f"{row['config']}_mean_score"] = row.get("mean_score")
        bucket[f"{row['config']}_n"] = row.get("n")
        if row.get("config") == "certified":
            bucket["reference_score"] = row.get("reference_score")
            bucket["delta_vs_reference"] = row.get("delta_vs_reference")
    for bucket in subtasks.values():
        dense = bucket.get("dense_mean_score")
        cert = bucket.get("certified_mean_score")
        bucket["delta_vs_dense"] = (
            float(cert) - float(dense) if cert is not None and dense is not None else None
        )

    results = payload.get("results", [])
    cert_results = [r for r in results if r.get("config") == "certified"]
    dense_results = [r for r in results if r.get("config") == "dense"]
    cert_timings = [r.get("timing", {}) for r in cert_results]
    dense_timings = [r.get("timing", {}) for r in dense_results]
    cert_telemetry = [
        t.get("certified_kv_telemetry", {})
        for t in cert_timings
        if isinstance(t.get("certified_kv_telemetry"), dict)
        and t.get("certified_kv_telemetry")
    ]

    def timing_mean(rows: list[dict[str, Any]], key: str) -> float | None:
        return mean([float(r[key]) for r in rows if r.get(key) is not None])

    def telemetry_sum(key: str) -> float:
        return sum(float(t.get(key, 0.0) or 0.0) for t in cert_telemetry)

    def telemetry_max(key: str) -> float | None:
        values = [float(t[key]) for t in cert_telemetry if t.get(key) is not None]
        return max(values) if values else None

    memory = cert_telemetry[-1] if cert_telemetry else {}
    return {
        "benchmark": "arxiv_v1_ruler",
        "context_length": payload.get("contexts", [None])[0],
        "samples": payload.get("samples"),
        "subtasks": [subtasks[k] for k in sorted(subtasks)],
        "dense_mean_ms_per_step": timing_mean(dense_timings, "ms_per_step"),
        "certified_mean_ms_per_step": timing_mean(cert_timings, "ms_per_step"),
        "dense_mean_decode_ms": timing_mean(dense_timings, "decode_ms"),
        "certified_mean_decode_ms": timing_mean(cert_timings, "decode_ms"),
        "certified_kv_metrics": {
            "certificate_e_key_max": telemetry_max("certificate_e_key_max"),
            "certificate_e_val_max": telemetry_max("certificate_e_val_max"),
            "certificate_bound_total_max": telemetry_max("certificate_bound_total_max"),
            "certificate_delta_tail_max": telemetry_max("certificate_delta_tail_max"),
            "certificate_true_tail_bound_max": telemetry_max("certificate_true_tail_bound_max"),
            "score_consistency_violations": telemetry_sum("score_consistency_violations"),
            "dense_fallback_layers": telemetry_sum("dense_fallback_layers"),
            "ranking_fallback_decode_heads": telemetry_sum("ranking_fallback_decode_heads"),
            "value_escalation_decode_heads": telemetry_sum("value_escalation_decode_heads"),
            "promoted_key_h2d_bytes": telemetry_sum("promoted_key_h2d_bytes"),
            "promoted_value_h2d_bytes": telemetry_sum("promoted_value_h2d_bytes"),
            "ranking_prefix_cache_hits": telemetry_sum("ranking_prefix_cache_hits"),
            "ranking_prefix_cache_misses": telemetry_sum("ranking_prefix_cache_misses"),
            "ranking_prefix_h2d_bytes": telemetry_sum("ranking_prefix_h2d_bytes"),
            "ranking_prefix_reuse_bytes": telemetry_sum("ranking_prefix_reuse_bytes"),
            "memory_tier1_compressed_vram_bytes": memory.get("memory_tier1_compressed_vram_bytes"),
            "memory_tier2_host_pinned_bytes": memory.get("memory_tier2_host_pinned_bytes"),
            "memory_tail_bf16_vram_bytes": memory.get("memory_tail_bf16_vram_bytes"),
            "memory_ranking_prefix_scratch_vram_bytes": memory.get(
                "memory_ranking_prefix_scratch_vram_bytes"
            ),
            "memory_dense_bf16_kv_vram_bytes": memory.get("memory_dense_bf16_kv_vram_bytes"),
            "memory_total_certified_vram_bytes": memory.get(
                "memory_total_certified_vram_bytes"
            ),
        },
        "critical_failures": payload.get("critical_failures"),
        "run_errors": payload.get("run_errors", []),
        "gate_failures": payload.get("gate_failures", []),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-context certified KV paper metrics")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--pg19-chunks", type=int, default=1)
    parser.add_argument(
        "--pg19-eval-start-frac",
        type=float,
        default=0.9375,
        help=(
            "Dense prefix fraction for certified PG-19 scoring. The default "
            "keeps a full context but scores only the final 1/16 with certified "
            "decode so 4K QA completes in a practical time."
        ),
    )
    parser.add_argument("--ruler-samples", type=int, default=1)
    parser.add_argument("--subtasks", nargs="+", default=DEFAULT_SUBTASKS)
    parser.add_argument("--source-text", type=Path, default=None)
    parser.add_argument("--reference-dir", type=Path, default=Path("/workspace/DotCache/benchmarks/results/arxiv_v1_20260420"))
    parser.add_argument("--output", type=Path, default=Path("target/certified_kv_paper_4k.json"))
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--skip-pg19", action="store_true")
    parser.add_argument("--skip-ruler", action="store_true")
    parser.add_argument("--no-fail-gates", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pg19_out = args.output.with_suffix(".pg19.json")
    ruler_out = args.output.with_suffix(".arxiv_v1.json")

    if not args.skip_pg19:
        pg19_cmd = [
            sys.executable,
            "oracle/pg19_smoke.py",
            "--binary",
            str(args.binary),
            "--model-dir",
            str(args.model_dir),
            "--contexts",
            str(args.context),
            "--num-chunks",
            str(args.pg19_chunks),
            "--config",
            "both",
            "--output",
            str(pg19_out),
            "--timeout",
            str(args.timeout),
            "--reference-dir",
            str(args.reference_dir),
            "--reference-smoke",
            "--emit-stage-timings",
            "--eval-start-frac",
            str(args.pg19_eval_start_frac),
        ]
        if args.source_text is not None:
            pg19_cmd.extend(["--source-text", str(args.source_text)])
        if args.no_fail_gates:
            pg19_cmd.extend(["--max-certified-delta", "1000000"])
        try:
            run_cmd(pg19_cmd, args.timeout)
        except Exception as exc:
            if not args.no_fail_gates:
                raise
            pg19_out.write_text(json.dumps({"run_error": str(exc)}, indent=2) + "\n")

    if not args.skip_ruler:
        ruler_cmd = [
            sys.executable,
            "oracle/arxiv_v1_smoke.py",
            "--binary",
            str(args.binary),
            "--model-dir",
            str(args.model_dir),
            "--reference-dir",
            str(args.reference_dir),
            "--contexts",
            str(args.context),
            "--samples",
            str(args.ruler_samples),
            "--subtasks",
            *args.subtasks,
            "--config",
            "both",
            "--timeout",
            str(args.timeout),
            "--output",
            str(ruler_out),
            "--emit-stage-timings",
            "--continue-on-error",
        ]
        if not args.no_fail_gates:
            ruler_cmd.extend(["--fail-on-critical"])
        run_cmd(ruler_cmd, args.timeout)

    payload: dict[str, Any] = {
        "benchmark": "certified_kv_paper_one_context",
        "model": "llama3.1-8b",
        "context_length": args.context,
        "pg19": None,
        "arxiv_v1_ruler": None,
    }
    if pg19_out.exists():
        pg19_payload = json.loads(pg19_out.read_text())
        payload["pg19"] = (
            {"benchmark": "pg19", "run_error": pg19_payload["run_error"]}
            if "run_error" in pg19_payload
            else summarize_pg19(pg19_payload)
        )
    if ruler_out.exists():
        payload["arxiv_v1_ruler"] = summarize_arxiv(json.loads(ruler_out.read_text()))

    failures: list[str] = []
    for section in ("pg19", "arxiv_v1_ruler"):
        if payload.get(section):
            failures.extend(payload[section].get("gate_failures", []))
            if payload[section].get("run_error"):
                failures.append(f"{section}: {payload[section]['run_error']}")
            for row in payload[section].get("run_errors", []):
                failures.append(
                    f"{section}:{row.get('config')}:{row.get('subtask')}:{row.get('sample_idx')}: {row.get('error')}"
                )
    payload["gate_failures"] = failures
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"JSON -> {args.output}")
    return 1 if failures and not args.no_fail_gates else 0


if __name__ == "__main__":
    raise SystemExit(main())
