#!/usr/bin/env python3
"""Summarize Qwen3.6 Metal SOTA gate reports into one decision artifact."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "qwen36-sota-gate-summary-v1"


@dataclass(frozen=True)
class GateSpec:
    gate_id: str
    label: str
    default_path: Path
    expected_schema: str
    gate_keys: tuple[str, ...]
    kind: str


GATE_SPECS = (
    GateSpec(
        gate_id="batched_prefill_variants",
        label="Batched-prefill variants",
        default_path=Path("target/qwen36_metal_batched_prefill_variant_sweep.json"),
        expected_schema="qwen36-metal-batched-prefill-variant-sweep-v2",
        gate_keys=("promotion_gate",),
        kind="runtime_promotion",
    ),
    GateSpec(
        gate_id="static_topn_runtime",
        label="Static top-N runtime",
        default_path=Path("target/qwen36_static_topn_runtime_sweep.json"),
        expected_schema="qwen36-static-topn-runtime-sweep-v3",
        gate_keys=("promotion_gate",),
        kind="runtime_promotion",
    ),
    GateSpec(
        gate_id="mps_resident_table",
        label="MPS resident table",
        default_path=Path("target/qwen36_mps_resident_table_probe.json"),
        expected_schema="qwen36-mps-resident-table-probe-v2",
        gate_keys=("viability_gate",),
        kind="viability",
    ),
    GateSpec(
        gate_id="mtp_acceptance",
        label="MTP acceptance",
        default_path=Path("target/qwen36_mtp_acceptance_sweep.json"),
        expected_schema="qwen36-moe-mtp-acceptance-sweep-v2",
        gate_keys=("promotion_gate",),
        kind="runtime_promotion",
    ),
)


def load_json_report(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing report: {path}"
    try:
        with path.open() as handle:
            loaded = json.load(handle)
    except json.JSONDecodeError as exc:
        return None, f"malformed json: {exc.msg} at line {exc.lineno} column {exc.colno}"
    if not isinstance(loaded, dict):
        return None, "malformed report: top-level JSON value must be an object"
    return loaded, None


def unique_strings(values: list[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value)
        if text not in out:
            out.append(text)
    return out


def candidate_label(candidate: dict[str, Any]) -> str:
    if candidate.get("mode") is not None:
        return str(candidate["mode"])
    if candidate.get("kind") is not None and candidate.get("capacity") is not None:
        return f"{candidate['kind']}:{candidate['capacity']}"
    if candidate.get("kind") is not None:
        return str(candidate["kind"])
    if candidate.get("capacity") is not None:
        return f"capacity:{candidate['capacity']}"
    return "candidate"


def summarize_candidates(gate: dict[str, Any]) -> dict[str, Any]:
    candidates = gate.get("candidates") or []
    if not isinstance(candidates, list):
        candidates = []
    passed: list[str] = []
    failed: list[dict[str, Any]] = []
    failures: list[Any] = list(gate.get("failures") or [])
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        label = candidate_label(raw)
        if bool(raw.get("passed", False)):
            passed.append(label)
        else:
            candidate_failures = list(raw.get("failures") or [])
            failures.extend(candidate_failures)
            failed.append(
                {
                    "candidate": label,
                    "failures": unique_strings(candidate_failures),
                }
            )
    passed_modes = gate.get("passed_modes")
    if isinstance(passed_modes, list):
        passed.extend(str(item) for item in passed_modes)
    best = gate.get("best_candidate")
    if isinstance(best, dict) and best.get("passed") and not passed:
        passed.append(candidate_label(best))
    return {
        "passed_candidates": unique_strings(passed),
        "failed_candidates": failed,
        "failures": unique_strings(failures),
    }


def extract_gate(report: dict[str, Any], spec: GateSpec) -> tuple[str | None, dict[str, Any] | None]:
    summary = report.get("summary")
    if not isinstance(summary, dict):
        return None, None
    for key in spec.gate_keys:
        gate = summary.get(key)
        if isinstance(gate, dict):
            return key, gate
    return None, None


def build_gate_row(spec: GateSpec, path: Path) -> dict[str, Any]:
    base: dict[str, Any] = {
        "gate_id": spec.gate_id,
        "label": spec.label,
        "kind": spec.kind,
        "path": str(path),
        "expected_schema": spec.expected_schema,
        "schema": None,
        "status": "unknown",
        "gate_key": None,
        "passed": None,
        "recommendation": None,
        "passed_candidates": [],
        "failed_candidates": [],
        "failures": [],
    }
    report, error = load_json_report(path)
    if error is not None:
        status = "missing" if error.startswith("missing report") else "malformed"
        return {**base, "status": status, "error": error}

    schema = report.get("schema")
    base["schema"] = schema
    gate_key, gate = extract_gate(report, spec)
    if gate is None:
        return {
            **base,
            "status": "missing_gate",
            "error": f"missing gate in summary; expected one of {','.join(spec.gate_keys)}",
        }

    candidate_summary = summarize_candidates(gate)
    status = "ok"
    error = None
    if schema != spec.expected_schema:
        status = "schema_mismatch"
        error = f"expected schema {spec.expected_schema}, got {schema!r}"

    return {
        **base,
        "status": status,
        "error": error,
        "gate_key": gate_key,
        "passed": bool(gate.get("passed", False)),
        "recommendation": gate.get("recommendation"),
        "passed_candidates": candidate_summary["passed_candidates"],
        "failed_candidates": candidate_summary["failed_candidates"],
        "failures": candidate_summary["failures"],
        "thresholds": gate.get("thresholds") or {},
    }


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def choose_next_action(rows: list[dict[str, Any]]) -> dict[str, Any]:
    input_failures = [row for row in rows if row.get("status") != "ok"]
    if input_failures:
        return {
            "action": "run_or_refresh_gate_reports",
            "reason": "one or more gate reports are missing, malformed, stale, or missing a gate object",
            "blocked_reason": ",".join(str(row["gate_id"]) for row in input_failures),
        }

    runtime_passes = [
        row
        for row in rows
        if row.get("kind") == "runtime_promotion" and row.get("passed") is True
    ]
    if runtime_passes:
        first = runtime_passes[0]
        candidates = first.get("passed_candidates") or []
        suffix = f":{candidates[0]}" if candidates else ""
        return {
            "action": f"prepare_runtime_promotion:{first['gate_id']}{suffix}",
            "reason": "a runtime promotion gate passed under the current harness thresholds",
            "blocked_reason": None,
        }

    viability_passes = [
        row
        for row in rows
        if row.get("kind") == "viability" and row.get("passed") is True
    ]
    if viability_passes:
        first = viability_passes[0]
        recommendation = first.get("recommendation") or "prototype_viable_path"
        return {
            "action": recommendation,
            "reason": "an estimate gate passed, but it still needs a runtime implementation and sweep",
            "blocked_reason": None,
        }

    return {
        "action": "keep_default_lane_and_select_next_measured_bottleneck",
        "reason": "all gate reports loaded, but no promotion or viability gate passed",
        "blocked_reason": "no_gate_passed",
    }


def recommendation_for_row(row: dict[str, Any]) -> str:
    if row.get("status") == "missing":
        return "run_harness"
    if row.get("status") in {"malformed", "schema_mismatch", "missing_gate"}:
        return "refresh_harness_or_parser"
    if row.get("passed") is True:
        if row.get("kind") == "viability":
            return str(row.get("recommendation") or "prototype_viable_path")
        candidates = row.get("passed_candidates") or []
        return "prepare_runtime_promotion" + (f":{candidates[0]}" if candidates else "")
    return "keep_disabled"


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bad_rows = [row for row in rows if row.get("status") != "ok"]
    failed_gate_rows = [
        row for row in rows if row.get("status") == "ok" and row.get("passed") is False
    ]
    passed_gate_rows = [
        row for row in rows if row.get("status") == "ok" and row.get("passed") is True
    ]
    next_action = choose_next_action(rows)
    return {
        "gate_count": len(rows),
        "status_counts": status_counts(rows),
        "input_failure_count": len(bad_rows),
        "input_failures": [
            {
                "gate_id": row["gate_id"],
                "status": row.get("status"),
                "error": row.get("error"),
            }
            for row in bad_rows
        ],
        "passed_gate_ids": [str(row["gate_id"]) for row in passed_gate_rows],
        "failed_gate_ids": [str(row["gate_id"]) for row in failed_gate_rows],
        "all_inputs_ok": not bad_rows,
        "all_loaded_gates_passed": not bad_rows and not failed_gate_rows,
        "next_action": next_action,
    }


def build_report(paths: dict[str, Path]) -> dict[str, Any]:
    rows = [build_gate_row(spec, paths.get(spec.gate_id, spec.default_path)) for spec in GATE_SPECS]
    for row in rows:
        row["recommendation_action"] = recommendation_for_row(row)
    return {
        "schema": SCHEMA,
        "model": "qwen3.6-35b-a3b",
        "backend": "metal",
        "rows": rows,
        "summary": build_summary(rows),
    }


def fmt_bool(value: Any) -> str:
    if value is None:
        return "-"
    return str(bool(value)).lower()


def fmt_list(values: Any) -> str:
    if not values:
        return "-"
    return ", ".join(str(value) for value in values)


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    next_action = summary["next_action"]
    lines = [
        "# Qwen3.6 Metal SOTA Gate Summary",
        "",
        f"- inputs_ok: `{summary['all_inputs_ok']}`",
        f"- loaded_gates_passed: `{summary['all_loaded_gates_passed']}`",
        f"- passed_gates: `{fmt_list(summary['passed_gate_ids'])}`",
        f"- failed_gates: `{fmt_list(summary['failed_gate_ids'])}`",
        f"- next_action: `{next_action['action']}`",
        f"- blocked_reason: `{next_action.get('blocked_reason') or '-'}`",
        "",
        "| Gate | Status | Passed | Candidates | Recommendation | Failures | Path |",
        "|:---|:---|:---:|:---|:---|:---|:---|",
    ]
    for row in report["rows"]:
        failures = row.get("failures") or []
        if row.get("error"):
            failures = [row["error"], *failures]
        lines.append(
            "| {label} | {status} | {passed} | {candidates} | {rec} | {failures} | `{path}` |".format(
                label=row["label"],
                status=row["status"],
                passed=fmt_bool(row.get("passed")),
                candidates=fmt_list(row.get("passed_candidates")),
                rec=row.get("recommendation_action") or "-",
                failures=fmt_list(failures),
                path=row["path"],
            )
        )
    lines.extend(
        [
            "",
            "This summary is an aggregation layer only. Missing reports are preserved as rows by default; use `--require` when a local validation run must fail closed on stale or absent gate artifacts.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prefill-json",
        type=Path,
        default=GATE_SPECS[0].default_path,
        help="batched-prefill variant sweep JSON",
    )
    parser.add_argument(
        "--static-runtime-json",
        type=Path,
        default=GATE_SPECS[1].default_path,
        help="static top-N runtime sweep JSON",
    )
    parser.add_argument(
        "--mps-json",
        type=Path,
        default=GATE_SPECS[2].default_path,
        help="MPS resident table probe JSON",
    )
    parser.add_argument(
        "--mtp-json",
        type=Path,
        default=GATE_SPECS[3].default_path,
        help="MTP acceptance sweep JSON",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("target/qwen36_sota_gate_summary.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("target/qwen36_sota_gate_summary.md"),
    )
    parser.add_argument(
        "--require",
        action="store_true",
        help="exit non-zero if any configured report is missing, malformed, stale, or missing a gate object",
    )
    parser.add_argument(
        "--fail-on-gate-failure",
        action="store_true",
        help="exit non-zero if inputs are not OK or any loaded gate did not pass",
    )
    return parser.parse_args(argv)


def paths_from_args(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "batched_prefill_variants": args.prefill_json,
        "static_topn_runtime": args.static_runtime_json,
        "mps_resident_table": args.mps_json,
        "mtp_acceptance": args.mtp_json,
    }


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    report = build_report(paths_from_args(args))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report))

    summary = report["summary"]
    print(
        "[qwen36-sota-gate-summary] gates={} inputs_ok={} passed={} failed={} next_action={}".format(
            summary["gate_count"],
            str(summary["all_inputs_ok"]).lower(),
            len(summary["passed_gate_ids"]),
            len(summary["failed_gate_ids"]),
            summary["next_action"]["action"],
        )
    )
    print(f"[wrote] {args.out_json}")
    print(f"[wrote] {args.out_md}")

    if args.fail_on_gate_failure and not summary["all_loaded_gates_passed"]:
        return 1
    if args.require and not summary["all_inputs_ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
