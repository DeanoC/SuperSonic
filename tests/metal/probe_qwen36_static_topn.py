#!/usr/bin/env python3
"""Probe static top-N resident expert tables for Qwen3.6 Metal.

This harness turns the route-profile dump into the first static resident-table
experiment: collect per-layer top-N experts from a calibration prompt, replay a
separate evaluation prompt with raw route calls, then report hit/fallback rates
plus native INT4 and FP16 MPS resident-table sizes.
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
SCHEMA = "qwen36-static-topn-mps-probe-v2"
DEFAULT_CAPACITIES = "2,4,8,16,32,64"
DEFAULT_LAYERS = 40
DEFAULT_HIDDEN = 2048
DEFAULT_MOE_INTERMEDIATE = 512
DEFAULT_GROUP_SIZE = 128

SMOKE_CALIBRATION_PROMPT = (
    "Inspect a local Apple Metal inference profile and identify the likely "
    "next bottleneck from route locality, command-buffer waits, and FFN time."
)
SMOKE_EVALUATION_PROMPT = (
    "Write a compact Rust helper that parses space-delimited key=value "
    "telemetry rows and returns a map of numeric fields."
)


def parse_key_values(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for part in line.split():
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        values[key] = raw.rstrip(",)")
    return values


def parse_int_list(raw: str | None) -> list[int]:
    if raw is None or raw == "":
        return []
    return [int(part) for part in raw.split(",") if part]


def parse_capacities(raw: str) -> list[int]:
    capacities = parse_int_list(raw)
    if not capacities or any(cap <= 0 for cap in capacities):
        raise ValueError(f"invalid capacities: {raw!r}")
    return capacities


def parse_topn_layers(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith("[qwen36-route-topn-layer]"):
            continue
        fields = parse_key_values(line)
        rows.append(
            {
                "capacity": int(fields["capacity"]),
                "layer": int(fields["layer"]),
                "experts": parse_int_list(fields.get("experts")),
                "counts": parse_int_list(fields.get("counts")),
                "covered": int(fields.get("covered", "0")),
                "total": int(fields.get("total", "0")),
                "coverage": float(fields.get("coverage", "0")),
            }
        )
    return rows


def parse_route_calls(output: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith("[qwen36-route-call]"):
            continue
        fields = parse_key_values(line)
        calls.append(
            {
                "call_idx": int(fields["call_idx"]),
                "layer": int(fields["layer"]),
                "experts": parse_int_list(fields.get("experts")),
            }
        )
    return calls


def build_static_tables(
    topn_rows: list[dict[str, Any]],
    capacities: list[int],
    layers: int,
) -> dict[int, dict[int, set[int]]]:
    tables: dict[int, dict[int, set[int]]] = {capacity: {} for capacity in capacities}
    for row in topn_rows:
        capacity = int(row["capacity"])
        if capacity not in tables:
            continue
        layer = int(row["layer"])
        if 0 <= layer < layers:
            tables[capacity][layer] = set(int(expert) for expert in row["experts"][:capacity])
    return tables


def export_static_tables(
    topn_rows: list[dict[str, Any]],
    capacities: list[int],
    layers: int,
) -> dict[str, dict[str, Any]]:
    exported: dict[str, dict[str, Any]] = {}
    for capacity in capacities:
        layer_rows = []
        rows = sorted(
            (row for row in topn_rows if int(row["capacity"]) == capacity),
            key=lambda row: int(row["layer"]),
        )
        for row in rows:
            layer = int(row["layer"])
            if not 0 <= layer < layers:
                continue
            experts = [int(expert) for expert in row["experts"][:capacity]]
            counts = [int(count) for count in row["counts"][: len(experts)]]
            layer_rows.append(
                {
                    "layer": layer,
                    "experts": experts,
                    "counts": counts,
                    "covered": int(row.get("covered", 0)),
                    "total": int(row.get("total", 0)),
                    "coverage": float(row.get("coverage", 0.0)),
                }
            )
        exported[str(capacity)] = {"layers": layer_rows}
    return exported


def evaluate_static_table(
    calls: list[dict[str, Any]],
    table: dict[int, set[int]],
) -> dict[str, Any]:
    assignments = 0
    covered = 0
    full_hit_calls = 0
    missing_layers: set[int] = set()
    touched_layers: set[int] = set()
    per_layer_total: dict[int, int] = {}
    per_layer_covered: dict[int, int] = {}

    for call in calls:
        layer = int(call["layer"])
        experts = [int(expert) for expert in call["experts"]]
        touched_layers.add(layer)
        resident = table.get(layer)
        if resident is None:
            missing_layers.add(layer)
            resident = set()
        local_total = len(experts)
        local_covered = sum(1 for expert in experts if expert in resident)
        assignments += local_total
        covered += local_covered
        per_layer_total[layer] = per_layer_total.get(layer, 0) + local_total
        per_layer_covered[layer] = per_layer_covered.get(layer, 0) + local_covered
        if local_total > 0 and local_covered == local_total:
            full_hit_calls += 1

    fallback_calls = len(calls) - full_hit_calls
    worst_layer = None
    for layer in sorted(touched_layers):
        total = per_layer_total.get(layer, 0)
        if total == 0:
            continue
        coverage = per_layer_covered.get(layer, 0) / total
        if worst_layer is None or coverage < worst_layer["coverage"]:
            worst_layer = {
                "layer": layer,
                "coverage": coverage,
                "covered": per_layer_covered.get(layer, 0),
                "total": total,
            }

    return {
        "calls": len(calls),
        "assignments": assignments,
        "covered": covered,
        "misses": assignments - covered,
        "coverage": covered / assignments if assignments else 0.0,
        "full_hit_calls": full_hit_calls,
        "fallback_calls": fallback_calls,
        "full_hit_call_rate": full_hit_calls / len(calls) if calls else 0.0,
        "missing_layers": sorted(missing_layers),
        "worst_layer": worst_layer,
    }


def ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def estimate_resident_native_int4_bytes(
    layers: int,
    capacity: int,
    hidden: int = DEFAULT_HIDDEN,
    moe_intermediate: int = DEFAULT_MOE_INTERMEDIATE,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> dict[str, Any]:
    gate_up_rows = 2 * moe_intermediate
    gate_up_weight_bytes_per_expert = gate_up_rows * ceil_div(hidden, 2)
    gate_up_sidecar_elems = ceil_div(gate_up_rows, group_size) * ceil_div(hidden, group_size)
    down_weight_bytes_per_expert = hidden * ceil_div(moe_intermediate, 2)
    down_sidecar_elems = ceil_div(hidden, group_size) * ceil_div(moe_intermediate, group_size)
    bytes_per_expert = (
        gate_up_weight_bytes_per_expert
        + down_weight_bytes_per_expert
        + 2 * gate_up_sidecar_elems * 2
        + 2 * down_sidecar_elems * 2
    )
    total = layers * capacity * bytes_per_expert
    return {
        "layers": layers,
        "capacity": capacity,
        "group_size": group_size,
        "bytes_per_expert": bytes_per_expert,
        "gate_up_weight_bytes_per_expert": gate_up_weight_bytes_per_expert,
        "down_weight_bytes_per_expert": down_weight_bytes_per_expert,
        "total_bytes": total,
        "total_gib": total / (1024.0**3),
    }


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
        "gate_up_rhs_bytes_per_expert": gate_up_rhs_bytes_per_expert,
        "down_rhs_bytes_per_expert": down_rhs_bytes_per_expert,
        "total_bytes": total,
        "total_gib": total / (1024.0**3),
    }


def summarize_calibration_rows(
    topn_rows: list[dict[str, Any]],
    capacities: list[int],
) -> dict[int, dict[str, Any]]:
    summary: dict[int, dict[str, Any]] = {}
    for capacity in capacities:
        rows = [row for row in topn_rows if row["capacity"] == capacity]
        covered = sum(int(row["covered"]) for row in rows)
        total = sum(int(row["total"]) for row in rows)
        summary[capacity] = {
            "covered": covered,
            "total": total,
            "coverage": covered / total if total else 0.0,
            "layers": len(rows),
        }
    return summary


def build_report(
    calibration_output: str,
    evaluation_output: str,
    capacities: list[int],
    layers: int,
    hidden: int = DEFAULT_HIDDEN,
    moe_intermediate: int = DEFAULT_MOE_INTERMEDIATE,
) -> dict[str, Any]:
    topn_rows = parse_topn_layers(calibration_output)
    route_calls = parse_route_calls(evaluation_output)
    tables = build_static_tables(topn_rows, capacities, layers)
    calibration_summary = summarize_calibration_rows(topn_rows, capacities)
    rows = []
    for capacity in capacities:
        evaluation = evaluate_static_table(route_calls, tables.get(capacity, {}))
        rows.append(
            {
                "capacity": capacity,
                "calibration_oracle_topn": calibration_summary.get(capacity, {}),
                "evaluation_static_topn": evaluation,
                "resident_native_int4": estimate_resident_native_int4_bytes(
                    layers,
                    capacity,
                    hidden,
                    moe_intermediate,
                ),
                "resident_mps_rhs": estimate_resident_mps_rhs_bytes(
                    layers,
                    capacity,
                    hidden,
                    moe_intermediate,
                ),
            }
        )
    return {
        "schema": SCHEMA,
        "model": MODEL,
        "layers": layers,
        "hidden": hidden,
        "moe_intermediate": moe_intermediate,
        "capacities": capacities,
        "calibration": {
            "topn_layer_rows": len(topn_rows),
            "has_all_requested_layers": all(
                calibration_summary.get(capacity, {}).get("layers") == layers
                for capacity in capacities
            ),
        },
        "evaluation": {
            "route_calls": len(route_calls),
            "assignments": sum(len(call["experts"]) for call in route_calls),
        },
        "static_tables": export_static_tables(topn_rows, capacities, layers),
        "rows": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Qwen3.6 Static Top-N MPS Probe",
        "",
        "| Capacity | Calib coverage | Eval coverage | Full-hit calls | Fallback calls | Worst layer | Native INT4 GiB | MPS RHS GiB |",
        "|---:|---:|---:|---:|---:|:---|---:|---:|",
    ]
    for row in report["rows"]:
        calib = row["calibration_oracle_topn"]
        eval_static = row["evaluation_static_topn"]
        native = row["resident_native_int4"]
        rhs = row["resident_mps_rhs"]
        worst_layer = eval_static.get("worst_layer")
        worst = (
            "L{layer} {coverage:.1%}".format(**worst_layer)
            if worst_layer
            else ""
        )
        lines.append(
            "| {cap} | {calib:.1%} | {eval_cov:.1%} | {full:.1%} | {fallback} | {worst} | {native_gib:.2f} | {gib:.2f} |".format(
                cap=row["capacity"],
                calib=calib.get("coverage", 0.0),
                eval_cov=eval_static.get("coverage", 0.0),
                full=eval_static.get("full_hit_call_rate", 0.0),
                fallback=eval_static.get("fallback_calls", 0),
                worst=worst,
                native_gib=native.get("total_gib", 0.0),
                gib=rhs.get("total_gib", 0.0),
            )
        )
    lines.extend(
        [
            "",
            "The native INT4 estimate is the resident-table footprint used by the opt-in packed Metal path. The resident RHS estimate includes FP16 gate/up and down matrices only; it excludes h_norm/output scratch and miss fallback cost.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def resolve_model_dir(raw_model_dir: Path | None, env: dict[str, str]) -> Path:
    if raw_model_dir is not None:
        return raw_model_dir
    if env.get("SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR"):
        return Path(env["SUPERSONIC_TEST_QWEN36_MOE_MODEL_DIR"])
    if env.get("SUPERSONIC_TEST_MODEL_ROOT"):
        return Path(env["SUPERSONIC_TEST_MODEL_ROOT"]) / MODEL
    return Path.home() / ".cache" / "supersonic-metal-models" / MODEL


def run_supersonic(
    args: argparse.Namespace,
    prompt: str,
    dump_topn_layers: bool,
    dump_calls: bool,
) -> tuple[str, list[str], float, int]:
    env = os.environ.copy()
    env["SUPERSONIC_BACKENDS"] = "metal"
    env["SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL"] = "0"
    env["SUPERSONIC_QWEN36_DENSE_PREFILL_TOKEN_LOOP"] = "1"
    env["SUPERSONIC_QWEN36_ROUTE_PROFILE"] = "1"
    env["SUPERSONIC_QWEN36_ROUTE_PROFILE_MAX_CALLS"] = str(args.route_profile_max_calls)
    if dump_topn_layers:
        env["SUPERSONIC_QWEN36_ROUTE_PROFILE_DUMP_TOPN_LAYERS"] = "1"
    if dump_calls:
        env["SUPERSONIC_QWEN36_ROUTE_PROFILE_DUMP_CALLS"] = "1"

    cmd = [
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
    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=args.timeout,
        env=env,
    )
    elapsed = time.monotonic() - start
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            "supersonic static top-N probe run failed with code {}:\n{}".format(
                proc.returncode,
                output[-5000:],
            )
        )
    return output, cmd, elapsed, proc.returncode


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--calibration-log", type=Path)
    parser.add_argument("--evaluation-log", type=Path)
    parser.add_argument("--calibration-prompt", default=SMOKE_CALIBRATION_PROMPT)
    parser.add_argument("--evaluation-prompt", default=SMOKE_EVALUATION_PROMPT)
    parser.add_argument("--capacities", default=DEFAULT_CAPACITIES)
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--moe-intermediate", type=int, default=DEFAULT_MOE_INTERMEDIATE)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--route-profile-max-calls", type=int, default=65536)
    parser.add_argument("--out-json", type=Path, default=Path("target/qwen36_static_topn_mps_probe.json"))
    parser.add_argument("--out-md", type=Path, default=Path("target/qwen36_static_topn_mps_probe.md"))
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="write a report even when route-profile rows are missing",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.model_dir = resolve_model_dir(args.model_dir, os.environ)
    capacities = parse_capacities(args.capacities)

    run_meta: dict[str, Any] = {}
    if args.calibration_log:
        calibration_output = args.calibration_log.read_text()
        run_meta["calibration_log"] = str(args.calibration_log)
    else:
        calibration_output, cmd, elapsed, returncode = run_supersonic(
            args,
            args.calibration_prompt,
            dump_topn_layers=True,
            dump_calls=False,
        )
        run_meta["calibration_run"] = {
            "command": cmd,
            "wall_seconds": elapsed,
            "returncode": returncode,
        }

    if args.evaluation_log:
        evaluation_output = args.evaluation_log.read_text()
        run_meta["evaluation_log"] = str(args.evaluation_log)
    else:
        evaluation_output, cmd, elapsed, returncode = run_supersonic(
            args,
            args.evaluation_prompt,
            dump_topn_layers=False,
            dump_calls=True,
        )
        run_meta["evaluation_run"] = {
            "command": cmd,
            "wall_seconds": elapsed,
            "returncode": returncode,
        }

    report = build_report(
        calibration_output,
        evaluation_output,
        capacities,
        args.layers,
        args.hidden,
        args.moe_intermediate,
    )
    report["run_meta"] = run_meta
    if not args.allow_empty and (
        report["calibration"]["topn_layer_rows"] == 0
        or report["evaluation"]["route_calls"] == 0
    ):
        print(
            "[qwen36-static-topn-mps-probe] error=missing_route_profile_rows "
            f"topn_layer_rows={report['calibration']['topn_layer_rows']} "
            f"route_calls={report['evaluation']['route_calls']}",
            file=sys.stderr,
        )
        return 2

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report))

    best = max(report["rows"], key=lambda row: row["evaluation_static_topn"]["coverage"])
    print(
        "[qwen36-static-topn-mps-probe] best_capacity={} eval_coverage={:.6} resident_rhs_gib={:.3f}".format(
            best["capacity"],
            best["evaluation_static_topn"]["coverage"],
            best["resident_mps_rhs"]["total_gib"],
        )
    )
    print(f"[wrote] {args.out_json}")
    print(f"[wrote] {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
