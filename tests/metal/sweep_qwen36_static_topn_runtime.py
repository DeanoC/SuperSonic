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
SCHEMA = "qwen36-static-topn-runtime-sweep-v3"

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
    "mps-static-partial": "mps-static-partial",
    "static-mps-partial": "mps-static-partial",
}
DEFAULT_MODES = "default,static,static-hotset,mps-static-partial"


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


def row_number(row: dict[str, Any], section: str, key: str) -> float | None:
    values = row.get(section) or {}
    value = values.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def headline_ms_per_token(row: dict[str, Any]) -> float | None:
    return row_number(row, "result", "ms_per_step") or row_number(
        row, "stage_timings", "total_ms_avg"
    )


def chain_ms(row: dict[str, Any], key: str) -> float | None:
    return row_number(row, "chain_breakdown", key)


def lm_head_ms(row: dict[str, Any]) -> float | None:
    return row_number(row, "stage_timings", "lm_head_ms_avg")


def profile_op_total(profile: dict[str, Any] | None, needle: str) -> float | None:
    if not profile:
        return None
    total = 0.0
    matched = False
    for entry in profile.get("entries") or []:
        op = str(entry.get("op") or "").lower()
        if needle not in op:
            continue
        matched = True
        total += float(entry.get("total_ms") or 0.0)
    return total if matched else None


def command_buffer_wait_ms(row: dict[str, Any]) -> float | None:
    return profile_op_total(row.get("metal_profile"), "command_buffer_wait")


def ratio(
    row: dict[str, Any],
    baseline: dict[str, Any],
    getter: Any,
) -> tuple[float | None, float | None, float | None]:
    row_value = getter(row)
    baseline_value = getter(baseline)
    if baseline_value is None or baseline_value == 0 or row_value is None:
        return row_value, baseline_value, None
    return row_value, baseline_value, row_value / baseline_value


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
    if mode in {"static", "static-hotset", "mps-static-partial"}:
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_STATIC_TOPN"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_FFN_EXPERT_STATIC_TOPN_FILE"] = str(
            args.static_table_json
        )
        if args.static_capacity is not None:
            overrides["SUPERSONIC_METAL_QWEN36_FFN_EXPERT_STATIC_TOPN_CAPACITY"] = str(
                args.static_capacity
            )
    if mode == "mps-static-partial":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_MPS_STATIC_TOPN_PARTIAL"] = "1"
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
            "metal_profile": parse_profile(output, "[metal-profile]", "[metal-profile-op]"),
            "hal_profile": parse_profile(output, "[hal-profile]", "[hal-profile-op]"),
            "output_tail": output_tail(output),
        }
        return row
    except subprocess.TimeoutExpired as exc:
        output = timeout_output(exc)
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
            "metal_profile": parse_profile(output, "[metal-profile]", "[metal-profile-op]"),
            "hal_profile": parse_profile(output, "[hal-profile]", "[hal-profile-op]"),
            "output_tail": output_tail(output),
        }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    reference_by_prompt: dict[str, list[int]] = {}
    prompt_summaries: dict[str, dict[str, Any]] = {}
    mismatches: list[dict[str, Any]] = []
    for row in ok_rows:
        prompt_id = str(row.get("prompt_id", ""))
        generated_ids = row.get("generated_ids", [])
        if prompt_id not in reference_by_prompt:
            reference_by_prompt[prompt_id] = generated_ids
        elif generated_ids != reference_by_prompt[prompt_id]:
            mismatches.append(
                {
                    "prompt_id": row.get("prompt_id"),
                    "mode": row.get("mode"),
                    "reference_generated_ids": reference_by_prompt[prompt_id],
                    "generated_ids": generated_ids,
                }
            )
    for prompt_id, reference_ids in reference_by_prompt.items():
        prompt_rows = [row for row in ok_rows if str(row.get("prompt_id", "")) == prompt_id]
        prompt_mismatches = [
            row for row in prompt_rows if row.get("generated_ids", []) != reference_ids
        ]
        prompt_summaries[prompt_id] = {
            "ok_rows": len(prompt_rows),
            "reference_generated_ids": reference_ids,
            "generated_ids_match": not prompt_mismatches,
        }
    return {
        "rows": len(rows),
        "ok_rows": len(ok_rows),
        "status_counts": {
            status: sum(1 for row in rows if row.get("status") == status)
            for status in sorted({str(row.get("status")) for row in rows})
        },
        "reference_generated_ids_by_prompt": reference_by_prompt,
        "generated_ids_match": not mismatches,
        "generated_id_mismatches": mismatches,
        "prompt_summaries": prompt_summaries,
    }


def append_ratio_gate(
    failures: list[str],
    prompt_result: dict[str, Any],
    name: str,
    row: dict[str, Any],
    baseline: dict[str, Any],
    getter: Any,
    max_ratio: float,
    missing_failure: str,
    regression_failure: str,
) -> None:
    row_value, baseline_value, metric_ratio = ratio(row, baseline, getter)
    prompt_result[name] = row_value
    prompt_result[f"baseline_{name}"] = baseline_value
    prompt_result[f"{name}_ratio"] = metric_ratio
    if metric_ratio is None:
        failures.append(missing_failure)
    elif metric_ratio > max_ratio:
        failures.append(regression_failure)


def build_promotion_gate(
    rows: list[dict[str, Any]],
    modes: list[str],
    max_headline_ratio: float = 0.999,
    max_ffn_ratio: float = 0.999,
    max_component_regression_ratio: float = 1.10,
    max_command_buffer_wait_ratio: float = 1.05,
    require_profile: bool = True,
) -> dict[str, Any]:
    prompt_ids: list[str] = []
    for row in rows:
        prompt_id = str(row.get("prompt_id", ""))
        if prompt_id and prompt_id not in prompt_ids:
            prompt_ids.append(prompt_id)
    rows_by_key = {
        (str(row.get("prompt_id", "")), row.get("mode")): row
        for row in rows
    }
    candidate_modes = [mode for mode in modes if mode != "default"]
    candidates: list[dict[str, Any]] = []
    for mode in candidate_modes:
        failures: list[str] = []
        prompt_results: list[dict[str, Any]] = []
        for prompt_id in prompt_ids:
            prompt_result: dict[str, Any] = {"prompt_id": prompt_id}
            baseline = rows_by_key.get((prompt_id, "default"))
            row = rows_by_key.get((prompt_id, mode))
            if baseline is None or baseline.get("status") != "ok":
                failures.append(f"prompt_{prompt_id}:missing_ok_default")
                prompt_result["passed"] = False
                prompt_result["failures"] = ["missing_ok_default"]
                prompt_results.append(prompt_result)
                continue
            if row is None or row.get("status") != "ok":
                failures.append(f"prompt_{prompt_id}:missing_ok_candidate")
                prompt_result["passed"] = False
                prompt_result["failures"] = ["missing_ok_candidate"]
                prompt_results.append(prompt_result)
                continue

            prompt_failures: list[str] = []
            if (row.get("generated_ids") or []) != (baseline.get("generated_ids") or []):
                prompt_failures.append("generated_ids_mismatch")
            append_ratio_gate(
                prompt_failures,
                prompt_result,
                "headline_ms_per_token",
                row,
                baseline,
                headline_ms_per_token,
                max_headline_ratio,
                "missing_headline_ms_per_token",
                "headline_not_improved",
            )
            append_ratio_gate(
                prompt_failures,
                prompt_result,
                "ffn_ms_avg",
                row,
                baseline,
                lambda item: chain_ms(item, "ffn_ms_avg"),
                max_ffn_ratio,
                "missing_ffn_ms_avg",
                "ffn_not_improved",
            )
            for component in ("full_attn_ms_avg", "linear_attn_ms_avg"):
                append_ratio_gate(
                    prompt_failures,
                    prompt_result,
                    component,
                    row,
                    baseline,
                    lambda item, metric_name=component: chain_ms(item, metric_name),
                    max_component_regression_ratio,
                    f"missing_{component}",
                    f"{component}_regressed",
                )
            append_ratio_gate(
                prompt_failures,
                prompt_result,
                "lm_head_ms_avg",
                row,
                baseline,
                lm_head_ms,
                max_component_regression_ratio,
                "missing_lm_head_ms_avg",
                "lm_head_ms_avg_regressed",
            )
            if require_profile:
                append_ratio_gate(
                    prompt_failures,
                    prompt_result,
                    "command_buffer_wait_ms",
                    row,
                    baseline,
                    command_buffer_wait_ms,
                    max_command_buffer_wait_ratio,
                    "missing_command_buffer_wait_profile",
                    "command_buffer_wait_regressed",
                )
            else:
                row_value, baseline_value, metric_ratio = ratio(
                    row, baseline, command_buffer_wait_ms
                )
                prompt_result["command_buffer_wait_ms"] = row_value
                prompt_result["baseline_command_buffer_wait_ms"] = baseline_value
                prompt_result["command_buffer_wait_ms_ratio"] = metric_ratio

            prompt_result["passed"] = not prompt_failures
            prompt_result["failures"] = prompt_failures
            failures.extend(f"prompt_{prompt_id}:{failure}" for failure in prompt_failures)
            prompt_results.append(prompt_result)
        candidates.append(
            {
                "mode": mode,
                "passed": not failures,
                "failures": failures,
                "prompts": prompt_results,
            }
        )

    passed_modes = [candidate["mode"] for candidate in candidates if candidate["passed"]]
    return {
        "passed": bool(passed_modes),
        "passed_modes": passed_modes,
        "candidate_count": len(candidates),
        "thresholds": {
            "max_headline_ratio": max_headline_ratio,
            "max_ffn_ratio": max_ffn_ratio,
            "max_component_regression_ratio": max_component_regression_ratio,
            "max_command_buffer_wait_ratio": max_command_buffer_wait_ratio,
            "require_profile": require_profile,
        },
        "candidates": candidates,
    }


def summarize_with_gate(
    rows: list[dict[str, Any]],
    modes: list[str],
    max_headline_ratio: float = 0.999,
    max_ffn_ratio: float = 0.999,
    max_component_regression_ratio: float = 1.10,
    max_command_buffer_wait_ratio: float = 1.05,
    require_profile: bool = True,
) -> dict[str, Any]:
    summary = summarize(rows)
    summary["promotion_gate"] = build_promotion_gate(
        rows,
        modes,
        max_headline_ratio,
        max_ffn_ratio,
        max_component_regression_ratio,
        max_command_buffer_wait_ratio,
        require_profile,
    )
    return summary


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
        "promotion_thresholds": {
            "max_headline_ratio": args.promotion_max_headline_ratio,
            "max_ffn_ratio": args.promotion_max_ffn_ratio,
            "max_component_regression_ratio": args.promotion_max_component_regression_ratio,
            "max_command_buffer_wait_ratio": args.promotion_max_command_buffer_wait_ratio,
            "require_profile": args.promotion_require_profile,
        },
        "summary": summarize_with_gate(
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


def render_float(value: Any, precision: int = 3) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def top_profile_op(profile: dict[str, Any] | None) -> dict[str, Any]:
    if not profile:
        return {}
    entries = profile.get("entries") or []
    return max(entries, key=lambda item: item.get("total_ms") or 0.0) if entries else {}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    promotion_gate = summary.get("promotion_gate") or {}
    lines = [
        "# Qwen3.6 Static Top-N Runtime Sweep",
        "",
        f"- prompt_set: `{report['prompt_set']}`",
        f"- modes: `{','.join(report['modes'])}`",
        f"- max_new_tokens: `{report['max_new_tokens']}`",
        f"- generated_ids_match: `{summary['generated_ids_match']}`",
        f"- promotion_gate_passed: `{promotion_gate.get('passed', False)}`",
        f"- promotion_gate_passed_modes: `{','.join(promotion_gate.get('passed_modes') or []) or '-'}`",
        "",
        "| Prompt | Mode | Status | IDs | Decode ms | FFN ms avg | Exact hit rate | Slot hit rate | Copied GiB | Top Metal op | Top Metal ms | HAL ms | Wall s |",
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
            "| {prompt} | {mode} | {status} | {ids} | {decode} | {ffn} | {exact} | {slot} | {copied} | {top_op} | {top_ms} | {hal_ms} | {wall} |".format(
                prompt=row.get("prompt_id", ""),
                mode=row.get("mode", ""),
                status=row.get("status", ""),
                ids=",".join(str(item) for item in row.get("generated_ids", [])),
                decode=render_float(result.get("decode_ms")),
                ffn=render_float(chain.get("ffn_ms_avg")),
                exact=render_float(residency.get("exact_hit_rate"), 6),
                slot=render_float(residency.get("slot_hit_rate"), 6),
                copied=render_float(copied_gib, 3),
                top_op=top_metal.get("op") or "-",
                top_ms=render_float(top_metal.get("total_ms")),
                hal_ms=render_float(hal_summary.get("total_ms")),
                wall=render_float(row.get("wall_seconds"), 1),
            )
        )
    lines.extend(
        [
            "",
            "Rows are separate process runs. Static modes measure within-run warm reuse across generated tokens; the first token still pays resident-table allocation on full-hit layers.",
        ]
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
        lines.append("")
        lines.append(
            "The gate is nonfatal for this script. A resident FFN mode passes only when generated IDs match the default mode for every prompt, headline ms/token and FFN time improve, full-attention/linear-attention/lm-head stay inside the configured regression threshold, and command-buffer-wait attribution is present and not regressed when profile evidence is required."
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
    parser.add_argument("--out-json", type=Path, default=Path("target/qwen36_static_topn_runtime_sweep.json"))
    parser.add_argument("--out-md", type=Path, default=Path("target/qwen36_static_topn_runtime_sweep.md"))
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.model_dir = resolve_model_dir(args.model_dir, os.environ)
    modes = parse_modes(args.modes)
    if (
        any(mode in {"static", "static-hotset", "mps-static-partial"} for mode in modes)
        and not args.static_table_json.exists()
    ):
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
