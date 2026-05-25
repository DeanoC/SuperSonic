#!/usr/bin/env python3
"""Sweep Qwen3.6 Metal full-attention decode handoff variants."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sweep_qwen36_linear_decode as base


MODEL = base.MODEL
SCHEMA = "qwen36-full-decode-sweep-v1"

MODE_ALIASES: dict[str, str] = {
    "baseline": "default",
    "default": "default",
    "direct-off": "default",
    "no-direct": "default",
    "old-handoff": "default",
    "direct": "direct",
    "direct-on": "direct",
    "enable-direct": "direct",
}
DEFAULT_MODES = "default,direct"


def parse_modes(raw: str) -> list[str]:
    modes: list[str] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        mode = MODE_ALIASES.get(stripped)
        if mode is None:
            raise ValueError(f"unknown mode {stripped!r}; expected one of {sorted(MODE_ALIASES)}")
        if mode not in modes:
            modes.append(mode)
    if not modes:
        raise ValueError("at least one mode is required")
    return modes


def build_env_overrides(args: argparse.Namespace, mode: str) -> dict[str, str]:
    overrides = {"SUPERSONIC_BACKENDS": "metal"}
    if args.metal_profile:
        overrides["SUPERSONIC_METAL_PROFILE"] = "1"
    if mode == "direct":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FULL_ATTN_DECODE_DIRECT"] = "1"
    return overrides


def full_attn_host_ms(row: dict[str, Any]) -> float | None:
    return base.profile_op_total_by_prefix(row.get("metal_profile"), "qwen36_full_attn", "host")


def command_buffer_wait_ms(row: dict[str, Any]) -> float | None:
    return base.profile_op_total(row.get("metal_profile"), "command_buffer_wait")


def copy_d2d_ms(row: dict[str, Any]) -> float | None:
    return base.profile_op_total(row.get("hal_profile"), "copy_d2d")


def run_row(args: argparse.Namespace, prompt_id: str, prompt: str, mode: str) -> dict[str, Any]:
    env = os.environ.copy()
    env_overrides = build_env_overrides(args, mode)
    env.update(env_overrides)
    command = base.build_command(args, prompt)
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
            "generated_ids": base.parse_generated_ids(output),
            "result": base.parse_result(output),
            "stage_timings": base.parse_stage_timings(output),
            "chain_breakdown": base.parse_chain_breakdown(output),
            "lifecycle_timings": base.parse_lifecycle_timings(output),
            "metal_profile": base.parse_profile(output, "[metal-profile]", "[metal-profile-op]"),
            "hal_profile": base.parse_profile(output, "[hal-profile]", "[hal-profile-op]"),
            "output_tail": base.output_tail(output),
        }
        row["full_attn_host_ms"] = full_attn_host_ms(row)
        row["copy_d2d_ms"] = copy_d2d_ms(row)
        row["command_buffer_wait_ms"] = command_buffer_wait_ms(row)
        return row
    except subprocess.TimeoutExpired as exc:
        output = base.timeout_output(exc)
        row = {
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
            "metal_profile": base.parse_profile(output, "[metal-profile]", "[metal-profile-op]"),
            "hal_profile": base.parse_profile(output, "[hal-profile]", "[hal-profile-op]"),
            "output_tail": base.output_tail(output),
        }
        row["full_attn_host_ms"] = full_attn_host_ms(row)
        row["copy_d2d_ms"] = copy_d2d_ms(row)
        row["command_buffer_wait_ms"] = command_buffer_wait_ms(row)
        return row


def build_promotion_gate(
    rows: list[dict[str, Any]],
    modes: list[str],
    max_headline_ratio: float = 0.999,
    max_full_ratio: float = 0.999,
    max_component_regression_ratio: float = 1.10,
    max_command_buffer_wait_ratio: float = 1.05,
    require_profile: bool = True,
    min_generated_tokens: int = 2,
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
    candidates: list[dict[str, Any]] = []
    for mode in [mode for mode in modes if mode != "default"]:
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
            if len(baseline.get("generated_ids") or []) < min_generated_tokens:
                prompt_failures.append("insufficient_generated_tokens_for_promotion")
            base.append_ratio_gate(
                prompt_failures,
                prompt_result,
                "headline_ms_per_token",
                row,
                baseline,
                base.headline_ms_per_token,
                max_headline_ratio,
                "missing_headline_ms_per_token",
                "headline_not_improved",
            )
            base.append_ratio_gate(
                prompt_failures,
                prompt_result,
                "full_attn_ms_avg",
                row,
                baseline,
                lambda item: base.chain_ms(item, "full_attn_ms_avg"),
                max_full_ratio,
                "missing_full_attn_ms_avg",
                "full_attn_not_improved",
            )
            for component in ("ffn_ms_avg", "linear_attn_ms_avg"):
                base.append_ratio_gate(
                    prompt_failures,
                    prompt_result,
                    component,
                    row,
                    baseline,
                    lambda item, metric_name=component: base.chain_ms(item, metric_name),
                    max_component_regression_ratio,
                    f"missing_{component}",
                    f"{component}_regressed",
                )
            base.append_ratio_gate(
                prompt_failures,
                prompt_result,
                "lm_head_ms_avg",
                row,
                baseline,
                base.lm_head_ms,
                max_component_regression_ratio,
                "missing_lm_head_ms_avg",
                "lm_head_ms_avg_regressed",
            )
            if require_profile:
                base.append_ratio_gate(
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
                row_value, baseline_value, metric_ratio = base.ratio(
                    row, baseline, command_buffer_wait_ms
                )
                prompt_result["command_buffer_wait_ms"] = row_value
                prompt_result["baseline_command_buffer_wait_ms"] = baseline_value
                prompt_result["command_buffer_wait_ms_ratio"] = metric_ratio
            prompt_result["full_attn_host_ms"] = row.get("full_attn_host_ms")
            prompt_result["copy_d2d_ms"] = row.get("copy_d2d_ms")
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
            "max_full_ratio": max_full_ratio,
            "max_component_regression_ratio": max_component_regression_ratio,
            "max_command_buffer_wait_ratio": max_command_buffer_wait_ratio,
            "require_profile": require_profile,
            "min_generated_tokens": min_generated_tokens,
        },
        "candidates": candidates,
    }


def summarize_with_gate(
    rows: list[dict[str, Any]],
    modes: list[str],
    max_headline_ratio: float = 0.999,
    max_full_ratio: float = 0.999,
    max_component_regression_ratio: float = 1.10,
    max_command_buffer_wait_ratio: float = 1.05,
    require_profile: bool = True,
    min_generated_tokens: int = 2,
) -> dict[str, Any]:
    summary = base.summarize(rows)
    summary["promotion_gate"] = build_promotion_gate(
        rows,
        modes,
        max_headline_ratio,
        max_full_ratio,
        max_component_regression_ratio,
        max_command_buffer_wait_ratio,
        require_profile,
        min_generated_tokens,
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
        "backend": "metal",
        "prompt_set": prompt_set,
        "modes": modes,
        "max_new_tokens": args.max_new_tokens,
        "context_size": args.context_size,
        "metal_profile": args.metal_profile,
        "promotion_thresholds": {
            "max_headline_ratio": args.promotion_max_headline_ratio,
            "max_full_ratio": args.promotion_max_full_ratio,
            "max_component_regression_ratio": args.promotion_max_component_regression_ratio,
            "max_command_buffer_wait_ratio": args.promotion_max_command_buffer_wait_ratio,
            "require_profile": args.promotion_require_profile,
            "min_generated_tokens": args.promotion_min_generated_tokens,
        },
        "summary": summarize_with_gate(
            rows,
            modes,
            args.promotion_max_headline_ratio,
            args.promotion_max_full_ratio,
            args.promotion_max_component_regression_ratio,
            args.promotion_max_command_buffer_wait_ratio,
            args.promotion_require_profile,
            args.promotion_min_generated_tokens,
        ),
        "rows": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    promotion_gate = summary.get("promotion_gate") or {}
    lines = [
        "# Qwen3.6 Full-Attention Decode Sweep",
        "",
        f"- prompt_set: `{report['prompt_set']}`",
        f"- modes: `{','.join(report['modes'])}`",
        f"- max_new_tokens: `{report['max_new_tokens']}`",
        f"- metal_profile: `{report['metal_profile']}`",
        f"- generated_ids_match: `{summary['generated_ids_match']}`",
        f"- promotion_gate_passed: `{promotion_gate.get('passed', False)}`",
        f"- promotion_gate_passed_modes: `{','.join(promotion_gate.get('passed_modes') or []) or '-'}`",
        "",
        "| Prompt | Mode | Status | IDs | Decode ms | Full-attn ms avg | Full host profile ms | copy_d2d ms | Wait ms | Top Metal op | Top Metal ms | Wall s |",
        "|:---|:---|:---|:---|---:|---:|---:|---:|---:|:---|---:|---:|",
    ]
    for row in report["rows"]:
        result = row.get("result") or {}
        chain = row.get("chain_breakdown") or {}
        top_metal = base.top_profile_op(row.get("metal_profile"))
        lines.append(
            "| {prompt} | {mode} | {status} | {ids} | {decode} | {full} | {host} | {copy_d2d} | {wait} | {top_op} | {top_ms} | {wall} |".format(
                prompt=row.get("prompt_id", ""),
                mode=row.get("mode", ""),
                status=row.get("status", ""),
                ids=",".join(str(item) for item in row.get("generated_ids", [])),
                decode=base.render_float(result.get("decode_ms")),
                full=base.render_float(chain.get("full_attn_ms_avg")),
                host=base.render_float(row.get("full_attn_host_ms")),
                copy_d2d=base.render_float(row.get("copy_d2d_ms")),
                wait=base.render_float(row.get("command_buffer_wait_ms")),
                top_op=top_metal.get("op") or "-",
                top_ms=base.render_float(top_metal.get("total_ms")),
                wall=base.render_float(row.get("wall_seconds"), 1),
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
        lines.append("")
        lines.append(
            "The gate is nonfatal. A full-attention decode mode passes only when generated IDs match default, the default row generated enough tokens for promotion, headline ms/token and full-attention time improve, FFN/linear-attention/lm-head stay inside the configured regression threshold, and command-buffer-wait attribution is present and not regressed when profile evidence is required."
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--prompt-set", choices=sorted(base.PROMPT_SETS), default="smoke")
    parser.add_argument("--prompt", action="append", help="custom prompt; repeat for a suite")
    parser.add_argument("--modes", default=DEFAULT_MODES)
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
        "--promotion-max-full-ratio",
        type=float,
        default=0.999,
        help="maximum candidate/default full_attn_ms_avg ratio for promotion",
    )
    parser.add_argument(
        "--promotion-max-component-regression-ratio",
        type=float,
        default=1.10,
        help="maximum allowed ratio for FFN, linear-attn, and lm-head buckets",
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
        "--promotion-min-generated-tokens",
        type=int,
        default=2,
        help="minimum generated IDs required before a row can pass the promotion gate",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("target/qwen36_full_decode_sweep.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("target/qwen36_full_decode_sweep.md"),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.model_dir = base.resolve_model_dir(args.model_dir, os.environ)
    modes = parse_modes(args.modes)
    prompts = base.select_prompts(args)
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
    gate = summary.get("promotion_gate") or {}
    print(
        "[qwen36-full-decode-sweep] rows={} ok={} generated_ids_match={} promotion_gate_passed={}".format(
            summary["rows"],
            summary["ok_rows"],
            str(summary["generated_ids_match"]).lower(),
            str(gate.get("passed", False)).lower(),
        )
    )
    print(f"[wrote] {args.out_json}")
    print(f"[wrote] {args.out_md}")
    return 0 if summary["ok_rows"] == summary["rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
