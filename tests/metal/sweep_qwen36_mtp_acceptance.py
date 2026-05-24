#!/usr/bin/env python3
"""Sweep Qwen3.6-MoE MTP acceptance across a small prompt suite."""

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

import probe_qwen36_mtp_acceptance as probe  # noqa: E402


SCHEMA = "qwen36-moe-mtp-acceptance-sweep-v2"
PROMPT_SETS: dict[str, list[tuple[str, str]]] = {
    "smoke": [
        (
            "profiling",
            "Explain how a local Metal inference runtime should decide whether "
            "native MTP is worth promoting from an experiment to a supported path.",
        ),
        (
            "coding",
            "Write a compact Rust helper that parses space-delimited key=value "
            "telemetry rows and returns drafted, accepted, and emitted token counts.",
        ),
    ],
    "comparison": [
        (
            "profiling",
            "Inspect a local Apple Metal inference profile and identify the next "
            "optimization target from route locality, FFN time, and command-buffer waits.",
        ),
        (
            "coding",
            "Write a compact Rust helper that parses space-delimited key=value "
            "telemetry rows and returns a typed summary with numeric fields.",
        ),
        (
            "reasoning",
            "A model accepts one draft token on some steps and rejects it on others. "
            "Explain how to compare acceptance rate against target steps per emitted token.",
        ),
        (
            "summary",
            "Summarize why resident expert layouts matter more than per-token expert "
            "packing for sparse MoE decode on a single Apple GPU.",
        ),
    ],
}


def select_prompts(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.prompt:
        return [(f"custom_{idx + 1}", prompt) for idx, prompt in enumerate(args.prompt)]
    return PROMPT_SETS[args.prompt_set]


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def build_promotion_gate(
    summary: dict[str, Any],
    promotion_min_acceptance: float,
    promotion_max_target_steps_per_emitted: float,
) -> dict[str, Any]:
    failures: list[str] = []
    status_counts_map = summary.get("status_counts") or {}
    if summary.get("measured_count") != summary.get("prompt_count"):
        failures.append("not_all_prompts_measured")
    if status_counts_map.get("policy_blocked", 0):
        failures.append("policy_blocked_rows")
    if summary.get("acceptance_rate", 0.0) < promotion_min_acceptance:
        failures.append("acceptance_below_threshold")
    target_steps = summary.get("target_steps_per_emitted", 0.0)
    if target_steps <= 0.0 or target_steps > promotion_max_target_steps_per_emitted:
        failures.append("target_steps_per_emitted_above_threshold")
    return {
        "passed": not failures,
        "failures": failures,
        "min_acceptance_rate": promotion_min_acceptance,
        "max_target_steps_per_emitted": promotion_max_target_steps_per_emitted,
    }


def build_summary(
    rows: list[dict[str, Any]],
    promotion_min_acceptance: float = 0.60,
    promotion_max_target_steps_per_emitted: float = 0.99,
) -> dict[str, Any]:
    measured = [row for row in rows if row.get("acceptance")]
    drafted = sum(int(row["acceptance"].get("drafted_tokens", 0)) for row in measured)
    accepted = sum(int(row["acceptance"].get("accepted_tokens", 0)) for row in measured)
    emitted = sum(int(row["acceptance"].get("emitted_tokens", 0)) for row in measured)
    target_steps = sum(
        int(row["acceptance"].get("base_steps", 0))
        + int(row["acceptance"].get("replay_steps", 0))
        for row in measured
    )
    full_accept_steps = sum(
        int(row["acceptance"].get("full_accept_steps", 0)) for row in measured
    )
    zero_accept_steps = sum(
        int(row["acceptance"].get("zero_accept_steps", 0)) for row in measured
    )
    summary = {
        "prompt_count": len(rows),
        "status_counts": status_counts(rows),
        "measured_count": len(measured),
        "wall_seconds": sum(float(row.get("wall_seconds", 0.0)) for row in rows),
        "drafted_tokens": drafted,
        "accepted_tokens": accepted,
        "acceptance_rate": accepted / drafted if drafted else 0.0,
        "emitted_tokens": emitted,
        "target_steps": target_steps,
        "target_steps_per_emitted": target_steps / emitted if emitted else 0.0,
        "full_accept_steps": full_accept_steps,
        "zero_accept_steps": zero_accept_steps,
    }
    summary["promotion_gate"] = build_promotion_gate(
        summary,
        promotion_min_acceptance,
        promotion_max_target_steps_per_emitted,
    )
    return summary


def trim_report(report: dict[str, Any], prompt_id: str, prompt_text: str) -> dict[str, Any]:
    row = {
        "prompt_id": prompt_id,
        "prompt": prompt_text,
        "backend": report["backend"],
        "mode": report["mode"],
        "status": report["status"],
        "returncode": report["returncode"],
        "wall_seconds": report["wall_seconds"],
        "command": report["command"],
        "acceptance": report.get("acceptance") or {},
        "policy_blocked": report.get("policy_blocked", False),
        "metal_profile": report.get("metal_profile"),
        "hal_profile": report.get("hal_profile"),
    }
    if report.get("output_tail"):
        row["output_tail"] = report["output_tail"]
    return row


def build_report(
    rows: list[dict[str, Any]],
    backend: str,
    mode: str,
    prompt_set: str,
    env_overrides: dict[str, str],
    promotion_min_acceptance: float = 0.60,
    promotion_max_target_steps_per_emitted: float = 0.99,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "model": probe.MODEL,
        "backend": backend,
        "mode": mode,
        "prompt_set": prompt_set,
        "env_overrides": env_overrides,
        "promotion_min_acceptance": promotion_min_acceptance,
        "promotion_max_target_steps_per_emitted": promotion_max_target_steps_per_emitted,
        "summary": build_summary(
            rows,
            promotion_min_acceptance,
            promotion_max_target_steps_per_emitted,
        ),
        "rows": rows,
    }


def render_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1%}"


def render_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def top_profile_op(profile: dict[str, Any] | None) -> dict[str, Any]:
    if not profile:
        return {}
    entries = profile.get("entries") or []
    return max(entries, key=lambda item: item.get("total_ms") or 0.0) if entries else {}


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    promotion_gate = summary.get("promotion_gate") or {}
    lines = [
        "# Qwen3.6 MTP Acceptance Sweep",
        "",
        f"- backend: `{report['backend']}`",
        f"- mode: `{report['mode']}`",
        f"- prompt_set: `{report['prompt_set']}`",
        f"- measured: `{summary['measured_count']}/{summary['prompt_count']}`",
        f"- aggregate_acceptance: `{summary['acceptance_rate']:.1%}`",
        f"- aggregate_target_steps_per_emitted: `{summary['target_steps_per_emitted']:.3f}`",
        f"- promotion_gate_passed: `{promotion_gate.get('passed', False)}`",
        f"- promotion_gate_failures: `{','.join(promotion_gate.get('failures') or []) or '-'}`",
        "",
        "| Prompt | Status | Drafted | Accepted | Acceptance | Emitted | Target steps/emitted | Top Metal op | Top Metal ms | HAL ms | Wall s |",
        "|:---|:---|---:|---:|---:|---:|---:|:---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        acceptance = row.get("acceptance") or {}
        rate = acceptance.get("acceptance_rate") if acceptance else None
        target = acceptance.get("target_steps_per_emitted") if acceptance else None
        top_metal = top_profile_op(row.get("metal_profile"))
        hal_total = ((row.get("hal_profile") or {}).get("summary") or {}).get("total_ms")
        lines.append(
            "| {prompt} | {status} | {drafted} | {accepted} | {rate} | {emitted} | {target} | {top_op} | {top_ms} | {hal_ms} | {wall:.1f} |".format(
                prompt=row["prompt_id"],
                status=row["status"],
                drafted=acceptance.get("drafted_tokens", "-"),
                accepted=acceptance.get("accepted_tokens", "-"),
                rate=render_percent(rate),
                emitted=acceptance.get("emitted_tokens", "-"),
                target=render_float(target),
                top_op=top_metal.get("op") or "-",
                top_ms=render_float(top_metal.get("total_ms")),
                hal_ms=render_float(hal_total),
                wall=float(row.get("wall_seconds", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "Aggregate acceptance is summed across measured rows only. Policy-blocked rows are preserved as status evidence but do not contribute to the acceptance denominator.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def run_prompt(args: argparse.Namespace, prompt_id: str, prompt_text: str) -> dict[str, Any]:
    prompt_args = copy.copy(args)
    prompt_args.prompt = prompt_text
    output, wall_seconds, returncode, command = probe.run_supersonic(prompt_args)
    report = probe.build_report(
        output,
        returncode,
        command,
        wall_seconds,
        args.backend,
        args.batched_spec_verify,
        probe.build_env_overrides(args),
    )
    return trim_report(report, prompt_id, prompt_text)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--backend", choices=("metal", "hip", "cuda"), default="metal")
    parser.add_argument("--prompt-set", choices=sorted(PROMPT_SETS), default="smoke")
    parser.add_argument(
        "--prompt",
        action="append",
        help="custom prompt; repeat to run a custom prompt suite instead of --prompt-set",
    )
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--batched-spec-verify", action="store_true")
    parser.add_argument(
        "--metal-experiment",
        action="store_true",
        help=f"set {probe.METAL_EXPERIMENT_ENV}=1 to run the env-gated Metal K=1 path",
    )
    parser.add_argument(
        "--metal-profile",
        action="store_true",
        help="set SUPERSONIC_METAL_PROFILE=1 and preserve parsed Metal/HAL profile rows",
    )
    parser.add_argument(
        "--promotion-min-acceptance",
        type=float,
        default=0.60,
        help="minimum aggregate acceptance rate for the reported promotion gate",
    )
    parser.add_argument(
        "--promotion-max-target-steps-per-emitted",
        type=float,
        default=0.99,
        help="maximum aggregate target-model steps per emitted token for promotion",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("target/qwen36_mtp_acceptance_sweep.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("target/qwen36_mtp_acceptance_sweep.md"),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.model_dir = probe.resolve_model_dir(args.model_dir, os.environ)
    prompts = select_prompts(args)
    prompt_set = "custom" if args.prompt else args.prompt_set
    rows = [run_prompt(args, prompt_id, text) for prompt_id, text in prompts]
    mode = "batched" if args.batched_spec_verify else "sequential"
    report = build_report(
        rows,
        args.backend,
        mode,
        prompt_set,
        probe.build_env_overrides(args),
        args.promotion_min_acceptance,
        args.promotion_max_target_steps_per_emitted,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report))

    summary = report["summary"]
    promotion_gate = summary.get("promotion_gate") or {}
    print(
        "[qwen36-mtp-acceptance-sweep] backend={} mode={} prompt_set={} measured={}/{} acceptance_rate={:.6f} target_steps_per_emitted={:.6f} promotion_gate_passed={}".format(
            report["backend"],
            report["mode"],
            report["prompt_set"],
            summary["measured_count"],
            summary["prompt_count"],
            summary["acceptance_rate"],
            summary["target_steps_per_emitted"],
            str(promotion_gate.get("passed", False)).lower(),
        )
    )
    print(f"[wrote] {args.out_json}")
    print(f"[wrote] {args.out_md}")

    bad_statuses = {"failed", "missing_acceptance", "measured_failed"}
    if any(row["status"] in bad_statuses for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
