#!/usr/bin/env python3
"""Sweep Qwen3.6 Metal fused routed INT4 decode variants."""

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
SCHEMA = "qwen36-fused-routed-int4-sweep-v5"
DEFAULT_MAX_FUSED_WALL_GPU_RATIO = 4.0
DEFAULT_MAX_WAIT_GPU_RATIO = 4.0

PROMPT_SETS: dict[str, list[tuple[str, str]]] = {
    "smoke": [("hello", "Hello")],
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
    "direct": "direct-gather",
    "direct-gather": "direct-gather",
    "direct-defer": "direct-defer-wait",
    "direct-defer-wait": "direct-defer-wait",
    "defer-direct-wait": "direct-defer-wait",
    "gpu-pack": "gpu-pack",
    "gpack": "gpu-pack",
    "full": "full-stage5",
    "full-stage5": "full-stage5",
    "native-stage5": "full-stage5",
    "stage5": "full-stage5",
    "router": "full-stage5-router",
    "router-stage5": "full-stage5-router",
    "full-router": "full-stage5-router",
    "full-stage5-router": "full-stage5-router",
    "router-defer": "router-defer-wait",
    "router-defer-wait": "router-defer-wait",
    "defer-router-wait": "router-defer-wait",
}
DEFAULT_MODES = "default,direct-gather,direct-defer-wait,gpu-pack,full-stage5,full-stage5-router,router-defer-wait"

FUSED_OP_NEEDLES = {
    "packed": "qwen36_ffn_int4_expert_packed_stage5",
    "direct-gather": "qwen36_ffn_int4_expert_direct_gather_stage5",
    "direct-defer-wait": "qwen36_ffn_int4_expert_direct_gather_stage5",
    "gpu-pack": "qwen36_ffn_int4_expert_gpu_pack_stage5",
    "full-stage5": "qwen36_ffn_int4_stage5",
    "full-stage5-router": "qwen36_ffn_int4_stage5_with_router",
    "router-defer-wait": "qwen36_ffn_int4_stage5_with_router",
}

FUSED_GPU_OP_PREFIXES = {
    "packed": ("command_buffer_gpu:qwen36_ffn_int4_expert_packed_stage5",),
    "direct-gather": ("command_buffer_gpu:qwen36_ffn_int4_expert_direct_gather_stage5",),
    "direct-defer-wait": ("command_buffer_gpu:qwen36_ffn_int4_expert_direct_gather_stage5",),
    "gpu-pack": ("command_buffer_gpu:qwen36_ffn_int4_expert_gpu_pack",),
    "full-stage5": (
        "command_buffer_gpu:qwen36_ffn_int4_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_up",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_scalar",
        "command_buffer_gpu:qwen36_ffn_int4_shared_down",
        "command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_tiled_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_expert_down_finalize",
    ),
    "full-stage5-router": (
        "command_buffer_gpu:qwen36_ffn_int4_stage5_with_router",
        "command_buffer_gpu:qwen36_ffn_int4_router_topk_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_up",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_scalar",
        "command_buffer_gpu:qwen36_ffn_int4_shared_down",
        "command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_tiled_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_expert_down_finalize",
    ),
    "router-defer-wait": (
        "command_buffer_gpu:qwen36_ffn_int4_stage5_with_router",
        "command_buffer_gpu:qwen36_ffn_int4_router_topk_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_up",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_scalar",
        "command_buffer_gpu:qwen36_ffn_int4_shared_down",
        "command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_tiled_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_expert_down_finalize",
    ),
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


def parse_router_parity_taps(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith("[qwen36-ffn-router-parity]"):
            continue
        fields = {
            key: parse_number(value)
            for key, value in parse_key_values(line).items()
        }
        rows.append(fields)
    return rows


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
    if getattr(args, "metal_profile_phases", False):
        overrides["SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES"] = "1"
    if getattr(args, "router_parity_tap", False):
        overrides["SUPERSONIC_METAL_QWEN36_FFN_ROUTER_STAGE5_PARITY_TAP"] = "1"
        max_calls = getattr(args, "router_parity_tap_max_calls", None)
        if max_calls:
            overrides["SUPERSONIC_METAL_QWEN36_FFN_ROUTER_STAGE5_PARITY_TAP_MAX_CALLS"] = str(
                max_calls
            )
    if mode == "packed":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACKED_STAGE5"] = "1"
    elif mode == "direct-gather":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DIRECT_GATHER_STAGE5"] = "1"
    elif mode == "direct-defer-wait":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DIRECT_GATHER_STAGE5"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DEFER_FFN_DIRECT_GATHER_STAGE5_WAIT"] = "1"
    elif mode == "gpu-pack":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACKED_STAGE5"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_GPU_PACK_STAGE5"] = "1"
    elif mode == "full-stage5":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5"] = "1"
    elif mode == "full-stage5-router":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
    elif mode == "router-defer-wait":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DEFER_FFN_ROUTER_STAGE5_WAIT"] = "1"
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


def profile_op_total_where(profile: dict[str, Any] | None, predicate: Any) -> float | None:
    if not profile:
        return None
    total = 0.0
    matched = False
    for entry in profile.get("entries") or []:
        if not predicate(entry):
            continue
        matched = True
        total += float(entry.get("total_ms") or 0.0)
    return total if matched else None


def command_buffer_wait_ms(row: dict[str, Any]) -> float | None:
    return profile_op_total(row.get("metal_profile"), "command_buffer_wait")


def fused_op_ms(row: dict[str, Any]) -> float | None:
    needle = FUSED_OP_NEEDLES.get(str(row.get("mode") or ""))
    if needle is None:
        return None
    return profile_op_total(row.get("metal_profile"), needle)


def fused_wall_ms(row: dict[str, Any]) -> float | None:
    needle = FUSED_OP_NEEDLES.get(str(row.get("mode") or ""))
    if needle is None:
        return None
    needle = needle.lower()

    def matches(entry: dict[str, Any]) -> bool:
        op = str(entry.get("op") or "").lower()
        path = str(entry.get("path") or "").lower()
        return needle in op and not op.startswith("command_buffer_gpu:") and path in {
            "",
            "native",
        }

    return profile_op_total_where(row.get("metal_profile"), matches)


def fused_gpu_ms(row: dict[str, Any]) -> float | None:
    prefixes = tuple(
        prefix.lower() for prefix in FUSED_GPU_OP_PREFIXES.get(str(row.get("mode") or ""), ())
    )
    if not prefixes:
        return None

    def matches(entry: dict[str, Any]) -> bool:
        op = str(entry.get("op") or "").lower()
        return any(op.startswith(prefix) for prefix in prefixes)

    return profile_op_total_where(row.get("metal_profile"), matches)


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def classify_ffn_attribution(row: dict[str, Any], max_wall_gpu_ratio: float, max_wait_gpu_ratio: float) -> str:
    if row.get("mode") == "default":
        return "host_or_default"
    if row.get("status") != "ok":
        return "unavailable"
    if not row.get("metal_profile"):
        return "missing_profile"
    gpu_ms = row.get("fused_gpu_ms")
    if gpu_ms is None or gpu_ms == 0:
        return "missing_gpu_profile"
    wait_gpu_ratio = row.get("wait_gpu_ratio")
    wall_gpu_ratio = row.get("fused_wall_gpu_ratio")
    if (
        wait_gpu_ratio is not None
        and wait_gpu_ratio > max_wait_gpu_ratio
        or wall_gpu_ratio is not None
        and wall_gpu_ratio > max_wall_gpu_ratio
    ):
        return "residency_or_submit_wait"
    return "gpu_arithmetic"


def annotate_ffn_profile_fields(
    rows: list[dict[str, Any]],
    max_wall_gpu_ratio: float = DEFAULT_MAX_FUSED_WALL_GPU_RATIO,
    max_wait_gpu_ratio: float = DEFAULT_MAX_WAIT_GPU_RATIO,
) -> None:
    for row in rows:
        row["command_buffer_wait_ms"] = command_buffer_wait_ms(row)
        row["fused_op_ms"] = fused_op_ms(row)
        row["fused_wall_ms"] = fused_wall_ms(row)
        row["fused_gpu_ms"] = fused_gpu_ms(row)
        row["fused_wall_gpu_ratio"] = safe_ratio(row["fused_wall_ms"], row["fused_gpu_ms"])
        row["wait_gpu_ratio"] = safe_ratio(row["command_buffer_wait_ms"], row["fused_gpu_ms"])
        row["ffn_attribution_class"] = classify_ffn_attribution(
            row,
            max_wall_gpu_ratio,
            max_wait_gpu_ratio,
        )


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


def top_profile_op(profile: dict[str, Any] | None) -> dict[str, Any]:
    if not profile:
        return {}
    entries = profile.get("entries") or []
    return max(entries, key=lambda item: item.get("total_ms") or 0.0) if entries else {}


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
            "metal_profile": parse_profile(output, "[metal-profile]", "[metal-profile-op]"),
            "hal_profile": parse_profile(output, "[hal-profile]", "[hal-profile-op]"),
            "router_parity_taps": parse_router_parity_taps(output),
            "output_tail": output_tail(output),
        }
        row["fused_op_ms"] = fused_op_ms(row)
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
            "metal_profile": parse_profile(output, "[metal-profile]", "[metal-profile-op]"),
            "hal_profile": parse_profile(output, "[hal-profile]", "[hal-profile-op]"),
            "router_parity_taps": parse_router_parity_taps(output),
            "fused_op_ms": None,
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
            prompt_result["fused_op_ms"] = row.get("fused_op_ms")
            prompt_result["fused_wall_ms"] = row.get("fused_wall_ms")
            prompt_result["fused_gpu_ms"] = row.get("fused_gpu_ms")
            prompt_result["fused_wall_gpu_ratio"] = row.get("fused_wall_gpu_ratio")
            prompt_result["wait_gpu_ratio"] = row.get("wait_gpu_ratio")
            prompt_result["ffn_attribution_class"] = row.get("ffn_attribution_class")
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


def build_ffn_residency_gap(
    rows: list[dict[str, Any]],
    modes: list[str],
    max_wall_gpu_ratio: float = DEFAULT_MAX_FUSED_WALL_GPU_RATIO,
    max_wait_gpu_ratio: float = DEFAULT_MAX_WAIT_GPU_RATIO,
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
        prompt_results: list[dict[str, Any]] = []
        classes: set[str] = set()
        for prompt_id in prompt_ids:
            row = rows_by_key.get((prompt_id, mode))
            baseline = rows_by_key.get((prompt_id, "default"))
            generated_ids_match_default = (
                None
                if row is None or baseline is None
                else (row.get("generated_ids") or []) == (baseline.get("generated_ids") or [])
            )
            row_class = str((row or {}).get("ffn_attribution_class") or "missing_candidate")
            classes.add(row_class)
            prompt_results.append(
                {
                    "prompt_id": prompt_id,
                    "status": (row or {}).get("status"),
                    "generated_ids_match_default": generated_ids_match_default,
                    "fused_op_ms": (row or {}).get("fused_op_ms"),
                    "fused_wall_ms": (row or {}).get("fused_wall_ms"),
                    "fused_gpu_ms": (row or {}).get("fused_gpu_ms"),
                    "fused_wall_gpu_ratio": (row or {}).get("fused_wall_gpu_ratio"),
                    "command_buffer_wait_ms": (row or {}).get("command_buffer_wait_ms"),
                    "wait_gpu_ratio": (row or {}).get("wait_gpu_ratio"),
                    "ffn_attribution_class": row_class,
                }
            )
        candidates.append(
            {
                "mode": mode,
                "classes": sorted(classes),
                "prompts": prompt_results,
            }
        )
    all_classes = {
        cls
        for candidate in candidates
        for cls in candidate.get("classes", [])
    }
    residency_modes = [
        candidate["mode"]
        for candidate in candidates
        if "residency_or_submit_wait" in candidate.get("classes", [])
    ]
    gpu_arithmetic_modes = [
        candidate["mode"]
        for candidate in candidates
        if "gpu_arithmetic" in candidate.get("classes", [])
    ]
    if residency_modes:
        recommendation = "prototype_ffn_residency_or_submit_wait_path"
        reason = "candidate GPU timestamps are much smaller than native wall or command-buffer wait totals"
    elif gpu_arithmetic_modes:
        recommendation = "prototype_ffn_gpu_arithmetic_tiling_path"
        reason = "candidate native wall time tracks GPU command time closely enough to focus on arithmetic"
    elif all_classes & {"missing_profile", "missing_gpu_profile"}:
        recommendation = "refresh_fused_ffn_sweep_with_metal_profile"
        reason = "candidate rows are missing enough GPU attribution to classify the FFN gap"
    else:
        recommendation = "inspect_fused_ffn_candidate_gap"
        reason = "candidate rows did not produce a dominant residency or GPU-arithmetic class"
    return {
        "thresholds": {
            "max_fused_wall_gpu_ratio": max_wall_gpu_ratio,
            "max_wait_gpu_ratio": max_wait_gpu_ratio,
        },
        "recommendation": recommendation,
        "reason": reason,
        "residency_or_submit_wait_modes": residency_modes,
        "gpu_arithmetic_modes": gpu_arithmetic_modes,
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
    max_fused_wall_gpu_ratio: float = DEFAULT_MAX_FUSED_WALL_GPU_RATIO,
    max_wait_gpu_ratio: float = DEFAULT_MAX_WAIT_GPU_RATIO,
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
    summary["ffn_residency_gap"] = build_ffn_residency_gap(
        rows,
        modes,
        max_fused_wall_gpu_ratio,
        max_wait_gpu_ratio,
    )
    return summary


def build_report(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    modes: list[str],
    prompt_set: str,
) -> dict[str, Any]:
    annotate_ffn_profile_fields(
        rows,
        args.promotion_max_fused_wall_gpu_ratio,
        args.promotion_max_wait_gpu_ratio,
    )
    return {
        "schema": SCHEMA,
        "model": MODEL,
        "backend": "metal",
        "prompt_set": prompt_set,
        "modes": modes,
        "max_new_tokens": args.max_new_tokens,
        "context_size": args.context_size,
        "metal_profile": args.metal_profile,
        "metal_profile_phases": getattr(args, "metal_profile_phases", False),
        "router_parity_tap": getattr(args, "router_parity_tap", False),
        "router_parity_tap_max_calls": getattr(args, "router_parity_tap_max_calls", None),
        "promotion_thresholds": {
            "max_headline_ratio": args.promotion_max_headline_ratio,
            "max_ffn_ratio": args.promotion_max_ffn_ratio,
            "max_component_regression_ratio": args.promotion_max_component_regression_ratio,
            "max_command_buffer_wait_ratio": args.promotion_max_command_buffer_wait_ratio,
            "max_fused_wall_gpu_ratio": args.promotion_max_fused_wall_gpu_ratio,
            "max_wait_gpu_ratio": args.promotion_max_wait_gpu_ratio,
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
            args.promotion_max_fused_wall_gpu_ratio,
            args.promotion_max_wait_gpu_ratio,
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


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    promotion_gate = summary.get("promotion_gate") or {}
    ffn_gap = summary.get("ffn_residency_gap") or {}
    lines = [
        "# Qwen3.6 Fused Routed INT4 Sweep",
        "",
        f"- prompt_set: `{report['prompt_set']}`",
        f"- modes: `{','.join(report['modes'])}`",
        f"- max_new_tokens: `{report['max_new_tokens']}`",
        f"- metal_profile: `{report['metal_profile']}`",
        f"- metal_profile_phases: `{report.get('metal_profile_phases', False)}`",
        f"- router_parity_tap: `{report.get('router_parity_tap', False)}`",
        f"- generated_ids_match: `{summary['generated_ids_match']}`",
        f"- promotion_gate_passed: `{promotion_gate.get('passed', False)}`",
        f"- promotion_gate_passed_modes: `{','.join(promotion_gate.get('passed_modes') or []) or '-'}`",
        f"- ffn_gap_recommendation: `{ffn_gap.get('recommendation') or '-'}`",
        "",
        "| Prompt | Mode | Status | IDs | Decode ms | FFN ms avg | Fused wall ms | Fused GPU ms | Wall/GPU | Wait/GPU | FFN class | Top Metal op | Top Metal ms | HAL ms | Wall s |",
        "|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|:---|:---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        result = row.get("result") or {}
        chain = row.get("chain_breakdown") or {}
        top_metal = top_profile_op(row.get("metal_profile"))
        hal_summary = (row.get("hal_profile") or {}).get("summary") or {}
        lines.append(
            "| {prompt} | {mode} | {status} | {ids} | {decode} | {ffn} | {fused_wall} | {fused_gpu} | {wall_gpu} | {wait_gpu} | {ffn_class} | {top_op} | {top_ms} | {hal_ms} | {wall} |".format(
                prompt=row.get("prompt_id", ""),
                mode=row.get("mode", ""),
                status=row.get("status", ""),
                ids=",".join(str(item) for item in row.get("generated_ids", [])),
                decode=render_float(result.get("decode_ms")),
                ffn=render_float(chain.get("ffn_ms_avg")),
                fused_wall=render_float(row.get("fused_wall_ms")),
                fused_gpu=render_float(row.get("fused_gpu_ms")),
                wall_gpu=render_float(row.get("fused_wall_gpu_ratio"), 2),
                wait_gpu=render_float(row.get("wait_gpu_ratio"), 2),
                ffn_class=row.get("ffn_attribution_class") or "-",
                top_op=top_metal.get("op") or "-",
                top_ms=render_float(top_metal.get("total_ms")),
                hal_ms=render_float(hal_summary.get("total_ms")),
                wall=render_float(row.get("wall_seconds"), 1),
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
            "The gate is nonfatal. A fused routed INT4 mode passes only when generated IDs match default, headline ms/token and FFN time improve, full-attention/linear-attention/lm-head stay inside the configured regression threshold, and command-buffer-wait attribution is present and not regressed when profile evidence is required."
        )
    gap_candidates = ffn_gap.get("candidates") or []
    if gap_candidates:
        lines.extend(
            [
                "",
                "## FFN Residency Gap",
                "",
                f"- recommendation: `{ffn_gap.get('recommendation') or '-'}`",
                f"- reason: {ffn_gap.get('reason') or '-'}",
                "",
                "| Mode | Prompt | Class | IDs Match | Fused wall ms | Fused GPU ms | Wall/GPU | Wait ms | Wait/GPU |",
                "|:---|:---|:---|:---:|---:|---:|---:|---:|---:|",
            ]
        )
        for candidate in gap_candidates:
            for prompt in candidate.get("prompts") or []:
                ids_match = prompt.get("generated_ids_match_default")
                lines.append(
                    "| {mode} | {prompt} | {cls} | {ids} | {wall} | {gpu} | {wall_gpu} | {wait} | {wait_gpu} |".format(
                        mode=candidate.get("mode"),
                        prompt=prompt.get("prompt_id"),
                        cls=prompt.get("ffn_attribution_class") or "-",
                        ids="-" if ids_match is None else str(bool(ids_match)).lower(),
                        wall=render_float(prompt.get("fused_wall_ms")),
                        gpu=render_float(prompt.get("fused_gpu_ms")),
                        wall_gpu=render_float(prompt.get("fused_wall_gpu_ratio"), 2),
                        wait=render_float(prompt.get("command_buffer_wait_ms")),
                        wait_gpu=render_float(prompt.get("wait_gpu_ratio"), 2),
                    )
                )
    tap_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in report["rows"]:
        for tap in row.get("router_parity_taps") or []:
            tap_rows.append((row, tap))
    if tap_rows:
        lines.extend(
            [
                "",
                "## Router Parity Tap",
                "",
                "| Prompt | Mode | Layer | Match | HNorm max | HNorm idx | Logit max | Logit idx | TopK weight max | Host idx | Metal idx |",
                "|:---|:---|---:|:---:|---:|---:|---:|---:|---:|:---|:---|",
            ]
        )
        for row, tap in tap_rows[:40]:
            lines.append(
                "| {prompt} | {mode} | {layer} | {match} | {hnorm} | {hnorm_idx} | {logits} | {logits_idx} | {weight} | {host_idx} | {metal_idx} |".format(
                    prompt=row.get("prompt_id", ""),
                    mode=row.get("mode", ""),
                    layer=tap.get("layer", "-"),
                    match=str(bool(tap.get("topk_idx_match"))).lower(),
                    hnorm=render_float(tap.get("h_norm_max_abs"), 8),
                    hnorm_idx=tap.get("h_norm_argmax", "-"),
                    logits=render_float(tap.get("logits_max_abs"), 8),
                    logits_idx=tap.get("logits_argmax", "-"),
                    weight=render_float(tap.get("topk_weight_max_abs"), 8),
                    host_idx=tap.get("host_idx", "-"),
                    metal_idx=tap.get("workspace_idx", tap.get("output_idx", "-")),
                )
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
    parser.add_argument("--metal-profile", action="store_true")
    parser.add_argument(
        "--metal-profile-phases",
        action="store_true",
        help="split Qwen3.6 FFN Metal profile runs into per-phase command buffers",
    )
    parser.add_argument(
        "--router-parity-tap",
        action="store_true",
        help="emit and parse Qwen3.6 full-stage5-router Metal-vs-host router parity rows",
    )
    parser.add_argument(
        "--router-parity-tap-max-calls",
        type=int,
        default=40,
        help="maximum router parity tap rows emitted by the runtime",
    )
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
        "--promotion-max-fused-wall-gpu-ratio",
        type=float,
        default=DEFAULT_MAX_FUSED_WALL_GPU_RATIO,
        help="maximum native fused FFN wall/GPU profile ratio before classifying the candidate as residency or submit wait bound",
    )
    parser.add_argument(
        "--promotion-max-wait-gpu-ratio",
        type=float,
        default=DEFAULT_MAX_WAIT_GPU_RATIO,
        help="maximum command_buffer_wait/GPU profile ratio before classifying the candidate as residency or submit wait bound",
    )
    parser.add_argument(
        "--promotion-require-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require command_buffer_wait profile evidence for promotion",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("target/qwen36_fused_routed_int4_sweep.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("target/qwen36_fused_routed_int4_sweep.md"),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.model_dir = resolve_model_dir(args.model_dir, os.environ)
    modes = parse_modes(args.modes)
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
    gate = summary.get("promotion_gate") or {}
    print(
        "[qwen36-fused-routed-int4-sweep] rows={} ok={} generated_ids_match={} promotion_gate_passed={}".format(
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
