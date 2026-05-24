#!/usr/bin/env python3
"""Sweep Qwen3.6 Metal fused routed INT4 decode variants."""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


MODEL = "qwen3.6-35b-a3b"
SCHEMA = "qwen36-fused-routed-int4-sweep-v20"
DEFAULT_MAX_FUSED_WALL_GPU_RATIO = 4.0
DEFAULT_MAX_WAIT_GPU_RATIO = 4.0
COARSE_BATCH_SERIAL_MODE = "full-stage5-router-batch"
COARSE_BATCH_SIMD_MODE = "full-stage5-router-simd-batch"
DEFERRED_BATCH_SERIAL_MODE = "full-stage5-router-batch-deferred-phases"
DEFERRED_BATCH_SIMD_MODE = "full-stage5-router-simd-batch-deferred-phases"
SHARED_TILED_BATCH_SIMD_MODE = "full-stage5-router-simd-batch-shared-tiled"
SHARED_TILED_DEFERRED_SIMD_MODE = "full-stage5-router-simd-batch-shared-tiled-deferred-phases"
SHARED_TILED_FFN_PHASE_SIMD_MODE = "full-stage5-router-simd-batch-shared-tiled-ffn-phases"
SHARED_GATE_UP_TILED_BATCH_SIMD_MODE = "full-stage5-router-simd-batch-shared-gate-up-tiled"
SHARED_SCALAR_SIMD_BATCH_SIMD_MODE = "full-stage5-router-simd-batch-shared-scalar-simd"
SHARED_DOWN_TILED_BATCH_SIMD_MODE = "full-stage5-router-simd-batch-shared-down-tiled"
SHARED_TILED_MODES = {
    SHARED_TILED_BATCH_SIMD_MODE,
    SHARED_TILED_DEFERRED_SIMD_MODE,
    SHARED_TILED_FFN_PHASE_SIMD_MODE,
}
SHARED_COMPONENT_MODES = {
    SHARED_GATE_UP_TILED_BATCH_SIMD_MODE,
    SHARED_SCALAR_SIMD_BATCH_SIMD_MODE,
    SHARED_DOWN_TILED_BATCH_SIMD_MODE,
}
BATCH_FAST_PROFILE_MODES = {
    "full-stage5-router-batch",
    "full-stage5-router-simd-batch",
    "full-stage5-router-batch-deferred-phases",
    "full-stage5-router-simd-batch-deferred-phases",
    SHARED_TILED_BATCH_SIMD_MODE,
    SHARED_TILED_DEFERRED_SIMD_MODE,
    SHARED_TILED_FFN_PHASE_SIMD_MODE,
    SHARED_GATE_UP_TILED_BATCH_SIMD_MODE,
    SHARED_SCALAR_SIMD_BATCH_SIMD_MODE,
    SHARED_DOWN_TILED_BATCH_SIMD_MODE,
    "full-stage5-router-batch-phases",
    "full-stage5-router-batch-ffn-phases",
    "full-stage5-router-simd-batch-phases",
    "full-stage5-router-simd-batch-ffn-phases",
}

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
    "router-simd": "full-stage5-router-simd",
    "router-stage5-simd": "full-stage5-router-simd",
    "full-router-simd": "full-stage5-router-simd",
    "full-stage5-router-simd": "full-stage5-router-simd",
    "router-batch": "full-stage5-router-batch",
    "batch-router": "full-stage5-router-batch",
    "full-router-batch": "full-stage5-router-batch",
    "full-stage5-router-batch": "full-stage5-router-batch",
    "router-simd-batch": "full-stage5-router-simd-batch",
    "batch-router-simd": "full-stage5-router-simd-batch",
    "full-router-simd-batch": "full-stage5-router-simd-batch",
    "full-stage5-router-simd-batch": "full-stage5-router-simd-batch",
    "router-simd-batch-shared-tiled": SHARED_TILED_BATCH_SIMD_MODE,
    "batch-router-simd-shared-tiled": SHARED_TILED_BATCH_SIMD_MODE,
    "full-router-simd-batch-shared-tiled": SHARED_TILED_BATCH_SIMD_MODE,
    SHARED_TILED_BATCH_SIMD_MODE: SHARED_TILED_BATCH_SIMD_MODE,
    "router-simd-batch-shared-gate-up-tiled": SHARED_GATE_UP_TILED_BATCH_SIMD_MODE,
    "batch-router-simd-shared-gate-up-tiled": SHARED_GATE_UP_TILED_BATCH_SIMD_MODE,
    SHARED_GATE_UP_TILED_BATCH_SIMD_MODE: SHARED_GATE_UP_TILED_BATCH_SIMD_MODE,
    "router-simd-batch-shared-scalar-simd": SHARED_SCALAR_SIMD_BATCH_SIMD_MODE,
    "batch-router-simd-shared-scalar-simd": SHARED_SCALAR_SIMD_BATCH_SIMD_MODE,
    SHARED_SCALAR_SIMD_BATCH_SIMD_MODE: SHARED_SCALAR_SIMD_BATCH_SIMD_MODE,
    "router-simd-batch-shared-down-tiled": SHARED_DOWN_TILED_BATCH_SIMD_MODE,
    "batch-router-simd-shared-down-tiled": SHARED_DOWN_TILED_BATCH_SIMD_MODE,
    SHARED_DOWN_TILED_BATCH_SIMD_MODE: SHARED_DOWN_TILED_BATCH_SIMD_MODE,
    "router-batch-deferred-phases": "full-stage5-router-batch-deferred-phases",
    "batch-router-deferred-phases": "full-stage5-router-batch-deferred-phases",
    "full-router-batch-deferred-phases": "full-stage5-router-batch-deferred-phases",
    "full-stage5-router-batch-deferred-phases": "full-stage5-router-batch-deferred-phases",
    "router-simd-batch-deferred-phases": "full-stage5-router-simd-batch-deferred-phases",
    "batch-router-simd-deferred-phases": "full-stage5-router-simd-batch-deferred-phases",
    "full-router-simd-batch-deferred-phases": "full-stage5-router-simd-batch-deferred-phases",
    "full-stage5-router-simd-batch-deferred-phases": "full-stage5-router-simd-batch-deferred-phases",
    "router-simd-batch-shared-tiled-deferred-phases": SHARED_TILED_DEFERRED_SIMD_MODE,
    "batch-router-simd-shared-tiled-deferred-phases": SHARED_TILED_DEFERRED_SIMD_MODE,
    "full-router-simd-batch-shared-tiled-deferred-phases": SHARED_TILED_DEFERRED_SIMD_MODE,
    SHARED_TILED_DEFERRED_SIMD_MODE: SHARED_TILED_DEFERRED_SIMD_MODE,
    "router-batch-phases": "full-stage5-router-batch-phases",
    "batch-router-phases": "full-stage5-router-batch-phases",
    "full-router-batch-phases": "full-stage5-router-batch-phases",
    "full-stage5-router-batch-phases": "full-stage5-router-batch-phases",
    "router-batch-ffn-phases": "full-stage5-router-batch-ffn-phases",
    "batch-router-ffn-phases": "full-stage5-router-batch-ffn-phases",
    "full-router-batch-ffn-phases": "full-stage5-router-batch-ffn-phases",
    "full-stage5-router-batch-ffn-phases": "full-stage5-router-batch-ffn-phases",
    "router-simd-batch-phases": "full-stage5-router-simd-batch-phases",
    "batch-router-simd-phases": "full-stage5-router-simd-batch-phases",
    "full-router-simd-batch-phases": "full-stage5-router-simd-batch-phases",
    "full-stage5-router-simd-batch-phases": "full-stage5-router-simd-batch-phases",
    "router-simd-batch-ffn-phases": "full-stage5-router-simd-batch-ffn-phases",
    "batch-router-simd-ffn-phases": "full-stage5-router-simd-batch-ffn-phases",
    "full-router-simd-batch-ffn-phases": "full-stage5-router-simd-batch-ffn-phases",
    "full-stage5-router-simd-batch-ffn-phases": "full-stage5-router-simd-batch-ffn-phases",
    "router-simd-batch-shared-tiled-ffn-phases": SHARED_TILED_FFN_PHASE_SIMD_MODE,
    "batch-router-simd-shared-tiled-ffn-phases": SHARED_TILED_FFN_PHASE_SIMD_MODE,
    "full-router-simd-batch-shared-tiled-ffn-phases": SHARED_TILED_FFN_PHASE_SIMD_MODE,
    SHARED_TILED_FFN_PHASE_SIMD_MODE: SHARED_TILED_FFN_PHASE_SIMD_MODE,
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
    "full-stage5-router-simd": "qwen36_ffn_int4_stage5_with_router",
    "full-stage5-router-batch": "qwen36_ffn_int4_stage5_with_router",
    "full-stage5-router-simd-batch": "qwen36_ffn_int4_stage5_with_router",
    "full-stage5-router-batch-deferred-phases": "qwen36_ffn_int4_stage5_with_router",
    "full-stage5-router-simd-batch-deferred-phases": "qwen36_ffn_int4_stage5_with_router",
    SHARED_TILED_BATCH_SIMD_MODE: "qwen36_ffn_int4_stage5_with_router",
    SHARED_TILED_DEFERRED_SIMD_MODE: "qwen36_ffn_int4_stage5_with_router",
    SHARED_TILED_FFN_PHASE_SIMD_MODE: "qwen36_ffn_int4",
    SHARED_GATE_UP_TILED_BATCH_SIMD_MODE: "qwen36_ffn_int4_stage5_with_router",
    SHARED_SCALAR_SIMD_BATCH_SIMD_MODE: "qwen36_ffn_int4_stage5_with_router",
    SHARED_DOWN_TILED_BATCH_SIMD_MODE: "qwen36_ffn_int4_stage5_with_router",
    "full-stage5-router-batch-phases": "qwen36_ffn_int4_stage5_with_router",
    "full-stage5-router-batch-ffn-phases": "qwen36_ffn_int4",
    "full-stage5-router-simd-batch-phases": "qwen36_ffn_int4_stage5_with_router",
    "full-stage5-router-simd-batch-ffn-phases": "qwen36_ffn_int4",
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
        "command_buffer_gpu:qwen36_ffn_int4_router_topk_stage5_simd",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_up",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_scalar",
        "command_buffer_gpu:qwen36_ffn_int4_shared_down",
        "command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_tiled_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_expert_down_finalize",
    ),
    "full-stage5-router-simd": (
        "command_buffer_gpu:qwen36_ffn_int4_stage5_with_router",
        "command_buffer_gpu:qwen36_ffn_int4_router_topk_stage5_simd",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_up",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_scalar",
        "command_buffer_gpu:qwen36_ffn_int4_shared_down",
        "command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_tiled_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_expert_down_finalize",
    ),
    "full-stage5-router-batch": (
        "command_buffer_gpu:qwen36_decode_batch",
        "command_buffer_gpu:qwen36_ffn_int4_stage5_with_router",
        "command_buffer_gpu:qwen36_ffn_int4_router_topk_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_up",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_scalar",
        "command_buffer_gpu:qwen36_ffn_int4_shared_down",
        "command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_tiled_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_expert_down_finalize",
    ),
    "full-stage5-router-simd-batch": (
        "command_buffer_gpu:qwen36_decode_batch",
        "command_buffer_gpu:qwen36_ffn_int4_stage5_with_router",
        "command_buffer_gpu:qwen36_ffn_int4_router_topk_stage5_simd",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_up",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_scalar",
        "command_buffer_gpu:qwen36_ffn_int4_shared_down",
        "command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_tiled_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_expert_down_finalize",
    ),
    "full-stage5-router-batch-deferred-phases": (
        "command_buffer_gpu:qwen36_decode_batch_ffn",
        "command_buffer_gpu:qwen36_ffn_int4_stage5_with_router",
        "command_buffer_gpu:qwen36_ffn_int4_router_topk_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_up",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_scalar",
        "command_buffer_gpu:qwen36_ffn_int4_shared_down",
        "command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_tiled_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_expert_down_finalize",
    ),
    "full-stage5-router-simd-batch-deferred-phases": (
        "command_buffer_gpu:qwen36_decode_batch_ffn",
        "command_buffer_gpu:qwen36_ffn_int4_stage5_with_router",
        "command_buffer_gpu:qwen36_ffn_int4_router_topk_stage5_simd",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_up",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_scalar",
        "command_buffer_gpu:qwen36_ffn_int4_shared_down",
        "command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_tiled_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_expert_down_finalize",
    ),
    "full-stage5-router-batch-phases": (
        "command_buffer_gpu:qwen36_decode_batch_ffn",
        "command_buffer_gpu:qwen36_ffn_int4_stage5_with_router",
        "command_buffer_gpu:qwen36_ffn_int4_router_topk_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_up",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_scalar",
        "command_buffer_gpu:qwen36_ffn_int4_shared_down",
        "command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_tiled_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_expert_down_finalize",
    ),
    "full-stage5-router-simd-batch-phases": (
        "command_buffer_gpu:qwen36_decode_batch_ffn",
        "command_buffer_gpu:qwen36_ffn_int4_stage5_with_router",
        "command_buffer_gpu:qwen36_ffn_int4_router_topk_stage5_simd",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_up",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_scalar",
        "command_buffer_gpu:qwen36_ffn_int4_shared_down",
        "command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_tiled_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_expert_down_finalize",
    ),
    "full-stage5-router-batch-ffn-phases": (
        "command_buffer_gpu:qwen36_ffn_int4_router_topk_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_up",
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_scalar",
        "command_buffer_gpu:qwen36_ffn_int4_shared_down",
        "command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_tiled_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_expert_down_finalize",
    ),
    "full-stage5-router-simd-batch-ffn-phases": (
        "command_buffer_gpu:qwen36_ffn_int4_router_topk_stage5_simd",
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

FUSED_GPU_OP_PREFIXES[SHARED_TILED_BATCH_SIMD_MODE] = FUSED_GPU_OP_PREFIXES[
    "full-stage5-router-simd-batch"
]
FUSED_GPU_OP_PREFIXES[SHARED_GATE_UP_TILED_BATCH_SIMD_MODE] = FUSED_GPU_OP_PREFIXES[
    "full-stage5-router-simd-batch"
]
FUSED_GPU_OP_PREFIXES[SHARED_SCALAR_SIMD_BATCH_SIMD_MODE] = FUSED_GPU_OP_PREFIXES[
    "full-stage5-router-simd-batch"
]
FUSED_GPU_OP_PREFIXES[SHARED_DOWN_TILED_BATCH_SIMD_MODE] = FUSED_GPU_OP_PREFIXES[
    "full-stage5-router-simd-batch"
]
FUSED_GPU_OP_PREFIXES[SHARED_TILED_DEFERRED_SIMD_MODE] = FUSED_GPU_OP_PREFIXES[
    "full-stage5-router-simd-batch-deferred-phases"
]
FUSED_GPU_OP_PREFIXES[SHARED_TILED_FFN_PHASE_SIMD_MODE] = FUSED_GPU_OP_PREFIXES[
    "full-stage5-router-simd-batch-ffn-phases"
]

BATCH_FFN_PHASE_GPU_FIELDS = {
    "decode_batch_ffn_router_topk_gpu_ms": (
        "command_buffer_gpu:qwen36_ffn_int4_router_topk_stage5",
        "command_buffer_gpu:qwen36_ffn_int4_router_topk_stage5_simd",
    ),
    "decode_batch_ffn_shared_gate_up_gpu_ms": (
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_up",
    ),
    "decode_batch_ffn_shared_scalar_gpu_ms": (
        "command_buffer_gpu:qwen36_ffn_int4_shared_gate_scalar",
    ),
    "decode_batch_ffn_shared_down_gpu_ms": ("command_buffer_gpu:qwen36_ffn_int4_shared_down",),
    "decode_batch_ffn_expert_gate_up_gpu_ms": (
        "command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_tiled_stage5",
    ),
    "decode_batch_ffn_expert_down_gpu_ms": (
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


def parse_shared_parity_taps(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith("[qwen36-ffn-shared-parity]"):
            continue
        fields = {
            key: parse_number(value)
            for key, value in parse_key_values(line).items()
        }
        rows.append(fields)
    return rows


def parse_decode_batch_shared_parity_taps(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith("[qwen36-decode-batch-shared-parity]"):
            continue
        fields = {
            key: parse_number(value)
            for key, value in parse_key_values(line).items()
        }
        rows.append(fields)
    return rows


def parse_decode_batch_routed_parity_taps(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith("[qwen36-decode-batch-routed-parity]"):
            continue
        fields = {
            key: parse_number(value)
            for key, value in parse_key_values(line).items()
        }
        rows.append(fields)
    return rows


def parse_final_hidden_taps(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith("[qwen36-final-hidden-tap]"):
            continue
        fields = {
            key: parse_number(value)
            for key, value in parse_key_values(line).items()
        }
        rows.append(fields)
    return rows


def parse_logits_taps(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith("[qwen36-logits-tap]"):
            continue
        fields = {
            key: parse_number(value)
            for key, value in parse_key_values(line).items()
        }
        rows.append(fields)
    return rows


def parse_layer_output_taps(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith("[qwen36-layer-output-tap]"):
            continue
        fields = {
            key: parse_number(value)
            for key, value in parse_key_values(line).items()
        }
        rows.append(fields)
    return rows


def parse_layer_output_delta_taps(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith("[qwen36-layer-output-delta-tap]"):
            continue
        fields = {
            key: parse_number(value)
            for key, value in parse_key_values(line).items()
        }
        rows.append(fields)
    return rows


def parse_decode_batch_route_snapshots(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith("[qwen36-decode-batch-route-snapshot]"):
            continue
        fields = {
            key: parse_number(value)
            for key, value in parse_key_values(line).items()
        }
        rows.append(fields)
    return rows


def summarize_router_parity_taps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    taps = [
        tap
        for row in rows
        for tap in (row.get("router_parity_taps") or [])
    ]
    paths = sorted({str(tap.get("router_path") or "-") for tap in taps})
    mismatches = [tap for tap in taps if not bool(tap.get("topk_idx_match"))]
    return {
        "tap_count": len(taps),
        "mismatch_count": len(mismatches),
        "paths": paths,
        "max_h_norm_abs": max(
            (float(tap.get("h_norm_max_abs") or 0.0) for tap in taps),
            default=0.0,
        ),
        "max_logits_abs": max(
            (float(tap.get("logits_max_abs") or 0.0) for tap in taps),
            default=0.0,
        ),
        "max_topk_weight_abs": max(
            (float(tap.get("topk_weight_max_abs") or 0.0) for tap in taps),
            default=0.0,
        ),
        "mismatch_examples": [
            {
                "path": tap.get("router_path") or "-",
                "layer": tap.get("layer"),
                "first_mismatch": tap.get("topk_first_mismatch"),
                "host_idx": tap.get("host_idx"),
                "metal_idx": tap.get("workspace_idx", tap.get("output_idx")),
                "host_top_logit_idx": tap.get("host_top_logit_idx"),
                "metal_top_logit_idx": tap.get("metal_top_logit_idx"),
            }
            for tap in mismatches[:20]
        ],
    }


def summarize_shared_parity_taps(
    rows: list[dict[str, Any]],
    field: str = "shared_parity_taps",
) -> dict[str, Any]:
    taps = [
        tap
        for row in rows
        for tap in (row.get(field) or [])
    ]
    paths = sorted({str(tap.get("shared_path") or "-") for tap in taps})

    def max_field(name: str) -> float:
        return max((float(tap.get(name) or 0.0) for tap in taps), default=0.0)

    ranked = sorted(
        taps,
        key=lambda tap: max(
            float(tap.get("shared_gate_max_abs") or 0.0),
            float(tap.get("shared_up_max_abs") or 0.0),
            float(tap.get("shared_mid_max_abs") or 0.0),
            float(tap.get("shared_out_max_abs") or 0.0),
        ),
        reverse=True,
    )
    return {
        "tap_count": len(taps),
        "paths": paths,
        "max_shared_gate_abs": max_field("shared_gate_max_abs"),
        "max_shared_up_abs": max_field("shared_up_max_abs"),
        "max_shared_mid_abs": max_field("shared_mid_max_abs"),
        "max_shared_scalar_abs": max_field("shared_scalar_abs"),
        "max_shared_out_abs": max_field("shared_out_max_abs"),
        "worst_examples": [
            {
                "path": tap.get("shared_path") or "-",
                "layer": tap.get("layer"),
                "shared_gate_max_abs": tap.get("shared_gate_max_abs"),
                "shared_up_max_abs": tap.get("shared_up_max_abs"),
                "shared_mid_max_abs": tap.get("shared_mid_max_abs"),
                "shared_scalar_abs": tap.get("shared_scalar_abs"),
                "shared_out_max_abs": tap.get("shared_out_max_abs"),
                "shared_out_argmax": tap.get("shared_out_argmax"),
            }
            for tap in ranked[:20]
        ],
    }


def summarize_routed_parity_taps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    taps = [
        tap
        for row in rows
        for tap in (row.get("decode_batch_routed_parity_taps") or [])
    ]
    paths = sorted({str(tap.get("router_path") or "-") for tap in taps})
    mismatches = [tap for tap in taps if not bool(tap.get("topk_idx_match"))]

    def max_field(name: str) -> float:
        return max((float(tap.get(name) or 0.0) for tap in taps), default=0.0)

    ranked = sorted(
        taps,
        key=lambda tap: max(
            float(tap.get("expert_mid_max_abs") or 0.0),
            float(tap.get("moe_out_max_abs") or 0.0),
            float(tap.get("final_out_max_abs") or 0.0),
            float(tap.get("topk_weight_max_abs") or 0.0),
        ),
        reverse=True,
    )
    return {
        "tap_count": len(taps),
        "mismatch_count": len(mismatches),
        "paths": paths,
        "max_topk_weight_abs": max_field("topk_weight_max_abs"),
        "max_expert_mid_abs": max_field("expert_mid_max_abs"),
        "max_moe_out_abs": max_field("moe_out_max_abs"),
        "max_final_out_abs": max_field("final_out_max_abs"),
        "worst_examples": [
            {
                "path": tap.get("router_path") or "-",
                "layer": tap.get("layer"),
                "topk_idx_match": tap.get("topk_idx_match"),
                "topk_weight_max_abs": tap.get("topk_weight_max_abs"),
                "expert_mid_max_abs": tap.get("expert_mid_max_abs"),
                "expert_mid_argmax": tap.get("expert_mid_argmax"),
                "moe_out_max_abs": tap.get("moe_out_max_abs"),
                "moe_out_argmax": tap.get("moe_out_argmax"),
                "final_out_max_abs": tap.get("final_out_max_abs"),
                "final_out_argmax": tap.get("final_out_argmax"),
            }
            for tap in ranked[:20]
        ],
    }


def iter_downstream_taps(
    rows: list[dict[str, Any]],
    field: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (row, tap)
        for row in rows
        for tap in (row.get(field) or [])
    ]


def downstream_tap_comparisons(
    rows: list[dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    baselines: dict[tuple[str, int], dict[str, Any]] = {}
    for row, tap in iter_downstream_taps(rows, field):
        if row.get("mode") != "default":
            continue
        key = (str(row.get("prompt_id", "")), int(tap.get("gen_index", 0)))
        baselines[key] = tap

    comparisons: list[dict[str, Any]] = []
    for row, tap in iter_downstream_taps(rows, field):
        if row.get("mode") == "default":
            continue
        key = (str(row.get("prompt_id", "")), int(tap.get("gen_index", 0)))
        baseline = baselines.get(key)
        comparison: dict[str, Any] = {
            "prompt_id": key[0],
            "mode": row.get("mode"),
            "gen_index": key[1],
            "path": tap.get("path", "-"),
            "checksum": tap.get("checksum"),
            "baseline_checksum": None if baseline is None else baseline.get("checksum"),
            "checksum_match": None if baseline is None else tap.get("checksum") == baseline.get("checksum"),
            "status": "missing_baseline" if baseline is None else "ok",
        }
        if field == "logits_taps":
            comparison["top1_idx"] = tap.get("top1_idx")
            comparison["baseline_top1_idx"] = None if baseline is None else baseline.get("top1_idx")
            comparison["top1_match"] = (
                None if baseline is None else tap.get("top1_idx") == baseline.get("top1_idx")
            )
        comparisons.append(comparison)
    return comparisons


def summarize_final_hidden_taps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    taps = [tap for _, tap in iter_downstream_taps(rows, "final_hidden_taps")]
    comparisons = downstream_tap_comparisons(rows, "final_hidden_taps")
    mismatches = [item for item in comparisons if item.get("checksum_match") is False]
    return {
        "tap_count": len(taps),
        "paths": sorted({str(tap.get("path") or "-") for tap in taps}),
        "comparison_count": len(comparisons),
        "checksum_mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "comparisons": comparisons,
    }


def summarize_logits_taps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    taps = [tap for _, tap in iter_downstream_taps(rows, "logits_taps")]
    comparisons = downstream_tap_comparisons(rows, "logits_taps")
    checksum_mismatches = [
        item for item in comparisons if item.get("checksum_match") is False
    ]
    top1_mismatches = [
        item for item in comparisons if item.get("top1_match") is False
    ]
    return {
        "tap_count": len(taps),
        "paths": sorted({str(tap.get("path") or "-") for tap in taps}),
        "comparison_count": len(comparisons),
        "checksum_mismatch_count": len(checksum_mismatches),
        "top1_mismatch_count": len(top1_mismatches),
        "first_checksum_mismatch": checksum_mismatches[0] if checksum_mismatches else None,
        "first_top1_mismatch": top1_mismatches[0] if top1_mismatches else None,
        "comparisons": comparisons,
    }


def iter_layer_output_taps(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (row, tap)
        for row in rows
        for tap in (row.get("layer_output_taps") or [])
    ]


def compare_layer_output_taps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row, tap in iter_layer_output_taps(rows):
        if row.get("mode") != "default":
            continue
        key = (
            str(row.get("prompt_id", "")),
            int(tap.get("position", 0)),
            int(tap.get("layer", -1)),
            str(tap.get("phase", "-")),
        )
        baselines[key] = tap

    comparisons: list[dict[str, Any]] = []
    for row, tap in iter_layer_output_taps(rows):
        if row.get("mode") == "default":
            continue
        key = (
            str(row.get("prompt_id", "")),
            int(tap.get("position", 0)),
            int(tap.get("layer", -1)),
            str(tap.get("phase", "-")),
        )
        baseline = baselines.get(key)
        comparisons.append(
            {
                "prompt_id": key[0],
                "mode": row.get("mode"),
                "position": key[1],
                "layer": key[2],
                "phase": key[3],
                "path": tap.get("path", "-"),
                "checksum": tap.get("checksum"),
                "baseline_checksum": None
                if baseline is None
                else baseline.get("checksum"),
                "checksum_match": None
                if baseline is None
                else tap.get("checksum") == baseline.get("checksum"),
                "status": "missing_baseline" if baseline is None else "ok",
            }
        )
    return comparisons


def summarize_layer_output_taps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    taps = [tap for _, tap in iter_layer_output_taps(rows)]
    comparisons = compare_layer_output_taps(rows)
    mismatches = [item for item in comparisons if item.get("checksum_match") is False]
    return {
        "tap_count": len(taps),
        "paths": sorted({str(tap.get("path") or "-") for tap in taps}),
        "comparison_count": len(comparisons),
        "checksum_mismatch_count": len(mismatches),
        "first_mismatch": mismatches[0] if mismatches else None,
        "comparisons": comparisons,
    }


def bf16_hex_to_bits(raw: Any) -> list[int]:
    if raw is None:
        return []
    return [
        int(part, 16)
        for part in str(raw).split(",")
        if part
    ]


def bf16_bits_to_f32(bits: int) -> float:
    raw = (bits & 0xFFFF) << 16
    return struct.unpack(">f", raw.to_bytes(4, "big"))[0]


def bf16_order_key(bits: int) -> int:
    bits &= 0xFFFF
    if bits & 0x8000:
        return (~bits) & 0xFFFF
    return bits | 0x8000


def compare_bf16_hex_values(expected_raw: Any, got_raw: Any) -> dict[str, Any]:
    expected_bits = bf16_hex_to_bits(expected_raw)
    got_bits = bf16_hex_to_bits(got_raw)
    limit = min(len(expected_bits), len(got_bits))
    max_abs_delta = 0.0
    max_abs_idx = 0
    max_ulp_delta = 0
    max_ulp_idx = 0
    differing_elems = abs(len(expected_bits) - len(got_bits))
    for idx in range(limit):
        a_bits = expected_bits[idx]
        b_bits = got_bits[idx]
        if a_bits != b_bits:
            differing_elems += 1
        delta = abs(bf16_bits_to_f32(a_bits) - bf16_bits_to_f32(b_bits))
        if delta > max_abs_delta:
            max_abs_delta = delta
            max_abs_idx = idx
        ulp_delta = abs(bf16_order_key(a_bits) - bf16_order_key(b_bits))
        if ulp_delta > max_ulp_delta:
            max_ulp_delta = ulp_delta
            max_ulp_idx = idx
    return {
        "elems": limit,
        "length_match": len(expected_bits) == len(got_bits),
        "differing_elems": differing_elems,
        "max_abs_delta": max_abs_delta,
        "max_abs_delta_idx": max_abs_idx,
        "max_ulp_delta": max_ulp_delta,
        "max_ulp_delta_idx": max_ulp_idx,
        "baseline_bf16_at_max_abs": (
            f"{expected_bits[max_abs_idx]:04x}" if expected_bits and max_abs_idx < len(expected_bits) else None
        ),
        "candidate_bf16_at_max_abs": (
            f"{got_bits[max_abs_idx]:04x}" if got_bits and max_abs_idx < len(got_bits) else None
        ),
        "baseline_value_at_max_abs": (
            bf16_bits_to_f32(expected_bits[max_abs_idx])
            if expected_bits and max_abs_idx < len(expected_bits)
            else None
        ),
        "candidate_value_at_max_abs": (
            bf16_bits_to_f32(got_bits[max_abs_idx])
            if got_bits and max_abs_idx < len(got_bits)
            else None
        ),
    }


def iter_layer_output_delta_taps(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (row, tap)
        for row in rows
        for tap in (row.get("layer_output_delta_taps") or [])
    ]


def compare_layer_output_delta_taps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for row, tap in iter_layer_output_delta_taps(rows):
        if row.get("mode") != "default":
            continue
        key = (
            str(row.get("prompt_id", "")),
            int(tap.get("position", 0)),
            int(tap.get("layer", -1)),
            str(tap.get("phase", "-")),
        )
        baselines[key] = tap

    comparisons: list[dict[str, Any]] = []
    for row, tap in iter_layer_output_delta_taps(rows):
        if row.get("mode") == "default":
            continue
        key = (
            str(row.get("prompt_id", "")),
            int(tap.get("position", 0)),
            int(tap.get("layer", -1)),
            str(tap.get("phase", "-")),
        )
        baseline = baselines.get(key)
        item = {
            "prompt_id": key[0],
            "mode": row.get("mode"),
            "position": key[1],
            "layer": key[2],
            "phase": key[3],
            "path": tap.get("path", "-"),
            "checksum": tap.get("checksum"),
            "baseline_checksum": None if baseline is None else baseline.get("checksum"),
            "checksum_match": None
            if baseline is None
            else tap.get("checksum") == baseline.get("checksum"),
            "status": "missing_baseline" if baseline is None else "ok",
        }
        if baseline is not None:
            item.update(compare_bf16_hex_values(baseline.get("bf16"), tap.get("bf16")))
        comparisons.append(item)
    return comparisons


def summarize_layer_output_delta_taps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    taps = [tap for _, tap in iter_layer_output_delta_taps(rows)]
    comparisons = compare_layer_output_delta_taps(rows)
    mismatches = [item for item in comparisons if item.get("checksum_match") is False]
    return {
        "tap_count": len(taps),
        "paths": sorted({str(tap.get("path") or "-") for tap in taps}),
        "comparison_count": len(comparisons),
        "checksum_mismatch_count": len(mismatches),
        "max_abs_delta": max(
            (float(item.get("max_abs_delta") or 0.0) for item in comparisons),
            default=0.0,
        ),
        "max_ulp_delta": max(
            (int(item.get("max_ulp_delta") or 0) for item in comparisons),
            default=0,
        ),
        "max_differing_elems": max(
            (int(item.get("differing_elems") or 0) for item in comparisons),
            default=0,
        ),
        "first_mismatch": mismatches[0] if mismatches else None,
        "comparisons": comparisons,
    }


def select_router_parity_tap_rows(
    tap_rows: list[tuple[dict[str, Any], dict[str, Any]]],
    limit: int = 40,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(tap_rows) <= limit:
        return tap_rows

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    selected_taps: set[int] = set()

    def add(pair: tuple[dict[str, Any], dict[str, Any]]) -> None:
        tap_id = id(pair[1])
        if tap_id not in selected_taps and len(selected) < limit:
            selected.append(pair)
            selected_taps.add(tap_id)

    for pair in tap_rows:
        if not bool(pair[1].get("topk_idx_match")):
            add(pair)

    grouped: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for pair in tap_rows:
        row, tap = pair
        key = (
            str(row.get("prompt_id", "")),
            str(row.get("mode", "")),
            str(tap.get("router_path", "-")),
        )
        grouped.setdefault(key, []).append(pair)

    remaining = max(0, limit - len(selected))
    quota = max(1, remaining // max(1, len(grouped)))
    for group in grouped.values():
        if len(selected) >= limit:
            break
        if len(group) <= quota:
            candidates = group
        elif quota == 1:
            candidates = [group[0]]
        else:
            candidates = group[: quota - 1] + [group[-1]]
        for pair in candidates:
            add(pair)

    for pair in tap_rows:
        if len(selected) >= limit:
            break
        add(pair)

    return selected


def select_shared_parity_tap_rows(
    tap_rows: list[tuple[dict[str, Any], dict[str, Any]]],
    limit: int = 40,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return sorted(
        tap_rows,
        key=lambda pair: max(
            float(pair[1].get("shared_gate_max_abs") or 0.0),
            float(pair[1].get("shared_up_max_abs") or 0.0),
            float(pair[1].get("shared_mid_max_abs") or 0.0),
            float(pair[1].get("shared_scalar_abs") or 0.0),
            float(pair[1].get("shared_out_max_abs") or 0.0),
        ),
        reverse=True,
    )[:limit]


def iter_decode_batch_route_snapshots(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [
        (row, snapshot)
        for row in rows
        for snapshot in (row.get("decode_batch_route_snapshots") or [])
    ]


def decode_batch_route_snapshot_comparisons(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs = iter_decode_batch_route_snapshots(rows)
    reference_by_key: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    for row, snapshot in pairs:
        prompt = str(row.get("prompt_id", ""))
        call = int(snapshot.get("call", 0))
        key = (prompt, call)
        current = reference_by_key.get(key)
        path = str(snapshot.get("router_path") or "")
        if current is None or (current[1].get("router_path") == "simd" and path != "simd"):
            reference_by_key[key] = (row, snapshot)

    comparisons: list[dict[str, Any]] = []
    for row, snapshot in pairs:
        prompt = str(row.get("prompt_id", ""))
        call = int(snapshot.get("call", 0))
        reference = reference_by_key.get((prompt, call))
        ref_row, ref_snapshot = reference if reference is not None else ({}, {})
        checksum_match = snapshot.get("checksum") == ref_snapshot.get("checksum")
        routes_match = snapshot.get("routes") == ref_snapshot.get("routes")
        comparisons.append(
            {
                "prompt_id": prompt,
                "mode": row.get("mode"),
                "path": snapshot.get("router_path") or "-",
                "call": call,
                "checksum": snapshot.get("checksum"),
                "routes": snapshot.get("routes"),
                "reference_mode": ref_row.get("mode"),
                "reference_path": ref_snapshot.get("router_path") or "-",
                "reference_checksum": ref_snapshot.get("checksum"),
                "match_reference": checksum_match and routes_match,
            }
        )
    return comparisons


def summarize_decode_batch_route_snapshots(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = iter_decode_batch_route_snapshots(rows)
    snapshots = [snapshot for _, snapshot in pairs]
    comparisons = decode_batch_route_snapshot_comparisons(rows)
    mismatches = [
        comparison
        for comparison in comparisons
        if not bool(comparison.get("match_reference"))
    ]
    return {
        "snapshot_count": len(snapshots),
        "mismatch_count": len(mismatches),
        "paths": sorted({str(snapshot.get("router_path") or "-") for snapshot in snapshots}),
        "max_captured_layers": max(
            (int(snapshot.get("captured_layers") or 0) for snapshot in snapshots),
            default=0,
        ),
        "mismatch_examples": mismatches[:20],
    }


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
    if getattr(args, "downstream_parity_tap", False):
        overrides["SUPERSONIC_QWEN36_DOWNSTREAM_PARITY_TAP"] = "1"
    if getattr(args, "layer_output_tap", False):
        overrides["SUPERSONIC_QWEN36_LAYER_OUTPUT_TAP"] = "1"
    if getattr(args, "layer_output_delta_tap", False):
        overrides["SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP"] = "1"
        layer = getattr(args, "layer_output_delta_layer", None)
        if layer is not None:
            overrides["SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_LAYER"] = str(layer)
        position = getattr(args, "layer_output_delta_position", None)
        if position is not None:
            overrides["SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_POSITION"] = str(position)
        phase = getattr(args, "layer_output_delta_phase", None)
        if phase:
            overrides["SUPERSONIC_QWEN36_LAYER_OUTPUT_DELTA_TAP_PHASE"] = str(phase)
    if getattr(args, "router_parity_tap", False):
        overrides["SUPERSONIC_METAL_QWEN36_FFN_ROUTER_STAGE5_PARITY_TAP"] = "1"
        max_calls = getattr(args, "router_parity_tap_max_calls", None)
        if max_calls:
            overrides["SUPERSONIC_METAL_QWEN36_FFN_ROUTER_STAGE5_PARITY_TAP_MAX_CALLS"] = str(
                max_calls
            )
    if getattr(args, "shared_parity_tap", False):
        overrides["SUPERSONIC_METAL_QWEN36_FFN_SHARED_STAGE5_PARITY_TAP"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SHARED_STAGE5_PARITY_TAP"] = "1"
        max_calls = getattr(args, "shared_parity_tap_max_calls", None)
        if max_calls:
            overrides["SUPERSONIC_METAL_QWEN36_FFN_SHARED_STAGE5_PARITY_TAP_MAX_CALLS"] = str(
                max_calls
            )
            overrides[
                "SUPERSONIC_METAL_QWEN36_DECODE_BATCH_SHARED_STAGE5_PARITY_TAP_MAX_CALLS"
            ] = str(max_calls)
    if getattr(args, "routed_parity_tap", False):
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTED_STAGE5_PARITY_TAP"] = "1"
        max_calls = getattr(args, "routed_parity_tap_max_calls", None)
        if max_calls:
            overrides[
                "SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTED_STAGE5_PARITY_TAP_MAX_CALLS"
            ] = str(max_calls)
    if getattr(args, "decode_batch_route_snapshot", False):
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTE_SNAPSHOT"] = "1"
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
    elif mode == "full-stage5-router-simd":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"] = "1"
    elif mode == "full-stage5-router-batch":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"] = "1"
    elif mode == "full-stage5-router-simd-batch":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"] = "1"
    elif mode == SHARED_TILED_BATCH_SIMD_MODE:
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_TILED"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"] = "1"
    elif mode == SHARED_GATE_UP_TILED_BATCH_SIMD_MODE:
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_GATE_UP_TILED"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"] = "1"
    elif mode == SHARED_SCALAR_SIMD_BATCH_SIMD_MODE:
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_SCALAR_SIMD"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"] = "1"
    elif mode == SHARED_DOWN_TILED_BATCH_SIMD_MODE:
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_DOWN_TILED"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"] = "1"
    elif mode == "full-stage5-router-batch-deferred-phases":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_PHASES_DEFERRED"] = "1"
    elif mode == "full-stage5-router-simd-batch-deferred-phases":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_PHASES_DEFERRED"] = "1"
    elif mode == SHARED_TILED_DEFERRED_SIMD_MODE:
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_TILED"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_PHASES_DEFERRED"] = "1"
    elif mode == "full-stage5-router-batch-phases":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_PHASES"] = "1"
    elif mode == "full-stage5-router-batch-ffn-phases":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"] = "1"
        overrides["SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_FFN_PHASES"] = "1"
    elif mode == "full-stage5-router-simd-batch-phases":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_PHASES"] = "1"
    elif mode == "full-stage5-router-simd-batch-ffn-phases":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"] = "1"
        overrides["SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_FFN_PHASES"] = "1"
    elif mode == SHARED_TILED_FFN_PHASE_SIMD_MODE:
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_STAGE5_SIMD"] = "1"
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_SHARED_TILED"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH"] = "1"
        overrides["SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DECODE_BATCH_PROFILE_FFN_PHASES"] = "1"
    elif mode == "router-defer-wait":
        overrides["SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5_ROUTER"] = "1"
        overrides["SUPERSONIC_METAL_QWEN36_DEFER_FFN_ROUTER_STAGE5_WAIT"] = "1"
    return overrides


def mode_emits_stage_timings(mode: str) -> bool:
    return mode not in BATCH_FAST_PROFILE_MODES


def build_command(args: argparse.Namespace, prompt: str, mode: str) -> list[str]:
    command = [
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
        "--emit-generated-json",
    ]
    if mode_emits_stage_timings(mode):
        command.append("--emit-stage-timings")
    return command


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


def profile_op_total_exact(profile: dict[str, Any] | None, op_name: str) -> float | None:
    op_name = op_name.lower()
    return profile_op_total_where(
        profile,
        lambda entry: str(entry.get("op") or "").lower() == op_name,
    )


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


def decode_batch_phase_gpu_ms(row: dict[str, Any], phase: str) -> float | None:
    return profile_op_total(
        row.get("metal_profile"),
        f"command_buffer_gpu:qwen36_decode_batch_{phase}",
    )


def decode_batch_gpu_ms(row: dict[str, Any]) -> float | None:
    return profile_op_total_exact(
        row.get("metal_profile"),
        "command_buffer_gpu:qwen36_decode_batch",
    )


def batch_ffn_subphase_gpu_ms(row: dict[str, Any], field: str) -> float | None:
    total = 0.0
    matched = False
    profile = row.get("metal_profile")
    if not profile:
        return None
    needles = {needle.lower() for needle in BATCH_FFN_PHASE_GPU_FIELDS[field]}
    for entry in profile.get("entries") or []:
        op = str(entry.get("op") or "").lower()
        if op not in needles:
            continue
        matched = True
        total += float(entry.get("total_ms") or 0.0)
    return total if matched else None


def batch_ffn_subphase_total_gpu_ms(row: dict[str, Any]) -> float | None:
    total = 0.0
    matched = False
    for field in BATCH_FFN_PHASE_GPU_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        matched = True
        total += float(value)
    return total if matched else None


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
        row["decode_batch_gpu_ms"] = decode_batch_gpu_ms(row)
        row["decode_batch_linear_gpu_ms"] = decode_batch_phase_gpu_ms(row, "linear_attn")
        row["decode_batch_ffn_gpu_ms"] = decode_batch_phase_gpu_ms(row, "ffn")
        for field in BATCH_FFN_PHASE_GPU_FIELDS:
            row[field] = batch_ffn_subphase_gpu_ms(row, field)
        if row["decode_batch_ffn_gpu_ms"] is None:
            row["decode_batch_ffn_gpu_ms"] = batch_ffn_subphase_total_gpu_ms(row)
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
    stage_timings_enabled = mode_emits_stage_timings(mode)
    command = build_command(args, prompt, mode)
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
            "stage_timings_enabled": stage_timings_enabled,
            "generated_ids": parse_generated_ids(output),
            "result": parse_result(output),
            "stage_timings": parse_stage_timings(output),
            "chain_breakdown": parse_chain_breakdown(output),
            "lifecycle_timings": parse_lifecycle_timings(output),
            "metal_profile": parse_profile(output, "[metal-profile]", "[metal-profile-op]"),
            "hal_profile": parse_profile(output, "[hal-profile]", "[hal-profile-op]"),
            "router_parity_taps": parse_router_parity_taps(output),
            "shared_parity_taps": parse_shared_parity_taps(output),
            "decode_batch_shared_parity_taps": parse_decode_batch_shared_parity_taps(output),
            "decode_batch_routed_parity_taps": parse_decode_batch_routed_parity_taps(output),
            "final_hidden_taps": parse_final_hidden_taps(output),
            "logits_taps": parse_logits_taps(output),
            "layer_output_taps": parse_layer_output_taps(output),
            "layer_output_delta_taps": parse_layer_output_delta_taps(output),
            "decode_batch_route_snapshots": parse_decode_batch_route_snapshots(output),
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
            "stage_timings_enabled": stage_timings_enabled,
            "generated_ids": [],
            "result": {},
            "stage_timings": {},
            "chain_breakdown": {},
            "lifecycle_timings": {},
            "metal_profile": parse_profile(output, "[metal-profile]", "[metal-profile-op]"),
            "hal_profile": parse_profile(output, "[hal-profile]", "[hal-profile-op]"),
            "router_parity_taps": parse_router_parity_taps(output),
            "shared_parity_taps": parse_shared_parity_taps(output),
            "decode_batch_shared_parity_taps": parse_decode_batch_shared_parity_taps(output),
            "decode_batch_routed_parity_taps": parse_decode_batch_routed_parity_taps(output),
            "final_hidden_taps": parse_final_hidden_taps(output),
            "logits_taps": parse_logits_taps(output),
            "layer_output_taps": parse_layer_output_taps(output),
            "layer_output_delta_taps": parse_layer_output_delta_taps(output),
            "decode_batch_route_snapshots": parse_decode_batch_route_snapshots(output),
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


def build_decode_batch_coarse_comparison(
    rows: list[dict[str, Any]],
    modes: list[str],
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
    comparisons: list[dict[str, Any]] = []
    missing_modes = [
        mode
        for mode in (COARSE_BATCH_SERIAL_MODE, COARSE_BATCH_SIMD_MODE)
        if mode not in modes
    ]
    for prompt_id in prompt_ids:
        baseline = rows_by_key.get((prompt_id, "default"))
        serial = rows_by_key.get((prompt_id, COARSE_BATCH_SERIAL_MODE))
        simd = rows_by_key.get((prompt_id, COARSE_BATCH_SIMD_MODE))
        if serial is None or simd is None:
            comparisons.append(
                {
                    "prompt_id": prompt_id,
                    "status": "missing_mode",
                    "missing_modes": [
                        mode
                        for mode, row in (
                            (COARSE_BATCH_SERIAL_MODE, serial),
                            (COARSE_BATCH_SIMD_MODE, simd),
                        )
                        if row is None
                    ],
                }
            )
            continue
        serial_decode = row_number(serial, "result", "decode_ms")
        simd_decode = row_number(simd, "result", "decode_ms")
        serial_gpu = serial.get("decode_batch_gpu_ms")
        simd_gpu = simd.get("decode_batch_gpu_ms")
        serial_wait = serial.get("command_buffer_wait_ms")
        simd_wait = simd.get("command_buffer_wait_ms")
        generated_ids_match_default = (
            None
            if baseline is None
            else (simd.get("generated_ids") or []) == (baseline.get("generated_ids") or [])
        )
        generated_ids_match_serial = (simd.get("generated_ids") or []) == (
            serial.get("generated_ids") or []
        )
        decode_ratio = safe_ratio(simd_decode, serial_decode)
        gpu_ratio = safe_ratio(simd_gpu, serial_gpu)
        wait_ratio = safe_ratio(simd_wait, serial_wait)
        wait_gpu_ratio = safe_ratio(simd_wait, simd_gpu)
        decode_gpu_ratio = safe_ratio(simd_decode, simd_gpu)
        if simd.get("status") != "ok" or serial.get("status") != "ok":
            blocker = "missing_ok_row"
        elif generated_ids_match_default is False or not generated_ids_match_serial:
            blocker = "correctness"
        elif wait_gpu_ratio is not None and wait_gpu_ratio > DEFAULT_MAX_WAIT_GPU_RATIO:
            blocker = "batch_wait_or_submit_overhead"
        elif decode_gpu_ratio is not None and decode_gpu_ratio > DEFAULT_MAX_FUSED_WALL_GPU_RATIO:
            blocker = "batch_wall_overhead"
        elif gpu_ratio is not None and gpu_ratio >= 0.999:
            blocker = "router_gpu_not_improved"
        else:
            blocker = "gpu_work_or_profile_needed"
        comparisons.append(
            {
                "prompt_id": prompt_id,
                "status": "ok",
                "generated_ids_match_default": generated_ids_match_default,
                "generated_ids_match_serial": generated_ids_match_serial,
                "serial_decode_ms": serial_decode,
                "simd_decode_ms": simd_decode,
                "decode_ratio": decode_ratio,
                "serial_decode_batch_gpu_ms": serial_gpu,
                "simd_decode_batch_gpu_ms": simd_gpu,
                "decode_batch_gpu_ratio": gpu_ratio,
                "serial_command_buffer_wait_ms": serial_wait,
                "simd_command_buffer_wait_ms": simd_wait,
                "command_buffer_wait_ratio": wait_ratio,
                "simd_wait_gpu_ratio": wait_gpu_ratio,
                "simd_decode_gpu_ratio": decode_gpu_ratio,
                "blocker": blocker,
            }
        )

    usable = [
        item
        for item in comparisons
        if item.get("status") == "ok"
    ]
    blockers = sorted({str(item.get("blocker")) for item in usable})
    mismatches = [
        item
        for item in usable
        if item.get("generated_ids_match_default") is False
        or item.get("generated_ids_match_serial") is False
    ]
    improved_decode = [
        item
        for item in usable
        if item.get("decode_ratio") is not None and float(item["decode_ratio"]) < 1.0
    ]
    improved_gpu = [
        item
        for item in usable
        if item.get("decode_batch_gpu_ratio") is not None
        and float(item["decode_batch_gpu_ratio"]) < 1.0
    ]
    if missing_modes:
        recommendation = "run_decode_batch_coarse_simd_sweep"
        reason = "serial and SIMD coarse decode-batch rows are both required"
    elif mismatches:
        recommendation = "fix_decode_batch_simd_correctness"
        reason = "SIMD coarse batch row does not match the default or serial generated IDs"
    elif usable and len(improved_decode) == len(usable) and len(improved_gpu) == len(usable):
        if any(
            item.get("blocker") in {"batch_wait_or_submit_overhead", "batch_wall_overhead"}
            for item in usable
        ):
            recommendation = "target_decode_batch_wait_or_wall_overhead"
            reason = "SIMD improves the coarse batch GPU label, but wall/wait ratios dominate"
        else:
            recommendation = "keep_simd_router_enabled_then_target_remaining_gpu_work"
            reason = "SIMD improves coarse batch decode and GPU labels without a wall/wait blocker"
    elif usable:
        recommendation = "keep_simd_router_as_attribution_only"
        reason = "SIMD coarse batch did not improve both decode wall time and batch GPU time"
    else:
        recommendation = "run_decode_batch_coarse_simd_sweep"
        reason = "no usable coarse decode-batch comparison rows were found"
    return {
        "serial_mode": COARSE_BATCH_SERIAL_MODE,
        "simd_mode": COARSE_BATCH_SIMD_MODE,
        "available": not missing_modes and bool(usable),
        "missing_modes": missing_modes,
        "comparison_count": len(usable),
        "mismatch_count": len(mismatches),
        "blockers": blockers,
        "recommendation": recommendation,
        "reason": reason,
        "comparisons": comparisons,
    }


def build_decode_batch_deferred_phase_summary(
    rows: list[dict[str, Any]],
    modes: list[str],
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
    rows_out: list[dict[str, Any]] = []
    missing_modes = [
        mode
        for mode in (DEFERRED_BATCH_SERIAL_MODE, DEFERRED_BATCH_SIMD_MODE)
        if mode not in modes
    ]
    for prompt_id in prompt_ids:
        baseline = rows_by_key.get((prompt_id, "default"))
        for mode in (DEFERRED_BATCH_SERIAL_MODE, DEFERRED_BATCH_SIMD_MODE):
            row = rows_by_key.get((prompt_id, mode))
            if row is None:
                continue
            linear_gpu = row.get("decode_batch_linear_gpu_ms")
            ffn_gpu = row.get("decode_batch_ffn_gpu_ms")
            total_gpu = None
            if linear_gpu is not None or ffn_gpu is not None:
                total_gpu = float(linear_gpu or 0.0) + float(ffn_gpu or 0.0)
            ffn_share = safe_ratio(ffn_gpu, total_gpu)
            wait_ms = row.get("command_buffer_wait_ms")
            wait_gpu_ratio = safe_ratio(wait_ms, total_gpu)
            generated_ids_match_default = (
                None
                if baseline is None
                else (row.get("generated_ids") or []) == (baseline.get("generated_ids") or [])
            )
            if row.get("status") != "ok":
                blocker = "missing_ok_row"
            elif generated_ids_match_default is False:
                blocker = "correctness"
            elif total_gpu is None:
                blocker = "missing_deferred_phase_profile"
            elif ffn_share is not None and ffn_share >= 0.60:
                blocker = "ffn_gpu_work"
            elif linear_gpu is not None and ffn_gpu is not None and linear_gpu > ffn_gpu:
                blocker = "linear_gpu_work"
            elif wait_gpu_ratio is not None and wait_gpu_ratio > DEFAULT_MAX_WAIT_GPU_RATIO:
                blocker = "wait_or_submit_overhead"
            else:
                blocker = "mixed_gpu_work"
            rows_out.append(
                {
                    "prompt_id": prompt_id,
                    "mode": mode,
                    "router_path": "simd" if "simd" in mode else "serial",
                    "status": row.get("status"),
                    "generated_ids_match_default": generated_ids_match_default,
                    "decode_ms": row_number(row, "result", "decode_ms"),
                    "linear_gpu_ms": linear_gpu,
                    "ffn_gpu_ms": ffn_gpu,
                    "total_phase_gpu_ms": total_gpu,
                    "ffn_share": ffn_share,
                    "command_buffer_wait_ms": wait_ms,
                    "wait_gpu_ratio": wait_gpu_ratio,
                    "blocker": blocker,
                }
            )

    usable = [row for row in rows_out if row.get("status") == "ok"]
    blockers = sorted({str(row.get("blocker")) for row in usable})
    if missing_modes:
        recommendation = "run_decode_batch_deferred_phase_sweep"
        reason = "serial and SIMD deferred phase rows are both required"
    elif any(row.get("blocker") == "correctness" for row in usable):
        recommendation = "fix_decode_batch_deferred_phase_correctness"
        reason = "a deferred phase row does not match default generated IDs"
    elif any(row.get("blocker") == "ffn_gpu_work" for row in usable):
        recommendation = "target_batched_ffn_gpu_work"
        reason = "deferred non-waiting phase labels show FFN dominates the coarse batch GPU work"
    elif any(row.get("blocker") == "linear_gpu_work" for row in usable):
        recommendation = "target_batched_linear_gpu_work"
        reason = "deferred non-waiting phase labels show linear attention dominates the coarse batch GPU work"
    elif any(row.get("blocker") == "wait_or_submit_overhead" for row in usable):
        recommendation = "target_decode_batch_wait_or_submit_overhead"
        reason = "deferred phase labels are small relative to command-buffer wait"
    elif usable:
        recommendation = "inspect_decode_batch_mixed_gpu_work"
        reason = "deferred phase labels do not point at one dominant component"
    else:
        recommendation = "run_decode_batch_deferred_phase_sweep"
        reason = "no usable deferred phase rows were found"
    return {
        "serial_mode": DEFERRED_BATCH_SERIAL_MODE,
        "simd_mode": DEFERRED_BATCH_SIMD_MODE,
        "available": not missing_modes and bool(usable),
        "missing_modes": missing_modes,
        "row_count": len(usable),
        "blockers": blockers,
        "recommendation": recommendation,
        "reason": reason,
        "rows": rows_out,
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
    summary["decode_batch_coarse"] = build_decode_batch_coarse_comparison(rows, modes)
    summary["decode_batch_deferred_phase"] = build_decode_batch_deferred_phase_summary(rows, modes)
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
    summary = summarize_with_gate(
        rows,
        modes,
        args.promotion_max_headline_ratio,
        args.promotion_max_ffn_ratio,
        args.promotion_max_component_regression_ratio,
        args.promotion_max_command_buffer_wait_ratio,
        args.promotion_require_profile,
        args.promotion_max_fused_wall_gpu_ratio,
        args.promotion_max_wait_gpu_ratio,
    )
    summary["router_parity"] = summarize_router_parity_taps(rows)
    summary["shared_parity"] = summarize_shared_parity_taps(rows)
    summary["decode_batch_shared_parity"] = summarize_shared_parity_taps(
        rows,
        "decode_batch_shared_parity_taps",
    )
    summary["decode_batch_routed_parity"] = summarize_routed_parity_taps(rows)
    summary["final_hidden_tap"] = summarize_final_hidden_taps(rows)
    summary["logits_tap"] = summarize_logits_taps(rows)
    summary["layer_output_tap"] = summarize_layer_output_taps(rows)
    summary["layer_output_delta_tap"] = summarize_layer_output_delta_taps(rows)
    summary["decode_batch_route_snapshot"] = summarize_decode_batch_route_snapshots(rows)
    return {
        "schema": SCHEMA,
        "model": MODEL,
        "backend": "metal",
        "prompt_set": prompt_set,
        "modes": modes,
        "max_new_tokens": args.max_new_tokens,
        "context_size": args.context_size,
        "stage_timing_modes": [mode for mode in modes if mode_emits_stage_timings(mode)],
        "fast_profile_modes": [mode for mode in modes if not mode_emits_stage_timings(mode)],
        "metal_profile": args.metal_profile,
        "metal_profile_phases": getattr(args, "metal_profile_phases", False),
        "downstream_parity_tap": getattr(args, "downstream_parity_tap", False),
        "layer_output_tap": getattr(args, "layer_output_tap", False),
        "layer_output_delta_tap": getattr(args, "layer_output_delta_tap", False),
        "layer_output_delta_layer": getattr(args, "layer_output_delta_layer", None),
        "layer_output_delta_position": getattr(args, "layer_output_delta_position", None),
        "layer_output_delta_phase": getattr(args, "layer_output_delta_phase", None),
        "router_parity_tap": getattr(args, "router_parity_tap", False),
        "router_parity_tap_max_calls": getattr(args, "router_parity_tap_max_calls", None),
        "shared_parity_tap": getattr(args, "shared_parity_tap", False),
        "shared_parity_tap_max_calls": getattr(args, "shared_parity_tap_max_calls", None),
        "routed_parity_tap": getattr(args, "routed_parity_tap", False),
        "routed_parity_tap_max_calls": getattr(args, "routed_parity_tap_max_calls", None),
        "decode_batch_route_snapshot": getattr(args, "decode_batch_route_snapshot", False),
        "promotion_thresholds": {
            "max_headline_ratio": args.promotion_max_headline_ratio,
            "max_ffn_ratio": args.promotion_max_ffn_ratio,
            "max_component_regression_ratio": args.promotion_max_component_regression_ratio,
            "max_command_buffer_wait_ratio": args.promotion_max_command_buffer_wait_ratio,
            "max_fused_wall_gpu_ratio": args.promotion_max_fused_wall_gpu_ratio,
            "max_wait_gpu_ratio": args.promotion_max_wait_gpu_ratio,
            "require_profile": args.promotion_require_profile,
        },
        "summary": summary,
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
    router_parity = summary.get("router_parity") or {}
    shared_parity = summary.get("shared_parity") or {}
    decode_batch_shared_parity = summary.get("decode_batch_shared_parity") or {}
    decode_batch_routed_parity = summary.get("decode_batch_routed_parity") or {}
    final_hidden_tap = summary.get("final_hidden_tap") or {}
    logits_tap = summary.get("logits_tap") or {}
    layer_output_tap = summary.get("layer_output_tap") or {}
    layer_output_delta_tap = summary.get("layer_output_delta_tap") or {}
    route_snapshot = summary.get("decode_batch_route_snapshot") or {}
    decode_batch_coarse = summary.get("decode_batch_coarse") or {}
    deferred_phase = summary.get("decode_batch_deferred_phase") or {}
    lines = [
        "# Qwen3.6 Fused Routed INT4 Sweep",
        "",
        f"- prompt_set: `{report['prompt_set']}`",
        f"- modes: `{','.join(report['modes'])}`",
        f"- max_new_tokens: `{report['max_new_tokens']}`",
        f"- stage_timing_modes: `{','.join(report.get('stage_timing_modes') or []) or '-'}`",
        f"- fast_profile_modes: `{','.join(report.get('fast_profile_modes') or []) or '-'}`",
        f"- metal_profile: `{report['metal_profile']}`",
        f"- metal_profile_phases: `{report.get('metal_profile_phases', False)}`",
        f"- downstream_parity_tap: `{report.get('downstream_parity_tap', False)}`",
        f"- layer_output_tap: `{report.get('layer_output_tap', False)}`",
        f"- layer_output_delta_tap: `{report.get('layer_output_delta_tap', False)}`",
        f"- router_parity_tap: `{report.get('router_parity_tap', False)}`",
        f"- shared_parity_tap: `{report.get('shared_parity_tap', False)}`",
        f"- routed_parity_tap: `{report.get('routed_parity_tap', False)}`",
        f"- decode_batch_route_snapshot: `{report.get('decode_batch_route_snapshot', False)}`",
        f"- generated_ids_match: `{summary['generated_ids_match']}`",
        f"- promotion_gate_passed: `{promotion_gate.get('passed', False)}`",
        f"- promotion_gate_passed_modes: `{','.join(promotion_gate.get('passed_modes') or []) or '-'}`",
        f"- ffn_gap_recommendation: `{ffn_gap.get('recommendation') or '-'}`",
        f"- decode_batch_coarse_recommendation: `{decode_batch_coarse.get('recommendation') or '-'}`",
        f"- decode_batch_deferred_phase_recommendation: `{deferred_phase.get('recommendation') or '-'}`",
        f"- router_parity_tap_count: `{router_parity.get('tap_count', 0)}`",
        f"- router_parity_mismatches: `{router_parity.get('mismatch_count', 0)}`",
        f"- shared_parity_tap_count: `{shared_parity.get('tap_count', 0)}`",
        f"- shared_parity_max_out_abs: `{render_float(shared_parity.get('max_shared_out_abs'), 8)}`",
        f"- decode_batch_shared_parity_tap_count: `{decode_batch_shared_parity.get('tap_count', 0)}`",
        f"- decode_batch_shared_parity_max_out_abs: `{render_float(decode_batch_shared_parity.get('max_shared_out_abs'), 8)}`",
        f"- decode_batch_routed_parity_tap_count: `{decode_batch_routed_parity.get('tap_count', 0)}`",
        f"- decode_batch_routed_parity_max_moe_out_abs: `{render_float(decode_batch_routed_parity.get('max_moe_out_abs'), 8)}`",
        f"- decode_batch_routed_parity_max_final_out_abs: `{render_float(decode_batch_routed_parity.get('max_final_out_abs'), 8)}`",
        f"- final_hidden_tap_count: `{final_hidden_tap.get('tap_count', 0)}`",
        f"- final_hidden_checksum_mismatches: `{final_hidden_tap.get('checksum_mismatch_count', 0)}`",
        f"- logits_tap_count: `{logits_tap.get('tap_count', 0)}`",
        f"- logits_top1_mismatches: `{logits_tap.get('top1_mismatch_count', 0)}`",
        f"- layer_output_tap_count: `{layer_output_tap.get('tap_count', 0)}`",
        f"- layer_output_checksum_mismatches: `{layer_output_tap.get('checksum_mismatch_count', 0)}`",
        f"- layer_output_delta_tap_count: `{layer_output_delta_tap.get('tap_count', 0)}`",
        f"- layer_output_delta_max_abs: `{render_float(layer_output_delta_tap.get('max_abs_delta'), 8)}`",
        f"- layer_output_delta_max_ulp: `{layer_output_delta_tap.get('max_ulp_delta', 0)}`",
        f"- decode_batch_route_snapshot_count: `{route_snapshot.get('snapshot_count', 0)}`",
        f"- decode_batch_route_snapshot_mismatches: `{route_snapshot.get('mismatch_count', 0)}`",
        "",
        "| Prompt | Mode | Status | IDs | Decode ms | FFN ms avg | Fused wall ms | Fused GPU ms | Batch GPU ms | Batch lin GPU ms | Batch FFN GPU ms | Wall/GPU | Wait/GPU | FFN class | Top Metal op | Top Metal ms | HAL ms | Wall s |",
        "|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|:---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        result = row.get("result") or {}
        chain = row.get("chain_breakdown") or {}
        top_metal = top_profile_op(row.get("metal_profile"))
        hal_summary = (row.get("hal_profile") or {}).get("summary") or {}
        lines.append(
            "| {prompt} | {mode} | {status} | {ids} | {decode} | {ffn} | {fused_wall} | {fused_gpu} | {batch_gpu} | {batch_linear} | {batch_ffn} | {wall_gpu} | {wait_gpu} | {ffn_class} | {top_op} | {top_ms} | {hal_ms} | {wall} |".format(
                prompt=row.get("prompt_id", ""),
                mode=row.get("mode", ""),
                status=row.get("status", ""),
                ids=",".join(str(item) for item in row.get("generated_ids", [])),
                decode=render_float(result.get("decode_ms")),
                ffn=render_float(chain.get("ffn_ms_avg")),
                fused_wall=render_float(row.get("fused_wall_ms")),
                fused_gpu=render_float(row.get("fused_gpu_ms")),
                batch_gpu=render_float(row.get("decode_batch_gpu_ms")),
                batch_linear=render_float(row.get("decode_batch_linear_gpu_ms")),
                batch_ffn=render_float(row.get("decode_batch_ffn_gpu_ms")),
                wall_gpu=render_float(row.get("fused_wall_gpu_ratio"), 2),
                wait_gpu=render_float(row.get("wait_gpu_ratio"), 2),
                ffn_class=row.get("ffn_attribution_class") or "-",
                top_op=top_metal.get("op") or "-",
                top_ms=render_float(top_metal.get("total_ms")),
                hal_ms=render_float(hal_summary.get("total_ms")),
                wall=render_float(row.get("wall_seconds"), 1),
            )
        )
    coarse_comparisons = [
        item
        for item in (decode_batch_coarse.get("comparisons") or [])
        if item.get("status") == "ok"
    ]
    if coarse_comparisons:
        lines.extend(
            [
                "",
                "## Decode Batch Coarse SIMD",
                "",
                f"- recommendation: `{decode_batch_coarse.get('recommendation') or '-'}`",
                f"- reason: {decode_batch_coarse.get('reason') or '-'}",
                "",
                "| Prompt | IDs match default | IDs match serial | Serial decode ms | SIMD decode ms | Decode ratio | Serial batch GPU ms | SIMD batch GPU ms | GPU ratio | Serial wait ms | SIMD wait ms | Wait ratio | SIMD wait/GPU | Blocker |",
                "|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
            ]
        )
        for comparison in coarse_comparisons:
            ids_default = comparison.get("generated_ids_match_default")
            ids_serial = comparison.get("generated_ids_match_serial")
            lines.append(
                "| {prompt} | {ids_default} | {ids_serial} | {serial_decode} | {simd_decode} | {decode_ratio} | {serial_gpu} | {simd_gpu} | {gpu_ratio} | {serial_wait} | {simd_wait} | {wait_ratio} | {wait_gpu} | {blocker} |".format(
                    prompt=comparison.get("prompt_id", ""),
                    ids_default="-" if ids_default is None else str(bool(ids_default)).lower(),
                    ids_serial="-" if ids_serial is None else str(bool(ids_serial)).lower(),
                    serial_decode=render_float(comparison.get("serial_decode_ms")),
                    simd_decode=render_float(comparison.get("simd_decode_ms")),
                    decode_ratio=render_float(comparison.get("decode_ratio"), 3),
                    serial_gpu=render_float(comparison.get("serial_decode_batch_gpu_ms")),
                    simd_gpu=render_float(comparison.get("simd_decode_batch_gpu_ms")),
                    gpu_ratio=render_float(comparison.get("decode_batch_gpu_ratio"), 3),
                    serial_wait=render_float(comparison.get("serial_command_buffer_wait_ms")),
                    simd_wait=render_float(comparison.get("simd_command_buffer_wait_ms")),
                    wait_ratio=render_float(comparison.get("command_buffer_wait_ratio"), 3),
                    wait_gpu=render_float(comparison.get("simd_wait_gpu_ratio"), 3),
                    blocker=comparison.get("blocker") or "-",
                )
            )
    deferred_rows = [
        item
        for item in (deferred_phase.get("rows") or [])
        if item.get("status") == "ok"
    ]
    if deferred_rows:
        lines.extend(
            [
                "",
                "## Decode Batch Deferred Phase",
                "",
                f"- recommendation: `{deferred_phase.get('recommendation') or '-'}`",
                f"- reason: {deferred_phase.get('reason') or '-'}",
                "",
                "| Prompt | Mode | Path | IDs match default | Decode ms | Linear GPU ms | FFN GPU ms | Phase GPU ms | FFN share | Wait ms | Wait/GPU | Blocker |",
                "|:---|:---|:---|:---:|---:|---:|---:|---:|---:|---:|---:|:---|",
            ]
        )
        for item in deferred_rows:
            ids_match = item.get("generated_ids_match_default")
            lines.append(
                "| {prompt} | {mode} | {path} | {ids} | {decode} | {linear} | {ffn} | {total} | {share} | {wait} | {wait_gpu} | {blocker} |".format(
                    prompt=item.get("prompt_id", ""),
                    mode=item.get("mode", ""),
                    path=item.get("router_path", "-"),
                    ids="-" if ids_match is None else str(bool(ids_match)).lower(),
                    decode=render_float(item.get("decode_ms")),
                    linear=render_float(item.get("linear_gpu_ms")),
                    ffn=render_float(item.get("ffn_gpu_ms")),
                    total=render_float(item.get("total_phase_gpu_ms")),
                    share=render_float(item.get("ffn_share"), 3),
                    wait=render_float(item.get("command_buffer_wait_ms")),
                    wait_gpu=render_float(item.get("wait_gpu_ratio"), 3),
                    blocker=item.get("blocker") or "-",
                )
            )
    if any(
        row.get(field) is not None
        for row in report["rows"]
        for field in BATCH_FFN_PHASE_GPU_FIELDS
    ):
        lines.extend(
            [
                "",
                "## Batch FFN Subphases",
                "",
                "| Prompt | Mode | Router top-k GPU ms | Shared gate/up GPU ms | Shared scalar GPU ms | Shared down GPU ms | Expert gate/up GPU ms | Expert down GPU ms | Total GPU ms |",
                "|:---|:---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["rows"]:
            if not any(row.get(field) is not None for field in BATCH_FFN_PHASE_GPU_FIELDS):
                continue
            lines.append(
                "| {prompt} | {mode} | {router} | {shared_gate} | {shared_scalar} | {shared_down} | {expert_gate} | {expert_down} | {total} |".format(
                    prompt=row.get("prompt_id", ""),
                    mode=row.get("mode", ""),
                    router=render_float(row.get("decode_batch_ffn_router_topk_gpu_ms")),
                    shared_gate=render_float(row.get("decode_batch_ffn_shared_gate_up_gpu_ms")),
                    shared_scalar=render_float(row.get("decode_batch_ffn_shared_scalar_gpu_ms")),
                    shared_down=render_float(row.get("decode_batch_ffn_shared_down_gpu_ms")),
                    expert_gate=render_float(row.get("decode_batch_ffn_expert_gate_up_gpu_ms")),
                    expert_down=render_float(row.get("decode_batch_ffn_expert_down_gpu_ms")),
                    total=render_float(row.get("decode_batch_ffn_gpu_ms")),
                )
            )
    snapshot_comparisons = decode_batch_route_snapshot_comparisons(report["rows"])
    if snapshot_comparisons:
        lines.extend(
            [
                "",
                "## Decode Batch Route Snapshot",
                "",
                "| Prompt | Mode | Path | Call | Pos | Captured | Checksum | Ref mode | Ref path | Match ref | Routes head |",
                "|:---|:---|:---|---:|---:|---:|:---|:---|:---|:---:|:---|",
            ]
        )
        snapshots_by_key = {
            (
                str(row.get("prompt_id", "")),
                str(row.get("mode", "")),
                int(snapshot.get("call", 0)),
            ): snapshot
            for row in report["rows"]
            for snapshot in (row.get("decode_batch_route_snapshots") or [])
        }
        for comparison in snapshot_comparisons[:40]:
            snapshot = snapshots_by_key.get(
                (
                    str(comparison.get("prompt_id", "")),
                    str(comparison.get("mode", "")),
                    int(comparison.get("call", 0)),
                ),
                {},
            )
            routes = str(snapshot.get("routes") or "-")
            lines.append(
                "| {prompt} | {mode} | {path} | {call} | {position} | {captured} | {checksum} | {ref_mode} | {ref_path} | {match} | {routes_head} |".format(
                    prompt=comparison.get("prompt_id", ""),
                    mode=comparison.get("mode", ""),
                    path=comparison.get("path", "-"),
                    call=comparison.get("call", "-"),
                    position=snapshot.get("position", "-"),
                    captured=snapshot.get("captured_layers", "-"),
                    checksum=comparison.get("checksum", "-"),
                    ref_mode=comparison.get("reference_mode", "-"),
                    ref_path=comparison.get("reference_path", "-"),
                    match=str(bool(comparison.get("match_reference"))).lower(),
                    routes_head=(routes[:96] + "...") if len(routes) > 96 else routes,
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
                "| Prompt | Mode | Path | Layer | Match | First mismatch | HNorm max | HNorm idx | Logit max | Logit idx | Host top | Metal top | Host@Metal | Metal@Host | TopK weight max | Host idx | Metal idx |",
                "|:---|:---|:---|---:|:---:|---:|---:|---:|---:|---:|:---|:---|---:|---:|---:|:---|:---|",
            ]
        )
        for row, tap in select_router_parity_tap_rows(tap_rows, limit=40):
            host_top = "{idx}:{value}".format(
                idx=tap.get("host_top_logit_idx", "-"),
                value=render_float(tap.get("host_top_logit"), 8),
            )
            metal_top = "{idx}:{value}".format(
                idx=tap.get("metal_top_logit_idx", "-"),
                value=render_float(tap.get("metal_top_logit"), 8),
            )
            lines.append(
                "| {prompt} | {mode} | {path} | {layer} | {match} | {first_mismatch} | {hnorm} | {hnorm_idx} | {logits} | {logits_idx} | {host_top} | {metal_top} | {host_at_metal} | {metal_at_host} | {weight} | {host_idx} | {metal_idx} |".format(
                    prompt=row.get("prompt_id", ""),
                    mode=row.get("mode", ""),
                    path=tap.get("router_path", "-"),
                    layer=tap.get("layer", "-"),
                    match=str(bool(tap.get("topk_idx_match"))).lower(),
                    first_mismatch=tap.get(
                        "topk_first_mismatch",
                        tap.get("workspace_first_idx_mismatch", "-"),
                    ),
                    hnorm=render_float(tap.get("h_norm_max_abs"), 8),
                    hnorm_idx=tap.get("h_norm_argmax", "-"),
                    logits=render_float(tap.get("logits_max_abs"), 8),
                    logits_idx=tap.get("logits_argmax", "-"),
                    host_top=host_top,
                    metal_top=metal_top,
                    host_at_metal=render_float(tap.get("host_logit_at_metal_top"), 8),
                    metal_at_host=render_float(tap.get("metal_logit_at_host_top"), 8),
                    weight=render_float(tap.get("topk_weight_max_abs"), 8),
                    host_idx=tap.get("host_idx", "-"),
                    metal_idx=tap.get("workspace_idx", tap.get("output_idx", "-")),
                )
            )
    shared_tap_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in report["rows"]:
        for tap in row.get("shared_parity_taps") or []:
            shared_tap_rows.append((row, tap))
    if shared_tap_rows:
        lines.extend(
            [
                "",
                "## Shared Expert Parity Tap",
                "",
                "| Prompt | Mode | Path | Layer | Gate max | Gate idx | Up max | Up idx | Mid max | Mid idx | Scalar max | Shared out max | Shared out idx | Host out | Metal out |",
                "|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row, tap in select_shared_parity_tap_rows(shared_tap_rows, limit=40):
            lines.append(
                "| {prompt} | {mode} | {path} | {layer} | {gate} | {gate_idx} | {up} | {up_idx} | {mid} | {mid_idx} | {scalar} | {out} | {out_idx} | {host_out} | {metal_out} |".format(
                    prompt=row.get("prompt_id", ""),
                    mode=row.get("mode", ""),
                    path=tap.get("shared_path", "-"),
                    layer=tap.get("layer", "-"),
                    gate=render_float(tap.get("shared_gate_max_abs"), 8),
                    gate_idx=tap.get("shared_gate_argmax", "-"),
                    up=render_float(tap.get("shared_up_max_abs"), 8),
                    up_idx=tap.get("shared_up_argmax", "-"),
                    mid=render_float(tap.get("shared_mid_max_abs"), 8),
                    mid_idx=tap.get("shared_mid_argmax", "-"),
                    scalar=render_float(tap.get("shared_scalar_abs"), 8),
                    out=render_float(tap.get("shared_out_max_abs"), 8),
                    out_idx=tap.get("shared_out_argmax", "-"),
                    host_out=render_float(tap.get("host_shared_out_at_argmax"), 8),
                    metal_out=render_float(tap.get("metal_shared_out_at_argmax"), 8),
                )
            )
    decode_batch_shared_tap_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in report["rows"]:
        for tap in row.get("decode_batch_shared_parity_taps") or []:
            decode_batch_shared_tap_rows.append((row, tap))
    if decode_batch_shared_tap_rows:
        lines.extend(
            [
                "",
                "## Decode-Batch Shared Expert Parity Tap",
                "",
                "| Prompt | Mode | Router | Phase profile | Path | Call | Position | Layer | Gate max | Up max | Mid max | Scalar max | Shared out max | Shared out idx | Host out | Metal out |",
                "|:---|:---|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row, tap in select_shared_parity_tap_rows(
            decode_batch_shared_tap_rows,
            limit=40,
        ):
            lines.append(
                "| {prompt} | {mode} | {router} | {phase} | {path} | {call} | {position} | {layer} | {gate} | {up} | {mid} | {scalar} | {out} | {out_idx} | {host_out} | {metal_out} |".format(
                    prompt=row.get("prompt_id", ""),
                    mode=row.get("mode", ""),
                    router=tap.get("router_path", "-"),
                    phase=str(bool(tap.get("phase_profile"))).lower(),
                    path=tap.get("shared_path", "-"),
                    call=tap.get("call", "-"),
                    position=tap.get("position", "-"),
                    layer=tap.get("layer", "-"),
                    gate=render_float(tap.get("shared_gate_max_abs"), 8),
                    up=render_float(tap.get("shared_up_max_abs"), 8),
                    mid=render_float(tap.get("shared_mid_max_abs"), 8),
                    scalar=render_float(tap.get("shared_scalar_abs"), 8),
                    out=render_float(tap.get("shared_out_max_abs"), 8),
                    out_idx=tap.get("shared_out_argmax", "-"),
                    host_out=render_float(tap.get("host_shared_out_at_argmax"), 8),
                    metal_out=render_float(tap.get("metal_shared_out_at_argmax"), 8),
                )
            )
    decode_batch_routed_tap_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in report["rows"]:
        for tap in row.get("decode_batch_routed_parity_taps") or []:
            decode_batch_routed_tap_rows.append((row, tap))
    if decode_batch_routed_tap_rows:
        ranked_routed = sorted(
            decode_batch_routed_tap_rows,
            key=lambda pair: max(
                float(pair[1].get("expert_mid_max_abs") or 0.0),
                float(pair[1].get("moe_out_max_abs") or 0.0),
                float(pair[1].get("final_out_max_abs") or 0.0),
                float(pair[1].get("topk_weight_max_abs") or 0.0),
            ),
            reverse=True,
        )
        lines.extend(
            [
                "",
                "## Decode-Batch Routed Expert Parity Tap",
                "",
                "| Prompt | Mode | Router | Phase profile | Call | Position | Layer | TopK match | TopK weight max | Expert mid max | Expert mid idx | Host mid | Metal mid | MoE out max | MoE out idx | Host MoE | Metal MoE | Final out max | Final out idx | Host final | Metal final |",
                "|:---|:---|:---|:---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row, tap in ranked_routed[:40]:
            lines.append(
                "| {prompt} | {mode} | {router} | {phase} | {call} | {position} | {layer} | {match} | {weight} | {mid} | {mid_idx} | {host_mid} | {metal_mid} | {moe} | {moe_idx} | {host_moe} | {metal_moe} | {final} | {final_idx} | {host_final} | {metal_final} |".format(
                    prompt=row.get("prompt_id", ""),
                    mode=row.get("mode", ""),
                    router=tap.get("router_path", "-"),
                    phase=str(bool(tap.get("phase_profile"))).lower(),
                    call=tap.get("call", "-"),
                    position=tap.get("position", "-"),
                    layer=tap.get("layer", "-"),
                    match=str(bool(tap.get("topk_idx_match"))).lower(),
                    weight=render_float(tap.get("topk_weight_max_abs"), 8),
                    mid=render_float(tap.get("expert_mid_max_abs"), 8),
                    mid_idx=tap.get("expert_mid_argmax", "-"),
                    host_mid=render_float(tap.get("host_expert_mid_at_argmax"), 8),
                    metal_mid=render_float(tap.get("metal_expert_mid_at_argmax"), 8),
                    moe=render_float(tap.get("moe_out_max_abs"), 8),
                    moe_idx=tap.get("moe_out_argmax", "-"),
                    host_moe=render_float(tap.get("host_moe_out_at_argmax"), 8),
                    metal_moe=render_float(tap.get("metal_moe_out_at_argmax"), 8),
                    final=render_float(tap.get("final_out_max_abs"), 8),
                    final_idx=tap.get("final_out_argmax", "-"),
                    host_final=render_float(tap.get("host_final_out_at_argmax"), 8),
                    metal_final=render_float(tap.get("metal_final_out_at_argmax"), 8),
                )
            )
    final_hidden_comparisons = (summary.get("final_hidden_tap") or {}).get("comparisons") or []
    if final_hidden_comparisons:
        lines.extend(
            [
                "",
                "## Final Hidden Tap",
                "",
                "| Prompt | Mode | Path | Gen | Checksum match | Baseline checksum | Candidate checksum |",
                "|:---|:---|:---|---:|:---:|:---|:---|",
            ]
        )
        for item in final_hidden_comparisons[:40]:
            lines.append(
                "| {prompt} | {mode} | {path} | {gen} | {match} | {base} | {checksum} |".format(
                    prompt=item.get("prompt_id", ""),
                    mode=item.get("mode", ""),
                    path=item.get("path", "-"),
                    gen=item.get("gen_index", "-"),
                    match=str(item.get("checksum_match")).lower(),
                    base=item.get("baseline_checksum") or "-",
                    checksum=item.get("checksum") or "-",
                )
            )
    logits_comparisons = (summary.get("logits_tap") or {}).get("comparisons") or []
    if logits_comparisons:
        lines.extend(
            [
                "",
                "## Logits Tap",
                "",
                "| Prompt | Mode | Path | Gen | Checksum match | Top1 match | Baseline top1 | Candidate top1 |",
                "|:---|:---|:---|---:|:---:|:---:|---:|---:|",
            ]
        )
        for item in logits_comparisons[:40]:
            lines.append(
                "| {prompt} | {mode} | {path} | {gen} | {checksum_match} | {top1_match} | {base_top1} | {top1} |".format(
                    prompt=item.get("prompt_id", ""),
                    mode=item.get("mode", ""),
                    path=item.get("path", "-"),
                    gen=item.get("gen_index", "-"),
                    checksum_match=str(item.get("checksum_match")).lower(),
                    top1_match=str(item.get("top1_match")).lower(),
                    base_top1=item.get("baseline_top1_idx", "-"),
                    top1=item.get("top1_idx", "-"),
                )
            )
    layer_output_comparisons = (summary.get("layer_output_tap") or {}).get("comparisons") or []
    if layer_output_comparisons:
        lines.extend(
            [
                "",
                "## Layer Output Tap",
                "",
                "| Prompt | Mode | Path | Position | Layer | Phase | Checksum match | Baseline checksum | Candidate checksum |",
                "|:---|:---|:---|---:|---:|:---|:---:|:---|:---|",
            ]
        )
        for item in layer_output_comparisons[:80]:
            lines.append(
                "| {prompt} | {mode} | {path} | {position} | {layer} | {phase} | {match} | {base} | {checksum} |".format(
                    prompt=item.get("prompt_id", ""),
                    mode=item.get("mode", ""),
                    path=item.get("path", "-"),
                    position=item.get("position", "-"),
                    layer=item.get("layer", "-"),
                    phase=item.get("phase", "-"),
                    match=str(item.get("checksum_match")).lower(),
                    base=item.get("baseline_checksum") or "-",
                    checksum=item.get("checksum") or "-",
                )
            )
    layer_output_delta_comparisons = (
        (summary.get("layer_output_delta_tap") or {}).get("comparisons") or []
    )
    if layer_output_delta_comparisons:
        lines.extend(
            [
                "",
                "## Layer Output Delta Tap",
                "",
                "| Prompt | Mode | Path | Position | Layer | Phase | Checksum match | Max abs delta | Max abs idx | Max ULP | ULP idx | Differing elems | Baseline | Candidate |",
                "|:---|:---|:---|---:|---:|:---|:---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in layer_output_delta_comparisons[:40]:
            lines.append(
                "| {prompt} | {mode} | {path} | {position} | {layer} | {phase} | {match} | {delta} | {delta_idx} | {ulp} | {ulp_idx} | {diffs} | {base} | {candidate} |".format(
                    prompt=item.get("prompt_id", ""),
                    mode=item.get("mode", ""),
                    path=item.get("path", "-"),
                    position=item.get("position", "-"),
                    layer=item.get("layer", "-"),
                    phase=item.get("phase", "-"),
                    match=str(item.get("checksum_match")).lower(),
                    delta=render_float(item.get("max_abs_delta"), 8),
                    delta_idx=item.get("max_abs_delta_idx", "-"),
                    ulp=item.get("max_ulp_delta", "-"),
                    ulp_idx=item.get("max_ulp_delta_idx", "-"),
                    diffs=item.get("differing_elems", "-"),
                    base=render_float(item.get("baseline_value_at_max_abs"), 8),
                    candidate=render_float(item.get("candidate_value_at_max_abs"), 8),
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
        "--downstream-parity-tap",
        action="store_true",
        help="emit final-hidden and lm-head logits signatures for default-vs-decode-batch comparison",
    )
    parser.add_argument(
        "--layer-output-tap",
        action="store_true",
        help="emit post-attention and post-FFN layer-output signatures for default-vs-decode-batch comparison",
    )
    parser.add_argument(
        "--layer-output-delta-tap",
        action="store_true",
        help="emit full BF16 layer-output rows for numeric default-vs-candidate delta comparison",
    )
    parser.add_argument(
        "--layer-output-delta-position",
        type=int,
        default=0,
        help="position filter for the layer-output delta tap",
    )
    parser.add_argument(
        "--layer-output-delta-layer",
        type=int,
        default=0,
        help="layer filter for the layer-output delta tap",
    )
    parser.add_argument(
        "--layer-output-delta-phase",
        choices=["attn", "ffn"],
        default="ffn",
        help="phase filter for the layer-output delta tap",
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
        "--shared-parity-tap",
        action="store_true",
        help="emit and parse Qwen3.6 full-stage5-router Metal-vs-host shared-expert parity rows",
    )
    parser.add_argument(
        "--shared-parity-tap-max-calls",
        type=int,
        default=40,
        help="maximum shared-expert parity tap rows emitted by the runtime",
    )
    parser.add_argument(
        "--routed-parity-tap",
        action="store_true",
        help="emit and parse Qwen3.6 decode-batch routed-expert parity rows",
    )
    parser.add_argument(
        "--routed-parity-tap-max-calls",
        type=int,
        default=40,
        help="maximum decode-batch routed-expert parity tap rows emitted by the runtime",
    )
    parser.add_argument(
        "--decode-batch-route-snapshot",
        action="store_true",
        help="capture per-layer decode-batch FFN top-k route snapshots at batch end",
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
