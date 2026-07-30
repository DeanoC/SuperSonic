#!/usr/bin/env python3
"""Run SuperSonic Qwen3.6 over the Lucebox HumanEval DFlash prompts.

This is a comparison harness, not a correctness gate. It can run either the
Lucebox-style Qwen3.6-27B GGUF target plus DFlash draft, or the native
Qwen3.6-35B-A3B MoE path over the same HumanEval prompt set.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_LUCEBOX_BENCH = Path("/home/deano/projects/lucebox-hub/server/scripts/bench_he.py")
DEFAULT_LUCEBOX_HE_JSONL = Path(
    "/home/deano/projects/lucebox-hub/harness/benchmarks/prompts/bench_he.jsonl"
)
DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR = Path("/mnt/data/tmp/qwen36-27b-dflash-q8-bf16")
DEFAULT_LUCEBOX_DRAFT_DIR = Path("/mnt/data/lucebox-hub/models/draft")
DEFAULT_27B_MODEL = "qwen3.6-27b"
DEFAULT_27B_MODEL_DIR = Path("/mnt/data/tmp/supersonic-qwen36-27b-lucebox")
DEFAULT_27B_QUANT = "q4km-gptq"
DEFAULT_35B_A3B_MODEL = "qwen3.6-35b-a3b"
DEFAULT_35B_A3B_MODEL_DIR = Path("/mnt/data/models/Qwen3.6-35B-A3B")
DEFAULT_35B_A3B_FLM_MODEL_DIR = Path(
    "/mnt/data/runs/geo-quant/"
    "qwen36-35b-a3b-supersonic-native-int4-current.flm"
)
DEFAULT_35B_A3B_QUANT = "int4"
DEFAULT_OUT_JSON = Path("target/qwen36_he_supersonic.json")
DEFAULT_35B_A3B_OUT_JSON = Path("target/qwen36_35b_a3b_he_supersonic.json")
DEFAULT_35B_A3B_FLM_OUT_JSON = Path("target/qwen36_35b_a3b_flm_he_supersonic.json")
DEFAULT_CONTEXT_SIZE = 512
LUCEBOX_SERVING_CONTEXT_SIZE = 1024
GIB = 1024.0 * 1024.0 * 1024.0
LUCEBOX_DRAFT_ALIASES = {
    "supersonic-q8-bf16": {
        "label": "supersonic-q8-bf16",
        "config_dir": DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR,
        "gguf": None,
    },
    "lucebox-q4-k-m": {
        "label": "lucebox-q4-k-m",
        "config_dir": DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR,
        "gguf": DEFAULT_LUCEBOX_DRAFT_DIR / "dflash-draft-3.6-q4_k_m.gguf",
    },
    "lucebox-q8-0": {
        "label": "lucebox-q8-0",
        "config_dir": DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR,
        "gguf": DEFAULT_LUCEBOX_DRAFT_DIR / "dflash-draft-3.6-q8_0.gguf",
    },
}
RESULT_RE = re.compile(
    r"\[result\]\s+prompt_tokens=(?P<prompt>\d+)\s+"
    r"generated_tokens=(?P<generated>\d+)\s+"
    r"decode_ms=(?P<decode_ms>[0-9.]+)\s+"
    r"ms_per_(?:step|tok)=(?P<ms_per_step>[0-9.]+)"
)
LIFECYCLE_RE = re.compile(r"\[qwen36-moe lifecycle-timings\]\s+(?P<body>.+)")
STARTUP_RE = re.compile(r"\[qwen36-moe startup-timings\]\s+(?P<body>.+)")
SPARSE_RESIDENCY_RE = re.compile(r"\[vmm\]\s+MoE island residency:\s+(?P<body>.+)")
SPARSE_BREAKDOWN_RE = re.compile(r"\[qwen36-moe sparse-breakdown\]\s+(?P<body>.+)")
TIMING_KV_RE = re.compile(r"(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)=(?P<value>-?[0-9]+(?:\.[0-9]+)?)")
INFERRED_MODEL_RE = re.compile(
    r"\[flm\]\s+inferred model (?P<model>\S+) from runtime descriptor"
)
FLM_WEIGHT_MODE_RE = re.compile(
    r"\[qwen36-moe\]\s+FLM weight mode:\s+(?P<mode>[^\r\n]+)"
)
FLM_DIRECT_PROFILE_RE = re.compile(
    r"\[qwen36-moe\]\s+FLM direct plans:\s+"
    r"required=(?P<required>\d+)\s+"
    r"raw_dense=(?P<raw_dense>\d+)\s+"
    r"native_int4=(?P<native_int4>\d+)\s+"
    r"bf16_fallback=(?P<bf16_fallback>\d+)"
)
FLM_READY_RE = re.compile(
    r"\[FLM runtime weights\]\s+ready-for-decode:\s+(?P<ready>YES|NO)"
    r"(?:\s+\((?P<detail>[^\r\n)]*)\))?"
)
RUNTIME_ENGINE_OWNERSHIP_RE = re.compile(
    r"\[qwen36-moe\]\s+runtime engine ready:\s+"
    r"load_sequence=(?P<load_sequence>\d+)\s+"
    r"source_open_count=(?P<source_open_count>\d+)"
)
HAL_PROFILE_OP_RE = re.compile(
    r"\[hal-profile-op\]\s+op=(?P<op>\S+)\s+"
    r"calls=(?P<calls>\d+)\s+"
    r"mean_ms=(?P<mean_ms>[0-9.]+)\s+"
    r"total_ms=(?P<total_ms>[0-9.]+)\s+"
    r"max_ms=(?P<max_ms>[0-9.]+)\s+"
    r"total_bytes=(?P<total_bytes>\d+)"
)


TARGET_PROFILES = {
    "qwen36-27b-lucebox": {
        "model": DEFAULT_27B_MODEL,
        "model_dir": DEFAULT_27B_MODEL_DIR,
        "quant": DEFAULT_27B_QUANT,
        "out_json": DEFAULT_OUT_JSON,
    },
    "qwen36-35b-a3b": {
        "model": DEFAULT_35B_A3B_MODEL,
        "model_dir": DEFAULT_35B_A3B_MODEL_DIR,
        "quant": DEFAULT_35B_A3B_QUANT,
        "out_json": DEFAULT_35B_A3B_OUT_JSON,
    },
    "qwen36-35b-a3b-flm": {
        "model": None,
        "model_dir": DEFAULT_35B_A3B_FLM_MODEL_DIR,
        "quant": "none",
        "out_json": DEFAULT_35B_A3B_FLM_OUT_JSON,
    },
}


def load_lucebox_script_prompts(path: Path) -> list[tuple[str, str]]:
    script_dir = str(path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    spec = importlib.util.spec_from_file_location("lucebox_bench_he", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to import Lucebox bench script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompts = getattr(module, "PROMPTS", None)
    if not prompts:
        raise RuntimeError(f"no PROMPTS found in {path}")
    return [(str(name), str(prompt)) for name, prompt in prompts]


QWEN3_NO_THINKING_PREFILL = "<think>\n\n</think>\n\n"


def render_chatml(messages: list[dict], *, no_thinking: bool = False) -> str:
    chunks = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        chunks.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    chunks.append("<|im_start|>assistant\n")
    if no_thinking:
        chunks.append(QWEN3_NO_THINKING_PREFILL)
    return "".join(chunks)


def load_lucebox_jsonl_prompts(path: Path, prompt_format: str) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            name = str(case.get("id") or case.get("name") or f"case_{idx:03d}")
            messages = case.get("messages")
            if isinstance(messages, list) and prompt_format in {
                "chatml",
                "chatml-no-thinking",
            }:
                prompt = render_chatml(
                    messages, no_thinking=prompt_format == "chatml-no-thinking"
                )
            elif isinstance(messages, list) and messages:
                prompt = "\n\n".join(str(m.get("content", "")) for m in messages)
            else:
                prompt = str(case.get("prompt") or case.get("content") or "")
            if not prompt:
                raise RuntimeError(f"empty prompt in {path}:{idx}")
            prompts.append((name, prompt))
    if not prompts:
        raise RuntimeError(f"no prompts found in {path}")
    return prompts


def load_prompts(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.prompt_source == "script":
        return load_lucebox_script_prompts(args.lucebox_bench)
    if args.prompt_source == "jsonl":
        return load_lucebox_jsonl_prompts(args.lucebox_jsonl, args.prompt_format)
    raise ValueError(f"unknown prompt source: {args.prompt_source}")


def resolve_dflash_draft(args: argparse.Namespace) -> tuple[Path, Path | None, str]:
    config_dir = args.dflash_draft_dir
    gguf = args.dflash_draft_gguf
    label = "custom"
    if args.dflash_draft_variant:
        alias = LUCEBOX_DRAFT_ALIASES[args.dflash_draft_variant]
        label = alias["label"]
        config_dir = alias["config_dir"]
        gguf = alias["gguf"]
    if config_dir is None:
        config_dir = DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR
    return Path(config_dir), Path(gguf) if gguf else None, label


def parse_lifecycle_timings(text: str) -> dict[str, float] | None:
    match = LIFECYCLE_RE.search(text)
    if not match:
        return None
    return parse_numeric_kv_body(match.group("body"))


def parse_startup_timings(text: str) -> dict[str, float] | None:
    match = STARTUP_RE.search(text)
    if not match:
        return None
    return parse_numeric_kv_body(match.group("body"))


def parse_numeric_kv_body(body: str) -> dict[str, float] | None:
    values = {
        item.group("key"): float(item.group("value"))
        for item in TIMING_KV_RE.finditer(body)
    }
    return values or None


def parse_sparse_residency(text: str) -> dict[str, float] | None:
    matches = list(SPARSE_RESIDENCY_RE.finditer(text))
    if not matches:
        return None
    return parse_numeric_kv_body(matches[-1].group("body"))


def parse_sparse_breakdown(text: str) -> dict[str, float] | None:
    matches = list(SPARSE_BREAKDOWN_RE.finditer(text))
    if not matches:
        return None
    return parse_numeric_kv_body(matches[-1].group("body"))


def parse_inferred_model(text: str) -> str | None:
    match = INFERRED_MODEL_RE.search(text)
    if not match:
        return None
    return match.group("model")


def parse_flm_weight_mode(text: str) -> str | None:
    match = FLM_WEIGHT_MODE_RE.search(text)
    if not match:
        return None
    return match.group("mode").strip()


def parse_flm_direct_profile(text: str) -> dict[str, int] | None:
    match = FLM_DIRECT_PROFILE_RE.search(text)
    if not match:
        return None
    return {
        "required": int(match.group("required")),
        "raw_dense": int(match.group("raw_dense")),
        "native_int4": int(match.group("native_int4")),
        "bf16_fallback": int(match.group("bf16_fallback")),
    }


def parse_flm_ready_for_decode(text: str) -> dict[str, bool | str] | None:
    match = FLM_READY_RE.search(text)
    if not match:
        return None
    result: dict[str, bool | str] = {"ready": match.group("ready") == "YES"}
    detail = match.group("detail")
    if detail:
        result["detail"] = detail
    return result


def parse_runtime_engine_ownership(text: str) -> list[dict[str, int]]:
    return [
        {
            "load_sequence": int(match.group("load_sequence")),
            "source_open_count": int(match.group("source_open_count")),
        }
        for match in RUNTIME_ENGINE_OWNERSHIP_RE.finditer(text)
    ]


def parse_hal_profile_ops(text: str) -> dict[str, dict[str, float | int]] | None:
    ops = {
        match.group("op"): {
            "calls": int(match.group("calls")),
            "mean_ms": float(match.group("mean_ms")),
            "total_ms": float(match.group("total_ms")),
            "max_ms": float(match.group("max_ms")),
            "total_bytes": int(match.group("total_bytes")),
        }
        for match in HAL_PROFILE_OP_RE.finditer(text)
    }
    return ops or None


def parse_runner_env_overrides(items: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"runner env override must be KEY=VALUE, got: {item!r}")
        overrides[key] = value
    return overrides


def requires_flm_first_class_evidence(args: argparse.Namespace) -> bool:
    model_dir = getattr(args, "model_dir", None)
    return isinstance(model_dir, Path) and model_dir.suffix == ".flm"


def flm_first_class_validation_errors(args: argparse.Namespace, row: dict) -> list[str]:
    errors: list[str] = []
    if (
        getattr(args, "model", None) is None
        and row.get("resolved_model") != DEFAULT_35B_A3B_MODEL
    ):
        errors.append("FLM run did not report inferred model from runtime descriptor")
    if row.get("flm_weight_mode") != "INT4 native FLM":
        errors.append("FLM run did not report INT4 native FLM weight mode")
    if row.get("flm_ready_for_decode") is not True:
        errors.append("FLM run did not report ready-for-decode YES")
    direct_profile = row.get("flm_direct_profile")
    if (
        not isinstance(direct_profile, dict)
        or int(direct_profile.get("native_int4", 0)) <= 0
        or int(direct_profile.get("bf16_fallback", 0)) != 0
    ):
        errors.append("FLM run did not report native INT4 direct plan coverage")
    ownership = row.get("runtime_engine_ownership_markers")
    if not isinstance(ownership, list) or len(ownership) != 1:
        errors.append("FLM run did not report exactly one runtime engine ownership marker")
    elif ownership[0] != {"load_sequence": 1, "source_open_count": 1}:
        errors.append(
            "FLM runtime engine ownership marker did not report "
            "load_sequence=1 source_open_count=1"
        )
    return errors


def apply_target_profile(args: argparse.Namespace) -> None:
    profile = TARGET_PROFILES[args.target_profile]
    if args.model is None:
        args.model = profile["model"]
    if args.model_dir is None:
        args.model_dir = profile["model_dir"]
    if args.quant is None:
        args.quant = profile["quant"]
    if args.out_json is None:
        args.out_json = profile["out_json"]


def apply_lucebox_serving_mode(args: argparse.Namespace) -> None:
    args.prompt_source = "jsonl"
    args.prompt_format = "chatml-no-thinking"
    args.ignore_eos = False
    if args.target_profile == "qwen36-27b-lucebox":
        args.dflash = True
        args.dflash_draft_variant = args.dflash_draft_variant or "lucebox-q4-k-m"
    if args.context_size == DEFAULT_CONTEXT_SIZE:
        args.context_size = LUCEBOX_SERVING_CONTEXT_SIZE


def run_one(args: argparse.Namespace, name: str, prompt: str, warmup: bool = False) -> dict:
    dflash_draft_dir, dflash_draft_gguf, dflash_draft_label = resolve_dflash_draft(args)
    cmd = [
        str(args.binary),
        "--backend",
        args.backend,
        "--model-dir",
        str(args.model_dir),
        "--prompt",
        prompt,
        "--context-size",
        str(args.context_size),
        "--max-new-tokens",
        str(args.warmup_new_tokens if warmup else args.n_gen),
        "--temperature",
        "0",
        "--top-k",
        "1",
        "--sampling-seed",
        str(args.seed),
        "--no-download",
    ]
    if args.model is not None:
        cmd.extend(["--model", args.model])
    if args.prompt_no_special_tokens:
        cmd.append("--prompt-no-special-tokens")
    allow_untested_gpu = getattr(args, "allow_untested_gpu", None)
    if allow_untested_gpu:
        cmd.extend(["--allow-untested-gpu", allow_untested_gpu])
    flm_virtual_transfer_backend = getattr(args, "flm_virtual_transfer_backend", None)
    if flm_virtual_transfer_backend:
        cmd.extend(["--flm-virtual-transfer-backend", flm_virtual_transfer_backend])
    if args.quant != "none":
        cmd.append(f"--{args.quant}")
    if args.ignore_eos:
        cmd.append("--ignore-eos")
    if args.emit_stage_timings:
        cmd.append("--emit-stage-timings")
    if args.kv_fp8:
        cmd.append("--kv-fp8")
    if args.dflash:
        cmd.append("--dflash")
        cmd.extend(["--dflash-draft-dir", str(dflash_draft_dir)])
        if args.dflash_block:
            cmd.extend(["--dflash-block", str(args.dflash_block)])

    env = os.environ.copy()
    env["SUPERSONIC_BACKENDS"] = args.backend
    if getattr(args, "hal_profile", False):
        # This runner hook emits gpu-hal rows as [hal-profile-op] lines.
        env["SUPERSONIC_METAL_PROFILE"] = "1"
    if dflash_draft_gguf is not None:
        env["SUPERSONIC_DFLASH_DRAFT_GGUF"] = str(dflash_draft_gguf)
    runner_env_overrides = parse_runner_env_overrides(getattr(args, "runner_env", []))
    env.update(runner_env_overrides)

    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=args.timeout,
        env=env,
    )
    elapsed = time.monotonic() - start
    combined = proc.stdout + "\n" + proc.stderr
    match = RESULT_RE.search(combined)
    row = {
        "name": name,
        "returncode": proc.returncode,
        "wall_s": elapsed,
        "stdout_tail": proc.stdout[-args.tail_chars :],
        "stderr_tail": proc.stderr[-args.tail_chars :],
    }
    if runner_env_overrides:
        row["runner_env"] = runner_env_overrides
    if flm_virtual_transfer_backend:
        row["flm_virtual_transfer_backend"] = flm_virtual_transfer_backend
    if match:
        row.update(
            {
                "prompt_tokens": int(match.group("prompt")),
                "generated_tokens": int(match.group("generated")),
                "requested_tokens": args.warmup_new_tokens if warmup else args.n_gen,
                "stopped_early": int(match.group("generated"))
                < (args.warmup_new_tokens if warmup else args.n_gen),
                "decode_ms": float(match.group("decode_ms")),
                "ms_per_step": float(match.group("ms_per_step")),
                "tok_s": 1000.0 / float(match.group("ms_per_step")),
            }
        )
    lifecycle_timings = parse_lifecycle_timings(combined)
    if lifecycle_timings:
        row["lifecycle_timings"] = lifecycle_timings
    startup_timings = parse_startup_timings(combined)
    if startup_timings:
        row["startup_timings"] = startup_timings
    sparse_residency = parse_sparse_residency(combined)
    if sparse_residency:
        row["sparse_residency"] = sparse_residency
    sparse_breakdown = parse_sparse_breakdown(combined)
    if sparse_breakdown:
        row["sparse_breakdown"] = sparse_breakdown
    resolved_model = parse_inferred_model(combined)
    if resolved_model:
        row["resolved_model"] = resolved_model
    flm_weight_mode = parse_flm_weight_mode(combined)
    if flm_weight_mode:
        row["flm_weight_mode"] = flm_weight_mode
    flm_direct_profile = parse_flm_direct_profile(combined)
    if flm_direct_profile:
        row["flm_direct_profile"] = flm_direct_profile
    flm_ready = parse_flm_ready_for_decode(combined)
    if flm_ready is not None:
        row["flm_ready_for_decode"] = flm_ready["ready"]
        if "detail" in flm_ready:
            row["flm_ready_for_decode_detail"] = flm_ready["detail"]
    runtime_ownership = parse_runtime_engine_ownership(combined)
    if runtime_ownership:
        row["runtime_engine_ownership_markers"] = runtime_ownership
    hal_profile_ops = parse_hal_profile_ops(combined)
    if hal_profile_ops:
        row["hal_profile_ops"] = hal_profile_ops
    if args.dflash:
        row["dflash_draft_label"] = dflash_draft_label
        row["dflash_draft_dir"] = str(dflash_draft_dir)
        row["dflash_draft_gguf"] = str(dflash_draft_gguf) if dflash_draft_gguf else None
    if proc.returncode == 0 and requires_flm_first_class_evidence(args):
        validation_errors = flm_first_class_validation_errors(args, row)
        if validation_errors:
            row["runner_returncode"] = proc.returncode
            row["returncode"] = 1
            row["benchmark_validation_errors"] = validation_errors
    return row


def mean_common_numeric_field(rows: list[dict], field: str) -> dict[str, float] | None:
    keys = sorted(
        {
            key
            for row in rows
            for key in row.get(field, {}).keys()
            if all(key in other.get(field, {}) for other in rows)
        }
    )
    if not keys:
        return None
    return {
        key: sum(row[field][key] for row in rows) / len(rows)
        for key in keys
    }


def gib_per_second(bytes_count: int | float, elapsed_ms: int | float) -> float:
    if elapsed_ms <= 0:
        return 0.0
    return (float(bytes_count) / GIB) / (float(elapsed_ms) / 1000.0)


def build_flm_load_speed_summary(rows: list[dict]) -> dict[str, float | int] | None:
    summary: dict[str, float | int] = {}
    h2d_rows = [
        row
        for row in rows
        if "layer_load_copy_h_to_d_bytes" in row.get("lifecycle_timings", {})
        and "layer_load_copy_h_to_d_ms" in row.get("lifecycle_timings", {})
    ]
    if h2d_rows:
        h2d_bytes = sum(
            int(row["lifecycle_timings"]["layer_load_copy_h_to_d_bytes"])
            for row in h2d_rows
        )
        h2d_ms = sum(
            float(row["lifecycle_timings"]["layer_load_copy_h_to_d_ms"])
            for row in h2d_rows
        )
        if h2d_bytes > 0 and h2d_ms > 0:
            summary.update(
                {
                    "layer_load_copy_h_to_d_bytes": h2d_bytes,
                    "layer_load_copy_h_to_d_ms": h2d_ms,
                    "layer_load_copy_h_to_d_gib_s": gib_per_second(h2d_bytes, h2d_ms),
                }
            )

    for op in ("copy_h2d", "copy_storage_to_device"):
        op_rows = [row for row in rows if op in row.get("hal_profile_ops", {})]
        if not op_rows:
            continue
        op_bytes = sum(
            int(row["hal_profile_ops"][op]["total_bytes"])
            for row in op_rows
        )
        op_ms = sum(
            float(row["hal_profile_ops"][op]["total_ms"])
            for row in op_rows
        )
        if op_bytes <= 0 or op_ms <= 0:
            continue
        summary.update(
            {
                f"{op}_bytes": op_bytes,
                f"{op}_ms": op_ms,
                f"{op}_gib_s": gib_per_second(op_bytes, op_ms),
            }
        )
    return summary or None


def build_summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("returncode") == 0 and "tok_s" in r]
    total_generated = sum(r["generated_tokens"] for r in ok)
    total_decode_ms = sum(r["decode_ms"] for r in ok)
    summary = {
        "count": len(ok),
        "mean_tok_s": sum(r["tok_s"] for r in ok) / len(ok),
        "weighted_tok_s": (1000.0 * total_generated / total_decode_ms)
        if total_decode_ms
        else 0.0,
        "mean_ms_per_step": sum(r["ms_per_step"] for r in ok) / len(ok),
        "min_tok_s": min(r["tok_s"] for r in ok),
        "max_tok_s": max(r["tok_s"] for r in ok),
        "total_generated_tokens": total_generated,
        "total_decode_ms": total_decode_ms,
        "stopped_early_count": sum(1 for r in ok if r.get("stopped_early")),
    }
    lifecycle_keys = sorted(
        {
            key
            for row in ok
            for key in row.get("lifecycle_timings", {}).keys()
            if all(key in other.get("lifecycle_timings", {}) for other in ok)
        }
    )
    if lifecycle_keys:
        summary["mean_lifecycle_timings"] = {
            key: sum(row["lifecycle_timings"][key] for row in ok) / len(ok)
            for key in lifecycle_keys
        }
    startup_keys = sorted(
        {
            key
            for row in ok
            for key in row.get("startup_timings", {}).keys()
            if all(key in other.get("startup_timings", {}) for other in ok)
        }
    )
    if startup_keys:
        summary["mean_startup_timings"] = {
            key: sum(row["startup_timings"][key] for row in ok) / len(ok)
            for key in startup_keys
        }
    mean_sparse_residency = mean_common_numeric_field(ok, "sparse_residency")
    if mean_sparse_residency:
        summary["mean_sparse_residency"] = mean_sparse_residency
    mean_sparse_breakdown = mean_common_numeric_field(ok, "sparse_breakdown")
    if mean_sparse_breakdown:
        summary["mean_sparse_breakdown"] = mean_sparse_breakdown
    flm_weight_modes = sorted(
        {row["flm_weight_mode"] for row in ok if row.get("flm_weight_mode")}
    )
    if flm_weight_modes:
        summary["flm_weight_modes"] = flm_weight_modes
    if any("flm_ready_for_decode" in row for row in ok):
        summary["flm_ready_for_decode_count"] = sum(
            1 for row in ok if row.get("flm_ready_for_decode")
        )
    if any("runtime_engine_ownership_markers" in row for row in ok):
        summary["runtime_engine_ready_count"] = sum(
            len(row.get("runtime_engine_ownership_markers", [])) for row in ok
        )
    flm_direct_profiles = []
    for row in ok:
        profile = row.get("flm_direct_profile")
        if profile and profile not in flm_direct_profiles:
            flm_direct_profiles.append(profile)
    if flm_direct_profiles:
        summary["flm_direct_profiles"] = flm_direct_profiles
    hal_op_names = sorted(
        {
            op
            for row in ok
            for op in row.get("hal_profile_ops", {}).keys()
            if all(op in other.get("hal_profile_ops", {}) for other in ok)
        }
    )
    if hal_op_names:
        hal_metrics = ("calls", "mean_ms", "total_ms", "max_ms", "total_bytes")
        summary["mean_hal_profile_ops"] = {
            op: {
                metric: sum(row["hal_profile_ops"][op][metric] for row in ok) / len(ok)
                for metric in hal_metrics
            }
            for op in hal_op_names
        }
    flm_load_speed = build_flm_load_speed_summary(ok)
    if flm_load_speed:
        summary["flm_load_speed"] = flm_load_speed
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument(
        "--target-profile",
        choices=sorted(TARGET_PROFILES),
        default="qwen36-27b-lucebox",
        help=(
            "Default model/model-dir/quant/output bundle. Explicit individual "
            "arguments still take precedence."
        ),
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--quant",
        choices=["q4km-gptq", "q4km", "int4", "none"],
        default=None,
    )
    parser.add_argument("--lucebox-bench", type=Path, default=DEFAULT_LUCEBOX_BENCH)
    parser.add_argument("--lucebox-jsonl", type=Path, default=DEFAULT_LUCEBOX_HE_JSONL)
    parser.add_argument("--prompt-source", choices=["script", "jsonl"], default="script")
    parser.add_argument(
        "--prompt-format",
        choices=["raw", "chatml", "chatml-no-thinking"],
        default="raw",
    )
    parser.add_argument(
        "--lucebox-serving-mode",
        action="store_true",
        help=(
            "Preset for Lucebox HTTP-style comparison: JSONL HE prompts, "
            "Qwen3 no-thinking ChatML prompt text, stop on EOS, and "
            "Lucebox Q4_K_M draft GGUF."
        ),
    )
    parser.add_argument("--backend", default="hip")
    parser.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE)
    parser.add_argument("--n-gen", type=int, default=256)
    parser.add_argument("--warmup-new-tokens", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--tail-chars", type=int, default=4000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-on-eos", action="store_true")
    parser.add_argument(
        "--prompt-no-special-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--emit-stage-timings", action="store_true")
    parser.add_argument(
        "--flm-virtual-transfer-backend",
        choices=["pageable-h2d", "gpu-direct-storage", "gds", "hipfile"],
        default=None,
        help=(
            "Forward SuperSonic's FLM virtual transfer backend selector. "
            "Explicit CLI selection takes precedence over runner env overrides."
        ),
    )
    parser.add_argument(
        "--hal-profile",
        action="store_true",
        help="Enable the runner HAL op dump and parse [hal-profile-op] rows into JSON.",
    )
    parser.add_argument(
        "--runner-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Forward an environment override to the SuperSonic process. "
            "Repeat for multiple overrides."
        ),
    )
    parser.add_argument("--kv-fp8", action="store_true")
    parser.add_argument(
        "--allow-untested-gpu",
        default=None,
        help=(
            "Forward SuperSonic's --allow-untested-gpu override, e.g. reuse "
            "gfx1100 registry policy on a newly detected HIP arch."
        ),
    )
    parser.add_argument("--dflash", action="store_true")
    parser.add_argument(
        "--dflash-draft-variant",
        choices=sorted(LUCEBOX_DRAFT_ALIASES),
        default=None,
        help=(
            "Convenience draft selection. Lucebox variants use their GGUF draft "
            "weights with the local SuperSonic DFlash config directory."
        ),
    )
    parser.add_argument(
        "--dflash-draft-dir",
        type=Path,
        default=DEFAULT_SUPERSONIC_DFLASH_DRAFT_DIR,
    )
    parser.add_argument(
        "--dflash-draft-gguf",
        type=Path,
        default=None,
        help=(
            "Optional Lucebox-style GGUF draft weights. The config still comes "
            "from --dflash-draft-dir."
        ),
    )
    parser.add_argument("--dflash-block", type=int, default=0)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    apply_target_profile(args)
    if args.lucebox_serving_mode:
        apply_lucebox_serving_mode(args)
    if args.stop_on_eos:
        args.ignore_eos = False
    runner_env_overrides = parse_runner_env_overrides(args.runner_env)

    if not args.binary.exists():
        raise FileNotFoundError(args.binary)
    if args.dflash:
        dflash_draft_dir, dflash_draft_gguf, _draft_label = resolve_dflash_draft(args)
        if not dflash_draft_dir.exists():
            raise FileNotFoundError(dflash_draft_dir)
        if dflash_draft_gguf is not None and not dflash_draft_gguf.exists():
            raise FileNotFoundError(dflash_draft_gguf)
    prompts = load_prompts(args)
    if args.start_index < 0 or args.start_index > len(prompts):
        raise ValueError(f"--start-index must be in 0..{len(prompts)}")
    prompts = prompts[args.start_index :]
    if args.limit > 0:
        prompts = prompts[: args.limit]

    rows = []
    if args.warmup and prompts:
        print(f"[warmup] {prompts[0][0]} {args.warmup_new_tokens} tokens", flush=True)
        warmup = run_one(args, prompts[0][0], prompts[0][1], warmup=True)
        if warmup["returncode"] != 0:
            print(warmup["stderr_tail"] or warmup["stdout_tail"], file=sys.stderr)

    print(f"{'prompt':28s} {'ptok':>5s} {'gen':>5s} {'ms/tok':>8s} {'tok/s':>8s} {'wall_s':>8s}")
    print("-" * 70)
    for name, prompt in prompts:
        row = run_one(args, name, prompt)
        rows.append(row)
        if row["returncode"] != 0 or "tok_s" not in row:
            print(f"{name:28s} FAILED rc={row['returncode']}")
            print(row["stderr_tail"] or row["stdout_tail"], file=sys.stderr)
            continue
        print(
            f"{name:28s} {row['prompt_tokens']:5d} {row['generated_tokens']:5d} "
            f"{row['ms_per_step']:8.2f} {row['tok_s']:8.2f} {row['wall_s']:8.1f}",
            flush=True,
        )

    ok = [r for r in rows if r.get("returncode") == 0 and "tok_s" in r]
    if not ok:
        return 1
    summary = build_summary(rows)
    dflash_draft_dir, dflash_draft_gguf, dflash_draft_label = resolve_dflash_draft(args)
    resolved_model = args.model or next(
        (row["resolved_model"] for row in rows if row.get("resolved_model")),
        None,
    )
    payload = {
        "schema": "supersonic-qwen36-he-comparison-v2",
        "model": args.model,
        "resolved_model": resolved_model,
        "model_dir": str(args.model_dir),
        "quant": args.quant,
        "dflash": args.dflash,
        "dflash_draft_label": dflash_draft_label if args.dflash else None,
        "dflash_draft_dir": str(dflash_draft_dir) if args.dflash else None,
        "dflash_draft_gguf": str(dflash_draft_gguf) if args.dflash and dflash_draft_gguf else None,
        "dflash_block": args.dflash_block if args.dflash_block else None,
        "backend": args.backend,
        "allow_untested_gpu": args.allow_untested_gpu,
        "flm_virtual_transfer_backend": args.flm_virtual_transfer_backend,
        "context_size": args.context_size,
        "n_gen": args.n_gen,
        "eos_policy": "ignore" if args.ignore_eos else "stop",
        "prompt_source": args.prompt_source,
        "prompt_format": args.prompt_format,
        "lucebox_bench": str(args.lucebox_bench),
        "lucebox_jsonl": str(args.lucebox_jsonl),
        "lucebox_serving_mode": args.lucebox_serving_mode,
        "runner_env": runner_env_overrides or None,
        "summary": summary,
        "rows": rows,
    }
    if "flm_weight_modes" in summary:
        payload["flm_weight_modes"] = summary["flm_weight_modes"]
    if "flm_ready_for_decode_count" in summary:
        payload["flm_ready_for_decode_count"] = summary["flm_ready_for_decode_count"]
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2))
    print("-" * 70)
    print(f"{'MEAN':28s} {'':5s} {'':5s} {summary['mean_ms_per_step']:8.2f} {summary['mean_tok_s']:8.2f}")
    print(f"[wrote] {args.out_json}")
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
