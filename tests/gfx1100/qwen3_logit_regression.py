#!/usr/bin/env python3
"""Compare Qwen3-30B-A3B one-token logits across SuperSonic binaries.

This is a small regression harness for persistent-attention experiments. It
runs deterministic one-token decodes, compares candidate persistent logits to
a reference binary, and optionally records the reference persistent-vs-chained
drift that already exists at HEAD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_DIR = Path(
    os.environ.get("SUPERSONIC_QWEN3_30B_A3B_DIR", "/mnt/data/models/Qwen3-30B-A3B")
)
DEFAULT_REFERENCE_BIN = Path("target/qwen3_reference/supersonic")
DEFAULT_CANDIDATE_BIN = Path("target/release/supersonic")
DEFAULT_OUT_JSON = Path("target/qwen3_logit_regression/report.json")
LONG_PROMPT_UNIT = (
    "Attention kernels dominate decode as context grows and this profiling prompt "
    "is repeated for Qwen3 persistent attention measurements."
)


@dataclass(frozen=True)
class Case:
    name: str
    prompt: str
    context_size: int


def parse_int_list(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("list must contain at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError("values must be positive integers")
    seen: set[int] = set()
    deduped: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def available_cases() -> dict[int, Case]:
    return {
        8: Case("ctx8", "Hello world", 8),
        64: Case(
            "ctx64",
            (
                "Qwen3 drift check prompt with enough words to exercise several "
                "attention cache positions while staying inside a sixty four token "
                "context window for deterministic one token comparison across "
                "persistent and chained decode paths."
            ),
            64,
        ),
        210: Case("ctx210", " ".join([LONG_PROMPT_UNIT] * 10), 256),
        504: Case("ctx504", " ".join([LONG_PROMPT_UNIT] * 24), 640),
    }


def select_cases(contexts: list[int]) -> list[Case]:
    cases = available_cases()
    missing = [ctx for ctx in contexts if ctx not in cases]
    if missing:
        valid = ", ".join(str(ctx) for ctx in sorted(cases))
        raise ValueError(f"unsupported context labels {missing}; valid labels: {valid}")
    return [cases[ctx] for ctx in contexts]


def parse_key_value_floats(raw: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for part in raw.split():
        if "=" not in part:
            continue
        key, value = part.strip("()").split("=", 1)
        try:
            values[key] = float(value)
        except ValueError:
            pass
    return values


def parse_stage_timings(output: str) -> dict[str, float | str]:
    match = re.search(r"\[qwen3-moe stage-timings\]\s+(.+)", output)
    if not match:
        return {}
    raw = match.group(1)
    timings: dict[str, float | str] = {}
    for part in raw.split():
        if "=" not in part:
            continue
        key, value = part.strip("()").split("=", 1)
        try:
            timings[key] = float(value)
        except ValueError:
            timings[key] = value
    return timings


def parse_prompt_tokens(output: str) -> int | None:
    match = re.search(r"prompt:\s+.*?\s+->\s+([0-9]+)\s+tokens?", output)
    return int(match.group(1)) if match else None


def parse_generated_count(output: str) -> int | None:
    match = re.search(r"Generated\s+([0-9]+)\s+tokens?", output)
    return int(match.group(1)) if match else None


def parse_last_logits(output: str) -> list[float]:
    match = re.search(r"^LAST_LOGITS:\s*(.+)$", output, flags=re.MULTILINE)
    if not match:
        raise ValueError("LAST_LOGITS line not found")
    raw = match.group(1).strip()
    if not raw:
        raise ValueError("LAST_LOGITS line is empty")
    return [float(part) for part in raw.split(",")]


def top_k_indices(values: list[float], k: int) -> list[int]:
    return sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)[:k]


def logit_hash(values: list[float]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(f"{value:.9g},".encode("ascii"))
    return h.hexdigest()


def compare_logits(reference: list[float], candidate: list[float], top_k: int = 10) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise ValueError(f"logit length mismatch: {len(reference)} vs {len(candidate)}")
    if not reference:
        raise ValueError("logits are empty")

    dot = 0.0
    ref_norm = 0.0
    cand_norm = 0.0
    max_abs = 0.0
    for ref, cand in zip(reference, candidate):
        dot += ref * cand
        ref_norm += ref * ref
        cand_norm += cand * cand
        max_abs = max(max_abs, abs(ref - cand))

    ref_top = top_k_indices(reference, top_k)
    cand_top = top_k_indices(candidate, top_k)
    ref_set = set(ref_top)
    cand_set = set(cand_top)
    ref_argmax = ref_top[0]
    cand_argmax = cand_top[0]
    denom = math.sqrt(ref_norm) * math.sqrt(cand_norm)
    cosine = dot / denom if denom else 0.0

    return {
        "length": len(reference),
        "reference_argmax": ref_argmax,
        "candidate_argmax": cand_argmax,
        "argmax_match": ref_argmax == cand_argmax,
        "top10_overlap": len(ref_set & cand_set),
        "top10_reference": ref_top,
        "top10_candidate": cand_top,
        "cosine": cosine,
        "max_abs": max_abs,
        "reference_hash": logit_hash(reference),
        "candidate_hash": logit_hash(candidate),
    }


def comparison_passes(
    metrics: dict[str, Any],
    *,
    require_argmax: bool,
    top10_overlap_floor: int,
    cosine_floor: float,
    max_abs_ceil: float,
) -> bool:
    return (
        (not require_argmax or bool(metrics["argmax_match"]))
        and int(metrics["top10_overlap"]) >= top10_overlap_floor
        and float(metrics["cosine"]) >= cosine_floor
        and float(metrics["max_abs"]) <= max_abs_ceil
    )


def run_one(
    binary: Path,
    model_dir: Path,
    case: Case,
    *,
    backend: str,
    persistent: bool,
    timeout: int,
    seed: int,
) -> dict[str, Any]:
    cmd = [
        str(binary),
        "--backend",
        backend,
        "--model",
        "qwen3-30b-a3b",
        "--model-dir",
        str(model_dir),
        "--prompt",
        case.prompt,
        "--max-new-tokens",
        "1",
        "--context-size",
        str(case.context_size),
        "--int4",
        "--no-download",
        "--temperature",
        "0",
        "--top-k",
        "1",
        "--sampling-seed",
        str(seed),
        "--dump-last-logits",
        "--emit-stage-timings",
    ]
    if not persistent:
        cmd.append("--no-persistent-decode")

    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return {
            "binary": str(binary),
            "path": "persistent" if persistent else "chained",
            "returncode": -1,
            "wall_seconds": elapsed,
            "timeout_seconds": timeout,
            "command": cmd,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-4000:],
            "error": f"timed out after {timeout} seconds",
        }

    elapsed = time.monotonic() - start
    output = proc.stdout + proc.stderr
    row: dict[str, Any] = {
        "binary": str(binary),
        "path": "persistent" if persistent else "chained",
        "returncode": proc.returncode,
        "wall_seconds": elapsed,
        "command": cmd,
        "prompt_tokens": parse_prompt_tokens(output),
        "generated_tokens": parse_generated_count(output),
        "stage": parse_stage_timings(output),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if proc.returncode != 0:
        row["error"] = output[-6000:]
        return row
    try:
        logits = parse_last_logits(output)
    except ValueError as exc:
        row["returncode"] = -2
        row["error"] = str(exc)
        return row
    row["logits"] = logits
    row["logits_len"] = len(logits)
    row["logits_hash"] = logit_hash(logits)
    row["argmax"] = top_k_indices(logits, 1)[0]
    row["top10"] = top_k_indices(logits, 10)
    return row


def summarize_run(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"logits", "command", "stdout_tail", "stderr_tail"}
    }


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    cases = select_cases(parse_int_list(args.contexts))
    chained_labels = set(parse_int_list(args.chained_baseline_contexts)) if args.chained_baseline_contexts else set()
    report_cases: list[dict[str, Any]] = []
    failures: list[str] = []

    for case in cases:
        print(f"[qwen3-logit] {case.name}: reference persistent", flush=True)
        reference = run_one(
            args.reference_bin,
            args.model_dir,
            case,
            backend=args.backend,
            persistent=True,
            timeout=args.timeout,
            seed=args.seed,
        )
        print(f"[qwen3-logit] {case.name}: candidate persistent", flush=True)
        candidate = run_one(
            args.candidate_bin,
            args.model_dir,
            case,
            backend=args.backend,
            persistent=True,
            timeout=args.timeout,
            seed=args.seed,
        )

        case_row: dict[str, Any] = {
            "name": case.name,
            "context_size": case.context_size,
            "prompt_chars": len(case.prompt),
            "reference": summarize_run(reference),
            "candidate": summarize_run(candidate),
        }

        if reference.get("returncode") != 0:
            failures.append(f"{case.name}: reference failed with {reference.get('returncode')}")
        if candidate.get("returncode") != 0:
            failures.append(f"{case.name}: candidate failed with {candidate.get('returncode')}")
        if reference.get("returncode") == 0 and candidate.get("returncode") == 0:
            metrics = compare_logits(reference["logits"], candidate["logits"])
            passed = comparison_passes(
                metrics,
                require_argmax=not args.allow_argmax_mismatch,
                top10_overlap_floor=args.top10_overlap_floor,
                cosine_floor=args.cosine_floor,
                max_abs_ceil=args.max_abs_ceil,
            )
            metrics["passed"] = passed
            case_row["candidate_vs_reference"] = metrics
            if not passed:
                failures.append(
                    f"{case.name}: argmax_match={metrics['argmax_match']} "
                    f"top10={metrics['top10_overlap']}/10 cosine={metrics['cosine']:.8f} "
                    f"max_abs={metrics['max_abs']:.6f}"
                )

        label = int(case.name.removeprefix("ctx"))
        if label in chained_labels:
            print(f"[qwen3-logit] {case.name}: reference chained baseline", flush=True)
            chained = run_one(
                args.reference_bin,
                args.model_dir,
                case,
                backend=args.backend,
                persistent=False,
                timeout=args.timeout,
                seed=args.seed,
            )
            case_row["reference_chained"] = summarize_run(chained)
            if chained.get("returncode") == 0 and reference.get("returncode") == 0:
                case_row["reference_persistent_vs_chained"] = compare_logits(
                    chained["logits"], reference["logits"]
                )
            elif chained.get("returncode") != 0:
                failures.append(f"{case.name}: reference chained failed with {chained.get('returncode')}")

        report_cases.append(case_row)

    report = {
        "schema": "qwen3-30b-a3b-logit-regression-v1",
        "model": "qwen3-30b-a3b",
        "model_dir": str(args.model_dir),
        "reference_bin": str(args.reference_bin),
        "candidate_bin": str(args.candidate_bin),
        "thresholds": {
            "require_argmax": not args.allow_argmax_mismatch,
            "top10_overlap_floor": args.top10_overlap_floor,
            "cosine_floor": args.cosine_floor,
            "max_abs_ceil": args.max_abs_ceil,
        },
        "cases": report_cases,
        "failures": failures,
    }
    return report, failures


def print_summary(report: dict[str, Any]) -> None:
    for case in report["cases"]:
        metrics = case.get("candidate_vs_reference")
        if not metrics:
            print(f"{case['name']}: no comparison")
            continue
        ref_stage = case.get("reference", {}).get("stage", {})
        cand_stage = case.get("candidate", {}).get("stage", {})
        ref_ms = ref_stage.get("decode_ms_avg")
        cand_ms = cand_stage.get("decode_ms_avg")
        print(
            f"{case['name']}: pass={metrics['passed']} "
            f"argmax={metrics['candidate_argmax']}/{metrics['reference_argmax']} "
            f"top10={metrics['top10_overlap']}/10 "
            f"cos={metrics['cosine']:.8f} max_abs={metrics['max_abs']:.6f} "
            f"decode_ms candidate/reference={cand_ms}/{ref_ms}"
        )
        baseline = case.get("reference_persistent_vs_chained")
        if baseline:
            print(
                f"{case['name']} chained-baseline: "
                f"argmax={baseline['candidate_argmax']}/{baseline['reference_argmax']} "
                f"top10={baseline['top10_overlap']}/10 "
                f"cos={baseline['cosine']:.8f} max_abs={baseline['max_abs']:.6f}"
            )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-bin", type=Path, default=DEFAULT_REFERENCE_BIN)
    parser.add_argument("--candidate-bin", type=Path, default=DEFAULT_CANDIDATE_BIN)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--backend", default="hip")
    parser.add_argument("--contexts", default="8,64,210,504")
    parser.add_argument(
        "--chained-baseline-contexts",
        default="8,64",
        help="context labels for recording reference persistent-vs-chained drift; empty disables",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cosine-floor", type=float, default=0.99997)
    parser.add_argument("--max-abs-ceil", type=float, default=0.25)
    parser.add_argument("--top10-overlap-floor", type=int, default=10)
    parser.add_argument("--allow-argmax-mismatch", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not args.reference_bin.exists():
        raise SystemExit(f"reference binary not found: {args.reference_bin}")
    if not args.candidate_bin.exists():
        raise SystemExit(f"candidate binary not found: {args.candidate_bin}")
    if not args.model_dir.exists():
        raise SystemExit(f"model dir not found: {args.model_dir}")

    report, failures = build_report(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    print_summary(report)
    print(f"wrote {args.out_json}")
    if failures:
        print("failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
