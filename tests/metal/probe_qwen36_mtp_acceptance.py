#!/usr/bin/env python3
"""Probe Qwen3.6-MoE MTP acceptance telemetry.

The Metal runtime still blocks ``--speculative-decode`` at policy level, but
this harness is the measurement gate we need before lifting that policy: it
captures the policy-blocked row today and parses ``[qwen36-mtp-acceptance]``
rows as soon as a backend can run the path.
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
SCHEMA = "qwen36-moe-mtp-acceptance-probe-v2"
POLICY_BLOCKED_NEEDLE = "does not wire the MTP/speculative decode path yet"
METAL_EXPERIMENT_ENV = "SUPERSONIC_QWEN36_METAL_MTP_EXPERIMENT"
ACCEPTANCE_PROFILE_ENV = "SUPERSONIC_QWEN36_MTP_ACCEPTANCE_PROFILE"
BATCHED_PREFILL_ENV = "SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL"
DEFAULT_PROMPT = (
    "Explain how a local inference runtime should measure native MTP "
    "acceptance separately from base-model FFN latency."
)

FLOAT_KEYS = {
    "acceptance_rate",
    "emitted_per_step",
    "target_steps_per_emitted",
}
INT_KEYS = {
    "steps",
    "drafted_tokens",
    "accepted_tokens",
    "emitted_tokens",
    "base_steps",
    "replay_steps",
    "full_accept_steps",
    "zero_accept_steps",
    "max_accept",
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


def parse_mtp_acceptance(output: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for line in output.splitlines():
        if not line.startswith("[qwen36-mtp-acceptance]"):
            continue
        fields = parse_key_values(line)
        row: dict[str, Any] = {}
        for key, value in fields.items():
            if key in INT_KEYS:
                row[key] = int(value)
            elif key in FLOAT_KEYS:
                row[key] = float(value)
            else:
                row[key] = value
        summary = row
    return summary


def classify_run(returncode: int, output: str, acceptance: dict[str, Any]) -> str:
    if acceptance:
        return "measured" if returncode == 0 else "measured_failed"
    if POLICY_BLOCKED_NEEDLE in output:
        return "policy_blocked"
    if returncode != 0:
        return "failed"
    return "missing_acceptance"


def build_report(
    output: str,
    returncode: int,
    command: list[str],
    wall_seconds: float,
    backend: str,
    batched_spec_verify: bool,
    env_overrides: dict[str, str],
) -> dict[str, Any]:
    acceptance = parse_mtp_acceptance(output)
    status = classify_run(returncode, output, acceptance)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "model": MODEL,
        "backend": backend,
        "mode": "batched" if batched_spec_verify else "sequential",
        "status": status,
        "returncode": returncode,
        "wall_seconds": wall_seconds,
        "command": command,
        "env_overrides": env_overrides,
        "acceptance": acceptance,
        "policy_blocked": status == "policy_blocked",
        "metal_profile": parse_profile(output, "[metal-profile]", "[metal-profile-op]"),
        "hal_profile": parse_profile(output, "[hal-profile]", "[hal-profile-op]"),
    }
    if status != "measured":
        report["output_tail"] = output[-5000:]
    return report


def top_profile_op(profile: dict[str, Any] | None) -> dict[str, Any]:
    if not profile:
        return {}
    entries = profile.get("entries") or []
    return max(entries, key=lambda item: item.get("total_ms") or 0.0) if entries else {}


def render_markdown(report: dict[str, Any]) -> str:
    acceptance = report.get("acceptance") or {}
    top_metal = top_profile_op(report.get("metal_profile"))
    hal_total = ((report.get("hal_profile") or {}).get("summary") or {}).get("total_ms")
    lines = [
        "# Qwen3.6 MTP Acceptance Probe",
        "",
        f"- backend: `{report.get('backend')}`",
        f"- mode: `{report.get('mode')}`",
        f"- status: `{report.get('status')}`",
        f"- returncode: `{report.get('returncode')}`",
        "",
    ]
    if acceptance:
        lines.extend(
            [
                "| Steps | Drafted | Accepted | Acceptance | Emitted | Base steps | Replay steps | Target steps/emitted |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
                "| {steps} | {drafted} | {accepted} | {rate:.1%} | {emitted} | {base} | {replay} | {target:.3f} |".format(
                    steps=acceptance.get("steps", 0),
                    drafted=acceptance.get("drafted_tokens", 0),
                    accepted=acceptance.get("accepted_tokens", 0),
                    rate=acceptance.get("acceptance_rate", 0.0),
                    emitted=acceptance.get("emitted_tokens", 0),
                    base=acceptance.get("base_steps", 0),
                    replay=acceptance.get("replay_steps", 0),
                    target=acceptance.get("target_steps_per_emitted", 0.0),
                ),
                "",
            ]
        )
    elif report.get("policy_blocked"):
        lines.extend(
            [
                "Metal speculative decode is still policy-blocked. This is the expected status until the MTP path is explicitly enabled for Metal.",
                "",
            ]
        )
    else:
        lines.extend(["No acceptance row was captured.", ""])
    if top_metal or hal_total is not None:
        lines.extend(
            [
                "| Top Metal op | Top Metal ms | HAL ms |",
                "|:---|---:|---:|",
                "| {op} | {metal_ms} | {hal_ms} |".format(
                    op=top_metal.get("op") or "-",
                    metal_ms=(
                        f"{top_metal.get('total_ms'):.3f}"
                        if top_metal.get("total_ms") is not None
                        else "-"
                    ),
                    hal_ms=f"{hal_total:.3f}" if hal_total is not None else "-",
                ),
                "",
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


def run_supersonic(args: argparse.Namespace) -> tuple[str, float, int, list[str]]:
    env = os.environ.copy()
    env.update(build_env_overrides(args))

    cmd = [
        str(args.binary),
        "--backend",
        args.backend,
        "--model",
        MODEL,
        "--model-dir",
        str(args.model_dir),
        "--int4",
        "--prompt",
        args.prompt,
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
        "--speculative-decode",
    ]
    if args.batched_spec_verify:
        cmd.append("--batched-spec-verify")

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=args.timeout,
            env=env,
        )
        output = proc.stdout + proc.stderr
        return output, time.monotonic() - start, proc.returncode, cmd
    except subprocess.TimeoutExpired as exc:
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
        return stdout + stderr, time.monotonic() - start, -1, cmd


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/release/supersonic"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--backend", choices=("metal", "hip", "cuda"), default="metal")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--context-size", type=int, default=256)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=5,
        help="5 allows the first extension to draft after the base token; Metal experiment forces K=1.",
    )
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--batched-spec-verify", action="store_true")
    parser.add_argument(
        "--metal-experiment",
        action="store_true",
        help=f"set {METAL_EXPERIMENT_ENV}=1 to run the env-gated Metal K=1 path",
    )
    parser.add_argument(
        "--metal-profile",
        action="store_true",
        help="set SUPERSONIC_METAL_PROFILE=1 and retain parsed Metal/HAL profile rows",
    )
    parser.add_argument(
        "--log",
        type=Path,
        help="Parse an existing combined stdout/stderr log instead of running supersonic.",
    )
    parser.add_argument(
        "--returncode",
        type=int,
        default=0,
        help="Return code to associate with --log input.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("target/qwen36_mtp_acceptance_probe.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("target/qwen36_mtp_acceptance_probe.md"),
    )
    return parser.parse_args(argv)


def build_env_overrides(args: argparse.Namespace) -> dict[str, str]:
    overrides = {
        "SUPERSONIC_BACKENDS": args.backend,
        ACCEPTANCE_PROFILE_ENV: "1",
    }
    if args.backend == "metal":
        overrides[BATCHED_PREFILL_ENV] = "0"
    if args.metal_experiment:
        overrides[METAL_EXPERIMENT_ENV] = "1"
    if getattr(args, "metal_profile", False):
        overrides["SUPERSONIC_METAL_PROFILE"] = "1"
    return overrides


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.model_dir = resolve_model_dir(args.model_dir, os.environ)
    env_overrides = build_env_overrides(args)

    if args.log:
        output = args.log.read_text()
        wall_seconds = 0.0
        returncode = args.returncode
        command = ["<parsed-log>", str(args.log)]
    else:
        output, wall_seconds, returncode, command = run_supersonic(args)

    report = build_report(
        output,
        returncode,
        command,
        wall_seconds,
        args.backend,
        args.batched_spec_verify,
        env_overrides,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(report))

    acceptance = report.get("acceptance") or {}
    if acceptance:
        print(
            "[qwen36-mtp-acceptance-probe] status={} backend={} mode={} acceptance_rate={:.6f} target_steps_per_emitted={:.6f}".format(
                report["status"],
                report["backend"],
                report["mode"],
                acceptance.get("acceptance_rate", 0.0),
                acceptance.get("target_steps_per_emitted", 0.0),
            )
        )
    else:
        print(
            "[qwen36-mtp-acceptance-probe] status={} backend={} mode={}".format(
                report["status"], report["backend"], report["mode"]
            )
        )
    print(f"[wrote] {args.out_json}")
    print(f"[wrote] {args.out_md}")

    if report["status"] in {"failed", "missing_acceptance", "measured_failed"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
