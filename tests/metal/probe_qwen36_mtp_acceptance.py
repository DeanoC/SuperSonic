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
SCHEMA = "qwen36-moe-mtp-acceptance-probe-v1"
POLICY_BLOCKED_NEEDLE = "does not wire the MTP/speculative decode path yet"
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
        "acceptance": acceptance,
        "policy_blocked": status == "policy_blocked",
    }
    if status != "measured":
        report["output_tail"] = output[-5000:]
    return report


def render_markdown(report: dict[str, Any]) -> str:
    acceptance = report.get("acceptance") or {}
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
    env["SUPERSONIC_BACKENDS"] = args.backend
    env["SUPERSONIC_QWEN36_MTP_ACCEPTANCE_PROFILE"] = "1"

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
        help="5 allows the first MTP extension to draft K=3 after the base token.",
    )
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--batched-spec-verify", action="store_true")
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


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.model_dir = resolve_model_dir(args.model_dir, os.environ)

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
