#!/usr/bin/env python3
"""Run the contributor-only, source-fixed raw-Q6 scalar generation example."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Mapping


OUTPUT_PREFIX = "[supersonic_json] "
OUTPUT_RE = re.compile(r"^\[supersonic_json\]\s+(?P<value>.+)$", re.MULTILINE)
ENGINE_NAME = "supersonic-scalar-lab"
ENGINE_VERSION = "scalar-head-lab-v1"
DEFAULT_BINARY = Path("target/release/examples/scalar_head_lab")
ROUTE_ENVIRONMENT_KEYS = frozenset(("SUPERSONIC_SCALAR_HEAD_ROUTE", "SUPERSONIC_HEAD_ROUTE"))


def build_command(
    *,
    binary: Path,
    model_dir: Path,
    artifact: Path,
    prompt: str,
    max_new_tokens: int,
    device: int,
    mode: str,
    chat: bool,
    honor_eos: bool,
) -> tuple[str, ...]:
    if mode not in ("ordinary", "mtp"):
        raise ValueError("mode must be ordinary or mtp")
    if not prompt:
        raise ValueError("prompt must be non-empty")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if device < 0:
        raise ValueError("device must be non-negative")
    command = [
        str(binary),
        "--model-dir",
        str(model_dir),
        "--artifact",
        str(artifact),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--device",
        str(device),
        "--mode",
        mode,
    ]
    if chat:
        command.append("--chat")
    if not honor_eos:
        command.append("--ignore-eos")
    return tuple(command)


def reject_route_environment(environment: Mapping[str, str]) -> None:
    present = sorted(key for key in ROUTE_ENVIRONMENT_KEYS if key in environment)
    if present:
        raise ValueError(f"route environment variables are forbidden: {', '.join(present)}")


def normalize_output(stdout: str, stderr: str) -> dict[str, object]:
    matches = OUTPUT_RE.findall(stdout)
    if len(matches) != 1:
        raise ValueError("scalar lab output must contain exactly one [supersonic_json] record")
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise ValueError("scalar lab record must be valid JSON") from exc
    required = {
        "decode_ms",
        "engine_name",
        "engine_version",
        "generated_text",
        "generated_tokens",
        "ms_per_tok",
        "prompt_tokens",
        "token_ids",
        "tokens_per_second",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("scalar lab record has unknown or missing fields")
    if payload["engine_name"] != ENGINE_NAME:
        raise ValueError(f"scalar lab engine_name must be {ENGINE_NAME}")
    if payload["engine_version"] != ENGINE_VERSION:
        raise ValueError(f"scalar lab engine_version must be {ENGINE_VERSION}")
    if not isinstance(payload["generated_text"], str):
        raise ValueError("scalar lab generated_text must be text")
    prompt_tokens = _positive_int(payload["prompt_tokens"], "prompt_tokens")
    generated_tokens = _positive_int(payload["generated_tokens"], "generated_tokens")
    token_ids = payload["token_ids"]
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != generated_tokens
        or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in token_ids)
    ):
        raise ValueError("scalar lab token_ids must match generated_tokens")
    decode_ms = _positive_number(payload["decode_ms"], "decode_ms")
    ms_per_tok = _positive_number(payload["ms_per_tok"], "ms_per_tok")
    tokens_per_second = _positive_number(payload["tokens_per_second"], "tokens_per_second")
    if not math.isclose(decode_ms, generated_tokens * ms_per_tok, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("scalar lab timing is inconsistent")
    if not math.isclose(tokens_per_second, 1000.0 / ms_per_tok, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("scalar lab timing rate is inconsistent")
    _ = prompt_tokens
    return payload


def run_command(command: tuple[str, ...], *, timeout_seconds: float) -> tuple[str, str]:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise TimeoutError(f"scalar lab timed out after {timeout_seconds:g}s") from exc
    if process.returncode != 0:
        detail = stderr[-4096:].strip() or f"exit status {process.returncode}"
        raise ValueError(f"scalar lab process failed: {detail}")
    return stdout, stderr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--mode", choices=("ordinary", "mtp"), required=True)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--honor-eos", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        reject_route_environment(os.environ)
        if not args.binary.is_file():
            raise ValueError(f"scalar lab binary is unavailable: {args.binary}")
        if not args.model_dir.is_dir():
            raise ValueError(f"model directory is unavailable: {args.model_dir}")
        if not args.artifact.is_file() or args.artifact.stat().st_size <= 0:
            raise ValueError(f"artifact is unavailable: {args.artifact}")
        command = build_command(
            binary=args.binary,
            model_dir=args.model_dir,
            artifact=args.artifact,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            mode=args.mode,
            chat=args.chat,
            honor_eos=args.honor_eos,
        )
        stdout, stderr = run_command(command, timeout_seconds=args.timeout_seconds)
        payload = normalize_output(stdout, stderr)
    except (OSError, TimeoutError, ValueError) as exc:
        print(f"supersonic-scalar-lab: {exc}", file=sys.stderr)
        return 2
    print(OUTPUT_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"scalar lab {label} must be a positive integer")
    return value


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"scalar lab {label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"scalar lab {label} must be finite and positive")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
