#!/usr/bin/env python3
"""Run one deterministic llama.cpp server request and emit exact timing JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from typing import Mapping
from urllib import error, request


OUTPUT_PREFIX = "[llama_cpp_json] "
DEFAULT_CONTEXT_SIZE = 32768
DEFAULT_STARTUP_TIMEOUT_SECONDS = 120.0


def build_server_command(
    *,
    server_binary: str,
    model: Path,
    context_size: int,
    port: int,
) -> tuple[str, ...]:
    if not server_binary.strip():
        raise ValueError("server_binary must be non-empty")
    if context_size <= 0:
        raise ValueError("context_size must be positive")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    return (
        server_binary,
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(context_size),
        "--parallel",
        "1",
        "--gpu-layers",
        "99",
        "--no-cache-prompt",
        "--no-warmup",
        "--log-disable",
    )


def normalize_response(
    response: Mapping[str, object],
    *,
    expected_generated_tokens: int,
    fixed_token_count: bool = True,
) -> dict[str, object]:
    if expected_generated_tokens <= 0:
        raise ValueError("expected_generated_tokens must be positive")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise ValueError("llama-server response must contain exactly one choice")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    allowed_finish_reasons = {"length"} if fixed_token_count else {"length", "stop"}
    if finish_reason not in allowed_finish_reasons:
        raise ValueError("llama-server finish_reason is inconsistent with the EOS policy")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("llama-server response choice is missing its message")
    reasoning = _optional_text(message.get("reasoning_content"), "reasoning_content")
    content = _optional_text(message.get("content"), "content")
    if reasoning:
        raise ValueError("llama-server returned reasoning content with thinking disabled")
    generated_text = content
    if not generated_text:
        raise ValueError("llama-server response contains no generated text")

    usage = _mapping(response.get("usage"), "usage")
    timings = _mapping(response.get("timings"), "timings")
    prompt_tokens = _positive_int(usage.get("prompt_tokens"), "usage.prompt_tokens")
    generated_tokens = _positive_int(usage.get("completion_tokens"), "usage.completion_tokens")
    total_tokens = _positive_int(usage.get("total_tokens"), "usage.total_tokens")
    if total_tokens != prompt_tokens + generated_tokens:
        raise ValueError("llama-server total token count is inconsistent")
    if fixed_token_count and generated_tokens != expected_generated_tokens:
        raise ValueError(
            "llama-server generated token count mismatch: "
            f"observed {generated_tokens}, expected {expected_generated_tokens}"
        )
    if not fixed_token_count and generated_tokens > expected_generated_tokens:
        raise ValueError("llama-server generated token count exceeded the quality token cap")
    if _positive_int(timings.get("prompt_n"), "timings.prompt_n") != prompt_tokens:
        raise ValueError("llama-server prompt token counts are inconsistent")
    if _positive_int(timings.get("predicted_n"), "timings.predicted_n") != generated_tokens:
        raise ValueError("llama-server generated token counts are inconsistent")

    decode_ms = _positive_number(timings.get("predicted_ms"), "timings.predicted_ms")
    ms_per_tok = _positive_number(
        timings.get("predicted_per_token_ms"),
        "timings.predicted_per_token_ms",
    )
    tokens_per_second = _positive_number(
        timings.get("predicted_per_second"),
        "timings.predicted_per_second",
    )
    if not math.isclose(decode_ms, generated_tokens * ms_per_tok, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("llama-server decode timing is inconsistent with per-token timing")
    if not math.isclose(tokens_per_second, 1000.0 / ms_per_tok, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("llama-server throughput is inconsistent with per-token timing")
    return {
        "decode_ms": decode_ms,
        "generated_text": generated_text,
        "generated_tokens": generated_tokens,
        "ms_per_tok": ms_per_tok,
        "prompt_tokens": prompt_tokens,
        "tokens_per_second": tokens_per_second,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-binary", default="llama-server")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--honor-eos", action="store_true")
    parser.add_argument("--startup-timeout-seconds", type=float, default=DEFAULT_STARTUP_TIMEOUT_SECONDS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_one_shot(
            server_binary=args.server_binary,
            model=args.model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            context_size=args.context_size,
            seed=args.seed,
            chat=args.chat,
            fixed_token_count=not args.honor_eos,
            startup_timeout_seconds=args.startup_timeout_seconds,
        )
    except (OSError, ValueError, TimeoutError) as exc:
        print(f"llama-cpp-peer: {exc}", file=sys.stderr)
        return 2
    print(OUTPUT_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


def run_one_shot(
    *,
    server_binary: str,
    model: Path,
    prompt: str,
    max_new_tokens: int,
    context_size: int,
    seed: int,
    chat: bool,
    fixed_token_count: bool = True,
    startup_timeout_seconds: float,
) -> dict[str, object]:
    if not model.is_file() or model.stat().st_size <= 0:
        raise ValueError(f"peer model is unavailable: {model}")
    if not prompt:
        raise ValueError("prompt must be non-empty")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if startup_timeout_seconds <= 0 or not math.isfinite(startup_timeout_seconds):
        raise ValueError("startup_timeout_seconds must be finite and positive")

    port = _available_port()
    command = build_server_command(
        server_binary=server_binary,
        model=model,
        context_size=context_size,
        port=port,
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    previous_handlers = _install_signal_handlers(process)
    try:
        base_url = f"http://127.0.0.1:{port}"
        _wait_until_healthy(process, base_url, startup_timeout_seconds)
        response = _post_chat(
            base_url,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            seed=seed,
            chat=chat,
            fixed_token_count=fixed_token_count,
        )
        return normalize_response(
            response,
            expected_generated_tokens=max_new_tokens,
            fixed_token_count=fixed_token_count,
        )
    finally:
        _restore_signal_handlers(previous_handlers)
        _stop_process(process)
        if process.stderr is not None:
            process.stderr.close()


def _post_chat(
    base_url: str,
    *,
    prompt: str,
    max_new_tokens: int,
    seed: int,
    chat: bool,
    fixed_token_count: bool,
) -> Mapping[str, object]:
    if not chat:
        raise ValueError("llama.cpp peer currently requires the shared chat-template path")
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "top_k": 0,
        "top_p": 1,
        "seed": seed,
        "cache_prompt": False,
        "ignore_eos": fixed_token_count,
        "timings_per_token": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    raw = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    req = request.Request(
        f"{base_url}/v1/chat/completions",
        data=raw,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=3600) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (error.URLError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid llama-server completion response: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("llama-server completion response must be an object")
    return payload


def _wait_until_healthy(process: subprocess.Popen[str], base_url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            detail = _stderr_tail(process)
            raise ValueError(f"llama-server exited during startup: {detail}")
        try:
            with request.urlopen(f"{base_url}/health", timeout=1) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, Mapping) and payload.get("status") == "ok":
                return
        except (error.URLError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(0.1)
    raise TimeoutError(f"llama-server startup timed out: {last_error}")


def _available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _install_signal_handlers(process: subprocess.Popen[str]) -> dict[int, object]:
    previous: dict[int, object] = {}

    def interrupted(signum, _frame):
        _stop_process(process)
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupted)
    return previous


def _restore_signal_handlers(previous: Mapping[int, object]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _stderr_tail(process: subprocess.Popen[str]) -> str:
    if process.stderr is None:
        return "no stderr captured"
    text = process.stderr.read(65536)
    return text[-4096:].strip() or f"exit status {process.returncode}"


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"llama-server response is missing {label}")
    return value


def _optional_text(value: object, label: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"llama-server {label} must be text")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"llama-server {label} must be a positive integer")
    return value


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"llama-server {label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"llama-server {label} must be finite and positive")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
