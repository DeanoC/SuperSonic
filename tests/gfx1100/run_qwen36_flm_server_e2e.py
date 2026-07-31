#!/usr/bin/env python3
"""Run the real Qwen3.6 FLM server through its OpenAI protocol surface."""

import argparse
import contextlib
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterator


ROOT = Path(__file__).resolve().parents[2]
COMPAT_SCRIPT = ROOT / "scripts" / "openai_compat_smoke.mjs"
AGENT_SCRIPT = ROOT / "scripts" / "openai_agent_tool_smoke.mjs"
DEFAULT_FLM = Path(
    "/mnt/data/runs/geo-quant/"
    "qwen36-35b-a3b-supersonic-native-int4-current.flm"
)
DEFAULT_BINARY = ROOT / "target" / "release" / "supersonic-serve"
DEFAULT_OUT_JSON = ROOT / "target" / "qwen36_35b_a3b_flm_server_e2e.json"
DEFAULT_OPENAI_SDK_DIR = ROOT / "target" / "openai-sdk-smoke"
EXPECTED_MODEL = "qwen3.6-35b-a3b"
EXPECTED_SOURCE = "flm"
EXPECTED_NATIVE_INT4 = 330
EXPECTED_BF16_FALLBACK = 0
EXPECTED_REQUIRED_WEIGHTS = 693
EXPECTED_RAW_DENSE_WEIGHTS = 363
OPENAI_SDK_VERSION = "6.49.0"
PROCESS_GRACE_SECONDS = 5
READY_POLL_SECONDS = 0.25
HTTP_TIMEOUT_SECONDS = 5.0
SMOKE_JSON_PREFIX = "SUPERSONIC_SMOKE_JSON="


class PhaseError(RuntimeError):
    pass


class SdkSmokeFailure(PhaseError):
    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


def server_command(args: argparse.Namespace, port: int) -> list[str]:
    command = [
        str(args.binary),
        "--flm-file",
        str(args.flm),
        "--backend",
        args.backend,
        "--device",
        str(args.device),
        "--max-context",
        str(args.max_context),
        "--host",
        args.host,
        "--port",
        str(port),
        "--api-key",
        args.api_key,
    ]
    if args.no_download:
        command.append("--no-download")
    return command


def allocate_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _log_tail(path: Path, limit: int = 16_384) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="replace").strip()
    except OSError as exc:
        return f"<unable to read server log: {exc}>"


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _process_group_exists(pgid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _reap_leader(
    process: subprocess.Popen[str],
    timeout: float,
) -> bool:
    try:
        process.communicate(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


def _terminate_and_reap_process_group(process: subprocess.Popen[str]) -> None:
    pgid = process.pid
    if _process_group_exists(pgid):
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    leader_reaped = _reap_leader(process, PROCESS_GRACE_SECONDS)
    group_gone = _wait_for_process_group_exit(pgid, PROCESS_GRACE_SECONDS)
    if group_gone:
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    if not leader_reaped:
        _reap_leader(process, PROCESS_GRACE_SECONDS)
    if not _wait_for_process_group_exit(pgid, PROCESS_GRACE_SECONDS):
        raise PhaseError(f"process group {pgid} survived SIGKILL")


def wait_for_ready(
    process: subprocess.Popen[str],
    base_url: str,
    api_key: str,
    *,
    timeout: float,
    log_tail: Callable[[], str] = lambda: "",
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "server did not answer"
    while True:
        returncode = process.poll()
        if returncode is not None:
            tail = log_tail()
            raise PhaseError(
                f"readiness failed: server exited with exit {returncode}; "
                f"log tail: {tail}"
            )
        remaining = deadline - time.monotonic()
        if remaining < 0:
            break
        request = urllib.request.Request(
            f"{base_url}/ready",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        try:
            with opener(
                request,
                timeout=max(0.1, min(HTTP_TIMEOUT_SECONDS, remaining or 0.1)),
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if response.status == 200 and payload.get("ready") is True:
                    return payload
                last_error = f"HTTP {response.status}: {payload}"
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        if timeout == 0:
            break
        time.sleep(min(READY_POLL_SECONDS, max(0.0, remaining)))
    raise PhaseError(
        f"server startup timed out after {timeout:g}s; "
        f"last readiness result: {last_error}; log tail: {log_tail()}"
    )


@contextlib.contextmanager
def running_server(
    args: argparse.Namespace,
    port: int,
    log_path: Path,
    *,
    popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    readiness: Callable[..., dict[str, Any]] = wait_for_ready,
) -> Iterator[dict[str, Any]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        command = server_command(args, port)
        try:
            process = popen_factory(
                command,
                cwd=ROOT,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            raise PhaseError(
                f"server failed to start: {exc}: {' '.join(command)}"
            ) from exc
        try:
            ready = readiness(
                process,
                f"http://{args.host}:{port}",
                args.api_key,
                timeout=args.startup_timeout,
                log_tail=lambda: _log_tail(log_path),
            )
            yield ready
        finally:
            _terminate_and_reap_process_group(process)


def _parse_scalar(value: str, label: str) -> int | float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise PhaseError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise PhaseError(f"{label} must be finite, got {value!r}")
    return int(parsed) if parsed.is_integer() else parsed


def parse_prometheus_metrics(text: str) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2 or "{" in fields[0]:
            raise PhaseError(
                f"unsupported Prometheus sample at line {line_number}: {raw!r}"
            )
        name, value = fields
        if name in metrics:
            raise PhaseError(f"duplicate metric {name}")
        metrics[name] = _parse_scalar(value, f"metric {name}")
    return metrics


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PhaseError(f"{label} must be an object")
    return value


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PhaseError(f"{label} must be an integer")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhaseError(f"{label} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise PhaseError(f"{label} must be finite")
    return parsed


def _expect(value: object, expected: object, label: str) -> None:
    if value != expected or (
        isinstance(expected, int)
        and not isinstance(expected, bool)
        and isinstance(value, bool)
    ):
        raise PhaseError(f"{label} must be {expected!r}, got {value!r}")


def _positive_int(value: object, label: str) -> int:
    parsed = _strict_int(value, label)
    if parsed <= 0:
        raise PhaseError(f"{label} must be positive")
    return parsed


def _validate_finite_tree(value: object, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_finite_tree(child, f"{label}.{key}")
        return
    _finite_number(value, label)


def _metric_int(
    metrics: dict[str, int | float],
    name: str,
    *,
    expected: int | None = None,
) -> int:
    if name not in metrics:
        raise PhaseError(f"required metric {name} is missing")
    value = _strict_int(metrics[name], f"metric {name}")
    if expected is not None:
        _expect(value, expected, f"metric {name}")
    return value


def validate_flm_evidence(
    capabilities: dict[str, Any],
    metrics: dict[str, int | float],
) -> dict[str, Any]:
    _expect(capabilities.get("model"), EXPECTED_MODEL, "capabilities model")
    _expect(capabilities.get("backend"), "HIP", "capabilities backend")
    _expect(capabilities.get("ready"), True, "capabilities ready")
    for field in ("chat", "responses", "streaming", "stream_usage", "tools", "reasoning"):
        _expect(capabilities.get(field), True, f"capabilities {field}")

    scheduler = _mapping(capabilities.get("scheduler"), "capabilities scheduler")
    for field in (
        "active_requests",
        "queued_requests",
        "max_queued_requests",
        "queue_timeout_ms",
    ):
        _strict_int(scheduler.get(field), f"scheduler {field}")
    _expect(scheduler.get("active_requests"), 0, "scheduler active_requests")
    _expect(scheduler.get("queued_requests"), 0, "scheduler queued_requests")

    flm = _mapping(capabilities.get("flm"), "capabilities flm")
    exact_fields = {
        "source": EXPECTED_SOURCE,
        "required_weights": EXPECTED_REQUIRED_WEIGHTS,
        "raw_dense_weights": EXPECTED_RAW_DENSE_WEIGHTS,
        "native_int4_direct_weights": EXPECTED_NATIVE_INT4,
        "bf16_fallback_weights": EXPECTED_BF16_FALLBACK,
        "load_sequence": 1,
        "source_open_count": 1,
    }
    for field, expected in exact_fields.items():
        if isinstance(expected, int):
            _strict_int(flm.get(field), f"FLM {field}")
        _expect(flm.get(field), expected, f"FLM {field}")
    _positive_int(
        flm.get("resident_allocation_count"),
        "FLM resident_allocation_count",
    )
    source_bytes = _positive_int(flm.get("source_bytes"), "FLM source_bytes")
    device_upload_bytes = _positive_int(
        flm.get("device_upload_bytes"),
        "FLM device_upload_bytes",
    )
    transfer_backend = flm.get("transfer_backend")
    if not isinstance(transfer_backend, str) or not transfer_backend:
        raise PhaseError("FLM transfer_backend must be a non-empty string")
    startup = _mapping(flm.get("startup"), "FLM startup")
    _validate_finite_tree(startup, "FLM startup")
    if _finite_number(startup.get("total_seconds"), "FLM startup.total_seconds") <= 0:
        raise PhaseError("FLM startup.total_seconds must be positive")

    metric_expectations = {
        "supersonic_ready": 1,
        "supersonic_active_requests": 0,
        "supersonic_queued_requests": 0,
        "supersonic_model_loads_total": 1,
        "supersonic_flm_native_int4_direct_weights": EXPECTED_NATIVE_INT4,
        "supersonic_flm_bf16_fallback_weights": EXPECTED_BF16_FALLBACK,
        "supersonic_flm_source_bytes": source_bytes,
        "supersonic_flm_device_upload_bytes": device_upload_bytes,
    }
    for name, expected in metric_expectations.items():
        _metric_int(metrics, name, expected=expected)
    startup_metric = metrics.get("supersonic_flm_startup_seconds")
    if _finite_number(startup_metric, "metric supersonic_flm_startup_seconds") <= 0:
        raise PhaseError("metric supersonic_flm_startup_seconds must be positive")

    return {
        "model": EXPECTED_MODEL,
        "source": EXPECTED_SOURCE,
        "load_sequence": 1,
        "native_int4": EXPECTED_NATIVE_INT4,
        "bf16_fallback": EXPECTED_BF16_FALLBACK,
        "model_loads_total": 1,
        "startup": startup,
        "transfer_backend": transfer_backend,
        "source_bytes": source_bytes,
        "device_upload_bytes": device_upload_bytes,
        "source_open_count": 1,
        "resident_allocation_count": flm["resident_allocation_count"],
        "scheduler": scheduler,
    }


def validate_load_invariance(
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    try:
        values = [
            _strict_int(before.get("load_sequence"), "before load_sequence"),
            _strict_int(before.get("model_loads_total"), "before model_loads_total"),
            _strict_int(after.get("load_sequence"), "after load_sequence"),
            _strict_int(after.get("model_loads_total"), "after model_loads_total"),
        ]
    except PhaseError as exc:
        raise PhaseError(f"single-load invariant failed: {exc}") from exc
    if values != [1, 1, 1, 1]:
        raise PhaseError(
            "single-load invariant failed: expected load sequence and count "
            f"[1, 1, 1, 1], got {values}"
        )


def parse_smoke_output(stdout: str, phase: str) -> dict[str, Any]:
    markers = [
        line[len(SMOKE_JSON_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(SMOKE_JSON_PREFIX)
    ]
    if len(markers) != 1:
        raise PhaseError(
            f"{phase} smoke emitted {len(markers)} structured reports; expected one"
        )
    try:
        payload = json.loads(markers[0])
    except json.JSONDecodeError as exc:
        raise PhaseError(f"{phase} smoke emitted invalid JSON: {exc}") from exc
    return _mapping(payload, f"{phase} smoke report")


def _path(mapping: dict[str, Any], *keys: str) -> object:
    current: object = mapping
    for index, key in enumerate(keys):
        label = ".".join(keys[:index]) or "report"
        current = _mapping(current, label).get(key)
        if current is None:
            raise PhaseError(f"report is missing {'.'.join(keys)}")
    return current


def _require_true(report: dict[str, Any], *path: str) -> None:
    _expect(_path(report, *path), True, ".".join(path))


def _exact_mapping(
    value: object,
    keys: set[str],
    label: str,
) -> dict[str, Any]:
    mapping = _mapping(value, label)
    if set(mapping) != keys:
        raise PhaseError(
            f"{label} keys must be {sorted(keys)}, got {sorted(mapping)}"
        )
    return mapping


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PhaseError(f"{label} must be a non-empty string")
    return value


def _validate_usage(value: object, label: str) -> None:
    usage = _exact_mapping(
        value,
        {"prompt_tokens", "completion_tokens", "total_tokens"},
        label,
    )
    prompt = _positive_int(usage["prompt_tokens"], f"{label}.prompt_tokens")
    completion = _positive_int(
        usage["completion_tokens"],
        f"{label}.completion_tokens",
    )
    total = _positive_int(usage["total_tokens"], f"{label}.total_tokens")
    if total != prompt + completion:
        raise PhaseError(f"{label}.total_tokens does not add up")


def _validate_auth_result(value: object, label: str) -> None:
    result = _exact_mapping(value, {"status", "error_type"}, label)
    _strict_int(result["status"], f"{label}.status")
    _expect(result["status"], 401, f"{label}.status")
    _expect(
        result["error_type"],
        "authentication_error",
        f"{label}.error_type",
    )


def _validate_canary(
    value: object,
    label: str,
    *,
    expected: str,
) -> None:
    canary = _exact_mapping(
        value,
        {"expected", "actual", "finish_reason", "passed"},
        label,
    )
    _expect(canary["expected"], expected, f"{label}.expected")
    _expect(canary["actual"], expected, f"{label}.actual")
    _expect(canary["finish_reason"], "stop", f"{label}.finish_reason")
    _expect(canary["passed"], True, f"{label}.passed")


def validate_compat_report(report: dict[str, Any]) -> dict[str, Any]:
    _exact_mapping(
        report,
        {"transport", "semantic_quality", "usage", "throughput"},
        "compat report",
    )
    transport = _exact_mapping(
        report["transport"],
        {
            "auth",
            "models",
            "tokenizer",
            "chat",
            "chat_stream",
            "completions",
            "responses",
            "responses_stream",
            "reasoning",
            "repeated_request",
        },
        "compat transport",
    )
    auth = _exact_mapping(
        transport["auth"],
        {"missing_key", "wrong_key", "protected_routes"},
        "compat transport.auth",
    )
    _validate_auth_result(auth["missing_key"], "compat auth.missing_key")
    _validate_auth_result(auth["wrong_key"], "compat auth.wrong_key")
    protected = _exact_mapping(
        auth["protected_routes"],
        {"/health", "/ready", "/metrics", "/v1/capabilities"},
        "compat auth.protected_routes",
    )
    for path, evidence in protected.items():
        _validate_auth_result(evidence, f"compat auth.protected_routes[{path}]")

    transport_shapes = {
        "models": {"listed", "retrieved"},
        "tokenizer": {"roundtrip", "token_count"},
        "chat": {"received"},
        "chat_stream": {
            "received_delta",
            "received_terminal",
            "received_usage",
        },
        "completions": {"received"},
        "responses": {"received", "stored_roundtrip"},
        "responses_stream": {
            "received_delta",
            "received_terminal",
            "received_usage",
        },
        "reasoning": {"request_accepted"},
        "repeated_request": {"received"},
    }
    for section, keys in transport_shapes.items():
        evidence = _exact_mapping(
            transport[section],
            keys,
            f"compat transport.{section}",
        )
        for key, value in evidence.items():
            if key == "token_count":
                _positive_int(value, "compat transport.tokenizer.token_count")
            else:
                _expect(value, True, f"compat transport.{section}.{key}")

    semantics = _exact_mapping(
        report["semantic_quality"],
        {
            "chat",
            "chat_stream",
            "completions",
            "responses",
            "responses_stream",
            "reasoning",
            "repeated_request",
            "passed",
        },
        "compat semantic_quality",
    )
    _validate_canary(semantics["chat"], "semantic chat", expected="hello")
    _validate_canary(
        semantics["completions"],
        "semantic completions",
        expected="hello",
    )
    _validate_canary(
        semantics["repeated_request"],
        "semantic repeated_request",
        expected="ready",
    )

    chat_stream = _exact_mapping(
        semantics["chat_stream"],
        {
            "expected",
            "actual",
            "finish_reason",
            "passed",
            "terminal_count",
            "terminal_last_before_usage",
            "usage_last",
        },
        "semantic chat_stream",
    )
    for key, expected in (
        ("expected", "hello"),
        ("actual", "hello"),
        ("finish_reason", "stop"),
        ("passed", True),
        ("terminal_count", 1),
        ("terminal_last_before_usage", True),
        ("usage_last", True),
    ):
        if key == "terminal_count":
            _strict_int(chat_stream[key], f"semantic chat_stream.{key}")
        _expect(chat_stream[key], expected, f"semantic chat_stream.{key}")

    responses = _exact_mapping(
        semantics["responses"],
        {
            "expected",
            "actual",
            "status",
            "stored_roundtrip",
            "passed",
        },
        "semantic responses",
    )
    for key, expected in (
        ("expected", "hello"),
        ("actual", "hello"),
        ("status", "completed"),
        ("stored_roundtrip", True),
        ("passed", True),
    ):
        _expect(responses[key], expected, f"semantic responses.{key}")

    responses_stream = _exact_mapping(
        semantics["responses_stream"],
        {
            "expected",
            "actual",
            "status",
            "terminal_count",
            "terminal_last",
            "passed",
        },
        "semantic responses_stream",
    )
    for key, expected in (
        ("expected", "hello"),
        ("actual", "hello"),
        ("status", "completed"),
        ("terminal_count", 1),
        ("terminal_last", True),
        ("passed", True),
    ):
        if key == "terminal_count":
            _strict_int(
                responses_stream[key],
                "semantic responses_stream.terminal_count",
            )
        _expect(
            responses_stream[key],
            expected,
            f"semantic responses_stream.{key}",
        )

    reasoning = _exact_mapping(
        semantics["reasoning"],
        {"accepted", "observed", "visible_think_tags", "passed"},
        "semantic reasoning",
    )
    _expect(reasoning["accepted"], True, "semantic reasoning.accepted")
    _expect(reasoning["observed"], True, "semantic reasoning.observed")
    _expect(
        reasoning["visible_think_tags"],
        False,
        "semantic reasoning.visible_think_tags",
    )
    _expect(reasoning["passed"], True, "semantic reasoning.passed")
    _expect(semantics["passed"], True, "semantic_quality.passed")

    usage = _exact_mapping(
        report["usage"],
        {
            "chat",
            "chat_stream",
            "completions",
            "responses",
            "responses_stream",
            "repeated_request",
        },
        "compat usage",
    )
    for section, counts in usage.items():
        _validate_usage(counts, f"compat usage.{section}")

    throughput = _exact_mapping(
        report["throughput"],
        {
            "first_token_seconds",
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
        },
        "compat throughput",
    )
    for key, value in throughput.items():
        if _finite_number(value, f"compat throughput.{key}") <= 0:
            raise PhaseError(f"compat throughput.{key} must be positive")
    return report


def _validate_scheduler_evidence(
    value: object,
    label: str,
    *,
    active: int,
    queued: int,
) -> None:
    snapshot = _exact_mapping(
        value,
        {
            "active_requests",
            "queued_requests",
            "model_loads_total",
            "metric_active_requests",
            "metric_queued_requests",
        },
        label,
    )
    expected = {
        "active_requests": active,
        "queued_requests": queued,
        "model_loads_total": 1,
        "metric_active_requests": active,
        "metric_queued_requests": queued,
    }
    for key, expected_value in expected.items():
        _strict_int(snapshot[key], f"{label}.{key}")
        _expect(snapshot[key], expected_value, f"{label}.{key}")


def validate_cancellation(value: object, label: str = "cancellation") -> dict[str, Any]:
    cancellation = _exact_mapping(
        value,
        {
            "nonterminal_delta",
            "abort_closed",
            "before",
            "after",
            "queued_request_completed",
            "release_seconds",
        },
        label,
    )
    for key in ("nonterminal_delta", "abort_closed", "queued_request_completed"):
        _expect(cancellation[key], True, f"{label}.{key}")
    _validate_scheduler_evidence(
        cancellation["before"],
        f"{label}.before",
        active=1,
        queued=1,
    )
    _validate_scheduler_evidence(
        cancellation["after"],
        f"{label}.after",
        active=0,
        queued=0,
    )
    if _finite_number(cancellation["release_seconds"], f"{label}.release_seconds") < 0:
        raise PhaseError(f"{label}.release_seconds must be non-negative")
    return cancellation


def _validate_agent_loop(name: str, value: object) -> dict[str, Any]:
    loop_specs = {
        "chat_tool_loop": {
            "terminal_key": "finish_reason",
            "terminal_value": "tool_calls",
            "continuation_terminal_key": "finish_reason",
            "continuation_terminal_value": "stop",
        },
        "responses_tool_loop": {
            "terminal_key": "status",
            "terminal_value": "completed",
            "continuation_terminal_key": "status",
            "continuation_terminal_value": "completed",
        },
    }
    spec = loop_specs[name]
    loop = _exact_mapping(
        value,
        {
            "call_count",
            "valid_tool_call",
            "call_id",
            "tool_name",
            "arguments",
            spec["terminal_key"],
            "suffix_content",
            "continuation",
            "elapsed_seconds",
        },
        f"agent requests.{name}",
    )
    _strict_int(loop["call_count"], f"agent {name}.call_count")
    _expect(loop["call_count"], 1, f"agent {name}.call_count")
    _expect(loop["valid_tool_call"], True, f"agent {name}.valid_tool_call")
    _nonempty_string(loop["call_id"], f"agent {name}.call_id")
    _expect(
        loop["tool_name"],
        "read_source_file",
        f"agent {name}.tool_name",
    )
    arguments = _exact_mapping(
        loop["arguments"],
        {"path"},
        f"agent {name}.arguments",
    )
    _expect(
        arguments["path"],
        "src/lib.rs",
        f"agent {name}.arguments.path",
    )
    _expect(
        loop[spec["terminal_key"]],
        spec["terminal_value"],
        f"agent {name}.{spec['terminal_key']}",
    )
    _expect(loop["suffix_content"], "", f"agent {name}.suffix_content")
    continuation = _exact_mapping(
        loop["continuation"],
        {
            "text",
            spec["continuation_terminal_key"],
            "tool_call_count",
        },
        f"agent {name}.continuation",
    )
    _nonempty_string(
        continuation["text"],
        f"agent {name}.continuation.text",
    )
    _expect(
        continuation[spec["continuation_terminal_key"]],
        spec["continuation_terminal_value"],
        f"agent {name}.continuation.{spec['continuation_terminal_key']}",
    )
    _strict_int(
        continuation["tool_call_count"],
        f"agent {name}.continuation.tool_call_count",
    )
    _expect(
        continuation["tool_call_count"],
        0,
        f"agent {name}.continuation.tool_call_count",
    )
    if _finite_number(loop["elapsed_seconds"], f"agent {name}.elapsed_seconds") <= 0:
        raise PhaseError(f"agent {name}.elapsed_seconds must be positive")
    return loop


def validate_agent_report(report: dict[str, Any]) -> dict[str, Any]:
    _exact_mapping(report, {"requests", "cancellation"}, "agent report")
    requests = _exact_mapping(
        report["requests"],
        {"chat_tool_loop", "responses_tool_loop"},
        "agent requests",
    )
    for name, value in requests.items():
        _validate_agent_loop(name, value)
    validate_cancellation(report["cancellation"], "agent cancellation")
    return report


def validate_agent_failure_report(report: dict[str, Any]) -> dict[str, Any]:
    partial = _exact_mapping(
        report,
        {"requests", "cancellation", "failure"},
        "agent failure report",
    )
    failure = _exact_mapping(
        partial["failure"],
        {"phase", "message", "raw"},
        "agent failure",
    )
    phase = failure["phase"]
    if phase not in {"cancellation", "chat_tool_loop", "responses_tool_loop"}:
        raise PhaseError(f"agent failure.phase is invalid: {phase!r}")
    _nonempty_string(failure["message"], "agent failure.message")

    requests = _mapping(partial["requests"], "agent failure.requests")
    expected_request_keys = {
        "cancellation": set(),
        "chat_tool_loop": set(),
        "responses_tool_loop": {"chat_tool_loop"},
    }[phase]
    if set(requests) != expected_request_keys:
        raise PhaseError(
            "agent failure.requests keys must be "
            f"{sorted(expected_request_keys)}, got {sorted(requests)}"
        )

    if phase == "cancellation":
        _expect(
            partial["cancellation"],
            None,
            "agent failure.cancellation",
        )
    else:
        validate_cancellation(
            partial["cancellation"],
            "agent failure.cancellation",
        )
        _mapping(failure["raw"], "agent failure.raw")
        for name, value in requests.items():
            _validate_agent_loop(name, value)
    return partial


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "model",
        "source",
        "load_sequence",
        "native_int4",
        "bf16_fallback",
        "requests",
        "startup",
        "throughput",
        "cancellation",
    }
    if set(report) != expected_keys:
        raise PhaseError(
            f"report keys must be {sorted(expected_keys)}, got {sorted(report)}"
        )
    _expect(report.get("model"), EXPECTED_MODEL, "report model")
    _expect(report.get("source"), EXPECTED_SOURCE, "report source")
    for field, expected in (
        ("load_sequence", 1),
        ("native_int4", EXPECTED_NATIVE_INT4),
        ("bf16_fallback", EXPECTED_BF16_FALLBACK),
    ):
        _strict_int(report.get(field), f"report {field}")
        _expect(report.get(field), expected, f"report {field}")

    requests = _exact_mapping(
        report.get("requests"),
        {"compat", "agent"},
        "report requests",
    )
    compat = validate_compat_report(
        _mapping(requests["compat"], "report requests.compat")
    )
    agent = validate_agent_report(
        _mapping(requests["agent"], "report requests.agent")
    )

    startup = _exact_mapping(
        report.get("startup"),
        {
            "ready_seconds",
            "total_seconds",
            "transfer_backend",
            "source_bytes",
            "device_upload_bytes",
            "source_open_count",
            "resident_allocation_count",
            "exclusive_components",
            "provenance",
        },
        "report startup",
    )
    for field in (
        "source_bytes",
        "device_upload_bytes",
        "source_open_count",
        "resident_allocation_count",
    ):
        _positive_int(startup.get(field), f"startup {field}")
    _expect(startup.get("source_open_count"), 1, "startup source_open_count")
    for field in ("ready_seconds", "total_seconds"):
        if _finite_number(startup.get(field), f"startup {field}") < 0:
            raise PhaseError(f"startup {field} must be non-negative")
    if not isinstance(startup.get("transfer_backend"), str):
        raise PhaseError("startup transfer_backend must be a string")
    _validate_finite_tree(
        startup["exclusive_components"],
        "startup exclusive_components",
    )
    provenance = _exact_mapping(
        startup["provenance"],
        {"artifact", "sdk"},
        "startup provenance",
    )
    artifact = _exact_mapping(
        provenance["artifact"],
        {"path", "sha256", "size_bytes"},
        "startup provenance.artifact",
    )
    _nonempty_string(artifact["path"], "startup provenance.artifact.path")
    digest = _nonempty_string(
        artifact["sha256"],
        "startup provenance.artifact.sha256",
    )
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise PhaseError("startup provenance.artifact.sha256 must be lowercase SHA-256")
    _positive_int(
        artifact["size_bytes"],
        "startup provenance.artifact.size_bytes",
    )
    sdk = _exact_mapping(
        provenance["sdk"],
        {"package", "version"},
        "startup provenance.sdk",
    )
    _expect(sdk["package"], "openai", "startup provenance.sdk.package")
    _expect(
        sdk["version"],
        OPENAI_SDK_VERSION,
        "startup provenance.sdk.version",
    )

    throughput = _mapping(report.get("throughput"), "report throughput")
    if throughput != compat["throughput"]:
        raise PhaseError("report throughput must equal compat throughput evidence")
    cancellation = validate_cancellation(report.get("cancellation"))
    if cancellation != agent["cancellation"]:
        raise PhaseError("report cancellation must equal agent cancellation evidence")

    def reject_nonfinite(value: object, label: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                reject_nonfinite(child, f"{label}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                reject_nonfinite(child, f"{label}[{index}]")
        elif isinstance(value, float) and not math.isfinite(value):
            raise PhaseError(f"{label} must be finite")

    reject_nonfinite(report, "report")
    return report


def _request(
    base_url: str,
    path: str,
    api_key: str,
    *,
    accept: str,
) -> bytes:
    request = urllib.request.Request(
        f"{base_url}{path}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": accept,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                raise PhaseError(f"GET {path} returned HTTP {response.status}")
            return response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise PhaseError(f"GET {path} failed: {exc}") from exc


def fetch_json(base_url: str, path: str, api_key: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            _request(
                base_url,
                path,
                api_key,
                accept="application/json",
            ).decode("utf-8")
        )
    except json.JSONDecodeError as exc:
        raise PhaseError(f"GET {path} returned invalid JSON: {exc}") from exc
    return _mapping(payload, f"GET {path}")


def fetch_metrics(base_url: str, api_key: str) -> dict[str, int | float]:
    text = _request(
        base_url,
        "/metrics",
        api_key,
        accept="text/plain",
    ).decode("utf-8")
    return parse_prometheus_metrics(text)


def run_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None,
    timeout: float,
    phase: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise PhaseError(f"{phase} failed to start: {exc}") from exc
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_and_reap_process_group(process)
        raise PhaseError(f"{phase} timed out after {timeout:g}s") from exc
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if check and result.returncode != 0:
        raise PhaseError(
            f"{phase} failed with exit {result.returncode}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return result


def _installed_openai_version(result: subprocess.CompletedProcess[str]) -> str | None:
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
        version = payload["dependencies"]["openai"]["version"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return version if isinstance(version, str) else None


def _sdk_probe(
    args: argparse.Namespace,
    sdk_dir: Path,
    *,
    phase: str,
) -> subprocess.CompletedProcess[str]:
    return run_process(
        [
            args.npm,
            "list",
            f"openai@{OPENAI_SDK_VERSION}",
            "--depth=0",
            "--json",
        ],
        cwd=sdk_dir,
        env=None,
        timeout=args.sdk_probe_timeout,
        phase=phase,
        check=False,
    )


def ensure_openai_sdk(args: argparse.Namespace) -> dict[str, Any]:
    sdk_dir = args.openai_sdk_dir.resolve()
    sdk_dir.mkdir(parents=True, exist_ok=True)
    probe = _sdk_probe(args, sdk_dir, phase="OpenAI SDK probe")
    if _installed_openai_version(probe) == OPENAI_SDK_VERSION:
        return {
            "directory": sdk_dir,
            "package": "openai",
            "version": OPENAI_SDK_VERSION,
        }
    run_process(
        [
            args.npm,
            "install",
            "--no-audit",
            "--no-fund",
            "--prefix",
            str(sdk_dir),
            f"openai@{OPENAI_SDK_VERSION}",
        ],
        cwd=ROOT,
        env=None,
        timeout=args.sdk_install_timeout,
        phase="OpenAI SDK install",
    )
    verify = _sdk_probe(args, sdk_dir, phase="OpenAI SDK verification")
    version = _installed_openai_version(verify)
    if version != OPENAI_SDK_VERSION:
        raise PhaseError(
            "OpenAI SDK verification expected "
            f"{OPENAI_SDK_VERSION}, got {version!r}: {verify.stderr.strip()}"
        )
    return {
        "directory": sdk_dir,
        "package": "openai",
        "version": version,
    }


def run_sdk_smoke(
    args: argparse.Namespace,
    script: Path,
    sdk_dir: Path,
    base_url: str,
    phase: str,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "SUPERSONIC_BASE_URL": base_url,
            "SUPERSONIC_API_KEY": args.api_key,
            "SUPERSONIC_SMOKE_MODEL": EXPECTED_MODEL,
            "SUPERSONIC_REQUEST_TIMEOUT_MS": str(
                max(1, int(args.request_timeout * 1000))
            ),
        }
    )
    result = run_process(
        [args.node, str(script)],
        cwd=sdk_dir,
        env=env,
        timeout=args.request_timeout,
        phase=phase,
        check=False,
    )
    if result.stderr.strip():
        print(result.stderr, file=sys.stderr, end="")
    print(result.stdout, end="")
    try:
        report = parse_smoke_output(result.stdout, phase)
    except PhaseError as exc:
        if result.returncode == 0:
            raise
        raise PhaseError(
            f"{phase} failed with exit {result.returncode} and no valid "
            f"structured report: {exc}\nstdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        ) from exc
    if result.returncode != 0:
        raise SdkSmokeFailure(
            f"{phase} failed with exit {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            report,
        )
    return report


def run_protocol_phases(
    args: argparse.Namespace,
    sdk_dir: Path,
    base_url: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "compat": None,
        "agent": None,
        "failures": [],
    }
    try:
        compat = run_sdk_smoke(
            args,
            COMPAT_SCRIPT,
            sdk_dir,
            base_url,
            "OpenAI compatibility",
        )
        result["compat"] = compat
        try:
            validate_compat_report(compat)
        except PhaseError as exc:
            result["failures"].append(
                {"phase": "compat_semantic", "message": str(exc)}
            )
    except PhaseError as exc:
        result["failures"].append(
            {"phase": "compat_transport", "message": str(exc)}
        )

    try:
        agent = run_sdk_smoke(
            args,
            AGENT_SCRIPT,
            sdk_dir,
            base_url,
            "OpenAI agent tool",
        )
        result["agent"] = agent
        try:
            validate_agent_report(agent)
        except PhaseError as exc:
            result["failures"].append(
                {"phase": "agent_protocol", "message": str(exc)}
            )
    except SdkSmokeFailure as exc:
        result["agent"] = validate_agent_failure_report(exc.report)
        result["failures"].append({"phase": "agent", "message": str(exc)})
    except PhaseError as exc:
        result["failures"].append({"phase": "agent", "message": str(exc)})
    return result


def build_report(
    before: dict[str, Any],
    compat: dict[str, Any],
    agent: dict[str, Any],
    *,
    ready_seconds: float,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    startup = {
        "ready_seconds": ready_seconds,
        "total_seconds": before["startup"]["total_seconds"],
        "transfer_backend": before["transfer_backend"],
        "source_bytes": before["source_bytes"],
        "device_upload_bytes": before["device_upload_bytes"],
        "source_open_count": before["source_open_count"],
        "resident_allocation_count": before["resident_allocation_count"],
        "exclusive_components": before["startup"].get("exclusive_components", {}),
        "provenance": provenance,
    }
    report = {
        "model": EXPECTED_MODEL,
        "source": EXPECTED_SOURCE,
        "load_sequence": before["load_sequence"],
        "native_int4": before["native_int4"],
        "bf16_fallback": before["bf16_fallback"],
        "requests": {
            "compat": validate_compat_report(compat),
            "agent": validate_agent_report(agent),
        },
        "startup": startup,
        "throughput": _mapping(compat.get("throughput"), "compat throughput"),
        "cancellation": _mapping(
            agent.get("cancellation"),
            "agent cancellation",
        ),
    }
    return validate_report(report)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        partial.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, path)
    except (OSError, ValueError) as exc:
        raise PhaseError(f"report write failed for {path}: {exc}") from exc


def clear_report_output(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
        for partial in path.parent.glob(f".{path.name}.partial-*"):
            partial.unlink(missing_ok=True)
    except OSError as exc:
        raise PhaseError(f"could not clear prior report {path}: {exc}") from exc


def artifact_provenance(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise PhaseError(f"artifact provenance failed for {path}: {exc}") from exc
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": size,
    }


def empty_final_evidence() -> dict[str, Any]:
    return {
        "health": None,
        "capabilities": None,
        "metrics": None,
        "flm_evidence": None,
        "load_invariance": {"passed": False, "error": "not collected"},
        "collection_errors": [],
    }


def collect_final_evidence(
    base_url: str,
    api_key: str,
    before: dict[str, Any] | None,
) -> dict[str, Any]:
    final = empty_final_evidence()

    def collect(label: str, operation: Callable[[], Any]) -> Any:
        try:
            return operation()
        except Exception as exc:
            final["collection_errors"].append(f"{label}: {exc}")
            return None

    final["health"] = collect(
        "health",
        lambda: fetch_json(base_url, "/health", api_key),
    )
    final["capabilities"] = collect(
        "capabilities",
        lambda: fetch_json(base_url, "/v1/capabilities", api_key),
    )
    final["metrics"] = collect(
        "metrics",
        lambda: fetch_metrics(base_url, api_key),
    )
    if final["capabilities"] is not None and final["metrics"] is not None:
        final["flm_evidence"] = collect(
            "FLM evidence",
            lambda: validate_flm_evidence(
                final["capabilities"],
                final["metrics"],
            ),
        )
    if before is not None and final["flm_evidence"] is not None:
        try:
            validate_load_invariance(before, final["flm_evidence"])
            final["load_invariance"] = {"passed": True, "error": None}
        except PhaseError as exc:
            final["load_invariance"] = {
                "passed": False,
                "error": str(exc),
            }
            final["collection_errors"].append(f"load invariance: {exc}")
    return final


def build_failure_report(
    *,
    phase: str,
    error: Exception,
    provenance: dict[str, Any],
    completed: dict[str, Any],
    final: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "failed",
        "phase": phase,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "provenance": provenance,
        "completed": completed,
        "final": final,
    }


def discover_inputs(args: argparse.Namespace) -> None:
    for path, label in (
        (args.binary, "server binary"),
        (args.flm, "FLM artifact"),
        (COMPAT_SCRIPT, "compatibility SDK smoke"),
        (AGENT_SCRIPT, "agent tool SDK smoke"),
    ):
        if not path.is_file():
            raise PhaseError(f"{label} does not exist: {path}")
    if args.host != "127.0.0.1":
        raise PhaseError("server harness host must be 127.0.0.1")
    if shutil.which(args.node) is None:
        raise PhaseError(f"Node executable not found: {args.node}")
    if shutil.which(args.npm) is None:
        raise PhaseError(f"npm executable not found: {args.npm}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    clear_report_output(args.out_json)
    phase = "inputs"
    completed: dict[str, Any] = {}
    final = empty_final_evidence()
    provenance: dict[str, Any] = {
        "artifact": {
            "path": str(args.flm.resolve()),
            "sha256": None,
            "size_bytes": None,
        },
        "sdk": None,
    }
    before: dict[str, Any] | None = None
    try:
        discover_inputs(args)
        provenance["artifact"] = artifact_provenance(args.flm)

        phase = "sdk"
        sdk = ensure_openai_sdk(args)
        provenance["sdk"] = {
            "package": sdk["package"],
            "version": sdk["version"],
        }

        port = allocate_loopback_port()
        base_url = f"http://{args.host}:{port}"
        log_path = args.out_json.with_suffix(".server.log")
        startup_begin = time.monotonic()
        with running_server(args, port, log_path):
            ready_seconds = time.monotonic() - startup_begin
            try:
                phase = "initial_evidence"
                before = validate_flm_evidence(
                    fetch_json(base_url, "/v1/capabilities", args.api_key),
                    fetch_metrics(base_url, args.api_key),
                )
                completed["initial_evidence"] = before

                phase = "protocol"
                protocol = run_protocol_phases(
                    args,
                    sdk["directory"],
                    base_url,
                )
                if protocol["compat"] is not None:
                    completed["compat"] = protocol["compat"]
                if protocol["agent"] is not None:
                    completed["agent"] = protocol["agent"]
                if protocol["failures"]:
                    completed["phase_failures"] = protocol["failures"]
                    phase = "+".join(
                        failure["phase"] for failure in protocol["failures"]
                    )
                    raise PhaseError(
                        "; ".join(
                            f"{failure['phase']}: {failure['message']}"
                            for failure in protocol["failures"]
                        )
                    )
                compat = protocol["compat"]
                agent = protocol["agent"]
            finally:
                final = collect_final_evidence(base_url, args.api_key, before)

            phase = "final_evidence"
            if final["collection_errors"]:
                raise PhaseError(
                    "final evidence failed: "
                    + "; ".join(final["collection_errors"])
                )
            if not final["load_invariance"]["passed"]:
                raise PhaseError(
                    "final load invariant failed: "
                    f"{final['load_invariance']['error']}"
                )
            phase = "report"
            report = build_report(
                before,
                compat,
                agent,
                ready_seconds=ready_seconds,
                provenance=provenance,
            )
            write_report(args.out_json, report)
            return report
    except Exception as exc:
        failure = build_failure_report(
            phase=phase,
            error=exc,
            provenance=provenance,
            completed=completed,
            final=final,
        )
        write_report(args.out_json, failure)
        if isinstance(exc, PhaseError):
            raise
        raise PhaseError(f"{phase} failed: {exc}") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flm", type=Path, default=DEFAULT_FLM)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--backend", default="hip", choices=["hip"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument("--request-timeout", type=float, default=1200.0)
    parser.add_argument("--sdk-probe-timeout", type=float, default=30.0)
    parser.add_argument("--sdk-install-timeout", type=float, default=300.0)
    parser.add_argument("--openai-sdk-dir", type=Path, default=DEFAULT_OPENAI_SDK_DIR)
    parser.add_argument("--node", default="node")
    parser.add_argument("--npm", default="npm")
    args = parser.parse_args(argv)
    args.no_download = True
    for field in (
        "startup_timeout",
        "request_timeout",
        "sdk_probe_timeout",
        "sdk_install_timeout",
    ):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.max_context <= 0:
        parser.error("--max-context must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except PhaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {args.out_json}")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
