#!/usr/bin/env python3
"""Run the real Qwen3.6 FLM server through its OpenAI protocol surface."""

import argparse
import contextlib
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
PROCESS_GRACE_SECONDS = 5
READY_POLL_SECONDS = 0.25
HTTP_TIMEOUT_SECONDS = 5.0
SMOKE_JSON_PREFIX = "SUPERSONIC_SMOKE_JSON="


class PhaseError(RuntimeError):
    pass


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


def _terminate_and_reap_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        process.communicate()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=PROCESS_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.communicate()


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

    requests = _mapping(report.get("requests"), "report requests")
    compat = _mapping(requests.get("compat"), "report requests.compat")
    agent = _mapping(requests.get("agent"), "report requests.agent")
    for section in (
        "auth",
        "models",
        "tokenizer",
        "chat",
        "chat_stream",
        "responses",
        "responses_stream",
        "reasoning",
        "usage_accounting",
        "repeated_request",
    ):
        _mapping(compat.get(section), f"report requests.compat.{section}")
    for section in ("chat_tool_loop", "responses_tool_loop"):
        _mapping(agent.get(section), f"report requests.agent.{section}")

    unauthorized = _path(report, "requests", "compat", "auth", "unauthorized_status")
    _strict_int(unauthorized, "auth unauthorized_status")
    _expect(unauthorized, 401, "auth unauthorized_status")
    true_paths = [
        ("requests", "compat", "models", "listed"),
        ("requests", "compat", "models", "retrieved"),
        ("requests", "compat", "tokenizer", "roundtrip"),
        ("requests", "compat", "chat", "assistant_result"),
        ("requests", "compat", "chat_stream", "saw_delta"),
        ("requests", "compat", "chat_stream", "saw_terminal"),
        ("requests", "compat", "chat_stream", "saw_usage"),
        ("requests", "compat", "responses", "assistant_result"),
        ("requests", "compat", "responses_stream", "saw_delta"),
        ("requests", "compat", "responses_stream", "saw_completed"),
        ("requests", "compat", "reasoning", "assistant_result"),
        ("requests", "compat", "reasoning", "request_accepted"),
        ("requests", "compat", "usage_accounting", "chat_valid"),
        ("requests", "compat", "usage_accounting", "chat_stream_valid"),
        ("requests", "compat", "usage_accounting", "responses_valid"),
        ("requests", "compat", "usage_accounting", "responses_stream_valid"),
        ("requests", "compat", "repeated_request", "assistant_result"),
        ("requests", "agent", "chat_tool_loop", "assistant_result"),
        ("requests", "agent", "responses_tool_loop", "assistant_result"),
    ]
    for path in true_paths:
        _require_true(report, *path)
    reasoning_observed = _path(
        report,
        "requests",
        "compat",
        "reasoning",
        "reasoning_observed",
    )
    if not isinstance(reasoning_observed, bool):
        raise PhaseError("reasoning reasoning_observed must be a boolean")
    _expect(
        _path(report, "requests", "compat", "reasoning", "visible_think_tags"),
        False,
        "reasoning visible_think_tags",
    )

    startup = _mapping(report.get("startup"), "report startup")
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

    throughput = _mapping(report.get("throughput"), "report throughput")
    for field in (
        "first_token_seconds",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
    ):
        if _finite_number(throughput.get(field), f"throughput {field}") <= 0:
            raise PhaseError(f"throughput {field} must be positive")

    cancellation = _mapping(report.get("cancellation"), "report cancellation")
    for field in (
        "aborted_after_first_delta",
        "saw_delta",
        "scheduler_released",
    ):
        _expect(cancellation.get(field), True, f"cancellation {field}")
    for field in ("active_requests", "queued_requests"):
        _strict_int(cancellation.get(field), f"cancellation {field}")
        _expect(cancellation.get(field), 0, f"cancellation {field}")
    if _finite_number(
        cancellation.get("release_seconds"),
        "cancellation release_seconds",
    ) < 0:
        raise PhaseError("cancellation release_seconds must be non-negative")

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
    if result.returncode != 0:
        raise PhaseError(
            f"{phase} failed with exit {result.returncode}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return result


def ensure_openai_sdk(args: argparse.Namespace) -> Path:
    sdk_dir = args.openai_sdk_dir.resolve()
    sdk_dir.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(
        [
            args.node,
            "-e",
            "console.log(require.resolve('openai'))",
        ],
        cwd=sdk_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if probe.returncode == 0:
        return sdk_dir
    run_process(
        [
            args.npm,
            "install",
            "--no-audit",
            "--no-fund",
            "--prefix",
            str(sdk_dir),
            "openai@6",
        ],
        cwd=ROOT,
        env=None,
        timeout=args.sdk_install_timeout,
        phase="OpenAI SDK install",
    )
    verify = subprocess.run(
        [args.node, "-e", "console.log(require.resolve('openai'))"],
        cwd=sdk_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if verify.returncode != 0:
        raise PhaseError(
            "OpenAI SDK install completed but the package cannot be resolved: "
            f"{verify.stderr.strip()}"
        )
    return sdk_dir


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
    )
    if result.stderr.strip():
        print(result.stderr, file=sys.stderr, end="")
    print(result.stdout, end="")
    return parse_smoke_output(result.stdout, phase)


def build_report(
    before: dict[str, Any],
    compat: dict[str, Any],
    agent: dict[str, Any],
    *,
    ready_seconds: float,
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
    }
    report = {
        "model": EXPECTED_MODEL,
        "source": EXPECTED_SOURCE,
        "load_sequence": before["load_sequence"],
        "native_int4": before["native_int4"],
        "bf16_fallback": before["bf16_fallback"],
        "requests": {
            "compat": _mapping(compat.get("requests"), "compat requests"),
            "agent": _mapping(agent.get("requests"), "agent requests"),
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
    discover_inputs(args)
    sdk_dir = ensure_openai_sdk(args)
    port = allocate_loopback_port()
    base_url = f"http://{args.host}:{port}"
    log_path = args.out_json.with_suffix(".server.log")
    startup_begin = time.monotonic()
    with running_server(args, port, log_path):
        ready_seconds = time.monotonic() - startup_begin
        before = validate_flm_evidence(
            fetch_json(base_url, "/v1/capabilities", args.api_key),
            fetch_metrics(base_url, args.api_key),
        )
        compat = run_sdk_smoke(
            args,
            COMPAT_SCRIPT,
            sdk_dir,
            base_url,
            "OpenAI compatibility",
        )
        agent = run_sdk_smoke(
            args,
            AGENT_SCRIPT,
            sdk_dir,
            base_url,
            "OpenAI agent tool",
        )
        after = validate_flm_evidence(
            fetch_json(base_url, "/v1/capabilities", args.api_key),
            fetch_metrics(base_url, args.api_key),
        )
        validate_load_invariance(before, after)
        report = build_report(
            before,
            compat,
            agent,
            ready_seconds=ready_seconds,
        )
        write_report(args.out_json, report)
        return report


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
    parser.add_argument("--sdk-install-timeout", type=float, default=300.0)
    parser.add_argument("--openai-sdk-dir", type=Path, default=DEFAULT_OPENAI_SDK_DIR)
    parser.add_argument("--node", default="node")
    parser.add_argument("--npm", default="npm")
    args = parser.parse_args(argv)
    args.no_download = True
    for field in ("startup_timeout", "request_timeout", "sdk_install_timeout"):
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
