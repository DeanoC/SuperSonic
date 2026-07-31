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
from dataclasses import dataclass
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
EXPECTED_FAMILY = "qwen3.6-moe"
EXPECTED_BACKEND = "HIP"
EXPECTED_SOURCE = "flm"
EXPECTED_MAX_CONTEXT = 4096
EXPECTED_NATIVE_INT4 = 330
EXPECTED_BF16_FALLBACK = 0
EXPECTED_REQUIRED_WEIGHTS = 693
EXPECTED_RAW_DENSE_WEIGHTS = 363
EXPECTED_ARCHITECTURE_ID = 2
EXPECTED_MODEL_ID = 2
EXPECTED_STORAGE_ABI_IDS = [8]
EXPECTED_TRANSFER_BACKEND = "pageable_h2d"
EXPECTED_SCHEDULER = {
    "active_requests": 0,
    "queued_requests": 0,
    "max_queued_requests": 32,
    "queue_timeout_ms": 30_000,
}
EXPECTED_FLM_FEATURES = {
    "plain_prefill_decode": True,
    "native_dflash_generate": False,
    "prefix_snapshot": False,
    "disk_prefix_snapshot": False,
}
OPENAI_SDK_VERSION = "6.49.0"
PROCESS_GRACE_SECONDS = 5
READY_POLL_SECONDS = 0.25
HTTP_TIMEOUT_SECONDS = 5.0
SMOKE_JSON_PREFIX = "SUPERSONIC_SMOKE_JSON="
EXPECTED_ENDPOINTS = [
    "/v1/models",
    "/v1/models/{model}",
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/tokenize",
    "/v1/detokenize",
    "/v1/responses",
    "/health",
    "/v1/health",
    "/ready",
    "/v1/ready",
    "/v1/capabilities",
    "/metrics",
]
FINAL_METRIC_KEYS = {
    "supersonic_ready",
    "supersonic_active_requests",
    "supersonic_queued_requests",
    "supersonic_generation_active",
    "supersonic_generation_queued",
    "supersonic_max_queued_requests",
    "supersonic_queue_timeout_ms",
    "supersonic_max_context",
    "supersonic_prefix_cache_enabled",
    "supersonic_prefix_cache_entries",
    "supersonic_prefix_cache_resident_bytes",
    "supersonic_prefix_cache_max_bytes",
    "supersonic_prefix_cache_hits",
    "supersonic_prefix_cache_misses",
    "supersonic_prefix_cache_cached_tokens",
    "supersonic_prefix_cache_evictions",
    "supersonic_prefix_cache_disk_writes",
    "supersonic_prefix_cache_disk_reads",
    "supersonic_prefix_cache_restore_failures",
    "supersonic_prefix_cache_admission_skips",
    "supersonic_dflash_last_rounds",
    "supersonic_dflash_last_accepted_total",
    "supersonic_dflash_last_decode_ms",
    "supersonic_model_loads_total",
    "supersonic_flm_native_int4_direct_weights",
    "supersonic_flm_bf16_fallback_weights",
    "supersonic_flm_source_bytes",
    "supersonic_flm_device_upload_bytes",
    "supersonic_flm_startup_seconds",
}


class PhaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class FlmProfileExpectations:
    storage_abi_ids: tuple[int, ...]
    row_group_int4: int
    tile_int4_v1: int
    native_int4: int
    bf16_fallback: int

    def __post_init__(self) -> None:
        if not isinstance(self.storage_abi_ids, tuple) or not self.storage_abi_ids:
            raise PhaseError("expected storage_abi_ids must be a non-empty tuple")
        if len(set(self.storage_abi_ids)) != len(self.storage_abi_ids):
            raise PhaseError("expected storage_abi_ids must be unique")
        for index, storage_abi_id in enumerate(self.storage_abi_ids):
            if isinstance(storage_abi_id, bool) or not isinstance(storage_abi_id, int):
                raise PhaseError(
                    f"expected storage_abi_ids[{index}] must be an integer"
                )
            if storage_abi_id <= 0 or storage_abi_id > 0xFFFF:
                raise PhaseError(
                    f"expected storage_abi_ids[{index}] must be in 1..65535"
                )
        for field in (
            "row_group_int4",
            "tile_int4_v1",
            "native_int4",
            "bf16_fallback",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise PhaseError(f"expected {field} must be an integer")
            if value < 0:
                raise PhaseError(f"expected {field} must be non-negative")
        if self.native_int4 != self.row_group_int4 + self.tile_int4_v1:
            raise PhaseError(
                "expected native_int4 must equal row_group_int4 + tile_int4_v1"
            )

    def as_json(self) -> dict[str, Any]:
        return {
            "storage_abi_ids": list(self.storage_abi_ids),
            "row_group_int4": self.row_group_int4,
            "tile_int4_v1": self.tile_int4_v1,
            "native_int4": self.native_int4,
            "bf16_fallback": self.bf16_fallback,
        }


LEGACY_FLM_PROFILE = FlmProfileExpectations(
    storage_abi_ids=tuple(EXPECTED_STORAGE_ABI_IDS),
    row_group_int4=0,
    tile_int4_v1=EXPECTED_NATIVE_INT4,
    native_int4=EXPECTED_NATIVE_INT4,
    bf16_fallback=EXPECTED_BF16_FALLBACK,
)


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


def _strict_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise PhaseError(f"{label} must be a boolean")
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


def _nonnegative_int(value: object, label: str) -> int:
    parsed = _strict_int(value, label)
    if parsed < 0:
        raise PhaseError(f"{label} must be non-negative")
    return parsed


def _validate_finite_tree(value: object, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_finite_tree(child, f"{label}.{key}")
        return
    _finite_number(value, label)


def _validate_startup_schema(value: object, label: str) -> dict[str, Any]:
    startup = _exact_mapping(
        value,
        {"total_seconds", "exclusive_components"},
        label,
    )
    components = _exact_mapping(
        startup["exclusive_components"],
        {"source_open", "tokenizer_seconds", "descriptor_seconds"},
        f"{label}.exclusive_components",
    )
    source_open = _exact_mapping(
        components["source_open"],
        {"total_seconds", "exclusive_phases"},
        f"{label}.exclusive_components.source_open",
    )
    phases = _exact_mapping(
        source_open["exclusive_phases"],
        {"store_open_seconds", "config_seconds", "direct_plan_seconds"},
        f"{label}.exclusive_components.source_open.exclusive_phases",
    )
    total = _finite_number(startup["total_seconds"], f"{label}.total_seconds")
    if total <= 0:
        raise PhaseError(f"{label}.total_seconds must be positive")
    for field, number in (
        ("exclusive_components.tokenizer_seconds", components["tokenizer_seconds"]),
        ("exclusive_components.descriptor_seconds", components["descriptor_seconds"]),
        (
            "exclusive_components.source_open.total_seconds",
            source_open["total_seconds"],
        ),
        (
            "exclusive_components.source_open.exclusive_phases.store_open_seconds",
            phases["store_open_seconds"],
        ),
        (
            "exclusive_components.source_open.exclusive_phases.config_seconds",
            phases["config_seconds"],
        ),
        (
            "exclusive_components.source_open.exclusive_phases.direct_plan_seconds",
            phases["direct_plan_seconds"],
        ),
    ):
        if _finite_number(number, f"{label}.{field}") < 0:
            raise PhaseError(f"{label}.{field} must be non-negative")
    return startup


def _validate_flm_payload_schema(
    value: object,
    label: str,
    *,
    expected_profile: FlmProfileExpectations = LEGACY_FLM_PROFILE,
) -> dict[str, Any]:
    flm = _exact_mapping(
        value,
        {
            "source",
            "file",
            "architecture_id",
            "model_id",
            "storage_abi_ids",
            "required_weights",
            "raw_dense_weights",
            "native_int4_direct_weights",
            "bf16_fallback_weights",
            "transfer_backend",
            "source_bytes",
            "device_upload_bytes",
            "startup_seconds",
            "startup",
            "load_sequence",
            "source_open_count",
            "resident_allocation_count",
            "features",
        },
        label,
    )
    exact_integers = {
        "architecture_id": EXPECTED_ARCHITECTURE_ID,
        "model_id": EXPECTED_MODEL_ID,
        "required_weights": EXPECTED_REQUIRED_WEIGHTS,
        "raw_dense_weights": EXPECTED_RAW_DENSE_WEIGHTS,
        "native_int4_direct_weights": expected_profile.native_int4,
        "bf16_fallback_weights": expected_profile.bf16_fallback,
        "load_sequence": 1,
        "source_open_count": 1,
    }
    for field, expected in exact_integers.items():
        _strict_int(flm[field], f"{label}.{field}")
        _expect(flm[field], expected, f"{label}.{field}")
    storage_abis = flm["storage_abi_ids"]
    if not isinstance(storage_abis, list):
        raise PhaseError(f"{label}.storage_abi_ids must be a list")
    for index, storage_abi in enumerate(storage_abis):
        _strict_int(storage_abi, f"{label}.storage_abi_ids[{index}]")
    _expect(
        storage_abis,
        list(expected_profile.storage_abi_ids),
        f"{label}.storage_abi_ids",
    )
    _expect(flm["source"], EXPECTED_SOURCE, f"{label}.source")
    filename = _nonempty_string(flm["file"], f"{label}.file")
    if not filename.endswith(".flm"):
        raise PhaseError(f"{label}.file must name an FLM artifact")
    _expect(
        flm["transfer_backend"],
        EXPECTED_TRANSFER_BACKEND,
        f"{label}.transfer_backend",
    )
    source_bytes = _positive_int(flm["source_bytes"], f"{label}.source_bytes")
    device_upload_bytes = _positive_int(
        flm["device_upload_bytes"],
        f"{label}.device_upload_bytes",
    )
    if device_upload_bytes > source_bytes:
        raise PhaseError(f"{label}.device_upload_bytes exceeds source_bytes")
    _positive_int(
        flm["resident_allocation_count"],
        f"{label}.resident_allocation_count",
    )
    startup_seconds = _finite_number(
        flm["startup_seconds"],
        f"{label}.startup_seconds",
    )
    if startup_seconds <= 0:
        raise PhaseError(f"{label}.startup_seconds must be positive")
    startup = _validate_startup_schema(flm["startup"], f"{label}.startup")
    _expect(
        startup_seconds,
        startup["total_seconds"],
        f"{label}.startup_seconds",
    )
    features = _exact_mapping(
        flm["features"],
        {
            "plain_prefill_decode",
            "native_dflash_generate",
            "prefix_snapshot",
            "disk_prefix_snapshot",
        },
        f"{label}.features",
    )
    for field, expected in EXPECTED_FLM_FEATURES.items():
        _strict_bool(features[field], f"{label}.features.{field}")
        _expect(features[field], expected, f"{label}.features.{field}")
    if (
        flm["raw_dense_weights"]
        + flm["native_int4_direct_weights"]
        + flm["bf16_fallback_weights"]
        != flm["required_weights"]
    ):
        raise PhaseError(f"{label} direct-profile counts do not add up")
    return flm


def _validate_prefix_cache_schema(value: object, label: str) -> dict[str, Any]:
    cache = _exact_mapping(
        value,
        {
            "enabled",
            "dir",
            "min_tokens",
            "max_entries",
            "max_bytes",
            "resident_bytes",
            "entries",
            "hits",
            "misses",
            "cached_tokens",
            "evictions",
            "disk_writes",
            "disk_reads",
            "restore_failures",
            "admission_skips",
        },
        label,
    )
    _strict_bool(cache["enabled"], f"{label}.enabled")
    _expect(cache["enabled"], False, f"{label}.enabled")
    if not isinstance(cache["dir"], str):
        raise PhaseError(f"{label}.dir must be a string")
    _expect(cache["dir"], "", f"{label}.dir")
    exact_integers = {
        "min_tokens": 128,
        "max_entries": 1,
        "resident_bytes": 0,
        "entries": 0,
        "hits": 0,
        "misses": 0,
        "cached_tokens": 0,
        "evictions": 0,
        "disk_writes": 0,
        "disk_reads": 0,
        "restore_failures": 0,
        "admission_skips": 0,
    }
    for field, expected in exact_integers.items():
        _strict_int(cache[field], f"{label}.{field}")
        _expect(cache[field], expected, f"{label}.{field}")
    _positive_int(cache["max_bytes"], f"{label}.max_bytes")
    return cache


def _validate_scheduler_schema(value: object, label: str) -> dict[str, Any]:
    scheduler = _exact_mapping(
        value,
        set(EXPECTED_SCHEDULER),
        label,
    )
    for field, expected in EXPECTED_SCHEDULER.items():
        _strict_int(scheduler[field], f"{label}.{field}")
        _expect(scheduler[field], expected, f"{label}.{field}")
    return scheduler


def _validate_endpoints(value: object, label: str) -> list[str]:
    if not isinstance(value, list):
        raise PhaseError(f"{label} must be a list")
    for index, endpoint in enumerate(value):
        _nonempty_string(endpoint, f"{label}[{index}]")
    _expect(value, EXPECTED_ENDPOINTS, label)
    return value


def _validate_capabilities_schema(
    value: object,
    label: str,
    *,
    expected_profile: FlmProfileExpectations = LEGACY_FLM_PROFILE,
) -> dict[str, Any]:
    capabilities = _exact_mapping(
        value,
        {
            "model",
            "family",
            "backend",
            "ready",
            "max_context",
            "endpoints",
            "chat",
            "completions",
            "responses",
            "streaming",
            "stream_usage",
            "tools",
            "reasoning",
            "scheduler",
            "prefix_cache",
            "flm",
        },
        label,
    )
    _expect(capabilities["model"], EXPECTED_MODEL, f"{label}.model")
    _expect(capabilities["family"], EXPECTED_FAMILY, f"{label}.family")
    _expect(capabilities["backend"], EXPECTED_BACKEND, f"{label}.backend")
    _strict_bool(capabilities["ready"], f"{label}.ready")
    _expect(capabilities["ready"], True, f"{label}.ready")
    _strict_int(capabilities["max_context"], f"{label}.max_context")
    _expect(
        capabilities["max_context"],
        EXPECTED_MAX_CONTEXT,
        f"{label}.max_context",
    )
    _validate_endpoints(capabilities["endpoints"], f"{label}.endpoints")
    for field in (
        "chat",
        "completions",
        "responses",
        "streaming",
        "stream_usage",
        "tools",
        "reasoning",
    ):
        _strict_bool(capabilities[field], f"{label}.{field}")
        _expect(capabilities[field], True, f"{label}.{field}")
    _validate_scheduler_schema(capabilities["scheduler"], f"{label}.scheduler")
    _validate_prefix_cache_schema(
        capabilities["prefix_cache"],
        f"{label}.prefix_cache",
    )
    _validate_flm_payload_schema(
        capabilities["flm"],
        f"{label}.flm",
        expected_profile=expected_profile,
    )
    return capabilities


def _validate_health_schema(
    value: object,
    label: str,
    *,
    expected_profile: FlmProfileExpectations = LEGACY_FLM_PROFILE,
) -> dict[str, Any]:
    health = _exact_mapping(
        value,
        {
            "status",
            "ready",
            "model",
            "max_context",
            "active_requests",
            "queued_requests",
            "max_queued_requests",
            "prefix_cache_entries",
            "flm",
        },
        label,
    )
    _expect(health["status"], "ok", f"{label}.status")
    _strict_bool(health["ready"], f"{label}.ready")
    _expect(health["ready"], True, f"{label}.ready")
    _expect(health["model"], EXPECTED_MODEL, f"{label}.model")
    exact_integers = {
        "max_context": EXPECTED_MAX_CONTEXT,
        "active_requests": 0,
        "queued_requests": 0,
        "max_queued_requests": EXPECTED_SCHEDULER["max_queued_requests"],
        "prefix_cache_entries": 0,
    }
    for field, expected in exact_integers.items():
        _strict_int(health[field], f"{label}.{field}")
        _expect(health[field], expected, f"{label}.{field}")
    _validate_flm_payload_schema(
        health["flm"],
        f"{label}.flm",
        expected_profile=expected_profile,
    )
    return health


def _validate_metrics_schema(
    value: object,
    label: str,
    *,
    capabilities: dict[str, Any] | None = None,
    health: dict[str, Any] | None = None,
    expected_profile: FlmProfileExpectations = LEGACY_FLM_PROFILE,
) -> dict[str, int | float]:
    metrics = _exact_mapping(value, FINAL_METRIC_KEYS, label)
    for name in FINAL_METRIC_KEYS - {"supersonic_flm_startup_seconds"}:
        _nonnegative_int(metrics[name], f"{label}.{name}")
    startup_seconds = _finite_number(
        metrics["supersonic_flm_startup_seconds"],
        f"{label}.supersonic_flm_startup_seconds",
    )
    if startup_seconds <= 0:
        raise PhaseError(f"{label}.supersonic_flm_startup_seconds must be positive")
    exact_metrics = {
        "supersonic_ready": 1,
        "supersonic_active_requests": 0,
        "supersonic_queued_requests": 0,
        "supersonic_generation_active": 0,
        "supersonic_generation_queued": 0,
        "supersonic_max_queued_requests": EXPECTED_SCHEDULER[
            "max_queued_requests"
        ],
        "supersonic_queue_timeout_ms": EXPECTED_SCHEDULER["queue_timeout_ms"],
        "supersonic_max_context": EXPECTED_MAX_CONTEXT,
        "supersonic_prefix_cache_enabled": 0,
        "supersonic_prefix_cache_entries": 0,
        "supersonic_prefix_cache_resident_bytes": 0,
        "supersonic_prefix_cache_hits": 0,
        "supersonic_prefix_cache_misses": 0,
        "supersonic_prefix_cache_cached_tokens": 0,
        "supersonic_prefix_cache_evictions": 0,
        "supersonic_prefix_cache_disk_writes": 0,
        "supersonic_prefix_cache_disk_reads": 0,
        "supersonic_prefix_cache_restore_failures": 0,
        "supersonic_prefix_cache_admission_skips": 0,
        "supersonic_dflash_last_rounds": 0,
        "supersonic_dflash_last_accepted_total": 0,
        "supersonic_dflash_last_decode_ms": 0,
        "supersonic_model_loads_total": 1,
        "supersonic_flm_native_int4_direct_weights": expected_profile.native_int4,
        "supersonic_flm_bf16_fallback_weights": expected_profile.bf16_fallback,
    }
    for name, expected in exact_metrics.items():
        _expect(metrics[name], expected, f"{label}.{name}")
    _positive_int(
        metrics["supersonic_prefix_cache_max_bytes"],
        f"{label}.supersonic_prefix_cache_max_bytes",
    )
    source_bytes = _positive_int(
        metrics["supersonic_flm_source_bytes"],
        f"{label}.supersonic_flm_source_bytes",
    )
    device_upload_bytes = _positive_int(
        metrics["supersonic_flm_device_upload_bytes"],
        f"{label}.supersonic_flm_device_upload_bytes",
    )
    if device_upload_bytes > source_bytes:
        raise PhaseError(
            f"{label}.supersonic_flm_device_upload_bytes exceeds source bytes"
        )

    if capabilities is not None:
        scheduler = capabilities["scheduler"]
        cache = capabilities["prefix_cache"]
        flm = capabilities["flm"]
        cross_fields = {
            "supersonic_ready": int(capabilities["ready"]),
            "supersonic_active_requests": scheduler["active_requests"],
            "supersonic_queued_requests": scheduler["queued_requests"],
            "supersonic_max_queued_requests": scheduler["max_queued_requests"],
            "supersonic_queue_timeout_ms": scheduler["queue_timeout_ms"],
            "supersonic_max_context": capabilities["max_context"],
            "supersonic_prefix_cache_enabled": int(cache["enabled"]),
            "supersonic_prefix_cache_entries": cache["entries"],
            "supersonic_prefix_cache_resident_bytes": cache["resident_bytes"],
            "supersonic_prefix_cache_max_bytes": cache["max_bytes"],
            "supersonic_prefix_cache_hits": cache["hits"],
            "supersonic_prefix_cache_misses": cache["misses"],
            "supersonic_prefix_cache_cached_tokens": cache["cached_tokens"],
            "supersonic_prefix_cache_evictions": cache["evictions"],
            "supersonic_prefix_cache_disk_writes": cache["disk_writes"],
            "supersonic_prefix_cache_disk_reads": cache["disk_reads"],
            "supersonic_prefix_cache_restore_failures": cache["restore_failures"],
            "supersonic_prefix_cache_admission_skips": cache["admission_skips"],
            "supersonic_flm_native_int4_direct_weights": flm[
                "native_int4_direct_weights"
            ],
            "supersonic_flm_bf16_fallback_weights": flm[
                "bf16_fallback_weights"
            ],
            "supersonic_flm_source_bytes": flm["source_bytes"],
            "supersonic_flm_device_upload_bytes": flm["device_upload_bytes"],
            "supersonic_flm_startup_seconds": flm["startup_seconds"],
        }
        for name, expected in cross_fields.items():
            _expect(metrics[name], expected, f"{label}.{name}")
    if health is not None:
        health_flm = health["flm"]
        health_cross_fields = {
            "supersonic_ready": int(health["ready"]),
            "supersonic_active_requests": health["active_requests"],
            "supersonic_queued_requests": health["queued_requests"],
            "supersonic_max_queued_requests": health["max_queued_requests"],
            "supersonic_max_context": health["max_context"],
            "supersonic_prefix_cache_entries": health["prefix_cache_entries"],
            "supersonic_model_loads_total": health_flm["load_sequence"],
            "supersonic_flm_native_int4_direct_weights": health_flm[
                "native_int4_direct_weights"
            ],
            "supersonic_flm_bf16_fallback_weights": health_flm[
                "bf16_fallback_weights"
            ],
            "supersonic_flm_source_bytes": health_flm["source_bytes"],
            "supersonic_flm_device_upload_bytes": health_flm[
                "device_upload_bytes"
            ],
            "supersonic_flm_startup_seconds": health_flm["startup_seconds"],
        }
        for name, expected in health_cross_fields.items():
            _expect(metrics[name], expected, f"{label}.{name}")
    return metrics


def validate_flm_evidence(
    capabilities: dict[str, Any],
    metrics: dict[str, int | float],
    *,
    expected_profile: FlmProfileExpectations = LEGACY_FLM_PROFILE,
) -> dict[str, Any]:
    capabilities = _validate_capabilities_schema(
        capabilities,
        "capabilities",
        expected_profile=expected_profile,
    )
    metrics = _validate_metrics_schema(
        metrics,
        "metrics",
        capabilities=capabilities,
        expected_profile=expected_profile,
    )
    flm = capabilities["flm"]
    scheduler = capabilities["scheduler"]
    return {
        "model": EXPECTED_MODEL,
        "source": EXPECTED_SOURCE,
        "load_sequence": 1,
        "expected_flm_profile": expected_profile.as_json(),
        "native_int4": expected_profile.native_int4,
        "bf16_fallback": expected_profile.bf16_fallback,
        "model_loads_total": 1,
        "startup": flm["startup"],
        "transfer_backend": flm["transfer_backend"],
        "source_bytes": flm["source_bytes"],
        "device_upload_bytes": flm["device_upload_bytes"],
        "source_open_count": 1,
        "resident_allocation_count": flm["resident_allocation_count"],
        "scheduler": scheduler,
    }


def _validate_flm_profile_snapshot(
    value: object,
    label: str,
    *,
    expected_profile: FlmProfileExpectations = LEGACY_FLM_PROFILE,
) -> dict[str, Any]:
    profile = _exact_mapping(
        value,
        {
            "storage_abi_ids",
            "row_group_int4",
            "tile_int4_v1",
            "native_int4",
            "bf16_fallback",
        },
        label,
    )
    storage_abi_ids = profile["storage_abi_ids"]
    if not isinstance(storage_abi_ids, list):
        raise PhaseError(f"{label}.storage_abi_ids must be a list")
    for index, storage_abi_id in enumerate(storage_abi_ids):
        _strict_int(storage_abi_id, f"{label}.storage_abi_ids[{index}]")
    for field in (
        "row_group_int4",
        "tile_int4_v1",
        "native_int4",
        "bf16_fallback",
    ):
        _nonnegative_int(profile[field], f"{label}.{field}")
    _expect(profile, expected_profile.as_json(), label)
    return profile


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


def _validate_compat_transport(value: object) -> None:
    transport = _exact_mapping(
        value,
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
        for key, field_value in evidence.items():
            if key == "token_count":
                _positive_int(
                    field_value,
                    "compat transport.tokenizer.token_count",
                )
            else:
                _expect(field_value, True, f"compat transport.{section}.{key}")


def _validate_compat_usage(value: object) -> None:
    usage = _exact_mapping(
        value,
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


def _validate_compat_throughput(value: object) -> None:
    throughput = _exact_mapping(
        value,
        {
            "first_token_seconds",
            "prefill_tokens_per_second",
            "decode_tokens_per_second",
        },
        "compat throughput",
    )
    for key, field_value in throughput.items():
        if _finite_number(field_value, f"compat throughput.{key}") <= 0:
            raise PhaseError(f"compat throughput.{key} must be positive")


def validate_compat_report(report: dict[str, Any]) -> dict[str, Any]:
    _exact_mapping(
        report,
        {"transport", "semantic_quality", "usage", "throughput"},
        "compat report",
    )
    _validate_compat_transport(report["transport"])

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

    _validate_compat_usage(report["usage"])
    _validate_compat_throughput(report["throughput"])
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


def validate_report(
    report: dict[str, Any],
    *,
    expected_profile: FlmProfileExpectations = LEGACY_FLM_PROFILE,
) -> dict[str, Any]:
    expected_keys = {
        "model",
        "source",
        "load_sequence",
        "expected_flm_profile",
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
    _validate_flm_profile_snapshot(
        report.get("expected_flm_profile"),
        "report expected_flm_profile",
        expected_profile=expected_profile,
    )
    for field, expected in (
        ("load_sequence", 1),
        ("native_int4", expected_profile.native_int4),
        ("bf16_fallback", expected_profile.bf16_fallback),
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
    timeout_error: subprocess.TimeoutExpired | None = None
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        timeout_error = exc
        stdout, stderr = "", ""
    finally:
        _terminate_and_reap_process_group(process)
    if timeout_error is not None:
        raise PhaseError(f"{phase} timed out after {timeout:g}s") from timeout_error
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
    expected_profile: FlmProfileExpectations = LEGACY_FLM_PROFILE,
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
        "expected_flm_profile": before["expected_flm_profile"],
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
    return validate_report(report, expected_profile=expected_profile)


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
    *,
    expected_profile: FlmProfileExpectations = LEGACY_FLM_PROFILE,
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
                expected_profile=expected_profile,
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


def _reject_nonfinite_json(value: object, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite_json(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite_json(child, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise PhaseError(f"{label} must be finite")


def _validate_flm_snapshot(
    value: object,
    label: str,
    *,
    expected_profile: FlmProfileExpectations = LEGACY_FLM_PROFILE,
) -> dict[str, Any]:
    snapshot = _exact_mapping(
        value,
        {
            "model",
            "source",
            "load_sequence",
            "expected_flm_profile",
            "native_int4",
            "bf16_fallback",
            "model_loads_total",
            "startup",
            "transfer_backend",
            "source_bytes",
            "device_upload_bytes",
            "source_open_count",
            "resident_allocation_count",
            "scheduler",
        },
        label,
    )
    _expect(snapshot["model"], EXPECTED_MODEL, f"{label}.model")
    _expect(snapshot["source"], EXPECTED_SOURCE, f"{label}.source")
    _validate_flm_profile_snapshot(
        snapshot["expected_flm_profile"],
        f"{label}.expected_flm_profile",
        expected_profile=expected_profile,
    )
    for field, expected in (
        ("load_sequence", 1),
        ("native_int4", expected_profile.native_int4),
        ("bf16_fallback", expected_profile.bf16_fallback),
        ("model_loads_total", 1),
        ("source_open_count", 1),
    ):
        _strict_int(snapshot[field], f"{label}.{field}")
        _expect(snapshot[field], expected, f"{label}.{field}")
    for field in (
        "source_bytes",
        "device_upload_bytes",
        "resident_allocation_count",
    ):
        _positive_int(snapshot[field], f"{label}.{field}")
    if snapshot["device_upload_bytes"] > snapshot["source_bytes"]:
        raise PhaseError(f"{label}.device_upload_bytes exceeds source_bytes")
    _expect(
        snapshot["transfer_backend"],
        EXPECTED_TRANSFER_BACKEND,
        f"{label}.transfer_backend",
    )
    _validate_startup_schema(snapshot["startup"], f"{label}.startup")
    _validate_scheduler_schema(
        snapshot["scheduler"],
        f"{label}.scheduler",
    )
    return snapshot


def _validate_partial_compat_report(report: object) -> dict[str, Any]:
    partial = _exact_mapping(
        report,
        {"transport", "semantic_quality", "usage", "throughput"},
        "partial compat report",
    )
    _validate_compat_transport(partial["transport"])
    _validate_compat_usage(partial["usage"])
    _validate_compat_throughput(partial["throughput"])
    semantics = _exact_mapping(
        partial["semantic_quality"],
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
        "partial compat semantic_quality",
    )
    child_results = []
    for name, expected in (
        ("chat", "hello"),
        ("completions", "hello"),
        ("repeated_request", "ready"),
    ):
        canary = _exact_mapping(
            semantics[name],
            {"expected", "actual", "finish_reason", "passed"},
            f"partial semantic {name}",
        )
        _expect(
            canary["expected"],
            expected,
            f"partial semantic {name}.expected",
        )
        if not isinstance(canary["actual"], str):
            raise PhaseError(f"partial semantic {name}.actual must be a string")
        finish_reason = _nonempty_string(
            canary["finish_reason"],
            f"partial semantic {name}.finish_reason",
        )
        if finish_reason not in {"stop", "length", "tool_calls", "content_filter"}:
            raise PhaseError(
                f"partial semantic {name}.finish_reason is unsupported: "
                f"{finish_reason!r}"
            )
        passed = _strict_bool(
            canary["passed"],
            f"partial semantic {name}.passed",
        )
        predicate = canary["actual"] == expected and finish_reason == "stop"
        _expect(passed, predicate, f"partial semantic {name}.passed")
        child_results.append(predicate)

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
        "partial semantic chat_stream",
    )
    _expect(
        chat_stream["expected"],
        "hello",
        "partial semantic chat_stream.expected",
    )
    if not isinstance(chat_stream.get("actual"), str):
        raise PhaseError("partial semantic chat_stream.actual must be a string")
    finish_reason = _nonempty_string(
        chat_stream["finish_reason"],
        "partial semantic chat_stream.finish_reason",
    )
    if finish_reason not in {"stop", "length", "tool_calls", "content_filter"}:
        raise PhaseError(
            "partial semantic chat_stream.finish_reason is unsupported: "
            f"{finish_reason!r}"
        )
    terminal_count = _strict_int(
        chat_stream["terminal_count"],
        "partial semantic chat_stream.terminal_count",
    )
    terminal_before_usage = _strict_bool(
        chat_stream["terminal_last_before_usage"],
        "partial semantic chat_stream.terminal_last_before_usage",
    )
    usage_last = _strict_bool(
        chat_stream["usage_last"],
        "partial semantic chat_stream.usage_last",
    )
    passed = _strict_bool(
        chat_stream["passed"],
        "partial semantic chat_stream.passed",
    )
    predicate = (
        chat_stream["actual"] == "hello"
        and finish_reason == "stop"
        and terminal_count == 1
        and terminal_before_usage
        and usage_last
    )
    _expect(passed, predicate, "partial semantic chat_stream.passed")
    child_results.append(predicate)

    responses = _exact_mapping(
        semantics["responses"],
        {"expected", "actual", "status", "stored_roundtrip", "passed"},
        "partial semantic responses",
    )
    _expect(
        responses["expected"],
        "hello",
        "partial semantic responses.expected",
    )
    if not isinstance(responses.get("actual"), str):
        raise PhaseError("partial semantic responses.actual must be a string")
    status = _nonempty_string(
        responses["status"],
        "partial semantic responses.status",
    )
    if status not in {
        "completed",
        "incomplete",
        "failed",
        "cancelled",
        "in_progress",
        "queued",
    }:
        raise PhaseError(
            f"partial semantic responses.status is unsupported: {status!r}"
        )
    stored_roundtrip = _strict_bool(
        responses["stored_roundtrip"],
        "partial semantic responses.stored_roundtrip",
    )
    passed = _strict_bool(
        responses["passed"],
        "partial semantic responses.passed",
    )
    predicate = (
        responses["actual"] == "hello"
        and status == "completed"
        and stored_roundtrip
    )
    _expect(passed, predicate, "partial semantic responses.passed")
    child_results.append(predicate)

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
        "partial semantic responses_stream",
    )
    _expect(
        responses_stream["expected"],
        "hello",
        "partial semantic responses_stream.expected",
    )
    if not isinstance(responses_stream.get("actual"), str):
        raise PhaseError("partial semantic responses_stream.actual must be a string")
    status = _nonempty_string(
        responses_stream["status"],
        "partial semantic responses_stream.status",
    )
    if status not in {
        "completed",
        "incomplete",
        "failed",
        "cancelled",
        "in_progress",
        "queued",
    }:
        raise PhaseError(
            "partial semantic responses_stream.status is unsupported: "
            f"{status!r}"
        )
    terminal_count = _strict_int(
        responses_stream["terminal_count"],
        "partial semantic responses_stream.terminal_count",
    )
    terminal_last = _strict_bool(
        responses_stream["terminal_last"],
        "partial semantic responses_stream.terminal_last",
    )
    passed = _strict_bool(
        responses_stream["passed"],
        "partial semantic responses_stream.passed",
    )
    predicate = (
        responses_stream["actual"] == "hello"
        and status == "completed"
        and terminal_count == 1
        and terminal_last
    )
    _expect(passed, predicate, "partial semantic responses_stream.passed")
    child_results.append(predicate)

    reasoning = _exact_mapping(
        semantics["reasoning"],
        {"accepted", "observed", "visible_think_tags", "passed"},
        "partial semantic reasoning",
    )
    for field in ("accepted", "observed", "visible_think_tags", "passed"):
        _strict_bool(reasoning.get(field), f"partial semantic reasoning.{field}")
    predicate = (
        reasoning["accepted"]
        and reasoning["observed"]
        and not reasoning["visible_think_tags"]
    )
    _expect(reasoning["passed"], predicate, "partial semantic reasoning.passed")
    child_results.append(predicate)

    aggregate = _strict_bool(
        semantics["passed"],
        "partial semantic_quality.passed",
    )
    _expect(
        aggregate,
        all(child_results),
        "partial semantic_quality.passed",
    )
    return partial


def _validate_failure_phase(phase: object) -> list[str]:
    value = _nonempty_string(phase, "failure phase")
    base_phases = {
        "inputs",
        "sdk",
        "initial_evidence",
        "protocol",
        "final_evidence",
        "report",
    }
    if value in base_phases:
        return [value]
    parts = value.split("+")
    allowed = {
        "compat_transport",
        "compat_semantic",
        "agent_protocol",
        "agent",
    }
    if any(part not in allowed for part in parts):
        raise PhaseError(f"failure phase grammar is invalid: {value!r}")
    if len(parts) != len(set(parts)):
        raise PhaseError("failure phase components must be unique")
    order = ["compat_transport", "compat_semantic", "agent_protocol", "agent"]
    if parts != sorted(parts, key=order.index):
        raise PhaseError("failure phase components are not in execution order")
    if len(set(parts) & {"compat_transport", "compat_semantic"}) > 1:
        raise PhaseError("failure phase has conflicting compatibility components")
    if len(set(parts) & {"agent_protocol", "agent"}) > 1:
        raise PhaseError("failure phase has conflicting agent components")
    return parts


def _validate_failure_provenance(
    value: object,
    *,
    phase_parts: list[str],
) -> dict[str, Any]:
    provenance = _exact_mapping(
        value,
        {"artifact", "sdk"},
        "failure provenance",
    )
    artifact = _exact_mapping(
        provenance["artifact"],
        {"path", "sha256", "size_bytes"},
        "failure provenance.artifact",
    )
    _nonempty_string(artifact["path"], "failure provenance.artifact.path")
    inputs_phase = phase_parts == ["inputs"]
    if inputs_phase:
        _expect(artifact["sha256"], None, "failure provenance.artifact.sha256")
        _expect(
            artifact["size_bytes"],
            None,
            "failure provenance.artifact.size_bytes",
        )
    else:
        digest = _nonempty_string(
            artifact["sha256"],
            "failure provenance.artifact.sha256",
        )
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise PhaseError("failure provenance artifact digest is not SHA-256")
        _positive_int(
            artifact["size_bytes"],
            "failure provenance.artifact.size_bytes",
        )
    if phase_parts in (["inputs"], ["sdk"]):
        _expect(provenance["sdk"], None, "failure provenance.sdk")
    else:
        sdk = _exact_mapping(
            provenance["sdk"],
            {"package", "version"},
            "failure provenance.sdk",
        )
        _expect(sdk["package"], "openai", "failure provenance.sdk.package")
        _expect(
            sdk["version"],
            OPENAI_SDK_VERSION,
            "failure provenance.sdk.version",
        )
    return provenance


def _validate_final_evidence(
    value: object,
    label: str,
    *,
    expected_profile: FlmProfileExpectations = LEGACY_FLM_PROFILE,
) -> dict[str, Any]:
    final = _exact_mapping(
        value,
        {
            "health",
            "capabilities",
            "metrics",
            "flm_evidence",
            "load_invariance",
            "collection_errors",
        },
        label,
    )
    errors = final["collection_errors"]
    if not isinstance(errors, list):
        raise PhaseError(f"{label}.collection_errors must be a list")
    for index, error in enumerate(errors):
        _nonempty_string(error, f"{label}.collection_errors[{index}]")
    if len(errors) != len(set(errors)):
        raise PhaseError(f"{label}.collection_errors must be unique")

    if final["health"] is not None:
        health = _validate_health_schema(
            final["health"],
            f"{label}.health",
            expected_profile=expected_profile,
        )
    else:
        health = None

    capabilities = final["capabilities"]
    if capabilities is not None:
        capabilities = _validate_capabilities_schema(
            capabilities,
            f"{label}.capabilities",
            expected_profile=expected_profile,
        )

    metrics = final["metrics"]
    if metrics is not None:
        metrics = _validate_metrics_schema(
            metrics,
            f"{label}.metrics",
            capabilities=capabilities,
            health=health,
            expected_profile=expected_profile,
        )

    if health is not None and capabilities is not None:
        cross_fields = {
            "ready": capabilities["ready"],
            "model": capabilities["model"],
            "max_context": capabilities["max_context"],
            "active_requests": capabilities["scheduler"]["active_requests"],
            "queued_requests": capabilities["scheduler"]["queued_requests"],
            "max_queued_requests": capabilities["scheduler"][
                "max_queued_requests"
            ],
            "prefix_cache_entries": capabilities["prefix_cache"]["entries"],
            "flm": capabilities["flm"],
        }
        for field, expected in cross_fields.items():
            _expect(health[field], expected, f"{label}.health.{field}")

    if capabilities is not None and metrics is not None:
        if final["flm_evidence"] is None:
            if not any(error.startswith("FLM evidence:") for error in errors):
                raise PhaseError(
                    f"{label}.flm_evidence may be null only with an FLM evidence error"
                )
            actual = None
        else:
            actual = _validate_flm_snapshot(
                final["flm_evidence"],
                f"{label}.flm_evidence",
                expected_profile=expected_profile,
            )
        if actual is None:
            capabilities = None
            metrics = None
        else:
            _expect(
                capabilities.get("model"),
                actual["model"],
                f"{label}.capabilities.model",
            )
    if capabilities is not None and metrics is not None:
        actual = _mapping(final["flm_evidence"], f"{label}.flm_evidence")
        flm = _mapping(capabilities.get("flm"), f"{label}.capabilities.flm")
        for field, snapshot_field in (
            ("source", "source"),
            ("load_sequence", "load_sequence"),
            ("native_int4_direct_weights", "native_int4"),
            ("bf16_fallback_weights", "bf16_fallback"),
            ("source_bytes", "source_bytes"),
            ("device_upload_bytes", "device_upload_bytes"),
            ("source_open_count", "source_open_count"),
            ("resident_allocation_count", "resident_allocation_count"),
            ("transfer_backend", "transfer_backend"),
        ):
            _expect(
                flm.get(field),
                actual[snapshot_field],
                f"{label}.capabilities.flm.{field}",
            )
        for metric_name, expected in (
            ("supersonic_model_loads_total", actual["model_loads_total"]),
            ("supersonic_active_requests", 0),
            ("supersonic_queued_requests", 0),
            (
                "supersonic_flm_native_int4_direct_weights",
                actual["native_int4"],
            ),
            ("supersonic_flm_bf16_fallback_weights", actual["bf16_fallback"]),
            ("supersonic_flm_source_bytes", actual["source_bytes"]),
            (
                "supersonic_flm_device_upload_bytes",
                actual["device_upload_bytes"],
            ),
        ):
            _expect(
                metrics.get(metric_name),
                expected,
                f"{label}.metrics.{metric_name}",
            )
    else:
        _expect(final["flm_evidence"], None, f"{label}.flm_evidence")

    load = _exact_mapping(
        final["load_invariance"],
        {"passed", "error"},
        f"{label}.load_invariance",
    )
    passed = _strict_bool(load["passed"], f"{label}.load_invariance.passed")
    if passed:
        _expect(load["error"], None, f"{label}.load_invariance.error")
    else:
        _nonempty_string(load["error"], f"{label}.load_invariance.error")
    return final


def validate_failure_report(
    report: dict[str, Any],
    *,
    expected_profile: FlmProfileExpectations = LEGACY_FLM_PROFILE,
) -> dict[str, Any]:
    failure = _exact_mapping(
        report,
        {
            "schema_version",
            "status",
            "phase",
            "error",
            "provenance",
            "completed",
            "final",
        },
        "failure report",
    )
    _strict_int(failure["schema_version"], "failure schema_version")
    _expect(failure["schema_version"], 1, "failure schema_version")
    _expect(failure["status"], "failed", "failure status")
    phase_parts = _validate_failure_phase(failure["phase"])
    error = _exact_mapping(
        failure["error"],
        {"type", "message"},
        "failure error",
    )
    _nonempty_string(error["type"], "failure error.type")
    _nonempty_string(error["message"], "failure error.message")
    _validate_failure_provenance(
        failure["provenance"],
        phase_parts=phase_parts,
    )

    completed = _mapping(failure["completed"], "failure completed")
    allowed_completed = {
        "initial_evidence",
        "compat",
        "agent",
        "phase_failures",
    }
    if not set(completed) <= allowed_completed:
        raise PhaseError(
            "failure completed has unsupported keys: "
            f"{sorted(set(completed) - allowed_completed)}"
        )
    pre_protocol = phase_parts[0] in {"inputs", "sdk", "initial_evidence"}
    if pre_protocol and completed:
        raise PhaseError("pre-protocol failure cannot contain completed evidence")
    if "initial_evidence" in completed:
        _validate_flm_snapshot(
            completed["initial_evidence"],
            "failure completed.initial_evidence",
            expected_profile=expected_profile,
        )
    if "compat" in completed:
        _validate_partial_compat_report(completed["compat"])
    if "agent" in completed:
        agent = _mapping(completed["agent"], "failure completed.agent")
        if "failure" in agent:
            validate_agent_failure_report(agent)
        else:
            validate_agent_report(agent)

    protocol_parts = [
        part
        for part in phase_parts
        if part
        in {
            "compat_transport",
            "compat_semantic",
            "agent_protocol",
            "agent",
        }
    ]
    if protocol_parts:
        if "initial_evidence" not in completed:
            raise PhaseError("protocol failure requires initial_evidence")
        phase_failures = completed.get("phase_failures")
        if not isinstance(phase_failures, list):
            raise PhaseError("protocol failure requires phase_failures list")
        observed = []
        for index, item in enumerate(phase_failures):
            entry = _exact_mapping(
                item,
                {"phase", "message"},
                f"failure completed.phase_failures[{index}]",
            )
            phase = _nonempty_string(
                entry["phase"],
                f"failure completed.phase_failures[{index}].phase",
            )
            _nonempty_string(
                entry["message"],
                f"failure completed.phase_failures[{index}].message",
            )
            observed.append(phase)
        if observed != protocol_parts:
            raise PhaseError(
                "failure phase_failures must exactly match phase grammar: "
                f"{observed!r} != {protocol_parts!r}"
            )
        if len(observed) != len(set(observed)):
            raise PhaseError("failure phase_failures must be unique")
        if "compat_semantic" in protocol_parts and "compat" not in completed:
            raise PhaseError("compat_semantic failure requires partial compat evidence")
        if "compat_transport" in protocol_parts and "compat" in completed:
            raise PhaseError("compat_transport failure cannot claim compat completion")
    elif "phase_failures" in completed:
        raise PhaseError("non-protocol failure cannot contain phase_failures")

    if phase_parts == ["protocol"] and set(completed) != {"initial_evidence"}:
        raise PhaseError("unclassified protocol failure requires only initial_evidence")
    if phase_parts in (["final_evidence"], ["report"]):
        if set(completed) != {"initial_evidence", "compat", "agent"}:
            raise PhaseError(
                f"{phase_parts[0]} failure requires all completed protocol evidence"
            )
    _validate_final_evidence(
        failure["final"],
        "failure final",
        expected_profile=expected_profile,
    )
    _reject_nonfinite_json(failure, "failure report")
    return report


def build_failure_report(
    *,
    phase: str,
    error: Exception,
    provenance: dict[str, Any],
    completed: dict[str, Any],
    final: dict[str, Any],
    expected_profile: FlmProfileExpectations = LEGACY_FLM_PROFILE,
) -> dict[str, Any]:
    report = {
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
    return validate_failure_report(report, expected_profile=expected_profile)


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
    expected_profile = getattr(args, "flm_profile", LEGACY_FLM_PROFILE)
    if not isinstance(expected_profile, FlmProfileExpectations):
        raise PhaseError("args.flm_profile must be FlmProfileExpectations")
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
                    expected_profile=expected_profile,
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
                final = collect_final_evidence(
                    base_url,
                    args.api_key,
                    before,
                    expected_profile=expected_profile,
                )

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
                expected_profile=expected_profile,
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
            expected_profile=expected_profile,
        )
        write_report(args.out_json, failure)
        if isinstance(exc, PhaseError):
            raise
        raise PhaseError(f"{phase} failed: {exc}") from exc


def _nonnegative_cli_int(value: str) -> int:
    if not value.isascii() or not value.isdecimal():
        raise argparse.ArgumentTypeError("must be a non-negative decimal integer")
    return int(value)


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
    parser.add_argument(
        "--expected-storage-abi-id",
        type=_nonnegative_cli_int,
        default=EXPECTED_STORAGE_ABI_IDS[0],
    )
    parser.add_argument(
        "--expected-row-group-int4",
        type=_nonnegative_cli_int,
        default=0,
    )
    parser.add_argument(
        "--expected-tile-int4-v1",
        type=_nonnegative_cli_int,
        default=EXPECTED_NATIVE_INT4,
    )
    parser.add_argument(
        "--expected-native-int4",
        type=_nonnegative_cli_int,
        default=EXPECTED_NATIVE_INT4,
    )
    parser.add_argument(
        "--expected-bf16-fallback",
        type=_nonnegative_cli_int,
        default=EXPECTED_BF16_FALLBACK,
    )
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
    try:
        args.flm_profile = FlmProfileExpectations(
            storage_abi_ids=(args.expected_storage_abi_id,),
            row_group_int4=args.expected_row_group_int4,
            tile_int4_v1=args.expected_tile_int4_v1,
            native_int4=args.expected_native_int4,
            bf16_fallback=args.expected_bf16_fallback,
        )
    except PhaseError as exc:
        parser.error(str(exc))
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
