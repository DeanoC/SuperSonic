from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Callable, Mapping, Sequence

from . import adapters
from .execution import ProcessResult, _run_process_with_telemetry


_PERSISTENT_RE = re.compile(
    r"^\[stage-timings\]\s+steps=(?P<steps>[0-9]+)\b.*?\bpersistent_ms=(?P<persistent>[0-9.]+)\b",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class SoakConfig:
    argv: tuple[str, ...]
    output: Path
    physical_gpu: str
    slow_persistent_ms_per_token: float
    max_runs: int
    trace_attempts: int
    timeout_seconds: float
    max_duration_seconds: float
    logical_gpu: int
    hip_visible_devices: str
    rocprof_binary: str = "rocprofv3"
    cwd: Path | None = None
    environment: Mapping[str, str] | None = None


SampleRunner = Callable[..., tuple[ProcessResult, Sequence[object], Sequence[str]]]


def run_soak(
    config: SoakConfig,
    *,
    sample_runner: SampleRunner = _run_process_with_telemetry,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    _validate_config(config)
    output = Path(config.output)
    if output.exists():
        raise ValueError(f"repeatability output already exists: {output}")
    logs = output / "logs"
    logs.mkdir(parents=True)

    result: dict[str, object] = {
        "schema_version": 1,
        "state": "running",
        "command": list(config.argv),
        "physical_gpu": config.physical_gpu,
        "slow_persistent_ms_per_token": config.slow_persistent_ms_per_token,
        "max_runs": config.max_runs,
        "max_duration_seconds": config.max_duration_seconds,
        "trace_attempts_requested": config.trace_attempts,
        "cache_state": "cold-load",
        "process_state": "fresh-process",
        "process_reuse": False,
        "device_mapping": {
            "physical_gpu": config.physical_gpu,
            "logical_gpu": config.logical_gpu,
            "HIP_VISIBLE_DEVICES": config.hip_visible_devices,
        },
        "samples": [],
        "trigger_run": None,
        "followup_traces": [],
    }
    expected_tokens: tuple[int, ...] | None = None
    deadline = monotonic() + config.max_duration_seconds
    _write_manifest(output, result)

    for run_index in range(1, config.max_runs + 1):
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            result["state"] = "duration-complete"
            _write_manifest(output, result)
            return result
        invocation_timeout = min(config.timeout_seconds, remaining)
        try:
            process, telemetry, notes = _invoke(
                config, config.argv, sample_runner, timeout=invocation_timeout
            )
        except KeyboardInterrupt:
            result.update(state="interrupted", error="interrupted")
            _write_manifest(output, result)
            return result
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            result.update(state="failed", error="runner-error", error_detail=str(exc))
            _write_manifest(output, result)
            return result
        _write_streams(logs, f"run-{run_index}", process)
        if process.returncode != 0 or process.timed_out or process.interrupted:
            if process.interrupted:
                result.update(state="interrupted", error="interrupted")
            elif process.timed_out and invocation_timeout < config.timeout_seconds:
                result.update(state="duration-complete", termination="aggregate-deadline")
            else:
                result.update(state="failed", error="process-failed")
            _write_manifest(output, result)
            return result
        try:
            parsed = adapters.parse_output("supersonic", process.stdout, process.stderr)
            timing = parse_persistent_timing(process.stdout, process.stderr)
        except (TypeError, ValueError) as exc:
            result.update(state="failed", error="invalid-output", error_detail=str(exc))
            _write_manifest(output, result)
            return result
        if expected_tokens is None:
            expected_tokens = parsed.token_ids
        elif parsed.token_ids != expected_tokens:
            result.update(state="failed", error="token-mismatch")
            _write_manifest(output, result)
            return result
        sample = {
            "run": run_index,
            **timing,
            "tokens_sha256": _tokens_digest(parsed.token_ids),
            "telemetry_samples": [_json_value(item) for item in telemetry],
            "telemetry_notes": list(notes),
        }
        result["samples"].append(sample)
        _write_manifest(output, result)
        if timing["persistent_ms_per_token"] >= config.slow_persistent_ms_per_token:
            result["trigger_run"] = run_index
            result["state"] = "slow-triggered"
            _write_manifest(output, result)
            valid_traces, interrupted = _capture_followups(
                config, result, sample_runner, deadline=deadline, monotonic=monotonic
            )
            if interrupted:
                result.update(state="interrupted", error="interrupted")
            elif valid_traces:
                result["state"] = "slow-captured"
            else:
                result.update(state="trace-failed", error="no-valid-followup-trace")
            _write_manifest(output, result)
            return result

    result["state"] = "no-slow-sample"
    _write_manifest(output, result)
    return result


def parse_persistent_timing(stdout: str, stderr: str = "") -> dict[str, float | int]:
    combined = f"{stdout.rstrip()}\n{stderr.lstrip()}" if stderr else stdout
    matches = list(_PERSISTENT_RE.finditer(combined))
    if len(matches) != 1:
        raise ValueError("SuperSonic output must contain exactly one persistent stage timing")
    steps = int(matches[0].group("steps"))
    persistent_ms = float(matches[0].group("persistent"))
    if steps <= 0 or not math.isfinite(persistent_ms) or persistent_ms <= 0.0:
        raise ValueError("persistent stage timing must contain positive finite values")
    return {
        "steps": steps,
        "persistent_ms": persistent_ms,
        "persistent_ms_per_token": persistent_ms / steps,
    }


def _capture_followups(
    config: SoakConfig,
    result: dict[str, object],
    runner: SampleRunner,
    *,
    deadline: float,
    monotonic: Callable[[], float],
) -> tuple[int, bool]:
    traces = result["followup_traces"]
    assert isinstance(traces, list)
    valid_traces = 0
    for attempt in range(1, config.trace_attempts + 1):
        remaining = deadline - monotonic()
        if remaining <= 0.0:
            break
        trace_dir = Path(config.output) / "traces" / f"attempt-{attempt}"
        trace_dir.mkdir(parents=True)
        argv = (
            config.rocprof_binary,
            "--kernel-trace",
            "--memory-allocation-trace",
            "--output-format",
            "json",
            "--output-directory",
            str(trace_dir),
            "--",
            *config.argv,
        )
        try:
            process, telemetry, notes = _invoke(
                config, argv, runner, timeout=min(config.timeout_seconds, remaining)
            )
        except KeyboardInterrupt:
            return valid_traces, True
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            traces.append(
                {
                    "attempt": attempt,
                    "relationship": "followup-reproduction",
                    "trace_directory": str(trace_dir.relative_to(config.output)),
                    "error": "runner-error",
                    "error_detail": str(exc),
                }
            )
            _write_manifest(Path(config.output), result)
            continue
        _write_streams(Path(config.output) / "logs", f"trace-{attempt}", process)
        trace: dict[str, object] = {
            "attempt": attempt,
            "relationship": "followup-reproduction",
            "returncode": process.returncode,
            "timed_out": process.timed_out,
            "interrupted": process.interrupted,
            "trace_directory": str(trace_dir.relative_to(config.output)),
            "telemetry_samples": [_json_value(item) for item in telemetry],
            "telemetry_notes": list(notes),
        }
        if process.returncode == 0 and not process.timed_out and not process.interrupted:
            try:
                trace.update(parse_persistent_timing(process.stdout, process.stderr))
                trace_files = _valid_json_traces(trace_dir)
                if trace_files:
                    trace["trace_files"] = trace_files
                    valid_traces += 1
                else:
                    trace["error"] = "missing-valid-json-trace"
            except (OSError, TypeError, ValueError) as exc:
                trace["error"] = "invalid-trace-output"
                trace["error_detail"] = str(exc)
        else:
            trace["error"] = "profiler-process-failed"
        traces.append(trace)
        _write_manifest(Path(config.output), result)
        if process.interrupted:
            return valid_traces, True
    return valid_traces, False


def _invoke(config: SoakConfig, argv: Sequence[str], runner: SampleRunner, *, timeout: float):
    return runner(
        tuple(argv),
        timeout=timeout,
        physical_gpu=config.physical_gpu,
        cwd=config.cwd,
        env=_effective_environment(config),
    )


def _valid_json_traces(trace_dir: Path) -> list[str]:
    valid: list[str] = []
    for candidate in sorted(trace_dir.rglob("*.json")):
        if not candidate.is_file() or candidate.stat().st_size == 0:
            continue
        with candidate.open(encoding="utf-8") as source:
            json.load(source)
        valid.append(str(candidate.relative_to(trace_dir)))
    return valid


def _write_streams(logs: Path, label: str, process: ProcessResult) -> None:
    _write_durable_text(logs / f"{label}.stdout.log", process.stdout)
    _write_durable_text(logs / f"{label}.stderr.log", process.stderr)
    _fsync_directory(logs)


def _write_manifest(output: Path, result: Mapping[str, object]) -> None:
    target = output / "manifest.json"
    temporary = output / ".manifest.json.tmp"
    _write_durable_text(temporary, json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    _fsync_directory(output)


def _write_durable_text(path: Path, value: str) -> None:
    with path.open("w", encoding="utf-8") as target:
        target.write(value)
        target.flush()
        os.fsync(target.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _effective_environment(config: SoakConfig) -> dict[str, str]:
    effective = dict(config.environment) if config.environment is not None else dict(os.environ)
    effective["HIP_VISIBLE_DEVICES"] = config.hip_visible_devices
    return effective


def _tokens_digest(tokens: tuple[int, ...] | None) -> str:
    if tokens is None:
        raise ValueError("repeatability soak requires exact SuperSonic token ids")
    encoded = " ".join(str(token) for token in tokens).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _json_value(value: object) -> object:
    try:
        return asdict(value)  # type: ignore[arg-type]
    except TypeError:
        return value


def _validate_config(config: SoakConfig) -> None:
    if not config.argv:
        raise ValueError("repeatability command must not be empty")
    if not str(config.physical_gpu).isdigit():
        raise ValueError("physical_gpu must be a numeric GPU ordinal")
    if config.max_runs < 1:
        raise ValueError("max_runs must be positive")
    if config.max_runs > 2160:
        raise ValueError("max_runs must not exceed 2160")
    if config.trace_attempts < 1:
        raise ValueError("trace_attempts must be positive")
    if config.trace_attempts > 3:
        raise ValueError("trace_attempts must not exceed 3")
    if not math.isfinite(config.slow_persistent_ms_per_token) or config.slow_persistent_ms_per_token <= 0:
        raise ValueError("slow persistent threshold must be positive")
    if not math.isfinite(config.timeout_seconds) or config.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if not math.isfinite(config.max_duration_seconds) or config.max_duration_seconds <= 0:
        raise ValueError("max_duration_seconds must be positive")
    if config.max_duration_seconds > 21600:
        raise ValueError("max_duration_seconds must not exceed 21600")
    visible = tuple(item.strip() for item in config.hip_visible_devices.split(",") if item.strip())
    if config.logical_gpu < 0 or config.logical_gpu >= len(visible):
        raise ValueError("logical_gpu must select an entry in HIP_VISIBLE_DEVICES")
    if visible[config.logical_gpu] != config.physical_gpu:
        raise ValueError("HIP_VISIBLE_DEVICES does not map logical_gpu to physical_gpu")
    if config.environment is not None and config.environment.get("HIP_VISIBLE_DEVICES") != config.hip_visible_devices:
        raise ValueError("effective child HIP_VISIBLE_DEVICES does not match recorded mapping")
