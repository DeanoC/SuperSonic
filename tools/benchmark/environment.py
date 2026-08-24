from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Callable, Mapping


ALLOWLISTED_ENVIRONMENT = (
    "HIP_ARCH",
    "HIP_VISIBLE_DEVICES",
    "ROCM_PATH",
    "HIP_PATH",
    "SUPERSONIC_DEVICE",
    "RUSTFLAGS",
)

_CACHE_STATES = frozenset(
    (
        "cold-load",
        "warm-resident",
        "prefix-cache-empty",
        "prefix-cache-populated",
        "prefix-cache-reset",
    )
)
_PREFIX_CACHE_STATE = {
    "prefix-cache-empty": "empty",
    "prefix-cache-populated": "populated",
    "prefix-cache-reset": "reset",
}
_ALLOWED_FLUSH_VALUES = frozenset(("verified", "unavailable", "not-requested"))
_SHOWALLINFO_COMMAND_PREFIX = (
    "timeout",
    "--foreground",
    "30s",
    "rocm-smi",
)
_DEFAULT_CPU_GOVERNOR_PATH = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")

_GPU_CLOCK_RE = re.compile(
    r"(?:GPU|sclk)\s+Clock\s+Level\s*:\s*(?:[A-Za-z0-9_-]+\s*:\s*)?\(?([0-9]+)\s*MHz",
    re.IGNORECASE,
)
_MEMORY_CLOCK_RE = re.compile(
    r"(?:Memory|mclk)\s+Clock\s+Level\s*:\s*(?:[A-Za-z0-9_-]+\s*:\s*)?\(?([0-9]+)\s*MHz",
    re.IGNORECASE,
)
_POWER_RE = re.compile(
    r"Average\s+Graphics\s+Package\s+Power\s+\(W\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_POWER_CAP_RE = re.compile(
    r"(?:Max|Power\s+Cap)\s+Graphics\s+Package\s+Power\s+\(W\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_TEMPERATURE_RE = re.compile(
    r"Temperature\s+\(Sensor\s+edge\)\s+\(C\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_GPU_USE_RE = re.compile(r"GPU\s+use\s*\(%\)\s*:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_VRAM_USE_RE = re.compile(
    r"GPU\s+Memory\s+Allocated\s+\(VRAM%\)\s*:\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_PERFORMANCE_LEVEL_RE = re.compile(r"Performance\s+Level\s*:\s*([A-Za-z0-9_-]+)", re.IGNORECASE)
_LOADED_GPU_UTILIZATION_PERCENT = 90.0


@dataclass(frozen=True, slots=True)
class RequestedTelemetry:
    gpu_clock_mhz: int | None
    clock_tolerance_mhz: int | None
    memory_clock_mhz: int | None
    power_cap_watts: int | None
    performance_level: str | None


@dataclass(frozen=True, slots=True)
class ObservedTelemetry:
    gpu_clock_mhz: int | None
    memory_clock_mhz: int | None
    power_cap_watts: int | None
    power_watts: float | None
    temperature_celsius: float | None
    gpu_utilization_percent: float | None
    memory_utilization_percent: float | None
    performance_level: str | None


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    offset_seconds: float
    gpu_clock_mhz: int | None
    memory_clock_mhz: int | None
    power_cap_watts: int | None
    power_watts: float | None
    temperature_celsius: float | None
    gpu_utilization_percent: float | None
    memory_utilization_percent: float | None
    performance_level: str | None


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    clock_policy: str
    requested: RequestedTelemetry
    requested_at: str
    observed_before: ObservedTelemetry
    observed_before_at: str
    observed_after: ObservedTelemetry
    observed_after_at: str
    telemetry_samples: tuple[TelemetrySample, ...]
    headline_eligible: bool
    physical_gpu: str
    logical_gpu: str
    cpu_governor: str | None
    allowlisted_environment: dict[str, str]
    cache_state: str
    cache_evidence: dict[str, object]
    process_reuse: bool
    verification_errors: tuple[str, ...]
    evidence_notes: tuple[str, ...]


def collect_snapshot(
    *,
    physical_gpu: str,
    clock_policy,
    cache_state: str,
    command_runner: Callable[[tuple[str, ...]], str],
    cache_evidence: Mapping[str, object] | None = None,
    environment_map: Mapping[str, str] | None = None,
    cpu_governor_reader: Callable[[], str] | None = None,
    sample_count: int = 0,
    wall_clock: Callable[[], datetime | str] | None = None,
    monotonic_clock: Callable[[], float] | None = None,
) -> EnvironmentSnapshot:
    if sample_count < 0:
        raise ValueError("sample_count must be non-negative")
    resolved_cache_evidence = dict(cache_evidence or _default_cache_evidence(cache_state))
    validate_cache_evidence(cache_state, resolved_cache_evidence)

    notes: list[str] = []
    env = dict(environment_map or {})
    command = _build_probe_command(physical_gpu)
    timestamp = wall_clock or _utc_now
    steady = monotonic_clock or (lambda: 0.0)

    requested_at = _timestamp(timestamp())
    before_started = steady()
    before = _parse_showallinfo(command_runner(command), notes)
    sample_values: list[TelemetrySample] = []
    for _ in range(sample_count):
        offset = steady() - before_started
        observed = _parse_showallinfo(command_runner(command), notes)
        sample_values.append(
            TelemetrySample(
                offset_seconds=offset,
                gpu_clock_mhz=observed.gpu_clock_mhz,
                memory_clock_mhz=observed.memory_clock_mhz,
                power_cap_watts=observed.power_cap_watts,
                power_watts=observed.power_watts,
                temperature_celsius=observed.temperature_celsius,
                gpu_utilization_percent=observed.gpu_utilization_percent,
                memory_utilization_percent=observed.memory_utilization_percent,
                performance_level=observed.performance_level,
            )
        )
    after = _parse_showallinfo(command_runner(command), notes)
    after_at = _timestamp(timestamp())
    requested = RequestedTelemetry(
        gpu_clock_mhz=_policy_value(clock_policy, "gpu_clock_mhz"),
        clock_tolerance_mhz=_policy_value(clock_policy, "clock_tolerance_mhz"),
        memory_clock_mhz=_policy_value(clock_policy, "memory_clock_mhz"),
        power_cap_watts=_policy_value(clock_policy, "power_cap_watts"),
        performance_level=_policy_value(clock_policy, "performance_level"),
    )
    cpu_governor = _read_cpu_governor(cpu_governor_reader, notes)
    verification_errors = verify_clock_policy(
        before,
        [
            ObservedTelemetry(
                gpu_clock_mhz=sample.gpu_clock_mhz,
                memory_clock_mhz=sample.memory_clock_mhz,
                power_cap_watts=sample.power_cap_watts,
                power_watts=sample.power_watts,
                temperature_celsius=sample.temperature_celsius,
                gpu_utilization_percent=sample.gpu_utilization_percent,
                memory_utilization_percent=sample.memory_utilization_percent,
                performance_level=sample.performance_level,
            )
            for sample in sample_values
        ],
        after,
        clock_policy,
    )
    for error in verification_errors:
        if error not in notes:
            notes.append(error)
    return EnvironmentSnapshot(
        clock_policy=str(_policy_value(clock_policy, "name") or ""),
        requested=requested,
        requested_at=requested_at,
        observed_before=before,
        observed_before_at=requested_at,
        observed_after=after,
        observed_after_at=after_at,
        telemetry_samples=tuple(sample_values),
        headline_eligible=_headline_eligible(clock_policy, verification_errors),
        physical_gpu=str(physical_gpu),
        logical_gpu=str(env.get("SUPERSONIC_DEVICE", "0")),
        cpu_governor=cpu_governor,
        allowlisted_environment={key: env[key] for key in ALLOWLISTED_ENVIRONMENT if key in env},
        cache_state=cache_state,
        cache_evidence=resolved_cache_evidence,
        process_reuse=bool(resolved_cache_evidence.get("process_reuse", False)),
        verification_errors=verification_errors,
        evidence_notes=tuple(notes),
    )


def snapshot_from_observations(
    *,
    physical_gpu: str,
    logical_gpu: str,
    clock_policy,
    cache_state: str,
    observed_before: ObservedTelemetry,
    observed_before_at: str,
    telemetry_samples: tuple[TelemetrySample, ...],
    observed_after: ObservedTelemetry,
    observed_after_at: str,
    environment_map: Mapping[str, str] | None = None,
    cache_evidence: Mapping[str, object] | None = None,
    cpu_governor_reader: Callable[[], str] | None = None,
    evidence_notes: tuple[str, ...] = (),
) -> EnvironmentSnapshot:
    resolved_cache_evidence = dict(cache_evidence or _default_cache_evidence(cache_state))
    validate_cache_evidence(cache_state, resolved_cache_evidence)
    requested = RequestedTelemetry(
        gpu_clock_mhz=_policy_value(clock_policy, "gpu_clock_mhz"),
        clock_tolerance_mhz=_policy_value(clock_policy, "clock_tolerance_mhz"),
        memory_clock_mhz=_policy_value(clock_policy, "memory_clock_mhz"),
        power_cap_watts=_policy_value(clock_policy, "power_cap_watts"),
        performance_level=_policy_value(clock_policy, "performance_level"),
    )
    observed = tuple(
        ObservedTelemetry(
            gpu_clock_mhz=sample.gpu_clock_mhz,
            memory_clock_mhz=sample.memory_clock_mhz,
            power_cap_watts=sample.power_cap_watts,
            power_watts=sample.power_watts,
            temperature_celsius=sample.temperature_celsius,
            gpu_utilization_percent=sample.gpu_utilization_percent,
            memory_utilization_percent=sample.memory_utilization_percent,
            performance_level=sample.performance_level,
        )
        for sample in telemetry_samples
    )
    verification_errors = verify_clock_policy(observed_before, list(observed), observed_after, clock_policy)
    notes = list(evidence_notes)
    for error in verification_errors:
        if error not in notes:
            notes.append(error)
    env = dict(environment_map or {})
    return EnvironmentSnapshot(
        clock_policy=str(_policy_value(clock_policy, "name") or ""),
        requested=requested,
        requested_at=observed_before_at,
        observed_before=observed_before,
        observed_before_at=observed_before_at,
        observed_after=observed_after,
        observed_after_at=observed_after_at,
        telemetry_samples=telemetry_samples,
        headline_eligible=_headline_eligible(clock_policy, verification_errors),
        physical_gpu=str(physical_gpu),
        logical_gpu=str(logical_gpu),
        cpu_governor=_read_cpu_governor(cpu_governor_reader, notes),
        allowlisted_environment={key: env[key] for key in ALLOWLISTED_ENVIRONMENT if key in env},
        cache_state=cache_state,
        cache_evidence=resolved_cache_evidence,
        process_reuse=bool(resolved_cache_evidence.get("process_reuse", False)),
        verification_errors=verification_errors,
        evidence_notes=tuple(notes),
    )


def verify_clock_policy(before, observed, after, policy) -> tuple[str, ...]:
    if _policy_value(policy, "name") == "uncontrolled-clocks":
        return ()
    return tuple(_clock_violations((before, *observed, after), policy))


def validate_cache_evidence(cache_state: str, evidence: Mapping[str, object]) -> None:
    if cache_state not in _CACHE_STATES:
        raise ValueError(f"unknown cache_state: {cache_state!r}")
    unknown = sorted(set(evidence) - {"process_state", "process_reuse", "filesystem_flush", "prefix_cache"})
    if unknown:
        raise ValueError(f"unknown cache evidence keys: {', '.join(unknown)}")
    process_reuse = evidence.get("process_reuse", False)
    if not isinstance(process_reuse, bool):
        raise ValueError("process_reuse must be a boolean")
    if process_reuse:
        raise ValueError("process_reuse must remain false for one-shot SuperSonic evidence")
    flush = evidence.get("filesystem_flush")
    if flush is not None:
        if flush == "claimed":
            raise ValueError("filesystem flush claims must be verified")
        if flush not in _ALLOWED_FLUSH_VALUES:
            raise ValueError(f"unknown filesystem_flush value: {flush!r}")
    if cache_state in ("cold-load", "warm-resident"):
        if evidence.get("process_state", "fresh-process") != "fresh-process":
            raise ValueError(f"{cache_state} evidence must use fresh-process wording")
        if "prefix_cache" in evidence:
            raise ValueError(f"{cache_state} evidence cannot declare prefix_cache")
        return
    expected = _PREFIX_CACHE_STATE[cache_state]
    if evidence.get("prefix_cache") != expected:
        raise ValueError(f"{cache_state} evidence requires prefix_cache={expected!r}")
    if "process_state" in evidence:
        raise ValueError(f"{cache_state} evidence cannot declare process_state")


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _timestamp(value: datetime | str) -> str:
    if isinstance(value, str):
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _policy_value(policy, field: str):
    if isinstance(policy, Mapping):
        return policy.get(field)
    return getattr(policy, field, None)


def _build_probe_command(physical_gpu: str) -> tuple[str, ...]:
    if not str(physical_gpu).isdigit():
        raise ValueError(f"physical GPU must be a numeric ordinal, got {physical_gpu!r}")
    return (*_SHOWALLINFO_COMMAND_PREFIX, "-d", str(physical_gpu), "--showallinfo")


def build_probe_command(physical_gpu: str) -> tuple[str, ...]:
    return _build_probe_command(physical_gpu)


def _default_cache_evidence(cache_state: str) -> dict[str, object]:
    if cache_state not in _CACHE_STATES:
        raise ValueError(f"unknown cache_state: {cache_state!r}")
    if cache_state in ("cold-load", "warm-resident"):
        return {"process_state": "fresh-process", "process_reuse": False}
    return {"prefix_cache": _PREFIX_CACHE_STATE[cache_state], "process_reuse": False}


def _headline_eligible(policy, verification_errors: tuple[str, ...]) -> bool:
    if _policy_value(policy, "name") != "locked":
        return False
    return not verification_errors


def _clock_violations(samples: tuple[ObservedTelemetry, ...], policy) -> list[str]:
    errors: list[str] = []
    requested_gpu = _policy_value(policy, "gpu_clock_mhz")
    requested_memory = _policy_value(policy, "memory_clock_mhz")
    requested_power_cap = _policy_value(policy, "power_cap_watts")
    requested_level = _policy_value(policy, "performance_level")
    requested_temperature_limit = _policy_value(policy, "temperature_limit_celsius")
    gpu_tolerance = int(_policy_value(policy, "clock_tolerance_mhz") or 0)
    memory_tolerance = int(_policy_value(policy, "memory_clock_tolerance_mhz") or gpu_tolerance)
    loaded_gpu_samples = 0
    for index, sample in enumerate(samples):
        gpu_is_loaded = (
            sample.gpu_utilization_percent is not None
            and sample.gpu_utilization_percent >= _LOADED_GPU_UTILIZATION_PERCENT
        )
        if requested_gpu is not None and gpu_is_loaded:
            loaded_gpu_samples += 1
            if sample.gpu_clock_mhz is None:
                errors.append(f"missing GPU clock verification at sample {index}")
            elif abs(sample.gpu_clock_mhz - requested_gpu) > gpu_tolerance:
                errors.append(
                    f"clock drift at sample {index}: observed {sample.gpu_clock_mhz} MHz, "
                    f"expected {requested_gpu}±{gpu_tolerance}"
                )
        if requested_memory is not None:
            if sample.memory_clock_mhz is None:
                errors.append(f"missing memory clock verification at sample {index}")
            elif abs(sample.memory_clock_mhz - requested_memory) > memory_tolerance:
                errors.append(
                    f"memory clock drift at sample {index}: observed {sample.memory_clock_mhz} MHz, "
                    f"expected {requested_memory}±{memory_tolerance}"
                )
        if requested_power_cap is not None:
            if sample.power_cap_watts is None:
                errors.append(f"missing power cap verification at sample {index}")
            elif sample.power_cap_watts != requested_power_cap:
                errors.append(
                    f"power cap mismatch at sample {index}: observed {sample.power_cap_watts} W, "
                    f"expected {requested_power_cap} W"
                )
        if requested_level is not None:
            if sample.performance_level is None:
                errors.append(f"missing performance level verification at sample {index}")
            elif sample.performance_level != requested_level:
                errors.append(
                    f"performance level mismatch at sample {index}: observed "
                    f"{sample.performance_level!r}, expected {requested_level!r}"
                )
        if requested_temperature_limit is not None and sample.temperature_celsius is not None:
            if sample.temperature_celsius > requested_temperature_limit:
                errors.append(
                    f"temperature exceeded limit at sample {index}: observed "
                    f"{sample.temperature_celsius}, limit {requested_temperature_limit}"
                )
    if requested_gpu is not None and loaded_gpu_samples == 0:
        errors.append("missing loaded GPU clock verification sample")
    return errors


def _read_cpu_governor(
    cpu_governor_reader: Callable[[], str] | None,
    notes: list[str],
) -> str | None:
    reader = cpu_governor_reader or (lambda: _DEFAULT_CPU_GOVERNOR_PATH.read_text(encoding="utf-8"))
    try:
        value = reader().strip()
    except OSError as exc:
        notes.append(f"CPU governor unavailable: {exc}")
        return None
    if not value:
        notes.append("CPU governor unavailable: empty reading")
        return None
    return value


def _parse_showallinfo(output: str, notes: list[str]) -> ObservedTelemetry:
    return ObservedTelemetry(
        gpu_clock_mhz=_extract_int(_GPU_CLOCK_RE, output, "GPU clock", notes),
        memory_clock_mhz=_extract_int(_MEMORY_CLOCK_RE, output, "memory clock", notes),
        power_cap_watts=_extract_intish(_POWER_CAP_RE, output, "power cap", notes),
        power_watts=_extract_float(_POWER_RE, output, "power draw", notes),
        temperature_celsius=_extract_float(_TEMPERATURE_RE, output, "temperature", notes),
        gpu_utilization_percent=_extract_float(_GPU_USE_RE, output, "GPU use (%)", notes),
        memory_utilization_percent=_extract_float(
            _VRAM_USE_RE,
            output,
            "GPU Memory Allocated (VRAM%)",
            notes,
        ),
        performance_level=_extract_str(_PERFORMANCE_LEVEL_RE, output, "performance level", notes),
    )


def parse_showallinfo(output: str, notes: list[str] | None = None) -> ObservedTelemetry:
    """Parse one bounded ROCm SMI telemetry probe for live monitoring."""

    return _parse_showallinfo(output, notes if notes is not None else [])


def _extract_int(
    pattern: re.Pattern[str],
    output: str,
    label: str,
    notes: list[str],
) -> int | None:
    value = _extract_str(pattern, output, label, notes)
    return int(value) if value is not None else None


def _extract_float(
    pattern: re.Pattern[str],
    output: str,
    label: str,
    notes: list[str],
) -> float | None:
    value = _extract_str(pattern, output, label, notes)
    return float(value) if value is not None else None


def _extract_intish(
    pattern: re.Pattern[str],
    output: str,
    label: str,
    notes: list[str],
) -> int | None:
    value = _extract_str(pattern, output, label, notes)
    return int(float(value)) if value is not None else None


def _extract_str(
    pattern: re.Pattern[str],
    output: str,
    label: str,
    notes: list[str],
) -> str | None:
    matches = pattern.findall(output)
    if not matches:
        notes.append(f"{label} unsupported by probe")
        return None
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {label} value, found {len(matches)}")
    return matches[0]
