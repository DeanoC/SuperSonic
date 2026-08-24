from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from . import validation


COMPARABILITY_FIELDS = (
    "environment.rocm_version",
    "environment.hip_version",
    "hardware.identity",
    "hardware.identity_kind",
    "hardware.architecture",
    "hardware.physical_gpu",
    "hardware.logical_gpu",
    "artifact.semantic_id",
    "artifact.quantization",
    "artifact.sha256",
    "artifact.tokenizer_sha256",
    "artifact.chat_template_sha256",
    "workload.case_id",
    "workload.prompt_sha256",
    "workload.context_limit",
    "workload.max_new_tokens",
    "workload.mode",
    "workload.stop_policy",
    "workload.cache_state",
    "workload.warmups",
    "workload.measurement_boundary",
    "environment.clock_policy",
    "environment.requested.gpu_clock_mhz",
    "environment.requested.memory_clock_mhz",
    "environment.requested.power_cap_watts",
    "environment.requested.performance_level",
    "environment.process_reuse",
)

# Engine identity distinguishes the two sides of a peer comparison and keeps
# independent engine series apart, but it is not a shared-input requirement:
# the comparator is specifically used to compare different engines.
SERIES_IDENTITY_FIELDS = ("engine.name", "engine.version", *COMPARABILITY_FIELDS)


@dataclass(frozen=True, slots=True)
class SampleSummary:
    values: tuple[float, ...]
    count: int
    minimum: float
    median: float
    maximum: float
    mad: float


@dataclass(frozen=True, slots=True)
class Comparison:
    left: SampleSummary
    right: SampleSummary
    comparable: bool
    reasons: tuple[str, ...]
    speedup: float | None


def summarize_samples(values: list[float] | tuple[float, ...]) -> SampleSummary:
    raw = tuple(_number(value, "sample") for value in values)
    if not raw:
        raise ValueError("sample summary requires at least one sample")
    median = _median(raw)
    deviations = tuple(abs(value - median) for value in raw)
    return SampleSummary(
        values=raw,
        count=len(raw),
        minimum=min(raw),
        median=median,
        maximum=max(raw),
        mad=_median(deviations),
    )


def series_key(record: dict[str, object]) -> tuple[str, ...]:
    return tuple(f"{field}={_field_value(record, field)}" for field in SERIES_IDENTITY_FIELDS)


def compare_records(left: dict[str, object], right: dict[str, object]) -> Comparison:
    reasons: list[str] = []
    for field in COMPARABILITY_FIELDS:
        if _field_value(left, field) != _field_value(right, field):
            _append_reason(reasons, _reason_name(field))

    for label, record in (("left", left), ("right", right)):
        try:
            validation.validate_record(record)
        except ValueError:
            _append_reason(reasons, f"{label}_invalid_record")

    if not _headline_eligible(left) or not _headline_eligible(right):
        _append_reason(reasons, "headline_eligible")

    left_summary = summarize_samples([sample["decode_ms"] for sample in left["samples"]])
    right_summary = summarize_samples([sample["decode_ms"] for sample in right["samples"]])

    comparable = not reasons
    speedup = None
    if comparable:
        if right_summary.median == 0:
            comparable = False
            _append_reason(reasons, "zero_median")
        else:
            speedup = left_summary.median / right_summary.median

    return Comparison(
        left=left_summary,
        right=right_summary,
        comparable=comparable,
        reasons=tuple(reasons),
        speedup=speedup,
    )


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} must be finite")
    return float(value)


def _median(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def _field_value(record: dict[str, object], field: str) -> object:
    value: Any = record
    for part in field.split("."):
        value = value[part]
    return value


def _reason_name(field: str) -> str:
    if field == "environment.clock_policy":
        return "clock_policy"
    if field == "workload.cache_state":
        return "cache_state"
    if field == "environment.requested.power_cap_watts":
        return "power_cap_watts"
    if field == "artifact.sha256":
        return "sha256"
    return field.rsplit(".", 1)[-1]


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _headline_eligible(record: dict[str, object]) -> bool:
    return validation.has_verified_headline(record)
