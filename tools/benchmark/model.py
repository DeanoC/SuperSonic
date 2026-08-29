from __future__ import annotations

from dataclasses import dataclass
import json
import math


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def parse_strict_json(text: str, *, context: str) -> object:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{context} must not contain duplicate keys: {key!r}")
            value[key] = item
        return value

    def reject_constant(token: str) -> object:
        raise ValueError(f"{context} must not contain non-finite numbers: {token}")

    try:
        value = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context} must be valid JSON") from exc

    _require_finite_json(value, context=context)
    return value


def _require_finite_json(value: object, *, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} must not contain non-finite numbers")
    if isinstance(value, list):
        for item in value:
            _require_finite_json(item, context=context)
    elif isinstance(value, dict):
        for item in value.values():
            _require_finite_json(item, context=context)


@dataclass(frozen=True, slots=True)
class PerformanceCase:
    id: str
    prompt: str
    max_new_tokens: int
    warmups: int
    repetitions: int
    mode: str
    cache_state: str
    timeout_seconds: int
    decoding_policy: str
    stop_policy: str
    engines: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SuiteManifest:
    version: int
    name: str
    budget_seconds: int
    minimum_duration_seconds: int
    quality_version: str
    quality_case_ids: tuple[str, ...]
    engines: tuple[str, ...]
    performance_cases: tuple[PerformanceCase, ...]
    decoding_policy: str


@dataclass(frozen=True, slots=True)
class QualityCase:
    id: str
    category: str
    prompt: str
    max_new_tokens: int
    scorer: str
    expected: object
    decoding_policy: str


@dataclass(frozen=True, slots=True)
class EngineManifest:
    version: int
    name: str
    binary: str
    version_command: tuple[str, ...]
    supported_modes: tuple[str, ...]
    version_pin_file: str | None
    pinned_version: str | None
