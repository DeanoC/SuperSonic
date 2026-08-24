from __future__ import annotations

from dataclasses import dataclass
import json


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


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


@dataclass(frozen=True, slots=True)
class SuiteManifest:
    version: int
    name: str
    budget_seconds: int
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
