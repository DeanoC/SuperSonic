from __future__ import annotations

from collections import Counter
from pathlib import Path
import tomllib

from .model import EngineManifest, PerformanceCase, QualityCase, SuiteManifest, canonical_json, parse_strict_json


ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ROOT / "benchmarks"
SUITES = BENCHMARKS / "suites"
QUALITY = BENCHMARKS / "quality"
ENGINES = BENCHMARKS / "engines"

SUITE_KEYS = {
    "version",
    "name",
    "budget_seconds",
    "quality_version",
    "quality_case_ids",
    "engines",
    "decoding_policy",
    "performance_cases",
}
PERFORMANCE_CASE_KEYS = {
    "id",
    "prompt",
    "max_new_tokens",
    "warmups",
    "repetitions",
    "mode",
    "cache_state",
    "timeout_seconds",
    "decoding_policy",
    "engines",
}
QUALITY_KEYS = {"version", "categories", "cases"}
QUALITY_CASE_KEYS = {
    "id",
    "category",
    "prompt",
    "max_new_tokens",
    "scorer",
    "expected",
    "decoding_policy",
}
ENGINE_BASE_KEYS = {"version", "name", "binary", "version_command", "supported_modes"}
ALLOWED_MODES = {"ordinary", "mtp"}
ALLOWED_CACHE_STATES = {
    "cold-load",
    "warm-resident",
    "prefix-cache-empty",
    "prefix-cache-populated",
    "prefix-cache-reset",
}
ALLOWED_SCORERS = {"exact_text", "exact_tokens", "structured_json"}
APPROVED_CATEGORIES = {
    "instruction-following",
    "structured-extraction",
    "arithmetic-and-reasoning",
    "code-completion",
    "long-context-retrieval",
    "chat-template-behavior",
    "repeated-run-determinism",
    "ordinary-vs-mtp-token-equality",
}


def load_suite(name: str) -> SuiteManifest:
    return load_suite_path(SUITES / f"{name}.toml")


def load_suite_path(path: Path) -> SuiteManifest:
    data = _load_toml(path)
    _require_exact_keys(data, SUITE_KEYS, f"suite {path.name}")

    version = _require_int(data, "version", minimum=1)
    if version != 1:
        raise ValueError(f"suite {path.name} must use version 1")
    name = _require_nonempty_str(data, "name")
    budget_seconds = _require_int(data, "budget_seconds", minimum=1)
    quality_version = _require_nonempty_str(data, "quality_version")
    decoding_policy = _require_greedy(data, "decoding_policy", f"suite {path.name}")

    quality_cases = load_quality(quality_version)
    quality_case_ids = _require_str_tuple(data, "quality_case_ids", minimum_items=1)
    if len(quality_case_ids) != len(set(quality_case_ids)):
        raise ValueError(f"suite {path.name} quality_case_ids must be unique")
    quality_case_set = {case.id for case in quality_cases}
    missing_quality = sorted(set(quality_case_ids) - quality_case_set)
    if missing_quality:
        raise ValueError(f"suite {path.name} references unknown quality cases: {missing_quality}")

    engines = _require_str_tuple(data, "engines", minimum_items=1)
    if len(engines) != len(set(engines)):
        raise ValueError(f"suite {path.name} engines must be unique")
    engine_manifests = tuple(load_engine(engine_name) for engine_name in engines)

    raw_cases = data["performance_cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"suite {path.name} performance_cases must be a non-empty array")
    performance_cases = tuple(
        _parse_performance_case(
            entry,
            suite_name=name,
            suite_decoding_policy=decoding_policy,
            suite_engines=engines,
            engine_manifests=engine_manifests,
        )
        for entry in raw_cases
    )
    case_ids = [case.id for case in performance_cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"suite {path.name} performance case ids must be unique")

    return SuiteManifest(
        version=version,
        name=name,
        budget_seconds=budget_seconds,
        quality_version=quality_version,
        quality_case_ids=quality_case_ids,
        engines=engines,
        performance_cases=performance_cases,
        decoding_policy=decoding_policy,
    )


def load_quality(version: str) -> tuple[QualityCase, ...]:
    path = QUALITY / f"{version}.json"
    data = _load_json(path)
    _require_exact_keys(data, QUALITY_KEYS, f"quality {path.name}")

    file_version = _require_nonempty_str(data, "version")
    if file_version != version:
        raise ValueError(f"quality file {path.name} version mismatch: {file_version!r}")
    categories = data["categories"]
    if not isinstance(categories, list) or not categories:
        raise ValueError(f"quality {path.name} categories must be a non-empty array")
    category_tuple = tuple(_require_plain_str(value, f"quality {path.name} categories[]") for value in categories)
    if len(category_tuple) != len(set(category_tuple)):
        raise ValueError(f"quality {path.name} categories must be unique")
    if set(category_tuple) != APPROVED_CATEGORIES:
        raise ValueError(f"quality {path.name} categories must match the approved set")

    raw_cases = data["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"quality {path.name} cases must be a non-empty array")
    cases = tuple(_parse_quality_case(entry, source=path.name) for entry in raw_cases)
    case_ids = [case.id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"quality {path.name} case ids must be unique")

    counts = Counter(case.category for case in cases)
    for category in category_tuple:
        if counts[category] < 2:
            raise ValueError(f"quality {path.name} category {category!r} needs at least two cases")
    return cases


def load_engine(name: str) -> EngineManifest:
    path = ENGINES / f"{name}.toml"
    data = _load_toml(path)
    allowed_keys = set(ENGINE_BASE_KEYS)
    if "version_pin_file" in data:
        allowed_keys.add("version_pin_file")
    _require_exact_keys(data, allowed_keys, f"engine {path.name}")

    version = _require_int(data, "version", minimum=1)
    if version != 1:
        raise ValueError(f"engine {path.name} must use version 1")
    engine_name = _require_nonempty_str(data, "name")
    if engine_name != name:
        raise ValueError(f"engine {path.name} name mismatch: expected {name!r}, got {engine_name!r}")
    binary = _require_nonempty_str(data, "binary")
    version_command_values = data["version_command"]
    if not isinstance(version_command_values, list) or not version_command_values:
        raise ValueError(f"engine {path.name} version_command must be a non-empty array")
    version_command = tuple(
        _require_plain_str(value, f"engine {path.name} version_command[]")
        for value in version_command_values
    )
    supported_modes = _require_str_tuple(data, "supported_modes", minimum_items=1)
    if len(supported_modes) != len(set(supported_modes)):
        raise ValueError(f"engine {path.name} supported_modes must be unique")
    unknown_modes = sorted(set(supported_modes) - ALLOWED_MODES)
    if unknown_modes:
        raise ValueError(f"engine {path.name} has unknown supported_modes: {unknown_modes}")

    version_pin_file = data.get("version_pin_file")
    pinned_version = None
    if version_pin_file is not None:
        version_pin_file = _require_plain_str(version_pin_file, f"engine {path.name} version_pin_file")
        pin_path = ROOT / version_pin_file
        pinned_version = _read_pin_file(pin_path)

    return EngineManifest(
        version=version,
        name=engine_name,
        binary=binary,
        version_command=version_command,
        supported_modes=supported_modes,
        version_pin_file=version_pin_file,
        pinned_version=pinned_version,
    )


def _parse_performance_case(
    data: object,
    *,
    suite_name: str,
    suite_decoding_policy: str,
    suite_engines: tuple[str, ...],
    engine_manifests: tuple[EngineManifest, ...],
) -> PerformanceCase:
    if not isinstance(data, dict):
        raise ValueError(f"suite {suite_name} performance case entries must be tables")
    _require_exact_keys(data, PERFORMANCE_CASE_KEYS, f"suite {suite_name} performance case")
    mode = _require_nonempty_str(data, "mode")
    if mode not in ALLOWED_MODES:
        raise ValueError(f"suite {suite_name} has unknown mode: {mode!r}")
    cache_state = _require_nonempty_str(data, "cache_state")
    if cache_state not in ALLOWED_CACHE_STATES:
        raise ValueError(f"suite {suite_name} has unknown cache_state: {cache_state!r}")
    decoding_policy = _require_greedy(
        data,
        "decoding_policy",
        f"suite {suite_name} performance case {data.get('id', '<unknown>')}",
    )
    if decoding_policy != suite_decoding_policy:
        raise ValueError(f"suite {suite_name} performance case decoding policy mismatch")
    scoped_engines = _require_str_tuple(data, "engines", minimum_items=1)
    if len(scoped_engines) != len(set(scoped_engines)):
        raise ValueError(f"suite {suite_name} performance case engines must be unique")
    unknown_engines = sorted(set(scoped_engines) - set(suite_engines))
    if unknown_engines:
        raise ValueError(
            f"suite {suite_name} performance case references engines not declared by the suite: {unknown_engines}"
        )
    engines_by_name = {engine.name: engine for engine in engine_manifests}
    unsupported_engines = sorted(
        engine_name
        for engine_name in scoped_engines
        if mode not in engines_by_name[engine_name].supported_modes
    )
    if unsupported_engines:
        raise ValueError(
            f"suite {suite_name} performance case mode {mode!r} is unsupported by engines: {unsupported_engines}"
        )
    return PerformanceCase(
        id=_require_nonempty_str(data, "id"),
        prompt=_require_nonempty_str(data, "prompt"),
        max_new_tokens=_require_int(data, "max_new_tokens", minimum=1),
        warmups=_require_int(data, "warmups", minimum=0),
        repetitions=_require_int(data, "repetitions", minimum=1),
        mode=mode,
        cache_state=cache_state,
        timeout_seconds=_require_int(data, "timeout_seconds", minimum=1),
        decoding_policy=decoding_policy,
        engines=scoped_engines,
    )


def _parse_quality_case(data: object, *, source: str) -> QualityCase:
    if not isinstance(data, dict):
        raise ValueError(f"quality {source} case entries must be objects")
    _require_exact_keys(data, QUALITY_CASE_KEYS, f"quality {source} case")
    category = _require_nonempty_str(data, "category")
    if category not in APPROVED_CATEGORIES:
        raise ValueError(f"quality {source} has unknown category: {category!r}")
    scorer = _require_nonempty_str(data, "scorer")
    if scorer not in ALLOWED_SCORERS:
        raise ValueError(f"quality {source} has unsupported scorer: {scorer!r}")
    return QualityCase(
        id=_require_nonempty_str(data, "id"),
        category=category,
        prompt=_require_nonempty_str(data, "prompt"),
        max_new_tokens=_require_int(data, "max_new_tokens", minimum=1),
        scorer=scorer,
        expected=data["expected"],
        decoding_policy=_require_greedy(data, "decoding_policy", f"quality {source} case"),
    )


def _load_toml(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing manifest file: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"manifest {path} must contain a top-level table")
    return data


def _load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing manifest file: {path}")
    data = parse_strict_json(path.read_text(encoding="utf-8"), context=f"manifest {path}")
    if not isinstance(data, dict):
        raise ValueError(f"manifest {path} must contain a top-level object")
    return data


def _read_pin_file(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"missing version pin file: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        candidate = line.strip()
        if candidate and not candidate.startswith("#"):
            return candidate
    raise ValueError(f"version pin file {path} does not contain a pinned version")


def _require_exact_keys(data: dict[str, object], allowed: set[str], context: str) -> None:
    unknown = sorted(set(data) - allowed)
    missing = sorted(allowed - set(data))
    if unknown:
        raise ValueError(f"{context} has unknown keys: {unknown}")
    if missing:
        raise ValueError(f"{context} is missing required keys: {missing}")


def _require_nonempty_str(data: dict[str, object], key: str) -> str:
    return _require_plain_str(data.get(key), key)


def _require_plain_str(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_int(data: dict[str, object], key: str, *, minimum: int) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value


def _require_greedy(data: dict[str, object], key: str, context: str) -> str:
    value = _require_nonempty_str(data, key)
    if value != "greedy":
        raise ValueError(f"{context} must use greedy decoding")
    return value


def _require_str_tuple(
    data: dict[str, object],
    key: str,
    *,
    minimum_items: int,
) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list) or len(value) < minimum_items:
        raise ValueError(f"{key} must contain at least {minimum_items} entries")
    return tuple(_require_plain_str(item, f"{key}[]") for item in value)


__all__ = [
    "EngineManifest",
    "PerformanceCase",
    "QualityCase",
    "SuiteManifest",
    "canonical_json",
    "load_engine",
    "load_quality",
    "load_suite",
    "load_suite_path",
]
