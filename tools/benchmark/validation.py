from __future__ import annotations

from pathlib import Path
import json
import math
import re

from . import environment
from .manifest import ROOT, load_suite


SCHEMA_PATH = ROOT / "benchmarks" / "schema" / "result-v1.schema.json"
_SCHEMA_KEYS = {
    "$id",
    "$schema",
    "additionalProperties",
    "const",
    "enum",
    "items",
    "minItems",
    "minimum",
    "pattern",
    "properties",
    "required",
    "type",
}
_JSON_TYPES = {"array", "boolean", "integer", "null", "number", "object", "string"}
_SECRET_KEY_PARTS = {
    "access_key",
    "apikey",
    "api_key",
    "auth",
    "authorization",
    "credential",
    "credentials",
    "password",
    "passwd",
    "private_key",
    "secret",
    "token",
}
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bbearer\s+[a-z0-9._~+/=-]+", re.IGNORECASE),
    re.compile(r"\bsk-[a-z0-9][a-z0-9_-]{8,}", re.IGNORECASE),
    re.compile(r"\b(?:ghp|github_pat|hf)_[a-z0-9_]{8,}", re.IGNORECASE),
)
_SAFE_ABSOLUTE_PREFIXES = (
    "/bin/",
    "/dev/",
    "/etc/",
    "/opt/rocm",
    "/sbin/",
    "/usr/",
)


def validate_record(record: object) -> None:
    if not isinstance(record, dict):
        raise ValueError("benchmark result record must be a JSON object")
    _reject_unsafe_json(record, "$")
    _validate_schema(record, _load_schema(), "$")
    _validate_record_consistency(record)


def validate_bundle(path: str | Path, require_complete: bool) -> tuple[Path, ...]:
    paths = _bundle_paths(Path(path))
    records: list[tuple[Path, dict[str, object]]] = []
    for result_path in paths:
        value = _load_json(result_path)
        validate_record(value)
        records.append((result_path, value))

    if require_complete:
        for result_path, record in records:
            _validate_publishable(record, result_path)
        _validate_complete_bundle(records)
    return paths


def _load_schema() -> dict[str, object]:
    value = _load_json(SCHEMA_PATH)
    if not isinstance(value, dict):
        raise ValueError(f"schema {SCHEMA_PATH} must be a JSON object")
    _check_schema_keywords(value, "$schema")
    return value


def _load_json(path: Path) -> object:
    if not path.is_file():
        raise ValueError(f"missing JSON file: {path}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{path} contains duplicate key: {key!r}")
            value[key] = item
        return value

    def reject_constant(token: str) -> object:
        raise ValueError(f"{path} contains non-finite number: {token}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} must contain valid JSON") from exc


def _bundle_paths(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    if not path.is_dir():
        raise ValueError(f"bundle path does not exist: {path}")
    paths = tuple(sorted(candidate for candidate in path.rglob("*.json") if candidate.is_file()))
    if not paths:
        raise ValueError(f"benchmark bundle contains no JSON results: {path}")
    return paths


def _check_schema_keywords(schema: object, path: str) -> None:
    if isinstance(schema, dict):
        unknown = sorted(str(key) for key in schema if key not in _SCHEMA_KEYS)
        if unknown:
            raise ValueError(f"{path} uses unsupported schema keywords: {unknown}")
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            unknown_types = [item for item in schema_type if item not in _JSON_TYPES]
            if unknown_types:
                raise ValueError(f"{path}.type has unsupported JSON types: {unknown_types}")
        elif isinstance(schema_type, str) and schema_type not in _JSON_TYPES:
            raise ValueError(f"{path}.type has unsupported JSON type: {schema_type!r}")
        for name, subschema in schema.get("properties", {}).items():
            _check_schema_keywords(subschema, f"{path}.properties.{name}")
        if "items" in schema:
            _check_schema_keywords(schema["items"], f"{path}.items")
    elif not isinstance(schema, (str, int, float, bool, list, type(None))):
        raise ValueError(f"{path} contains unsupported schema value: {type(schema).__name__}")


def _validate_schema(value: object, schema: dict[str, object], path: str) -> None:
    if "type" in schema and not _matches_type(value, schema["type"]):
        raise ValueError(f"{path} must match schema type {schema['type']!r}")

    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} must equal schema const {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must match one of schema enum values")
    if "minimum" in schema and isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema["minimum"]:
            raise ValueError(f"{path} must be >= {schema['minimum']}")
    if "pattern" in schema and isinstance(value, str):
        if re.fullmatch(str(schema["pattern"]), value) is None:
            raise ValueError(f"{path} does not match schema pattern {schema['pattern']!r}")

    if isinstance(value, dict):
        required = schema.get("required", [])
        if not isinstance(required, list):
            raise ValueError(f"{path}.required schema entry must be an array")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"{path} is missing required fields: {missing}")
        properties = schema.get("properties", {})
        if properties is not None and not isinstance(properties, dict):
            raise ValueError(f"{path}.properties schema entry must be an object")
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                raise ValueError(f"{path} contains additional properties: {unknown}")
        for key, item in value.items():
            if key in properties:
                child_schema = properties[key]
                if not isinstance(child_schema, dict):
                    raise ValueError(f"{path}.{key} schema entry must be an object")
                _validate_schema(item, child_schema, f"{path}.{key}")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            raise ValueError(f"{path} must contain at least {schema['minItems']} items")
        if "items" in schema:
            item_schema = schema["items"]
            if not isinstance(item_schema, dict):
                raise ValueError(f"{path}.items schema entry must be an object")
            for index, item in enumerate(value):
                _validate_schema(item, item_schema, f"{path}[{index}]")


def _matches_type(value: object, expected: object) -> bool:
    if isinstance(expected, list):
        return any(_matches_type(value, item) for item in expected)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and _finite(value)
    if expected == "null":
        return value is None
    raise ValueError(f"unsupported schema type: {expected!r}")


def _reject_unsafe_json(value: object, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains non-finite number")
    if isinstance(value, str):
        _reject_unsafe_string(value, path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_unsafe_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string object key")
            if _secret_like_key(key):
                raise ValueError(f"{path}.{key} contains secret-like key")
            _reject_unsafe_json(item, f"{path}.{key}")


def _reject_unsafe_string(value: str, path: str) -> None:
    if value.startswith("/") and not _safe_absolute_path(value):
        raise ValueError(f"{path} contains unsafe absolute path")
    if any(pattern.search(value) for pattern in _SECRET_VALUE_PATTERNS):
        raise ValueError(f"{path} contains secret-like value")


def _safe_absolute_path(value: str) -> bool:
    if value in ("/bin", "/dev", "/etc", "/opt/rocm", "/sbin", "/usr"):
        return True
    return any(value.startswith(prefix) for prefix in _SAFE_ABSOLUTE_PREFIXES)


def _secret_like_key(key: str) -> bool:
    lowered = key.lower()
    parts = [part for part in re.split(r"[^a-z0-9]+", lowered) if part]
    joined_pairs = {f"{left}_{right}" for left, right in zip(parts, parts[1:])}
    matched = (set(parts) | joined_pairs) & _SECRET_KEY_PARTS
    if not matched:
        return False
    if matched - {"token", "auth"}:
        return True
    secret_context = {"access", "api", "bearer", "github", "hf", "openai", "private"}
    return len(parts) == 1 or bool(set(parts) & secret_context)


def _finite(value: int | float) -> bool:
    return not isinstance(value, float) or math.isfinite(value)


def _validate_record_consistency(record: dict[str, object]) -> None:
    run = record["run"]
    engine_info = record["engine"]
    workload = record["workload"]
    quality = record["quality"]
    env = record["environment"]
    hardware = record["hardware"]
    if not isinstance(run, dict) or not isinstance(engine_info, dict) or not isinstance(workload, dict):
        raise ValueError("record sections must be objects")
    suite = load_suite(str(run["suite"]))
    if run["suite_version"] != suite.version:
        raise ValueError("run suite_version does not match suite manifest")
    if run["quality_version"] != suite.quality_version:
        raise ValueError("run quality_version does not match suite manifest")
    if run["case_id"] != workload["case_id"]:
        raise ValueError("run case_id must match workload case_id")

    matching_cases = [case for case in suite.performance_cases if case.id == run["case_id"]]
    if not matching_cases:
        raise ValueError(f"run case_id is not in suite manifest: {run['case_id']!r}")
    case = matching_cases[0]
    engine_name = str(engine_info["name"])
    if engine_name not in case.engines:
        raise ValueError(f"engine {engine_name!r} is not configured for case {case.id!r}")
    if workload["max_new_tokens"] != case.max_new_tokens:
        raise ValueError("workload max_new_tokens does not match suite manifest")
    if workload["warmups"] != case.warmups:
        raise ValueError("workload warmups does not match suite manifest")
    if workload["mode"] != case.mode:
        raise ValueError("workload mode does not match suite manifest")
    if workload["cache_state"] != case.cache_state:
        raise ValueError("workload cache_state does not match suite manifest")
    sample_count = len(record["samples"])
    if sample_count != case.repetitions:
        raise ValueError(f"sample count must be exactly {case.repetitions} for {case.id}, got {sample_count}")

    if env["cache_state"] != workload["cache_state"]:
        raise ValueError("environment cache_state must match workload cache_state")
    if env["clock_policy"] != hardware["clock_policy"]:
        raise ValueError("environment clock_policy must match hardware clock_policy")
    environment.validate_cache_evidence(str(env["cache_state"]), env["cache_evidence"])
    if env["process_reuse"] is not False:
        raise ValueError("process_reuse must remain false")

    _validate_quality_summary(quality, suite.quality_case_ids)


def _validate_quality_summary(quality: dict[str, object], required_case_ids: tuple[str, ...]) -> None:
    cases = quality["cases"]
    if not isinstance(cases, list):
        raise ValueError("quality cases must be an array")
    case_ids = [case["id"] for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("quality case ids must be unique")
    if set(case_ids) != set(required_case_ids):
        raise ValueError("quality cases must exactly match suite quality_case_ids")
    passed = sum(1 for case in cases if case["passed"])
    failed = len(cases) - passed
    if quality["passed"] != passed or quality["failed"] != failed or quality["total"] != len(cases):
        raise ValueError("quality summary counts do not match cases")
    if quality["missing_case_ids"]:
        raise ValueError("quality summary reports missing cases")

    category_counts: dict[str, dict[str, int]] = {}
    for case in cases:
        bucket = category_counts.setdefault(str(case["category"]), {"passed": 0, "failed": 0, "total": 0})
        bucket["total"] += 1
        if case["passed"]:
            bucket["passed"] += 1
        else:
            bucket["failed"] += 1
    if quality["categories"] != category_counts:
        raise ValueError("quality category counts do not match cases")


def _validate_publishable(record: dict[str, object], path: Path) -> None:
    if record["run"]["dirty"]:
        raise ValueError(f"{path} is dirty and cannot be published")
    if record["status"]["state"] != "complete":
        raise ValueError(f"{path} has failed or incomplete status")
    if record["errors"]:
        error_words = " ".join(f"{item['code']} {item['message']}" for item in record["errors"])
        if "missing-configured-input" in error_words:
            raise ValueError(f"{path} has configured missing inputs")
        raise ValueError(f"{path} has configured errors")
    if record["quality"]["failed"] != 0:
        raise ValueError(f"{path} has failing quality results")
    if not all(case["passed"] for case in record["quality"]["cases"]):
        raise ValueError(f"{path} has failing quality cases")
    if record["quality"]["missing_case_ids"]:
        raise ValueError(f"{path} has missing quality cases")
    if record["environment"]["headline_eligible"] is not True or record["environment"]["verification_errors"]:
        raise ValueError(f"{path} lacks verified headline eligibility")


def _validate_complete_bundle(records: list[tuple[Path, dict[str, object]]]) -> None:
    by_suite: dict[str, set[tuple[str, str]]] = {}
    for _, record in records:
        by_suite.setdefault(str(record["run"]["suite"]), set()).add(
            (str(record["run"]["case_id"]), str(record["engine"]["name"]))
        )
    for suite_name, actual in by_suite.items():
        suite = load_suite(suite_name)
        expected = {
            (case.id, engine_name)
            for case in suite.performance_cases
            for engine_name in case.engines
        }
        missing = sorted(expected - actual)
        if missing:
            raise ValueError(f"incomplete benchmark bundle for suite {suite_name}: missing {missing}")
