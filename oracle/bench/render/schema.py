"""Shared JSON schema for run-dir artifacts. Mirrors crates/bench/src/runs.rs."""
try:
    import jsonschema
except ModuleNotFoundError:  # pragma: no cover - local bare env fallback
    jsonschema = None

META_SCHEMA = {
    "type": "object",
    "required": [
        "schema_version", "run_id", "timestamp_utc", "git_sha", "hostname",
        "arch", "rocminfo", "rocm_smi_u", "runner_version",
    ],
    "properties": {
        "schema_version": {"type": "integer", "enum": [1, 2]},
        "run_id": {"type": "string"},
        "timestamp_utc": {"type": "string"},
        "git_sha": {"type": "string"},
        "hostname": {"type": "string"},
        "arch": {"type": "string"},
        "rocminfo": {"type": "string"},
        "rocm_smi_u": {"type": "string"},
        "gpu_temp_c_pre": {"type": ["number", "null"]},
        "gpu_temp_c_post": {"type": ["number", "null"]},
        "runner_version": {"type": "string"},
    },
}

_PERF_BASE = {
    "schema_version": {"type": "integer", "enum": [1, 2]},
    "model": {"type": "string"},
    "quant": {"type": "string"},
    "arch": {"type": "string"},
    "backend": {"type": "string"},
    "prompt": {"type": "string"},
    "max_new_tokens": {"type": "integer", "minimum": 1},
    "stage_timings": {
        "type": "object",
        "additionalProperties": {"type": "number"},
    },
    "chain_breakdown": {
        "type": "object",
        "additionalProperties": {"type": "number"},
    },
    "lifecycle_timings": {
        "type": "object",
        "additionalProperties": {"type": "number"},
    },
    "gpu_temp_c_end": {"type": ["number", "null"]},
}
_PERF_REQUIRED = [
    "schema_version", "model", "quant", "prompt", "max_new_tokens", "gpu_temp_c_end",
]

PERF_CELL_SCHEMA = {
    "type": "object",
    "allOf": [
        {
            "if": {"properties": {"schema_version": {"const": 2}}},
            "then": {"required": ["arch", "backend"]},
        }
    ],
    "oneOf": [
        {
            "required": _PERF_REQUIRED + ["status", "ms_per_step", "ms_per_tok", "samples"],
            "properties": {**_PERF_BASE,
                           "status": {"const": "ok"},
                           "ms_per_step": {"type": "number"},
                           "ms_per_tok": {"type": "number"},
                           "samples": {"type": "array", "items": {"type": "number"}}},
        },
        {
            "required": _PERF_REQUIRED + ["status", "reason"],
            "properties": {**_PERF_BASE,
                           "status": {"const": "skipped"},
                           "reason": {"type": "string"}},
        },
        {
            "required": _PERF_REQUIRED + ["status", "stderr_tail"],
            "properties": {**_PERF_BASE,
                           "status": {"const": "error"},
                           "stderr_tail": {"type": "string"}},
        },
    ],
}

QUALITY_CELL_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "model", "quant", "eval", "metric", "value"],
    "properties": {
        "schema_version": {"type": "integer", "const": 1},
        "model": {"type": "string"},
        "quant": {"type": "string"},
        "eval": {"type": "string"},        # e.g. "perplexity_pg19", "golden_diff", "niah_4k"
        "metric": {"type": "string"},      # e.g. "ppl", "exact_match", "score"
        "value": {"type": "number"},
        "extras": {"type": "object"},      # eval-specific extras (free-form)
    },
}

EXTERNAL_CELL_SCHEMA = {
    "type": "object",
    "required": ["schema_version", "engine", "engine_version", "model", "quant", "status"],
    "properties": {
        "schema_version": {"type": "integer", "const": 1},
        "engine": {"type": "string"},
        "engine_version": {"type": "string"},
        "model": {"type": "string"},
        "quant": {"type": "string"},
        "status": {"enum": ["ok", "unsupported_by_engine", "error"]},
        "ms_per_step": {"type": ["number", "null"]},
        "samples": {"type": ["array", "null"], "items": {"type": "number"}},
        "stderr_tail": {"type": ["string", "null"]},
    },
}


def validate_meta(d: dict) -> None:
    _validate(d, META_SCHEMA)


def validate_perf_cell(d: dict) -> None:
    if jsonschema is None:
        _validate_one_of(d, PERF_CELL_SCHEMA)
        return
    _validate(d, PERF_CELL_SCHEMA)


def validate_quality_cell(d: dict) -> None:
    _validate(d, QUALITY_CELL_SCHEMA)


def validate_external_cell(d: dict) -> None:
    _validate(d, EXTERNAL_CELL_SCHEMA)


def _validate(d: dict, schema: dict) -> None:
    if jsonschema is not None:
        jsonschema.validate(d, schema)
        return
    for key in schema.get("required", []):
        if key not in d:
            raise ValueError(f"missing required field {key!r}")


def _validate_one_of(d: dict, schema: dict) -> None:
    failures = []
    for variant in schema.get("oneOf", []):
        try:
            _validate(d, variant)
            _validate_consts(d, variant)
            return
        except Exception as exc:
            failures.append(str(exc))
    raise ValueError("object did not match any schema variant: " + "; ".join(failures))


def _validate_consts(d: dict, schema: dict) -> None:
    for key, prop in schema.get("properties", {}).items():
        if key in d and "const" in prop and d[key] != prop["const"]:
            raise ValueError(f"field {key!r} expected {prop['const']!r}, got {d[key]!r}")
