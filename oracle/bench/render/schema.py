"""Shared JSON schema for run-dir artifacts. Mirrors crates/bench/src/runs.rs."""
import jsonschema

META_SCHEMA = {
    "type": "object",
    "required": [
        "schema_version", "run_id", "timestamp_utc", "git_sha", "hostname",
        "arch", "rocminfo", "rocm_smi_u", "runner_version",
    ],
    "properties": {
        "schema_version": {"type": "integer", "const": 1},
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
    "schema_version": {"type": "integer", "const": 1},
    "model": {"type": "string"},
    "quant": {"type": "string"},
    "prompt": {"type": "string"},
    "max_new_tokens": {"type": "integer", "minimum": 1},
    "gpu_temp_c_end": {"type": ["number", "null"]},
}

PERF_CELL_SCHEMA = {
    "type": "object",
    "oneOf": [
        {
            "required": list(_PERF_BASE) + ["status", "ms_per_step", "ms_per_tok", "samples"],
            "properties": {**_PERF_BASE,
                           "status": {"const": "ok"},
                           "ms_per_step": {"type": "number"},
                           "ms_per_tok": {"type": "number"},
                           "samples": {"type": "array", "items": {"type": "number"}}},
        },
        {
            "required": list(_PERF_BASE) + ["status", "reason"],
            "properties": {**_PERF_BASE,
                           "status": {"const": "skipped"},
                           "reason": {"type": "string"}},
        },
        {
            "required": list(_PERF_BASE) + ["status", "stderr_tail"],
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
    jsonschema.validate(d, META_SCHEMA)


def validate_perf_cell(d: dict) -> None:
    jsonschema.validate(d, PERF_CELL_SCHEMA)


def validate_quality_cell(d: dict) -> None:
    jsonschema.validate(d, QUALITY_CELL_SCHEMA)


def validate_external_cell(d: dict) -> None:
    jsonschema.validate(d, EXTERNAL_CELL_SCHEMA)
