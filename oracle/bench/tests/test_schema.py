"""Tests for the shared run-dir JSON schema."""
import json
from pathlib import Path

import pytest

from oracle.bench.render.schema import (
    META_SCHEMA, PERF_CELL_SCHEMA, validate_external_cell, validate_meta, validate_perf_cell,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "run_minimal"


def test_meta_fixture_validates():
    meta = json.loads((FIXTURE_DIR / "meta.json").read_text())
    validate_meta(meta)  # raises if invalid


def test_perf_cell_fixture_validates():
    cell = json.loads((FIXTURE_DIR / "perf" / "qwen3.5-0.8b_bf16.json").read_text())
    validate_perf_cell(cell)


def test_perf_cell_skipped_status():
    cell = {
        "schema_version": 1,
        "model": "qwen3.5-9b",
        "quant": "bf16",
        "prompt": "x",
        "max_new_tokens": 1,
        "status": "skipped",
        "reason": "OOM at preflight",
        "gpu_temp_c_end": None,
    }
    validate_perf_cell(cell)


def test_perf_cell_error_status():
    cell = {
        "schema_version": 1,
        "model": "qwen3.5-9b",
        "quant": "int4",
        "prompt": "x",
        "max_new_tokens": 1,
        "status": "error",
        "stderr_tail": "panicked at line 42",
        "gpu_temp_c_end": None,
    }
    validate_perf_cell(cell)


def test_perf_cell_v2_attribution_status():
    cell = {
        "schema_version": 2,
        "model": "qwen3.6-35b-a3b",
        "quant": "int4",
        "arch": "apple-m5-max",
        "backend": "metal",
        "prompt": "x",
        "max_new_tokens": 16,
        "status": "ok",
        "ms_per_step": 27.5,
        "ms_per_tok": 27.5,
        "samples": [27.4, 27.5, 27.6],
        "stage_timings": {"chain_ms_avg": 25.0},
        "chain_breakdown": {"ffn_ms_avg": 12.0},
        "lifecycle_timings": {"prefill_total_ms": 1000.0},
        "gpu_temp_c_end": None,
    }
    validate_perf_cell(cell)


def test_perf_cell_invalid_status_rejected():
    cell = {
        "schema_version": 1,
        "model": "qwen3.5-0.8b",
        "quant": "bf16",
        "prompt": "x",
        "max_new_tokens": 1,
        "status": "ok",
        # missing ms_per_step and samples
        "gpu_temp_c_end": None,
    }
    with pytest.raises(Exception):
        validate_perf_cell(cell)


def test_external_cell_accepts_workload_metadata():
    cell = {
        "schema_version": 1,
        "engine": "llama.cpp",
        "engine_version": "llama.cpp build 1234",
        "model": "qwen3.5-35b-a3b",
        "quant": "q4km",
        "status": "ok",
        "ms_per_step": 11.0,
        "tok_per_s": 90.9,
        "samples": [10.9, 11.0, 11.1],
        "stderr_tail": None,
        "command": ["llama-bench", "-m", "/models/qwen.gguf", "-n", "512"],
        "workload": {
            "prompt": "",
            "prompt_tokens": 0,
            "max_new_tokens": 512,
            "context_size": 1024,
            "warmup_runs": 1,
            "measurement_runs": 5,
        },
        "extras": {"batch_context": "llama-bench defaults"},
    }
    validate_external_cell(cell)
