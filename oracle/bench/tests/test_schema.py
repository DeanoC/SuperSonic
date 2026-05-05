"""Tests for the shared run-dir JSON schema."""
import json
from pathlib import Path

import pytest

from oracle.bench.render.schema import (
    META_SCHEMA, PERF_CELL_SCHEMA, validate_meta, validate_perf_cell,
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
