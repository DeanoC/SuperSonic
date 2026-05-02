#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models/supersonic-cdna/qwen3.6-35b-a3b-fp8}"
BINARY="${BINARY:-target/release/supersonic}"
CONTEXTS="${CONTEXTS:-8K}"
FAMILIES="${FAMILIES:-pg19}"
MODES="${MODES:-fp8}"
DECODE_PATHS="${DECODE_PATHS:-chained}"
REPEATS="${REPEATS:-1}"
TIMEOUT="${TIMEOUT:-1800}"
PG19_SOURCE="${PG19_SOURCE:-synthetic}"
OUT="${OUT:-target/qwen36_verify_long_context.json}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv-verify/bin/python" ]]; then
    PYTHON_BIN=".venv-verify/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

cargo build --release --bin supersonic

"${PYTHON_BIN}" oracle/qwen36_verify_suite.py \
  --binary "${BINARY}" \
  --model-dir "${MODEL_DIR}" \
  --backend hip \
  --contexts "${CONTEXTS}" \
  --families "${FAMILIES}" \
  --modes "${MODES}" \
  --decode-paths "${DECODE_PATHS}" \
  --repeats "${REPEATS}" \
  --timeout "${TIMEOUT}" \
  --pg19-source "${PG19_SOURCE}" \
  --continue-on-error \
  --emit-stage-timings \
  --out "${OUT}"
