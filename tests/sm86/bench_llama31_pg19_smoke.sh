#!/usr/bin/env bash
set -euo pipefail

# Llama 3.1 8B PG-19 teacher-forced smoke harness.
#
# Environment overrides:
#   CONTEXTS=512                 comma-separated token contexts, e.g. 512,4096
#   NUM_CHUNKS=1                 PG-19 chunks per context
#   CONFIG=both                  dense, certified, or both
#   OUTPUT=target/pg19_smoke.json
#   TIMEOUT=900                  per SuperSonic scorer invocation
#   EMIT_STAGE_TIMINGS=0         include per-stage GPU/runtime timing telemetry
#   CERTIFIED_EXTRA_ARGS=""      extra args forwarded only to certified scorer
#   MAX_CERTIFIED_DELTA=0.10     fail if certified ppl exceeds dense by this much
#   FAIL_ABOVE_REFERENCE=0       compare to DotCache arxiv_v1 reference when present
#   REFERENCE_TOLERANCE=0.05     additive ppl tolerance vs DotCache reference
#   EVAL_START_FRAC=             certified dense-prefix fraction; defaults to
#                                 0.5 with reference checks, else full scoring
#   SOURCE_TEXT=/path/text.txt   optional local text instead of HF PG-19 streaming

MODEL_DIR="${1:-${SUPERSONIC_MODEL_DIR_LLAMA31:-}}"
if [[ -z "$MODEL_DIR" ]]; then
  echo "Usage: $0 /path/to/Meta-Llama-3.1-8B" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}"
cargo build --release --bin supersonic

ARGS=(
  --binary ./target/release/supersonic
  --model-dir "$MODEL_DIR"
  --contexts "${CONTEXTS:-512}"
  --num-chunks "${NUM_CHUNKS:-1}"
  --config "${CONFIG:-both}"
  --output "${OUTPUT:-target/pg19_smoke.json}"
  --timeout "${TIMEOUT:-900}"
  --reference-tolerance "${REFERENCE_TOLERANCE:-0.05}"
)

if [[ -n "${SOURCE_TEXT:-}" ]]; then
  ARGS+=(--source-text "$SOURCE_TEXT")
fi
if [[ -n "${MAX_CERTIFIED_DELTA:-0.10}" ]]; then
  ARGS+=(--max-certified-delta "${MAX_CERTIFIED_DELTA:-0.10}")
fi
if [[ "${FAIL_ABOVE_REFERENCE:-0}" == "1" ]]; then
  ARGS+=(--fail-above-reference)
fi
if [[ "${REFERENCE_SMOKE:-0}" == "1" ]]; then
  ARGS+=(--reference-smoke)
fi
if [[ -n "${EVAL_START_FRAC:-}" ]]; then
  ARGS+=(--eval-start-frac "$EVAL_START_FRAC")
fi
if [[ "${EMIT_STAGE_TIMINGS:-0}" == "1" ]]; then
  ARGS+=(--emit-stage-timings)
fi
if [[ -n "${CERTIFIED_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${CERTIFIED_EXTRA_ARGS}"
  for arg in "${EXTRA_ARGS[@]}"; do
    ARGS+=("--certified-extra-arg=$arg")
  done
fi
python3 oracle/pg19_smoke.py "${ARGS[@]}"
