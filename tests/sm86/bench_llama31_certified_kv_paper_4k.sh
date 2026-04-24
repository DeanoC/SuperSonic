#!/usr/bin/env bash
#
# One-context 4K certified KV paper-metric benchmark.
#
# Runs:
#   - PG-19 teacher-forced perplexity, one chunk by default.
#   - arxiv_v1/RULER generated QA, one sample for each paper subtask by default.
#
# Environment overrides:
#   CONTEXT=4096
#   PG19_CHUNKS=1
#   PG19_EVAL_START_FRAC=0.9375
#   RULER_SAMPLES=1
#   SUBTASKS="niah_single niah_multikey niah_multivalue niah_multiquery vt cwe fwe"
#   SOURCE_TEXT=/path/to/text.txt
#   OUTPUT=target/certified_kv_paper_4k.json
#   TIMEOUT=7200
#   NO_FAIL_GATES=0
set -euo pipefail

MODEL_DIR="${1:-${SUPERSONIC_MODEL_DIR_LLAMA31_8B:-${SUPERSONIC_MODEL_DIR_LLAMA31:-}}}"
if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <path-to-Meta-Llama-3.1-8B>"
    echo "  or set SUPERSONIC_MODEL_DIR_LLAMA31_8B"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f /root/.cargo/env ]; then
    . /root/.cargo/env
fi

export SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}"

CONTEXT="${CONTEXT:-4096}"
PG19_CHUNKS="${PG19_CHUNKS:-1}"
PG19_EVAL_START_FRAC="${PG19_EVAL_START_FRAC:-0.9375}"
RULER_SAMPLES="${RULER_SAMPLES:-1}"
OUTPUT="${OUTPUT:-$REPO_ROOT/target/certified_kv_paper_4k.json}"
TIMEOUT="${TIMEOUT:-7200}"
REFERENCE_DIR="${ARXIV_V1_REFERENCE_DIR:-/workspace/DotCache/benchmarks/results/arxiv_v1_20260420}"
read -r -a SUBTASK_ARGS <<< "${SUBTASKS:-niah_single niah_multikey niah_multivalue niah_multiquery vt cwe fwe}"

echo "=== SuperSonic Llama 3.1 certified KV paper-metric 4K benchmark ==="
echo "Model dir:     $MODEL_DIR"
echo "Context:       $CONTEXT"
echo "PG-19 chunks:  $PG19_CHUNKS"
echo "PG-19 eval:    certified tail starts at ${PG19_EVAL_START_FRAC} * context"
echo "RULER samples: $RULER_SAMPLES"
echo "Subtasks:      ${SUBTASK_ARGS[*]}"
echo "Reference dir: $REFERENCE_DIR"
echo "Output:        $OUTPUT"
echo ""

cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic

EXTRA_ARGS=()
if [ -n "${SOURCE_TEXT:-}" ]; then
    EXTRA_ARGS+=(--source-text "$SOURCE_TEXT")
fi
if [ "${NO_FAIL_GATES:-0}" = "1" ]; then
    EXTRA_ARGS+=(--no-fail-gates)
fi

python3 "$REPO_ROOT/oracle/certified_kv_paper_bench.py" \
    --binary "$REPO_ROOT/target/release/supersonic" \
    --model-dir "$MODEL_DIR" \
    --context "$CONTEXT" \
    --pg19-chunks "$PG19_CHUNKS" \
    --pg19-eval-start-frac "$PG19_EVAL_START_FRAC" \
    --ruler-samples "$RULER_SAMPLES" \
    --subtasks "${SUBTASK_ARGS[@]}" \
    --reference-dir "$REFERENCE_DIR" \
    --output "$OUTPUT" \
    --timeout "$TIMEOUT" \
    "${EXTRA_ARGS[@]}"
