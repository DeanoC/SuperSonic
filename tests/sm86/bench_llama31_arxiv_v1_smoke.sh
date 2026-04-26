#!/usr/bin/env bash
#
# Run a small arxiv_v1-compatible Llama 3.1 QA sweep against SuperSonic CUDA.
#
# The default lane uses the DotCache arxiv_v1 4K smoke references and runs two
# RULER retrieval subtasks with three samples each for dense INT8 and certified
# KV.  Override CONTEXTS/SUBTASKS/SAMPLES to trade runtime for coverage.
#
# Usage:
#   ./tests/sm86/bench_llama31_arxiv_v1_smoke.sh /path/to/Meta-Llama-3.1-8B
set -euo pipefail

MODEL_DIR="${1:-${SUPERSONIC_MODEL_DIR_LLAMA31_8B:-}}"
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
REFERENCE_DIR="${ARXIV_V1_REFERENCE_DIR:-/workspace/DotCache/benchmarks/results/arxiv_v1_20260420}"
CONTEXTS="${CONTEXTS:-4096}"
SAMPLES="${SAMPLES:-3}"
CONFIG="${CONFIG:-both}"
OUTPUT="${OUTPUT:-$REPO_ROOT/target/arxiv_v1_smoke.json}"
TIMEOUT="${TIMEOUT:-900}"
MIN_SCORE="${MIN_SCORE:-}"
FAIL_BELOW_REFERENCE="${FAIL_BELOW_REFERENCE:-1}"
REFERENCE_TOLERANCE="${REFERENCE_TOLERANCE:-0.0}"
FAIL_ON_CRITICAL="${FAIL_ON_CRITICAL:-1}"

read -r -a CONTEXT_ARGS <<< "$CONTEXTS"
read -r -a SUBTASK_ARGS <<< "${SUBTASKS:-niah_single niah_multikey}"

echo "=== SuperSonic sm86 Llama 3.1 arxiv_v1 smoke QA ==="
echo "Model dir:      $MODEL_DIR"
echo "Reference dir:  $REFERENCE_DIR"
echo "Contexts:       ${CONTEXT_ARGS[*]}"
echo "Subtasks:       ${SUBTASK_ARGS[*]}"
echo "Samples:        $SAMPLES"
echo "Config:         $CONFIG"
echo "Min score:      ${MIN_SCORE:-<unset>}"
echo "Ref gate:       $FAIL_BELOW_REFERENCE tolerance=$REFERENCE_TOLERANCE"
echo "Critical gate:  $FAIL_ON_CRITICAL"
echo "Output:         $OUTPUT"
echo ""

cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic

EXTRA_ARGS=()
if [ -n "$MIN_SCORE" ]; then
    EXTRA_ARGS+=(--min-score "$MIN_SCORE")
fi
if [ "$FAIL_BELOW_REFERENCE" = "1" ]; then
    EXTRA_ARGS+=(--fail-below-reference --reference-tolerance "$REFERENCE_TOLERANCE")
fi
if [ "$FAIL_ON_CRITICAL" = "1" ]; then
    EXTRA_ARGS+=(--fail-on-critical)
fi

python3 "$REPO_ROOT/oracle/arxiv_v1_smoke.py" \
    --model-dir "$MODEL_DIR" \
    --binary "$REPO_ROOT/target/release/supersonic" \
    --reference-dir "$REFERENCE_DIR" \
    --contexts "${CONTEXT_ARGS[@]}" \
    --subtasks "${SUBTASK_ARGS[@]}" \
    --samples "$SAMPLES" \
    --config "$CONFIG" \
    --timeout "$TIMEOUT" \
    --output "$OUTPUT" \
    "${EXTRA_ARGS[@]}"
