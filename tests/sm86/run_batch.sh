#!/usr/bin/env bash
#
# E2E validation test for batched decode on NVIDIA sm86 (RTX 3090-class)
# Model: Qwen3.5-4B
#
# Usage:
#   ./tests/sm86/run_batch.sh [model_dir]
#
set -euo pipefail

MODEL_DIR="${1:-${SUPERSONIC_MODEL_DIR_4B:-}}"
if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <path-to-Qwen3.5-4B>"
    echo "  or set SUPERSONIC_MODEL_DIR_4B environment variable"
    exit 1
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: $MODEL_DIR/config.json not found — is this a valid model directory?"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f /root/.cargo/env ]; then
    . /root/.cargo/env
fi
MAX_DELTA_THRESHOLD="${MAX_DELTA_THRESHOLD:-1.0}"
TIMEOUT="${TIMEOUT:-300}"
CORPUS_TIMEOUT="${CORPUS_TIMEOUT:-600}"
TMPOUT=$(mktemp)
trap "rm -f $TMPOUT" EXIT

echo "=== SuperSonic Batch Decode Test: sm86 / qwen3.5-4b ==="
echo "Model dir:  $MODEL_DIR"
echo "Batch size: 2"
echo "Threshold:  $MAX_DELTA_THRESHOLD"
echo ""

echo "--- Building (release) ---"
SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}" cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic 2>&1
echo ""

SUPERSONIC="$REPO_ROOT/target/release/supersonic"

run_test() {
    local label="$1"
    shift
    echo "--- $label ---"
    if ! timeout "$TIMEOUT" "$SUPERSONIC" \
        --backend cuda \
        --oracle-device cuda:0 \
        --model qwen3.5-4b \
        --model-dir "$MODEL_DIR" \
        --prompt "Hello" \
        --max-new-tokens 4 \
        --batch-size 2 \
        --validate \
        "$@" \
        > "$TMPOUT" 2>&1 </dev/null; then
        cat "$TMPOUT"
        echo ""
        echo "FAIL: process exited non-zero or timed out (${TIMEOUT}s limit)"
        return 1
    fi
    cat "$TMPOUT"

    DELTA=$(grep -oP 'decode_max_delta=\K[0-9.]+' "$TMPOUT" || echo "MISSING")
    if [ "$DELTA" = "MISSING" ]; then
        echo "FAIL: could not extract decode_max_delta from output"
        return 1
    fi
    echo ""
    echo "Max delta ($label): $DELTA"
    PASS=$(python3 -c "print('PASS' if float('$DELTA') <= float('$MAX_DELTA_THRESHOLD') else 'FAIL')")
    echo "Result: $PASS (threshold=$MAX_DELTA_THRESHOLD)"
    if [ "$PASS" != "PASS" ]; then
        return 1
    fi
    echo ""
}

run_test "Test 1: Baked path"
run_test "Test 2: Safetensors path (--no-bake)" --no-bake

GOLDEN="$REPO_ROOT/tests/corpus/golden_4b_batch2.json"
if [ -f "$GOLDEN" ]; then
    CORPUS_TIMEOUT="$CORPUS_TIMEOUT" \
        "$REPO_ROOT/tests/corpus/run_golden.sh" \
        qwen3.5-4b "$MODEL_DIR" "$GOLDEN" "$SUPERSONIC" \
        --backend cuda --oracle-device cuda:0 --batch-size 2
else
    echo "--- Skipping batch golden corpus (not found: $GOLDEN) ---"
fi

if [ -f "$GOLDEN" ]; then
    CORPUS_TIMEOUT="$CORPUS_TIMEOUT" \
        "$REPO_ROOT/tests/corpus/run_golden.sh" \
        qwen3.5-4b "$MODEL_DIR" "$GOLDEN" "$SUPERSONIC" \
        --backend cuda --oracle-device cuda:0 --batch-size 2 --no-bake
fi

echo "=== All batch tests passed ==="
