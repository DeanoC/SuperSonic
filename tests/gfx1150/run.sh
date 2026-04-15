#!/usr/bin/env bash
#
# E2E validation test for SuperSonic on gfx1150 (RDNA 3.5)
# Runs Qwen3.5-0.8B decode with oracle validation and checks max delta.
#
# Prerequisites:
#   - ROCm/HIP runtime installed
#   - Python 3 with torch + transformers (for oracle)
#   - Model weights downloaded (set MODEL_DIR below or pass as $1)
#
# Usage:
#   ./tests/gfx1150/run.sh [model_dir]
#
set -euo pipefail

MODEL_DIR="${1:-${SUPERSONIC_MODEL_DIR:-}}"
if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <path-to-Qwen3.5-0.8B>"
    echo "  or set SUPERSONIC_MODEL_DIR environment variable"
    exit 1
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: $MODEL_DIR/config.json not found — is this a valid model directory?"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAX_DELTA_THRESHOLD="${MAX_DELTA_THRESHOLD:-1.0}"
TIMEOUT="${TIMEOUT:-120}"
PROMPT="Hello"
MAX_NEW_TOKENS=4
TMPOUT=$(mktemp)
trap "rm -f $TMPOUT" EXIT

echo "=== SuperSonic E2E Test: gfx1150 / qwen3.5-0.8b ==="
echo "Model dir:  $MODEL_DIR"
echo "Threshold:  $MAX_DELTA_THRESHOLD"
echo ""

# Build
echo "--- Building (release) ---"
cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic 2>&1
echo ""

SUPERSONIC="$REPO_ROOT/target/release/supersonic"

run_test() {
    local label="$1"
    shift
    echo "--- $label ---"
    if ! timeout "$TIMEOUT" "$SUPERSONIC" \
        --model qwen3.5-0.8b \
        --model-dir "$MODEL_DIR" \
        --prompt "$PROMPT" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --validate \
        "$@" \
        > "$TMPOUT" 2>&1 </dev/null; then
        cat "$TMPOUT"
        echo ""
        echo "FAIL: process exited non-zero or timed out (${TIMEOUT}s limit)"
        echo "      If GPU hung, you may need to reset it (e.g. reboot or rocm-smi --resetgpu)"
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

# Test 1: Baked path (auto-bakes on first run)
run_test "Test 1: Baked path"

# Test 2: Safetensors path (--no-bake)
run_test "Test 2: Safetensors path (--no-bake)" --no-bake

# Test 3: Golden corpus — baked weights
GOLDEN="$REPO_ROOT/tests/corpus/golden_0.8b.json"
if [ -f "$GOLDEN" ]; then
    "$REPO_ROOT/tests/corpus/run_golden.sh" qwen3.5-0.8b "$MODEL_DIR" "$GOLDEN" "$SUPERSONIC"
else
    echo "--- Skipping golden corpus (not found: $GOLDEN) ---"
fi

# Test 4: Golden corpus — raw safetensors (--no-bake)
if [ -f "$GOLDEN" ]; then
    "$REPO_ROOT/tests/corpus/run_golden.sh" qwen3.5-0.8b "$MODEL_DIR" "$GOLDEN" "$SUPERSONIC" --no-bake
fi

echo "=== All tests passed ==="
