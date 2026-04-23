#!/usr/bin/env bash
#
# E2E validation test for SuperSonic on NVIDIA sm86 (RTX 3090-class)
# Model: Meta-Llama-3.1-8B BF16 raw safetensors
#
# Usage:
#   ./tests/sm86/run_llama31_8b.sh [model_dir]
#
set -euo pipefail

MODEL_DIR="${1:-${SUPERSONIC_MODEL_DIR_LLAMA31_8B:-}}"
if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <path-to-Meta-Llama-3.1-8B>"
    echo "  or set SUPERSONIC_MODEL_DIR_LLAMA31_8B environment variable"
    exit 1
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: $MODEL_DIR/config.json not found — is this a valid model directory?"
    exit 1
fi

if ! find "$MODEL_DIR" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.safetensors.index.json" \) | grep -q .; then
    echo "ERROR: $MODEL_DIR has config/tokenizer files but no raw safetensors."
    echo "Need local Hugging Face safetensors under $MODEL_DIR"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f /root/.cargo/env ]; then
    . /root/.cargo/env
fi
MAX_DELTA_THRESHOLD="${MAX_DELTA_THRESHOLD:-1.0}"
ORACLE_DEVICE="${ORACLE_DEVICE:-cuda:0}"
TIMEOUT="${TIMEOUT:-900}"
PROMPT="${PROMPT:-Hello}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
TMPOUT=$(mktemp)
trap "rm -f $TMPOUT" EXIT

echo "=== SuperSonic E2E Test: sm86 / llama3.1-8b ==="
echo "Model dir:      $MODEL_DIR"
echo "Prompt:         $PROMPT"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Threshold:      $MAX_DELTA_THRESHOLD"
echo "Oracle device:  $ORACLE_DEVICE"
echo ""

echo "--- Building (release) ---"
SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}" cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic 2>&1
echo ""

SUPERSONIC="$REPO_ROOT/target/release/supersonic"

echo "--- Test 1: Raw safetensors + oracle validate ---"
if ! timeout "$TIMEOUT" "$SUPERSONIC" \
    --backend cuda \
    --oracle-device "$ORACLE_DEVICE" \
    --model llama3.1-8b \
    --model-dir "$MODEL_DIR" \
    --prompt "$PROMPT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --validate \
    > "$TMPOUT" 2>&1 </dev/null; then
    cat "$TMPOUT"
    echo ""
    echo "FAIL: process exited non-zero or timed out (${TIMEOUT}s limit)"
    exit 1
fi

cat "$TMPOUT"

DELTA=$(grep -oP '\[validate\] max_delta=\K[0-9.]+' "$TMPOUT" || echo "MISSING")
if [ "$DELTA" = "MISSING" ]; then
    echo "FAIL: could not extract [validate] max_delta from output"
    exit 1
fi
if grep -q 'MISMATCH' "$TMPOUT"; then
    echo "FAIL: validate reported token mismatch"
    exit 1
fi

echo ""
echo "Max delta: $DELTA"
PASS=$(python3 -c "print('PASS' if float('$DELTA') <= float('$MAX_DELTA_THRESHOLD') else 'FAIL')")
echo "Result: $PASS (threshold=$MAX_DELTA_THRESHOLD)"
if [ "$PASS" != "PASS" ]; then
    exit 1
fi

echo ""
echo "=== All tests passed ==="
