#!/usr/bin/env bash
#
# E2E validation test for Gemma 4 E2B BF16 on NVIDIA sm86.
#
# Usage:
#   ./tests/sm86/run_gemma4.sh [model_dir]
#
set -euo pipefail

MODEL_DIR="${1:-${GEMMA_E2B_DIR:-${SUPERSONIC_GEMMA_E2B_DIR:-}}}"
if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <path-to-gemma-4-E2B>"
    echo "  or set GEMMA_E2B_DIR / SUPERSONIC_GEMMA_E2B_DIR"
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

TIMEOUT="${TIMEOUT:-900}"
PROMPT="${PROMPT:-Hello, world}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
TMPOUT=$(mktemp)
trap "rm -f $TMPOUT" EXIT

echo "=== SuperSonic E2E Test: sm86 / gemma4-e2b BF16 ==="
echo "Model dir: $MODEL_DIR"
echo ""

echo "--- Building (release) ---"
SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}" cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic 2>&1
echo ""

SUPERSONIC="$REPO_ROOT/target/release/supersonic"

echo "--- Gemma 4 E2B BF16 validate ---"
if ! timeout "$TIMEOUT" "$SUPERSONIC" \
    --backend cuda \
    --model gemma4-e2b \
    --model-dir "$MODEL_DIR" \
    --prompt "$PROMPT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size 1 \
    --validate \
    > "$TMPOUT" 2>&1 </dev/null; then
    cat "$TMPOUT"
    echo ""
    echo "FAIL: process exited non-zero or timed out (${TIMEOUT}s limit)"
    exit 1
fi
cat "$TMPOUT"

if grep -Fq "MISMATCH" "$TMPOUT"; then
    echo ""
    echo "FAIL: Gemma 4 CUDA validation reported token mismatch"
    exit 1
fi

echo ""
echo "=== Gemma 4 E2B CUDA validation passed ==="
