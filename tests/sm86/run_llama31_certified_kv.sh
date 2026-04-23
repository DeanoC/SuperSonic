#!/usr/bin/env bash
#
# E2E validation for the staged certified KV Llama 3.1 CUDA path.
# The current Rust/CUDA milestone intentionally runs the certified contract
# through unconditional dense fallback while compressed kernels are being wired.
#
# Usage:
#   ./tests/sm86/run_llama31_certified_kv.sh [model_dir]
#
set -euo pipefail

MODEL_DIR="${1:-${SUPERSONIC_MODEL_DIR_LLAMA31_8B:-}}"
if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <path-to-Meta-Llama-3.1-8B>"
    echo "  or set SUPERSONIC_MODEL_DIR_LLAMA31_8B"
    exit 1
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: $MODEL_DIR/config.json not found"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f /root/.cargo/env ]; then
    . /root/.cargo/env
fi

MAX_DELTA_THRESHOLD="${MAX_DELTA_THRESHOLD:-2.0}"
ORACLE_DEVICE="${ORACLE_DEVICE:-cuda:0}"
TIMEOUT="${TIMEOUT:-1200}"
PROMPT="${PROMPT:-Hello}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
CERTIFIED_KV_SHADOW_VALIDATE="${CERTIFIED_KV_SHADOW_VALIDATE:-0}"
if [ -n "${CERTIFIED_KV_TELEMETRY:-}" ]; then
    TELEMETRY="$CERTIFIED_KV_TELEMETRY"
    TELEMETRY_CREATED=0
else
    TELEMETRY="$(mktemp)"
    TELEMETRY_CREATED=1
fi
TMPOUT=$(mktemp)
cleanup() {
    rm -f "$TMPOUT"
    if [ "$TELEMETRY_CREATED" = "1" ]; then
        rm -f "$TELEMETRY"
    fi
}
trap cleanup EXIT

echo "=== SuperSonic E2E Test: sm86 / llama3.1-8b / certified-kv ==="
echo "Model dir:      $MODEL_DIR"
echo "Prompt:         $PROMPT"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Threshold:      $MAX_DELTA_THRESHOLD"
echo "Oracle device:  $ORACLE_DEVICE"
echo "Telemetry:      $TELEMETRY"
echo "Shadow validate:$CERTIFIED_KV_SHADOW_VALIDATE"
echo ""

echo "--- Building (release) ---"
SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}" cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic
echo ""

SUPERSONIC="$REPO_ROOT/target/release/supersonic"

echo "--- Certified KV dense-fallback Rust/CUDA validation ---"
cmd=(
    "$SUPERSONIC"
    --backend cuda
    --oracle-device "$ORACLE_DEVICE"
    --model llama3.1-8b
    --model-dir "$MODEL_DIR"
    --prompt "$PROMPT"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --int8
    --certified-kv
    --certified-kv-telemetry "$TELEMETRY"
    --validate
)
if [ "$CERTIFIED_KV_SHADOW_VALIDATE" = "1" ]; then
    cmd+=(--certified-kv-shadow-validate)
fi
if ! timeout "$TIMEOUT" "${cmd[@]}" > "$TMPOUT" 2>&1 </dev/null; then
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
if ! grep -q '"mode":"certified_kv_dense_fallback"' "$TELEMETRY"; then
    echo "FAIL: telemetry did not record certified_kv_dense_fallback"
    cat "$TELEMETRY" || true
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
echo "--- Certified KV telemetry ---"
cat "$TELEMETRY"
echo ""
echo "=== All tests passed ==="
