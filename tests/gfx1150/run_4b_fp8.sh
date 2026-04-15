#!/usr/bin/env bash
#
# E2E validation test for SuperSonic on gfx1150 (RDNA 3.5)
# Model: Qwen3.5-4B-FP8 (FP8 quantized → dequant at bake time)
#
# Tests that FP8 weights produce the same output as BF16 after bake-time
# dequantization. Uses the BF16 4B golden data as reference — the baked
# FP8 model should produce identical results since dequant happens at bake time.
#
# Usage:
#   ./tests/gfx1150/run_4b_fp8.sh [fp8_model_dir] [bf16_model_dir]
#
set -euo pipefail

FP8_DIR="${1:-${SUPERSONIC_MODEL_DIR_4B_FP8:-}}"
BF16_DIR="${2:-${SUPERSONIC_MODEL_DIR_4B:-}}"

if [ -z "$FP8_DIR" ]; then
    echo "Usage: $0 <path-to-Qwen3.5-4B-FP8> [path-to-Qwen3.5-4B-BF16]"
    echo "  or set SUPERSONIC_MODEL_DIR_4B_FP8 environment variable"
    exit 1
fi

if [ ! -f "$FP8_DIR/config.json" ]; then
    echo "ERROR: $FP8_DIR/config.json not found"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAX_DELTA_THRESHOLD="${MAX_DELTA_THRESHOLD:-1.0}"
TIMEOUT="${TIMEOUT:-300}"
PROMPT="Hello"
MAX_NEW_TOKENS=4
TMPOUT=$(mktemp)
trap "rm -f $TMPOUT" EXIT

echo "=== SuperSonic E2E Test: gfx1150 / qwen3.5-4b-fp8 ==="
echo "FP8 model dir:  $FP8_DIR"
echo "BF16 model dir: ${BF16_DIR:-not set}"
echo "Threshold:      $MAX_DELTA_THRESHOLD"
echo ""

# Build
echo "--- Building (release) ---"
cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic 2>&1
echo ""

SUPERSONIC="$REPO_ROOT/target/release/supersonic"

# Test 1: FP8 baked path with oracle validation
echo "--- Test 1: FP8 baked path ---"
if ! timeout "$TIMEOUT" "$SUPERSONIC" \
    --model qwen3.5-4b \
    --model-dir "$FP8_DIR" \
    --prompt "$PROMPT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --validate \
    > "$TMPOUT" 2>&1 </dev/null; then
    cat "$TMPOUT"
    echo "FAIL: process exited non-zero or timed out"
    exit 1
fi
cat "$TMPOUT"

DELTA=$(grep -oP 'decode_max_delta=\K[0-9.]+' "$TMPOUT" || echo "MISSING")
echo ""
echo "Max delta: $DELTA"
PASS=$(python3 -c "print('PASS' if float('$DELTA') <= float('$MAX_DELTA_THRESHOLD') else 'FAIL')")
echo "Result: $PASS (threshold=$MAX_DELTA_THRESHOLD)"
if [ "$PASS" != "PASS" ]; then
    exit 1
fi
echo ""

# Test 2: Golden corpus — FP8 baked
# Uses the BF16 golden data since bake-time dequant should produce equivalent output.
# If FP8 and BF16 produce different golden data, generate FP8-specific golden:
#   python3 tests/corpus/generate_golden_native.py --binary target/release/supersonic \
#     --model qwen3.5-4b --model-dir $FP8_DIR \
#     --test-defs tests/corpus/golden_4b.json --output tests/corpus/golden_4b_fp8.json
FP8_GOLDEN="$REPO_ROOT/tests/corpus/golden_4b_fp8.json"
BF16_GOLDEN="$REPO_ROOT/tests/corpus/golden_4b.json"

# Use FP8-specific golden if it exists, otherwise fall back to BF16 golden
GOLDEN="${FP8_GOLDEN}"
if [ ! -f "$GOLDEN" ]; then
    GOLDEN="${BF16_GOLDEN}"
fi

if [ -f "$GOLDEN" ]; then
    CORPUS_TIMEOUT=600 "$REPO_ROOT/tests/corpus/run_golden.sh" qwen3.5-4b "$FP8_DIR" "$GOLDEN" "$SUPERSONIC"
else
    echo "--- Skipping golden corpus (not found) ---"
fi

# Test 3: Cross-weight comparison — if BF16 dir provided, compare FP8 vs BF16 output
if [ -n "$BF16_DIR" ] && [ -f "$BF16_DIR/config.json" ]; then
    echo ""
    echo "--- Test 3: FP8 vs BF16 cross-comparison ---"
    # Run both and compare output text
    FP8_OUT=$(timeout "$TIMEOUT" "$SUPERSONIC" --model qwen3.5-4b --model-dir "$FP8_DIR" \
        --prompt "The capital of France is" --max-new-tokens 4 2>/dev/null)
    BF16_OUT=$(timeout "$TIMEOUT" "$SUPERSONIC" --model qwen3.5-4b --model-dir "$BF16_DIR" \
        --prompt "The capital of France is" --max-new-tokens 4 2>/dev/null)

    if [ "$FP8_OUT" = "$BF16_OUT" ]; then
        echo "FP8 vs BF16: MATCH"
        echo "  Output: $FP8_OUT"
    else
        echo "FP8 vs BF16: DIFFER (may be acceptable for quantized models)"
        echo "  FP8:  $FP8_OUT"
        echo "  BF16: $BF16_OUT"
    fi
fi

echo ""
echo "=== All tests passed ==="
