#!/usr/bin/env bash
#
# Quantization-path regression canary: for each (model, --int4/--kv-fp8 combo),
# generate the first 6 tokens of a known-stable prompt and compare against a
# hardcoded token-ID sequence.
#
# Why not a golden-text corpus? The full 10-prompt corpus proved flaky under
# --kv-fp8: marginal-token flips from FP8 accumulation noise cause the
# argmax to drift on ~20-30% of runs for prompts like "Hello, world" and
# the Chinese one. "The quick brown fox" is rock-stable across all four
# quant combos on both 2B and 4B.
#
# The fp8_lut-init regression (2ca5c77) made token 2 onwards = 248319
# (vocab_size-1). This canary catches that class of bug.
#
# Usage:
#   ./tests/gfx1150/run_quant_regression.sh
#     (uses $SUPERSONIC_MODEL_DIR_4B and $SUPERSONIC_MODEL_DIR_2B)
#   MODEL_DIR_4B=... MODEL_DIR_2B=... ./tests/gfx1150/run_quant_regression.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUPERSONIC="$REPO_ROOT/target/release/supersonic"
PROMPT="The quick brown fox"
MAX_NEW=6
TIMEOUT="${TIMEOUT:-120}"

MODEL_DIR_4B="${MODEL_DIR_4B:-${SUPERSONIC_MODEL_DIR_4B:-}}"
MODEL_DIR_2B="${MODEL_DIR_2B:-${SUPERSONIC_MODEL_DIR_2B:-}}"
MODEL_DIR_08B="${MODEL_DIR_08B:-${SUPERSONIC_MODEL_DIR_08B:-}}"

if [ ! -x "$SUPERSONIC" ]; then
    echo "ERROR: $SUPERSONIC not found. Run: cargo build --release"
    exit 1
fi

PASSED=0
FAILED=0

# check_case <label> <model> <model_dir> <expected_tokens> <flags...>
check_case() {
    local label="$1"; shift
    local model="$1"; shift
    local model_dir="$1"; shift
    local expected="$1"; shift
    local flags=("$@")

    if [ -z "$model_dir" ] || [ ! -f "$model_dir/config.json" ]; then
        printf "  %-30s SKIP (model dir missing)\n" "$label"
        return
    fi

    local out
    if ! out=$(timeout "$TIMEOUT" "$SUPERSONIC" \
        --model "$model" --model-dir "$model_dir" \
        --prompt "$PROMPT" --max-new-tokens "$MAX_NEW" \
        "${flags[@]}" 2>/dev/null); then
        printf "  %-30s FAIL (binary exited non-zero)\n" "$label"
        FAILED=$((FAILED + 1))
        return
    fi

    local tokens
    tokens=$(echo "$out" | grep '^\[tokens\]' | sed 's/^\[tokens\] //' || true)
    if [ "$tokens" = "$expected" ]; then
        printf "  %-30s PASS\n" "$label"
        PASSED=$((PASSED + 1))
    else
        printf "  %-30s FAIL\n" "$label"
        echo "    expected: $expected"
        echo "    got:      $tokens"
        FAILED=$((FAILED + 1))
    fi
}

echo "=== SuperSonic quant-path regression canary (gfx1150) ==="
echo "Prompt:        \"$PROMPT\""
echo "Max new tokens: $MAX_NEW"
echo ""

echo "--- 4B ---"
check_case "4B bf16"                qwen3.5-4b "$MODEL_DIR_4B" "33075 888 279 15217 5388 13"
check_case "4B --int4"              qwen3.5-4b "$MODEL_DIR_4B" "33075 888 279 15217 5388 13" --int4
check_case "4B --kv-fp8"            qwen3.5-4b "$MODEL_DIR_4B" "33075 888 279 15217 5388 13" --kv-fp8
check_case "4B --int4 --kv-fp8"     qwen3.5-4b "$MODEL_DIR_4B" "33075 888 279 15217 5388 13" --int4 --kv-fp8

echo ""
echo "--- 2B ---"
check_case "2B bf16"                qwen3.5-2b "$MODEL_DIR_2B" "33075 888 279 15217 5388 13"
check_case "2B --int4"              qwen3.5-2b "$MODEL_DIR_2B" "369 264 220 17 15 16" --int4
check_case "2B --kv-fp8"            qwen3.5-2b "$MODEL_DIR_2B" "33075 888 279 15217 5388 13" --kv-fp8
check_case "2B --int4 --kv-fp8"     qwen3.5-2b "$MODEL_DIR_2B" "369 264 220 17 15 16" --int4 --kv-fp8

echo ""
echo "--- 0.8B ---"
# 0.8B-native decode kernel has no INT4 megakernel path; --int4 forces use_4b_kernel.
# Guards against two regressions: (a) the 0.8B INT4 memory-fault bug fixed 2026-04-18
# (commit 5b27c41) and (b) any future silent fall-through where the dispatch reverts
# to the BF16-only 0.8B kernel.
check_case "0.8B --int4"            qwen3.5-0.8b "$MODEL_DIR_08B" "369 264 5243 321 5243 9572" --int4

echo ""
echo "Quant regression: $PASSED passed, $FAILED failed"
if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
