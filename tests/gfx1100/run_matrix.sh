#!/usr/bin/env bash
#
# Smoke-survey of the HIP supported matrix on gfx1100 (AMD RX 7900 XTX, 24 GiB).
# Mirrors tests/gfx1150/run_quant_regression.sh in spirit: short prompt, fixed
# token count, pass/fail by token sequence — but no oracle dependency since
# decode parity has already been validated against the BF16 oracle when bringing
# up each variant.
#
# Set per-model dirs via env vars or skip the section. Honors the same
# SUPERSONIC_MODEL_DIR_* names used by the gfx1150 scripts.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUPERSONIC="$REPO_ROOT/target/release/supersonic"
PROMPT="The quick brown fox"
MAX_NEW=6
TIMEOUT="${TIMEOUT:-180}"

MODEL_DIR_08B="${MODEL_DIR_08B:-${SUPERSONIC_MODEL_DIR_08B:-${SUPERSONIC_MODEL_DIR:-}}}"
MODEL_DIR_2B="${MODEL_DIR_2B:-${SUPERSONIC_MODEL_DIR_2B:-}}"
MODEL_DIR_4B="${MODEL_DIR_4B:-${SUPERSONIC_MODEL_DIR_4B:-}}"
MODEL_DIR_9B="${MODEL_DIR_9B:-${SUPERSONIC_MODEL_DIR_9B:-}}"
MODEL_DIR_GEMMA_E2B="${MODEL_DIR_GEMMA_E2B:-${SUPERSONIC_MODEL_DIR_GEMMA_E2B:-}}"
MODEL_DIR_GEMMA_E4B="${MODEL_DIR_GEMMA_E4B:-${SUPERSONIC_MODEL_DIR_GEMMA_E4B:-}}"
MODEL_DIR_PHI4="${MODEL_DIR_PHI4:-${SUPERSONIC_MODEL_DIR_PHI4:-}}"

if [ ! -x "$SUPERSONIC" ]; then
    echo "ERROR: $SUPERSONIC not found. Run: cargo build --release"
    exit 1
fi

PASSED=0
FAILED=0
SKIPPED=0

# check_case <label> <model> <model_dir> <expected_tokens> <flags...>
check_case() {
    local label="$1"; shift
    local model="$1"; shift
    local model_dir="$1"; shift
    local expected="$1"; shift
    local flags=("$@")

    if [ -z "$model_dir" ] || [ ! -f "$model_dir/config.json" ]; then
        printf "  %-32s SKIP (model dir missing)\n" "$label"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    local out
    if ! out=$(timeout "$TIMEOUT" "$SUPERSONIC" \
        --model "$model" --model-dir "$model_dir" \
        --prompt "$PROMPT" --max-new-tokens "$MAX_NEW" \
        "${flags[@]}" 2>&1); then
        printf "  %-32s FAIL (binary exited non-zero / GPU fault)\n" "$label"
        FAILED=$((FAILED + 1))
        return
    fi

    local tokens
    tokens=$(echo "$out" | grep '^\[tokens\]' | sed 's/^\[tokens\] //' || true)
    if [ "$tokens" = "$expected" ]; then
        printf "  %-32s PASS\n" "$label"
        PASSED=$((PASSED + 1))
    else
        printf "  %-32s FAIL\n" "$label"
        echo "    expected: $expected"
        echo "    got:      $tokens"
        FAILED=$((FAILED + 1))
    fi
}

echo "=== SuperSonic gfx1100 supported-matrix smoke (RX 7900 XTX) ==="
echo "Prompt:        \"$PROMPT\""
echo "Max new tokens: $MAX_NEW"
echo ""

EXPECTED_QWEN_FOX="33075 888 279 15217 5388 13"
# 0.8B argmax drifts on marginal flips ("lazy dog" vs "green hedge"). gfx1100
# produces "green hedge" tokens for the same prompt; this matches gpu_oracle
# internal parity and is within the BF16 oracle threshold (delta 0.28 < 1.0).
EXPECTED_QWEN_08B_FOX="33075 888 279 5983 40289 13"
EXPECTED_QWEN_08B_INT4="369 264 1865 9572 303 279"      # gfx1150 reference; gfx1100 INT4 currently faults
EXPECTED_QWEN_2B_INT4="369 264 42140 3542 494 279"
EXPECTED_GEMMA_FOX="38167 1024 506 31770 4799 236761"

echo "--- Qwen3.5 0.8B ---"
check_case "0.8B bf16"               qwen3.5-0.8b "$MODEL_DIR_08B" "$EXPECTED_QWEN_08B_FOX"
check_case "0.8B --fp8-runtime"      qwen3.5-0.8b "$MODEL_DIR_08B" "$EXPECTED_QWEN_08B_FOX" --fp8-runtime
check_case "0.8B --kv-fp8"           qwen3.5-0.8b "$MODEL_DIR_08B" "$EXPECTED_QWEN_08B_FOX" --kv-fp8
check_case "0.8B --int4"             qwen3.5-0.8b "$MODEL_DIR_08B" "369 264 1546 5243 321 5243" --int4

echo ""
echo "--- Qwen3.5 2B ---"
check_case "2B bf16"                 qwen3.5-2b   "$MODEL_DIR_2B"  "$EXPECTED_QWEN_FOX"
check_case "2B --fp8-runtime"        qwen3.5-2b   "$MODEL_DIR_2B"  "$EXPECTED_QWEN_FOX" --fp8-runtime
check_case "2B --kv-fp8"             qwen3.5-2b   "$MODEL_DIR_2B"  "$EXPECTED_QWEN_FOX" --kv-fp8
check_case "2B --int4"               qwen3.5-2b   "$MODEL_DIR_2B"  "369 264 3542 303 279 220" --int4

echo ""
echo "--- Qwen3.5 4B ---"
check_case "4B bf16"                 qwen3.5-4b   "$MODEL_DIR_4B"  "$EXPECTED_QWEN_FOX"
check_case "4B --fp8-runtime"        qwen3.5-4b   "$MODEL_DIR_4B"  "$EXPECTED_QWEN_FOX" --fp8-runtime
check_case "4B --kv-fp8"             qwen3.5-4b   "$MODEL_DIR_4B"  "$EXPECTED_QWEN_FOX" --kv-fp8
check_case "4B --int4"               qwen3.5-4b   "$MODEL_DIR_4B"  "33075 888 279 15217 7993 13" --int4

echo ""
echo "--- Qwen3.5 9B ---"
check_case "9B bf16"                 qwen3.5-9b   "$MODEL_DIR_9B"  "$EXPECTED_QWEN_FOX"
check_case "9B --fp8-runtime"        qwen3.5-9b   "$MODEL_DIR_9B"  "$EXPECTED_QWEN_FOX" --fp8-runtime
check_case "9B --kv-fp8"             qwen3.5-9b   "$MODEL_DIR_9B"  "$EXPECTED_QWEN_FOX" --kv-fp8
check_case "9B --int4"               qwen3.5-9b   "$MODEL_DIR_9B"  "$EXPECTED_QWEN_FOX" --int4

echo ""
echo "--- Gemma 4 ---"
check_case "E2B bf16"                gemma4-e2b   "$MODEL_DIR_GEMMA_E2B" "$EXPECTED_GEMMA_FOX"
check_case "E2B --int4"              gemma4-e2b   "$MODEL_DIR_GEMMA_E2B" "563 496 3823 8864 37423 236761" --int4
check_case "E4B bf16"                gemma4-e4b   "$MODEL_DIR_GEMMA_E4B" "$EXPECTED_GEMMA_FOX"

echo ""
echo "--- Phi-4-mini ---"
check_case "phi4 bf16"               phi4-mini    "$MODEL_DIR_PHI4" "65613 1072 290 29082 6446 13"
# phi4-mini --int4 needs a local bake (oracle/bake_int4_phi4.py, ~4 min on gfx1100).
# No GitHub-release asset; once baked, the runtime --int4 path picks it up automatically.
check_case "phi4 --int4"             phi4-mini    "$MODEL_DIR_PHI4" "65613 1072 290 29082 6446 13" --int4

echo ""
echo "Summary: $PASSED passed, $FAILED failed, $SKIPPED skipped"
if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
