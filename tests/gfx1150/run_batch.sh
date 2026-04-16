#!/usr/bin/env bash
#
# E2E validation tests for batched decode on gfx1150 (RDNA 3.5)
# Tests --batch-size 2 on all model variants that use the 4B kernel.
#
# Usage:
#   ./tests/gfx1150/run_batch.sh [model_dir_2b] [model_dir_4b] [model_dir_4b_fp8] [model_dir_9b_fp8]
#
# Or set environment variables:
#   SUPERSONIC_MODEL_DIR_2B, SUPERSONIC_MODEL_DIR_4B,
#   SUPERSONIC_MODEL_DIR_4B_FP8, SUPERSONIC_MODEL_DIR_9B_FP8
#
set -euo pipefail

DIR_2B="${1:-${SUPERSONIC_MODEL_DIR_2B:-}}"
DIR_4B="${2:-${SUPERSONIC_MODEL_DIR_4B:-}}"
DIR_4B_FP8="${3:-${SUPERSONIC_MODEL_DIR_4B_FP8:-}}"
DIR_9B_FP8="${4:-${SUPERSONIC_MODEL_DIR_9B_FP8:-}}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TIMEOUT="${TIMEOUT:-300}"
TMPOUT=$(mktemp)
trap "rm -f $TMPOUT" EXIT

echo "=== SuperSonic Batch Decode Tests: gfx1150 ==="
echo "Batch size: 2"
echo ""

# Build
echo "--- Building (release) ---"
cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic 2>&1
echo ""

SUPERSONIC="$REPO_ROOT/target/release/supersonic"
TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_SKIP=0

run_batch_golden() {
    local label="$1"
    local model="$2"
    local model_dir="$3"
    local golden="$4"
    shift 4
    local extra_flags=("$@")

    if [ -z "$model_dir" ] || [ ! -f "$model_dir/config.json" ]; then
        echo "--- $label: SKIP (model dir not found) ---"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        return 0
    fi
    if [ ! -f "$golden" ]; then
        echo "--- $label: SKIP (golden file not found: $golden) ---"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        return 0
    fi

    echo "--- $label ---"

    # Quick smoke test first
    printf "  smoke test (Hello, 4 tokens)... "
    if ! timeout "$TIMEOUT" "$SUPERSONIC" \
        --model "$model" --model-dir "$model_dir" \
        --prompt "Hello" --max-new-tokens 4 \
        --batch-size 2 "${extra_flags[@]}" \
        > "$TMPOUT" 2>&1 </dev/null; then
        echo "FAIL (crash)"
        cat "$TMPOUT" | tail -3 | sed 's/^/    /'
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        return 1
    fi
    echo "PASS"

    # Golden corpus
    if CORPUS_TIMEOUT=600 "$REPO_ROOT/tests/corpus/run_golden.sh" \
        "$model" "$model_dir" "$golden" "$SUPERSONIC" --batch-size 2 "${extra_flags[@]}"; then
        TOTAL_PASS=$((TOTAL_PASS + 1))
    else
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        return 1
    fi
    echo ""
}

# 2B BF16
run_batch_golden "2B BF16 batch=2" qwen3.5-2b "$DIR_2B" \
    "$REPO_ROOT/tests/corpus/golden_2b_batch2.json"

# 4B BF16
run_batch_golden "4B BF16 batch=2" qwen3.5-4b "$DIR_4B" \
    "$REPO_ROOT/tests/corpus/golden_4b_batch2.json"

# 4B FP8 runtime
run_batch_golden "4B FP8 runtime batch=2" qwen3.5-4b "$DIR_4B_FP8" \
    "$REPO_ROOT/tests/corpus/golden_4b_fp8_runtime_batch2.json" --fp8-runtime

# 9B FP8 runtime
run_batch_golden "9B FP8 runtime batch=2" qwen3.5-9b "$DIR_9B_FP8" \
    "$REPO_ROOT/tests/corpus/golden_9b_fp8_runtime_batch2.json" --fp8-runtime

echo ""
echo "=== Batch decode results: $TOTAL_PASS passed, $TOTAL_FAIL failed, $TOTAL_SKIP skipped ==="

if [ "$TOTAL_FAIL" -gt 0 ]; then
    exit 1
fi
if [ "$TOTAL_PASS" -eq 0 ]; then
    echo "WARNING: no tests ran (set model dir env vars or pass paths)"
    exit 1
fi
