#!/usr/bin/env bash
#
# Negative coverage for CUDA v1 on NVIDIA sm86.
# Verifies unsupported flag combinations and failure modes stay explicit.
#
# Usage:
#   ./tests/sm86/run_negative.sh [model_dir_0_8b] [model_dir_4b]
#
set -euo pipefail

MODEL_DIR_0_8B="${1:-${SUPERSONIC_MODEL_DIR:-}}"
MODEL_DIR_4B="${2:-${SUPERSONIC_MODEL_DIR_4B:-}}"
if [ -z "$MODEL_DIR_0_8B" ] || [ -z "$MODEL_DIR_4B" ]; then
    echo "Usage: $0 <path-to-Qwen3.5-0.8B> <path-to-Qwen3.5-4B>"
    echo "  or set SUPERSONIC_MODEL_DIR and SUPERSONIC_MODEL_DIR_4B"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f /root/.cargo/env ]; then
    . /root/.cargo/env
fi
TIMEOUT="${TIMEOUT:-120}"
TMPOUT=$(mktemp)
trap "rm -f $TMPOUT" EXIT

echo "=== SuperSonic Negative CUDA Tests: sm86 ==="
echo ""

echo "--- Building (release) ---"
SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}" cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic 2>&1
echo ""

SUPERSONIC="$REPO_ROOT/target/release/supersonic"

run_fail() {
    local label="$1"
    local expected="$2"
    shift 2
    echo "--- $label ---"
    set +e
    timeout "$TIMEOUT" "$SUPERSONIC" "$@" > "$TMPOUT" 2>&1 </dev/null
    local status=$?
    set -e
    cat "$TMPOUT"
    if [ "$status" -eq 0 ]; then
        echo ""
        echo "FAIL: command unexpectedly succeeded"
        return 1
    fi
    if ! grep -Fq -- "$expected" "$TMPOUT"; then
        echo ""
        echo "FAIL: expected error substring not found: $expected"
        return 1
    fi
    echo ""
}

COMMON_0_8B=(
    --backend cuda
    --model qwen3.5-0.8b
    --model-dir "$MODEL_DIR_0_8B"
    --prompt "Hello"
    --max-new-tokens 1
)

COMMON_4B=(
    --backend cuda
    --model qwen3.5-4b
    --model-dir "$MODEL_DIR_4B"
    --prompt "Hello"
    --max-new-tokens 1
)

COMMON_QWEN36_27B=(
    --backend cuda
    --model qwen3.6-27b
    --model-dir "$MODEL_DIR_4B"
    --prompt "Hello"
    --max-new-tokens 1
)

run_fail "Test 1: CUDA rejects --int4 outside Qwen3.5" \
    "CUDA --int4 currently supports only Qwen3.5 on sm86" \
    "${COMMON_QWEN36_27B[@]}" --int4

run_fail "Test 2: CUDA rejects --kv-fp8 outside Qwen3.5" \
    "CUDA --kv-fp8 currently supports only Qwen3.5 on sm86" \
    "${COMMON_QWEN36_27B[@]}" --kv-fp8

run_fail "Test 3: CUDA rejects out-of-range batch size" \
    "--batch-size must be 1.." \
    "${COMMON_4B[@]}" --int4 --batch-size 0

run_fail "Test 4: unknown CUDA override stays explicit on unsupported model" \
    "--allow-untested-gpu=sm999: no registry entry for model=gemma4-e2b backend=CUDA arch=sm999" \
    --backend cuda \
    --allow-untested-gpu sm999 \
    --model gemma4-e2b \
    --model-dir "$MODEL_DIR_4B" \
    --prompt "Hello" \
    --max-new-tokens 1

COMMON_GEMMA_E2B=(
    --backend cuda
    --model gemma4-e2b
    --model-dir "$MODEL_DIR_4B"
    --prompt "Hello"
    --max-new-tokens 1
)

run_fail "Test 5: Gemma CUDA rejects INT4" \
    "Gemma 4 CUDA v1 supports BF16 only; --int4 is not wired" \
    "${COMMON_GEMMA_E2B[@]}" --int4

run_fail "Test 6: Gemma CUDA rejects FP8 runtime" \
    "Gemma 4 CUDA v1 supports BF16 only; --fp8-runtime is not wired" \
    "${COMMON_GEMMA_E2B[@]}" --fp8-runtime

run_fail "Test 7: Gemma CUDA rejects KV-FP8" \
    "Gemma 4 CUDA v1 supports BF16 KV only; --kv-fp8 is not wired" \
    "${COMMON_GEMMA_E2B[@]}" --kv-fp8

run_fail "Test 8: Gemma CUDA rejects batch size > 1" \
    "Gemma 4 CUDA v1 supports only --batch-size=1" \
    "${COMMON_GEMMA_E2B[@]}" --batch-size 2

run_fail "Test 9: Gemma E4B has no CUDA v1 registry entry" \
    "No optimized kernel for model=gemma4-e4b backend=CUDA arch=sm86" \
    --backend cuda \
    --model gemma4-e4b \
    --model-dir "$MODEL_DIR_4B" \
    --prompt "Hello" \
    --max-new-tokens 1

echo "=== All negative CUDA tests passed ==="
