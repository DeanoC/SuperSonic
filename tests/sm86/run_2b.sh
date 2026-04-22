#!/usr/bin/env bash
#
# E2E validation test for SuperSonic on NVIDIA sm86 (RTX 3090-class)
# Model: Qwen3.5-2B
#
# Usage:
#   ./tests/sm86/run_2b.sh [model_dir]
#
set -euo pipefail

MODEL_DIR="${1:-${SUPERSONIC_MODEL_DIR_2B:-}}"
if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <path-to-Qwen3.5-2B>"
    echo "  or set SUPERSONIC_MODEL_DIR_2B environment variable"
    exit 1
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: $MODEL_DIR/config.json not found — is this a valid model directory?"
    exit 1
fi

HAS_SAFETENSORS=0
if find "$MODEL_DIR" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.safetensors.index.json" \) | grep -q .; then
    HAS_SAFETENSORS=1
fi
HAS_BAKE=0
if [ -f "$MODEL_DIR/.supersonic/v1/manifest.json" ]; then
    HAS_BAKE=1
fi
if [ "$HAS_SAFETENSORS" -eq 0 ] && [ "$HAS_BAKE" -eq 0 ]; then
    echo "ERROR: $MODEL_DIR has config/tokenizer files but no raw safetensors and no BF16 bake."
    echo "Need one of:"
    echo "  - local HuggingFace safetensors under $MODEL_DIR"
    echo "  - a published/local BF16 bake under $MODEL_DIR/.supersonic/v1/"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f /root/.cargo/env ]; then
    . /root/.cargo/env
fi
MAX_DELTA_THRESHOLD="${MAX_DELTA_THRESHOLD:-1.0}"
GPU_VALIDATE_DELTA_THRESHOLD="${GPU_VALIDATE_DELTA_THRESHOLD:-0.5}"
TIMEOUT="${TIMEOUT:-600}"
PROMPT="Hello"
MAX_NEW_TOKENS=4
TMPOUT=$(mktemp)
trap "rm -f $TMPOUT" EXIT

echo "=== SuperSonic E2E Test: sm86 / qwen3.5-2b ==="
echo "Model dir:  $MODEL_DIR"
echo "Threshold:  $MAX_DELTA_THRESHOLD"
echo "GPU validate threshold:  $GPU_VALIDATE_DELTA_THRESHOLD"
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
        --model qwen3.5-2b \
        --model-dir "$MODEL_DIR" \
        --prompt "$PROMPT" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
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

run_gpu_validate_test() {
    local label="$1"
    shift
    echo "--- $label ---"
    if ! timeout "$TIMEOUT" "$SUPERSONIC" \
        --backend cuda \
        --model qwen3.5-2b \
        --model-dir "$MODEL_DIR" \
        --prompt "2 + 2 =" \
        --max-new-tokens 4 \
        --gpu-validate \
        "$@" \
        > "$TMPOUT" 2>&1 </dev/null; then
        cat "$TMPOUT"
        echo ""
        echo "FAIL: process exited non-zero or timed out (${TIMEOUT}s limit)"
        return 1
    fi
    cat "$TMPOUT"

    DELTA=$(grep -oP 'gpu_oracle_max_delta=\K[0-9.]+' "$TMPOUT" || echo "MISSING")
    if [ "$DELTA" = "MISSING" ]; then
        echo "FAIL: could not extract gpu_oracle_max_delta from output"
        return 1
    fi
    if grep -q 'MISMATCH' "$TMPOUT"; then
        echo "FAIL: gpu-validate reported token mismatch"
        return 1
    fi
    echo ""
    echo "GPU validate delta ($label): $DELTA"
    PASS=$(python3 -c "print('PASS' if float('$DELTA') <= float('$GPU_VALIDATE_DELTA_THRESHOLD') else 'FAIL')")
    echo "Result: $PASS (threshold=$GPU_VALIDATE_DELTA_THRESHOLD)"
    if [ "$PASS" != "PASS" ]; then
        return 1
    fi
    echo ""
}

run_test "Test 1: Baked path"
if [ "$HAS_SAFETENSORS" -eq 1 ]; then
    run_test "Test 2: Safetensors path (--no-bake)" --no-bake
else
    echo "--- Skipping safetensors path (--no-bake): no raw safetensors in $MODEL_DIR ---"
fi
run_gpu_validate_test "Test 3: GPU validate replay path"
if [ "$HAS_SAFETENSORS" -eq 1 ]; then
    run_gpu_validate_test "Test 4: GPU validate replay path (--no-bake)" --no-bake
else
    echo "--- Skipping GPU validate (--no-bake): no raw safetensors in $MODEL_DIR ---"
fi

GOLDEN="$REPO_ROOT/tests/corpus/golden_2b.json"
GOLDEN_MODEL_ID=""
GOLDEN_MODEL_MATCH=0
if [ -f "$GOLDEN" ]; then
    GOLDEN_MODEL_ID="$(python3 -c "import json; print(json.load(open('$GOLDEN')).get('model_id', ''))")"
    GOLDEN_MODEL_BASENAME="$(basename "$GOLDEN_MODEL_ID")"
    if [ "$GOLDEN_MODEL_ID" = "Qwen/Qwen3.5-2B" ] || [ "$GOLDEN_MODEL_BASENAME" = "Qwen3.5-2B" ]; then
        GOLDEN_MODEL_MATCH=1
    fi
fi
if [ -f "$GOLDEN" ] && [ "$GOLDEN_MODEL_MATCH" -eq 1 ]; then
    CORPUS_TIMEOUT="${CORPUS_TIMEOUT:-600}" \
        "$REPO_ROOT/tests/corpus/run_golden.sh" \
        qwen3.5-2b "$MODEL_DIR" "$GOLDEN" "$SUPERSONIC" \
        --backend cuda --oracle-device cuda:0
elif [ -f "$GOLDEN" ]; then
    echo "--- Skipping golden corpus: $GOLDEN targets $GOLDEN_MODEL_ID, not Qwen/Qwen3.5-2B ---"
else
    echo "--- Skipping golden corpus (not found: $GOLDEN) ---"
fi

if [ -f "$GOLDEN" ] && [ "$GOLDEN_MODEL_MATCH" -eq 1 ] && [ "$HAS_SAFETENSORS" -eq 1 ]; then
    CORPUS_TIMEOUT="${CORPUS_TIMEOUT:-600}" \
        "$REPO_ROOT/tests/corpus/run_golden.sh" \
        qwen3.5-2b "$MODEL_DIR" "$GOLDEN" "$SUPERSONIC" \
        --backend cuda --oracle-device cuda:0 --no-bake
fi

echo "=== All tests passed ==="
