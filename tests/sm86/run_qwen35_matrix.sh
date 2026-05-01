#!/usr/bin/env bash
#
# CUDA sm86 supported-matrix smoke for Qwen3.5 low-bit modes.
#
# This is the CUDA counterpart to the Qwen rows in tests/gfx1100/run_matrix.sh.
# It intentionally skips model dirs that are not available on the machine so a
# single 3090 box can validate the subset it has downloaded.
#
# Usage:
#   SUPERSONIC_MODEL_DIR_08B=/models/Qwen3.5-0.8B \
#   SUPERSONIC_MODEL_DIR_2B=/models/Qwen3.5-2B \
#   SUPERSONIC_MODEL_DIR_4B=/models/Qwen3.5-4B \
#   SUPERSONIC_MODEL_DIR_9B=/models/Qwen3.5-9B \
#     ./tests/sm86/run_qwen35_matrix.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f /root/.cargo/env ]; then
    . /root/.cargo/env
fi

MODEL_DIR_08B="${MODEL_DIR_08B:-${SUPERSONIC_MODEL_DIR_08B:-${SUPERSONIC_MODEL_DIR:-}}}"
MODEL_DIR_2B="${MODEL_DIR_2B:-${SUPERSONIC_MODEL_DIR_2B:-}}"
MODEL_DIR_4B="${MODEL_DIR_4B:-${SUPERSONIC_MODEL_DIR_4B:-}}"
MODEL_DIR_9B="${MODEL_DIR_9B:-${SUPERSONIC_MODEL_DIR_9B:-}}"

PROMPT="${PROMPT:-Hello}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2}"
TIMEOUT="${TIMEOUT:-900}"
FETCH_TIMEOUT="${FETCH_TIMEOUT:-1800}"
ORACLE_DEVICE="${ORACLE_DEVICE:-cpu}"
MAX_DELTA_THRESHOLD="${MAX_DELTA_THRESHOLD:-2.0}"
VALIDATE="${VALIDATE:-0}"

TMPOUT=$(mktemp)
trap "rm -f $TMPOUT" EXIT

echo "=== SuperSonic sm86 Qwen3.5 CUDA matrix smoke ==="
echo "Prompt:        $PROMPT"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Oracle device: $ORACLE_DEVICE"
echo "Validate:      $VALIDATE"
echo ""

echo "--- Building (release) ---"
SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}" \
    cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic
echo ""

SUPERSONIC="$REPO_ROOT/target/release/supersonic"

PASSED=0
FAILED=0
SKIPPED=0

run_case() {
    local label="$1"; shift
    local model="$1"; shift
    local model_dir="$1"; shift
    local effective_timeout="$1"; shift
    local flags=("$@")

    if [ -z "$model_dir" ]; then
        printf "  %-36s SKIP (model dir missing)\n" "$label"
        SKIPPED=$((SKIPPED + 1))
        return
    fi
    mkdir -p "$model_dir"

    local validate_flags=()
    if [ "$VALIDATE" = "1" ]; then
        validate_flags=(--validate --oracle-device "$ORACLE_DEVICE")
    fi

    if ! timeout "$effective_timeout" "$SUPERSONIC" \
        --backend cuda \
        --model "$model" \
        --model-dir "$model_dir" \
        --prompt "$PROMPT" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        "${validate_flags[@]}" \
        "${flags[@]}" \
        > "$TMPOUT" 2>&1 </dev/null; then
        printf "  %-36s FAIL (process error / timeout)\n" "$label"
        cat "$TMPOUT"
        FAILED=$((FAILED + 1))
        return
    fi

    if [ "$VALIDATE" != "1" ]; then
        local tokens
        tokens=$(grep '^\[tokens\]' "$TMPOUT" | sed 's/^\[tokens\] //' || true)
        printf "  %-36s PASS tokens=%s\n" "$label" "${tokens:-<none>}"
        PASSED=$((PASSED + 1))
        return
    fi

    local delta
    delta=$(grep -oP 'decode_max_delta=\K[0-9.]+' "$TMPOUT" || echo "MISSING")
    if [ "$delta" = "MISSING" ]; then
        printf "  %-36s FAIL (missing decode_max_delta)\n" "$label"
        cat "$TMPOUT"
        FAILED=$((FAILED + 1))
        return
    fi

    local pass
    pass=$(python3 -c "print('PASS' if float('$delta') <= float('$MAX_DELTA_THRESHOLD') else 'FAIL')")
    if [ "$pass" = "PASS" ]; then
        printf "  %-36s PASS delta=%s\n" "$label" "$delta"
        PASSED=$((PASSED + 1))
    else
        printf "  %-36s FAIL delta=%s threshold=%s\n" "$label" "$delta" "$MAX_DELTA_THRESHOLD"
        cat "$TMPOUT"
        FAILED=$((FAILED + 1))
    fi
}

run_qwen_model() {
    local short="$1"
    local model="$2"
    local model_dir="$3"
    local has_release_bf16="$4"
    local has_release_fp8="$5"
    local has_raw=0

    if [ -n "$model_dir" ] && find "$model_dir" -maxdepth 1 -type f \( -name "*.safetensors" -o -name "*.safetensors.index.json" \) 2>/dev/null | grep -q .; then
        has_raw=1
    fi

    echo "--- $model ---"
    if [ -n "$model_dir" ] && { [ "$has_raw" -eq 1 ] || [ -f "$model_dir/.supersonic/v2/manifest.json" ] || [ "$has_release_bf16" -eq 1 ]; }; then
        run_case "$short bf16" "$model" "$model_dir" "$FETCH_TIMEOUT"
    else
        printf "  %-36s SKIP (no BF16 release/raw checkpoint)\n" "$short bf16"
        SKIPPED=$((SKIPPED + 1))
    fi
    run_case "$short --int4" "$model" "$model_dir" "$FETCH_TIMEOUT" --int4
    if [ -n "$model_dir" ] && { [ "$has_raw" -eq 1 ] || [ -f "$model_dir/.supersonic/v2-fp8/manifest.json" ] || [ "$has_release_fp8" -eq 1 ]; }; then
        run_case "$short --fp8-runtime" "$model" "$model_dir" "$FETCH_TIMEOUT" --fp8-runtime
    else
        printf "  %-36s SKIP (no FP8-native release/raw checkpoint)\n" "$short --fp8-runtime"
        SKIPPED=$((SKIPPED + 1))
    fi
    if [ -n "$model_dir" ] && { [ "$has_raw" -eq 1 ] || [ -f "$model_dir/.supersonic/v2/manifest.json" ] || [ "$has_release_bf16" -eq 1 ]; }; then
        run_case "$short --kv-fp8" "$model" "$model_dir" "$FETCH_TIMEOUT" --kv-fp8
    else
        printf "  %-36s SKIP (no BF16 release/raw checkpoint)\n" "$short --kv-fp8"
        SKIPPED=$((SKIPPED + 1))
    fi
    echo ""
}

run_qwen_model "0.8B" "qwen3.5-0.8b" "$MODEL_DIR_08B" 1 1
run_qwen_model "2B" "qwen3.5-2b" "$MODEL_DIR_2B" 1 1
run_qwen_model "4B" "qwen3.5-4b" "$MODEL_DIR_4B" 1 1
run_qwen_model "9B" "qwen3.5-9b" "$MODEL_DIR_9B" 0 1

echo "Summary: $PASSED passed, $FAILED failed, $SKIPPED skipped"
if [ "$FAILED" -gt 0 ]; then
    exit 1
fi
