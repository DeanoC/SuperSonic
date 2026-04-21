#!/usr/bin/env bash
#
# Nsight Compute helper for the qwen3.5-0.8b sm86 decode path.
# Profiles the persistent decode kernel for a single generated token.
#
# Usage:
#   ./tests/sm86/profile_qwen08_decode.sh [model_dir]
#
# Environment:
#   PROFILE_MODE=hero        hero path (default)
#   PROFILE_MODE=fast        disable hero, keep CUDA fast-greedy
#   PROFILE_MODE=legacy      disable hero and fast-greedy
#   PROMPT=Hello             prompt text
#   MAX_NEW_TOKENS=1         generated tokens
#   NCU_METRICS=...          comma-separated ncu metrics
#
set -euo pipefail

MODEL_DIR="${1:-${SUPERSONIC_MODEL_DIR:-}}"
if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <path-to-Qwen3.5-0.8B>"
    echo "  or set SUPERSONIC_MODEL_DIR"
    exit 1
fi

if ! command -v /usr/local/cuda/bin/ncu >/dev/null 2>&1; then
    echo "ncu not found at /usr/local/cuda/bin/ncu"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f /root/.cargo/env ]; then
    . /root/.cargo/env
fi

PROFILE_MODE="${PROFILE_MODE:-hero}"
PROMPT="${PROMPT:-Hello}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"
NCU_METRICS="${NCU_METRICS:-sm__warps_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active,dram__throughput.avg.pct_of_peak_sustained_elapsed}"
export SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}"

case "$PROFILE_MODE" in
    hero)
        export SUPERSONIC_ENABLE_CUDA_08B_HERO=1
        unset SUPERSONIC_DISABLE_CUDA_08B_HERO || true
        unset SUPERSONIC_DISABLE_CUDA_FAST_GREEDY || true
        ;;
    fast)
        unset SUPERSONIC_ENABLE_CUDA_08B_HERO || true
        export SUPERSONIC_DISABLE_CUDA_08B_HERO=1
        unset SUPERSONIC_DISABLE_CUDA_FAST_GREEDY || true
        ;;
    legacy)
        unset SUPERSONIC_ENABLE_CUDA_08B_HERO || true
        export SUPERSONIC_DISABLE_CUDA_08B_HERO=1
        export SUPERSONIC_DISABLE_CUDA_FAST_GREEDY=1
        ;;
    *)
        echo "Unsupported PROFILE_MODE: $PROFILE_MODE"
        echo "Expected one of: hero, fast, legacy"
        exit 1
        ;;
esac

echo "=== SuperSonic sm86 Qwen3.5-0.8B Decode Profile ==="
echo "Model dir:      $MODEL_DIR"
echo "Profile mode:   $PROFILE_MODE"
echo "Prompt:         $PROMPT"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Metrics:        $NCU_METRICS"
echo ""

echo "--- Building (release) ---"
cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic
echo ""

SUPERSONIC="$REPO_ROOT/target/release/supersonic"

/usr/local/cuda/bin/ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:dotcache_qwen35_persistent_decode_kernel' \
  --launch-count 1 \
  --metrics "$NCU_METRICS" \
  "$SUPERSONIC" \
    --backend cuda \
    --model qwen3.5-0.8b \
    --model-dir "$MODEL_DIR" \
    --prompt "$PROMPT" \
    --max-new-tokens "$MAX_NEW_TOKENS"
