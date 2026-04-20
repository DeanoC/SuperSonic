#!/usr/bin/env bash
#
# Run the validated CUDA sm86 suite in one command:
#   - Qwen3.5-0.8B
#   - Qwen3.5-0.8B long-context CPU oracle
#   - Qwen3.5-0.8B fast-greedy regression
#   - Qwen3.5-4B
#   - Qwen3.5-4B long-context CPU oracle
#   - Qwen3.5-4B batch_size=2
#   - negative CUDA v1 coverage
#
# Usage:
#   ./tests/sm86/run_all.sh [model_dir_0_8b] [model_dir_4b]
#
# Or set:
#   SUPERSONIC_MODEL_DIR
#   SUPERSONIC_MODEL_DIR_4B
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_DIR_0_8B="${1:-${SUPERSONIC_MODEL_DIR:-}}"
MODEL_DIR_4B="${2:-${SUPERSONIC_MODEL_DIR_4B:-}}"

if [ -z "$MODEL_DIR_0_8B" ] || [ -z "$MODEL_DIR_4B" ]; then
    echo "Usage: $0 <path-to-Qwen3.5-0.8B> <path-to-Qwen3.5-4B>"
    echo "  or set SUPERSONIC_MODEL_DIR and SUPERSONIC_MODEL_DIR_4B"
    exit 1
fi

export SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}"

echo "=== SuperSonic sm86 CUDA Suite ==="
echo ""

TIMEOUT="${TIMEOUT:-600}" "$SCRIPT_DIR/run.sh" "$MODEL_DIR_0_8B"
echo ""
TIMEOUT="${TIMEOUT:-900}" "$SCRIPT_DIR/run_long.sh" "$MODEL_DIR_0_8B"
echo ""
TIMEOUT="${TIMEOUT:-300}" "$SCRIPT_DIR/run_fast_greedy.sh" "$MODEL_DIR_0_8B"
echo ""
TIMEOUT="${TIMEOUT:-900}" "$SCRIPT_DIR/run_4b.sh" "$MODEL_DIR_4B"
echo ""
TIMEOUT="${TIMEOUT:-1200}" "$SCRIPT_DIR/run_4b_long.sh" "$MODEL_DIR_4B"
echo ""
TIMEOUT="${TIMEOUT:-900}" "$SCRIPT_DIR/run_batch.sh" "$MODEL_DIR_4B"
echo ""
TIMEOUT="${TIMEOUT:-120}" "$SCRIPT_DIR/run_negative.sh" "$MODEL_DIR_0_8B" "$MODEL_DIR_4B"
