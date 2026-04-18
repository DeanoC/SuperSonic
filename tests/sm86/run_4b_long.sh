#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-${SUPERSONIC_MODEL_DIR_4B:-}}"
if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <path-to-Qwen3.5-4B>"
    echo "  or set SUPERSONIC_MODEL_DIR_4B environment variable"
    exit 1
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: $MODEL_DIR/config.json not found — is this a valid model directory?"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f /root/.cargo/env ]; then
    . /root/.cargo/env
fi

TIMEOUT="${TIMEOUT:-1200}"
CORPUS_TIMEOUT="${CORPUS_TIMEOUT:-1200}"

echo "=== SuperSonic Long-Context Test: sm86 / qwen3.5-4b ==="
echo "Model dir:  $MODEL_DIR"
echo "Oracle:     cpu"
echo ""

echo "--- Building (release) ---"
SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}" cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic 2>&1
echo ""

SUPERSONIC="$REPO_ROOT/target/release/supersonic"
GOLDEN="$REPO_ROOT/tests/corpus/golden_4b_long.json"

CORPUS_TIMEOUT="$CORPUS_TIMEOUT" \
    "$REPO_ROOT/tests/corpus/run_golden.sh" \
    qwen3.5-4b "$MODEL_DIR" "$GOLDEN" "$SUPERSONIC" \
    --backend cuda --oracle-device cpu

echo "=== Long-context tests passed ==="
