#!/usr/bin/env bash
#
# Parity gate: assert that the Rust bench-perf orchestrator produces the same
# ms/step values as the legacy bash bench_matrix.sh on a pinned model subset,
# within ±3%.
#
# This is the gate for deleting tests/gfx1100/bench_matrix.sh.
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUPERSONIC="$REPO_ROOT/target/release/supersonic"
BENCH_PERF="$REPO_ROOT/target/release/bench-perf"

if [ ! -x "$SUPERSONIC" ] || [ ! -x "$BENCH_PERF" ]; then
    echo "ERROR: build supersonic + bench-perf first" >&2
    exit 1
fi

MODEL_DIR_08B="${MODEL_DIR_08B:-/mnt/data/models/Qwen3.5-0.8B}"

echo "[parity] running legacy bench_matrix.sh on qwen3.5-0.8b BF16..."
LEGACY_OUT="$(MODEL_DIR_08B="$MODEL_DIR_08B" "$SCRIPT_DIR/bench_matrix.sh" 2>/dev/null \
    | grep "qwen3.5-0.8b" | head -n1)"
LEGACY_BF16="$(echo "$LEGACY_OUT" | awk -F'|' '{print $3}' | tr -d ' ')"

echo "[parity] running Rust bench-perf on qwen3.5-0.8b BF16..."
RUN_DIR="$(mktemp -d)"
"$BENCH_PERF" --arch gfx1100 --models qwen3.5-0.8b --quants bf16 \
    --model-dir "qwen3.5-0.8b=$MODEL_DIR_08B" \
    --run-root "$RUN_DIR" 2>&1 | tee "$RUN_DIR/log"

CELL_JSON="$(find "$RUN_DIR" -name 'qwen3.5-0.8b_bf16.json' | head -n1)"
RUST_BF16="$(jq -r '.ms_per_step // empty' "$CELL_JSON")"

if [ -z "$LEGACY_BF16" ] || [ -z "$RUST_BF16" ]; then
    echo "[parity] FAIL: missing measurement (legacy='$LEGACY_BF16' rust='$RUST_BF16')" >&2
    exit 1
fi

DELTA_PCT="$(awk -v l="$LEGACY_BF16" -v r="$RUST_BF16" 'BEGIN { d=(r-l)/l*100; if(d<0)d=-d; print d }')"
echo "[parity] legacy=$LEGACY_BF16 rust=$RUST_BF16 delta=${DELTA_PCT}%"
LIMIT="${PARITY_LIMIT_PCT:-3.0}"
PASS="$(awk -v d="$DELTA_PCT" -v l="$LIMIT" 'BEGIN { print (d<=l) ? "yes" : "no" }')"
if [ "$PASS" != "yes" ]; then
    echo "[parity] FAIL: delta ${DELTA_PCT}% exceeds limit ${LIMIT}%" >&2
    exit 1
fi
echo "[parity] PASS"
