#!/usr/bin/env bash
#
# Benchmark the full HIP gfx1100 quant matrix and emit a markdown table to
# stdout. Mirrors the gfx1150 numbers in `docs/performance.md` (same prompt,
# same `MAX_NEW`) so cross-arch comparisons stay apples-to-apples.
#
# Each cell: 3s cooldown → 1 warmup run → median of 3 measurement runs.
# The cooldown + median was added 2026-05-04 after a previous version of
# the matrix without these controls produced numbers biased upward by
# thermal accumulation across cells (see docs/performance.md § Methodology).
#
# Full sweep on a 7900 XTX with warm bakes is ~3 minutes (was ~1 minute
# before the 3-run-median; the extra runs stabilize the larger-model cells).
#
# Usage:
#   tests/gfx1100/bench_matrix.sh > /tmp/gfx1100_matrix.md
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUPERSONIC="$REPO_ROOT/target/release/supersonic"
PROMPT="${PROMPT:-The quick brown fox jumps over}"
MAX_NEW="${MAX_NEW:-16}"
WARMUP_NEW="${WARMUP_NEW:-2}"

if [ ! -x "$SUPERSONIC" ]; then
    echo "ERROR: $SUPERSONIC not found. Run: cargo build --release" >&2
    exit 1
fi

MODEL_DIR_08B="${MODEL_DIR_08B:-/mnt/data/models/Qwen3.5-0.8B}"
MODEL_DIR_2B="${MODEL_DIR_2B:-/mnt/data/models/Qwen3.5-2B}"
MODEL_DIR_4B="${MODEL_DIR_4B:-/mnt/data/models/Qwen3.5-4B}"
MODEL_DIR_9B="${MODEL_DIR_9B:-/mnt/data/models/Qwen3.5-9B}"
MODEL_DIR_GEMMA_E2B="${MODEL_DIR_GEMMA_E2B:-/mnt/data/models/gemma-4-E2B}"
MODEL_DIR_GEMMA_E4B="${MODEL_DIR_GEMMA_E4B:-/mnt/data/models/gemma-4-E4B}"
MODEL_DIR_PHI4="${MODEL_DIR_PHI4:-/mnt/data/models/Phi-4-mini-instruct}"

# Capture ms/step from one run; "skip" on error so the matrix completes.
# Args: model model_dir [extra_flags...]
bench_one() {
    local model="$1"; shift
    local model_dir="$1"; shift
    if [ ! -d "$model_dir" ]; then
        echo "skip"
        return 0
    fi
    # 3s cooldown to bleed thermal state from the previous cell. Without
    # this, serial benches over 7 models accumulate enough heat that the
    # larger Qwen3.5 cells measure 1.5-2x slower than steady-state. See
    # docs/performance.md § Methodology.
    sleep 3
    # Warm up once (kernel JIT, page cache).
    "$SUPERSONIC" --model "$model" --model-dir "$model_dir" \
        --prompt "$PROMPT" --max-new-tokens "$WARMUP_NEW" "$@" \
        >/dev/null 2>&1 || true
    # Median of 3 measurement runs.
    local m1 m2 m3
    extract_ms() {
        local raw="$1"
        local val
        val="$(printf '%s' "$raw" | sed -n 's/.*ms_per_step=\([0-9.]*\).*/\1/p' | tail -n1)"
        if [ -z "$val" ]; then
            val="$(printf '%s' "$raw" | sed -n 's/.*ms_per_tok=\([0-9.]*\).*/\1/p' | tail -n1)"
        fi
        printf '%s' "$val"
    }
    m1=$(extract_ms "$("$SUPERSONIC" --model "$model" --model-dir "$model_dir" \
        --prompt "$PROMPT" --max-new-tokens "$MAX_NEW" "$@" 2>&1 || true)")
    sleep 1
    m2=$(extract_ms "$("$SUPERSONIC" --model "$model" --model-dir "$model_dir" \
        --prompt "$PROMPT" --max-new-tokens "$MAX_NEW" "$@" 2>&1 || true)")
    sleep 1
    m3=$(extract_ms "$("$SUPERSONIC" --model "$model" --model-dir "$model_dir" \
        --prompt "$PROMPT" --max-new-tokens "$MAX_NEW" "$@" 2>&1 || true)")
    if [ -z "$m1" ] && [ -z "$m2" ] && [ -z "$m3" ]; then
        echo "—"
        return 0
    fi
    # Median: sort numerically, take middle.
    printf '%s\n' "$m1" "$m2" "$m3" | grep -v '^$' | sort -n | sed -n '2p'
}

row() {
    local label="$1"; shift
    local model="$1"; shift
    local model_dir="$1"; shift
    local bf16 int4 fp8r kvfp8
    bf16="$(bench_one  "$model" "$model_dir")"
    int4="$(bench_one  "$model" "$model_dir" --int4)"
    fp8r="$(bench_one  "$model" "$model_dir" --fp8-runtime)"
    kvfp8="$(bench_one "$model" "$model_dir" --kv-fp8)"
    printf "| %-15s | %5s | %5s | %5s | %5s |\n" \
        "$label" "$bf16" "$int4" "$fp8r" "$kvfp8"
}

echo "| Model           | BF16  | INT4  | FP8r  | KV-FP8 |"
echo "|-----------------|------:|------:|------:|-------:|"
row "qwen3.5-0.8b"   qwen3.5-0.8b   "$MODEL_DIR_08B"
row "qwen3.5-2b"     qwen3.5-2b     "$MODEL_DIR_2B"
row "qwen3.5-4b"     qwen3.5-4b     "$MODEL_DIR_4B"
row "qwen3.5-9b"     qwen3.5-9b     "$MODEL_DIR_9B"
row "gemma4-e2b"     gemma4-e2b     "$MODEL_DIR_GEMMA_E2B"
row "gemma4-e4b"     gemma4-e4b     "$MODEL_DIR_GEMMA_E4B"
row "phi4-mini"      phi4-mini      "$MODEL_DIR_PHI4"
