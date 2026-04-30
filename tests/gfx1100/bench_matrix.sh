#!/usr/bin/env bash
#
# Benchmark the full HIP gfx1100 quant matrix and emit a markdown table to
# stdout. Mirrors the gfx1150 numbers in `docs/performance.md` (same prompt,
# same `MAX_NEW`) so cross-arch comparisons stay apples-to-apples.
#
# Each cell records `ms/step` from the `[result] decode_ms=N ms_per_step=M`
# line emitted by the runner. Runs serial; full sweep on a 7900 XTX with
# warm bakes is ~1 minute.
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
    # Warm up once (kernel JIT, page cache) then take a steady-state run.
    "$SUPERSONIC" --model "$model" --model-dir "$model_dir" \
        --prompt "$PROMPT" --max-new-tokens "$WARMUP_NEW" "$@" \
        >/dev/null 2>&1 || true
    local out
    out="$("$SUPERSONIC" --model "$model" --model-dir "$model_dir" \
        --prompt "$PROMPT" --max-new-tokens "$MAX_NEW" "$@" 2>&1 || true)"
    # Runner emits either `ms_per_step=N` (Gemma 4, Phi-4) or
    # `ms_per_tok=N` (Qwen 3.5). Both are per-decode-step timings.
    local mspt
    mspt="$(printf '%s' "$out" | sed -n 's/.*ms_per_step=\([0-9.]*\).*/\1/p' | tail -n1)"
    if [ -z "$mspt" ]; then
        mspt="$(printf '%s' "$out" | sed -n 's/.*ms_per_tok=\([0-9.]*\).*/\1/p' | tail -n1)"
    fi
    if [ -z "$mspt" ]; then
        # Some FP8 / INT4 combos may bail with a clear message (e.g. unsupported
        # combo) — record as "—" rather than fail the whole sweep.
        echo "—"
    else
        echo "$mspt"
    fi
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
