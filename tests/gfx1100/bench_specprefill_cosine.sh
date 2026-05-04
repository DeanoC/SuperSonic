#!/usr/bin/env bash
# Manual TTFT comparison for SpecPrefill cosine scoring on gfx1100.
# Captures TTFT (ms) for each mode on a 1353-token prompt + 4 generated tokens.
# Warmup pass + 3 measurement runs + 3s cooldown + median per cell.
#
# Usage:
#   tests/gfx1100/bench_specprefill_cosine.sh > /tmp/specprefill_cosine_ttft.md

set -e
SUPERSONIC=./target/release/supersonic
MD=/mnt/data/models/Qwen3.5-9B
DD=/mnt/data/models/Qwen3.5-0.8B
PROMPT_FILE="${PROMPT_FILE:-/tmp/specprefill_long.txt}"
MAX_NEW="${MAX_NEW:-4}"

if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: $PROMPT_FILE not found. Use the 1353-token prompt from prior SpecPrefill measurements." >&2
    exit 1
fi
if [ ! -x "$SUPERSONIC" ]; then
    echo "ERROR: $SUPERSONIC not built. Run: cargo build --release --bin supersonic" >&2
    exit 1
fi
PROMPT=$(cat "$PROMPT_FILE")

# Returns total TTFT ms by summing speculator+target lines (sparse) or
# the single prefill-done line (dense). Echoes integer ms.
extract_ttft() {
    local raw="$1"
    local spec_ms target_ms dense_ms
    spec_ms=$(printf '%s' "$raw" | sed -n 's/.*speculator (cosine|lookahead) done in \([0-9]*\)ms.*/\1/p' | tail -n1)
    target_ms=$(printf '%s' "$raw" | sed -n 's/.*target prefill done in \([0-9]*\)ms.*/\1/p' | tail -n1)
    dense_ms=$(printf '%s' "$raw" | sed -n 's/.*native GPU prefill done in \([0-9]*\)ms.*/\1/p' | tail -n1)
    if [ -n "$spec_ms" ] && [ -n "$target_ms" ]; then
        echo $(( spec_ms + target_ms ))
    elif [ -n "$dense_ms" ]; then
        echo "$dense_ms"
    else
        echo "—"
    fi
}

run_cell() {
    local label="$1"; shift
    local m1 m2 m3
    sleep 3  # cooldown
    "$SUPERSONIC" --backend hip --model qwen3.5-9b --model-dir "$MD" \
        --prompt "$PROMPT" --max-new-tokens 2 "$@" >/dev/null 2>&1 || true  # warmup
    sleep 1
    local raw
    raw=$("$SUPERSONIC" --backend hip --model qwen3.5-9b --model-dir "$MD" \
        --prompt "$PROMPT" --max-new-tokens "$MAX_NEW" "$@" 2>&1 || true)
    m1=$(extract_ttft "$raw")
    sleep 1
    raw=$("$SUPERSONIC" --backend hip --model qwen3.5-9b --model-dir "$MD" \
        --prompt "$PROMPT" --max-new-tokens "$MAX_NEW" "$@" 2>&1 || true)
    m2=$(extract_ttft "$raw")
    sleep 1
    raw=$("$SUPERSONIC" --backend hip --model qwen3.5-9b --model-dir "$MD" \
        --prompt "$PROMPT" --max-new-tokens "$MAX_NEW" "$@" 2>&1 || true)
    m3=$(extract_ttft "$raw")
    local median
    median=$(printf '%s\n' "$m1" "$m2" "$m3" | grep -v '^—$' | sort -n | sed -n '2p')
    [ -z "$median" ] && median="—"
    printf "| %-30s | %5s |\n" "$label" "$median"
}

echo "| Mode                           | TTFT ms |"
echo "|--------------------------------|--------:|"
run_cell "dense (no specprefill)"
run_cell "cosine, shallowest (A)"     --specprefill-draft-dir "$DD" --specprefill-algorithm cosine --specprefill-keep-ratio 0.50
SUPERSONIC_SPECPREFILL_LAYERS=all_max run_cell "cosine, all_max (B)" --specprefill-draft-dir "$DD" --specprefill-algorithm cosine --specprefill-keep-ratio 0.50
run_cell "lookahead (Phase C)"        --specprefill-draft-dir "$DD" --specprefill-algorithm lookahead --specprefill-keep-ratio 0.50
