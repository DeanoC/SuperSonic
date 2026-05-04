#!/usr/bin/env bash
# Chained-vs-persistent A/B for sparse-prefill TTFT.
# Same prompt, same model, same load conditions — only the kernel path
# differs. --no-persistent-decode forces the chained sibling fns even
# when cache_pos is set; default keeps the new persistent-with-cache_pos
# path. Cleaner than the R1-baseline comparison because it removes
# inter-day variance.

set -u
# Resolve repo-relative paths so the script works from any worktree.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUPERSONIC="${SUPERSONIC:-$REPO_ROOT/target/release/supersonic}"
MD="${SUPERSONIC_QWEN36_35B_A3B_DIR:-/mnt/data/models/Qwen3.6-35B-A3B}"
DD="${SUPERSONIC_QWEN35_0_8B_DIR:-/mnt/data/models/Qwen3.5-0.8B}"
FIXTURES="$REPO_ROOT/tests/fixtures/specprefill"
SUMMARY=/tmp/qwen36_ttft_ab_summary.tsv

declare -A PROMPTS
PROMPTS[c0088]=$FIXTURES/specprefill_c0088_target88_actual88.txt
PROMPTS[c0349]=$FIXTURES/specprefill_c0349_target349_actual349.txt
PROMPTS[c1393]=$FIXTURES/specprefill_c1393_target1393_actual1393.txt
PROMPTS[c4177]=$FIXTURES/specprefill_c4177_target4177_actual4177.txt
PROMPTS[c8353]=$FIXTURES/specprefill_c8353_target8353_actual8353.txt

# 8k skipped from the A/B by default — chained at 8k took ~12 min in R1.
# Pass `INCLUDE_8K=1 bash ...` to add it.
CELLS=(c0088 c0349 c1393 c4177)
[ "${INCLUDE_8K:-0}" = "1" ] && CELLS+=(c8353)

echo -e "cell\tmode\tprefill_chain_ms\tprefill_total_ms\tspeculator_ms\tprefill_steps" > "$SUMMARY"

run_cell() {
    local cell="$1"
    local mode="$2"   # persistent | chained
    local prompt_file="${PROMPTS[$cell]}"
    local log="/tmp/qwen36_ttft_ab_${cell}_${mode}.log"
    local prompt; prompt=$(cat "$prompt_file")

    local kernel_flag
    case "$mode" in
        persistent) kernel_flag="--persistent-decode" ;;
        chained)    kernel_flag="--no-persistent-decode" ;;
        *) echo "bad mode $mode"; exit 2 ;;
    esac

    echo "[ab] cell=$cell mode=$mode" >&2
    HIP_ARCH=gfx1100 \
    "$SUPERSONIC" \
        --backend hip \
        --model qwen3.6-35b-a3b \
        --model-dir "$MD" \
        --int4 \
        --prompt "$prompt" \
        --max-new-tokens 1 \
        --emit-stage-timings \
        $kernel_flag \
        --specprefill-draft-dir "$DD" \
        --specprefill-algorithm cosine \
        --specprefill-keep-ratio 0.50 \
        >"$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[ab] cell=$cell mode=$mode FAILED rc=$rc" >&2
        echo -e "$cell\t$mode\tFAILED\tFAILED\tFAILED\tFAILED" >> "$SUMMARY"
        return
    fi

    chain_ms=$(grep -oE 'prefill_chain_ms=[0-9.]+' "$log" | tail -n1 | sed 's/prefill_chain_ms=//')
    total_ms=$(grep -oE 'prefill_total_ms=[0-9.]+' "$log" | tail -n1 | sed 's/prefill_total_ms=//')
    spec_ms=$(grep -oE 'speculator \(cosine\) done in [0-9]+ms' "$log" | grep -oE '[0-9]+' | tail -n1)
    steps=$(grep -oE 'prefill_steps=[0-9]+' "$log" | tail -n1 | sed 's/prefill_steps=//')
    [ -z "$spec_ms" ] && spec_ms=0
    echo -e "$cell\t$mode\t${chain_ms:--}\t${total_ms:--}\t${spec_ms}\t${steps:--}" >> "$SUMMARY"
    echo "[ab]   chain=${chain_ms} total=${total_ms} spec=${spec_ms} steps=${steps}" >&2
}

for cell in "${CELLS[@]}"; do
    run_cell "$cell" persistent
    sleep 2
    run_cell "$cell" chained
    sleep 2
done

echo "[ab] DONE"
cat "$SUMMARY"
