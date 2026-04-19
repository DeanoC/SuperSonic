#!/bin/bash
# DFlash M4.3b A/B: `--dflash-verify prefill` vs `--dflash-verify fused`
# on the same 15 prompts used by corpus_smoke.sh. The fused path (one
# persistent_decode_4b megakernel launch per verify) must emit the same
# or higher MATCH count as the prefill path on the same corpus — they
# can disagree on individual prompts (marginal-argmax flips differ by
# kernel path), but the total count is the correctness signal.
#
# Run from the repo root:
#   ./tests/dflash/fused_vs_prefill_smoke.sh
set -eu

BASE_CMD="./target/release/supersonic --model qwen3.5-9b --model-dir /home/deano/models/qwen3.5-9b --int4 --force-kernel-decode"
# Prefill verify runs at the shipped default B=4 (best corpus point per
# project_m4_2_findings). Fused verify is capped at B=3 on 9B because
# (block_size + B*hidden + fp8_lut) * 4 must fit in 64 KiB LDS on
# gfx1150 — see the diagnostic in verify_block_fused_decode. Compare
# each mode at its own best-allowed block size.
PREFILL_CMD="$BASE_CMD --dflash --dflash-draft-dir /home/deano/models/qwen35-9b-dflash --dflash-verify prefill --dflash-block 4"
FUSED_CMD="$BASE_CMD --dflash --dflash-draft-dir /home/deano/models/qwen35-9b-dflash --dflash-verify fused   --dflash-block 3"

declare -a PROMPTS=(
  "Hello"
  "The quick brown fox jumps"
  "Hi"
  "Good morning"
  "Q: What is the capital of France?\nA:"
  "Why is the sky blue?"
  "User: Hello\nAssistant:"
  "100 + 200 ="
  "The sum of 2 and 3 is"
  "def fibonacci(n):\n    if n <= 1:\n        return n\n    return"
  "for i in range(10):"
  "Once upon a time in a faraway land,"
  "In the year 2050,"
  "Bonjour le monde"
  "1. apple\n2. banana\n3."
)
MAX_NEW=12
OK_BASE_PF=0
OK_BASE_FU=0
OK_PF_FU=0
TOTAL=0
BASE_MS_TOTAL=0
PF_MS_TOTAL=0
FU_MS_TOTAL=0
for PROMPT in "${PROMPTS[@]}"; do
  TOTAL=$((TOTAL+1))
  P=$(printf '%b' "$PROMPT")
  echo "===== [$TOTAL/${#PROMPTS[@]}] prompt=$(echo "$P" | head -c 40 | tr '\n' ' ')..."
  base_log=$(mktemp)
  pf_log=$(mktemp)
  fu_log=$(mktemp)
  $BASE_CMD     --prompt "$P" --max-new-tokens $MAX_NEW > "$base_log" 2>&1
  $PREFILL_CMD  --prompt "$P" --max-new-tokens $MAX_NEW > "$pf_log"   2>&1
  $FUSED_CMD    --prompt "$P" --max-new-tokens $MAX_NEW > "$fu_log"   2>&1
  base_tokens=$(grep '^\[tokens\]' "$base_log" || echo "MISSING")
  pf_tokens=$(grep '^\[tokens\]'   "$pf_log"   || echo "MISSING")
  fu_tokens=$(grep '^\[tokens\]'   "$fu_log"   || echo "MISSING")
  base_ms=$(grep -oE 'decode_ms=[0-9]+' "$base_log" | head -1 | cut -d= -f2)
  pf_ms=$(grep -oE 'decode_ms=[0-9]+'   "$pf_log"   | head -1 | cut -d= -f2)
  fu_ms=$(grep -oE 'decode_ms=[0-9]+'   "$fu_log"   | head -1 | cut -d= -f2)
  pf_acc=$(grep -oE 'mean_accepted_per_round=[0-9]+\.[0-9]+' "$pf_log" | head -1 | cut -d= -f2)
  fu_acc=$(grep -oE 'mean_accepted_per_round=[0-9]+\.[0-9]+' "$fu_log" | head -1 | cut -d= -f2)
  verdict_pf="MISMATCH"
  verdict_fu="MISMATCH"
  verdict_ab="MISMATCH"
  [[ "$base_tokens" == "$pf_tokens" ]] && { OK_BASE_PF=$((OK_BASE_PF+1)); verdict_pf="MATCH"; }
  [[ "$base_tokens" == "$fu_tokens" ]] && { OK_BASE_FU=$((OK_BASE_FU+1)); verdict_fu="MATCH"; }
  [[ "$pf_tokens" == "$fu_tokens" ]]   && { OK_PF_FU=$((OK_PF_FU+1));     verdict_ab="MATCH"; }
  printf "  base %4sms | prefill %4sms acc=%s %s-vs-base | fused %4sms acc=%s %s-vs-base | pf-vs-fu %s\n" \
    "$base_ms" "$pf_ms" "$pf_acc" "$verdict_pf" "$fu_ms" "$fu_acc" "$verdict_fu" "$verdict_ab"
  [[ "$verdict_fu" == "MISMATCH" ]] && {
    echo "    base:    $base_tokens"
    echo "    fused:   $fu_tokens"
  }
  BASE_MS_TOTAL=$((BASE_MS_TOTAL + base_ms))
  PF_MS_TOTAL=$((PF_MS_TOTAL + pf_ms))
  FU_MS_TOTAL=$((FU_MS_TOTAL + fu_ms))
  rm "$base_log" "$pf_log" "$fu_log"
done
echo ""
echo "Summary (vs --force-kernel-decode baseline): prefill $OK_BASE_PF / $TOTAL   fused $OK_BASE_FU / $TOTAL"
echo "Summary (prefill vs fused agree): $OK_PF_FU / $TOTAL"
echo "Total decode_ms: baseline=$BASE_MS_TOTAL  prefill=$PF_MS_TOTAL  fused=$FU_MS_TOTAL"
if [[ "$BASE_MS_TOTAL" -gt 0 ]]; then
  echo "  prefill/baseline = $(echo "scale=2; $PF_MS_TOTAL / $BASE_MS_TOTAL" | bc -l)x"
  echo "  fused/baseline   = $(echo "scale=2; $FU_MS_TOTAL / $BASE_MS_TOTAL" | bc -l)x"
fi
if [[ "$PF_MS_TOTAL" -gt 0 ]]; then
  echo "  fused/prefill    = $(echo "scale=2; $FU_MS_TOTAL / $PF_MS_TOTAL" | bc -l)x"
fi
