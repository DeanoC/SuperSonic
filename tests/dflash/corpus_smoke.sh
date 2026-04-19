#!/bin/bash
# DFlash correctness + wall-clock smoke against the Qwen3.5-9B INT4 baseline.
#
# Emits per-prompt MATCH/MISMATCH + decode_ms for both paths and a
# summary line with total time + mean acceptance.
#
# Uses `--force-kernel-decode` for the baseline so the two sides share the
# same per-step kernel (the default single-sequence 4B baseline replays a
# full prefill per step "for correctness", which is slower AND produces
# different argmaxes on marginal tokens than the decode kernel used by
# dflash's re-decode). See `project_m4_2_findings.md` for the discussion.
#
# Run from the repo root:
#   ./tests/dflash/corpus_smoke.sh
set -eu

BASE_CMD="./target/release/supersonic --model qwen3.5-9b --model-dir /home/deano/models/qwen3.5-9b --int4 --force-kernel-decode"
DFLASH_CMD="$BASE_CMD --dflash --dflash-draft-dir /home/deano/models/qwen35-9b-dflash --dflash-verify prefill"

# Prompts drawn from the Gemma4 corpus, excluding empty/whitespace cases
# that the dflash engine bails on.
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
OK=0
TOTAL=0
BASE_MS_TOTAL=0
DF_MS_TOTAL=0
ACC_SUM=0.0
ROUNDS_TOTAL=0
for PROMPT in "${PROMPTS[@]}"; do
  TOTAL=$((TOTAL+1))
  P=$(printf '%b' "$PROMPT")
  echo "===== [$TOTAL/${#PROMPTS[@]}] prompt=$(echo "$P" | head -c 40 | tr '\n' ' ')..."
  base_log=$(mktemp)
  df_log=$(mktemp)
  $BASE_CMD --prompt "$P" --max-new-tokens $MAX_NEW > "$base_log" 2>&1
  $DFLASH_CMD --prompt "$P" --max-new-tokens $MAX_NEW > "$df_log" 2>&1
  base_tokens=$(grep '^\[tokens\]' "$base_log" || echo "MISSING")
  df_tokens=$(grep '^\[tokens\]' "$df_log" || echo "MISSING")
  base_ms=$(grep -oE 'decode_ms=[0-9]+' "$base_log" | head -1 | cut -d= -f2)
  df_ms=$(grep -oE 'decode_ms=[0-9]+' "$df_log" | head -1 | cut -d= -f2)
  df_rounds=$(grep -oE 'rounds=[0-9]+' "$df_log" | head -1 | cut -d= -f2)
  df_acc=$(grep -oE 'mean_accepted_per_round=[0-9]+\.[0-9]+' "$df_log" | head -1 | cut -d= -f2)
  if [[ "$base_tokens" == "$df_tokens" ]]; then
    OK=$((OK+1))
    verdict="MATCH"
  else
    verdict="MISMATCH"
  fi
  printf "  base %4sms | dflash %4sms | rounds=%2s accept=%s | %s\n" \
    "$base_ms" "$df_ms" "$df_rounds" "$df_acc" "$verdict"
  [[ "$verdict" == "MISMATCH" ]] && {
    echo "    base:   $base_tokens"
    echo "    dflash: $df_tokens"
  }
  BASE_MS_TOTAL=$((BASE_MS_TOTAL + base_ms))
  DF_MS_TOTAL=$((DF_MS_TOTAL + df_ms))
  ROUNDS_TOTAL=$((ROUNDS_TOTAL + df_rounds))
  ACC_SUM=$(echo "$ACC_SUM + $df_acc" | bc -l)
  rm "$base_log" "$df_log"
done
echo ""
echo "Summary: $OK / $TOTAL match"
MEAN_ACC=$(echo "scale=2; $ACC_SUM / $TOTAL" | bc -l)
echo "Total decode_ms: baseline=$BASE_MS_TOTAL  dflash=$DF_MS_TOTAL  (dflash/baseline = $(echo "scale=2; $DF_MS_TOTAL / $BASE_MS_TOTAL" | bc -l)x)"
echo "DFlash rounds total=$ROUNDS_TOTAL  mean_accept_per_round=$MEAN_ACC"
