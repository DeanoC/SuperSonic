#!/bin/bash
# DFlash wall-clock vs block size on two representative prompts.
#
# Small block = lower fixed verify cost per round but also smaller max
# commit per round (accepted_len is capped at block_size - 1 by the
# engine). Empirically B=4 beat B=16 on both prompts tested 2026-04-19
# against Qwen3.5-9B INT4 on gfx1150. See `project_m4_2_findings.md`.
set -eu

BASE_CMD="./target/release/supersonic --model qwen3.5-9b --model-dir /home/deano/models/qwen3.5-9b --int4 --force-kernel-decode"
DFLASH_CMD_PREFIX="$BASE_CMD --dflash --dflash-draft-dir /home/deano/models/qwen35-9b-dflash --dflash-verify prefill"
MAX_NEW=12

# Low-accept + high-accept representatives.
for PROMPT in "Hello" "The sum of 2 and 3 is"; do
  echo ""
  echo "========= prompt: $PROMPT ========="
  base_log=$(mktemp)
  $BASE_CMD --prompt "$PROMPT" --max-new-tokens $MAX_NEW > "$base_log" 2>&1
  base_ms=$(grep -oE 'decode_ms=[0-9]+' "$base_log" | head -1 | cut -d= -f2)
  base_tokens=$(grep '^\[tokens\]' "$base_log")
  printf "baseline (fkd)   : %5sms  tokens=%s\n" "$base_ms" "$base_tokens"
  rm "$base_log"
  for B in 4 8 12 16; do
    df_log=$(mktemp)
    $DFLASH_CMD_PREFIX --dflash-block $B --prompt "$PROMPT" --max-new-tokens $MAX_NEW > "$df_log" 2>&1
    ms=$(grep -oE 'decode_ms=[0-9]+' "$df_log" | head -1 | cut -d= -f2)
    tokens=$(grep '^\[tokens\]' "$df_log")
    rounds=$(grep -oE 'rounds=[0-9]+' "$df_log" | head -1 | cut -d= -f2)
    acc=$(grep -oE 'mean_accepted_per_round=[0-9]+\.[0-9]+' "$df_log" | head -1 | cut -d= -f2)
    match="MATCH"
    [[ "$tokens" != "$base_tokens" ]] && match="MISMATCH"
    printf "dflash B=%-2d      : %5sms  rounds=%2s accept=%s  %s\n" "$B" "$ms" "$rounds" "$acc" "$match"
    rm "$df_log"
  done
done
