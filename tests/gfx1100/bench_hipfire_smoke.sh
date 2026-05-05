#!/usr/bin/env bash
# Smoke: hipfire adapter on one combo, gated by check-versions.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR_08B="${MODEL_DIR_08B:-/mnt/data/models/Qwen3.5-0.8B}"

cd "$REPO_ROOT"
"$REPO_ROOT/tools/external/check-versions.sh"

RUN_ROOT="$(mktemp -d)/bench-runs"
mkdir -p "$RUN_ROOT/2026-05-05-smoke"

# Bench-perf needs to have run first to create a run-dir for the external script to attach to.
./target/release/bench-perf --arch gfx1100 --models qwen3.5-0.8b --quants bf16 \
    --model-dir "qwen3.5-0.8b=$MODEL_DIR_08B" \
    --run-root "$RUN_ROOT" >/dev/null

python -m oracle.bench.external.external_main --engine hipfire \
    --models qwen3.5-0.8b --quants bf16 \
    --run-root "$RUN_ROOT" \
    --model-dir "qwen3.5-0.8b=$MODEL_DIR_08B"

LATEST="$(ls -t "$RUN_ROOT" | head -n1)"
test -s "$RUN_ROOT/$LATEST/external/hipfire/qwen3.5-0.8b_bf16.json" \
    || { echo "[hipfire-smoke] FAIL" >&2; exit 1; }
echo "[hipfire-smoke] PASS"
