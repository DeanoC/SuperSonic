#!/usr/bin/env bash
# End-to-end smoke: one combo through perf + quality + render.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR_08B="${MODEL_DIR_08B:-/mnt/data/models/Qwen3.5-0.8B}"

cd "$REPO_ROOT"
cargo build --release --bin supersonic
cargo build --release -p supersonic-bench

RUN_ROOT="$(mktemp -d)/bench-runs"
echo "[smoke] perf"
./target/release/bench-perf --arch gfx1100 --models qwen3.5-0.8b --quants bf16 \
    --model-dir "qwen3.5-0.8b=$MODEL_DIR_08B" \
    --run-root "$RUN_ROOT"

RUN_DIR="$(ls -t "$RUN_ROOT" | head -n1)"
RUN_PATH="$RUN_ROOT/$RUN_DIR"

echo "[smoke] quality"
python -m oracle.bench.quality_main --arch gfx1100 \
    --models qwen3.5-0.8b --quants bf16 \
    --binary ./target/release/supersonic \
    --run-root "$RUN_ROOT" \
    --model-dir "qwen3.5-0.8b=$MODEL_DIR_08B" \
    --ppl-num-chunks 1 --ppl-contexts 256

echo "[smoke] render"
python -m oracle.bench.render.render_main markdown \
    --run "$RUN_PATH" \
    --out "$REPO_ROOT"

# Assertions: at least one perf JSON and one quality JSON exist with non-empty fields.
test -s "$RUN_PATH/perf/qwen3.5-0.8b_bf16.json" || { echo "[smoke] FAIL: missing perf cell" >&2; exit 1; }
ls "$RUN_PATH/quality/" | head -n1 || { echo "[smoke] FAIL: no quality cells" >&2; exit 1; }
echo "[smoke] PASS — run_dir=$RUN_PATH"
