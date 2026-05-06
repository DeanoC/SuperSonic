#!/usr/bin/env bash
#
# CUDA sm86 Qwen3.6 SpecPrefill perf + quality gate.
#
# Usage:
#   tests/sm86/bench_specprefill_quality.sh \
#       /workspace/models/Qwen3.6-35B-A3B-FP8 \
#       /workspace/models/Qwen3.5-0.8B
#
# Environment:
#   PROMPT_REPEAT=12
#   PROMPT_SET=quick|standard   # default: standard
#   KEEP_RATIOS=0.75            # enforced by default; use 0.50,0.75 to gate balanced + conservative
#   INCLUDE_BALANCED=1          # include int4-spec050 in perf run and quality output
#   INCLUDE_EXPLORATORY=1       # include int4-spec025 in perf run and quality output
#   FAIL_EXPLORATORY=1          # make exploratory threshold misses fail the script
#   MEASUREMENT_RUNS=1
#   RUN_ROOT=target/bench-runs-sm86
#
set -euo pipefail

TARGET_DIR="${1:-${SUPERSONIC_MODEL_DIR_QWEN36:-}}"
DRAFT_DIR="${2:-${SUPERSONIC_MODEL_DIR_QWEN08:-}}"
if [ -z "$TARGET_DIR" ] || [ -z "$DRAFT_DIR" ]; then
    echo "Usage: $0 <path-to-Qwen3.6-35B-A3B-FP8> <path-to-Qwen3.5-0.8B>" >&2
    echo "  or set SUPERSONIC_MODEL_DIR_QWEN36 and SUPERSONIC_MODEL_DIR_QWEN08" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f /root/.cargo/env ]; then
    . /root/.cargo/env
fi

PROMPT_REPEAT="${PROMPT_REPEAT:-12}"
PROMPT_SET="${PROMPT_SET:-standard}"
KEEP_RATIOS="${KEEP_RATIOS:-0.75}"
MEASUREMENT_RUNS="${MEASUREMENT_RUNS:-1}"
RUN_ROOT="${RUN_ROOT:-target/bench-runs-sm86}"
export SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}"

QUANTS="int4,int4-spec075"
if [ "${INCLUDE_BALANCED:-0}" = "1" ] || [[ ",$KEEP_RATIOS," == *",0.50,"* ]]; then
    QUANTS="int4,int4-spec050,int4-spec075"
    if [[ ",$KEEP_RATIOS," != *",0.50,"* ]]; then
        KEEP_RATIOS="${KEEP_RATIOS},0.50"
    fi
fi

if [ "${INCLUDE_EXPLORATORY:-0}" = "1" ]; then
    QUANTS="${QUANTS},int4-spec025"
    if [[ ",$KEEP_RATIOS," != *",0.25,"* ]]; then
        KEEP_RATIOS="${KEEP_RATIOS},0.25"
    fi
fi

cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic

PROMPT="$(
    python3 - "$PROMPT_REPEAT" <<'PY'
import sys
repeat = int(sys.argv[1])
print(("SuperSonic CUDA dense prefill profiling sentence. " * repeat).strip())
PY
)"

pushd "$REPO_ROOT" >/dev/null

cargo run --release -p supersonic-bench --bin bench-perf -- \
    --arch sm86 \
    --models qwen3.6-35b-a3b \
    --quants "$QUANTS" \
    --model-dir "qwen3.6-35b-a3b=$TARGET_DIR" \
    --specprefill-draft-dir "qwen3.6-35b-a3b=$DRAFT_DIR" \
    --prompt "$PROMPT" \
    --max-new-tokens 1 \
    --warmup-tokens 1 \
    --measurement-runs "$MEASUREMENT_RUNS" \
    --cooldown-seconds 0 \
    --run-root "$RUN_ROOT"

RUN_DIR="$(find "$RUN_ROOT" -maxdepth 1 -type d -name '20*' -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-)"
if [ -z "$RUN_DIR" ]; then
    echo "failed to locate latest run dir under $RUN_ROOT" >&2
    exit 1
fi

quality_args=()
if [ "${FAIL_EXPLORATORY:-0}" = "1" ]; then
    quality_args+=(--fail-exploratory)
fi

python3 -m oracle.bench.specprefill_quality \
    --binary ./target/release/supersonic \
    --model-dir "$TARGET_DIR" \
    --draft-dir "$DRAFT_DIR" \
    --run "$RUN_DIR" \
    --keep-ratios "$KEEP_RATIOS" \
    --prompt-set "$PROMPT_SET" \
    --enforce-thresholds \
    "${quality_args[@]}"

python3 -m oracle.bench.render.render_main markdown \
    --run "$RUN_DIR" \
    --out . \
    --perf-zone-key bench-perf-matrix-sm86

echo "run_dir=$RUN_DIR"

popd >/dev/null
