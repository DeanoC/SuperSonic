#!/usr/bin/env bash
#
# Starter smoke matrix for HIP gfx1201 (RDNA 4 / Radeon AI PRO R9700).
#
# This is intentionally narrower than tests/gfx1100/run_matrix.sh while the
# target is in bring-up: kernel-level RDNA4 WMMA coverage first, then the
# Qwen3.6 27B Q4KM-GPTQ/Lucebox lane that motivated the new target. Broader
# Qwen3.5 matrix promotion should add explicit token/parity checks here.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INT4_TEST="$REPO_ROOT/target/release/int4_test"
SUPERSONIC="$REPO_ROOT/target/release/supersonic"

HIP_ARCH="${HIP_ARCH:-gfx1100,gfx1201}"
BUILD="${BUILD:-1}"
TIMEOUT="${TIMEOUT:-900}"
PROMPT="${PROMPT:-The quick brown fox}"
QWEN36_27B_MODEL_DIR="${QWEN36_27B_MODEL_DIR:-${SUPERSONIC_MODEL_DIR_QWEN36_27B:-/mnt/data/tmp/supersonic-qwen36-27b-lucebox}}"
QWEN36_CONTEXT="${QWEN36_CONTEXT:-512}"
QWEN36_DENSE_TOKENS="${QWEN36_DENSE_TOKENS:-2}"

RUN_LUCEBOX="${RUN_LUCEBOX:-auto}"
LUCEBOX_N_GEN="${LUCEBOX_N_GEN:-16}"
LUCEBOX_WARMUP_NEW_TOKENS="${LUCEBOX_WARMUP_NEW_TOKENS:-4}"
LUCEBOX_LIMIT="${LUCEBOX_LIMIT:-1}"
LUCEBOX_OUT_JSON="${LUCEBOX_OUT_JSON:-$REPO_ROOT/target/qwen36_he_supersonic_gfx1201_smoke.json}"
LUCEBOX_DRAFT_DIR="${LUCEBOX_DRAFT_DIR:-/mnt/data/tmp/qwen36-27b-dflash-q8-bf16}"
LUCEBOX_Q8_DRAFT_GGUF="${LUCEBOX_Q8_DRAFT_GGUF:-/mnt/data/lucebox-hub/models/draft/dflash-draft-3.6-q8_0.gguf}"

echo "=== SuperSonic gfx1201 starter matrix (RDNA 4 / R9700) ==="
echo "HIP_ARCH: ${HIP_ARCH}"
if [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
    echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES}"
fi
echo ""

if [ "$BUILD" != "0" ]; then
    echo "--- Build dual-arch HIP binaries ---"
    env SUPERSONIC_BACKENDS=hip HIP_ARCH="$HIP_ARCH" \
        cargo build --release --bin int4_test --bin supersonic
    echo ""
fi

if [ ! -x "$INT4_TEST" ]; then
    echo "ERROR: $INT4_TEST not found. Run with BUILD=1 or build int4_test first." >&2
    exit 1
fi
if [ ! -x "$SUPERSONIC" ]; then
    echo "ERROR: $SUPERSONIC not found. Run with BUILD=1 or build supersonic first." >&2
    exit 1
fi

echo "--- RDNA4 WMMA / INT4 kernel harness ---"
timeout "$TIMEOUT" env SUPERSONIC_BACKENDS=hip "$INT4_TEST"
echo ""

echo "--- Qwen3.6 27B Q4KM-GPTQ direct smoke ---"
if [ -f "$QWEN36_27B_MODEL_DIR/config.json" ]; then
    timeout "$TIMEOUT" env SUPERSONIC_BACKENDS=hip "$SUPERSONIC" \
        --backend hip \
        --model qwen3.6-27b \
        --model-dir "$QWEN36_27B_MODEL_DIR" \
        --prompt "$PROMPT" \
        --context-size "$QWEN36_CONTEXT" \
        --max-new-tokens "$QWEN36_DENSE_TOKENS" \
        --temperature 0 \
        --top-k 1 \
        --sampling-seed 20260620 \
        --q4km-gptq \
        --no-download
else
    echo "SKIP: QWEN36_27B_MODEL_DIR missing config.json: $QWEN36_27B_MODEL_DIR"
fi
echo ""

lucebox_available=0
if [ -f "$QWEN36_27B_MODEL_DIR/config.json" ] && \
   [ -d "$LUCEBOX_DRAFT_DIR" ] && \
   [ -f "$LUCEBOX_Q8_DRAFT_GGUF" ]; then
    lucebox_available=1
fi

case "$RUN_LUCEBOX" in
    auto)
        run_lucebox="$lucebox_available"
        require_lucebox=0
        ;;
    1|true|TRUE|yes|YES)
        run_lucebox=1
        require_lucebox=1
        ;;
    0|false|FALSE|no|NO)
        run_lucebox=0
        require_lucebox=0
        ;;
    *)
        echo "ERROR: RUN_LUCEBOX must be auto, 1, or 0 (got '$RUN_LUCEBOX')" >&2
        exit 1
        ;;
esac

echo "--- Qwen3.6 27B Lucebox/DFlash smoke ---"
if [ "$run_lucebox" = "1" ]; then
    if [ "$lucebox_available" != "1" ]; then
        echo "ERROR: Lucebox artifacts are missing:" >&2
        echo "  QWEN36_27B_MODEL_DIR=$QWEN36_27B_MODEL_DIR" >&2
        echo "  LUCEBOX_DRAFT_DIR=$LUCEBOX_DRAFT_DIR" >&2
        echo "  LUCEBOX_Q8_DRAFT_GGUF=$LUCEBOX_Q8_DRAFT_GGUF" >&2
        exit 1
    fi
    timeout "$TIMEOUT" env SUPERSONIC_BACKENDS=hip python3 \
        "$REPO_ROOT/tests/gfx1100/bench_qwen36_he_supersonic.py" \
        --binary "$SUPERSONIC" \
        --target-profile qwen36-27b-lucebox \
        --model-dir "$QWEN36_27B_MODEL_DIR" \
        --backend hip \
        --dflash \
        --dflash-draft-dir "$LUCEBOX_DRAFT_DIR" \
        --dflash-draft-gguf "$LUCEBOX_Q8_DRAFT_GGUF" \
        --n-gen "$LUCEBOX_N_GEN" \
        --warmup-new-tokens "$LUCEBOX_WARMUP_NEW_TOKENS" \
        --limit "$LUCEBOX_LIMIT" \
        --no-warmup \
        --out-json "$LUCEBOX_OUT_JSON"
else
    if [ "$require_lucebox" = "1" ]; then
        exit 1
    fi
    echo "SKIP: Lucebox artifacts not available or RUN_LUCEBOX=0"
fi

echo ""
echo "gfx1201 starter matrix complete"
