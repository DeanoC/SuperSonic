#!/usr/bin/env bash
#
# HAL/lifecycle prefill profile for Qwen3.6-MoE 35B-A3B on CUDA sm86.
#
# Usage:
#   ./tests/sm86/profile_qwen36_prefill.sh [model_dir]
#
# Environment:
#   PROMPT_REPEAT=12
#   PROFILE_JSON=/tmp/supersonic-qwen36-prefill-profile.json
#
set -euo pipefail

MODEL_DIR="${1:-${SUPERSONIC_MODEL_DIR_QWEN36:-${SUPERSONIC_MODEL_DIR:-}}}"
if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <path-to-Qwen3.6-35B-A3B>"
    echo "  or set SUPERSONIC_MODEL_DIR_QWEN36"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f /root/.cargo/env ]; then
    . /root/.cargo/env
fi

PROMPT_REPEAT="${PROMPT_REPEAT:-12}"
PROFILE_JSON="${PROFILE_JSON:-/tmp/supersonic-qwen36-prefill-profile.json}"
export SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}"

cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic

PROMPT="$(
    python3 - "$PROMPT_REPEAT" <<'PY'
import sys
repeat = int(sys.argv[1])
print(("SuperSonic CUDA dense prefill profiling sentence. " * repeat).strip())
PY
)"

"$REPO_ROOT/target/release/supersonic" \
    --backend cuda \
    --model qwen3.6-35b-a3b \
    --model-dir "$MODEL_DIR" \
    --prompt "$PROMPT" \
    --max-new-tokens 1 \
    --profile-prefill \
    --profile-prefill-json "$PROFILE_JSON" \
    --emit-stage-timings

echo "profile_json=$PROFILE_JSON"
