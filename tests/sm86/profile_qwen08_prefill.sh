#!/usr/bin/env bash
#
# HAL-level prefill profile for Qwen3.5-0.8B on CUDA sm86.
#
# Usage:
#   ./tests/sm86/profile_qwen08_prefill.sh [model_dir]
#
# Environment:
#   TARGET_PROMPT_TOKENS=520
#   PROFILE_JSON=/tmp/supersonic-qwen08-prefill-profile.json
#   SUPERSONIC_CUDA_PREFILL_FORCE_SCALAR=1  compare against scalar prefill GEMM
#
set -euo pipefail

MODEL_DIR="${1:-${SUPERSONIC_MODEL_DIR:-}}"
if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <path-to-Qwen3.5-0.8B>"
    echo "  or set SUPERSONIC_MODEL_DIR"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f /root/.cargo/env ]; then
    . /root/.cargo/env
fi

TARGET_PROMPT_TOKENS="${TARGET_PROMPT_TOKENS:-520}"
PROFILE_JSON="${PROFILE_JSON:-/tmp/supersonic-qwen08-prefill-profile.json}"
export SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}"

cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic

python3 - "$REPO_ROOT/target/release/supersonic" "$MODEL_DIR" "$TARGET_PROMPT_TOKENS" "$PROFILE_JSON" <<'PY'
import subprocess
import sys

binary, model_dir, target_prompt_tokens, profile_json = sys.argv[1:]
target_prompt_tokens = int(target_prompt_tokens)
base = "SuperSonic CUDA prefill profile sentence for prompt calibration. "

def make_prompt(repeat: int) -> str:
    return (base * repeat).strip()

repeat = max(1, target_prompt_tokens // 14)
prompt = make_prompt(repeat)
cmd = [
    binary,
    "--backend", "cuda",
    "--model", "qwen3.5-0.8b",
    "--model-dir", model_dir,
    "--prompt", prompt,
    "--max-new-tokens", "1",
    "--profile-prefill",
    "--profile-prefill-json", profile_json,
]
subprocess.run(cmd, check=True)
print(f"profile_json={profile_json}")
PY
