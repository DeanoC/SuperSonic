#!/usr/bin/env bash
#
# Regression check for the CUDA fast-greedy path on qwen3.5-0.8b.
# Compares generated token ids between the fast CUDA argmax path and the legacy
# full-logits host-sampling path across short, medium, and long prompts.
#
# Usage:
#   ./tests/sm86/run_fast_greedy.sh [model_dir]
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

TIMEOUT="${TIMEOUT:-300}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
export SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}"

echo "=== SuperSonic Fast-Greedy Regression: sm86 / qwen3.5-0.8b ==="
echo "Model dir:  $MODEL_DIR"
echo ""

echo "--- Building (release) ---"
cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic
echo ""

SUPERSONIC="$REPO_ROOT/target/release/supersonic"

python3 - "$SUPERSONIC" "$MODEL_DIR" "$TIMEOUT" "$MAX_NEW_TOKENS" <<'PY'
import os
import re
import subprocess
import sys

binary, model_dir, timeout_s, max_new_tokens = sys.argv[1:]
timeout_s = int(timeout_s)
max_new_tokens = int(max_new_tokens)

token_pat = re.compile(r"^\[tokens\] (.*)$", re.MULTILINE)

prompts = {
    "short": "Summarize the phrase 'safety before speed' in three words.",
    "medium": "Explain why persistent GPU decode kernels can help inference throughput without changing the model's math.",
    "long": ("Archive note alpha tracks a cache refill and kernel launch. " * 80).strip(),
}

def run(prompt: str, legacy: bool):
    env = os.environ.copy()
    env["SUPERSONIC_DISABLE_CUDA_08B_HERO"] = "1"
    if legacy:
        env["SUPERSONIC_DISABLE_CUDA_FAST_GREEDY"] = "1"
    cmd = [
        binary,
        "--backend", "cuda",
        "--model", "qwen3.5-0.8b",
        "--model-dir", model_dir,
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--emit-stage-timings",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout_s)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit("decode run failed")
    token_match = token_pat.search(proc.stdout)
    if not token_match:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit("missing [tokens] line")
    if "[stage-timings]" not in proc.stderr:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit("missing [stage-timings] line")
    return token_match.group(1).strip()

for label, prompt in prompts.items():
    fast_tokens = run(prompt, legacy=False)
    legacy_tokens = run(prompt, legacy=True)
    print(f"{label}: fast=[{fast_tokens}] legacy=[{legacy_tokens}]")
    if fast_tokens != legacy_tokens:
        raise SystemExit(f"{label}: generated token mismatch")

print("all prompt comparisons matched")
PY
