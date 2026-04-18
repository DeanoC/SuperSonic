#!/usr/bin/env bash
#
# Lightweight sm86 CUDA benchmark harness for validated CUDA paths.
# Measures native prefill and decode timing on qwen3.5-0.8b, qwen3.5-4b,
# and qwen3.5-4b batched decode.
#
# Usage:
#   ./tests/sm86/bench.sh <model_dir_0_8b> <model_dir_4b>
#
# Environment:
#   RUNS=3                  number of repeated runs per model
#   MAX_NEW_TOKENS=32       decode length
#   PROMPT_REPEAT=32        repetition count for the synthetic prompt body
#   BATCH_SIZE_4B=2         batch size for the 4B batch benchmark
#   SUPERSONIC_BACKENDS=cuda
#
set -euo pipefail

MODEL_DIR_0_8B="${1:-${SUPERSONIC_MODEL_DIR:-}}"
MODEL_DIR_4B="${2:-${SUPERSONIC_MODEL_DIR_4B:-}}"

if [ -z "$MODEL_DIR_0_8B" ] || [ -z "$MODEL_DIR_4B" ]; then
    echo "Usage: $0 <path-to-Qwen3.5-0.8B> <path-to-Qwen3.5-4B>"
    echo "  or set SUPERSONIC_MODEL_DIR and SUPERSONIC_MODEL_DIR_4B"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f /root/.cargo/env ]; then
    . /root/.cargo/env
fi

RUNS="${RUNS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
PROMPT_REPEAT="${PROMPT_REPEAT:-32}"
BATCH_SIZE_4B="${BATCH_SIZE_4B:-2}"
export SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}"

echo "=== SuperSonic sm86 CUDA Benchmark ==="
echo "Runs:            $RUNS"
echo "Max new tokens:  $MAX_NEW_TOKENS"
echo "Prompt repeat:   $PROMPT_REPEAT"
echo "4B batch size:   $BATCH_SIZE_4B"
echo ""

echo "--- Building (release) ---"
cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic
echo ""

SUPERSONIC="$REPO_ROOT/target/release/supersonic"

PROMPT="$(python3 - <<PY
repeat = int("$PROMPT_REPEAT")
base = "SuperSonic CUDA benchmark context for qwen validation and decode throughput. "
print((base * repeat).strip())
PY
)"

run_model() {
    local label="$1"
    local model="$2"
    local model_dir="$3"
    local batch_size="$4"

    echo "--- $label ($model, batch_size=$batch_size) ---"
    python3 - "$SUPERSONIC" "$model" "$model_dir" "$PROMPT" "$MAX_NEW_TOKENS" "$RUNS" "$batch_size" <<'PY'
import re
import statistics
import subprocess
import sys

binary, model, model_dir, prompt, max_new_tokens, runs, batch_size = sys.argv[1:]
max_new_tokens = int(max_new_tokens)
runs = int(runs)
batch_size = int(batch_size)

prefill_pat = re.compile(r"\[prefill\] native GPU prefill done in (\d+)ms")
result_pat = re.compile(
    r"\[result\] prompt_tokens=(\d+) generated_tokens=(\d+) decode_ms=(\d+) ms_per_tok=(\d+)"
)

prefill_ms = []
prompt_tokens = []
decode_ms = []
generated_tokens = []

for idx in range(runs):
    cmd = [
        binary,
        "--backend", "cuda",
        "--model", model,
        "--model-dir", model_dir,
        "--prompt", prompt,
        "--max-new-tokens", str(max_new_tokens),
        "--batch-size", str(batch_size),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        raise SystemExit(f"benchmark run {idx + 1} failed")

    prefill_match = prefill_pat.search(proc.stderr)
    result_match = result_pat.search(proc.stderr)
    if not prefill_match or not result_match:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"failed to parse benchmark output for run {idx + 1}")

    prefill_ms.append(int(prefill_match.group(1)))
    prompt_tokens.append(int(result_match.group(1)))
    generated_tokens.append(int(result_match.group(2)))
    decode_ms.append(int(result_match.group(3)))

ptoks = prompt_tokens[0]
gtoks = generated_tokens[0]
prefill_mean = statistics.mean(prefill_ms)
decode_mean = statistics.mean(decode_ms)
prefill_toks_s = (ptoks / prefill_mean) * 1000.0 if prefill_mean else 0.0
decode_toks_s = ((gtoks * batch_size) / decode_mean) * 1000.0 if decode_mean else 0.0

print(f"prompt_tokens={ptoks} generated_tokens={gtoks}")
print(f"prefill_ms={prefill_ms} mean={prefill_mean:.1f} prefill_toks_s={prefill_toks_s:.1f}")
print(f"decode_ms={decode_ms} mean={decode_mean:.1f} decode_toks_s={decode_toks_s:.1f} aggregate_batch_tokens={gtoks * batch_size}")
PY
    echo ""
}

run_model "Model 1" "qwen3.5-0.8b" "$MODEL_DIR_0_8B" 1
run_model "Model 2" "qwen3.5-4b" "$MODEL_DIR_4B" 1
run_model "Model 3" "qwen3.5-4b" "$MODEL_DIR_4B" "$BATCH_SIZE_4B"
