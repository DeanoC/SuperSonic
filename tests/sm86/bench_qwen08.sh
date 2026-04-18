#!/usr/bin/env bash
#
# Lucebox-style Qwen3.5-0.8B CUDA parity benchmark for sm86.
# Measures warmed batch-1 BF16 decode with optional legacy comparison.
#
# Usage:
#   ./tests/sm86/bench_qwen08.sh [model_dir]
#
# Environment:
#   TARGET_PROMPT_TOKENS=520  approximate prompt length target
#   MAX_NEW_TOKENS=128        generated tokens per timed run
#   WARMUP_RUNS=10           untimed warmup runs
#   TIMED_RUNS=20            timed runs to average
#   COMPARE_LEGACY=1         also run the legacy host-logits sampling path
#   SUPERSONIC_BACKENDS=cuda
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
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
WARMUP_RUNS="${WARMUP_RUNS:-10}"
TIMED_RUNS="${TIMED_RUNS:-20}"
COMPARE_LEGACY="${COMPARE_LEGACY:-1}"
export SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}"

echo "=== SuperSonic sm86 Qwen3.5-0.8B Parity Benchmark ==="
echo "Target prompt tokens: $TARGET_PROMPT_TOKENS"
echo "Max new tokens:       $MAX_NEW_TOKENS"
echo "Warmup runs:          $WARMUP_RUNS"
echo "Timed runs:           $TIMED_RUNS"
echo "Compare legacy:       $COMPARE_LEGACY"
echo ""

echo "--- Building (release) ---"
cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic
echo ""

SUPERSONIC="$REPO_ROOT/target/release/supersonic"

python3 - "$SUPERSONIC" "$MODEL_DIR" "$TARGET_PROMPT_TOKENS" "$MAX_NEW_TOKENS" "$WARMUP_RUNS" "$TIMED_RUNS" "$COMPARE_LEGACY" <<'PY'
import os
import re
import statistics
import subprocess
import sys

binary, model_dir, target_prompt_tokens, max_new_tokens, warmup_runs, timed_runs, compare_legacy = sys.argv[1:]
target_prompt_tokens = int(target_prompt_tokens)
max_new_tokens = int(max_new_tokens)
warmup_runs = int(warmup_runs)
timed_runs = int(timed_runs)
compare_legacy = compare_legacy not in ("0", "false", "False")

prefill_pat = re.compile(r"\[prefill\] native GPU prefill done in (\d+)ms")
result_pat = re.compile(
    r"\[result\] prompt_tokens=(\d+) generated_tokens=(\d+) decode_ms=(\d+) ms_per_tok=(\d+)"
)
stage_pat = re.compile(
    r"\[stage-timings\] steps=(\d+) persistent_ms=([0-9.]+) rms_norm_ms=([0-9.]+) "
    r"lm_head_ms=([0-9.]+) logits_d2h_ms=([0-9.]+) host_sampling_ms=([0-9.]+) "
    r"gpu_argmax_ms=([0-9.]+) token_d2h_ms=([0-9.]+) total_native_decode_ms=([0-9.]+)"
)

base = "SuperSonic parity benchmark sentence for warmed Qwen decode throughput. "

def make_prompt(repeat: int) -> str:
    return (base * repeat).strip()

def run_once(prompt: str, legacy: bool, run_max_new_tokens: int):
    env = os.environ.copy()
    if legacy:
        env["SUPERSONIC_DISABLE_CUDA_08B_HERO"] = "1"
        env["SUPERSONIC_DISABLE_CUDA_FAST_GREEDY"] = "1"
    cmd = [
        binary,
        "--backend", "cuda",
        "--model", "qwen3.5-0.8b",
        "--model-dir", model_dir,
        "--prompt", prompt,
        "--max-new-tokens", str(run_max_new_tokens),
        "--emit-stage-timings",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        print(proc.stdout, file=sys.stderr)
        raise SystemExit("benchmark run failed")
    stderr = proc.stderr
    prefill = prefill_pat.search(stderr)
    result = result_pat.search(stderr)
    stage = stage_pat.search(stderr)
    if not prefill or not result:
        print(stderr, file=sys.stderr)
        raise SystemExit("failed to parse benchmark output")
    if not stage:
        print(stderr, file=sys.stderr)
        raise SystemExit("failed to parse stage timings output")
    return {
        "prompt_tokens": int(result.group(1)),
        "generated_tokens": int(result.group(2)),
        "prefill_ms": int(prefill.group(1)),
        "decode_ms": int(result.group(3)),
        "steps": int(stage.group(1)),
        "persistent_ms": float(stage.group(2)),
        "rms_norm_ms": float(stage.group(3)),
        "lm_head_ms": float(stage.group(4)),
        "logits_d2h_ms": float(stage.group(5)),
        "host_sampling_ms": float(stage.group(6)),
        "gpu_argmax_ms": float(stage.group(7)),
        "token_d2h_ms": float(stage.group(8)),
        "total_native_decode_ms": float(stage.group(9)),
    }

repeat = 32
prompt = make_prompt(repeat)
probe = run_once(prompt, legacy=False, run_max_new_tokens=1)
for _ in range(6):
    if probe["prompt_tokens"] >= target_prompt_tokens:
        break
    scale = max(target_prompt_tokens / max(probe["prompt_tokens"], 1), 1.1)
    repeat = max(int(repeat * scale) + 1, repeat + 1)
    prompt = make_prompt(repeat)
    probe = run_once(prompt, legacy=False, run_max_new_tokens=1)

prompt = make_prompt(repeat)
prompt_tokens = probe["prompt_tokens"]
print(f"Calibrated prompt_repeat={repeat} actual_prompt_tokens={prompt_tokens}")

def benchmark(label: str, legacy: bool):
    for _ in range(warmup_runs):
        run_once(prompt, legacy=legacy, run_max_new_tokens=max_new_tokens)

    runs = [run_once(prompt, legacy=legacy, run_max_new_tokens=max_new_tokens) for _ in range(timed_runs)]
    generated_tokens = runs[0]["generated_tokens"]
    prefill_mean = statistics.mean(r["prefill_ms"] for r in runs)
    decode_mean = statistics.mean(r["decode_ms"] for r in runs)
    persistent_mean = statistics.mean(r["persistent_ms"] for r in runs)
    rms_mean = statistics.mean(r["rms_norm_ms"] for r in runs)
    lm_head_mean = statistics.mean(r["lm_head_ms"] for r in runs)
    logits_d2h_mean = statistics.mean(r["logits_d2h_ms"] for r in runs)
    host_sampling_mean = statistics.mean(r["host_sampling_ms"] for r in runs)
    gpu_argmax_mean = statistics.mean(r["gpu_argmax_ms"] for r in runs)
    token_d2h_mean = statistics.mean(r["token_d2h_ms"] for r in runs)
    native_mean = statistics.mean(r["total_native_decode_ms"] for r in runs)

    prefill_toks_s = (prompt_tokens / prefill_mean) * 1000.0 if prefill_mean else 0.0
    decode_toks_s = (generated_tokens / decode_mean) * 1000.0 if decode_mean else 0.0
    print(f"\n--- {label} ---")
    print(f"prompt_tokens={prompt_tokens} generated_tokens={generated_tokens}")
    print(f"prefill_ms_mean={prefill_mean:.1f} prefill_toks_s={prefill_toks_s:.1f}")
    print(f"decode_ms_mean={decode_mean:.1f} decode_toks_s={decode_toks_s:.1f}")
    print(
        "stage_ms_mean "
        f"persistent={persistent_mean:.3f} rms_norm={rms_mean:.3f} lm_head={lm_head_mean:.3f} "
        f"logits_d2h={logits_d2h_mean:.3f} host_sampling={host_sampling_mean:.3f} "
        f"gpu_argmax={gpu_argmax_mean:.3f} token_d2h={token_d2h_mean:.3f} "
        f"native_total={native_mean:.3f}"
    )

benchmark("fast-greedy", legacy=False)
if compare_legacy:
    benchmark("legacy-host-logits", legacy=True)
PY
