#!/usr/bin/env bash
#
# Warmed Llama 3.1 8B CUDA benchmark for the staged certified KV path.
# Current Rust/CUDA mode is the correctness-first dense-fallback baseline.
#
# Usage:
#   ./tests/sm86/bench_llama31_certified_kv.sh [model_dir]
#
set -euo pipefail

MODEL_DIR="${1:-${SUPERSONIC_MODEL_DIR_LLAMA31_8B:-}}"
if [ -z "$MODEL_DIR" ]; then
    echo "Usage: $0 <path-to-Meta-Llama-3.1-8B>"
    echo "  or set SUPERSONIC_MODEL_DIR_LLAMA31_8B"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f /root/.cargo/env ]; then
    . /root/.cargo/env
fi

TARGET_PROMPT_TOKENS="${TARGET_PROMPT_TOKENS:-520}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
WARMUP_RUNS="${WARMUP_RUNS:-3}"
TIMED_RUNS="${TIMED_RUNS:-5}"
CERTIFIED_KV_SHADOW_VALIDATE="${CERTIFIED_KV_SHADOW_VALIDATE:-0}"
export SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}"

echo "=== SuperSonic sm86 Llama 3.1 8B Certified KV Benchmark ==="
echo "Target prompt tokens: $TARGET_PROMPT_TOKENS"
echo "Max new tokens:       $MAX_NEW_TOKENS"
echo "Warmup runs:          $WARMUP_RUNS"
echo "Timed runs:           $TIMED_RUNS"
echo "Shadow validate:      $CERTIFIED_KV_SHADOW_VALIDATE"
echo ""

echo "--- Building (release) ---"
cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin supersonic
echo ""

SUPERSONIC="$REPO_ROOT/target/release/supersonic"

python3 - "$SUPERSONIC" "$MODEL_DIR" "$TARGET_PROMPT_TOKENS" "$MAX_NEW_TOKENS" "$WARMUP_RUNS" "$TIMED_RUNS" <<'PY'
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile

binary, model_dir, target_prompt_tokens, max_new_tokens, warmup_runs, timed_runs = sys.argv[1:]
target_prompt_tokens = int(target_prompt_tokens)
max_new_tokens = int(max_new_tokens)
warmup_runs = int(warmup_runs)
timed_runs = int(timed_runs)

prefill_pat = re.compile(r"\[prefill\] (\d+) tokens in (\d+)ms")
result_pat = re.compile(
    r"\[result\] prompt_tokens=(\d+) generated_tokens=(\d+) decode_ms=([0-9.]+) ms_per_step=([0-9.]+)"
)
stage_pat = re.compile(
    r"\[stage\] tokens=(\d+) total_ms=([0-9.]+) per_tok_ms=([0-9.]+) "
    r"layer_compute=([0-9.]+) full_attn=([0-9.]+) full_attn_proj=([0-9.]+) "
    r"full_attn_core=([0-9.]+) full_attn_out=([0-9.]+) linear=([0-9.]+) "
    r"mlp=([0-9.]+) rms_norm=([0-9.]+) lm_head=([0-9.]+) "
    r"logits_d2h=([0-9.]+) host_sampling=([0-9.]+)"
)

base = "SuperSonic CUDA certified KV benchmark context for Meta Llama three point one decode throughput. "

def make_prompt(repeat: int) -> str:
    return (base * repeat).strip()

def run_once(prompt: str, run_max_new_tokens: int):
    with tempfile.NamedTemporaryFile(prefix="certified-kv-", suffix=".jsonl") as telemetry:
        cmd = [
            binary,
            "--backend", "cuda",
            "--model", "llama3.1-8b",
            "--model-dir", model_dir,
            "--prompt", prompt,
            "--max-new-tokens", str(run_max_new_tokens),
            "--int8",
            "--certified-kv",
            "--certified-kv-telemetry", telemetry.name,
            "--emit-stage-timings",
        ]
        if os.environ.get("CERTIFIED_KV_SHADOW_VALIDATE") == "1":
            cmd.append("--certified-kv-shadow-validate")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            print(proc.stdout, file=sys.stderr)
            raise SystemExit("benchmark run failed")
        stderr = proc.stderr
        prefill = prefill_pat.search(stderr)
        result = result_pat.search(stderr)
        if not prefill or not result:
            print(stderr, file=sys.stderr)
            raise SystemExit("failed to parse benchmark output")
        stage = stage_pat.search(stderr)
        if run_max_new_tokens > 1 and not stage:
            print(stderr, file=sys.stderr)
            raise SystemExit("failed to parse stage timings output")
        telemetry.seek(0)
        telemetry_lines = [json.loads(line) for line in telemetry.read().decode().splitlines() if line.strip()]
    return {
        "prompt_tokens": int(result.group(1)),
        "generated_tokens": int(result.group(2)),
        "prefill_ms": int(prefill.group(2)),
        "decode_ms": float(result.group(3)),
        "ms_per_step": float(result.group(4)),
        "stage_tokens": int(stage.group(1)) if stage else 0,
        "stage_total_ms": float(stage.group(2)) if stage else 0.0,
        "stage_per_tok_ms": float(stage.group(3)) if stage else 0.0,
        "layer_compute_ms": float(stage.group(4)) if stage else 0.0,
        "full_attn_ms": float(stage.group(5)) if stage else 0.0,
        "full_attn_proj_ms": float(stage.group(6)) if stage else 0.0,
        "full_attn_core_ms": float(stage.group(7)) if stage else 0.0,
        "full_attn_out_ms": float(stage.group(8)) if stage else 0.0,
        "linear_ms": float(stage.group(9)) if stage else 0.0,
        "mlp_ms": float(stage.group(10)) if stage else 0.0,
        "rms_norm_ms": float(stage.group(11)) if stage else 0.0,
        "lm_head_ms": float(stage.group(12)) if stage else 0.0,
        "logits_d2h_ms": float(stage.group(13)) if stage else 0.0,
        "host_sampling_ms": float(stage.group(14)) if stage else 0.0,
        "telemetry": telemetry_lines[-1] if telemetry_lines else {},
    }

repeat = 32
prompt = make_prompt(repeat)
probe = run_once(prompt, run_max_new_tokens=1)
for _ in range(6):
    if probe["prompt_tokens"] >= target_prompt_tokens:
        break
    scale = max(target_prompt_tokens / max(probe["prompt_tokens"], 1), 1.1)
    repeat = max(int(repeat * scale) + 1, repeat + 1)
    prompt = make_prompt(repeat)
    probe = run_once(prompt, run_max_new_tokens=1)

prompt = make_prompt(repeat)
prompt_tokens = probe["prompt_tokens"]
print(f"Calibrated prompt_repeat={repeat} actual_prompt_tokens={prompt_tokens}")

for _ in range(warmup_runs):
    run_once(prompt, run_max_new_tokens=max_new_tokens)

runs = [run_once(prompt, run_max_new_tokens=max_new_tokens) for _ in range(timed_runs)]
generated_tokens = runs[0]["generated_tokens"]
prefill_mean = statistics.mean(r["prefill_ms"] for r in runs)
decode_mean = statistics.mean(r["decode_ms"] for r in runs)
ms_per_step_mean = statistics.mean(r["ms_per_step"] for r in runs)
stage_tokens = runs[0]["stage_tokens"]
stage_total_mean = statistics.mean(r["stage_total_ms"] for r in runs)
layer_compute_mean = statistics.mean(r["layer_compute_ms"] for r in runs)
full_attn_mean = statistics.mean(r["full_attn_ms"] for r in runs)
full_attn_core_mean = statistics.mean(r["full_attn_core_ms"] for r in runs)
mlp_mean = statistics.mean(r["mlp_ms"] for r in runs)
rms_mean = statistics.mean(r["rms_norm_ms"] for r in runs)
lm_head_mean = statistics.mean(r["lm_head_ms"] for r in runs)
logits_d2h_mean = statistics.mean(r["logits_d2h_ms"] for r in runs)
host_sampling_mean = statistics.mean(r["host_sampling_ms"] for r in runs)
rung4_steps = statistics.mean(r["telemetry"].get("rung4_forced_dense_steps", 0) for r in runs)
shadow_runs = [r["telemetry"] for r in runs if r["telemetry"].get("shadow_layers") is not None]

prefill_toks_s = (prompt_tokens / prefill_mean) * 1000.0 if prefill_mean else 0.0
decode_toks_s = (generated_tokens / decode_mean) * 1000.0 if decode_mean else 0.0

print("")
print("--- llama3.1-8b certified-kv dense-fallback lane ---")
print(f"prompt_tokens={prompt_tokens} generated_tokens={generated_tokens} batch_size=1 mode=certified_kv_dense_fallback")
print(f"prefill_ms_mean={prefill_mean:.1f} prefill_toks_s={prefill_toks_s:.1f}")
print(f"decode_ms_mean={decode_mean:.1f} decode_toks_s={decode_toks_s:.1f} ms_per_step_mean={ms_per_step_mean:.1f}")
print(
    "stage_ms_mean "
    f"stage_tokens={stage_tokens} total={stage_total_mean:.3f} layer_compute={layer_compute_mean:.3f} "
    f"full_attn={full_attn_mean:.3f} full_attn_core={full_attn_core_mean:.3f} mlp={mlp_mean:.3f} "
    f"rms_norm={rms_mean:.3f} lm_head={lm_head_mean:.3f} logits_d2h={logits_d2h_mean:.3f} "
    f"host_sampling={host_sampling_mean:.3f} forced_rung4_steps={rung4_steps:.1f}"
)
if shadow_runs:
    shadow_layers = statistics.mean(r.get("shadow_layers", 0) for r in shadow_runs)
    shadow_tokens = statistics.mean(r.get("shadow_aligned_tokens", 0) for r in shadow_runs)
    shadow_bytes = statistics.mean(r.get("shadow_tier1_bytes", 0) for r in shadow_runs)
    shadow_ms = statistics.mean(r.get("shadow_quantize_ms", 0.0) for r in shadow_runs)
    shadow_err = max(r.get("shadow_max_value_error", 0.0) for r in shadow_runs)
    shadow_score_layers = statistics.mean(r.get("shadow_score_layers", 0) for r in shadow_runs)
    shadow_score_ms = statistics.mean(r.get("shadow_score_ms", 0.0) for r in shadow_runs)
    shadow_score_delta = max(r.get("shadow_max_score_ref_delta", 0.0) for r in shadow_runs)
    shadow_selector_heads = statistics.mean(r.get("shadow_selector_heads", 0) for r in shadow_runs)
    shadow_selector_mean_k = statistics.mean(r.get("shadow_selector_mean_k", 0.0) for r in shadow_runs)
    shadow_selector_tail = max(r.get("shadow_selector_max_tail_mass", 0.0) for r in shadow_runs)
    shadow_selector_rung1 = statistics.mean(r.get("shadow_selector_rung1_heads", 0) for r in shadow_runs)
    print(
        "shadow_mean "
        f"layers={shadow_layers:.1f} aligned_tokens={shadow_tokens:.1f} "
        f"tier1_bytes={shadow_bytes:.0f} quantize_ms={shadow_ms:.3f} "
        f"max_value_error={shadow_err:.6f} score_layers={shadow_score_layers:.1f} "
        f"score_ms={shadow_score_ms:.3f} max_score_ref_delta={shadow_score_delta:.6f} "
        f"selector_heads={shadow_selector_heads:.1f} selector_mean_k={shadow_selector_mean_k:.2f} "
        f"selector_max_tail_mass={shadow_selector_tail:.6f} selector_rung1_heads={shadow_selector_rung1:.1f}"
    )
PY
