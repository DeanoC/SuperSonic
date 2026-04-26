#!/usr/bin/env bash
#
# Warmed synthetic benchmark for the Llama3.1 certified-KV PyTorch oracle.
#
# This benchmarks the reference algorithm at one attention layer, not the full
# SuperSonic runtime. It gives stable baseline numbers and telemetry before the
# Rust/CUDA path is wired into llama31_engine.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEVICE="${CERT_KV_ORACLE_DEVICE:-cuda}"
if ! python3 - >/dev/null 2>&1 <<'PY'; then
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
    DEVICE="cpu"
fi

TOKENS="${CERT_KV_BENCH_TOKENS:-512,2048,8192}"
WARMUP_RUNS="${WARMUP_RUNS:-3}"
TIMED_RUNS="${TIMED_RUNS:-10}"

echo "=== SuperSonic sm86 Llama3.1 certified-KV oracle benchmark ==="
echo "Device:      $DEVICE"
echo "Tokens:      $TOKENS"
echo "Warmups:     $WARMUP_RUNS"
echo "Timed runs:  $TIMED_RUNS"
echo ""

python3 - "$REPO_ROOT" "$DEVICE" "$TOKENS" "$WARMUP_RUNS" "$TIMED_RUNS" <<'PY'
import statistics
import sys
import time
from pathlib import Path

import torch

repo_root, device, tokens_csv, warmup_runs, timed_runs = sys.argv[1:]
sys.path.insert(0, str(Path(repo_root)))

from oracle.certified_kv_llama31 import (  # noqa: E402
    CertifiedKvConfig,
    build_tiered_kv_cache,
    certified_attention_step,
)

warmup_runs = int(warmup_runs)
timed_runs = int(timed_runs)
token_counts = [int(x) for x in tokens_csv.split(",") if x.strip()]

torch.manual_seed(20260423)
cfg = CertifiedKvConfig()
kv_heads = 8
gqa_group = 4
q_heads = kv_heads * gqa_group
head_dim = 128

def sync():
    if device.startswith("cuda"):
        torch.cuda.synchronize()

for tokens in token_counts:
    keys = torch.randn(kv_heads, tokens, head_dim, dtype=torch.float32, device=device)
    values = torch.randn_like(keys)
    q = torch.randn(q_heads, head_dim, dtype=torch.float32, device=device)

    sync()
    t0 = time.perf_counter()
    cache = build_tiered_kv_cache(keys, values, cfg)
    sync()
    build_ms = (time.perf_counter() - t0) * 1000.0

    for _ in range(warmup_runs):
        certified_attention_step(q, cache, gqa_group)
    sync()

    samples = []
    last_telem = None
    for _ in range(timed_runs):
        sync()
        start = time.perf_counter()
        _, last_telem = certified_attention_step(q, cache, gqa_group)
        sync()
        samples.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(samples)
    p50_ms = statistics.median(samples)
    p95_ms = sorted(samples)[max(0, int(len(samples) * 0.95) - 1)]
    print(
        "oracle_certified_kv "
        f"tokens={tokens} blocks={cache.num_blocks} build_ms={build_ms:.3f} "
        f"step_ms_mean={mean_ms:.3f} step_ms_p50={p50_ms:.3f} step_ms_p95={p95_ms:.3f} "
        f"k_star_mean={statistics.mean(last_telem['k_star']):.2f} "
        f"tail_mass_max={max(last_telem['tail_mass_int8_est']):.6f} "
        f"rung1={int(last_telem['rung1_fired'])} rung2={int(last_telem['rung2_fired'])}"
    )
PY

