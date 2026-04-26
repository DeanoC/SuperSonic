#!/usr/bin/env bash
#
# Synthetic certified-KV kernel microbench.
#
# This bypasses full model decode and profiles the certified-KV CUDA phases over
# synthetic Llama 3.1-shaped query/KV tensors. The BF16 source corpus is device
# staged so kernels can be timed directly; this is a kernel-scaling benchmark,
# not a runtime memory-residency conformance test.
#
# Environment:
#   CONTEXTS=4096,8192,16384,32768
#   ITERS=20
#   WARMUP=5
#   OUTPUT=target/llama31_certified_kv_microbench.json
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f /root/.cargo/env ]; then
    . /root/.cargo/env
fi

export SUPERSONIC_BACKENDS="${SUPERSONIC_BACKENDS:-cuda}"

CONTEXTS="${CONTEXTS:-4096,8192,16384,32768}"
ITERS="${ITERS:-20}"
WARMUP="${WARMUP:-5}"
OUTPUT="${OUTPUT:-$REPO_ROOT/target/llama31_certified_kv_microbench.json}"

echo "=== SuperSonic Llama 3.1 Certified KV Microbench ==="
echo "Contexts: $CONTEXTS"
echo "Iters:    $ITERS"
echo "Warmup:   $WARMUP"
echo "Output:   $OUTPUT"
echo ""

cargo build --release --manifest-path "$REPO_ROOT/Cargo.toml" --bin llama31_certified_kv_microbench

mkdir -p "$(dirname "$OUTPUT")"
"$REPO_ROOT/target/release/llama31_certified_kv_microbench" \
    --contexts "$CONTEXTS" \
    --iters "$ITERS" \
    --warmup "$WARMUP" \
    > "$OUTPUT"

cat "$OUTPUT"
