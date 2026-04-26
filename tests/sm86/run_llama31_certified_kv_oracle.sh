#!/usr/bin/env bash
#
# Synthetic correctness smoke for the Llama3.1 certified-KV PyTorch oracle.
#
# This does not require model weights. It validates the reference math and emits
# one deterministic telemetry record for the Rust/CUDA implementation to mirror.
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

echo "=== SuperSonic sm86 Llama3.1 certified-KV oracle smoke ==="
echo "Device: $DEVICE"
echo ""

python3 -m unittest "$REPO_ROOT/tests/test_certified_kv_llama31_oracle.py"
echo ""

python3 "$REPO_ROOT/oracle/certified_kv_llama31.py" \
    --self-test \
    --device "$DEVICE" \
    --tokens "${CERT_KV_TOKENS:-33}" \
    --kv-heads "${CERT_KV_HEADS:-2}" \
    --gqa-group "${CERT_KV_GQA_GROUP:-4}" \
    --head-dim "${CERT_KV_HEAD_DIM:-128}" \
    --block-size 16 \
    --value-group-size 16 \
    --tau-cov 0.995 \
    --k-min 2 \
    --k-max 128 \
    --v-tol 0.05
