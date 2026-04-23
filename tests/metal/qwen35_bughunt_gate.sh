#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

default_model_dir="$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17"
model_dir="${QWEN35_MODEL_DIR:-$default_model_dir}"
manifest="${QWEN35_BUGHUNT_MANIFEST:-$repo_root/crates/runner/bughunt/qwen35_metal_manifest.json}"
report_json="${QWEN35_BUGHUNT_REPORT_JSON:-/tmp/qwen35_bughunt_gate.json}"
oracle_device="${QWEN35_ORACLE_DEVICE:-cpu}"
backend="${QWEN35_BUGHUNT_BACKEND:-metal}"

if [[ ! -d "$model_dir" ]]; then
  echo "Qwen3.5 0.8B model dir not found: $model_dir" >&2
  echo "Set QWEN35_MODEL_DIR=/path/to/Qwen3.5-0.8B and rerun." >&2
  exit 2
fi

cd "$repo_root"

cargo build -p runner --bin qwen35_bughunt

cmd=(
  "$repo_root/target/debug/qwen35_bughunt"
  --model-dir "$model_dir"
  --backend "$backend"
  --oracle-device "$oracle_device"
  --mode gate
  --prompt-manifest "$manifest"
  --report-json "$report_json"
)

if [[ -n "${QWEN35_BUGHUNT_PROMPT:-}" ]]; then
  cmd+=(--prompt "$QWEN35_BUGHUNT_PROMPT")
fi

if [[ -d "$repo_root/.venv/bin" ]]; then
  PATH="$repo_root/.venv/bin:$PATH" "${cmd[@]}"
else
  "${cmd[@]}"
fi
