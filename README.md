# SuperSonic

SuperSonic is a performance-specialized ROCm/HIP inference engine for
Qwen3.8-27B. Its narrow product surface is built around maximum measured performance,
reproducible correctness, and a custom GQH GGUF artifact on
supported AMD GPUs.

The supported contract is one Qwen3.8-27B sequence at a time with deterministic
greedy generation, optional Qwen3.8 NextN/MTP generation, and the
`gfx1100` and `gfx1201` ROCm targets. The model directory and the weight
artifact are separate inputs: `--model-dir` supplies `config.json`, tokenizer
data, and, only when `--chat` is used, the chat template from
`tokenizer_config.json`; `--gguf-file` supplies the matching custom GQH GGUF
weights.

## Quick start

On a R9700 host, discover the physical GPU ordinal first. The selector reads
the AMD SMI static ASIC record and accepts an override only after validating
that record; it intentionally does not assume physical GPU zero:

```bash
set -euo pipefail
static_report="${TMPDIR:-/tmp}/supersonic-amd-smi-static.json"
requested_gpu="${SUPERSONIC_R9700_GPU_ID:-}"
unset SUPERSONIC_R9700_GPU_ID SUPERSONIC_R9700_GPU_ARCH HIP_VISIBLE_DEVICES SUPERSONIC_DEVICE
timeout --foreground 30s amd-smi static --asic --json > "$static_report"
selection="$(
  python3 tools/select-r9700-device.py \
    --input "$static_report" \
    --override "$requested_gpu"
)"

declare -A selected=()
while IFS='=' read -r name value; do
  [[ -n "$name" && -n "$value" ]] || {
    echo "invalid R9700 selector output: empty name or value" >&2
    exit 1
  }
  case "$name" in
    SUPERSONIC_R9700_GPU_ID|SUPERSONIC_R9700_GPU_ARCH|SUPERSONIC_GPU_IDENTITY|SUPERSONIC_GPU_IDENTITY_KIND|SUPERSONIC_GPU_LOGICAL|HIP_VISIBLE_DEVICES|SUPERSONIC_DEVICE) ;;
    *)
      echo "invalid R9700 selector output: unexpected key $name" >&2
      exit 1
      ;;
  esac
  [[ -z "${selected[$name]+present}" ]] || {
    echo "invalid R9700 selector output: duplicate key $name" >&2
    exit 1
  }
  selected["$name"]="$value"
done <<< "$selection"

[[ "${#selected[@]}" -eq 7 ]]
[[ "${selected[SUPERSONIC_R9700_GPU_ID]}" =~ ^[0-9]+$ ]]
[[ "${selected[SUPERSONIC_R9700_GPU_ARCH]}" == "gfx1201" ]]
[[ "${selected[HIP_VISIBLE_DEVICES]}" == "${selected[SUPERSONIC_R9700_GPU_ID]}" ]]
[[ "${selected[SUPERSONIC_DEVICE]}" == "0" ]]
[[ -n "${selected[SUPERSONIC_GPU_IDENTITY]}" ]]
[[ "${selected[SUPERSONIC_GPU_IDENTITY_KIND]}" == "pci_bdf" || "${selected[SUPERSONIC_GPU_IDENTITY_KIND]}" == "uuid" ]]
[[ "${selected[SUPERSONIC_GPU_LOGICAL]}" == "${selected[SUPERSONIC_DEVICE]}" ]]
export SUPERSONIC_R9700_GPU_ID="${selected[SUPERSONIC_R9700_GPU_ID]}"
export SUPERSONIC_R9700_GPU_ARCH="${selected[SUPERSONIC_R9700_GPU_ARCH]}"
export HIP_VISIBLE_DEVICES="${selected[HIP_VISIBLE_DEVICES]}"
export SUPERSONIC_DEVICE="${selected[SUPERSONIC_DEVICE]}"
export SUPERSONIC_GPU_IDENTITY="${selected[SUPERSONIC_GPU_IDENTITY]}"
export SUPERSONIC_GPU_IDENTITY_KIND="${selected[SUPERSONIC_GPU_IDENTITY_KIND]}"
export SUPERSONIC_GPU_LOGICAL="${selected[SUPERSONIC_GPU_LOGICAL]}"
```

Build for the selected ROCm target, then run the one direct GQH path:

```bash
HIP_ARCH=gfx1201 cargo build --release --workspace
```

```bash
HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
  cargo run --release --bin supersonic -- \
  --model qwen3.8-27b \
  --model-dir /data/models/Qwen3.8-27B \
  --gguf-file /home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq.gguf \
  --prompt "Hello, world" \
  --max-new-tokens 8 \
  --device 0
```

The model directory must contain `config.json` and `tokenizer.json`. Only a
`--chat` run additionally requires the chat template in
`tokenizer_config.json`. The GGUF must be the matching project-specific GQH
artifact; generic GGUF files are not part of this contract. See the [artifact
contract](docs/artifact-format.md) before running a correctness gate.

Any unlisted model, architecture, artifact format, or non-greedy generation
combination fails explicitly. A configured GPU or artifact that cannot be
validated fails closed before the correctness run starts.

## Active documentation

- [Build and run](docs/build-and-run.md)
- [Supported matrix](docs/supported-matrix.md)
- [Artifact format](docs/artifact-format.md)
- [Testing gates](docs/testing.md)
- [Benchmarks](docs/benchmarks.md)
- [Performance](docs/performance.md)

Validate the checked product boundaries with:

```bash
python3 tools/check-support-matrix.py
python3 tools/check-active-docs.py
```

Performance numbers are published only when the exact commit, GPU target,
artifact, workload, measurement method, and correctness result are recorded.
Until that evidence is attached, this page intentionally makes no standalone
throughput claim.
