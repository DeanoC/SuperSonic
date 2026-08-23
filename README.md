# SuperSonic

SuperSonic is the product path for Qwen3.8-27B inference with custom GQH
artifacts on HIP. The active support contract is intentionally small: the
`gfx1100` and `gfx1201` lanes share the same model and artifact roles.

## Quick start

On a R9700 host, discover the physical GPU ordinal first. The selector reads
the AMD SMI static ASIC record and accepts an override only after validating
that record; it intentionally does not assume physical GPU zero:

```bash
amd-smi static --asic --json > /tmp/supersonic-amd-smi-static.json
while IFS='=' read -r name value; do
  case "$name" in
    SUPERSONIC_R9700_GPU_ID|SUPERSONIC_R9700_GPU_ARCH|HIP_VISIBLE_DEVICES|SUPERSONIC_DEVICE)
      export "$name=$value" ;;
  esac
done < <(
  python3 tools/select-r9700-device.py \
    --input /tmp/supersonic-amd-smi-static.json \
    --override "${SUPERSONIC_R9700_GPU_ID:-}"
)
```

```bash
HIP_ARCH=gfx1201 cargo build --release --workspace

HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
  cargo run --release --bin supersonic -- \
  --model qwen3.8-27b \
  --model-dir /data/models/Qwen3.8-27B \
  --gguf-file /home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq.gguf \
  --prompt "Hello, world" \
  --max-new-tokens 8 \
  --device 0
```

The GGUF and model directory are separate roles. Keep the artifact pair on a
local filesystem and follow the [artifact contract](docs/artifact-format.md)
before running a correctness gate.

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
