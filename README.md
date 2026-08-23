# SuperSonic

SuperSonic is the product path for Qwen3.8-27B inference with custom GQH
artifacts on HIP. The active support contract is intentionally small: the
`gfx1100` and `gfx1201` lanes share the same model and artifact roles.

## Quick start

```bash
HIP_ARCH=gfx1201 cargo build --release --workspace

HIP_VISIBLE_DEVICES=0 \
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
