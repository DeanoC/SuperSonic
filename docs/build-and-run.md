# Build and run

This page documents the active HIP path for the Qwen3.8-27B GQH artifact
lane. The [support matrix](supported-matrix.md) is the authority for the two
architecture rows.

## Build

Use a ROCm development environment with a pinned toolchain and select the
compile target explicitly. Compilation does not require a visible GPU.

```bash
HIP_ARCH=gfx1100 cargo build --release --workspace
HIP_ARCH=gfx1201 cargo build --release --workspace
```

## Run

Provide both the model directory and the matching custom GGUF artifact:

```bash
HIP_VISIBLE_DEVICES=0 \
  ./target/release/supersonic \
  --model qwen3.8-27b \
  --model-dir /data/models/Qwen3.8-27B \
  --gguf-file /home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq.gguf \
  --prompt "Hello" \
  --max-new-tokens 8 \
  --device 0
```

The R9700 runner discovers its physical device from AMD SMI output and then
masks it to logical device zero. A local run should make the same mapping
explicit; do not assume that physical ordinal zero is the target.

For the full pairing and header checks, see
[artifact format](artifact-format.md). For serial large-artifact gates, see
[testing](testing.md).

## MTP equivalence

The correctness gate runs ordinary and MTP generation with the same prompt
and compares the emitted token sequence. Reproduce that comparison only
after the ordinary GQH crawl passes.
