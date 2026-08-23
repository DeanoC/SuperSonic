# Benchmarks

Benchmark runs are measurements, not support claims. Record the commit,
architecture, artifact paths, prompt, generated-token count, and the
structured stage timings for every run.

## Preparation

```bash
git status --short --branch
HIP_ARCH=gfx1100 cargo build --release --workspace
rocm-smi --showproductname --showuse --showmemuse --showtemp
```

Select the target device explicitly using the [README discovery
snippet](../README.md), or set `SUPERSONIC_R9700_GPU_ID` to a validated
physical ordinal before exporting `HIP_VISIBLE_DEVICES`. Leave enough free
memory for the artifact. The R9700 workflow records nonblocking throughput
telemetry after the correctness steps; telemetry failure must not hide a
correctness result.

## Reproducible command

```bash
HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
  ./target/release/supersonic \
  --model qwen3.8-27b \
  --model-dir /data/models/Qwen3.8-27B \
  --gguf-file /home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq.gguf \
  --prompt "Hello" \
  --max-new-tokens 128 \
  --emit-generated-json \
  --emit-stage-timings \
  --device 0
```

Compare ordinary and MTP token logs before comparing throughput. See
[testing](testing.md) for the correctness gate and
[performance](performance.md) for the reporting fields.
