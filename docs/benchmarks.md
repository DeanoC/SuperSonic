# Benchmarks

Benchmark runs measure the supported Qwen3.8-27B GQH path; they do not expand
the support matrix. Report maximum measured performance only with enough
evidence for another contributor to reproduce the same workload and correctness
gate.

## Record before measuring

Save the exact commit, ROCm version, target architecture, physical device
selection, model-directory path, GGUF filename, prompt, context size, and
generated-token count. Keep the structured stage-timing output and the
ordinary-versus-MTP token comparison with the run.

For `gfx1201`, use the validated physical-device selection in the
[README](../README.md). The selection exports `SUPERSONIC_R9700_GPU_ID` and
`HIP_VISIBLE_DEVICES`; the process still uses logical `--device 0`. Never
replace discovery with an assumed physical ordinal.

## Reproducible command

Build for the target, then run warmups followed by repeated measurements:

```bash
HIP_ARCH=gfx1201 cargo build --release --workspace

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

Run the correctness gate in [testing](testing.md) first. For each measured
run, record warmup count, measured-run count, prefill time, median decode
`ms_per_tok`, derived `tok/s`, and the generated-token equality result. A
telemetry failure does not replace or weaken the correctness result.

The committed `gfx1201` workflow writes the same evidence as
`target/ci/qwen38-gfx1201/reproducibility.json`. It records the commit,
ROCm/HIP versions, physical-to-logical GPU mapping, artifact basename and
SHA-256 digest, prompt and token count, correctness hash, ordinary-versus-MTP
equality, and each warmup/measured prefill and decode timing. Absolute
artifact and model-directory paths are intentionally omitted from this
portable record.

## Comparing runs

Compare runs only when the artifact, prompt, generation length, build target,
and device selection match. Keep `gfx1100` and `gfx1201` results in separate
series. If the workload or artifact changes, start a new dated series rather
than combining numbers.
