# Performance

SuperSonic is tuned for maximum measured inference performance on the
Qwen3.8-27B custom GQH GGUF path. Its ROCm/HIP implementation combines fused
decode, prefill, dequantization, and device-specific scheduling; the public
identity is the measured result and its reproducible evidence, not one kernel
shape.

## Measurement contract

Publish a result only after the [correctness gate](testing.md) passes. Every
reported number must identify:

- the repository commit and ROCm/HIP toolchain;
- the target architecture and validated physical device selection;
- the exact model-directory and GQH GGUF artifact (including an artifact
  digest when one is available);
- prompt text or a stable workload description, context size, and generated
  token count;
- warmup count, measured-run count, prefill timing, median decode
  `ms_per_tok`, and derived `tok/s`;
- the ordinary-versus-NextN/MTP generated-token equality result.

A headline claim is not a benchmark without the exact commit, artifact,
workload, measurement method, and correctness result. If that evidence is not
available, omit the number and report the missing evidence instead.

## Reproduce

Use the command and run-record fields in [benchmarks](benchmarks.md). Keep
`gfx1100` and `gfx1201` as separate series, and do not compare different GQH
artifacts as if they were one model build. CI throughput telemetry is
nonblocking until repeated runs establish variance and a documented threshold.
