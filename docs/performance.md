# Performance

The active performance surface is the Qwen3.8-27B GQH artifact on HIP
`gfx1100` and `gfx1201`. This page records the measurement contract; headline
numbers are published only with reproducible command lines and structured
logs.

## Required fields

Record:

- architecture and physical device selection;
- artifact names and hashes;
- prompt and generated-token counts;
- prefill and decode timings from `--emit-stage-timings`;
- ordinary-versus-MTP token equality;
- warmup count, measured-run count, median `ms_per_tok`, and derived `tok/s`.

The serial correctness gate runs before telemetry. A telemetry failure is
diagnostic and does not replace the correctness result. Use the command in
[benchmarks](benchmarks.md) and keep the resulting logs with the run.
