# Build and run

This page documents the supported Qwen3.8-27B path: a custom GQH GGUF
artifact loaded by the ROCm/HIP runtime on `gfx1100` or `gfx1201`. The
[support matrix](supported-matrix.md) is the authority for those two targets.

## Prerequisites

Install a Rust toolchain and a ROCm installation that provides HIP and the
matching device runtime. Keep `ROCM_PATH` and `HIP_PATH` pointed at that
installation when the tools are not on the default path. `HIP_ARCH` selects
the compile target; the supported values are `gfx1100` and `gfx1201`.

The public input pair is:

- `--model-dir <directory>` containing `config.json` and `tokenizer.json`;
  `tokenizer_config.json` with the Qwen3.8 chat template is needed only when
  `--chat` is used;
- `--gguf-file <file>` containing the matching project-specific GQH GGUF
  weights.

The startup validator reads the required files before GPU allocation, including
chat-template metadata when `--chat` is set. Missing files, invalid Qwen3.8
geometry, a missing GQH header, or an incompatible tensor layout is an error
with the path and required role in the message.

## Build

Compilation is CPU-safe, but the target architecture must still be explicit:

```bash
HIP_ARCH=gfx1100 cargo build --release --workspace
HIP_ARCH=gfx1201 cargo build --release --workspace
```

## Select a device

For a self-hosted `gfx1201` run, use the validated physical-device discovery
snippet in the [README](../README.md). It reads `amd-smi static --asic
--json`, requires one R9700 record, and exports the selected physical ordinal
through `HIP_VISIBLE_DEVICES` while the program uses logical `--device 0`.
`SUPERSONIC_R9700_GPU_ID` is an override only after the same record validation.

Do not assume physical ordinal zero. If discovery, the override, or the
selected architecture is ambiguous, the setup stops without exporting a
selection. The self-hosted workflow performs bounded idle probes before its
artifact gate; this local page does not duplicate that polling loop.

## Run

After device selection, invoke the direct GQH command. `--model` is explicit
and accepts only `qwen3.8-27b`:

```bash
HIP_VISIBLE_DEVICES="$HIP_VISIBLE_DEVICES" \
  cargo run --release --bin supersonic -- \
  --model qwen3.8-27b \
  --model-dir /data/models/Qwen3.8-27B \
  --gguf-file /home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq.gguf \
  --prompt "Hello" \
  --max-new-tokens 8 \
  --device 0
```

Use `--chat` when the prompt should be rendered with the template in
`tokenizer_config.json`. The generation controls remain deterministic greedy
defaults: `--temperature 0`, `--top-k 0`, and `--top-p 1`. An unsupported
model value, artifact role, architecture, or generation mode fails directly.

For reproducible logs, add `--emit-generated-json` and
`--emit-stage-timings`. `--prefill-chunk-size` and `--context-size` are
supported device/context controls; keep one sequence per process.

## Optional NextN/MTP generation

If the GQH file contains the complete Qwen3.8 NextN block, pass
`--speculative-decode`. The runtime validates the block before launch and
keeps the generated token sequence equivalent to ordinary greedy generation.
The [testing](testing.md) page describes the serial equivalence gate.
