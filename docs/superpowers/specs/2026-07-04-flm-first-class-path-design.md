# FLM First-Class Path Design

## Goal

Make an FLM file exported by geo-quant a first-class SuperSonic model source. A normal run should pass the FLM path as `--model-dir`, require no Hugging Face snapshot, require no quantization flag such as `--int4`, and load the executable weight mode directly from the file metadata.

## Current State

The current merged path can load the latest native INT4 Qwen3.6 MoE FLM only when the user also passes `--int4`. Without that flag SuperSonic opens the FLM, reads the runtime config and tokenizer, then rejects the file because the default CLI quantization profile is BF16. That makes the CLI flag, not the FLM, the source of truth.

The latest geo-quant native INT4 artifact proves the direct path exists: with `--int4`, SuperSonic reports `INT4 native FLM` and reaches the dry-run ready-for-decode path. The first-class gap is selection and ergonomics, not the underlying native tensor layout.

The e2e implementation also exposed a stricter direct-layout requirement: Qwen3.6 linear-attention raw tensors must be stored in the SuperSonic runtime layout inside the FLM. In particular, `linear_attn.conv1d.weight` is the squeezed depthwise-conv view, `linear_attn.dt_bias` is `[1, 1, H]`, and `linear_attn.A_log` is exponentiated BF16 `[1, 1, H]`. A runnable FLM should contain those direct values; SuperSonic labels and validates them, but does not reshape HF checkpoint payloads on the normal path.

## Architecture

SuperSonic should treat `.flm` as authoritative for FLM-backed runs:

- `--model-dir /path/model.flm` or `--flm-file /path/model.flm` selects an FLM model source.
- FLM open options expose logical/direct INT4 aliases for FLM sources without needing a CLI quantization profile.
- Qwen3.6 MoE weight-mode selection probes the already-open FLM store and chooses native INT4 when the logical probe is `LayoutTag::Int4Quantized` with `u8`, or BF16 fallback when the probe is `LayoutTag::Raw` with `bf16`.
- Raw direct-plan aliases are labeled with their runtime layout tags when the FLM plan target shape already matches the runtime view, for example `DepthwiseConvSqueezed`, `HeadBiasReshaped`, and `HeadExpReshaped` for Qwen3.6 linear attention.
- Explicit incompatible quantization flags remain validation errors. The no-flag path is file-driven.

This keeps the direct path first-class while preserving fallbacks for FLM artifacts that carry CT-style or BF16 views.

## Data Flow

1. CLI resolves an effective FLM source from `--flm-file` or `.flm` `--model-dir`.
2. SuperSonic opens the FLM once with runtime aliases enabled and optional BLAKE3 payload verification based on `--verify-flm-hashes`.
3. Runtime config and tokenizer are loaded from FLM assets.
4. Qwen3.6 MoE loader probes the logical weight view in the open store and selects the runtime weight mode from the file.
5. Decode uses the selected mode and the already-open FLM store. No HF `config.json`, `tokenizer.json`, safetensors, fetch, or bake path is consulted.

## Error Handling

If a Qwen3.6 MoE FLM has no recognized executable probe, the loader should fail with a message naming the missing/incompatible probe. If the user asks for an incompatible non-INT4 quant mode with an FLM source, the CLI should continue to reject it before load. Payload hash verification remains opt-in because it reads the payload bytes and is not representative of fast-load latency.

## Testing

The primary red test is the current failing behavior: the latest geo-quant native INT4 FLM must dry-run under SuperSonic without `--int4`.

Automated coverage should include:

- unit tests proving FLM open options enable runtime aliases independent of CLI quant flags;
- unit tests proving Qwen3.6 MoE FLM weight selection is file-probe driven;
- model-store tests proving Qwen3.6 linear-attention direct raw values match the SuperSonic runtime bake byte-for-byte;
- env-gated integration tests that run the real 35B A3B FLM with no HF snapshot and no `--int4`;
- docs with canonical geo-quant export and SuperSonic run commands.

Manual/e2e verification should use the current 35B native artifact and capture `--emit-stage-timings` output so load and inference speeds are measurable.
