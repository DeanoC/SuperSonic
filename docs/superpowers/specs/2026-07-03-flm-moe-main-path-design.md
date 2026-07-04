# Qwen3.6 MoE FLM Main Path Design

## Goal

Make a Qwen3.6 35B-A3B MoE FLM file a complete SuperSonic model source for
the normal HIP decode path.

The target user path is:

```bash
cargo run -q -p runner --bin supersonic -- \
  --backend hip \
  --model qwen3.6-35b-a3b \
  --model-dir /path/to/qwen36-35b-a3b.flm \
  --int4 \
  --verify-flm-hashes \
  --prompt "Hello" \
  --max-new-tokens 1
```

For this stage, SuperSonic must not require a Hugging Face snapshot,
`config.json`, `tokenizer.json`, safetensors shards, or a `.supersonic` bake
directory when the FLM contains the required runtime directory, tokenizer
assets, and MoE tensor payloads.

## Motivation

The dense Qwen3.6 27B path now proves the right lifecycle shape: one opened FLM
source supplies config, tokenizer, and weights. The next useful FLM milestone is
not broader format surface area. It is one larger production-shaped model that
loads and executes directly from FLM.

Qwen3.6 35B-A3B is the right slice because it exercises the things FLM must be
good at:

- a binary MoE runtime descriptor rather than JSON config;
- native tokenizer assets rather than `tokenizer.json`;
- many expert tensors and sidecars;
- ROCm/HIP GPU upload from the FLM-backed `BakedStore`;
- a real decode path that currently expects a bake directory.

## Current Baseline

SuperSonic already has these pieces:

- `model-store` parses `ARCH_QWEN3_6_MOE` runtime directories and exposes
  `FlmQwen36MoeConfig`.
- `model-store` tests cover small and real-shape Qwen3.6 35B low-bit FLM
  loadability and HIP upload views.
- `runner` has `FlmModelSource` for a single opened FLM source.
- Dense startup loads config, tokenizer, and weights from one `FlmModelSource`.
- The MoE decode path already consumes a `model_store::BakedStore` after opening
  a selected bake directory.

The missing part is the runner boundary for MoE. `qwen36_moe::engine` still
expects:

- `config.json` through `run_qwen36_moe_dry_run`;
- `tokenizer.json` through `prepare_prompt`;
- a selected `.supersonic` bake through `select_decode_bake`;
- optional bake download through `ensure_qwen36_bake`.

## Scope

In scope:

- `qwen3.6-35b-a3b` only.
- HIP/ROCm decode first.
- INT4-compatible MoE weight mode first, using the existing MoE direct-loading
  path.
- `--model-dir model.flm` and compatibility `--flm-file model.flm`.
- Config, tokenizer, and weights loaded from one opened FLM source.
- Optional payload verification through `--verify-flm-hashes`.
- Env-gated one-token end-to-end smoke coverage on a real local FLM artifact.

Out of scope:

- DFlash and specprefill from FLM sources.
- q4km, q4km-gptq, int8, and BF16 FLM execution.
- New GPU kernels or new MoE tensor layouts.
- GPUDirect, O_DIRECT, pinned staging, or plan-set load performance work.
- Removing dense compatibility aliases.
- Requiring the inference engine to read Hugging Face JSON files for normal FLM
  use.

## Format Contract

The producer-side FLM must be accepted by the existing no-HF validation profile
and must carry the MoE runtime ABI:

- `ARCH_QWEN3_6_MOE`
- `MODEL_QWEN3_6_MOE_V1`
- `TENSOR_ABI_QWEN3_6_MOE_MIXED_LOWBIT_V1`
- `QUANT_PROFILE_QWEN3_6_MOE_MIXED_LOWBIT_V1`
- native tokenizer descriptor and assets
- Stage 3 tensor tables and payload block hashes

SuperSonic should not depend on a Hugging Face snapshot to fill in missing
normal-use fields. If a field is required to build `qwen36_moe::config::Config`,
tokenize a prompt, or load decode weights, it belongs in the FLM runtime
directory or tensor table.

JSON compatibility assets may remain as optional transition extensions, but
they are not part of the main-path contract. The smoke test should prove no
`config.json`, `tokenizer.json`, safetensors shard, or bake directory is read.

## Runner Architecture

Keep `FlmModelSource` as the runner-facing source object. Do not add a second
FLM storage abstraction for MoE.

The MoE path should resolve one optional FLM source before dry-run accounting:

```text
Cli
  -> effective_flm_source(cli)
  -> Option<Qwen36MoeFlmSource>
       -> FlmModelSource
       -> qwen36_moe::config::Config
       -> tokenizers::Tokenizer
       -> Qwen36WeightMode
  -> DryRunReport
  -> decode_text
       -> tokenizer from FLM source
       -> BakedStore from FLM source
       -> existing layer/session/decode path
```

`Qwen36MoeFlmSource` can be a small runner helper that owns the opened
`FlmModelSource` and the already converted MoE startup values. It should not
change the storage interface consumed by lower-level MoE loaders.

## Config Conversion

The conversion from `model_store::FlmQwen36MoeConfig` to
`qwen36_moe::config::Config` should live in the `qwen36_moe` crate, matching the
dense pattern in `qwen35`.

The conversion must build:

- `TextConfig::vocab_size`
- hidden, layer, attention, KV, and head dimensions
- linear-attention projection dimensions
- MoE expert counts and intermediate sizes
- `layer_types` from `full_attention_layers`
- `rope_parameters` from `rope_theta`, `partial_rotary_factor`,
  `mrope_interleaved`, and `mrope_section`
- `eos_token_id` from `eos_token_ids`
- `hidden_act` from `activation_id`
- `tie_word_embeddings` and `attn_output_gate`

It must reject invalid descriptors before decode:

- zero sizes for required dimensions and expert counts;
- `num_experts_per_tok > num_experts`;
- full-attention layer indices outside `num_hidden_layers`;
- unknown `activation_id`;
- invalid rotary dimension;
- invalid attention head divisibility.

The converted config should then pass the MoE crate's existing validation.

## Weight Mode

The FLM describes the model artifact and tensor storage. The runner maps a
supported FLM MoE runtime descriptor to the existing execution-side
`Qwen36WeightMode` at the boundary.

For this stage:

- accept the MoE mixed-lowbit runtime profile only for `--int4` or the backend
  path that already requires an INT4-compatible load;
- map that accepted profile to `Qwen36WeightMode::Int4`;
- fail clearly for profiles the current MoE decode path cannot execute;
- do not model `DEQUANT_IN_MMA` as a separate FLM fallback strategy.

Whether dequantization is done by dedicated hardware, software inside the MMA
path, or a pre-load transform is an execution-engine choice. The FLM contract
only needs to say which tensors and storage bindings are present.

## Runtime Behavior

When the effective model source is FLM and `--model qwen3.6-35b-a3b`:

1. Validate incompatible global flags before downloads or bake checks.
2. Open the FLM once with `FlmModelSourceOptions` from the CLI.
3. Require the MoE runtime directory and convert it to `qwen36_moe::Config`.
4. Load the tokenizer from FLM native assets.
5. Skip `ensure_qwen36_bake`.
6. Build the dry-run report from the FLM config without requiring bake-directory
   accounting.
7. Tokenize the prompt using the FLM tokenizer.
8. Load MoE layers from the already-open FLM `BakedStore`.
9. Run the existing MoE decode path.

When the effective model source is not FLM, existing HF-directory and bake
behavior stays unchanged.

## Logging Contract

The FLM MoE path should emit a single source-open line:

```text
[flm] opening model source at /path/qwen36-35b-a3b.flm (FLM logical INT4 aliases enabled) (BLAKE3 hash verification enabled)
```

The decode path should then make the reuse explicit:

```text
[qwen36-moe] loading config from FLM runtime descriptor
[qwen36-moe] loading tokenizer from FLM assets
[qwen36-moe] loading weights from already-open FLM source at /path/qwen36-35b-a3b.flm (INT4 GPTQ)
```

The smoke test should assert one FLM source-open line and no `[fetch]`, `[bake]`,
`config.json`, `tokenizer.json`, safetensors, or `.supersonic` references.

## Testing

Unit coverage:

- FLM model-source policy accepts `qwen3.6-27b` and `qwen3.6-35b-a3b`.
- FLM model-source policy still rejects unrelated model variants.
- MoE FLM config conversion accepts the real 35B-A3B geometry.
- MoE FLM config conversion rejects bad expert counts, bad full-attention layer
  indices, and unknown activation ids.
- MoE FLM source opening rejects dense FLMs for the MoE model.
- MoE FLM source opening rejects unsupported quant profiles.
- MoE prompt setup uses an already-loaded tokenizer when one is provided.
- MoE bake selection is bypassed only when an FLM source exists.

Integration coverage:

- `SUPERSONIC_QWEN36_35B_A3B_NO_HF_FLM=/path/model.flm cargo test -q -p runner --test flm_moe_main_path -- --nocapture`
  runs one generated token on HIP.
- The smoke passes `--verify-flm-hashes`.
- The smoke asserts config, tokenizer, and weights came from FLM.
- The smoke asserts the FLM is opened exactly once.
- The smoke asserts no HF snapshot or bake path is used.

## Success Criteria

- `cargo test -q -p qwen36_moe flm` passes.
- `cargo test -q -p runner flm_source` passes.
- `cargo test -q -p runner qwen36_moe` passes.
- `cargo test -q -p model-store flm` passes.
- With a local Qwen3.6 35B-A3B no-HF FLM artifact, the env-gated runner smoke
  executes a one-token HIP decode with hash verification enabled.
- Existing HF-directory MoE decode behavior is unchanged.

## Follow-Up

After this stage lands:

1. Expand the same FLM-source path to DFlash and specprefill where the runtime
   contract supports them.
2. Add binary runtime assets for any remaining compatibility-only JSON metadata.
3. Add load-performance work around transfer plans, staging, and fewer page
   faults.
4. Reduce compatibility aliases once direct runtime storage bindings are stable.
