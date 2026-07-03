# Qwen3.6 Dense FLM Main Path Design

## Goal

Make a Qwen3.6 27B dense FLM file a complete, first-class SuperSonic model source.
The normal user path should be:

```bash
cargo run -q -p runner --bin supersonic -- \
  --model qwen3.6-27b \
  --model-dir /path/to/qwen36-27b.flm \
  --int4 \
  --prompt "..."
```

For this stage, SuperSonic must not require a Hugging Face snapshot, `config.json`,
`tokenizer.json`, safetensors shards, or a `.supersonic` bake directory when the FLM
runtime directory contains the required native descriptors and assets.

## Current Baseline

The merged FLM support already provides these pieces:

- `--model-dir model.flm` and `--flm-file model.flm` are recognized as FLM sources.
- Qwen3.6 dense config is loaded from the FLM runtime directory.
- Qwen BPE tokenizer assets are loaded from native FLM runtime assets.
- Qwen3.6 dense FLM weights are exposed through `BakedStore::open_flm_with_options`.
- Direct upload views exist for INT4, NVFP4, MXFP4, MXFP8, and FP8 Stage 3 storage
  bindings.
- The policy layer rejects unsupported FLM combinations such as non-Qwen3.6 dense,
  q4km, int8, DFlash, and specprefill.

This design turns that baseline into a complete main path by tightening lifecycle,
validation, error behavior, smoke coverage, and compatibility boundaries.

## Scope

In scope:

- Qwen3.6 27B dense FLM as the target model.
- FLM as the authoritative source for model config, tokenizer, and weights.
- INT4 runtime path first, using existing Qwen35/Qwen3.6 dense runtime plumbing.
- Optional payload hash verification through the existing `--verify-flm-hashes`.
- Clear failures for incomplete or unsupported FLM files.
- A real end-to-end smoke path that exercises config, tokenizer, weight loading, and
  one prompt run without HF files.

Out of scope:

- Qwen3.6 35B-A3B MoE main-path execution.
- DFlash and specprefill with FLM sources.
- q4km, q4km-gptq, int8, or alternate quant paths.
- GPUDirect, O_DIRECT, or plan-set performance work.
- Removing compatibility aliases from the FLM loader.
- Rewriting SuperSonic's dense runtime around FLM-specific descriptors.

## Architecture

The main path should keep a single authoritative abstraction:

```text
Cli
  -> EffectiveModelSource
       -> HfDirectory(model_dir)
       -> FlmFile(path)
  -> StartupBundle
       -> Config
       -> Tokenizer
       -> WeightStore
  -> Qwen dense runtime
```

Today FLM checks are split across `bakes.rs`, `qwen35_startup.rs`, policy checks, and
weight loading. This stage should make the boundary explicit without over-refactoring:

- `effective_flm_source(cli)` remains the compatibility detector.
- A new or tightened `EffectiveModelSource` helper should describe whether the model
  source is HF-directory-backed or FLM-backed.
- Startup code should open the FLM once where practical and pass the parsed runtime
  information to config/tokenizer/weight loading. Re-opening is acceptable in tests or
  narrow helpers, but the production path should avoid independently parsing the same
  FLM three times.
- `BakedStore` remains the weight store interface consumed by existing Qwen dense
  weight code.

The important behavior is that the dense runtime sees normal `Config`, `Tokenizer`,
and `Qwen35Weights` values. FLM is a source format, not a second dense model runtime.

## Data Flow

1. CLI/policy resolves the model variant and model source.
2. If the source is FLM:
   - Reject unsupported model variants and quant profiles before any download or bake
     behavior.
   - Open the FLM with hash verification matching `--verify-flm-hashes`.
   - Require a runtime directory with `QWEN3_6_DENSE_V1`.
   - Build the normalized Qwen dense config from the runtime descriptor.
   - Build the tokenizer from native FLM assets.
   - Load weights through `BakedStore::open_flm_with_options`.
3. If the source is HF-directory-backed, keep the existing config/tokenizer/bake flow.
4. The dense runtime receives the same `Config`, prompt ids, and `Qwen35Weights` shape
   it already expects.

The FLM path must not call metadata bootstrap, bake download, bake lock acquisition, or
local bake generation. If those appear in logs for an FLM model source, that is a bug.

## Validation And Error Handling

FLM startup should fail early with messages that identify the missing contract piece:

- Missing runtime directory: "FLM has no runtime directory".
- Wrong architecture: "FLM is not Qwen3.6 dense v1".
- Missing tokenizer descriptor or asset: identify the asset kind and id.
- Unsupported tokenizer algorithm: include the algorithm id and expected id.
- Missing required tensor or unsupported Stage 3 binding: surface the `model-store`
  error with the tensor/logical name.
- Incompatible CLI flags: reject before opening weights and mention the flag pair.
- Payload hash mismatch with `--verify-flm-hashes`: fail before GPU upload.

Default behavior should keep hash verification opt-in for local developer iteration,
while the error path and smoke tests must prove that enabling it works. The public
format recommendation remains that distributable runnable FLMs should carry payload
hashes.

## Testing

Unit coverage:

- CLI/model-source selection treats `.flm` as authoritative.
- FLM source rejects unsupported variants and incompatible quant flags.
- FLM config loading rejects missing/wrong runtime descriptors.
- FLM tokenizer loading rejects missing assets, wrong asset kinds, duplicate ids, and
  unsupported tokenizer algorithm ids.
- FLM weight loading rejects incomplete required tensor manifests.
- `--verify-flm-hashes` rejects a corrupted payload fixture.

Integration coverage:

- An env-gated Qwen3.6 27B dense FLM smoke test opens a real no-HF FLM artifact,
  loads config/tokenizer/weights, uploads representative tensors to HIP, and runs one
  short prompt path.
- The smoke test should assert logs or counters proving no HF metadata bootstrap,
  bake download, bake lock, or bake directory access happened.
- Existing `cargo test -q -p model-store` and runner unit tests remain clean.

Producer/validator coverage in geo-quant:

- Keep `runnable-no-hf` and `strict-mainline --verify-payload-hashes` validation as the
  gate for any artifact used by SuperSonic smoke tests.
- If the SuperSonic smoke uncovers a missing runtime descriptor, fix the producer and
  validator first, then update SuperSonic.

## Migration Boundary

The existing `--flm-file` flag remains as a compatibility override, but the preferred
main path is `--model-dir model.flm`. Help text and tests should continue to say that.

The old HF-directory path must remain unchanged. A user with a normal HF snapshot and
existing bake should see no behavior change.

Unsupported FLM combinations should stay explicit failures rather than silent fallback
to HF files. This is how we preserve the FLM guarantee that a runnable FLM does not need
an external snapshot for normal use.

## Success Criteria

- `cargo test -q -p model-store` passes.
- Relevant runner unit tests pass.
- With a no-HF Qwen3.6 27B dense FLM artifact:
  - `--model-dir artifact.flm --model qwen3.6-27b --int4 --verify-flm-hashes` opens
    successfully.
  - The native FLM tokenizer produces prompt ids.
  - Representative FLM tensors upload to HIP using direct views.
  - A short prompt smoke path reaches model execution without reading HF files or a
    `.supersonic` bake.
- The implementation has clear, documented rejection behavior for MoE, DFlash,
  specprefill, q4km, and int8 FLM sources.

## Follow-Up Stages

After this path is complete:

1. Add Qwen3.6 35B-A3B MoE FLM as a normal model source.
2. Add runtime support for FLM-backed DFlash/specprefill where it makes sense.
3. Reduce compatibility aliases in favor of direct runtime storage bindings.
4. Add plan-set and load-performance work such as larger sequential reads, pinned
   staging buffers, and GPUDirect-shaped transfer plans.
