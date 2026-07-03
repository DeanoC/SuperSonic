# FLM Single Source Lifecycle Design

## Goal

Make the Qwen3.6 27B dense FLM main path use one authoritative opened FLM
source for config, tokenizer, and weights.

The supported user path remains:

```bash
cargo run -q -p runner --bin supersonic -- \
  --model qwen3.6-27b \
  --model-dir /path/to/qwen36-27b.flm \
  --int4 \
  --verify-flm-hashes \
  --prompt "Hello"
```

With this stage, `--verify-flm-hashes` means the runnable FLM is verified once at
source open, and the verified mmap-backed store is then used for every FLM-backed
runtime asset and tensor load in the dense main path.

## Motivation

The merged dense FLM main path makes a no-HF Qwen3.6 27B FLM runnable, but the
lifecycle is still split:

- startup opens an FLM source for config and tokenizer;
- startup deliberately disables payload hash verification to avoid a duplicate
  full-payload BLAKE3 pass;
- weight loading opens the same FLM again with `--verify-flm-hashes` applied.

That is correct enough for integrity, but it is not the shape we want long term.
FLM is meant to be the model source, not a metadata source plus a separate weight
source. The runner should carry one opened FLM source through startup and engine
setup, so the code structure matches the format's contract.

## Scope

In scope:

- Qwen3.6 27B dense FLM main path.
- `--model-dir model.flm` and compatibility `--flm-file model.flm`.
- Sharing the opened `FlmModelSource` from startup through dense weight loading.
- Opening the FLM with final runtime options, including `--verify-flm-hashes`.
- Preserving existing HF-directory and bake behavior.
- Test/log coverage proving the normal binary path does not reopen the FLM for
  weights.

Out of scope:

- Qwen3.6 35B-A3B MoE FLM support.
- DFlash and specprefill FLM support.
- Changing the FLM binary format.
- Adding lazy or per-tensor hash verification to `model-store`.
- Removing `--flm-file`.
- Redesigning `Qwen35Weights` around FLM-specific runtime descriptors.

## Design

The runner should treat the FLM source as a startup-owned resource that is moved
forward into engine setup:

```text
Cli
  -> Qwen35Startup
       -> TextConfig
       -> Tokenizer
       -> Prompt ids
       -> Optional<FlmModelSource>
  -> Qwen35EngineSetup
       -> Qwen35Weights from existing store when FLM source exists
       -> existing HF/bake path otherwise
```

`FlmModelSource` already owns:

- the FLM file path;
- the `BakedStore`;
- runtime directory accessors;
- Qwen config and tokenizer helpers.

This stage should not add a second source abstraction unless the implementation
needs it. The main change is ownership: `load_qwen35_startup` opens the FLM with
the same options that weights require and returns the opened source inside
`Qwen35Startup`.

## Hash Semantics

`model-store::BakedStore::open_flm_with_options` verifies FLM BLAKE3 payload
hashes during open when `verify_block_hashes` is true. This is a full referenced
payload pass.

For this stage:

- `--verify-flm-hashes` is threaded into the single startup open.
- A hash mismatch fails before config/tokenizer use and before any GPU upload.
- Weight loading uses the already-open, already-verified `BakedStore`.
- The default remains opt-in for local developer iteration.

This preserves the current public safety behavior while removing the extra open
and aligning the runner with FLM's "single model source" intent.

## Runtime Behavior

When the effective model source is FLM:

1. Validate that the selected model is `qwen3.6-27b`.
2. Validate incompatible flags such as `--no-bake`, q4km, and int8.
3. Compute one `FlmModelSourceOptions` value from the CLI quant profile.
4. Open the FLM once during startup.
5. Load config and tokenizer from the opened source.
6. Move the opened source through `Qwen35Startup`.
7. Load weights from that source's `BakedStore`.

When the effective model source is not FLM, the existing HF metadata bootstrap,
bake selection, and weight loading behavior stay unchanged.

## Logging Contract

The FLM main path should emit one source-open log line, for example:

```text
[flm] opening model source at /path/model.flm (FLM logical INT4 aliases enabled) (BLAKE3 hash verification enabled)
```

Weight loading should make clear that it is consuming the existing source:

```text
[weights] loading FLM weights from already-open source at /path/model.flm
```

The env-gated smoke test should assert that the source-open log appears exactly
once and that the old "loading FLM container" reopen log is absent from the
main-path binary run.

## Error Handling

Errors should keep the current user-facing context:

- source open failures: `opening FLM startup source <path>: <error>`;
- missing runtime directory: `FLM <path> has no runtime directory`;
- wrong runtime architecture: `FLM <path> is not Qwen3.6 dense v1`;
- tokenizer failures: `loading FLM Qwen tokenizer: <error>`;
- weight failures: `load FLM weights: <error>`.

If the startup already owns an FLM source, the weight path should not attempt a
fallback open. A failure to load required tensors from that source is a format or
artifact error and should surface directly.

## Testing

Unit coverage:

- shared FLM source open options enable INT4 aliases for the native INT4 runtime;
- shared FLM source open options enable payload verification when
  `--verify-flm-hashes` is set;
- startup no longer has a special "defer hash verification to weights" option;
- non-FLM weight loading still uses the existing bake/HF path.

Integration coverage:

- the existing env-gated Qwen3.6 27B no-HF FLM smoke still runs one token;
- the smoke asserts config and tokenizer came from FLM;
- the smoke asserts weights came from the already-open FLM source;
- the smoke asserts exactly one FLM source open log line;
- the smoke asserts no `[fetch]`, `[bake]`, `config.json`, `tokenizer.json`, or
  `.supersonic` access appears in the binary output.

## Success Criteria

- `cargo test -q -p runner qwen35_startup` passes.
- `cargo test -q -p runner flm` passes.
- `cargo test -q -p model-store flm` passes.
- With the local Qwen3.6 27B INT4 FLM artifact, the no-HF runner smoke passes
  with `--verify-flm-hashes`.
- Smoke output proves a single FLM source open and already-open weight loading.
- No behavior changes for normal HF-directory model sources.

## Follow-Up

After this lifecycle is complete, the next strong FLM stages are:

1. Qwen3.6 35B-A3B MoE FLM as a normal model source.
2. FLM binary runtime assets for any remaining JSON compatibility assets.
3. Load-performance work such as larger sequential reads, pinned staging, or
   GPUDirect-shaped transfer plans.
