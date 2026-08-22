# Qwen3.8 ROCm Product Slimming Design

**Date:** 2026-08-23

**Status:** Approved design pending implementation planning

## Purpose

Replace the broad, internally inconsistent SuperSonic product surface with a
deliberately narrow inference engine optimized for Qwen3.8-27B on ROCm/HIP.
The refactor must preserve the working custom GQH GGUF inference path and its
measured performance while removing dormant models, backends, features, CLI
options, tests, and documentation.

FLM remains an internal background project and the intended long-term loader
format. It is not part of the public product contract until its Qwen3.8 path
passes the same artifact, correctness, generation, and performance gates as
the direct GQH GGUF path.

## Product Contract

The supported product is exactly:

- ROCm/HIP inference.
- Qwen3.8-27B.
- Direct loading of project-specific custom GQH GGUF artifacts.
- AMD `gfx1100` and `gfx1201` targets.
- Single-sequence inference.
- Greedy generation.
- Optional Qwen3.8 NextN/MTP speculative generation.
- Explicit model selection with `--model qwen3.8-27b`.
- A model directory containing `config.json`, tokenizer data, and the chat
  template.
- A GQH GGUF supplied with `--gguf-file`.

The product makes no compatibility commitment for CUDA, Metal, other Qwen
versions, other model families, generic GGUF files, external batching, or
removed flags and environment variables.

## Positioning

SuperSonic is a performance-specialized ROCm/HIP inference engine for
Qwen3.8-27B. It executes custom GQH GGUF artifacts through hand-tuned fused
decode, prefill, and dequantization paths for supported AMD GPUs. Its
intentionally narrow product surface is built around maximum measured
performance with reproducible correctness and benchmark evidence.

The product is not described primarily as a megakernel engine. Persistent
megakernels remain one implementation technique alongside split kernels, HIP
graphs, fused paths, and future optimizations. Performance claims must identify
the GPU, artifact, workload, build, measurement method, and correctness gate.
Use "maximum measured" rather than unqualified "fastest" or "ultimate" claims.

## Architectural Approach

Use a surgical in-place reduction. First freeze the known-working Qwen3.8 GQH
behavior with focused tests. Then simplify the public entry point and remove
unreachable systems in dependency order. Do not create a clean-room rewrite or
retain disabled legacy code merely behind build flags.

The initial runtime chain remains:

```text
runner
  -> Qwen3.8 loader currently housed in qwen35
  -> runtime decode and prefill
  -> HIP 4B, GQH, and MTP kernel interfaces
  -> gpu-hal HIP
```

Several required Qwen3.8 components currently have legacy names:

- `qwen35`, `Qwen35Weights`, and `ModelFamily::Qwen35` contain the current
  Qwen3.8 loader and 4B geometry path.
- `build_int4_scale_descs` carries GQH sidecar/header pointers required by the
  current kernel ABI.
- DFlash-named fused verification helpers are used by Qwen3.8 MTP.
- `MetalV2DecodeScratch` is used by the HIP MTP path.

These components must be isolated and renamed before their surrounding legacy
systems are deleted. File- or crate-level deletion must follow call-site proof,
not naming assumptions.

## Retained Workspace Boundary

The target product workspace contains:

- `core`, narrowed to Qwen3.8 and AMD architecture data.
- `gpu-hal`, narrowed to HIP.
- `kernel-ffi`, narrowed to the HIP 4B, prefill, GQH, and MTP interfaces.
- `model-store`, retaining custom GGUF/GQH readers and internal FLM
  foundations.
- The current `qwen35` implementation, renamed to Qwen3.8 after its required
  dependencies are isolated.
- `runtime`, narrowed to direct Qwen3.8 generation.
- `runner`, narrowed to the public CLI.

Remove from the product workspace:

- `qwen35_dflash` and outer DFlash tree/rollback functionality.
- `qwen36_moe`.
- The current server, whose input path does not implement the public GQH GGUF
  contract.
- The broad benchmark, kernel-lab, and machine-profile frameworks.
- Obsolete runner diagnostics and model-specific binaries.

Small performance reporting and machine-identification capabilities required
for reproducible Qwen3.8 measurements may be retained or rebuilt as narrow
modules owned by the Qwen3.8 executable.

## Loader and Artifact Boundary

The public loader accepts a project-specific GQH GGUF, not arbitrary GGUF.
The artifact may contain multiple required encodings, including Q2_K/Q3_K
embeddings, Q8_0, GGML-K variants, ROCmFP mixes, and GQH variants. The
`model-store` reduction must preserve every encoding exercised by the canonical
Qwen3.8 artifact.

The GGUF supplies weights but not the full startup contract. `--model-dir`
continues to supply `config.json`, tokenizer data, and chat-template metadata.
Documentation and CLI errors must make this pairing explicit.

FLM codec and model-store foundations remain internal, compile-tested, and
clearly identified as background work. Remove Qwen3.6-specific FLM product
lanes, public FLM CLI options, upload probes, bake/download flows, and public
FLM claims. Promotion of FLM into the public CLI requires a later design and
the same gates as GQH GGUF.

## CLI Contract

The public command retains:

```text
supersonic
  --model qwen3.8-27b
  --model-dir <config-and-tokenizer-directory>
  --gguf-file <custom-gqh-gguf>
  --prompt <text> | --chat
  --max-new-tokens <count>
  [greedy generation controls]
  [Qwen3.8 MTP controls]
  [device, context, and prefill controls]
  [structured timing and generated-token output]
```

HIP is implicit. Remove `--backend` and `SUPERSONIC_BACKENDS`. Keep standard
ROCm controls such as `HIP_ARCH`, `HIP_VISIBLE_DEVICES`, `ROCM_PATH`, and
`HIP_PATH`. Retain only Qwen3.8/GQH tuning variables that control a supported
or explicitly diagnostic path; rename required `QWEN35` variables to Qwen3.8
terminology.

Remove public options for:

- FLM, safetensor bakes, downloads, and source selection.
- CUDA and Metal.
- Qwen3.5, Qwen3.6, MoE, Gemma, Phi, and Llama.
- External batch generation.
- Outer DFlash and SpecPrefill.
- Certified-KV and KV-FP8.
- Generic quantization selection and old Q4KM/GPTQ bake paths.
- Oracle, teacher-forced, replay, and legacy validation modes.
- Legacy traces, forced kernels, and component compatibility switches not
  required by the Qwen3.8 product gates.

There is no backward-compatibility period. Removed options fail through normal
CLI unknown-option or invalid-value handling. `--model` stays explicit as the
model contract and future extension point; its only accepted value is
`qwen3.8-27b`.

## Runtime and Kernel Reduction Order

1. Freeze direct GQH GGUF loading, decode, prefill, chat generation, and MTP
   behavior with focused tests.
2. Replace family/backend dispatch in `runner` with the direct Qwen3.8 path and
   reduce the CLI.
3. Remove Qwen3.6 MoE, outer DFlash, SpecPrefill, Certified-KV, oracle, bake,
   download, and obsolete runner paths.
4. Rehome shared MTP verification and scratch types, then remove Metal and CUDA
   modules and build branches.
5. Reduce model/architecture registries, kernel groups, and model-store modules
   to the retained contract plus internal FLM foundations.
6. Remove FP8, KV-FP8, old VMM, and stale descriptor/state fields only after
   call-site checks prove the direct GQH and MTP paths do not use them.
7. Rename legacy Qwen3.5 identifiers to Qwen3.8 terminology after all retained
   product gates are green.

Each phase must leave production binaries buildable and the gates appropriate
to that phase green. Deletions are reviewed as behavior changes, not treated as
mechanical cleanup.

## Testing Contract

### CPU-Safe Pull-Request CI

The final CPU-safe gate runs:

- `git diff --check`.
- `cargo fmt --check` after the retained baseline is formatted once.
- `cargo check --workspace --all-targets`.
- GQH codec, header, pointer registry, and official-vector CPU tests.
- Qwen3.8 MTP acceptance-policy tests.
- CLI contract tests for the retained model and required paths.
- Negative CLI tests proving removed flags and unsupported model names fail.
- Narrow support-matrix and kernel-source manifest validators.
- Documentation-link and forbidden stale-term checks over active public docs.

All unit tests in this tier must be safe on a host without a GPU. GPU-dependent
tests must skip only when they are outside a GPU workflow and must never turn a
configured but missing CI artifact into a pass.

### ROCm gfx1201 Self-Hosted CI

The R9700 workflow must:

- Select the `gfx1201` device explicitly.
- Build with `HIP_ARCH=gfx1201`.
- Assert that the configured GQH GGUF and model-directory inputs exist.
- Run official GQH vector decode and matvec parity serially.
- Run the canonical Qwen3.8 artifact acceptance suite.
- Run deterministic component decode.
- Run chat-template generation.
- Compare ordinary greedy and MTP-generated token sequences.
- Record warmup and median throughput telemetry.

Performance telemetry is initially nonblocking. A throughput threshold becomes
blocking only after variance has been measured over repeated CI runs and the
threshold is documented with its artifact and workload.

### Test Removal

Delete tests for unsupported models, backends, and removed features, including
Qwen3.5, Qwen3.6, MoE, outer DFlash, SpecPrefill, CUDA, Metal, Gemma, Phi,
Llama, Certified-KV, obsolete oracles, old architecture matrices, and legacy
bakes. Preserve or rewrite only tests that exercise a dependency required by
the direct Qwen3.8 GQH or internal FLM foundation.

Canonical product tests include:

- GQH model-store vectors.
- GQH kernel-FFI codec, header, decode, and matvec parity.
- Qwen3.8 GQH GGUF geometry and artifact crawl.
- Qwen3.8 deterministic decode and chat generation.
- MTP token acceptance and ordinary-versus-MTP token equivalence.

Artifact-dependent tests accept paths through explicit environment variables.
Local runs may skip when no artifact is configured. GPU CI validates the paths
before invoking tests, making a missing artifact a job failure.

## CI Workflows

Replace the broad gfx1100 kernel-lab workflow with:

1. A CPU-safe pull-request workflow covering formatting, all-target
   compilation, focused unit tests, manifests, CLI behavior, and documentation.
2. A self-hosted `gfx1201` workflow covering artifact ingestion, GPU parity,
   deterministic generation, MTP equivalence, and performance telemetry.

GPU tests run serially with an explicit device and bounded job timeout. The
workflow includes GPU-idle checks and publishes structured correctness and
performance artifacts.

## Documentation Architecture

The active public documentation is limited to:

- `README.md`: scope, prerequisites, one quickstart, and a qualified current
  measurement.
- `docs/build-and-run.md`: ROCm setup, `HIP_ARCH`, artifact pairing, CLI, and
  MTP usage.
- `docs/supported-matrix.md`: Qwen3.8 GQH GGUF on supported AMD targets, with
  status and gates.
- `docs/artifact-format.md`: the project-specific GQH GGUF contract and its
  distinction from generic GGUF support.
- `docs/testing.md`: CPU and GPU gates.
- `docs/benchmarks.md`: reproducible measurement and profiling recipes.
- `docs/performance.md`: dated, artifact-qualified results.

Contributor documentation covers the narrowed repository architecture,
retained kernel interfaces, internal FLM status, and validation conventions.

Delete obsolete active-branch documentation rather than maintaining an archive
inside the product tree. Git history remains the archive. This includes old
feature matrices, detailed historical performance tables, DFlash, SpecPrefill,
Certified-KV, multi-backend memory documentation, obsolete bring-up and
optimization notes, historical plans/specs unrelated to this design, and
papers for removed product features.

`AGENTS.md` is rewritten as the Qwen3.8/ROCm contributor contract. Duplicate or
contradictory guidance such as `CLAUDE.md` is removed or reduced to a pointer to
the canonical file.

## Error Handling

- Missing `--model-dir`, `config.json`, tokenizer data, or `--gguf-file`
  produces a direct, actionable startup error.
- A non-GQH or incompatible GGUF is rejected before GPU allocation where
  practical.
- An unsupported `--model` value is rejected by CLI parsing.
- Unsupported AMD architectures fail explicitly with the supported target
  list.
- Missing GPU CI artifacts fail during workflow preflight rather than being
  reported as skipped tests.
- Internal FLM paths are not reachable through public CLI arguments.

## Success Criteria

The refactor is complete when:

- The workspace and public CLI expose only the approved Qwen3.8 ROCm product.
- `--model qwen3.8-27b` remains explicit and is the only accepted model.
- `--backend` and `SUPERSONIC_BACKENDS` are gone.
- Direct custom GQH GGUF generation and optional MTP remain correct.
- FLM foundations remain internal and compile-tested without public claims.
- CPU CI passes `cargo check --workspace --all-targets` and all focused tests
  without requiring a GPU.
- The R9700 workflow passes artifact, kernel parity, deterministic generation,
  and MTP equivalence gates.
- Active public documentation contains no unsupported model/backend/feature
  instructions.
- Performance wording and published numbers are qualified and reproducible.
- The retained code no longer carries removable CUDA, Metal, other-model,
  Certified-KV, DFlash, SpecPrefill, or broad framework surface.

## Explicit Non-Goals

- Backward compatibility for removed CLI flags, environment variables, or
  model identifiers.
- Generic GGUF compatibility.
- Public FLM inference before parity work is separately designed and validated.
- HTTP serving.
- Multi-user or batched serving.
- CUDA or Metal support.
- Support for model families other than Qwen3.8-27B.
- Reintroducing removed features preemptively for possible future use.
