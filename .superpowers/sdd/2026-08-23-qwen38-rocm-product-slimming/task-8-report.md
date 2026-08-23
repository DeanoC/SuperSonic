# Task 8 report — keep GQH with internal FLM foundations

## Scope and base

Implemented Task 8 from base `ad30d20`.  The retained product artifact path is
the custom GQH GGUF loader.  FLM remains an internal, CPU-only format
foundation: its module is `#[doc(hidden)]`, it is not re-exported from the
model-store root, and no runner startup path or `--flm-file` option reaches it.

## TDD evidence

The characterization test was created first at
`crates/model-store/tests/flm_internal_contract.rs` and run before adding the
production APIs:

```text
cargo test -p model-store --test flm_internal_contract -- --nocapture
RED: failed to compile: model_store::codec and
flm::FlmTensorDescriptor were intentionally not present yet
```

After adding the legacy-surface assertions, the second RED run proved the
deletion test was active against the old tree:

```text
cargo test -p model-store --test flm_internal_contract -- --nocapture
3 passed, 1 failed: model-store still exposes pub mod baker;
```

The final GREEN boundary test covers four contracts: GGUF qtype ↔ internal
FLM codec identity, a CPU-only internal tensor-descriptor encode/decode
round-trip, absence of a runner FLM startup route, and absence of the legacy
model-store bake/distribution modules and Qwen3.8 bake loader surface.

## Retained API boundary

- `model_store::gguf`, `gqh`, `dmix2`, `q2k`, and `q3k` remain available for
  direct GQH GGUF loading and the encodings exercised by the canonical artifact.
- `model_store::codec` owns the neutral GGUF qtype / FLM codec ID table:
  GQH3, GQH2-H, GQH2-C, and GQH4 map to GGUF qtypes 108–111 and FLM codec IDs
  13–16.  `gqh.rs` and `flm.rs` consume this one table rather than Qwen runtime
  descriptors.
- `flm::FlmTensorDescriptor` aliases the existing internal logical-tensor
  primitive and has CPU-only `encode`/`decode` (`to_bytes`/`from_bytes`) checks.
  The descriptor wire helper validates version, reserved fields, rank/shape,
  UTF-8 name, and exact length; it does not allocate GPU state or load a model.
- Root-level bake/store/manifest exports and GPU-direct FLM product loading
  are absent.  The remaining FLM runtime-directory parser and format structs
  are retained as internal background foundations explicitly covered by the
  task brief.

## Deletion and dependency inventory

Deleted from `crates/model-store/src`:

- `baker.rs`: HF/safetensors baking, Q4KM/GPTQ conversion, and bake assembly;
- `fetch.rs`: published-bake/Hugging Face fetch and archive handling;
- `manifest.rs`: old artifact/layout manifests and layout tags;
- `store.rs`: baked-store/GPU-direct loading and storage arena surface; and
- `transforms.rs`: legacy weight transformation helpers.

Removed the dead Qwen3.8 safetensors/HF loader (`crates/qwen38/src/loader.rs`),
its `BakedStore`/`LayoutTag`/`load_baked` path, and the corresponding unused
`memmap2`/`safetensors` Qwen dependency.  `model-store` now retains only
`gpu-hal`, `memmap2`, `half`, and `thiserror`; its removed serialization,
network, archive, hashing, locking, and temporary-file dependencies are no
longer declared.  `Cargo.lock` was regenerated through Cargo's dependency
resolution.

## GREEN verification

Required Task 8 checks:

```text
cargo test -p model-store --lib 'gqh::tests::' -- --nocapture
5 passed

cargo test -p model-store --test flm_internal_contract -- --nocapture
4 passed

HIP_ARCH=gfx1201 cargo check --workspace --all-targets
finished successfully (warnings are pre-existing unused/dead-code warnings)
```

Additional checks:

```text
cargo test -p model-store --lib -- --nocapture
30 passed
cargo test -p runner --test qwen38_cli_contract -- --nocapture
5 passed
python3 tools/check-retained-source-terms.py
passed
python3 tools/check-kernel-groups.py
kernel groups ok: 2 groups, 4 bridge sources, 6 tracked support sources
python3 tools/check-support-matrix.py
support matrix ok: 2 entries cover 2 arches
rustfmt --edition 2021 --check <all changed Rust files>
passed
cargo metadata --locked --offline --no-deps --format-version 1
passed
git diff --check
passed
```

No GPU execution was attempted in this CPU/container pass; the R9700/GQH
runtime gates remain the CI responsibility.  The source boundary test is
deliberately negative and includes the forbidden FLM spellings as test data;
it scans only `crates/runner/src` when checking the product route.

## Commit

Required commit subject:

```text
refactor(model-store): keep GQH with internal FLM foundations
```

## Concerns

- `flm` remains module-visible for the integration characterization test, but
  `#[doc(hidden)]` and the lack of root re-exports mark it as internal rather
  than a supported product API.
- Optional legacy-looking INT4 fields remain in the Qwen3.8 runtime struct for
  descriptor layout compatibility, but no bake, conversion, safetensors, or
  FLM loader populates or dispatches them on the direct GQH path.
- GPU-backed artifact loading and the R9700 correctness/throughput gates still
  need to run on the self-hosted HIP environment.
