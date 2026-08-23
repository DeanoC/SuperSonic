# Task 8 report — keep GQH with internal FLM foundations

## Scope and base

Implemented Task 8 from base `ad30d20`.  The retained product artifact path is
the custom GQH GGUF loader.  FLM remains an internal, CPU-only format
foundation: its `codec` and `flm` modules are crate-private (`mod`, not
`pub mod`), and no runner startup path or `--flm-file` option reaches it.

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

The final GREEN boundary test covers four external contracts: public GGUF
qtype mappings, absence of a runner FLM startup route, absence of the legacy
model-store bake/distribution modules and Qwen3.8 bake loader surface, and
absence of public FLM modules/FLM-named GQH methods.  Internal FLM
characterization was moved into `flm.rs` unit tests.

## Retained API boundary

- `model_store::gguf`, `gqh`, `dmix2`, `q2k`, and `q3k` remain available for
  direct GQH GGUF loading and the encodings exercised by the canonical artifact.
- Private `codec` owns the neutral GGUF qtype / internal FLM codec ID table:
  GQH3, GQH2-H, GQH2-C, and GQH4 map to GGUF qtypes 108–111 and FLM codec IDs
  13–16.  `gqh.rs` and `flm.rs` consume this one table rather than Qwen runtime
  descriptors.  The GQH methods that mention FLM are `pub(crate)` only.
- Private `flm` unit coverage parses the production logical-tensor row table
  and string pool, compares the parsed descriptor semantics, and parses a
  minimal Qwen3.8 `FlmRuntimeDirectory`; there is no standalone FTD1 helper
  format or external FLM type path.
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
31 passed
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

The source boundary test is deliberately negative and includes the forbidden
FLM spellings as test data; it scans only `crates/runner/src` when checking the
product route.  GPU execution remains an external R9700 gate.

## Commit

Required commit subject:

```text
refactor(model-store): keep GQH with internal FLM foundations
```

## Concerns

- `flm` and `codec` are now genuinely crate-private; the integration contract
  does not import either module.
- Optional legacy-looking INT4 fields remain in the Qwen3.8 runtime struct for
  descriptor layout compatibility, but no bake, conversion, safetensors, or
  FLM loader populates or dispatches them on the direct GQH path.
- GPU-backed artifact loading and the R9700 correctness/throughput gates still
  need to run on the self-hosted HIP environment.

## Fix round 1 — review findings

### RED evidence

The external contract test was rewritten first so it no longer imported
private FLM modules or the invented descriptor format.  Against the Task 8
commit it correctly failed on the still-public codec module:

```text
cargo test -p model-store --test flm_internal_contract -- --nocapture
3 passed, 1 failed: assertion failed: !lib_rs.contains("pub mod codec;")
```

The new internal FLM unit test was then added before the identity/parser
implementation.  It failed to compile because the Qwen3.8 architecture
constant did not yet exist:

```text
cargo test -p model-store --lib flm -- --nocapture
RED: cannot find value ARCH_QWEN3_8_DENSE
```

### Redesign and GREEN evidence

- `codec` and `flm` changed from `pub mod` to private `mod`; the integration
  test now observes only `model_store::gqh` public behavior and source/runner
  absence.
- Removed the FTD1 alias and all encode/decode helpers.  Internal tests now use
  the production logical-tensor row-table format (44-byte rows plus string
  pool) and compare direct row parsing with the same descriptor returned by
  `FlmRuntimeDirectory::parse`.
- `GqhRung::from_flm_codec` and `flm_codec` are `pub(crate)`, while public GQH
  callers see only GGUF qtypes.  Model-store unit coverage checks all four
  pairs, including 111 ↔ 16; kernel-FFI coverage now explicitly checks codec
  16 → `RUNG_GQH4`.
- Removed the Qwen3.6 dense/MoE schema structs, architecture/model/tensor ABI
  IDs, MoE parser and accessors, and stale Qwen3.6 chat-template identity.
  The retained private runtime parser accepts only the internal Qwen3.8 dense
  identity (`ARCH_QWEN3_8_DENSE = 3`, model/tensor/profile IDs = 3).

Focused and full GREEN results:

```text
cargo test -p model-store --test flm_internal_contract -- --nocapture
4 passed

cargo test -p model-store --lib flm -- --nocapture
25 passed

cargo test -p model-store --lib 'gqh::tests::' -- --nocapture
5 passed

cargo test -p model-store --lib -- --nocapture
31 passed

cargo test -p kernel-ffi --lib 'gqh::tests::maps_gguf_and_flm_ids' -- --nocapture
1 passed (including FLM codec 16)

cargo test -p kernel-ffi --lib -- --nocapture
9 passed, 1 ignored (R9700 artifact test)

cargo test -p runner --test qwen38_cli_contract -- --nocapture
5 passed

cargo test -p runner --test qwen38_startup_contract -- --nocapture
12 passed

HIP_ARCH=gfx1201 cargo check --workspace --all-targets
finished successfully
```

The source validators, locked/offline metadata check, rustfmt check, and
`git diff --check` also pass after the fix.

### Canonical artifact and R9700 readiness

No artifact was downloaded, generated, or modified.  The approved paths are
present in this environment:

```text
/home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq.gguf
  10,675,462,336 bytes
/home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq-8192.gguf
  10,676,036,416 bytes
/data/models/Qwen3.8-27B/config.json
/data/models/Qwen3.8-27B/tokenizer.json
/data/models/Qwen3.8-27B/tokenizer_config.json
```

With
`SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1`,
`SUPERSONIC_GQH_GGUF=/home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq.gguf`,
and `SUPERSONIC_QWEN38_MODEL_DIR=/data/models/Qwen3.8-27B`, the CPU host
artifact checks `rung2_hf_config_matches_gguf_geometry` and
`rung3_q2k_embed_row_is_finite` each pass.  The runner startup contract also
passes its real-artifact checks when those canonical files are available.

The sandbox blocker for GPU gates is exact: `/dev/kfd` and `/dev/dri` are not
present.  `rocm-smi` can enumerate GPU 1 as AMD Radeon AI PRO R9700,
`gfx1201`, but HIP tests report `hipGetDevice`/status 100 without those device
nodes.  The controller must run the ignored GQH upload/decode/throughput gates
on the R9700 host with those device nodes exposed, set
`HIP_ARCH=gfx1201`, select the R9700 via `HIP_VISIBLE_DEVICES`, and provide the
artifact/model-dir environment variables above.  The 8192 variant additionally
uses `SUPERSONIC_GQH_8192_GGUF` pointing at the `-8192.gguf` path.

## Fix-round commit

Required focused commit subject:

```text
refactor(model-store): make FLM foundation private and Qwen3.8-native
```
