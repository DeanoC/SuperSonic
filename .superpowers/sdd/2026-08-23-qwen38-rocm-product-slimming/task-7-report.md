# Task 7 report — make Qwen3.8 the sole model identity

## Scope and base

Implemented from Task 6 base `e168187` (`fix(hip): drop disabled non-hip
compatibility surfaces`).  Task 6 deleted `crates/core/src/backend.rs`; it was
not recreated.  Registry/runtime code continues to use the HIP-only backend
type supplied by `gpu-hal`.

## TDD evidence

The registry tests were added before the implementation changes.  The RED
run was:

```text
cargo test -p supersonic-core registry -- --nocapture
2 failed
```

The failures were intentional: the old parser still accepted `qwen38-27b`,
and the registry still reported the Qwen35 family and unsupported architecture
rows.  The GREEN run is:

```text
cargo test -p supersonic-core registry -- --nocapture
2 passed
```

The tests now require exact `qwen3.8-27b` parsing, exactly `gfx1100` and
`gfx1201`, HIP/Qwen3.8 on every row, and rejection of removed aliases and
architectures.

## Rename map

- `crates/qwen35/` → `crates/qwen38/` via `git mv`; package and workspace
  dependency names are now `qwen38`.
- `Qwen35Weights` → `Qwen38Weights`, with corresponding loader, descriptor,
  state, runtime, runner, test, and import updates.
- Runner modules and entry points were renamed from `qwen35_*` to `qwen38_*`.
- `ModelFamily::Qwen35`/`FamilyParams::Qwen35`/`Qwen35KernelParams` became
  `Qwen38`/`Qwen38`/`Qwen38KernelParams`.
- Model-store's active Qwen baker/export and manifest fixture use `qwen38`.
- Kernel FFI's Rust module and manifest path are `qwen38`; the stale launch
  preset/fallback surface was removed rather than carried as an alias.
- The active support matrix and tool inventory no longer advertise Qwen3.5;
  matrix coverage is limited to the two HIP Qwen3.8 GQH lanes.

## Deliberate ABI/wire exceptions

The following spellings are not public model identities and were retained with
boundary comments:

- Historical `supersonic_qwen35_*` HIP bridge symbols in the Rust extern blocks
  and retained bridge sources, including the Task 6 MTP restore symbol.
- Historical bridge object/archive names (`qwen35_megakernel_hip*.o` and
  `libqwen35_megakernel_hip.a`) required by the link ABI.
- The C++ descriptor mirror spellings (`Qwen35DecodeLayerDesc` and
  `Qwen35INT4ScaleDesc`) whose field order/layout is an external ABI contract.
- The custom GQH artifact's historical `general.architecture = "qwen35"`
  wire key.  Startup accepts it only while validating the Qwen3.8 geometry;
  no qwen35 model alias or compatibility route was added.

Kernel boundary comments identify these as retained ABI/compiler spellings,
not alternate products.  The retained GQH schema and descriptor layouts are
unchanged.  Internal FLM `qwen36` schema and docs were left for Tasks 8 and 10,
respectively.

## GREEN verification

```text
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
finished successfully

cargo test -p runner --test qwen38_cli_contract -- --nocapture
5 passed

cargo test -p runner --test qwen38_startup_contract -- --nocapture
12 passed

HIP_ARCH=gfx1201 cargo test -p supersonic-runtime mtp_accept_tests --lib -- --nocapture
3 passed

cargo test -p kernel-ffi --lib 'gqh::tests::maps_gguf_and_flm_ids' -- --nocapture
1 passed

cargo test -p model-store --lib 'gqh::tests::' -- --nocapture
5 passed

python3 -m unittest tests.test_kernel_groups -v
10 passed

python3 tools/check-kernel-groups.py
kernel groups ok

python3 tools/check-support-matrix.py
support matrix ok: 2 entries cover 2 arches

python3 tools/check-retained-source-terms.py
retained Qwen3.8 MTP source-boundary check passed

HIP_ARCH=gfx1201 cargo run -p runner --bin supersonic -- --help
passed; help advertises only Qwen3.8/GQH/ROCm options

git diff --check
passed
```

The complete `cargo test -p model-store --lib` suite was also attempted: 129
tests passed and one existing GPU-I/O test failed:
`store::tests::virtual_arena_range_load_names_gpu_direct_backend_without_fallback`.
VMM capability passed, then `hipFileSetParameterBool AllowCompatMode=false`
returned `Internal GPU IO library error`.  This is an environment/GPU-I/O
limitation, not a Qwen3.8 rename failure.

## Commit

Required commit subject:

```text
refactor(qwen38): make Qwen3.8 the sole model identity
```

## Concerns

- GPU execution and VMM-backed model-store coverage remain unavailable in this
  container; run the retained GQH/MTP suites on the `gfx1201` CI runner.
- Historical Qwen3.5 strings remain only in the documented HIP ABI/object/wire
  boundaries and negative legacy-name validators described above.  No active
  Rust product import, type, parser alias, registry row, or support-matrix lane
  uses the old identity.

## Fix round 1 — active kernel environment controls

The first fix-round RED check was:

```text
python3 -m unittest \
  tests.test_kernel_groups.HipOnlyBuildSurfaceTests.test_active_bridges_have_no_legacy_qwen35_environment_controls -v
FAIL: 9 legacy QWEN35 environment controls found in the retained bridges
```

The retained bridge controls were renamed to their QWEN38 forms, and the
Qwen3.5 comments in the retained HIP source/bridges were corrected to Qwen3.8
terminology.  The focused GREEN evidence is:

```text
python3 -m unittest \
  tests.test_kernel_groups.HipOnlyBuildSurfaceTests.test_active_bridges_have_no_legacy_qwen35_environment_controls -v
passed

python3 -m unittest tests.test_kernel_groups -v
11 passed

python3 tools/check-retained-source-terms.py
retained Qwen3.8 MTP source-boundary check passed

HIP_ARCH=gfx1201 cargo check -p kernel-ffi --lib
finished successfully

HIP_ARCH=gfx1201 cargo test -p kernel-ffi --lib 'gqh::tests::maps_gguf_and_flm_ids' -- --nocapture
1 passed

git diff --check
passed
```

Fix-round commit: `fix(qwen38): remove legacy kernel environment controls`

## Fix round 2 — canonical kernel geometry comments

The second fix-round RED checks were added before the final source edits:

```text
python3 -m unittest \
  tests.test_kernel_groups.HipOnlyBuildSurfaceTests.test_retained_qwen38_kernel_comments_use_canonical_geometry \
  tests.test_kernel_groups.HipOnlyBuildSurfaceTests.test_retained_source_validator_scans_kernel_geometry -v
FAIL: retained HIP comments still said "Processes all 24 decoder layers";
      the validator fixture also found no Qwen3.5/"24 total" violations because
      the validator did not yet scan the retained HIP sources
```

The retained HIP comments now use the canonical Qwen3.8-27B 64-layer contract,
including the partial rotary dimension.  `check-retained-source-terms.py` now
scans both retained full-attention HIP sources, rejects dotted Qwen3.5 product
references and stale non-canonical geometry/count language, and requires the
canonical layer/rotary markers.  Historical `qwen35` ABI/compiler spellings
remain intentionally outside this product-reference check.

The focused GREEN evidence is:

```text
python3 -m unittest \
  tests.test_kernel_groups.HipOnlyBuildSurfaceTests.test_retained_qwen38_kernel_comments_use_canonical_geometry \
  tests.test_kernel_groups.HipOnlyBuildSurfaceTests.test_retained_source_validator_scans_kernel_geometry -v
2 passed

python3 -m unittest tests.test_kernel_groups -v
13 passed

python3 tools/check-retained-source-terms.py
retained Qwen3.8 MTP source-boundary check passed

python3 tools/check-kernel-groups.py
kernel groups ok: 2 groups, 4 bridge sources, 6 tracked support sources

python3 tools/check-support-matrix.py
support matrix ok: 2 entries cover 2 arches

HIP_ARCH=gfx1201 cargo check -p kernel-ffi --lib
finished successfully

HIP_ARCH=gfx1201 cargo test -p kernel-ffi --lib 'gqh::tests::maps_gguf_and_flm_ids' -- --nocapture
1 passed

git diff --check
passed
```

Fix-round commit: `fix(qwen38): correct canonical kernel geometry comments`
