# Deterministic Raw-Q6 Output-Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and prove a private `gfx1201` raw-Q6 scalar output-head reference without changing SuperSonic's production route.

**Architecture:** Add an exact CPU oracle in `model-store`, then a shape-locked HIP kernel and private Rust FFI wrapper that produce F32 logits for explicit BF16-RNE argmax. A contributor-only Cargo feature lets ignored artifact tests exercise the scalar route; the ordinary production build remains WMMA. A generated-code validator and fresh-process locked-clock diagnostic gate decide whether certified correction and overnight qualification deserve separate follow-on plans.

**Tech Stack:** Rust 2021, ROCm/HIP C++17, AMDGPU code objects, Python 3 `unittest`, existing `gpu-hal` buffers and benchmark telemetry helpers.

**Spec:** `docs/superpowers/specs/2026-08-25-deterministic-raw-q6-output-head-design.md`

## Global Constraints

- The public model remains explicit `qwen3.8-27b` with paired `--model-dir` and `--gguf-file` inputs.
- This plan supports only BF16 `batch=1`, `m=1`, `k=5120`, raw GGML Q6_K `qtype=14`, `n=248320`, and AMD `gfx1201`.
- Unsupported dtype, shape, qtype, AWQ, architecture, or required-pointer combinations fail explicitly; there is no fallback from the private scalar entry point.
- The production runner, public CLI, default runtime route, `gfx1100` route, prefill route, and MTP verification route do not change in this plan.
- Coefficients use `RN_F32(RN_F32(d * s) * q)`; accumulation uses 160 ordered F32 FMAs per lane and the fixed shuffle/add offsets `16, 8, 4, 2, 1`.
- A successfully audited finite/non-overflowing scalar operation graph uses `gamma(165) = 165 * 2^-24 / (1 - 165 * 2^-24)` in the later correction study.
- Observable logits use explicit finite F32-to-BF16 round-to-nearest-even and lowest-index tie selection.
- The audited kernel must contain ordinary scalar F32 MUL/FMA/ADD, no `v_fma_mix*`, WMMA, or MFMA, no spills, and an FP32 RNE/denormal-preserve descriptor.
- Artifact tests require `SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1`, the configured model/GGUF pair, `RUST_TEST_THREADS=1`, and explicit ignored-test selection.
- Performance evidence requires verified locked clocks, a declared cache/process state, balanced fresh-process routes, raw samples, and no throttle or telemetry violation.
- `.superpowers/sdd-plans/` is pre-existing private operator state and must never be staged or modified.

## File Map

- `crates/model-store/src/q6_bound.rs`: exact host-side scalar coefficient, dot, BF16, and argmax oracle.
- `crates/model-store/tests/q6_bound.rs`: CPU-safe numerical and tie-contract tests.
- `kernels/full_attention_4b.hip`: dedicated one-wave-per-row raw-Q6 scalar kernel and explicit arithmetic helpers.
- `kernels/full_attention_bridge_4b.cpp`: exact `gfx1201`/shape validation and private kernel launch.
- `crates/kernel-ffi/src/prefill_ffi.rs`: private extern declaration, Rust buffer validation, and profile label.
- `crates/kernel-ffi/tests/q6_scalar_head.rs`: ignored artifact-backed full-row CPU/GPU contract test.
- `tools/check-scalar-head-code-object.py`: bounded disassembly and descriptor validator for the named scalar symbol.
- `tests/test_scalar_head_code_object.py`: CPU-safe validator fixtures and rejection tests.
- `crates/runtime/Cargo.toml`: contributor-only `scalar-head-lab` feature.
- `crates/runtime/src/decode_engine.rs`: feature-gated lab route using existing F32 scratch and F32-as-BF16 argmax.
- `crates/runtime/tests/qwen38_gqh_decode_rung11.rs`: fixed-route repeated generation and timing tests.
- `tests/test_active_docs.py`: retain this plan in the active design allowlist.

---

### Task 1: Exact CPU scalar oracle

**Files:**
- Modify: `crates/model-store/src/q6_bound.rs`
- Modify: `crates/model-store/tests/q6_bound.rs`

**Interfaces:**
- Consumes: `decode_q6_k_block(block: &[u8]) -> Result<DecodedQ6Block, String>` and canonical 210-byte Q6_K blocks.
- Produces: `SCALAR_HEAD_ACCUMULATION_DEPTH: usize`, `raw_q6_scalar_row_f32(row: &[u8], activation_bf16: &[u16]) -> Result<f32, String>`, `f32_to_bf16_rne_finite(value: f32) -> Result<u16, String>`, and `argmax_f32_as_bf16(logits: &[f32]) -> Result<usize, String>`.

- [ ] **Step 1: Write failing coefficient, reduction, and tie tests.**

Add tests that construct two Q6 blocks with hand-selected FP16 scales, signed
subscales, low/high nibble values, and BF16 activations. Compute the expected
row with a test-local lane array and `mul_add`, then require the public oracle
to match bit-for-bit. Include the following tie cases:

```rust
assert_eq!(f32_to_bf16_rne_finite(1.003_906_25).unwrap(), 0x3f80);
assert!(f32_to_bf16_rne_finite(f32::INFINITY).is_err());
assert_eq!(argmax_f32_as_bf16(&[1.0, 1.003_0, 0.0]).unwrap(), 0);
assert_eq!(argmax_f32_as_bf16(&[-0.0, 0.0]).unwrap(), 0);
assert!(raw_q6_scalar_row_f32(&[0; 209], &[0; 5120]).is_err());
assert!(raw_q6_scalar_row_f32(&[0; 20 * 210], &[0; 5119]).is_err());
```

- [ ] **Step 2: Run the focused test and verify RED.**

Run:

```bash
cargo test -p model-store --test q6_bound -- --nocapture
```

Expected: compile failure for the missing oracle symbols.

- [ ] **Step 3: Implement the fixed host operation graph.**

Add non-generic helpers that keep the two coefficient multiplies separate and
use exactly the kernel traversal:

```rust
pub const SCALAR_HEAD_ACCUMULATION_DEPTH: usize = 165;

#[inline(never)]
fn rn_mul(left: f32, right: f32) -> f32 {
    left * right
}

pub fn raw_q6_scalar_row_f32(row: &[u8], activation_bf16: &[u16]) -> Result<f32, String> {
    if row.len() != 20 * Q6_K_BYTES || activation_bf16.len() != 5120 {
        return Err("raw Q6 scalar row requires 4200 weight bytes and 5120 BF16 values".into());
    }
    let mut lanes = [0.0f32; 32];
    for block_index in 0..20 {
        let block = decode_q6_k_block(
            &row[block_index * Q6_K_BYTES..(block_index + 1) * Q6_K_BYTES],
        )?;
        for lane in 0..32 {
            for t in 0..8 {
                let coordinate = lane + 32 * t;
                let weight = rn_mul(
                    rn_mul(block.d, f32::from(block.scales[coordinate])),
                    f32::from(block.quants[coordinate]),
                );
                let x = bf16::from_bits(activation_bf16[block_index * 256 + coordinate]).to_f32();
                lanes[lane] = weight.mul_add(x, lanes[lane]);
            }
        }
    }
    for offset in [16usize, 8, 4, 2, 1] {
        let before = lanes;
        for lane in 0..32 {
            lanes[lane] = before[lane] + before[lane ^ offset];
        }
    }
    lanes[0].is_finite().then_some(lanes[0]).ok_or_else(|| "scalar row is non-finite".into())
}
```

Implement BF16 RNE with the explicit `0x7fff + lsb` integer bias already used
by the HIP helper. `argmax_f32_as_bf16` converts every finite value with that
helper and updates the winner only on strict BF16 greater-than, preserving the
lowest index.

- [ ] **Step 4: Run the focused and complete crate tests.**

Run:

```bash
cargo test -p model-store --test q6_bound -- --nocapture
cargo test -p model-store --all-targets
```

Expected: all tests pass with no ignored configured artifact.

- [ ] **Step 5: Commit the oracle.**

```bash
git add crates/model-store/src/q6_bound.rs crates/model-store/tests/q6_bound.rs
git commit -m "test: define raw-q6 scalar head oracle"
```

### Task 2: Private `gfx1201` scalar kernel and FFI

**Files:**
- Modify: `kernels/full_attention_4b.hip`
- Modify: `kernels/full_attention_bridge_4b.cpp`
- Modify: `crates/kernel-ffi/src/prefill_ffi.rs`

**Interfaces:**
- Consumes: BF16 activation `GpuBuffer[5120]`, raw Q6_K `GpuBuffer[248320 * 4200 bytes]`, F32 output `GpuBuffer[248320]`, and the Task 1 numerical contract.
- Produces: `prefill_ffi::q6_k_scalar_head_f32(ordinal: usize, lhs: &GpuBuffer, rhs_q6: &GpuBuffer, out: &mut GpuBuffer, row_start: usize, row_count: usize) -> Result<(), GpuError>`, `prefill_ffi::device_supports_q6_scalar_head(ordinal: usize) -> Result<bool, GpuError>`, and C ABI symbol `supersonic_qwen38_hip_q6_k_scalar_head_f32`.

- [ ] **Step 1: Write Rust validation tests before declaring the extern.**

Factor a CPU-safe private validator with this exact input record:

```rust
struct Q6ScalarHeadShape {
    lhs_dtype: ScalarType,
    lhs_elems: usize,
    rhs_dtype: ScalarType,
    rhs_bytes: usize,
    out_dtype: ScalarType,
    out_elems: usize,
    row_start: usize,
    row_count: usize,
}
```

Test the valid full row and 16-row tile. Reject BF16 output, F32 lhs, short Q6
storage, zero rows, `row_start + row_count > 248320`, and arithmetic overflow.
The wrapper must reject a non-HIP backend before FFI.

- [ ] **Step 2: Run the kernel-FFI library tests and verify RED.**

Run:

```bash
cargo test -p kernel-ffi --lib q6_scalar_head -- --nocapture
```

Expected: compile failure for the missing validator/wrapper.

- [ ] **Step 3: Add explicit device arithmetic helpers and the kernel.**

Add a dedicated kernel, not a new mode on the generic matvec template:

```cpp
__device__ __forceinline__ float q6_scalar_rn_mul(float a, float b) {
    return a * b;
}

__device__ __forceinline__ float q6_scalar_rn_fma(float a, float b, float c) {
    return __builtin_fmaf(a, b, c);
}

__global__ void supersonic_qwen38_q6_k_scalar_head_f32_kernel(
    const uint16_t* lhs_bf16,
    const uint8_t* rhs_q6,
    float* out_f32,
    int row_start,
    int row_count) {
    // Four wave32 rows per 128-thread block. Decode the established Q6_K
    // mapping, traverse block then t, xor-reduce 16/8/4/2/1, lane 0 stores.
}
```

Place `#pragma clang fp reassociate(off)` and `#pragma clang fp contract(off)`
around the scalar coefficient code; the explicit `__builtin_fmaf` remains the
only fused operation requested by source. Keep these helpers local to this
kernel and let Task 3's disassembly—not source appearance—decide whether Clang
honored the contract. The row pointer is
`rhs_q6 + (row_start + local_row) * 20 * 210` and the activation conversion is
the existing exact `bf16_bits_to_f32` helper.

- [ ] **Step 4: Add the fail-closed bridge and Rust wrapper.**

The bridge checks `dtype == 2`, exact global dimensions, nulls, range, and
`hipGetDeviceProperties(...).gcnArchName` equal to `gfx1201` before enqueueing.
Return a dedicated `360..=369` status family for unsupported/invalid scalar
head calls and other nonzero statuses for HIP launch/sync failures. The Rust
wrapper validates buffer dtype and size first, then maps every nonzero status
to `GpuError`; unlike optional kernels, it never returns `Ok(false)`.
The support query uses the existing `kernel_ffi::query_gpu_info` architecture
string and returns true only for the exact `gfx1201` token.

- [ ] **Step 5: Run CPU contract tests and both architecture compile gates.**

Run:

```bash
cargo test -p kernel-ffi --lib q6_scalar_head -- --nocapture
HIP_ARCH=gfx1201 cargo check -p kernel-ffi
HIP_ARCH=gfx1100 cargo check -p kernel-ffi
```

Expected: validation tests pass; both targets compile; the private bridge would
reject execution on `gfx1100`.

- [ ] **Step 6: Commit the private kernel/FFI.**

```bash
git add kernels/full_attention_4b.hip kernels/full_attention_bridge_4b.cpp crates/kernel-ffi/src/prefill_ffi.rs
git commit -m "feat(kernel): add private raw-q6 scalar head"
```

### Task 3: Generated-code contract validator

**Files:**
- Create: `tools/check-scalar-head-code-object.py`
- Create: `tests/test_scalar_head_code_object.py`

**Interfaces:**
- Consumes: one exact `full_attention_bridge_4b.o`, `llvm-objdump`, and `llvm-readobj` output for `supersonic_qwen38_q6_k_scalar_head_f32_kernel`.
- Produces: exit 0 plus canonical JSON `{symbol, sha256, vgpr_count, spill_count, fp32_round_mode, fp32_denorm_mode, instruction_counts}` or exit 1 with precise violations.

- [ ] **Step 1: Write fixture-driven parser tests.**

Use in-memory disassembly/metadata strings. The passing fixture contains
`v_mul_f32`, `v_fma_f32`, `v_add_f32`, five `ds_bpermute_b32`/shuffle stages,
`COMPUTE_PGM_RSRC1` decoding to FP32 round mode 0 and denormal mode 3, a VGPR
count, and zero scratch/spills. Separate tests must reject each of:

```text
v_fma_mix_f32
v_wmma_f32_16x16x16_bf16
v_mfma_f32_16x16x16bf16
scratch_size: 16
FP32 round mode != 0
FP32 denormal mode != 3
missing named symbol
unexpected shuffle/add count
```

- [ ] **Step 2: Run the validator tests and verify RED.**

Run:

```bash
python3 -m unittest tests.test_scalar_head_code_object -v
```

Expected: import/file failure because the validator does not exist.

- [ ] **Step 3: Implement bounded argv-only inspection.**

Expose pure `analyze(disassembly: str, metadata: str, symbol: str) -> dict` and
`find_violations(report: dict) -> list[str]` functions for tests. The CLI accepts
only `--object PATH` and `--symbol NAME`, verifies a regular nonempty file, and
runs each LLVM tool with `subprocess.run([...], timeout=30, check=True,
capture_output=True, text=True)`. It hashes the object bytes, prints sorted
canonical JSON on success, and prints every violation to stderr on failure.

- [ ] **Step 4: Run the unit test and the real `gfx1201` object audit.**

Run:

```bash
python3 -m unittest tests.test_scalar_head_code_object -v
CARGO_TARGET_DIR=target/scalar-head-audit HIP_ARCH=gfx1201 cargo build --release -p kernel-ffi
object_path="$(find target/scalar-head-audit/release/build -path '*/out/full_attention_bridge_4b.o' -type f -print -quit)"
test -n "$object_path"
python3 tools/check-scalar-head-code-object.py \
  --object "$object_path" \
  --symbol supersonic_qwen38_q6_k_scalar_head_f32_kernel
```

Expected: validator exits 0, reports no mixed/WMMA/MFMA instructions or spills,
and reports FP32 mode `RNE`/`preserve`.

- [ ] **Step 5: Commit the audit gate.**

```bash
git add tools/check-scalar-head-code-object.py tests/test_scalar_head_code_object.py
git commit -m "test(kernel): audit scalar head code object"
```

### Task 4: Artifact-backed full-row CPU/GPU proof

**Files:**
- Create: `crates/kernel-ffi/tests/q6_scalar_head.rs`
- Modify: `crates/kernel-ffi/Cargo.toml`

**Interfaces:**
- Consumes: Task 1 oracle, Task 2 FFI, configured Q3KXL GGUF, and a deterministic 5120-value BF16 activation fixture generated in the test.
- Produces: ignored tests `q6_scalar_head_full_row_matches_cpu_oracle` and `q6_scalar_head_tiled_matches_full_row`.

- [ ] **Step 1: Write the ignored tests and verify RED.**

Load `output.weight` through `model_store::gguf::GgufFile`, require qtype 14 and
dimensions `[5120, 248320]`, upload its bytes, and generate activation bits with
this stable expression:

```rust
let activation: Vec<u16> = (0..5120)
    .map(|i| half::bf16::from_f32(((i % 257) as f32 - 128.0) / 128.0).to_bits())
    .collect();
```

Run the full GPU row twice in fresh buffers, require identical F32 bits, and
compare every row against `raw_q6_scalar_row_f32`. Require GPU and CPU BF16
bits and `argmax_f32_as_bf16` winners to match. The tiled test launches
contiguous 16-row ranges into the corresponding F32 output offsets and requires
the final byte vector to equal the full launch.

Run:

```bash
SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1 \
SUPERSONIC_GQH_GGUF="$SUPERSONIC_GQH_GGUF" \
RUST_TEST_THREADS=1 HIP_ARCH=gfx1201 \
cargo test --release -p kernel-ffi --test q6_scalar_head \
  -- --include-ignored --test-threads=1 --nocapture
```

Expected: compile/test failure until all artifact loading and buffer plumbing is
complete; a configured missing artifact must fail rather than skip.

- [ ] **Step 2: Complete only the test plumbing needed by the declared FFI.**

Add `model-store` and `half` dev dependencies only if not already present. Do
not add a runtime accessor, production route, environment selector, or host
fallback. If the full CPU scan exceeds 15 minutes, stop and optimize the
offline row iteration; do not weaken the comparison or sample vocabulary rows.

- [ ] **Step 3: Run the full-row and tiled gates twice.**

Use the exact command above in two fresh `cargo test` processes. Expected:
both processes report the same full-row digest and winner; all 248,320 CPU/GPU
BF16 comparisons and tiled/full F32 comparisons pass.

- [ ] **Step 4: Commit the artifact proof.**

```bash
git add crates/kernel-ffi/Cargo.toml crates/kernel-ffi/tests/q6_scalar_head.rs
git commit -m "test(kernel): prove raw-q6 scalar head parity"
```

### Task 5: Contributor-only runtime lab route

**Files:**
- Modify: `crates/runtime/Cargo.toml`
- Modify: `crates/runtime/src/decode_engine.rs`
- Modify: `crates/runtime/tests/qwen38_gqh_decode_rung11.rs`

**Interfaces:**
- Consumes: `prefill_ffi::q6_k_scalar_head_f32`, existing `logits_f32_buf`, `argmax_f32_as_bf16_rows`, and exact Q6_K model identity.
- Produces under feature `scalar-head-lab`: `ScalarHeadLabRoute::{ProductionWmma, RawQ6Scalar}` and `DecodeEngine::set_scalar_head_lab_route(route) -> Result<()>`. Builds without the feature expose neither symbol and remain production-WMMA.

- [ ] **Step 1: Write feature-gated route and repeatability tests.**

Add ignored tests with fixed route selection in source—not an environment
variable:

```rust
#[cfg(feature = "scalar-head-lab")]
#[test]
#[ignore = "requires configured gfx1201 Q3KXL artifact"]
fn scalar_head_lab_hello_is_repeatable() { /* two fresh engines, 32 exact tokens */ }

#[cfg(feature = "scalar-head-lab")]
#[test]
#[ignore = "requires configured gfx1201 Q3KXL artifact"]
fn scalar_head_lab_cold_chat_is_repeatable() { /* two fresh engines, 32 exact tokens */ }
```

Both use `ChatTemplate::render(..., true)`, assert Hello has 13 prompt tokens
and the established cold prompt has 23, call `prepare_hip_gqh_decode`, and
require the two scalar token vectors to be identical. Also run one production
WMMA engine from the same restored prefix and print the first divergence as
structured evidence without asserting global token equality.

- [ ] **Step 2: Run with the feature and verify RED.**

Run:

```bash
SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1 \
RUST_TEST_THREADS=1 HIP_ARCH=gfx1201 \
cargo test --release -p supersonic-runtime --features scalar-head-lab \
  --test qwen38_gqh_decode_rung11 scalar_head_lab_ \
  -- --include-ignored --test-threads=1 --nocapture
```

Expected: compile failure for the missing feature/route API.

- [ ] **Step 3: Add the compile-time-only route.**

Default `DecodeEngine` to `ProductionWmma`. The setter validates HIP, qtype 14,
hidden 5120, vocabulary 248320, and successful `gfx1201` scalar FFI support
without launching an approximate path. In `decode_step_single_kernel_impl`,
`RawQ6Scalar` writes `logits_f32_buf`; fast greedy uses
`argmax_f32_as_bf16_rows`, and host-logit mode downloads F32 then converts each
finite value through `half::bf16::from_f32(value).to_f32()` before sampling.
Production WMMA retains the existing BF16 buffer and argmax path byte-for-byte.
Do not alter `lm_head_lowbit`, prefill, or MTP block verification in this task.

- [ ] **Step 4: Prove feature isolation and deterministic generation.**

Run:

```bash
cargo test -p supersonic-runtime --lib
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
HIP_ARCH=gfx1100 cargo check --workspace --all-targets
SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1 RUST_TEST_THREADS=1 HIP_ARCH=gfx1201 \
cargo test --release -p supersonic-runtime --features scalar-head-lab \
  --test qwen38_gqh_decode_rung11 scalar_head_lab_ \
  -- --include-ignored --test-threads=1 --nocapture
```

Expected: normal builds do not contain the lab API; both architectures compile;
the `gfx1201` lab tests repeat exactly and print candidate/control divergence.

- [ ] **Step 5: Commit the private runtime route.**

```bash
git add crates/runtime/Cargo.toml crates/runtime/src/decode_engine.rs crates/runtime/tests/qwen38_gqh_decode_rung11.rs
git commit -m "test(runtime): exercise private scalar head route"
```

### Task 6: Locked-clock diagnostic decision gate

**Files:**
- Verify: all files from Tasks 1–5
- Record outside the active tree: `target/scalar-head-evidence/`

**Interfaces:**
- Consumes: fixed WMMA/scalar ignored tests, audited code-object JSON, host-prepared locked clocks, and existing benchmark telemetry collection.
- Produces: a non-publishable diagnostic bundle with balanced fresh-process raw samples and a pass/fail decision for the next plan.

- [ ] **Step 1: Run the complete CPU/build gate.**

```bash
python3 -m unittest discover -s tests -v
cargo test --workspace --all-targets
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
HIP_ARCH=gfx1100 cargo check --workspace --all-targets
cargo fmt --all --check
python3 tools/check-active-docs.py
git diff --check
```

Expected: every command exits 0; configured artifact tests are not silently
skipped when the strict environment is set in subsequent steps.

- [ ] **Step 2: Capture static identity and verify the operator's locked state.**

Use the bounded commands from `docs/benchmarks.md` to create merged AMD SMI
provenance plus ROCm and HIP version files under
`target/scalar-head-evidence/`. Call `tools.benchmark.environment` with the
declared `locked` policy before measurement. Abort on missing requested fields,
wrong physical/logical mapping, clock drift, performance-level mismatch,
power-cap mismatch, or throttle state; never change privileged host settings
from the test harness.

- [ ] **Step 3: Run balanced fresh-process WMMA/scalar rounds.**

Execute seven alternating rounds. Each route is a separate invocation of its
fixed ignored test, with the order reversed every round. Use the same Hello
chat prompt, 32 generated tokens, artifact, binary, code object, GPU, and
measurement boundary. Record the emitted `lm_head_ms`, exact token vector,
process start/end timestamps, and telemetry samples after every invocation.
Every invocation is a fresh process and is labeled `cold-load`; do not claim a
filesystem cache flush.

- [ ] **Step 4: Apply the diagnostic thresholds.**

From the seven scalar samples, calculate median and nearest-rank p95. Pass only
when median is at most 2.20 ms/token, p95 is at most 2.40 ms/token, all scalar
token vectors repeat exactly, the CPU/GPU and code-object gates remain valid,
and telemetry verification reports no errors. Always retain the same-session
WMMA samples; do not publish a speedup from this diagnostic bundle.

- [ ] **Step 5: Review scope and stop at the checkpoint.**

Inspect `git diff`, `git status --short`, and the evidence bundle. Ensure the
default runtime route is still WMMA, `gfx1100` is unchanged, no public selector
exists, and `.superpowers/sdd-plans/` is untouched. If the gate passes, write
separate plans for (a) the raw-reference 64-state certified-correction study and
(b) candidate/control/llama.cpp six-hour qualification. If it fails, remove
the contributor-only route in a reviewed cleanup rather than retaining an
unqualified second implementation.

## Out-of-Scope Follow-on Plans

This plan intentionally stops before three outcome-dependent changes:

1. self-describing outward-rounded correction sidecars and Q8 proposal/tile
   correction;
2. benchmark-schema support for distinct WMMA and scalar SuperSonic engine
   identities, followed by the six-hour quality run; and
3. tagging, production promotion, golden-series changes, and removal of the
   superseded `gfx1201` head route.

None should be planned in implementation detail until the locked scalar gate
above provides valid data.
