# Raw-Reference 64-State Certified-Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, on 64 fixed decode states, whether a Q8 proposal plus bounded exact raw-Q6 tile correction can reproduce the scalar oracle without relying on an undocumented WMMA error model.

**Architecture:** Keep this as an offline, contributor-only study. Capture the same 32 Hello-chat and 32 cold-chat states through a feature-gated test hook, treat the existing ordered raw-Q6 scalar path as the reference, and evaluate a Q8 proposal whose exclusion decisions use outward-rounded bounds and tie-aware BF16 comparisons. Exact raw-Q6 correction handles every non-excludable 16-row tile; exceeding 16 tiles is a measured fallback, never an approximate answer.

**Tech Stack:** Rust, ROCm/HIP `gfx1201`, custom GQH/Q6_K readers, BF16/F32, existing model-store bound analyzer, ignored artifact tests.

**Spec:** `docs/superpowers/specs/2026-08-25-deterministic-raw-q6-output-head-design.md`

## Global Constraints

- Keep `--model qwen3.8-27b`, paired `--model-dir`/`--gguf-file`, and the public runner unchanged.
- Support only the configured Qwen3.8-27B GQH artifact and exact `gfx1201` study route; do not add fallback or compatibility behavior.
- The ordered raw-Q6 scalar oracle, including BF16 observation and lowest-index tie semantics, is authoritative.
- Do not claim a formal bound for baseline WMMA: public documentation does not specify its internal accumulation order or accuracy.
- Every exclusion bound must round outward. Uncertified top-K or heuristic exclusion cannot satisfy correctness.
- Keep the capture hook feature-gated, ignored, strict-artifact, and absent from normal runtime artifacts.
- Stop after the study report. Do not promote a runtime route, publish a speedup, or change `gfx1100`.

---

### Task 1: Freeze the 64-state corpus contract

**Files:**
- Modify: `crates/runtime/tests/qwen38_gqh_decode_rung11.rs`
- Create: `crates/runtime/tests/fixtures/scalar_correction_64_state_manifest.json`
- Test: `crates/runtime/tests/qwen38_gqh_decode_rung11.rs`

**Interfaces:**
- Consumes: `DecodeEngine`, `scalar-head-lab`, the fixed Hello and cold-chat templates, and strict artifact preflight.
- Produces: `capture_scalar_correction_corpus(output: &Path) -> Result<(), String>` writing 64 records. Each record names exactly two separate files: `<state-id>.hidden.bf16le` containing 5,120 little-endian `u16` words (10,240 bytes), and `<state-id>.scalar.f32le` containing 248,320 little-endian finite `f32` words (993,280 bytes), plus each file's SHA-256, prompt ID/SHA-256, generation index, selected lowest-index BF16 token, artifact SHA-256, scalar contract version, and row-mapping version. Offsets into shared files and alternate encodings are forbidden in v1.

- [ ] **Step 1: Write the failing manifest and strict-capture tests.** Assert exactly two cases, 32 states per case, prompt lengths 13 and 23, contiguous generation indices, the established 32-token vectors, hidden length 5,120, logit length 248,320, unique state IDs, canonical ordering, and failure when a configured artifact/device is unavailable.
- [ ] **Step 2: Run RED.**

```bash
cargo test -p supersonic-runtime --features scalar-head-lab --test qwen38_gqh_decode_rung11 scalar_correction_manifest_ -- --nocapture
```

Expected: fail because the corpus manifest/capture API is absent.

- [ ] **Step 3: Implement the minimal feature-gated capture.** Reuse chat rendering with `add_generation_prompt=true`; capture immediately before each m=1 output head; store binary arrays little-endian and a canonical JSON manifest containing relative paths, byte counts, SHA-256, artifact digest, commit, route `RawQ6Scalar`, and exact token.
- [ ] **Step 4: Run the CPU contract tests and two strict ignored captures.**

```bash
cargo test -p supersonic-runtime --features scalar-head-lab --test qwen38_gqh_decode_rung11 scalar_correction_manifest_ -- --nocapture
SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1 \
SUPERSONIC_GQH_GGUF=/home/deano/models/qwen38-gqh-shaped.gguf \
SUPERSONIC_QWEN38_MODEL_DIR=/data/models/Qwen3.8-27B \
HIP_VISIBLE_DEVICES=1 HIP_ARCH=gfx1201 RUST_TEST_THREADS=1 \
cargo test --release -p supersonic-runtime --features scalar-head-lab \
  --test qwen38_gqh_decode_rung11 capture_scalar_correction_corpus \
  -- --exact --include-ignored --test-threads=1 --nocapture
```

Expected: 64/64 records validate and both full token lists match the fixture.

- [ ] **Step 5: Remove any general runtime accessor not needed by the fixed ignored test, prove normal-artifact isolation, and commit.**

```bash
cargo test -p supersonic-runtime --lib
cargo test -p supersonic-runtime --features scalar-head-lab --lib
git add crates/runtime/tests/qwen38_gqh_decode_rung11.rs crates/runtime/tests/fixtures/scalar_correction_64_state_manifest.json
git commit -m "test(runtime): freeze scalar correction corpus"
```

### Task 2: Make Q8 error accounting outward-rounded and self-describing

**Files:**
- Modify: `crates/model-store/src/q6_bound.rs`
- Modify: `crates/model-store/tests/q6_bound.rs`

**Interfaces:**
- Consumes: Q6_K row bytes, 5,120 BF16 activations, and the existing `q8_1_reconstruct` helpers.
- Produces the following exact Rust surface (serialization is explicit and never `repr(C)`):

```rust
pub struct BoundSidecarHeaderV1 {
    pub artifact_sha256: [u8; 32],
    pub payload_sha256: [u8; 32],
    pub scalar_instruction_sha256: [u8; 32],
    pub proposal_instruction_sha256: [u8; 32],
    pub toolchain_sha256: [u8; 32],
}
pub struct BoundNormPairV1 { pub w_f16_bits: u16, pub d_f16_bits: u16 }
pub struct BoundSidecarV1 {
    pub header: BoundSidecarHeaderV1,
    pub norms: Vec<BoundNormPairV1>, // exactly 248320 * 20, row-major
}
pub struct OutwardInterval { pub lower: f32, pub upper: f32 }
pub enum BoundError { Shape(String), Contract(String), Digest(String), NonFinite(String), Overflow(String) }
pub fn build_bound_sidecar_v1(artifact: &std::path::Path, scalar_instruction_sha256: [u8; 32], proposal_instruction_sha256: [u8; 32], toolchain_sha256: [u8; 32]) -> Result<BoundSidecarV1, BoundError>;
pub fn encode_bound_sidecar_v1(sidecar: &BoundSidecarV1) -> Result<Vec<u8>, BoundError>;
pub fn decode_bound_sidecar_v1(bytes: &[u8], expected_artifact_sha256: [u8; 32], expected_scalar_instruction_sha256: [u8; 32], expected_proposal_instruction_sha256: [u8; 32], expected_toolchain_sha256: [u8; 32]) -> Result<BoundSidecarV1, BoundError>;
pub fn q8_row_interval(row: usize, proposal_center: f32, activation_bf16: &[u16; 5120], sidecar: &BoundSidecarV1) -> Result<OutwardInterval, BoundError>;
```

- [ ] **Step 1: Write RED tests** for magic/version/geometry, 20-Q6-block rows, exactly 19,865,600 payload bytes and 19,865,856 total bytes for 248,320 rows, deterministic SHA, upward FP16 rounding, malformed/truncated rejection, finite-only values, and intervals containing adversarial real-dot errors.
- [ ] **Step 2: Run RED.**

```bash
cargo test -p model-store --test q6_bound bound_sidecar_v1_ -- --nocapture
```

Expected: fail on missing `BoundSidecarV1` APIs.

- [ ] **Step 3: Implement the format and arithmetic.** Use a fixed 256-byte little-endian header followed by a 19,865,600-byte row-major payload `[248320][20][W_f16,D_f16]`. Header offsets are: `0..8` magic `SQ6BND1\0`; `8..12` format version `1`; `12..16` header bytes `256`; `16..20` rows `248320`; `20..24` K `5120`; `24..28` Q6 block values `256`; `28..32` groups per row `20`; `32..36` norms per group `2`; `36..40` element bytes `2`; `40..44` row bytes `80`; `44..48` scalar arithmetic contract `1`; `48..52` Q8 proposal arithmetic contract `1`; `52..56` output-row mapping contract `1`; `56..60` group-layout contract `1`; `60..64` norm encoding `1` (`FP16_UPWARD`); `64..96` source artifact SHA-256; `96..128` payload SHA-256; `128..160` scalar selected-symbol instruction SHA-256; `160..192` Q8 proposal selected-symbol instruction SHA-256; `192..224` SHA-256 of the canonical `hipcc --version` plus `llvm-objdump --version` toolchain identity; and `224..256` zero reserved bytes. Reject any nonzero reserved byte, unknown contract, or mismatched mapping/toolchain/fingerprint. Store upward-rounded group norms and compute every product, sum, radius, and endpoint with an outward primitive (`mul_up`, `add_up`, `next_down_f32`, `next_up_f32`), rejecting overflow/non-finite values.
- [ ] **Step 4: Bind and derive the arithmetic proof.** Row mapping contract 1 is `row index == tokenizer vocabulary token ID`, with 248,320 consecutive rows and exactly 4,200 Q6 bytes per row in the artifact tensor descriptor; Q6 mapping contract 1 is the canonical 210-byte GGML Q6_K logical-K decode already tested in `decode_q6_k_block`; scalar-reference contract 1 is the selected symbol pinned by the parent deterministic raw-Q6 plan's code-object audit. Freeze the scalar center as two ordered F32 multiplies that materialize each weight operand, one FMA per contribution, 160 contributions per lane, then XOR reductions `16,8,4,2,1`. Define `w` as the exact F32 bit pattern produced by those two ordered RN multiplies and build sidecar norms from that value; the subsequent dot dependency depth is 160 FMA roundings plus five reduction adds, `Cscalar=165`.

  Proposal contract 1 is exact and source-independent: split each 256-value Q6 block into eight ordered 32-value Q8_1 groups; choose `d8_f32 = RN_F32(amax/127)`, store `d8=RN_f16(d8_f32)`, choose each signed `q8=clamp(round_ties_away_from_zero(x/d8_f32),-127,127)`, and define reconstructed `a` as the exact real product of FP16 `d8` and signed-I8 `q8`. For wave lane `lane`, define `bq8_offset=4*(lane/16)+(lane%16)/8`, `within=(lane%8)*4`; for pair `p in 0..2`, Q8 group is `block*8+bq8_offset+2*p`, and Q6 logical start is `(bq8_offset+2*p)*32+within`. Pack the four consecutive signed Q6 and signed Q8 values in increasing logical-coordinate order. The audited GPU graph performs two exact signed DOT4 operations, exact I32→F32 conversions, `RN_F32(d8_f16*dot)` via `v_fma_mix_f32(...,-0)`, `RN_F32(result*i8_scale)`, adds the two pair values, multiplies by Q6 FP16 `d6`, accumulates at most three blocks per warp, serially accumulates eight warp partials, then performs the five XOR reductions. The rounded dependency depths are `2+1+1+2+7+5=18`; zero additions are excluded only when the audited instruction operands prove exact zero. Pin launch shape, signed packing order, scale lane mapping, and the selected-symbol digest; any change requires re-derivation.

  Let `P_j` be the finite F32 proposal emitted by that graph. Define `t_jg` only for the proof as the exact real coefficient `(exact FP16 d6) * (exact signed-I8 scale) * (exact signed-Q6 value)`—the proposal does not materialize a rounded per-weight F32 operand. Let `a_g` be the exact real reconstruction above and `x_g` the exact BF16-expanded scalar activation. Define upward norms `W_jg=||w_jg||2`, `D_jg=||w_jg-t_jg||2`, `E_g=||x_g-a_g||2`, `A_g=||a_g||2`, and `X_g=||x_g||2`. Form `Q_j=up(sum_g(up(W_jg*E_g)+up(D_jg*A_g)))`, `Sscalar_j=up(sum_g W_jg*X_g)`, and `Sprop_j=up(sum_g up(W_jg+D_jg)*A_g)`. With `gamma(n)=up(n*2^-24/(1-n*2^-24))`, set `R_j=up(Q_j + up(gamma(165)*Sscalar_j) + up(gamma(18)*Sprop_j))` and emit `[down(P_j-R_j), up(P_j+R_j)]`. Every f64→f32/f16 conversion rounds upward explicitly; signed zero is canonicalized only for norms, subnormal inputs/outputs are preserved, and any overflow, NaN, infinity, or unrepresentable endpoint returns `BoundError` and forces full exact fallback. Tests must fail if an operand graph, depth, artifact digest, row mapping, norm encoding, or selected-symbol digest changes. Do not use a WMMA term.
- [ ] **Step 5: Add property tests** covering subnormals, signed zero, BF16 half ties, equal-logit row-index ties, maximum finite scale, one-ULP boundary inclusion, every header binding, both arithmetic depths, every radius term, and deliberate inward-rounding mutations.
- [ ] **Step 6: Run and commit.**

```bash
cargo test -p model-store --all-targets
cargo fmt --all --check
git add crates/model-store/src/q6_bound.rs crates/model-store/tests/q6_bound.rs
git commit -m "feat(model-store): encode outward q6 correction bounds"
```

### Task 3: Implement tie-aware exact-tile certification offline

**Files:**
- Modify: `crates/model-store/src/q6_bound.rs`
- Modify: `crates/model-store/examples/q6_bound_spike.rs`
- Modify: `crates/model-store/tests/q6_bound.rs`

**Interfaces:**
- Consumes: Q8 proposal centers, outward intervals, 16-row tiles, and a callback that computes requested rows with `raw_q6_scalar_row_f32` and returns BF16 observations.
- Produces this exact API; tile indices are zero-based, sorted, unique, and map to half-open row ranges `[tile*tile_rows, min(rows,(tile+1)*tile_rows))`:

```rust
pub struct ProposalRow { pub center: f32, pub interval: OutwardInterval }
pub struct CorrectionDecision {
    pub proposal_winner: u32,
    pub non_excludable_rows: u32,
    pub exact_tiles: Vec<u32>,
    pub corrected_winner: u32,
    pub fell_back: bool,
}
pub enum CorrectionError { Shape(String), Callback(String), NonFinite(String), Contract(String) }
pub fn certify_with_exact_tiles<F, G>(
    proposals: &[ProposalRow],
    tile_rows: usize,
    max_tiles: usize,
    exact_tile: F,
    exact_all: G,
) -> Result<CorrectionDecision, CorrectionError>
where
    F: FnMut(std::ops::Range<usize>) -> Result<Vec<u16>, CorrectionError>,
    G: FnOnce() -> Result<Vec<u16>, CorrectionError>;
```

  Each callback vector is ascending-row BF16 bits and must exactly match the requested range length; `exact_all` must return exactly `proposals.len()` entries. Empty input, zero tile size/limit, malformed intervals, unexpected callback length, or nonfinite proposal center fails closed. The API must not accept or inspect a precomputed full exact-logit vector except through the explicit `exact_all` fallback callback.

- [ ] **Step 1: Write RED tests** where a lower row tied after BF16 conversion remains eligible, a higher tied row is excludable, intervals touching the winner remain eligible, rows map deterministically to 16-row tiles, duplicate tiles collapse, and `max_tiles + 1` triggers full exact fallback.
- [ ] **Step 2: Run RED.**

```bash
cargo test -p model-store --test q6_bound certified_tile_ -- --nocapture
```

- [ ] **Step 3: Implement candidate-independent certification.** Never use baseline-WMMA logits, a precomputed oracle vector, or `gamma(10240)`. Exact-compute the proposal winner's tile through `exact_tile`, compare every other outward upper endpoint against that exact BF16 winner with lowest-index tie handling, then exact-compute every retained tile. If more than `max_tiles` remain, call `exact_all` once and mark fallback. The caller validates the returned winner against the separately captured oracle only after certification returns.
- [ ] **Step 4: Mutation-test the certificate.** Temporarily use inward rounding and strict equality exclusion; the touching-boundary and tie tests must fail. Revert mutations.
- [ ] **Step 5: Run and commit.**

```bash
cargo test -p model-store --all-targets
git add crates/model-store/src/q6_bound.rs crates/model-store/examples/q6_bound_spike.rs crates/model-store/tests/q6_bound.rs
git commit -m "feat(model-store): certify exact q6 correction tiles"
```

### Task 4: Evaluate all 64 states against the raw scalar oracle

**Files:**
- Modify: `crates/model-store/examples/q6_bound_spike.rs`
- Create: `crates/model-store/tests/fixtures/q6_bound_64_state_expected.json`
- Verify: `target/scalar-correction-study/`

**Interfaces:**
- Consumes: the Task 1 corpus, Task 2 sidecar, Task 3 certificate, and the configured GQH artifact.
- Produces: canonical per-state JSONL and aggregate JSON with Q8 proposal mismatch, bound violations, exact-tile distribution, fallback count, corrected-token equality, and timing.

- [ ] **Step 1: Write RED parser/aggregate tests** requiring 64 unique states, stable case/index sorting, all finite metrics, p50/p95/p99/max nearest-rank tile counts, exact mismatch IDs, and fail-closed digest validation.
- [ ] **Step 2: Run RED.**

```bash
cargo test -p model-store --test q6_bound corpus_report_ -- --nocapture
```

- [ ] **Step 3: Add the bounded CLI** with explicit `--artifact`, `--corpus-manifest`, `--sidecar`, `--max-tiles 16`, `--output`, and `--time-limit-seconds 900`; reject unknown arguments and unsafe output paths.
- [ ] **Step 4: Run the release study once.**

```bash
timeout --foreground 900s cargo run --release -p model-store --example q6_bound_spike -- \
  --artifact /home/deano/models/qwen38-gqh-shaped.gguf \
  --corpus-manifest target/scalar-correction-study/corpus/manifest.json \
  --sidecar target/scalar-correction-study/q6-bound-v1.bin \
  --max-tiles 16 \
  --time-limit-seconds 900 \
  --output target/scalar-correction-study/report.json
```

Acceptance: 64/64 corrected winners equal the raw scalar oracle; zero interval violations over 15,892,480 state/rows; zero silent fallback; maximum exact tiles at most 16; all fallbacks, if any, are full exact computation and explicitly counted.

- [ ] **Step 5: Pin only structural expected data, not host timings, and commit.**

```bash
git add crates/model-store/examples/q6_bound_spike.rs crates/model-store/tests/fixtures/q6_bound_64_state_expected.json
git commit -m "test(model-store): qualify 64-state q6 correction"
```

### Task 5: Final study gate and cleanup decision

**Files:**
- Verify: all Task 1–4 files
- Record: `.superpowers/sdd/2026-08-25-raw-reference-64-state-certified-correction/certified-correction-report.md`

**Interfaces:**
- Consumes: reviewed commits and the 64-state report.
- Produces: a PASS/FAIL study decision; no runtime route.

- [ ] **Step 1: Run complete affected gates.**

```bash
python3 -m unittest discover -s tests -v
cargo test --workspace --all-targets
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
HIP_ARCH=gfx1100 cargo check --workspace --all-targets
cargo fmt --all --check
python3 tools/check-active-docs.py
git diff --check
```

- [ ] **Step 2: Request whole-change review** focused on outward rounding, tie semantics, digest binding, fallback exactness, normal-feature isolation, and absence of WMMA assumptions.
- [ ] **Step 3: Write the evidence report** with exact commit/artifact/corpus/sidecar digests, all 64 decisions, tile distribution, bound violations, fallback behavior, commands, exits, and failures.
- [ ] **Step 4: Stop.** PASS permits a separate runtime-design proposal; FAIL removes the temporary capture route and retains only generally useful offline tests/helpers after review.
