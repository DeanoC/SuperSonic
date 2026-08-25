# Deterministic Raw-Q6 Output-Head Design

**Date:** 2026-08-25

**Status:** Proposed design pending review

## Purpose

Replace the `gfx1201` single-token Q6_K output-head reference with a
deterministic scalar implementation whose arithmetic is fully specified and
whose generated code can be audited. This creates a reference that can support
certified Q8 proposal-and-correction later, while also testing the substantially
faster scalar result as a possible production route.

The replacement is not assumed to preserve the current WMMA logits or tokens.
The existing route truncates dequantized weights to BF16 and uses a WMMA
accumulation whose internal reduction and accuracy are not publicly specified.
The proposed route consumes the raw Q6_K values with documented scalar F32
operations. Promotion therefore requires both deterministic correctness gates
and the versioned six-hour quality suite, not merely a matching first token or
an isolated speed result.

No public CLI option, environment switch, or silent fallback is added. Until
all gates pass, the current production route remains unchanged and the scalar
route is reachable only through focused tests and benchmark-only builds.

## Decision and Alternatives

The recommended reference is raw-Q6 scalar arithmetic with one fixed wave32
reduction per vocabulary row. A feasibility spike produced deterministic
full-vocabulary logits and repeated token streams, used 36 VGPRs without
spills, and measured about 1.77 ms per output-head token. That timing is
promising but is not a headline result: the spike was not compared with WMMA
under the same locked-clock session and the host reported throttling.

Three alternatives were considered:

1. **Raw-Q6 scalar reference (recommended).** It removes BF16 weight
   truncation from the reference and makes the operation graph auditable. It
   was deterministic in the spike and is materially faster than the historical
   WMMA measurement, but changed the cold-chat token stream.
2. **BF16-truncated scalar control.** It preserves the current WMMA operand
   values and is useful for separating operand changes from reduction-order
   changes. It retains an avoidable error term and still cannot guarantee WMMA
   token equivalence because its reduction tree differs.
3. **Retain WMMA as the certifiable reference.** This cannot provide the
   required proof from public AMD documentation: the internal WMMA
   accumulation order, precision, rounding points, and accuracy bound are not
   specified. WMMA remains the production control while this proposal is
   evaluated, not the long-term certified reference.

## Scope

The first implementation is restricted to the validated `gfx1201` output-head
shape:

```text
model       qwen3.8-27b
lhs         BF16, batch=1, m=1, k=5120
rhs         raw GGML Q6_K, qtype=14, n=248320
output      F32 scratch followed by explicit BF16 RNE conversion
architecture gfx1201
```

Every unsupported dtype, shape, quantization, AWQ combination, architecture,
or null required pointer fails explicitly. `gfx1100` continues to use its
existing validated architecture-specific route; it is not a fallback and is
not assumed to share the new numerical contract. Extending the scalar route to
`gfx1100` requires its own generated-code, correctness, performance, and
quality evidence.

This design does not add another model, weight format, multi-sequence path,
FLM route, approximate public mode, runtime sidecar generator, compatibility
alias, or general matrix multiplication implementation.

## Numerical Contract

For output row `j`, Q6 block `b`, and coordinate `i`, define:

```text
d       = exact F16-to-F32 conversion of the block scale
s       = exact signed-I8-to-F32 conversion of the Q6 subscale
q       = decoded signed Q6 integer in [-32, 31]
w       = RN_F32(RN_F32(d * s) * q)
x       = exact BF16-bits-to-F32 conversion of the activation
```

The Q6 bit mapping remains the established project mapping, including 20
blocks, each block's four 64-value subscales per half, and `q - 32` centering.
The two multiplications defining `w` are separate ordinary F32 multiplies;
contraction and reassociation across them are forbidden.

One wave32 owns each output row. Lane `l` evaluates coordinates in this exact
order:

```text
p = +0.0f
for block in 0..20:
    for t in 0..8:
        i = block * 256 + l + 32 * t
        p = FMA_F32(w(i), x(i), p)

for offset in [16, 8, 4, 2, 1]:
    p = ADD_F32(p, shfl_xor(p, offset))
```

Lane zero writes the F32 row result. A separate shared helper converts that
result exactly once with explicit finite F32-to-BF16 round-to-nearest-even bit
semantics. Greedy selection compares the resulting BF16 values and keeps the
lowest vocabulary index on ties, including signed-zero equality.

The contract uses explicit ordinary scalar F32 multiply, FMA, and add
instructions. It does not depend on source-level expressions being compiled
as intended. The implementation must prevent fast-math reassociation and must
not emit `v_fma_mix*`, WMMA, or MFMA instructions. NaN, infinity, overflow, or
an unrecognized floating-point mode fails the affected experimental gate.

For finite non-overflowing values, the fixed lane path contains 160 FMA updates
and the wave tree contributes five adds. A scalar reference bound may therefore
use `gamma(165)`, subject to successful instruction and mode verification.
The coefficient construction is part of the reference definition rather than
an unaccounted runtime approximation.

## Kernel and Runtime Architecture

The HIP implementation belongs with the existing Q6_K kernels and reuses the
established Q6 decoder rather than introducing a second mapping. It adds a
private, versioned FFI entry point rather than renaming a historical symbol or
changing an existing descriptor layout. Its inputs are BF16 activations, raw
Q6_K weights, an F32 output buffer, and an optional contiguous row range for
tile correction.

The F32 output scratch, correction-tile buffers, and eventual bound metadata
are owned by `DecodeEngine` and follow its device lifetime. There is no global
mutable scratch. Prefill, host-logit decode, MTP verification, and non-Q6 paths
remain unchanged unless a later promotion plan explicitly covers them.

The implementation proceeds in two layers:

- a private full-row scalar reference used by tests and benchmark-only route
  selection; and
- after the reference is proven, an optional Q8 proposal path that recomputes
  certified candidate tiles with the same scalar entry point and falls back to
  the complete scalar vocabulary when certification fails.

The Q8 proposal can never return an uncertified token. Runtime telemetry for
that later layer records proposal execution, certification status, exact tile
count, and full-scalar fallback. The public route is promoted only in a
separate reviewed change.

## Generated-Code Contract

Correct source is insufficient. A supported build pins the ROCm/compiler
identity and verifies the generated `gfx1201` code object. The verification
must establish:

- expected scalar `V_MUL_F32`, `V_FMA_F32`, and `V_ADD_F32` operations;
- the fixed five-step shuffle/add reduction;
- no `v_fma_mix*`, WMMA, MFMA, or unexpected contraction/reassociation;
- no spills and a recorded VGPR count;
- FP32 round-to-nearest-even and input/output denormal preservation in the
  code-object descriptor; and
- a recorded code-object digest and stable instruction fingerprint.

A fingerprint or descriptor mismatch is a build/test failure, not permission
to use a different instruction sequence. The spike's `v_fma_mix_f32`
instructions are specifically disallowed because the public instruction entry
does not provide the accuracy contract needed by this design.

## Correctness Gates

Correctness is established before performance measurement.

### CPU and contract tests

A CPU oracle implements the exact coefficient construction, lane traversal,
FMA sequence, reduction tree, BF16 conversion, and lowest-index tie rule. Tests
cover official Q6 vectors, nibble boundaries, signed scales, subnormal and
signed-zero behavior, ties on both sides of the current winner, and all startup
rejections. The BF16-truncated scalar alternative remains a test-only control,
not a second production implementation.

### GPU numerical tests

The private GPU route must pass:

- exact repeated full-row output across fresh processes for all 248,320 logits;
- GPU agreement with the CPU oracle at the contract's declared comparison
  boundary;
- identical argmax and tie behavior between full-row and tiled execution;
- the official deterministic decode/chat vectors;
- two repeated 32-token established workloads; and
- ordinary-versus-MTP token equality under the candidate semantics.

Any nondeterminism, unexplained CPU disagreement, non-finite value, skipped
configured artifact, or token-count discrepancy stops the experiment. A token
difference from legacy WMMA is recorded and reviewed but is not itself a
numerical-correctness failure, because the proposed arithmetic contract is
intentionally different.

### Sixty-four-state study

The existing captured-state analyzer is updated to treat the audited raw-Q6
scalar result as canonical. All 64 chat-template states are evaluated with:

- exact proposal-versus-reference error observations;
- outward-rounded per-row norm bounds;
- tie-aware BF16 interval exclusion;
- exact selected-tile equality with the full scalar reference; and
- exact full-fallback equality.

The bound sidecar is self-describing and bound to the model artifact, Q6
mapping version, reference-contract version, compiler/code-object fingerprint,
grouping, dtype, dimensions, and digest. A mismatch prevents runtime use.
Nearest-rounded norms are diagnostic only; every normative norm, product, sum,
and interval endpoint is rounded outward.

The Q8 path remains a proposal. Its audited operation graph supplies its own
error term. The raw-Q6 reference removes the legacy BF16-weight-truncation
residual from the reference, but proposal coefficient error is still included
where the proposal consumes different values.

## Locked-Clock Performance Gate

The throwaway spike demonstrated feasibility, not an accepted performance
result. The hardened implementation is compared with the retained WMMA route
in the same host session under the benchmark system's `locked` policy.

Each route uses fresh processes, identical artifact and prompt inputs, the same
warmup/repetition policy, and a declared cache state. Run order is balanced to
reduce thermal bias. Records include requested and observed graphics, memory,
and fabric clocks; power cap and performance level; temperature, utilization,
and throttle state; code-object and binary digests; raw samples; and exact
measurement boundaries.

The initial promotion thresholds are:

- scalar output-head median no greater than 2.20 ms per generated token;
- scalar output-head p95 no greater than 2.40 ms per generated token;
- no more than 5% regression from the locked-clock hardened scalar baseline
  established in the same compiler/toolchain series; and
- no clock drift, thermal throttle, cache-state violation, or incomplete
  balanced round.

The same-session WMMA result is always reported. Speedup is published only
when the comparison validator marks the records comparable. An
`uncontrolled-clocks` run may guide development but cannot satisfy this gate
or produce a headline claim.

## Six-Hour Quality and Reproducibility Gate

After the correctness, instruction, 64-state, and short performance gates pass,
run the versioned full benchmark suite for up to six hours. The overnight run
contains three pinned participants:

- retained WMMA SuperSonic as the semantic and performance control;
- hardened raw-Q6 scalar SuperSonic as the candidate; and
- the pinned comparable llama.cpp/GGUF peer.

The two SuperSonic variants are selected through private benchmark builds or a
test harness, never a new public CLI flag or runtime environment switch. Each
participant has a distinct binary and code-object digest. The existing suite
clock, cache, failure, completeness, and publication rules apply unchanged.

Candidate acceptance requires:

- every one of the 16 versioned deterministic quality cases passes, with no
  failed case hidden by an aggregate;
- repeated scalar runs produce identical logits and tokens;
- ordinary and NextN/MTP generation remain token-equal under candidate
  semantics;
- every legacy-versus-candidate token divergence is retained in the result
  bundle and reviewed by case, including the known cold-chat divergence;
- peer comparisons retain all artifact, tokenizer, chat-template, stopping,
  clock, cache, and measurement-boundary qualifiers; and
- the full suite completes within its six-hour policy without throttle or
  invalid telemetry.

The candidate is not required to reproduce legacy WMMA tokens globally. A
deliberate semantics change starts a new quality series: prompts, expected
results, token evidence, and scoring remain versioned, while newly accepted
golden tokens are changed only in the later promotion commit. Performance
prompts without a deterministic answer retain token diffs for inspection but
do not become quality successes merely because their text appears coherent.

Incomplete or uncontrolled overnight evidence remains diagnostic and cannot
approve promotion or publication.

## Promotion and Slim Evolution

Promotion is a separate reviewed change after all evidence is available. Before
removing a superseded implementation, tag the last commit containing the old
`gfx1201` canonical output-head route and push that tag to GitHub. Then replace
the route and remove its superseded head-specific code, tests, private switches,
and documentation together. WMMA helpers still used by other validated
operations remain; this design does not remove unrelated kernels.

Architecture registry dispatch remains explicit: `gfx1201` selects the new
validated scalar reference if promoted, while `gfx1100` selects its own current
validated implementation. Neither branch is a compatibility fallback.

The promotion commit updates the quality-series version and accepted golden
evidence, records the benchmark bundle, and updates public performance claims
only from qualified locked-clock records. If any gate fails, the production
route remains WMMA and the scalar work remains an experimental result rather
than a permanently active second implementation.

## Implementation Stages

1. **Hardened private scalar reference.** Add the CPU oracle, explicit scalar
   HIP operations, private FFI, generated-code audit, and focused full-row and
   token tests. Do not change production routing.
2. **Controlled evaluation.** Run same-session locked-clock WMMA/scalar A/B,
   then update and execute the 64-state raw-reference bound study.
3. **Overnight qualification.** Run the complete six-hour candidate/control/
   peer benchmark and produce a reviewable result bundle.
4. **Promotion decision.** Review quality divergences and reproducibility. If
   accepted, tag the old implementation and make the narrow replacement in a
   separate change; otherwise remove the experimental route.

Likely ownership follows existing crate boundaries: HIP kernels and bridges in
`crates/kernel-ffi`, engine-owned scratch and route state in `crates/runtime`,
offline bound analysis in `crates/model-store`, model/architecture validation
in `crates/core` and `crates/qwen38`, and diagnostic variant orchestration and
records under `benchmarks/` and `tools/`.

## Success Criteria

This design succeeds only when SuperSonic has a deterministic, instruction-
audited raw-Q6 reference; exact CPU/GPU and tiled/full behavior; a valid
64-state correction certificate; a same-session locked-clock performance win;
and a complete six-hour quality result that supports the deliberate numerical
semantics change. Until then, the public product and its documented performance
claims do not change.
