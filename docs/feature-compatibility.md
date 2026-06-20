# Feature Compatibility

Compatibility tracker for SuperSonic's runtime features: which combinations
of feature × model × architecture are validated, which are mutually
exclusive, and which use cases each combination targets.

This doc tracks **correctness**. For measured speedups see
[detailed_performance.md § Runtime feature impact](detailed_performance.md#runtime-feature-impact).
For the model × quant × arch baseline see
[supported-matrix.md](supported-matrix.md).

CUDA `sm90` currently follows the `sm86` CUDA feature surface: H100-class
devices compile native SM90 CUDA kernels but reuse the validated `sm86`
registry geometry and feature gates. Dedicated H100 validation is tracked in
[supported-matrix.md § CUDA on sm90](supported-matrix.md#cuda-on-sm90) and
[detailed_performance.md § CUDA — sm90](detailed_performance.md#cuda--sm90-nvidia-h100-80gb-hbm3).

Metal `apple-m5-max` has a broader model/quant coverage surface than Apple M4
now, but it is still a component/chained decode surface. Qwen3.6-35B-A3B INT4,
Qwen3-30B-A3B INT4, Gemma 4 BF16/INT4, and Phi-4 mini BF16/INT4/FP8-runtime
coverage lives in [supported-matrix.md](supported-matrix.md#metal-on-apple-m4--apple-m5-max).
The advanced runtime features below remain unsupported on Metal unless a row
explicitly says otherwise: Qwen3.6 persistent decode, KV-FP8, VMM,
SpecPrefill/DFlash-style speculative paths, MoE prefetch, and batching are not
part of the Apple M5 Max feature surface today.

HIP `gfx1201` is a new RDNA 4 bring-up lane. It has first-class registry/build
support, RDNA4 BF16/i8 WMMA kernel coverage, and Qwen3.6 27B Q4KM-GPTQ/DFlash
smokes; broader feature cells stay TBM until the `tests/gfx1201` matrix
promotes them with per-model validation.

## How to read

- **✅** = validated end-to-end (parity test or oracle agreement).
- **❌** = explicitly unsupported (CLI guard rejects, or kernel returns
  a clear "not implemented" error).
- **—** = combination doesn't exist (e.g. "FP8 weights for a model
  family that has no FP8 bake").
- **TBM** = to be measured / to be validated.
- Footnotes capture caveats (memory bound, requires specific quant, etc.).

## Runtime features

### 1. Weight quantization

What & why. The base axis: which numeric format the weight matmuls
consume at runtime. BF16 is the reference; INT4 (GPTQ) and FP8 reduce
weight VRAM and (often) compute time at a quality cost. Q4KM is a
GGUF-style INT4 packing used by the CUDA reference path.

Flags: `--int4`, `--fp8-runtime`, `--q4km`, `--q4km-gptq`.

Support per (model, arch): see
[supported-matrix.md](supported-matrix.md). The runtime features
below depend on this baseline being supported first.

### 2. KV-FP8

What & why. KV cache stored in FP8 E4M3 (1 byte) instead of BF16 (2
bytes), halving KV VRAM. Optional sidecar window keeps the most-recent
N tokens in BF16 for higher decode quality. Used when context length
× layers × heads makes the KV cache the binding VRAM constraint.

Flags: `--kv-fp8`. The BF16 sidecar window is enabled by default for
Qwen3.6-MoE; window size can be overridden via env var
`SUPERSONIC_DEBUG_KV_FP8_BF16_SIDECAR_WINDOW=N` (debug knob, not a
stable CLI surface).

Support:

The `sm86` CUDA column also applies to `sm90` unless a per-arch note says
otherwise; the current H100 path is an inherited CUDA compatibility lane.

| Model            | gfx1100 | gfx1150 | gfx1201 | gfx942 | sm86  | apple-m4 |
|------------------|:-------:|:-------:|:-------:|:------:|:-----:|:--------:|
| qwen3.5-0.8b     |    ✅   |   ✅    |   TBM   |  ✅¹  |  ✅   |    —     |
| qwen3.5-2b       |    ✅   |   ✅    |   TBM   |  ✅¹  |  ✅   |    —     |
| qwen3.5-4b       |    ✅   |   ✅    |   TBM   |  ✅¹  |  ✅   |    —     |
| qwen3.5-9b       |    ✅   |   ✅    |   TBM   |  ✅¹  |  ✅   |    —     |
| qwen3.6-35b-a3b  |   ✅⁴   |    —    |    —    |   —   |   —   |    —     |
| gemma4-e2b       |   ✅²   |    —    |    —    |   —   |   —   |    —     |
| gemma4-e4b       |   ✅²   |    —    |    —    |   —   |   —   |    —     |
| phi4-mini        |    ✅   |   ✅    |    —    |  ✅²  |  ✅   |    —     |
| llama3.1-8b      |    —    |    —    |    —    |   —   |  ✅³  |    —     |

¹ gfx942 KV-FP8 uses replayed GPU prefill for the single-sequence path.
² Gemma 4 KV-FP8 requires `--batch-size 1`, cannot combine with `--int4`.
   Phi-4-mini gfx942 KV-FP8 uses the correctness-first single-block fallback.
³ Llama 3.1 8B KV-FP8 only validated alongside `--int8` and certified-KV.
⁴ qwen3.6-35b-a3b KV-FP8 requires --int4 (the only quant lane shipped
  for this model). Sidecar window is configured via env var
  `SUPERSONIC_DEBUG_KV_FP8_BF16_SIDECAR_WINDOW=N`, NOT a CLI flag —
  the earlier "--kv-fp8-sidecar-window" mention in this doc was wrong.
  Default sidecar enables full BF16 coverage for the resident KV slice.

### 3. VMM (virtual KV cache)

What & why. KV cache backed by virtual memory with on-demand resident
mapping. Eviction-to-host + restore lets a workload exceed nominal
VRAM by paging cold KV out. Currently Qwen3.6-MoE only (its 40 layers
+ MTP make KV a dominant footprint).

Flags: `--virtual-kv` (default ON for Qwen3.6-MoE on HIP per
[lowlevel-memory.md](lowlevel-memory.md)).

Support:

The CUDA `sm86` column also applies to `sm90` for VMM: no VMM surface is
registered for CUDA dense/Qwen3.6-MoE today.

| Model            | gfx1100 | gfx1150 | gfx1201 | gfx942 | sm86  |
|------------------|:-------:|:-------:|:-------:|:------:|:-----:|
| qwen3.6-35b-a3b  |    ✅   |    —    |    —    |   —    |   —   |

Other models do not enable VMM today; the dense KV allocator is
sufficient.

### 4. SpecPrefill

What & why. Speculator-driven sparse target prefill for long-prompt
TTFT. See [specprefill.md](specprefill.md) for the user-facing
summary and [research/2026-05-03-specprefill-feasibility.md](research/2026-05-03-specprefill-feasibility.md)
for the design.

Flags: `--specprefill-draft-dir <path>` plus tuning flags (see
specprefill.md).

**Performance note:** As of Phase D (2026-05-04), SpecPrefill defaults
to cosine-similarity scoring (`--specprefill-algorithm cosine`) plus
drafter early-exit (the drafter stops after the scoring layer instead
of running its full model), and the speedup **compounds with prompt
length** on gfx1100 (Qwen3.5-9B BF16, `keep_ratio=0.50`):
**2.07× faster than dense at 1.3k tokens, 2.63× at 4k, 3.36× at 8k**
(2.39s / 11.16s / 36.94s with SpecPrefill vs 4.94s / 29.37s / 124.22s
dense — saving ~87 seconds on a single 8k-prompt prefill). The legacy
`lookahead` algorithm remains available but is slower than dense at
short prompts and only pulls ahead at ≥4k tokens (still beaten by
`cosine` at every measured length) — kept for parity-test coverage
and future research, not recommended for production. See
[specprefill.md § Performance](specprefill.md#performance) for the
full prompt-length sweep.

Support:

The CUDA `sm86` rejection also applies to `sm90`.

| Target × Draft                 | gfx1100 | gfx1150 | gfx1201 | gfx942 | sm86 |
|--------------------------------|:-------:|:-------:|:-------:|:------:|:----:|
| qwen3.5-9b BF16 + qwen3.5-0.8b |    ✅   |   TBM   |   TBM   |  TBM   |  ❌¹ |

¹ CUDA bridge returns "not implemented" for the look-ahead and
RoPE-indirect kernels. Validation rejects upfront.

### 5. DFlash speculative decode

What & why. Single-model speculative decode using a small "DFlash"
draft head trained per-target. Runs the megakernel verify path with
B-block batched candidates. Targets the steady-state decode rate, not
TTFT.

Flags: `--dflash`, `--dflash-draft-dir <path>`, `--dflash-block <N>`.

See [dflash.md](dflash.md) for the design and the M3/M4 milestones.

Support:

The CUDA `sm86` rejection also applies to `sm90`.

| Target                                      | gfx1100 | gfx1150 | gfx1201 | gfx942 | sm86 |
|---------------------------------------------|:-------:|:-------:|:-------:|:------:|:----:|
| qwen3.5-9b INT4                             |    ✅   |   ✅    |   TBM   |  TBM   |  ❌  |
| qwen3.6-27b Q4KM-GPTQ + Lucebox q8_0 draft  |    ✅¹  |    —    |    ✅¹   |   —    |  ❌  |

CUDA support is not currently planned; the B-block fused verify is
HIP-megakernel-specific.

¹ Qwen3.6 27B DFlash is the Lucebox comparison lane. The `gfx1201` cell is a
  starter RDNA4 smoke, not a completed sustained-decode tuning result.

### 6. MoE expert prefetch

What & why. Asynchronous prefetch of MoE expert weights from the rolling
admission window so per-token routed expert dispatch doesn't stall on
weight load. Qwen3.6-MoE only.

Flags: governed by `--qwen36-moe-prefetch-policy <name>` and a few
`--qwen36-moe-*` tuning flags. See
[plans/qwen36-moe-plan.md](plans/qwen36-moe-plan.md).

Support:

The CUDA `sm86` column also applies to `sm90`: no CUDA MoE prefetch lane is
registered today.

| Model            | gfx1100 | gfx1150 | gfx1201 | gfx942 | sm86 |
|------------------|:-------:|:-------:|:-------:|:------:|:----:|
| qwen3.6-35b-a3b  |    ✅   |    —    |    —    |   —    |   —  |

### 7. Certified KV (Llama 3.1)

What & why. KV provenance and content certification for Llama 3.1
INT8 on CUDA. Used in retrieval / safety-critical contexts where the
KV cache integrity matters.

Flags: `--certified-kv`, `--certified-kv-shadow-validate`. Requires
`--int8` and Llama 3.1 family. See
[certified-kv-audit-map.md](certified-kv-audit-map.md).

Support:

The CUDA `sm86` column also applies to `sm90`; H100 inherits the same
certified-KV feature gates pending dedicated quality/performance runs.

| Model        | gfx1100 | gfx1150 | gfx1201 | gfx942 | sm86 |
|--------------|:-------:|:-------:|:-------:|:------:|:----:|
| llama3.1-8b  |    —    |    —    |    —    |   —    |  ✅  |

CUDA-only; the BF16 step-copy fallback added in PR #177 unblocks the
non-certified BF16 component decode on HIP, but certified mode itself
is still CUDA-specific.

## Feature × feature compatibility

A ✅ means the two flags can be combined; ❌ means the CLI rejects the
combo (or one feature implicitly requires the other to be off).

|                  | Quant | KV-FP8 | VMM | SpecPrefill | DFlash | MoE prefetch | Certified KV |
|------------------|:-----:|:------:|:---:|:-----------:|:------:|:------------:|:------------:|
| **Quant**        |   —   |   ✅¹  | ✅  |     ✅      |   ✅   |     ✅       |     ✅²      |
| **KV-FP8**       |  ✅¹  |   —    | ✅³ |     ✅⁵     |   ❌   |     ✅       |     ✅       |
| **VMM**          |  ✅   |   ✅³  |  —  |     —⁴      |   —⁴   |     ✅       |     —⁴       |
| **SpecPrefill**  |  ✅   |   ✅⁵  | —⁴  |      —      |   ❌   |     —⁴       |     —⁴       |
| **DFlash**       |  ✅   |   ❌   | —⁴  |     ❌      |   —    |     —⁴       |     —⁴       |
| **MoE prefetch** |  ✅   |   ✅   | ✅  |     —⁴      |   —⁴   |      —       |     —⁴       |
| **Certified KV** |  ✅²  |   ✅   | —⁴  |     —⁴      |   —⁴   |     —⁴       |      —       |

¹ KV-FP8 + INT4: see per-model footnotes in §KV-FP8.
² Certified KV requires `--int8` and Llama 3.1.
³ VMM and KV-FP8 are independently configured for Qwen3.6-MoE; the
  sidecar window applies to the resident slice.
⁴ Dash means "no validated combo exists today" — the underlying
  features apply to disjoint model families (e.g. SpecPrefill is
  Qwen3.5-9B; MoE prefetch is Qwen3.6-MoE).
⁵ SpecPrefill + KV-FP8 is supported on the default `cosine` algorithm
  (since 2026-05-04). The first prefill stays sparse via `prefill_kept`
  so the SpecPrefill TTFT win is preserved; each decode step then runs
  a dense replay-prefill (`rebuild_prefill_state`) on the full
  unsparsified history — the same fallback plain `--kv-fp8` already
  uses on Qwen3.5-9B because component decode lacks an FP8 attention
  path. Per-token decode is bounded by replay-prefill cost. The
  legacy `--specprefill-algorithm lookahead` + `--kv-fp8` combo is
  still rejected upfront because the speculator's drafter decode
  has no replay-prefill workaround.

## Picker recipes — "I want to ..."

### ... validate the SpecPrefill correctness chain on Qwen3.5-9B (HIP)

The default `cosine` algorithm + drafter early-exit runs 2.07× faster
than dense prefill on gfx1100; the legacy `lookahead` algorithm remains
slower than dense (see
[specprefill.md § Performance](specprefill.md#performance)). For
production TTFT use the default; pass `--specprefill-algorithm
lookahead` only if you're actively researching the legacy path.

```bash
supersonic --backend hip --model qwen3.5-9b --model-dir /path/to/9B \
    --specprefill-draft-dir /path/to/0.8B \
    --prompt "<long prompt>" --max-new-tokens 32
```

### ... maximize tokens/sec on Qwen3.5-9B greedy decode (HIP)

Use DFlash. INT4 target, DFlash draft head, B=3 default.

```bash
supersonic --backend hip --model qwen3.5-9b --model-dir /path/to/9B \
    --int4 --dflash --dflash-draft-dir /path/to/dflash-draft \
    --prompt "..." --max-new-tokens 64
```

### ... fit Qwen3.6-35B-A3B in 24 GiB on gfx1100

Use INT4 GPTQ + VMM (default ON for this model on HIP). For long
contexts, add `--kv-fp8` for additional KV headroom (validated
2026-05-03; ~1% step-time overhead, see
[detailed_performance.md § Runtime feature impact](detailed_performance.md#runtime-feature-impact)).

```bash
supersonic --backend hip --model qwen3.6-35b-a3b \
    --model-dir /path/to/35B-A3B \
    --int4 \
    --prompt "..." --max-new-tokens 32
```

### ... run a long-context retrieval QA with Llama 3.1 8B (CUDA)

Use INT8 + certified-KV.

```bash
supersonic --backend cuda --model llama3.1-8b --model-dir /path/to/Llama-3.1-8B \
    --int8 --certified-kv \
    --prompt "..." --max-new-tokens 64
```

### ... benchmark steady-state decode on Qwen3.5-0.8B (HIP)

No runtime feature flags needed — the persistent megakernel default
path is the fastest.

```bash
supersonic --backend hip --model qwen3.5-0.8b --model-dir /path/to/0.8B \
    --prompt "Hello, world" --max-new-tokens 32
```

## Where the perf numbers live

This doc is correctness-only. For measured impact (ms/step, % TTFT,
VRAM delta) see
[detailed_performance.md § Runtime feature impact](detailed_performance.md#runtime-feature-impact).
