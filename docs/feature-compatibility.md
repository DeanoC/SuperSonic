# Feature Compatibility

Compatibility tracker for SuperSonic's runtime features: which combinations
of feature × model × architecture are validated, which are mutually
exclusive, and which use cases each combination targets.

This doc tracks **correctness**. For measured speedups see
[performance.md § Runtime feature impact](performance.md#runtime-feature-impact).
For the model × quant × arch baseline see
[supported-matrix.md](supported-matrix.md).

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

Flags: `--kv-fp8`, `--kv-fp8-sidecar-window <N>` (Qwen3.6-MoE only).

Support:

| Model            | gfx1100 | gfx1150 | gfx942 | sm86  | apple-m4 |
|------------------|:-------:|:-------:|:------:|:-----:|:--------:|
| qwen3.5-0.8b     |    ✅   |   ✅    |  ✅¹  |  ✅   |    —     |
| qwen3.5-2b       |    ✅   |   ✅    |  ✅¹  |  ✅   |    —     |
| qwen3.5-4b       |    ✅   |   ✅    |  ✅¹  |  ✅   |    —     |
| qwen3.5-9b       |    ✅   |   ✅    |  ✅¹  |  ✅   |    —     |
| qwen3.6-35b-a3b  |    —    |    —    |   —   |   —   |    —     |
| gemma4-e2b       |   ✅²   |    —    |   —   |   —   |    —     |
| gemma4-e4b       |   ✅²   |    —    |   —   |   —   |    —     |
| phi4-mini        |    ✅   |   ✅    |  ✅²  |  ✅   |    —     |
| llama3.1-8b      |    —    |    —    |   —   |  ✅³  |    —     |

¹ gfx942 KV-FP8 uses replayed GPU prefill for the single-sequence path.
² Gemma 4 KV-FP8 requires `--batch-size 1`, cannot combine with `--int4`.
   Phi-4-mini gfx942 KV-FP8 uses the correctness-first single-block fallback.
³ Llama 3.1 8B KV-FP8 only validated alongside `--int8` and certified-KV.

### 3. VMM (virtual KV cache)

What & why. KV cache backed by virtual memory with on-demand resident
mapping. Eviction-to-host + restore lets a workload exceed nominal
VRAM by paging cold KV out. Currently Qwen3.6-MoE only (its 40 layers
+ MTP make KV a dominant footprint).

Flags: `--virtual-kv` (default ON for Qwen3.6-MoE on HIP per
[lowlevel-memory.md](lowlevel-memory.md)).

Support:

| Model            | gfx1100 | gfx1150 | gfx942 | sm86  |
|------------------|:-------:|:-------:|:------:|:-----:|
| qwen3.6-35b-a3b  |    ✅   |    —    |   —    |   —   |

Other models do not enable VMM today; the dense KV allocator is
sufficient.

### 4. SpecPrefill

What & why. Speculator-driven sparse target prefill for long-prompt
TTFT. See [specprefill.md](specprefill.md) for the user-facing
summary and [research/2026-05-03-specprefill-feasibility.md](research/2026-05-03-specprefill-feasibility.md)
for the design.

Flags: `--specprefill-draft-dir <path>` plus tuning flags (see
specprefill.md).

Support:

| Target × Draft                 | gfx1100 | gfx1150 | gfx942 | sm86 |
|--------------------------------|:-------:|:-------:|:------:|:----:|
| qwen3.5-9b BF16 + qwen3.5-0.8b |    ✅   |   TBM   |  TBM   |  ❌¹ |

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

| Target          | gfx1100 | gfx1150 | gfx942 | sm86 |
|-----------------|:-------:|:-------:|:------:|:----:|
| qwen3.5-9b INT4 |    ✅   |   ✅    |  TBM   |  ❌  |

CUDA support is not currently planned; the B-block fused verify is
HIP-megakernel-specific.

### 6. MoE expert prefetch

What & why. Asynchronous prefetch of MoE expert weights from the rolling
admission window so per-token routed expert dispatch doesn't stall on
weight load. Qwen3.6-MoE only.

Flags: governed by `--qwen36-moe-prefetch-policy <name>` and a few
`--qwen36-moe-*` tuning flags. See
[qwen36-moe-plan.md](qwen36-moe-plan.md).

Support:

| Model            | gfx1100 | gfx1150 | gfx942 | sm86 |
|------------------|:-------:|:-------:|:------:|:----:|
| qwen3.6-35b-a3b  |    ✅   |    —    |   —    |   —  |

### 7. Certified KV (Llama 3.1)

What & why. KV provenance and content certification for Llama 3.1
INT8 on CUDA. Used in retrieval / safety-critical contexts where the
KV cache integrity matters.

Flags: `--certified-kv`, `--certified-kv-shadow-validate`. Requires
`--int8` and Llama 3.1 family. See
[certified-kv-audit-map.md](certified-kv-audit-map.md).

Support:

| Model        | gfx1100 | gfx1150 | gfx942 | sm86 |
|--------------|:-------:|:-------:|:------:|:----:|
| llama3.1-8b  |    —    |    —    |   —    |  ✅  |

CUDA-only; the BF16 step-copy fallback added in PR #177 unblocks the
non-certified BF16 component decode on HIP, but certified mode itself
is still CUDA-specific.

## Feature × feature compatibility

A ✅ means the two flags can be combined; ❌ means the CLI rejects the
combo (or one feature implicitly requires the other to be off).

|                  | Quant | KV-FP8 | VMM | SpecPrefill | DFlash | MoE prefetch | Certified KV |
|------------------|:-----:|:------:|:---:|:-----------:|:------:|:------------:|:------------:|
| **Quant**        |   —   |   ✅¹  | ✅  |     ✅      |   ✅   |     ✅       |     ✅²      |
| **KV-FP8**       |  ✅¹  |   —    | ✅³ |     TBM     |   ❌   |     ✅       |     ✅       |
| **VMM**          |  ✅   |   ✅³  |  —  |     —⁴      |   —⁴   |     ✅       |     —⁴       |
| **SpecPrefill**  |  ✅   |   TBM  | —⁴  |      —      |   ❌   |     —⁴       |     —⁴       |
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

## Picker recipes — "I want to ..."

### ... reduce time-to-first-token on a long Qwen3.5-9B prompt (HIP)

Use SpecPrefill at default keep ratio. The 0.8B draft amortizes
selection in ~700 ms; target then prefills only ~50% of the prompt.

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

Use INT4 GPTQ + VMM (default ON for this model on HIP). KV-FP8
optional for additional KV headroom on long contexts.

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
[performance.md § Runtime feature impact](performance.md#runtime-feature-impact).
