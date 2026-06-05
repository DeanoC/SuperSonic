# Lower-Precision Quantization Track

This track is for quality-aware lower-precision experiments before any Metal
kernel promotion. The benchmark matrix owns stable lane names early so generated
artifacts, docs, and future kernels do not drift.

## Candidate Lanes

| Lane | First model | Target | Artifact source | Runtime status |
|:---|:---|:---|:---|:---|
| `int3` | `qwen3.5-0.8b` | weight-only INT3 probe | AutoRound / SignRoundV2 with `enable_alg_ext` | skipped |
| `int2-4-mixed` | `qwen3.5-0.8b` | adaptive INT2/INT4 mixed-bit probe | AutoRound AutoScheme / SignRoundV2 | skipped |
| `mxfp4` | `qwen3.5-0.8b` | MXFP4 format probe | AutoRound / SignRoundV2 | skipped |
| `int2-4-mixed` | `qwen3.5-35b-a3b` | target-model placeholder | AutoRound AutoScheme / SignRoundV2 | skipped |

These rows are intentionally separate from `SUPPORTED_COMBOS`. A requested
candidate writes a skipped `bench-perf` cell with `quant_artifact` metadata
instead of silently falling through to BF16 or a generic unsupported row.

## Artifact Metadata

Lower-precision artifacts must record:

- `quant_method.profile`
- producer and producer version
- source format and source quant
- average bits per weight
- candidate bit set, such as `["int2", "int4"]`
- mixed-precision assignment payload when the artifact uses per-layer or
  per-projection bit allocation
- calibration corpus, sample count, sequence length, and seed

The benchmark-side `quant_artifact` field mirrors the high-level artifact
identity in skipped/perf cells. The bake manifest remains the source of truth
once an artifact exists.

## Acceptance Gates

Before adding runtime kernels for a candidate:

1. Generate the small-model artifact first.
2. Run single-projection dequant/matvec checks against a BF16 or GGUF oracle.
3. Pass 1-token smoke and 8-token deterministic decode.
4. Pass 128-token drift classification and 512-token divergence checks.
5. Compare memory footprint and tok/s against `q4km` and `q4km-gptq` controls.

If quality passes but tok/s does not improve after unpack/dequant overhead, keep
the lane as a memory-capacity experiment rather than a promoted performance
lane.
