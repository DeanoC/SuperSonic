# Supported Matrix

Which models, quantization lanes, and runtime features are validated on
which GPU architecture. The cells below track *correctness* — see
[docs/performance.md](performance.md) for measured decode throughput
and [docs/feature-compatibility.md](feature-compatibility.md) for the
runtime-feature compatibility grid.

Six backend surfaces are validated or in bring-up today:

- **HIP / `gfx1100`** — AMD Radeon RX 7900 XTX (RDNA 3, 24 GiB)
- **HIP / `gfx1150`** — AMD Radeon 890M iGPU (RDNA 3.5)
- **HIP / `gfx942`** — AMD Instinct MI300X-class (CDNA 3, wave64 bring-up)
- **CUDA / `sm86`** — NVIDIA RTX 3090-class (Ampere)
- **CUDA / `sm90`** — NVIDIA H100 80GB HBM3 (Hopper)
- **Metal / `apple-m4`, `apple-m5-max`** — Apple silicon, BF16 Qwen3.5
  (CLI + `supersonic-serve` on M4; CLI + bughunt gate on M5 Max)

### HIP on `gfx1100`

| Model            | BF16 | INT4 | FP8 runtime | FP8 KV |
|------------------|:----:|:----:|:-----------:|:------:|
| qwen3.5-0.8b     |  ✅  |  ✅  |      ✅     |   ✅   |
| qwen3.5-2b       |  ✅  |  ✅  |      ✅     |   ✅   |
| qwen3.5-4b       |  ✅  |  ✅  |      ✅     |   ✅   |
| qwen3.5-9b       |  ✅  |  ✅  |      ✅     |   ✅   |
| qwen3.6-35b-a3b  |  ❌³ |  ✅³ |      —      |    —   |
| gemma4-e2b       |  ✅  |  ✅  |     ✅²    |   ✅²  |
| gemma4-e4b       |  ✅  |  ✅¹ |     ✅²    |   ✅²  |
| phi4-mini        |  ✅  |  ✅  |      ✅     |   ✅   |

¹ Gemma E4B INT4 needs `--group-size 64` at calibration time (the default
  128 produces gibberish — see fix in `oracle/bake_all.sh`). The published
  release bake is the gs=64 version; consumers fetch it automatically.
² Gemma 4 `--kv-fp8` and `--fp8-runtime` are wired into the single-batch
  persistent decode kernel only — both require `--batch-size=1` and
  cannot combine with `--int4` (the INT4 kernel doesn't yet route the
  FP8 paths). Prefill under either FP8 mode runs per-token through the
  same persistent kernel rather than the BF16 prefill primitive chain.
³ `qwen3.6-35b-a3b` is the Qwen3.6 hybrid linear/full-attention MoE
  (40 layers, 256 experts, top-8 routing, ~3B active per token; HF
  release ships the large decoder weights in FP8). BF16 is intentionally
  unsupported for this model: expanding the FP8-native checkpoint into
  BF16 is a derived debug artifact, not a native model lane. INT4-GPTQ is
  the only HIP lane today: the bake is ~16.9 GiB on-disk and ~21 GiB at
  runtime (KV and scratch). Calibration needs more host RAM than typical 7900 XTX
  rigs carry, so consumers pull the published bake from GitHub
  releases (see [docs/bake-distribution.md](bake-distribution.md));
  producer workflow is unchanged. `--fp8-runtime` and `--kv-fp8` are
  not wired for the MoE family.

### HIP on `gfx1150`

| Model            | BF16 | INT4 | FP8 runtime | FP8 KV |
|------------------|:----:|:----:|:-----------:|:------:|
| qwen3.5-0.8b     |  ✅  |  ✅  |      ✅     |   ✅   |
| qwen3.5-2b       |  ✅  |  ✅  |      ✅     |   ✅   |
| qwen3.5-4b       |  ✅  |  ✅  |      ✅     |   ✅   |
| qwen3.5-9b       |  ✅  |  ✅¹ |      ✅     |   ✅   |
| gemma4-e2b       |  ✅  |  ✅  |      —      |    —   |
| gemma4-e4b       |  ✅  |  ✅² |      —      |    —   |
| phi4-mini        |  ✅  |  ✅  |      ✅     |   ✅   |

¹ GPTQ calibration for 9B INT4 needs ≥24 GiB; consumers pull the released
  bake from GitHub releases. See [docs/bake-distribution.md](bake-distribution.md).
² E4B INT4 uses the published gs=64 bake (see footnote ¹ in the gfx1100
  matrix above); on gfx1150 it decodes at ~280 ms/step.

DFlash speculative decode is available for `qwen3.5-9b` INT4 on HIP —
see [docs/dflash.md](dflash.md).

### HIP on `gfx942`

| Model            | BF16 | INT4 | FP8 runtime | FP8 KV |
|------------------|:----:|:----:|:-----------:|:------:|
| qwen3.5-0.8b     | ✅¹  | ✅²  |     ✅¹⁰    |  ✅¹¹  |
| qwen3.5-2b       | ✅¹  | ✅²  |     ✅¹⁰    |  ✅¹¹  |
| qwen3.5-4b       | ✅¹  | ✅²  |     ✅¹⁰    |  ✅¹¹  |
| qwen3.5-9b       | ✅¹  | ✅²  |     ✅¹⁰    |  ✅¹¹  |
| qwen3.6-35b-a3b  |  ❌⁴ | ✅⁴  |      —      |   —    |
| gemma4-e2b       | ✅³  | ✅⁵  |      —      |   —    |
| gemma4-e4b       | ✅⁶  |  —   |      —      |   —    |
| phi4-mini        | ✅⁷  | ✅⁸  |     ✅⁹     |  ✅¹²  |

¹ CDNA single-sequence decode uses the persistent megakernel by default.
  `--force-replay-decode` remains available as the slower GPU-prefill
  reference path.
² INT4 uses the published GPTQ bake. The PyTorch oracle is BF16-only, so INT4
  bring-up checks exact token agreement and BF16-scale logit drift against
  replayed GPU prefill.
³ Gemma 4 E2B BF16 validates against the PyTorch oracle; Gemma does not yet
  have the Qwen GPU replay validator wired up.
⁴ Qwen3.6 35B A3B currently uses the INT4 GPTQ bake and the host-orchestrated
  HIP chained decode path; performance work is still pending. BF16 is not
  supported because the source checkpoint is FP8-native, so a BF16 bake would
  be an expansion artifact rather than a real model lane.
⁵ Gemma 4 E2B INT4 uses the GPTQ bake and the existing Gemma 4 INT4 primitive
  chain on CDNA; performance tuning is still pending.
⁶ Gemma 4 E4B BF16 uses the existing Gemma 4 HIP kernel path on CDNA;
  performance tuning is still pending.
⁷ Phi-4-mini BF16 uses a correctness-first single-block CDNA fallback in the
  existing Phi-4 HIP kernel path; performance tuning is still pending.
⁸ Phi-4-mini INT4 uses the GPTQ bake and the same correctness-first
  single-block CDNA fallback as BF16.
⁹ Phi-4-mini FP8-runtime uses the FP8-native bake and the same
  correctness-first single-block CDNA fallback as BF16.
¹⁰ Qwen3.5 FP8-runtime uses the published FP8-native bakes and matches the
   existing HIP golden-token outputs with zero GPU replay delta.
¹¹ Qwen3.5 KV-FP8 uses replayed GPU prefill for the single-sequence path and
   matches the existing HIP golden-token outputs with zero GPU replay delta.
¹² Phi-4-mini KV-FP8 uses the correctness-first single-block CDNA fallback and
   matches the PyTorch oracle tokens; logit drift is higher than BF16 but below
   the INT4 lane observed during bring-up.

### CUDA on `sm86`

| Model            | BF16 | INT4 | FP8 runtime | FP8 KV |
|------------------|:----:|:----:|:-----------:|:------:|
| qwen3.5-0.8b     |  ✅  |  ✅  |      ✅     |   ✅   |
| qwen3.5-2b       |  ✅  |  ✅  |      ✅     |   ✅   |
| qwen3.5-4b       |  ✅  |  ✅  |      ✅     |   ✅   |
| qwen3.5-9b       |  ✅¹ |  ✅  |      ✅     |   ✅¹  |
| gemma4-e2b       |  ✅³ |  —   |      —      |    —   |
| phi4-mini        |  ✅² |  ✅² |      ✅²    |   ⏳²  |

¹ `qwen3.5-9b` BF16 and KV-FP8 use the `bakes-v2` BF16 release artifact for
  release-backed downloads. `bakes-v2` also publishes `qwen3.5-9b` INT4 GPTQ
  and FP8-native artifacts; BF16, KV-FP8, INT4 GPTQ, and FP8-runtime
  release-backed paths have all been smoke-tested on `sm86`.
² `phi4-mini` BF16 CUDA is wired and validated on `sm86` with the CPU oracle.
  INT4 uses the downloadable bake and passes the `12/12` reconstructed-bake
  corpus with the kernel-accurate deterministic Python oracle. The live
  PyTorch BF16 oracle remains `10/12`, which layer replay diagnostics attribute
  to PyTorch BF16 accumulation sensitivity. See
  [docs/phi4-cuda-parity.md](phi4-cuda-parity.md). FP8-runtime uses the
  downloadable FP8-native bake and passes the PyTorch oracle on CUDA. FP8-KV
  has descriptor/kernel hooks but still needs CUDA bake/validation work.
³ Gemma 4 CUDA v1 is native `gemma4-e2b` BF16 only and requires
  `--batch-size=1`. `gemma4-e4b`, `--int4`, `--fp8-runtime`, and `--kv-fp8`
  are intentionally not registered for CUDA yet.

CUDA support is validated on NVIDIA `sm86` hardware (RTX 3090-class) with
hand-maintained CUDA sources and no generic fallback backend. Qwen3.5 BF16,
INT4, FP8-runtime, and KV-FP8 lanes are now at parity with the HIP matrix for
0.8B, 2B, and 4B; `gemma4-e2b` BF16 and `phi4-mini` BF16/FP8-runtime have
native CUDA persistent-decode lanes. `phi4-mini` INT4 now matches the kernel-accurate
deterministic corpus oracle; the live PyTorch oracle is advisory for that lane
because BF16 accumulation choices change near-tie generations. The former
Qwen3.5 9B CUDA artifact gap is closed: BF16, INT4 GPTQ, and FP8-native release
bakes are present in `bakes-v2`, and the BF16-dependent 9B KV-FP8 lane has
been smoke-tested with the same release-backed BF16 weights.

CUDA KV-FP8 notes:

- checked against the CPU oracle on the CUDA test machine
- batched `--batch-size 2` uses the real persistent kernel path
- single-sequence decode uses replayed prefill only for `--kv-fp8`
- the BF16 KV sidecar is a bring-up aid for parity-sensitive reads/debugging, not part of normal BF16 CUDA runs
- CUDA caps that sidecar to the most recent 128 KV positions by default for `--kv-fp8`, because full-prefix sidecar reads destabilized long-context parity on `sm86`; opt back into the full-prefix debug sidecar with `SUPERSONIC_DEBUG_ENABLE_CUDA_KV_FP8_BF16_SIDECAR=1`
- normal CUDA BF16 runs do not allocate or use the sidecar
- the sidecar can be disabled for A/B work with `SUPERSONIC_DEBUG_DISABLE_KV_FP8_BF16_SIDECAR=1`

CUDA batched decode is currently validated only for:

- `qwen3.5-4b`
- NVIDIA `sm86`
- `--batch-size 2`

`--gpu-validate` is now part of the checked `sm86` debug surface for `qwen3.5-0.8b` and `qwen3.5-4b`.
It is intentionally slower than normal decode: each step replays the full token history through the validated GPU prefill path and compares the resulting last-token logits against native decode.
For `qwen3.5-4b` on `sm86`, normal BF16 single-sequence decode now uses the
kernel path by default; replayed-prefill decode is legacy debugging behavior
behind `--force-replay-decode`. CUDA `--kv-fp8` single-sequence decode still
uses replayed GPU prefill for correctness.

### CUDA on `sm90`

`sm90` currently reuses the CUDA `sm86` registry geometry and kernel families,
while `kernel-ffi/build.rs` compiles the CUDA sources as native SM90 cubins
from the detected H100 compute capability. This is a compatibility bring-up
lane, not a Hopper-specific retune.

| Model            | BF16 | INT4 | FP8 runtime | FP8 KV |
|------------------|:----:|:----:|:-----------:|:------:|
| qwen3.5-0.8b     |  ✅¹ |  ✅² |      ✅²    |   ✅²  |
| qwen3.5-2b       |  ✅² |  ✅² |      ✅²    |   ✅²  |
| qwen3.5-4b       |  ✅¹ |  ✅² |      ✅²    |   ✅²  |
| qwen3.5-9b       |  ✅² |  ✅² |      ✅²    |   ✅²  |
| gemma4-e2b       |  ✅² |  —   |      —      |    —   |
| phi4-mini        |  ✅² |  ✅² |      ✅²    |   ⏳²  |
| llama3.1-8b      |  ✅² |  —   |      —      |   ✅²  |

¹ Smoke-tested and benchmarked on an NVIDIA H100 80GB HBM3 on 2026-05-07 with
  CUDA 13.0 / driver 580.126.09, using the `bakes-v2` BF16 release artifacts.
  See [docs/detailed_performance.md](detailed_performance.md#cuda--sm90-nvidia-h100-80gb-hbm3).
² Registered on `sm90` by reusing the validated CUDA `sm86` entry for this
  model/lane. Dedicated H100 parity coverage is pending; if you need strict
  certification for one of these inherited lanes, run the matching `tests/sm86`
  validation script on the H100 and record the result here.

### Metal on `apple-m4` / `apple-m5-max`

| Model            | BF16 | INT4 | FP8 runtime | FP8 KV |
|------------------|:----:|:----:|:-----------:|:------:|
| qwen3.5-0.8b     |  ✅  |  —   |      —      |    —   |
| qwen3.5-2b       |  ✅  |  —   |      —      |    —   |
| qwen3.5-4b       |  ✅¹ |  —   |      —      |    —   |
| qwen3.5-9b       |  ✅¹ |  —   |      —      |    —   |
| qwen3-30b-a3b    |  —   |  ✅¹ |      —      |    —   |
| qwen3.6-35b-a3b  |  —   |  ✅¹ |      —      |    —   |
| gemma4-e2b       |  ✅¹ |  ✅¹ |      —      |    —   |
| gemma4-e4b       |  ✅¹ |  ✅¹ |      —      |    —   |
| phi4-mini        |  ✅¹ |  ✅¹ |     ✅¹     |    —   |

Metal v2 is a single supported surface:

- BF16 single-sequence decode for `qwen3.5-0.8b`, `qwen3.5-2b`, `qwen3.5-4b`,
  and `qwen3.5-9b`
- Apple M5 Max is validated on the `qwen3.5-0.8b` CLI + bughunt gate path;
  it uses the same Metal v2 path as Apple M4.
- `qwen3.5-4b` and `qwen3.5-9b` are validated on Apple M5 Max with one-token
  `--validate --oracle-device cpu` smokes through Metal v2 incremental decode.
- `qwen3-30b-a3b` INT4 has an Apple M5 Max registry row and uses a
  correctness-first chained Metal fallback for decode-layer attention, MoE
  routing, expert matmul, KV updates, and the final INT4 lm-head. The
  persistent Qwen3-MoE megakernel remains HIP-only.
- `qwen3.6-35b-a3b` INT4 is supported on Apple M5 Max through the
  host-orchestrated chained decode route with Metal fallbacks for BF16
  full-attention stages 1-5, linear-attention stages 1-5, FFN stages 1-5,
  the final lm-head helpers, MTP pre-fusion helper, and INT4 sidecars for the
  projection/expert matvecs. The validation smoke is:
  `SUPERSONIC_TEST_MODEL_DIR=/path/to/Qwen3.6-35B-A3B cargo test --release -p runner --test qwen36_moe_metal_smoke -- --ignored --nocapture`.
  Persistent decode, FP8-runtime, KV-FP8, and speculative decode remain
  unsupported on this Metal path.
- `phi4-mini` has an Apple M5 Max registry row and uses a component Metal decode
  path assembled from existing RMSNorm, GEMV, INT4 runtime-dequant matvec,
  FP8 runtime-dequant host fallback matvec, RoPE, attention, SwiGLU, and
  residual kernels.
- `gemma4-e2b` and `gemma4-e4b` have Apple M5 Max registry rows. BF16 uses a
  runner-level Metal component path for RMSNorm, BF16 matvec, RoPE, attention,
  residual, MLP, PLE, K/V append, and sliding-window cache slicing. INT4 uses
  the same component decode route with Metal INT4 matvec fallbacks.
- Apple M4 remains limited to the smaller Metal validation lane.
- both the `supersonic` CLI and `supersonic-serve` HTTP server work; `/v1/completions`
  and `/v1/chat/completions` (streaming and non-streaming) are exercised end-to-end
- decode is implemented as **incremental per-token decode**: each generated token runs
  a single length-1 forward pass (O(N) per step). Conv and recurrent state are carried
  across tokens in persistent GPU buffers; KV cache grows with the sequence
- INT4 GPTQ kernel coverage includes the supported `qwen3.6-35b-a3b`,
  `qwen3-30b-a3b`, `gemma4-e2b`, `gemma4-e4b`, and `phi4-mini` Apple M5 Max
  smokes.
- `--fp8-runtime`, `--kv-fp8`, `--batch-size > 1`, `--force-kernel-decode`,
  and `--force-component-decode` are all rejected at startup

¹ Apple M5 Max only.

Metal is not yet at mode-level HIP parity: persistent megakernel decode,
Qwen3.6 FP8-runtime, Qwen3.6 KV-FP8, speculative decode, Metal VMM, and
batching remain unsupported.

The Apple M5 Max large-model smoke is:

```bash
SUPERSONIC_TEST_MODEL_ROOT=/path/to/supersonic-metal-models \
  cargo test --release -p runner --test metal_large_model_smoke -- --ignored --nocapture
```
