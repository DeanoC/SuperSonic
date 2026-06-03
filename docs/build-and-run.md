# Build and Run

Per-backend build commands and validated `supersonic` invocations for
each (model, GPU arch) combination. The
[Supported Matrix](supported-matrix.md) lists which combinations are
validated; this doc shows how to run them.

For runtime-feature flags (DFlash, SpecPrefill, KV-FP8, MoE prefetch,
VMM, etc.) see [feature-compatibility.md](feature-compatibility.md).

## Quick Start

```bash
# Build with the backend(s) you want compiled in.
# Omit SUPERSONIC_BACKENDS to build the default configured backend set.
SUPERSONIC_BACKENDS=cuda cargo build --release

# Run (auto-bakes weights on first run)
SUPERSONIC_BACKENDS=cuda cargo run --release --bin supersonic -- \
  --backend cuda \
  --model qwen3.5-0.8b \
  --model-dir /path/to/Qwen3.5-0.8B \
  --prompt "Hello, world" \
  --max-new-tokens 8
```

On first run, SuperSonic bakes the HuggingFace safetensors into an optimized format at `{model_dir}/.supersonic/v1/`. Subsequent runs load from this baked format for faster startup.

If a local bake is missing and one is available in the repo's GitHub releases,
SuperSonic can download that package instead of rebuilding it locally. Pass
`--no-download` to disable network fetches. See [bake-distribution.md](bake-distribution.md)
for the producer workflow and release layout.

## Producing and Publishing Bakes

On a machine with enough VRAM/RAM for GPTQ calibration, `oracle/bake_all.sh`
can bake and optionally upload every configured model in one pass. Unset model
directory env vars are skipped, so this is the reference "big producer box"
command:

```bash
pip install -r oracle/requirements-upload.txt
gh auth login

QWEN_0_8B_DIR=/models/Qwen3.5-0.8B \
QWEN_2B_DIR=/models/Qwen3.5-2B \
QWEN_4B_DIR=/models/Qwen3.5-4B \
QWEN_9B_DIR=/models/Qwen3.5-9B \
GEMMA_E2B_DIR=/models/gemma-4-E2B \
GEMMA_E4B_DIR=/models/gemma-4-E4B \
./oracle/bake_all.sh --upload
```

By default this produces INT4-GPTQ bakes for every configured model. Add
`--bf16` to also publish Qwen BF16 bakes, `--fp8-native` for FP8-native
bakes (Qwen only; source checkpoint must ship FP8 tensors), `--force` to
rebuild, or drop `--upload` to keep the output local.

**9B INT4 note:** `qwen3.5-9b` GPTQ calibration loads the full BF16 model
(~18 GiB) into GPU memory, so it OOMs on ≤16 GiB cards (including gfx1150).
Run it on a box with ≥24 GiB GPU RAM — the small-VRAM consumer then pulls
the resulting bake from the release. Leave `QWEN_9B_DIR` unset to skip.

## CUDA

### Build requirements

- NVIDIA driver + CUDA runtime/toolkit usable from the build machine
- Rust toolchain able to build this repo
- Python 3 with `torch` and `transformers` for oracle validation
- local model weights for `Qwen3.5-0.8B` and/or `Qwen3.5-4B`

### Validated commands

These are the checked CUDA `sm86` commands:

```bash
# Qwen3.5-0.8B
SUPERSONIC_BACKENDS=cuda TIMEOUT=600 ./tests/sm86/run.sh /path/to/Qwen3.5-0.8B

# Qwen3.5-0.8B long-context CPU-oracle check
SUPERSONIC_BACKENDS=cuda TIMEOUT=900 ./tests/sm86/run_long.sh /path/to/Qwen3.5-0.8B

# Qwen3.5-4B
SUPERSONIC_BACKENDS=cuda TIMEOUT=900 CORPUS_TIMEOUT=600 ./tests/sm86/run_4b.sh /path/to/Qwen3.5-4B

# Qwen3.5-4B long-context CPU-oracle check
SUPERSONIC_BACKENDS=cuda TIMEOUT=1200 CORPUS_TIMEOUT=1200 ./tests/sm86/run_4b_long.sh /path/to/Qwen3.5-4B

# Qwen3.5-4B batched decode
SUPERSONIC_BACKENDS=cuda TIMEOUT=900 CORPUS_TIMEOUT=600 ./tests/sm86/run_batch.sh /path/to/Qwen3.5-4B

# Phi-4-mini FP8-runtime release bake + PyTorch oracle
HF_HOME=/dev/shm/hf_home SUPERSONIC_BACKENDS=cuda ./target/release/supersonic \
  --backend cuda \
  --model phi4-mini \
  --model-dir /dev/shm/Phi-4-mini \
  --prompt "Hello, world" \
  --max-new-tokens 8 \
  --fp8-runtime \
  --validate

# Llama 3.1 8B INT8 component decode
SUPERSONIC_BACKENDS=cuda ./target/release/supersonic \
  --backend cuda \
  --model llama3.1-8b \
  --model-dir /path/to/Meta-Llama-3.1-8B \
  --prompt "Hello" \
  --max-new-tokens 32 \
  --int8

# Llama 3.1 8B arxiv_v1 retrieval smoke QA
CONTEXTS='4096' SUBTASKS='niah_single niah_multikey niah_multiquery' \
  SAMPLES=1 CONFIG=both TIMEOUT=900 \
  ./tests/sm86/bench_llama31_arxiv_v1_smoke.sh \
  /path/to/Meta-Llama-3.1-8B

# Llama 3.1 8B PG-19 teacher-forced smoke QA
CONTEXTS='512' NUM_CHUNKS=1 CONFIG=both \
  ./tests/sm86/bench_llama31_pg19_smoke.sh \
  /path/to/Meta-Llama-3.1-8B

# Llama 3.1 8B PG-19 DotCache reference smoke QA
CONTEXTS='4096' NUM_CHUNKS=1 CONFIG=both REFERENCE_SMOKE=1 \
  FAIL_ABOVE_REFERENCE=1 TIMEOUT=1200 \
  ./tests/sm86/bench_llama31_pg19_smoke.sh \
  /path/to/Meta-Llama-3.1-8B

# Combined wrapper
SUPERSONIC_BACKENDS=cuda ./tests/sm86/run_all.sh \
  /path/to/Qwen3.5-0.8B \
  /path/to/Qwen3.5-4B
```

The warm `phi4-mini --fp8-runtime --validate` command above passes on `sm86`
with `token_mismatches=0`, `max_delta=1.5000`, and `34.7 ms/step` over 8
generated tokens.

Each `sm86` script currently validates:

- baked `.supersonic/v1` loading
- direct `--no-bake` loading
- oracle logit deltas
- replay-based `--gpu-validate` deltas and token agreement
- golden corpus coverage
- CUDA `4B --kv-fp8` on the validated `sm86` lane

`tests/sm86/run_batch.sh` adds `qwen3.5-4b --batch-size 2` coverage on the same `sm86` target.
`tests/sm86/run_fast_greedy.sh` checks that the CUDA fast-greedy 0.8B path
matches the legacy host-logits sampling path on short, medium, and long prompts.
`llama3.1-8b --int8` is checked with the PyTorch oracle, `--gpu-validate`, and
fast-greedy/full-logits token regression runs.
`tests/sm86/bench_llama31_arxiv_v1_smoke.sh` covers generated RULER/NIAH-style
retrieval smoke QA, and `tests/sm86/bench_llama31_pg19_smoke.sh` covers
teacher-forced PG-19/perplexity smoke QA for dense INT8 vs certified KV.
The CUDA certified-KV runtime validates Tier-1 compressed KV plus adaptive
BF16-key promotion, with BF16 originals retained in host-pinned Tier-2 storage
and promoted keys/values paged into compact device scratch by the fallback path.
`tests/sm86/run_negative.sh` covers unsupported CUDA v1 flags and explicit failure modes.
The default short/medium `sm86` scripts still validate against the CUDA oracle.
The long-context scripts use the CPU oracle on this box, because that is the stable reference
for longer `4B` prompts today.
`tests/sm86/run_long.sh` and `tests/sm86/run_4b_long.sh` add explicit long-context coverage
against the CPU oracle using focused long-only golden corpora.

The CUDA KV-FP8 lane is validated separately with commands like:

```bash
target/release/supersonic --backend cuda --oracle-device cpu \
  --model qwen3.5-4b --model-dir /path/to/Qwen3.5-4B \
  --prompt '中国的首都是' --max-new-tokens 8 \
  --batch-size 2 --kv-fp8 --validate

CORPUS_TIMEOUT=1200 tests/corpus/run_golden.sh \
  qwen3.5-4b /path/to/Qwen3.5-4B tests/corpus/golden_4b_batch2.json \
  target/release/supersonic --backend cuda --oracle-device cpu \
  --batch-size 2 --kv-fp8
```

### Benchmark baseline

For CUDA baseline measurements on `sm86`, use:

```bash
SUPERSONIC_BACKENDS=cuda ./tests/sm86/bench.sh \
  /path/to/Qwen3.5-0.8B \
  /path/to/Qwen3.5-4B
```

For a warmed Lucebox-style `qwen3.5-0.8b` parity run on `sm86`, use:

```bash
SUPERSONIC_BACKENDS=cuda ./tests/sm86/bench_qwen08.sh \
  /path/to/Qwen3.5-0.8B
```

That harness defaults to batch-1 BF16, a roughly `pp520` prompt target,
`tg128`, `10` warmup runs, `20` timed runs, and prints aggregated native decode
stage timings from `--emit-stage-timings`.

For a warmed single-sequence native-kernel `qwen3.5-4b` run on `sm86`, use:

```bash
SUPERSONIC_BACKENDS=cuda ./tests/sm86/bench_qwen4b_single.sh \
  /path/to/Qwen3.5-4B
```

That harness forces `--force-kernel-decode` so the run measures the native
single-sequence `4B` kernel instead of the default replayed-prefill
correctness path.

The current `qwen3.5-0.8b` CUDA `sm86` optimization record, benchmark progression,
remaining gap to Lucebox, and carry-forward process for the other supported Qwen3.5
CUDA models are tracked in [docs/qwen35-sm86-optimization.md](qwen35-sm86-optimization.md).

For a one-token Nsight Compute pass over the non-4B persistent decode kernel on
`sm86`, use:

```bash
SUPERSONIC_BACKENDS=cuda ./tests/sm86/profile_qwen08_decode.sh \
  /path/to/Qwen3.5-0.8B
```

Set `PROFILE_MODE=fast` to disable the hero path while keeping CUDA fast-greedy,
or `PROFILE_MODE=legacy` to force the old host-logits decode path.

Current behavior on this `sm86` box now defaults to the native kernel path for
single-sequence `qwen3.5-4b`; the older replayed-prefill decode path is opt-in
via `--force-replay-decode`.

With a quick harness pass (`PROMPT_REPEAT=8`, `MAX_NEW_TOKENS=8`, `RUNS=1`):

- `qwen3.5-0.8b`: prefill `206 ms` for 112 prompt tokens (`544 tok/s`), decode `75 ms` for 8 generated tokens (`106.7 tok/s`)
- `qwen3.5-4b --batch-size 1`: prefill `898 ms` for 112 prompt tokens (`124.7 tok/s`), decode `308 ms` for 8 generated tokens (`26.0 tok/s`)
- `qwen3.5-4b --batch-size 2`: prefill `911 ms` for 112 prompt tokens (`122.9 tok/s`), decode `1042 ms` for 16 aggregate generated tokens (`15.4 tok/s`)

There is also an explicit native single-sequence `4B` CUDA hero lane behind
`--force-kernel-decode`. The exact lane is:

- CUDA + `sm86`
- `qwen3.5-4b`
- BF16
- baked load
- `--force-kernel-decode`
- `--batch-size 1`
- warmed `pp533 / tg16`

Current best verified result on this box for that lane is commit `5a34190`:

- prefill `5252 ms` (`101.5 tok/s`)
- decode `727 ms` (`22.0 tok/s`)
- persistent decode stage `655 ms`

That single-stream lane is for Lucebox-style native-kernel optimization work.
`qwen3.5-4b --batch-size 2` remains the validated batched throughput lane.
Detailed CUDA `sm86` history for both the `0.8B` and `4B` hero lanes lives in
[docs/qwen35-sm86-optimization.md](qwen35-sm86-optimization.md).

## Metal

Metal support is currently an Apple-silicon lane validated on Apple M4 and
Apple M5 Max, with Qwen3.5 BF16 coverage, Qwen3.6-35B-A3B INT4 coverage, and
large Apple M5 Max component/chained-path coverage.
The core decode path is now O(N) incremental decode — no replay overhead.

Supported Metal scope:

- `qwen3.5-0.8b`, `qwen3.5-2b`
- `qwen3.5-4b`, `qwen3.5-9b` on Apple M5 Max
- `qwen3-30b-a3b` INT4 on Apple M5 Max is supported through a correctness-first
  chained Metal fallback
- `qwen3.6-35b-a3b` INT4 on Apple M5 Max is supported through the
  correctness-first chained Metal decode route
- `gemma4-e2b`, `gemma4-e4b` BF16 and INT4 on Apple M5 Max are supported
  through a component Metal decode path
- `phi4-mini` BF16, INT4, and FP8-runtime on Apple M5 Max are supported through
  a component Metal decode path
- Apple M4 / `apple-m4`
- Apple M5 Max / `apple-m5-max`
- BF16 prefill parity against the Python CPU oracle
- CLI and `supersonic-serve` HTTP server
- native Metal greedy prefill
- Metal v2 incremental decode: length-1 forward pass per token with persistent conv/recurrent/KV state
- checked token-ID prompt corpus via `qwen35_bughunt`

Metal currently rejects or defers:

- `--fp8-runtime` outside the Phi4 component path
- `--kv-fp8`
- batched decode
- persistent megakernel decode (all ops fused into one dispatch)
- `qwen3-30b-a3b` INT4 on Apple M5 Max has a correctness-first chained Metal
  fallback for decode-layer attention, MoE routing, expert matmul, KV updates,
  and the final INT4 lm-head. The persistent Qwen3-MoE megakernel remains
  HIP-only.
- `qwen3.6-35b-a3b` INT4 on Apple M5 Max uses the host-orchestrated chained
  decode route with Metal fallbacks for BF16 full-attention and FFN stages,
  plus INT4 sidecars for projection and expert matvecs. Stage-5
  linear-attention projection/recurrent work uses a fused native Metal INT4
  path by default; set `SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_INT4_STAGE5=1`
  to force the host fallback. Decode publishes that native linear output
  directly into the next residual buffer by default; set
  `SUPERSONIC_METAL_DISABLE_QWEN36_LINEAR_DECODE_DIRECT=1` to retain the older
  D2D-copy handoff. Stage-5 FFN work stays on the host fallback by default,
  with routed expert gate/up and down rows batched across top-k
  experts; `SUPERSONIC_METAL_PROFILE=1` profiles that default lane rather than
  switching implementations. The experimental native FFN path remains explicit
  opt-in via `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_INT4_STAGE5=1`; routed expert
  gate/up and down matvecs are the next measured optimization target. The
  tiled routed-expert gate/up kernel can be tested with
  `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_GATE_UP_TILED=1`, but it is
  diagnostic-only because expert down/finalize still has to read the
  intermediate through the host path. A combined routed expert gate/up +
  down/finalize experiment keeps `expert_mid` device-side behind
  `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_TILED_STAGE5=1`; it also remains
  diagnostic-only because the current real decode profile is dominated by
  command-buffer wait and large bake-buffer residency rather than the GPU
  arithmetic row. A direct-gather follow-up,
  `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DIRECT_GATHER_STAGE5=1`, keeps
  original expert IDs and uses a wider tiled down/finalize kernel, but it is
  still diagnostic-only because the measured FFN time remains above the default
  host-orchestrated lane. A packed active-expert variant,
  `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACKED_STAGE5=1`, copies only the
  routed top-k expert slabs into compact scratch buffers before running the
  same combined shader. It is also diagnostic-only: it proves the giant-buffer
  residency fault is real, but the CPU packing cost is still too high for the
  default lane. The raw GGML expert-down/finalize lane now uses the stable
  one-row top-k-parallel variant by default for `top_k <= 8`; set
  `SUPERSONIC_METAL_DISABLE_QWEN36_FFN_EXPERT_DOWN_TOPK_PARALLEL=1` to force
  the older multirow finalizer for A/B checks. The explicit
  `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DOWN_TOPK_PARALLEL=1` switch is
  still accepted for older scripts, but is no longer required for raw `--q4km`.
  `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DOWN_MULTIROW_TOPK_PARALLEL=1`
  tests a two-row-per-threadgroup top-k variant that keeps one-row top-k
  semantics with isolated per-row scratch; it is diagnostic-only until it shows
  a real speedup over the one-row default.
  `SUPERSONIC_METAL_DIAG_QWEN36_FFN_EXPERT_DOWN_ROWPAIR_TOPK_PARALLEL=1`
  tests a two-row, top-k-parallel raw Q4_K_M expert-down/finalize variant. It
  is diagnostic-only/quarantined: a 128-token A/B preserved the stream and
  reduced the profiled expert-down phase, but repeated 512-token gates showed
  nondeterministic late divergence.
  `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_DOWN_GATHERED=1` tests the
  MLX-shaped selected-expert down path: compute `[top_k, hidden]` down outputs
  into the existing FFN workspace, then combine top-k weights in a separate
  finalizer. It is parity-safe for the current raw Q4_K_M lane, but remains
  diagnostic-only until repeated runs show a real decode-speed win.
  `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACK_CACHE=1` caches exact
  per-layer active sets, while
  `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACK_HOTSET=1` keeps an LRU
  resident slot pool sized by
  `SUPERSONIC_METAL_QWEN36_FFN_EXPERT_HOTSET_CAPACITY` (default 16). Neither is
  promoted because measured route churn still leaves too much slab copy and
  residency overhead. A static native INT4 top-N probe can also be stacked on
  the packed path with
  `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_STATIC_TOPN=1` and
  `SUPERSONIC_METAL_QWEN36_FFN_EXPERT_STATIC_TOPN_FILE=target/qwen36_static_topn_mps_probe.json`;
  set `SUPERSONIC_METAL_QWEN36_FFN_EXPERT_STATIC_TOPN_CAPACITY` to choose one
  exported capacity, otherwise the largest table is used. Full hits run from
  resident slots and misses fall through to the existing packed/hotset path. A
  GPU-side active-slab pack probe can be stacked on the
  packed path with
  `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_GPU_PACK_STAGE5=1`. It keeps the
  CPU out of the per-token copy loop and remaps top-k expert IDs inside the
  same Metal command buffer, but remains diagnostic-only because the measured
  wall time is worse than the CPU pack path on this machine. Profile runs keep
  Qwen3.6 linear stage-5 aggregate by default; set
  `SUPERSONIC_METAL_PROFILE_QWEN36_LINEAR_PHASES=1` with
  `SUPERSONIC_METAL_PROFILE=1` only when you need per-phase linear command-buffer
  attribution and accept the extra waited submits. Profile runs keep
  Qwen3.6 FFN candidate stages aggregate by default; set
  `SUPERSONIC_METAL_PROFILE_QWEN36_FFN_PHASES=1` with
  `SUPERSONIC_METAL_PROFILE=1` only when you need per-phase FFN command-buffer
  attribution and accept the extra waited submits. Add
  `SUPERSONIC_METAL_PROFILE_QWEN36_ROUTER_PHASES=1` to split the router block
  into norm/logits/top-k labels. Decode-batch router parity can be tapped
  without forcing the older chained router path by setting
  `SUPERSONIC_METAL_QWEN36_DECODE_BATCH_ROUTER_STAGE5_PARITY_TAP=1`; narrow it
  with the matching `_MAX_CALLS`, `_POSITION`, and `_LAYER` suffixes. The legacy
  non-batch router tap also supports
  `SUPERSONIC_METAL_QWEN36_FFN_ROUTER_STAGE5_PARITY_TAP_LAYER`.
  `SUPERSONIC_METAL_ENABLE_QWEN36_FFN_ROUTER_FUSED_EXACT=1` selects the
  monolithic exact router candidate instead of the default split exact-SIMD
  router. It is useful for launch-count and local router-parity attribution, but
  remains diagnostic-only because its stream-stable command-buffer cadence is
  slower than the default exact-SIMD router path. When this fused router is
  enabled, decode batch now avoids splitting each FFN into its own deferred
  command buffer by default; set
  `SUPERSONIC_METAL_QWEN36_DECODE_BATCH_FFN_COMMIT_INTERVAL=N` to override that
  cadence during scheduling experiments. Profile runs emit
  `[qwen36-expert-residency]` and `[qwen36-expert-residency-policy]`; set
  `SUPERSONIC_QWEN36_EXPERT_RESIDENCY_PROFILE=1` to collect the same residency
  counters without enabling the full Metal/HAL profile tables. The one-token
  local smoke is:
  `cargo run --release --bin supersonic -- --backend metal --model qwen3.6-35b-a3b --model-dir /path/to/Qwen3.6-35B-A3B --int4 --prompt "Hello" --max-new-tokens 1 --emit-stage-timings`.
  FP8-runtime, KV-FP8, speculative decode, and persistent decode remain
  unsupported on this Metal path.
- `phi4-mini` INT4 and FP8-runtime use the component Metal route; KV-FP8 is
  still unsupported
- `gemma4-e2b` and `gemma4-e4b` INT4 use the component Metal route with
  release-hosted INT4 GPTQ bakes

Native Metal kernels used in the hot path:

- matmul RHS-transposed (BF16 + INT4 GPTQ dequant)
- FP8 runtime-dequant matmul has a correctness-first Metal host fallback for
  Phi4 component decode
- full-attention prefill core
- lm-head argmax, including the Qwen3.6 greedy lm-head tail experiment gated by
  `SUPERSONIC_METAL_ENABLE_QWEN36_LM_HEAD_GPU_ARGMAX=1`
- RMSNorm rows
- Qwen3.6 stage-5 linear-attention INT4 fused projection/recurrent path
- Qwen3.6 stage-5 FFN INT4 projection kernels for explicit opt-in
  experiments
- Qwen3.6 routed-expert gate/up tiled INT4 kernel for focused FFN
  microbenching and explicit opt-in attribution
- Qwen3.6 routed-expert gate/up + down/finalize tiled INT4 kernel for focused
  FFN microbenching and explicit opt-in attribution
- Qwen3.6 packed active-expert routed FFN experiment for explicit
  residency-attribution runs
- linear prefill conv pack
- element add
- cast
- scalar multiply
- SHD-to-HSD transpose
- QKV split
- Q-gate split

Current Apple M4 / Apple M5 Max checkpoint:

- `qwen35_bughunt --mode gate`: PASS for `hello_world`, `forest_prompt`, and `code_prompt`
- `supersonic --backend metal --model qwen3.5-0.8b --prompt "Hello, world" --max-new-tokens 8`:
  - prefill about `112 ms`
  - incremental decode about `34 ms/token` (constant across context lengths)
- `--gpu-validate` on 16-token sequence: `gpu_oracle_max_delta=0.0000` every step
- Apple M5 Max validation also covers `qwen3.5-0.8b --validate --oracle-device cpu`
  and the checked-in bughunt gate.
- Apple M5 Max validation also covers one-token `qwen3.5-4b` and `qwen3.5-9b`
  `--validate --oracle-device cpu` smokes through Metal v2 incremental decode.
- Apple M5 Max validation also covers one-token large-model smokes for
  `qwen3-30b-a3b` INT4, `gemma4-e2b` BF16/INT4, `gemma4-e4b` BF16/INT4, and
  `phi4-mini` BF16/INT4/FP8-runtime.

The current Qwen3.6 optimization target is measured from the Apple M5 Max
bench/profile loop rather than guessed up front. As of the latest local
checkpoint, the retained headline is `58.3 ms/token` (`17.2 tok/s`) from
`target/bench-runs/2026-05-25-d20a655-9`, with samples `58.3`, `60.3`, and
`58.2`. The unprofiled stage table is `ffn_ms_avg=32.611`,
`linear_attn_ms_avg=15.954`, `full_attn_ms_avg=5.737`, and
`lm_head_ms_avg=4.659`. The profile pass still names `command_buffer_wait` and
`qwen36_linear_int4_stage5` first, followed by FFN host expert gate/up/down, so
the next measured target is the Metal linear stage/wait pair plus the remaining
FFN expert rows.

### Large model setup

The large Apple M5 Max Metal smoke suite uses one canonical model root:

```bash
export SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models"
export HF_HOME="$HOME/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$SUPERSONIC_TEST_MODEL_ROOT"

hf download Qwen/Qwen3-30B-A3B --local-dir "$SUPERSONIC_TEST_MODEL_ROOT/qwen3-30b-a3b"
hf download google/gemma-4-E2B --local-dir "$SUPERSONIC_TEST_MODEL_ROOT/gemma4-e2b"
hf download google/gemma-4-E4B --local-dir "$SUPERSONIC_TEST_MODEL_ROOT/gemma4-e4b"
hf download microsoft/Phi-4-mini-instruct --local-dir "$SUPERSONIC_TEST_MODEL_ROOT/phi4-mini"

# Qwen3.6 is large enough that this machine keeps the HF snapshot as the
# source of truth and exposes a stable SuperSonic alias. Replace `<snapshot>`
# with the hash under `$HUGGINGFACE_HUB_CACHE/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/`.
hf download Qwen/Qwen3.6-35B-A3B-FP8
ln -sfn "$HUGGINGFACE_HUB_CACHE/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/<snapshot>" \
  "$SUPERSONIC_TEST_MODEL_ROOT/qwen3.6-35b-a3b"
```

Quantized lanes use release-hosted bakes and install them under each model's
`.supersonic/` directory on first run:

- `v2-int4-gptq` for `qwen3.6-35b-a3b`
- `v2-int4-gptq` for `qwen3-30b-a3b`, `gemma4-e2b`, `gemma4-e4b`, and
  `phi4-mini --int4`
- `v2-fp8` for `phi4-mini --fp8-runtime`

On this development machine, the same convention is sourced from
`~/.config/supersonic/env.zsh` by `~/.zshenv`.

The ignored large-model gate is:

```bash
SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  cargo test --release -p runner --test metal_large_model_smoke -- --ignored --nocapture
```

The Qwen3.6 Apple M5 Max main-target loop is:

```bash
SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  cargo test --release -p runner --test qwen36_moe_metal_smoke -- --ignored --nocapture

SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  cargo run --release -p supersonic-bench --bin bench-perf -- \
  --arch apple-m5-max --models qwen3.6-35b-a3b --quants int4

SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  python3 tests/metal/bench_qwen36_longctx.py --preset smoke

SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  python3 tests/metal/bench_qwen36_longctx.py --preset smoke --metal-profile

SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  python3 tests/metal/bench_qwen36_longctx.py --preset smoke \
  --batched-prefill-feasibility

SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  python3 tests/metal/bench_qwen36_longctx.py --preset smoke \
  --batched-prefill-prototype

SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  python3 tests/metal/sweep_qwen36_batched_prefill_variants.py --metal-profile

SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  python3 tests/metal/audit_qwen36_mtp.py --require-complete-bake

SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  python3 tests/metal/probe_qwen36_mtp_acceptance.py

SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  python3 tests/metal/probe_qwen36_mtp_acceptance.py --metal-experiment

SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  python3 tests/metal/sweep_qwen36_mtp_acceptance.py --prompt-set smoke \
  --metal-experiment

SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  python3 tests/metal/probe_qwen36_static_topn.py

SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  python3 tests/metal/sweep_qwen36_static_topn_runtime.py \
  --modes default,static,static-hotset,mps-static-partial --metal-profile

SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  python3 tests/metal/sweep_qwen36_linear_decode.py \
  --prompt-set smoke --metal-profile

SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  python3 tests/metal/sweep_qwen36_full_decode.py \
  --prompt-set smoke --metal-profile

SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  python3 tests/metal/sweep_qwen36_lm_head_tail.py \
  --prompt-set smoke --metal-profile

SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  python3 tests/metal/probe_qwen36_mps_resident_table.py \
  --run-pilot --require-pilot

SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  python3 tests/metal/sweep_qwen36_route_residency.py --prompt-set smoke

python3 tests/metal/summarize_qwen36_sota_gates.py --require

cargo build --release -p runner --bin qwen36_ffn_expert_microbench
target/release/qwen36_ffn_expert_microbench --iters 20 --warmup 3
```

Use the long-context harness with `--preset comparison` before choosing the
next Metal runtime optimization target. The report records generated-token
sanity, NIAH hit/miss, stage timings, chain breakdown, and lifecycle timings
for the supported INT4 chained-decode lane. With `--metal-profile`, it also sets
`SUPERSONIC_METAL_PROFILE=1` and records parsed `metal_profile` and
`hal_profile` summaries from the machine-readable profile lines. With
`--batched-prefill-feasibility`, it keeps the supported Metal per-token prefill
path but records grouped-MoE router/permutation occupancy from actual route
choices via `[qwen36-batched-prefill-feasibility]` plus
`[qwen36-batched-prefill-plan]` candidate chunk rows. The plan rows compare
64/128/256/512/1024-token chunks by grouped-expert segment count, scalar tails,
WMMA16 assignment coverage, and padded-row overhead; set
`SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL_PLAN_CHUNKS=...` to override the
candidate list for a local experiment. The 512-token smoke is
intentionally slow on the current chained Metal prefill path; treat the
comparison preset as a long-running sweep rather than a per-commit unit gate.
On the first 512-token smoke, the 512/1024-token plans tied because the prompt
had 417 profiled prefill tokens: 82.7% WMMA16 assignment coverage, 23,048
scalar-tail assignments, and 54.6% WMMA16 padding overhead. Smaller chunks
increased scalar tails sharply.
The Metal long-context harness now runs the Metal batched-prefill route by
default. This path uses Metal batched full-attention and a direct routed-expert
INT4 kernel pair for grouped MoE prefill while keeping router/top-k on the
existing host path; set `--legacy-prefill-baseline` to force the older
per-token Metal prefill baseline for A/B reports. `--batched-prefill-prototype`
is retained as a compatibility/provenance tag for reports that still separate
prototype-default from baseline modes.
`--batched-prefill-variant` makes the env-gated prototype probes reproducible
from the harness: `linear-direct-off`, `full-attn-vec-off`, `full-attn-tmajor`,
`split-qgate`, `router-topk`, `fused-residual-off`, and
`shared-expert-batch-off`. The JSON rows record both the variant name and the
exact env overrides used for the run.
`tests/metal/sweep_qwen36_batched_prefill_variants.py` runs the supported
`baseline`, `prototype-default`, and selected named variants against the same
deterministic NIAH prompt per context. It writes
`target/qwen36_metal_batched_prefill_variant_sweep.json` and
`target/qwen36_metal_batched_prefill_variant_sweep.md`, preserving generated-ID
parity, prefill ratios versus baseline, stage/lifecycle rows, and optional
Metal/HAL attribution with `--metal-profile`. The v2 report also includes a
nonfatal `promotion_gate`: variants must preserve generated IDs, improve
prefill, headline decode, and `ffn_ms_avg`, avoid material full-attention,
linear-attention, or lm-head regressions, and carry command-buffer-wait profile
evidence unless `--no-promotion-require-profile` is used. The first local
512-token normal smoke generated the same `[271]` one-token sanity row and
reduced prefill to 22.15s; the profiled smoke measured 34.20s because it splits
the routed-expert gate/up and down/combine phases for attribution. A follow-up
Metal row-scalar sigmoid kernel now avoids expanding the shared-expert scalar
gate through host memory; the 512-token normal prototype smoke measured 12.47s
prefill and 172.54 ms/token with the same `[271]` sanity row. The profiled row
shows `sigmoid_mul_row_scalar` at 11.859 ms native wall and 0.725 ms GPU time
across 40 layers, so the next measured pressure remains command-buffer
orchestration, linear-attention volume, full-attention prefill, and routed
expert direct work. A follow-up linear-attention orchestration slice keeps the
native stage-5 output scratch separate from the final row destination, batches
each linear layer's per-token Metal launches into one command buffer, and
writes the final residual directly into the chunk row. With
`SUPERSONIC_QWEN36_MOE_METAL_LINEAR_PREFILL_DIRECT=0` as the control, the
512-token smoke measured 13.30s prefill; with the direct-row batch enabled it
measured 10.89s prefill and the same `[271]` sanity row. Profile runs keep the
waited per-token path so phase attribution remains readable.
The next full-attention pass keeps reusable prefill scratch on by default and
leaves two measured layout shortcuts behind opt-in env gates. Reusing
Q/K-after-norm and KV-prefix buffers avoids per-layer allocation churn. The
time-major KV attention probe
(`SUPERSONIC_QWEN36_MOE_METAL_FULL_ATTN_TMAJOR=1`) measured 11.09s prefill,
and the native Q/gate split probe
(`SUPERSONIC_QWEN36_MOE_METAL_SPLIT_QGATE=1`) measured 11.00s prefill, so both
remain off by default. The default 512-token smoke with the probes off measured
10.73s prefill and preserved `[271]`; the next target remains routed expert
compute/residency.
The MTP audit writes `target/qwen36_mtp_audit.json` and
`target/qwen36_mtp_audit.md`. It checks the local safetensors index for the
split source MTP tensors, checks the INT4 bake manifest for the 19 folded
runtime tensors consumed by `mtp_loader.rs`, and can fail closed with
`--require-complete-bake`. The current local Qwen3.6 FP8 snapshot reports
1,560 `mtp.*` source tensors and the INT4 bake reports all 19 runtime MTP
tensors, but speculative decode remains unsupported on the Metal policy path
until the MTP path is explicitly enabled. The acceptance probe writes
`target/qwen36_mtp_acceptance_probe.json` and
`target/qwen36_mtp_acceptance_probe.md`. Today on Metal it records the expected
`policy_blocked` row; when run on a backend with speculative decode enabled it
parses `[qwen36-mtp-acceptance]` telemetry with drafted tokens, accepted
tokens, emitted tokens, base verify steps, replay steps, and target
steps/emitted so one-draft and multi-draft acceptance can be measured apart
from FFN latency. `--metal-experiment` sets
`SUPERSONIC_QWEN36_METAL_MTP_EXPERIMENT=1`, enabling the experimental Metal
sequential K=1 path only. The probe also sets
`SUPERSONIC_QWEN36_MOE_BATCHED_PREFILL=0` on Metal so it stays on the
supported per-token prefill lane; `--batched-spec-verify` remains blocked.
The first local K=1 smoke completed in 24.3s and recorded
`drafted_tokens=2`, `accepted_tokens=1`, `acceptance_rate=0.5`, and
`target_steps_per_emitted=1.0`, so it is a correctness/acceptance foothold, not
a throughput win by itself. The sweep harness writes
`target/qwen36_mtp_acceptance_sweep.json` and
`target/qwen36_mtp_acceptance_sweep.md`, then aggregates measured rows across a
small prompt suite so K=1 promotion is based on prompt breadth rather than a
single acceptance sample. It also reports a non-fatal `promotion_gate` summary
using aggregate acceptance and target steps per emitted token; add
`--metal-profile` to retain parsed Metal/HAL attribution rows for each prompt.
The first two-prompt Metal smoke measured both rows:
profiling accepted 0/2 drafts, coding accepted 1/2 drafts, aggregate acceptance
was 25.0%, and aggregate `target_steps_per_emitted` stayed 1.0. That keeps the
path useful for telemetry, but not ready for Metal policy promotion.
The static top-N probe writes `target/qwen36_static_topn_mps_probe.json` and
`target/qwen36_static_topn_mps_probe.md`. It uses the route profiler's gated
`[qwen36-route-topn-layer]` and `[qwen36-route-call]` rows to choose per-layer
oracle top-N experts from a calibration prompt, replay a separate
coding-shaped evaluation prompt, and report assignment coverage, full-hit call
rate, miss fallback calls, worst layer, and the FP16 MPS RHS footprint for each
resident capacity. This is a resident-table sizing and hit-rate probe only; it
does not yet replace FFN execution with a static MPS table.
On the first two-prompt local smoke, capacity 16 covered only 35.8% of
evaluation assignments with 876/880 FFN calls still requiring fallback, while
requiring 3.75 GiB of resident FP16 RHS storage. That argues against a tiny
static table as the next default path.
`tests/metal/sweep_qwen36_static_topn_runtime.py` consumes the generated
`static_tables` JSON and compares warm decode modes such as `default`,
`static`, and `static-hotset`. It writes
`target/qwen36_static_topn_runtime_sweep.json` and
`target/qwen36_static_topn_runtime_sweep.md`, preserving generated IDs per
prompt, stage timings, chain breakdown, lifecycle timings, and expert-residency
policy rows for each mode. Add `--metal-profile` to keep parsed
`metal_profile` / `hal_profile` objects in each row and render the top
attribution rows in Markdown. The v3 report includes a nonfatal
`promotion_gate`: a resident mode must preserve generated IDs versus `default`,
improve headline ms/token and `ffn_ms_avg`, keep full-attention,
linear-attention, and lm-head inside the configured regression ratio, and carry
command-buffer-wait profile evidence unless `--no-promotion-require-profile` is
used.
`tests/metal/sweep_qwen36_lm_head_tail.py` compares the default full-logit
host-sampling tail against `gpu-argmax`, which enables
`SUPERSONIC_METAL_ENABLE_QWEN36_LM_HEAD_GPU_ARGMAX=1`. The opt-in path is
limited to greedy Metal runs that do not request host logits; it preserves
full-logit readback for non-greedy sampling, `--dump-last-logits`, and
`SUPERSONIC_QWEN36_DUMP_LOGITS`. Its v1 promotion gate requires generated IDs
to match default, headline and `lm_head_ms_avg` to improve, the chain buckets
to stay inside the regression ratio, and command-buffer-wait profile evidence
when profiling is required.
The MPS resident-table probe writes
`target/qwen36_mps_resident_table_probe.json` and
`target/qwen36_mps_resident_table_probe.md`. Its v2 `viability_gate` is an
estimate gate, not a runtime promotion gate: full-hit-only and partial-hit
candidates must fit the configured resident RHS budget and meet projected FFN
speedup, coverage, and full-hit-rate thresholds. A passing partial-hit estimate
still requires a runtime path that avoids per-token FP16 RHS rebuilds.
`tests/metal/sweep_qwen36_route_residency.py` runs the existing route profiler
across a small prompt suite and writes `target/qwen36_route_residency_sweep.json`
and `target/qwen36_route_residency_sweep.md`. Its v1 `decision_gate` compares
per-layer LRU hit rates against oracle static top-N coverage so the next
resident-expert branch can be chosen from prompt-shaped route evidence instead
of one Hello run.
`tests/metal/sweep_qwen36_fused_routed_int4.py` compares the `default`,
`direct-gather`, and `gpu-pack` routed-expert INT4 decode paths under the same
prompt suite and writes `target/qwen36_fused_routed_int4_sweep.json` and
`target/qwen36_fused_routed_int4_sweep.md`. Its v1 `promotion_gate` preserves
the fused INT4 fork as a measured runtime decision: generated IDs must match
default, headline decode and FFN time must improve, component timings must stay
inside the regression ratio, and command-buffer-wait profile evidence is
required unless explicitly disabled.
`tests/metal/summarize_qwen36_sota_gates.py` is the cross-harness summary for
the current roadmap gates. It reads the batched-prefill variant sweep, static
top-N runtime sweep, fused routed INT4 runtime sweep, MPS resident-table probe,
route residency sweep, MTP acceptance sweep, LRU resident-cache sweep, linear
decode sweep, and full-attention decode sweep JSON reports, then writes
`target/qwen36_sota_gate_summary.json`
and `target/qwen36_sota_gate_summary.md`. Missing reports remain visible as rows
by default, and each row includes the command that refreshes the corresponding
gate artifact. Add `--require` to make a local validation run fail closed on
missing, malformed, schema-mismatched, stale, or missing-gate artifacts. Use
`--max-age-hours N` when the summary must reject old target reports instead of
quietly reusing yesterday's gate decisions.
The v9 summary marks an estimate or decision gate as superseded when a newer
runtime sweep has already measured and rejected that candidate, keeping
`next_action` pointed at untried implementation work.
`tests/metal/refresh_qwen36_sota_gates.py` turns those row-level refresh
commands into a dry-run plan at
`target/qwen36_sota_gate_refresh_plan.{json,md}`. It selects missing/stale/bad
inputs by default, supports `--only <gate_id>` for a deliberate one-gate
refresh, and executes the selected commands only when `--run` is provided.
When the summary's `next_action` is
`keep_default_lane_and_select_next_measured_bottleneck`, run
`tests/metal/select_qwen36_next_bottleneck.py`. It writes
`target/qwen36_next_bottleneck.{json,md}` by ranking the profiled default-lane
decode buckets and skipping FFN as an action target once the resident/static/
fused/MPS/LRU FFN forks are all measured negative. Linear and full attention
are likewise skipped only after their decode variant gates have failed.
The current M5 Max perf gate points at linear-attention as the next measured
multi-token per-token bucket after FFN fallback tightening. The 512-token
`--metal-profile` smoke currently reports roughly 269 ms/token, 71.7 s prefill,
and a clear NIAH miss row; its one-token row is lm-head/tail dominated, while
the Metal profile totals still point at FFN command-buffer wait. The routed
expert microbench reports `mean_ms=0.4490` for gate/up and `mean_ms=0.3332`
for combined gate/up + down/finalize with zero mismatches on the exact Qwen3.6
stage-5 INT4 geometry. Wired into real decode with
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_TILED_STAGE5=1`, the combined path
still generates `[11]`, but it regresses badly: the profiled one-token smoke
measured `ffn_ms_avg=1458.677`, with only `19.760 ms` total GPU time for
`command_buffer_gpu:qwen36_ffn_int4_expert_gate_up_down_finalize_tiled` across
40 layers and `1561.827 ms` in `command_buffer_wait`. The packed active-expert
variant still generates `[11]` and cuts the unprofiled one-token result to
roughly `337 ms/token`, but profile attribution shows `100.287 ms` in
`qwen36_ffn_int4_expert_pack_stage5`, `43.556 ms` in
`qwen36_ffn_int4_expert_packed_stage5`, and `182.291 ms` in
`command_buffer_wait`. A short four-token cache probe is available with
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACK_CACHE=1`, but the measured
result was not promotable: it reduced HAL alloc/free churn while regressing the
profiled control from `217.6 ms/token` to `234.7 ms/token` on the same prompt.
The hotset probe,
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_PACK_HOTSET=1`, is also
diagnostic-only. A four-token M5 Max profile generated the expected
`[11, 353, 599, 264]` with 16 resident slots and cut copied bytes from
`2014248960` to `1356470784`, but it measured `298.0 ms/token`; 32 resident
slots cut copied bytes only slightly further to `1345455360` and measured
`304.2 ms/token`. The GPU-side active-slab pack probe,
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_GPU_PACK_STAGE5=1`, also preserves
the expected four-token output but measured `777.3 ms/token` with
`ffn_ms_avg=641.772`; it removes CPU slab copying but moves the same per-token
active slab materialization into Metal command-buffer waits. These probes point
away from slab rebuilding, whether CPU or GPU driven, and toward a different
fused INT4 addressing scheme or eliminating the per-token packed slab
path.
`SUPERSONIC_METAL_QWEN36_MPS_EXPERT_PILOT=1` adds a resident-FP16
Metal Performance Shaders attribution row for the active-expert GEMV shapes. The
current retained `bench-perf` profile writes this as `mps_expert_pilot` in
schema-v9 JSON. The retained M5 Max run
`target/bench-runs/2026-05-25-d20a655-9` measured `58.3 ms/token` median and
recorded a resident-MPS pilot of `gate_up_ms=0.627`, `down_ms=0.343`,
`gate_up_tflops=5.355`, and `down_tflops=4.894`. That row is attribution only:
the real default lane remains the INT4 host fallback because the measured MPS
bridge candidates still lose once active expert materialization is included.
`SUPERSONIC_METAL_ENABLE_QWEN36_FFN_EXPERT_MPS_BRIDGE=1` enables the first
real bridge experiment: active GPTQ experts are transposed/dequantized to FP16
MPS buffers, MPS runs gate/up and down, and a small Metal finalizer writes the
decode output. It still generated `[11]`, but it is not promotable yet: a warm
one-token profile measured `1707.3 ms/token`, with `1365.389 ms` in CPU
INT4-to-FP16 packing and only `34.032 ms` in bridge command-buffer GPU time.
The follow-up GPU-side transcode uses a 16-entry threadgroup LUT per GPTQ
scale/zero group and keeps the CPU path available with
`SUPERSONIC_METAL_QWEN36_MPS_BRIDGE_CPU_TRANSCODE=1`. It is correct and the
normal async smoke improved slightly to `1683.6 ms/token`, but it is still not
promotable: the profiled GPU-transcode run measured `1766.4 ms/token` with
`qwen36_ffn_int4_expert_mps_transcode_int4_f16=1404.914 ms` wall time across 40
layers while the Metal command-buffer GPU timestamp for the transcode work was
only `66.239 ms`. That points away from more per-token MPS slab materialization
and toward either a fully fused routed-expert INT4 path or persistent resident
expert buffers that avoid rebuilding/consuming large FP16 MPS matrices every
token. The CPU fallback also has an opt-in tiled packed-byte LUT pack experiment
via `SUPERSONIC_METAL_QWEN36_MPS_BRIDGE_CPU_TRANSCODE_LUT=1`: the release
microbench improves scalar packing from `44.725 ms` to `16.602 ms` for the
50.4 MB active slab by mapping each packed INT4 byte to a pair of FP16 values
and transposing through cache-sized tiles. With the LUT path enabled, the bridge
materializes directly into the Metal shared buffers instead of first building
large intermediate CPU slabs and copying them into `MTLBuffer` storage. An
additional opt-in one-way store mode,
`SUPERSONIC_METAL_QWEN36_MPS_BRIDGE_CPU_TRANSCODE_STREAM=1`, uses paired
non-temporal ARM stores for the transposed FP16 flush. It still generated
`[11]`, but it is not promotable: the stream-store smoke measured
`868.1 ms/token` unprofiled, and the profiled run measured `907.5 ms/token`
with `qwen36_ffn_int4_expert_mps_bridge_pack_f16_lut=556.425 ms` across 40
layers. On Apple UMA this is not PCIe upload cost; the remaining blocker is
per-token FP16 MPS slab rebuild/consumption, so the next target is either
resident reused slabs or a fully fused routed-expert INT4 path. Metal profile
runs also emit Qwen3.6 route-locality lines:
`[qwen36-route-profile]`, `[qwen36-route-cache-sim]`, and
`[qwen36-route-topn]`. The route profiler defaults to capacities
2/4/8/16/32/64 and accepts `SUPERSONIC_QWEN36_ROUTE_PROFILE_CAPACITIES` for
custom resident-table budgets. Use those rows to choose a static table before
enabling the opt-in native INT4 top-N runtime probe.

### Metal validation

The canonical Apple silicon gate is:

```bash
SUPERSONIC_BACKENDS=metal \
QWEN35_MODEL_DIR=/path/to/Qwen3.5-0.8B \
QWEN35_BUGHUNT_REPORT_JSON=/tmp/qwen35_bughunt_gate.json \
./tests/metal/qwen35_bughunt_gate.sh
```

The script builds `qwen35_bughunt`, runs the checked-in manifest at
`crates/runner/bughunt/qwen35_metal_manifest.json`, and compares native Metal
prefill, selected hidden rows, and final prefill logits against the Python oracle
on CPU.

To run one prompt from the manifest:

```bash
SUPERSONIC_BACKENDS=metal \
QWEN35_BUGHUNT_PROMPT=code_prompt \
QWEN35_MODEL_DIR=/path/to/Qwen3.5-0.8B \
./tests/metal/qwen35_bughunt_gate.sh
```

Current checkpoint quality on Apple M4:

- `hello_world`: PASS against Python CPU oracle
- `forest_prompt`: PASS against Python CPU oracle
- `code_prompt`: PASS against Python CPU oracle
