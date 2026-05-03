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

Metal support is currently a Qwen3.5 0.8B Apple-silicon lane validated on Apple M4.
The core decode path is now O(N) incremental decode — no replay overhead.

Validated Metal scope:

- `qwen3.5-0.8b`, `qwen3.5-2b`
- Apple M4 / `apple-m4`
- BF16 prefill parity against the Python CPU oracle
- CLI and `supersonic-serve` HTTP server
- native Metal greedy prefill
- Metal v2 incremental decode: length-1 forward pass per token with persistent conv/recurrent/KV state
- checked token-ID prompt corpus via `qwen35_bughunt`

Metal currently rejects or defers:

- models other than `qwen3.5-0.8b` and `qwen3.5-2b`
- `--fp8-runtime`
- `--kv-fp8`
- batched decode
- persistent megakernel decode (all ops fused into one dispatch)

Native Metal kernels used in the hot path:

- matmul RHS-transposed (BF16 + INT4 GPTQ dequant)
- full-attention prefill core
- lm-head argmax
- RMSNorm rows
- linear prefill conv pack
- element add
- cast
- scalar multiply
- SHD-to-HSD transpose
- QKV split
- Q-gate split

Current Apple M4 checkpoint on this machine:

- `qwen35_bughunt --mode gate`: PASS for `hello_world`, `forest_prompt`, and `code_prompt`
- `supersonic --backend metal --model qwen3.5-0.8b --prompt "Hello, world" --max-new-tokens 8`:
  - prefill about `112 ms`
  - incremental decode about `34 ms/token` (constant across context lengths)
- `--gpu-validate` on 16-token sequence: `gpu_oracle_max_delta=0.0000` every step

The next optimization target is a persistent Metal megakernel — collapsing the
per-token command-buffer round-trips into a single dispatch, equivalent to the
HIP persistent decode path.

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
