# Benchmarks

Operator-facing benchmark and profiling recipes. Keep this file focused on
commands that are useful to repeat; long optimization narratives live under
[`optimization/`](optimization/), bring-up notes under
[`bringup/`](bringup/), and implementation plans under [`plans/`](plans/).

For published headline numbers, see [performance.md](performance.md) and
[detailed_performance.md](detailed_performance.md).

## Before A Run

Start from a clean tree, build the target binary, and capture machine state:

```bash
cd /home/deano/projects/SuperSonicBase
git status --short --branch
HIP_ARCH=gfx1100 cargo build --release --bin supersonic
rocm-smi
rocm-smi --showclocks --showpower --showtemp --showfan
```

For dual RDNA3/RDNA4 work:

```bash
SUPERSONIC_BACKENDS=hip HIP_ARCH=gfx1100,gfx1201 \
  cargo build --release --bin supersonic --bin int4_test
```

## Correctness Gates

Run per-arch scripts from [testing.md](testing.md) before trusting benchmark
numbers.

```bash
# RDNA4 / R9700 starter lane: WMMA/int4 harness, direct Qwen3.6 27B smoke,
# and optional Lucebox/DFlash smoke when artifacts are available.
HIP_VISIBLE_DEVICES=1 ./tests/gfx1201/run_matrix.sh

# RDNA3 / 7900 XTX component smoke.
HIP_VISIBLE_DEVICES=0 SUPERSONIC_BACKENDS=hip target/release/int4_test

# Harness unit tests for the Qwen3.6 Lucebox benchmark runner.
python3 -m unittest tests.test_qwen36_he_supersonic_bench
```

## Bench Perf

`supersonic-bench` is the stable matrix runner used by support docs and
release-facing rows.

```bash
cargo run --release -p supersonic-bench --bin bench-perf -- \
  --arch gfx1100 \
  --models qwen3.5-9b \
  --quants int4
```

Apple M5 Max Qwen3.5 raw-Q4 comparison lane:

```bash
SUPERSONIC_TEST_MODEL_ROOT="$HOME/.cache/supersonic-metal-models" \
  cargo run --release -p supersonic-bench --bin bench-perf -- \
  --preset qwen35-q4km-m5-max-gen512
```

## Lucebox Qwen3.6 27B, 7900 XTX

This is the benchmark used for the RX 7900 XTX / `gfx1100` Lucebox Qwen3.6
27B 100 tok/s optimization run.

Canonical workload:

- Hardware: RX 7900 XTX, HIP `gfx1100`, 24 GiB.
- Target: Qwen3.6-27B Q4_K_M-compatible SuperSonic bake.
- Draft: Lucebox DFlash Q8 draft, 5 layers, block size 16.
- Prompt set: Lucebox HumanEval 10-prompt suite.
- Serving mode: no-thinking ChatML prompt wrapping, stop on EOS.
- Generation cap: `n_gen=256`.
- Main success metric: mean decode tok/s across the 10 prompts.
- Guardrails: prompt-level generated-token counts and combined generated-id
  hash must remain stable.

Default local paths used in the historical runs:

```bash
MODEL_DIR=/mnt/data/tmp/supersonic-qwen36-27b-lucebox
DRAFT_DIR=/mnt/data/tmp/qwen36-27b-dflash-q8-bf16
JSONL=/home/deano/projects/lucebox-hub/harness/benchmarks/prompts/bench_he.jsonl
```

SuperSonic full-suite run:

```bash
mkdir -p target/qwen36_100tok_profile2

HIP_VISIBLE_DEVICES=0 \
python3 tests/gfx1100/bench_qwen36_he_supersonic.py \
  --binary target/release/supersonic \
  --target-profile qwen36-27b-lucebox \
  --model-dir "$MODEL_DIR" \
  --backend hip \
  --context-size 512 \
  --n-gen 256 \
  --dflash \
  --dflash-draft-dir "$DRAFT_DIR" \
  --prompt-source jsonl \
  --prompt-format chatml-no-thinking \
  --lucebox-jsonl "$JSONL" \
  --out-json target/qwen36_100tok_profile2/current_10x256.json
```

The best historical artifact was:

```text
target/qwen36_100tok_profile2/append_recurrent_warp32_direct_10x256.json
```

It reported `100.86 tok/s` mean, `99.40 tok/s` weighted, generated `1654`
tokens, stopped early on all 10 prompts, and kept combined output hash
`032209e65467e8aa6c74025dc8b70b325f0ec767054ddff614e04550bb11f3bf`.

Repeat artifact:

```text
target/qwen36_100tok_profile2/append_recurrent_warp32_direct_10x256_repeat.json
```

It reported `100.79 tok/s` mean and `99.38 tok/s` weighted.

Production refactor validation from 2026-06-28, after the runtime/server
boundary cleanup, used the same local 27B target and Q8 DFlash draft artifacts
but forced the full 256-token cap for every HumanEval prompt. The artifact:

```text
target/qwen36_production_refactor/current_10x256.json
```

It reported `42.641 ms/tok`, `24.38 tok/s` mean, `23.45 tok/s` weighted,
generated `2560` total tokens, and stopped early on `0` prompts.

Reference notes:

- Algorithm summary:
  [`optimization/qwen36-100tok-algorithms-and-optimizations.md`](optimization/qwen36-100tok-algorithms-and-optimizations.md).
- Working log:
  [`optimization/qwen36-lucebox-parity-log.md`](optimization/qwen36-lucebox-parity-log.md).
- Roofline/profiler setup:
  [`optimization/qwen36-lucebox-next-roofline.md`](optimization/qwen36-lucebox-next-roofline.md).

## Lucebox Reference

Run Lucebox from its checkout and record the exact command alongside any
SuperSonic comparison:

```bash
cd /home/deano/projects/lucebox-hub/server
python3 scripts/bench_he.py \
  --ddtree-budget 8 \
  --n-gen 256
```

For Qwen3.6-35B-A3B local parity work:

```bash
cd /home/deano/projects/lucebox-hub/server
.venv/bin/python scripts/bench_he.py \
  --target-profile qwen36-35b-a3b \
  --n-gen 256 \
  --skip-tokenize
```

Keep prompt source, prompt formatting, generation length, target quant, draft
quant, backend, and thermal state matched before treating Lucebox and
SuperSonic numbers as comparable.

## RDNA4 / R9700 Smoke Benchmark

The starter `gfx1201` lane reuses the 27B Lucebox harness but keeps generation
short by default:

```bash
HIP_VISIBLE_DEVICES=1 \
BUILD=0 \
RUN_LUCEBOX=1 \
QWEN36_27B_MODEL_DIR=/mnt/data/tmp/supersonic-qwen36-27b-lucebox \
LUCEBOX_HE_JSONL=/home/deano/projects/lucebox-hub/harness/benchmarks/prompts/bench_he.jsonl \
./tests/gfx1201/run_matrix.sh
```

Use this as a correctness and availability check before doing sustained RDNA4
profiling. Longer R9700 performance sweeps should write to a dedicated
`target/qwen36_gfx1201_*` directory and record `HIP_VISIBLE_DEVICES`, clocks,
and whether the binary was built for both `gfx1100,gfx1201`.

The 2026-06-28 production refactor validation ran the full starter matrix with
Lucebox/DFlash artifacts available:

```bash
HIP_VISIBLE_DEVICES=1 RUN_LUCEBOX=1 ./tests/gfx1201/run_matrix.sh
```

The matrix passed the RDNA4 WMMA/int4 harness, direct Qwen3.6 27B smoke, and
one-prompt Lucebox/DFlash smoke. The smoke artifact
`target/qwen36_he_supersonic_gfx1201_smoke.json` reported `60.24 ms/tok`,
`16.60 tok/s`, `16` generated tokens, and `0` early stops.

## Production Server Smoke: Qwen3.6 27B DFlash

This smoke checks the production-facing path rather than the CLI benchmark
path. It assumes `supersonic-serve` was built with HIP and the local Qwen3.6
27B target/draft artifacts exist.

```bash
SUPERSONIC_BACKENDS=hip HIP_ARCH=gfx1100,gfx1201 \
  cargo build --release -p server

SUPERSONIC_BACKENDS=hip HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
target/release/supersonic-serve \
  --backend hip \
  --model qwen3.6-27b \
  --model-dir "$MODEL_DIR" \
  --max-context 4096 \
  --q4km-gptq \
  --dflash \
  --dflash-draft-dir "$DRAFT_DIR" \
  --host 127.0.0.1 \
  --port 8013 \
  --api-key local \
  --no-download \
  --prefix-cache-disable
```

Then run the OpenAI-compatible smoke:

```bash
SUPERSONIC_BASE_URL=http://127.0.0.1:8013 \
SUPERSONIC_API_KEY=local \
node scripts/openai_compat_smoke.mjs
```

## ROCm Profiling

Use the isolated ROCm 7.1.1 profiler/runtime stack on the RX 7900 XTX machine:

```bash
env -u HSA_OVERRIDE_GFX_VERSION \
  LD_LIBRARY_PATH=/opt/rocm-7.1.1/lib \
  SUPERSONIC_BACKENDS=hip \
  /opt/rocm-7.1.1/bin/rocprofv3 \
  --kernel-trace \
  --output-directory target/rocprof_run \
  -- \
  target/release/supersonic ...
```

Counter passes should be narrow. `SQ_WAVES` has been useful; large combined PMC
sets can exceed hardware collection limits and leave a child benchmark process
running.

## External References

External engines are measured through the adapter harness so JSON cells retain
the exact command, engine version, workload settings, samples, median
`ms_per_step`, derived `tok_per_s`, and notes.

```bash
python3 -m oracle.bench.external.external_main \
  --engine llama.cpp \
  --models qwen3.5-35b-a3b \
  --quants q4km \
  --model-dir qwen3.5-35b-a3b=/path/to/qwen3.5-35b-a3b-q4_k_m.gguf \
  --prompt "" \
  --prompt-tokens 0 \
  --context-size 1024 \
  --max-new-tokens 512 \
  --measurement-runs 5
```

```bash
python3 -m oracle.bench.external.external_main \
  --engine mlx-lm \
  --models qwen3.5-35b-a3b \
  --quants q4km \
  --model-dir qwen3.5-35b-a3b=/path/to/mlx/qwen3.5-35b-a3b-q4 \
  --prompt "" \
  --prompt-tokens 0 \
  --context-size 1024 \
  --max-new-tokens 512 \
  --measurement-runs 5
```
