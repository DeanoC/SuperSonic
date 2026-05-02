# Qwen3.6 Verification Suite

`oracle/qwen36_verify_suite.py` is the first gate for making the native FP8
Qwen3.6-35B-A3B path usable as a verification base for quantized variants.
It does not assume correctness from PyTorch. Instead, it checks whether a
given SuperSonic lane is repeatable enough to become an oracle candidate.

The suite runs deterministic prompts, captures
`SUPERSONIC_QWEN36_DUMP_FINAL_HIDDEN` and `SUPERSONIC_QWEN36_DUMP_LOGITS`,
hashes the dumped BF16 bytes, and fails if identical runs produce different
hidden states, logits, generated ids, or all-zero logits.

## Setup

The harness only needs the Rust binary plus the lightweight Hugging Face
`tokenizers` package:

```bash
python3 -m venv .venv-verify
. .venv-verify/bin/activate
pip install tokenizers
```

For real streamed PG-19 instead of the default synthetic PG-19 text, install
`datasets` in the same venv and pass `--pg19-source dataset`.

## Quick Smoke

This is the smallest repeatability gate for checking that the harness can run
the native FP8 and INT4 lanes and compare repeated outputs:

```bash
. .venv-verify/bin/activate
python oracle/qwen36_verify_suite.py \
  --binary target/release/supersonic \
  --model-dir /models/supersonic-cdna/qwen3.6-35b-a3b-fp8 \
  --backend hip \
  --contexts 8 \
  --families pg19 \
  --modes fp8,int4 \
  --repeats 2 \
  --pg19-source synthetic \
  --out target/qwen36_verify_repeat_smoke.json
```

## gfx942 Sweep

The wrapper builds `supersonic` and runs a broader PG-19 plus RULER-like
repeatability matrix through 2K context. The default excludes 8K because the
current long-context path can exceed the subprocess timeout and is tracked as a
separate performance investigation.

```bash
tests/gfx942/run_qwen36_verify_suite.sh
```

Useful overrides:

```bash
MODEL_DIR=/models/supersonic-cdna/qwen3.6-35b-a3b-fp8 \
CONTEXTS=8,128,512,2K \
FAMILIES=pg19,ruler \
MODES=fp8,int4 \
REPEATS=3 \
TIMEOUT=900 \
PG19_SOURCE=synthetic \
OUT=target/qwen36_verify_results.json \
tests/gfx942/run_qwen36_verify_suite.sh
```

For a real PG-19 run:

```bash
PG19_SOURCE=dataset tests/gfx942/run_qwen36_verify_suite.sh
```

## Long Context Probe

8K is opt-in while the FP8 path is still a timeout/performance investigation:

```bash
tests/gfx942/run_qwen36_verify_long_context.sh
```

The long-context wrapper defaults to `CONTEXTS=8K`, `FAMILIES=pg19`,
`MODES=fp8`, `REPEATS=1`, and `TIMEOUT=1800`. It passes
`--continue-on-error` so timeout failures are written to the result JSON when
possible. INT4 and RULER can be opted in explicitly:

```bash
MODES=fp8,int4 FAMILIES=pg19,ruler tests/gfx942/run_qwen36_verify_long_context.sh
```

## Reading Results

The output JSON has:

- `summary`: per case and mode, including deterministic flags for logits,
  hidden state, and generated ids.
- `runs`: per subprocess run, including dump hashes, generated ids, vector
  stats, timing, and stdout/stderr tails.
- `failures`: the exact gate failures that caused a non-zero exit.

FP8 and INT4 now pass the repeatability gate through 2K context, so FP8 is the
native deterministic base for those verification cases. 8K remains a
performance/timeout blocker until the long-context probe completes.
