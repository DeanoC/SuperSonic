# Benchmarks

The benchmark harness measures the supported Qwen3.8-27B GQH path without
expanding the runner contract. A result is a reviewable candidate first; it
becomes a public number only after validation, correctness, and evidence review.
No measured throughput number is published in this repository yet.

## Tiers and time budgets

The suite manifests own the measurement budgets:

- `quick` has a 600-second hard budget (10 minutes). It is the development and GPU
  smoke candidate, with ordinary and DFlash2 short performance arms plus the
  representative quality cases.
- `full` runs for about six hours: it has a 20,700-second minimum inside a
  21,600-second hard budget. It is
  manually triggered after quick review or for an overnight run, and includes
  the complete quality corpus, the verified cold-load series, MTP and DFlash2
  cases, and the configured peer engine. The seeded case/engine order runs in
  balanced rounds until the minimum is reached; the reserved 900 seconds bound the final
  round and manifest finalization.

The GitHub Actions job caps include checkout, host probes, artifact preflight,
and the release build in addition to the harness. The quick workflow cap is
30 minutes and the full workflow cap is 450 minutes. The harness budgets and
the workflow caps are different limits; neither changes the other. A full run
is `workflow_dispatch` only and is not part of push or pull-request CI.

## Read-only host preparation

The host operator may prepare clocks, power, and performance level outside the
repository. The harness does not apply privileged clock, power, or operating-
system cache changes. It reads the configured state and fails closed when a
locked request cannot be verified.

Capture the static AMD SMI evidence used for physical-device provenance before
running the CLI:

```bash
set -euo pipefail
export BENCHMARK_OUTPUT_ROOT=target/benchmarks/manual
mkdir -p "$BENCHMARK_OUTPUT_ROOT"
timeout --foreground 30s amd-smi static --asic --bus --json \
  > "$BENCHMARK_OUTPUT_ROOT/amd-smi-static-asic-bus.json"
timeout --foreground 30s amd-smi list -e --json \
  > "$BENCHMARK_OUTPUT_ROOT/amd-smi-enumeration.json"
python3 tools/merge-amd-smi-provenance.py \
  --asic-bus "$BENCHMARK_OUTPUT_ROOT/amd-smi-static-asic-bus.json" \
  --enumeration "$BENCHMARK_OUTPUT_ROOT/amd-smi-enumeration.json" \
  --output "$BENCHMARK_OUTPUT_ROOT/amd-smi-provenance.json"
```

Use the selector in the [README](../README.md) with that merged record. Export
the selected `SUPERSONIC_R9700_GPU_ID`, `SUPERSONIC_R9700_GPU_ARCH`,
`SUPERSONIC_GPU_LOGICAL`, `HIP_VISIBLE_DEVICES`, and `SUPERSONIC_DEVICE`
values from its validated output. The physical ordinal is retained for SMI
evidence; masking it makes the selected device logical `0` to the process.
Do not assume physical GPU zero or use `eval` on selector output.

Capture the bounded toolchain identities after selecting the physical device;
the CLI consumes these files and stores only normalized values in records:

```bash
timeout --foreground 30s hipcc --version \
  | tee "$BENCHMARK_OUTPUT_ROOT/hipcc-version.txt"
timeout --foreground 30s rocm-smi -d "$SUPERSONIC_R9700_GPU_ID" \
  --showdriverversion | tee "$BENCHMARK_OUTPUT_ROOT/rocm-driver-version.txt"
```

## Release-build prerequisite

Before either local quick or full run, build the canonical release workspace
with the same HIP target used by the selected device:

```bash
HIP_ARCH=gfx1201 cargo build --release --workspace
```

The harness invokes that release binary; it does not compile a different
development profile implicitly.

## Quick candidate

The following is the implemented CLI shape used by the quick workflow. All
paths and clock values are explicit inputs; the static JSON is required by the
CLI and is the authoritative provenance source. The DFlash2 arm requires the
canonical `qwen38-dflash2-q8_0-canonical.gguf` drafter; do not substitute the
similarly named vendor or converted drafters. DFlash2 results record the
active width-16 verify block and are not comparable with obsolete width-8
evidence.

```bash
set -euo pipefail
run_id="quick-manual-$(date +%s)"
RUST_TEST_THREADS=1 timeout --foreground 660s \
  python3 tools/supersonic-bench.py run \
  --suite quick \
  --model-dir "$SUPERSONIC_QWEN38_MODEL_DIR" \
  --artifact "$SUPERSONIC_GQH_GGUF" \
  --draft-artifact "$SUPERSONIC_DFLASH_DRAFT_GGUF" \
  --draft-artifact-semantic-id qwen3.8-27b-dflash2-q8-canonical \
  --draft-artifact-quantization Q8_0 \
  --artifact-semantic-id qwen3.8-27b-gqh-q3kxl-hf-91bc7e33 \
  --artifact-quantization GQH-Q3KXL \
  --tokenizer-sha256 0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3 \
  --chat-template-sha256 c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041 \
  --physical-gpu "$SUPERSONIC_R9700_GPU_ID" \
  --gpu-static-json "$BENCHMARK_OUTPUT_ROOT/amd-smi-provenance.json" \
  --rocm-version-file "$BENCHMARK_OUTPUT_ROOT/rocm-driver-version.txt" \
  --hip-version-file "$BENCHMARK_OUTPUT_ROOT/hipcc-version.txt" \
  --logical-gpu "$SUPERSONIC_GPU_LOGICAL" \
  --gpu-arch "$SUPERSONIC_R9700_GPU_ARCH" \
  --device "$SUPERSONIC_DEVICE" \
  --chat \
  --clock-policy locked \
  --gpu-clock-mhz "$SUPERSONIC_BENCHMARK_GPU_CLOCK_MHZ" \
  --gpu-clock-tolerance-mhz "$SUPERSONIC_BENCHMARK_GPU_CLOCK_TOLERANCE_MHZ" \
  --memory-clock-mhz "$SUPERSONIC_BENCHMARK_MEMORY_CLOCK_MHZ" \
  --power-cap-watts "$SUPERSONIC_BENCHMARK_POWER_CAP_WATTS" \
  --performance-level "$SUPERSONIC_BENCHMARK_PERFORMANCE_LEVEL" \
  --seed 1 \
  --run-id "$run_id" \
  --output target/benchmarks/candidate
```

The `660s` wrapper leaves time for process cleanup around the exact 600-second
suite budget. The workflow stores the candidate under
`target/benchmarks/quick/candidate/<run-id>` and uploads that directory, logs,
and host diagnostics even when a later step fails. Validate the exact bundle,
not a hand-selected JSON file:

```bash
bundle=target/benchmarks/candidate/<run-id>
python3 tools/supersonic-bench.py validate --publishable "$bundle"
```

## Full candidate

The full workflow is the same CLI with the full suite and its separate pinned
peer engine. Before performance, it runs the complete serial `gfx1201`
artifact gate from [Testing](testing.md#gfx1201-artifact-gate) and
reverifies that the GPU is idle. Both engines consume the exact local copy of
[Qwen3.8-27B-GQH-Q3KXL.gguf](https://huggingface.co/Geometric-AI/Qwen3.8-27B-GQH-Q3KXL-GGUF/blob/91bc7e33c1912856dcd8d2ca4499dd8ccad13ac4/Qwen3.8-27B-GQH-Q3KXL.gguf),
verified as SHA-256
`c710b03bf5bf224107d0ae1567b97f1c8638ef35c5f431c39479a3ecc963bd98`.
Do not substitute the similarly named `qwen38-gqh-32gb-a.gguf`; it is a
different artifact. Before a local full run, `llama-server` must be on `PATH`, must
match the first non-comment line in `tools/external/llama-cpp-version.txt`,
and the peer artifact must be readable:

```bash
export SUPERSONIC_GQH_GGUF=/home/deano/models/qwen38-gqh-shaped.gguf
export SUPERSONIC_LLAMA_CPP_ARTIFACT="$SUPERSONIC_GQH_GGUF"
export SUPERSONIC_GQH_GGUF_SHA256=c710b03bf5bf224107d0ae1567b97f1c8638ef35c5f431c39479a3ecc963bd98
export SUPERSONIC_LLAMA_CPP_ARTIFACT_SHA256="$SUPERSONIC_GQH_GGUF_SHA256"
command -v llama-server
test -r "$SUPERSONIC_LLAMA_CPP_ARTIFACT"
test "$(sha256sum "$SUPERSONIC_GQH_GGUF" | awk '{print $1}')" = "$SUPERSONIC_GQH_GGUF_SHA256"
pinned_peer_version="$(grep -v '^#' tools/external/llama-cpp-version.txt | grep -v '^$' | head -n 1)"
test "$(llama-server --version 2>&1 | head -n 1)" = "$pinned_peer_version"
```

The one-shot peer adapter starts a fresh `llama-server` for every invocation,
passes `cache_prompt=false`, disables the server warmup, reads exact JSON token
counts and timings, and stops the server before returning. The workflow supplies
the actual peer path and binary pin. The local command
must pass the exact readable artifact that the preflight checked:
`--peer-artifact "$SUPERSONIC_LLAMA_CPP_ARTIFACT"`.

Performance cases ignore EOS so every measured sample decodes the declared
token count. Quality cases instead honor EOS and treat `max_new_tokens` as a
cap, matching ordinary answer generation while retaining deterministic greedy
sampling. Both engines explicitly disable Qwen3.8 thinking in the shared chat
template for these cases.

```bash
set -euo pipefail
run_id="full-manual-$(date +%s)"
RUST_TEST_THREADS=1 timeout --foreground 21660s \
  python3 tools/supersonic-bench.py run \
  --suite full \
  --model-dir "$SUPERSONIC_QWEN38_MODEL_DIR" \
  --artifact "$SUPERSONIC_GQH_GGUF" \
  --draft-artifact "$SUPERSONIC_DFLASH_DRAFT_GGUF" \
  --draft-artifact-semantic-id qwen3.8-27b-dflash2-q8-canonical \
  --draft-artifact-quantization Q8_0 \
  --peer-artifact "$SUPERSONIC_LLAMA_CPP_ARTIFACT" \
  --artifact-semantic-id qwen3.8-27b-gqh-q3kxl-hf-91bc7e33 \
  --artifact-quantization GQH-Q3KXL \
  --tokenizer-sha256 0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3 \
  --chat-template-sha256 c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041 \
  --physical-gpu "$SUPERSONIC_R9700_GPU_ID" \
  --gpu-static-json "$BENCHMARK_OUTPUT_ROOT/amd-smi-provenance.json" \
  --rocm-version-file "$BENCHMARK_OUTPUT_ROOT/rocm-driver-version.txt" \
  --hip-version-file "$BENCHMARK_OUTPUT_ROOT/hipcc-version.txt" \
  --logical-gpu "$SUPERSONIC_GPU_LOGICAL" \
  --gpu-arch "$SUPERSONIC_R9700_GPU_ARCH" \
  --device "$SUPERSONIC_DEVICE" \
  --chat \
  --clock-policy locked \
  --gpu-clock-mhz "$SUPERSONIC_BENCHMARK_GPU_CLOCK_MHZ" \
  --gpu-clock-tolerance-mhz "$SUPERSONIC_BENCHMARK_GPU_CLOCK_TOLERANCE_MHZ" \
  --memory-clock-mhz "$SUPERSONIC_BENCHMARK_MEMORY_CLOCK_MHZ" \
  --power-cap-watts "$SUPERSONIC_BENCHMARK_POWER_CAP_WATTS" \
  --performance-level "$SUPERSONIC_BENCHMARK_PERFORMANCE_LEVEL" \
  --seed 1 \
  --run-id "$run_id" \
  --output target/benchmarks/candidate
```

The `21660s` wrapper bounds cleanup around the 20,700-second minimum and
21,600-second hard budget. Each full performance subprocess has a 60-second
fail-closed timeout. The manual full workflow validates its candidate for diagnosis
and uploads it even if the harness is incomplete. An incomplete, failed, or
quality-failed bundle is diagnostic only and cannot be promoted.

## Cache and clock terminology

Cache state is part of the series identity. The executable suites currently
support only `cold-load`: every sample uses a fresh process and reports model
loading separately from decode. `warm-resident` fails preflight until an
adapter can warm and measure the same resident process. Prefix-cache states
remain rejected until their transitions are verified. A filesystem cache
flush is never claimed unless its mechanism and verification are attached to
the evidence.

For a focused investigation of the observed fresh-process decode bimodality,
run the bounded repeatability diagnostic:

```bash
HIP_VISIBLE_DEVICES=1 python3 tools/supersonic-bench.py repeatability \
  --model-dir /data/models/Qwen3.8-27B \
  --artifact /home/deano/models/qwen38-gqh-shaped.gguf \
  --physical-gpu 1 \
  --output target/benchmarks/repeatability/<run-id>
```

The persistent-stage threshold is a diagnostic trigger, not a performance
gate. Every invocation retains raw streams and live clock, power, temperature,
and utilization telemetry. A token mismatch fails immediately. On the first
slow sample, the exact trigger logs are preserved and bounded rocprof follow-up
reproductions collect kernel and allocation traces. Those traces are labeled
`followup-reproduction` because profiling cannot be attached retroactively to
the process that triggered them. The default wall-clock ceiling is 21,600
seconds (six hours), independent of the per-process and run-count limits.
`HIP_VISIBLE_DEVICES` must map the recorded physical GPU to the selected logical
device; a mismatched or missing mapping fails before the first sample. The
manifest explicitly records cold-load, fresh-process, non-reuse semantics and
is updated after every sample. At least one parseable rocprof JSON trace is
required before the terminal state can be `slow-captured`.

The schema also names `prefix-cache-empty`, `prefix-cache-populated`, and
`prefix-cache-reset`. Those cases are explicitly unsupported by the current
execution boundary until adapter transitions are verified. Do not add them to a
candidate or describe a prefix-cache transition as measured before that gate
exists.

`locked` means the host operator prepared the requested clock and power state.
The harness retains every observation and reports GPU clock drift when
three consecutive loaded samples are outside the recorded tolerance. It also verifies
memory clock, power cap, and performance level strictly
before, during, and after each measured case. Idle edge samples are retained
but are not compared to the nominal GPU clock because RDNA power gating lowers
the instantaneous sclk while idle. `uncontrolled-clocks` records retain observed telemetry for
diagnosis but are excluded from headline performance and peer speedup claims.

## Evidence and review

Every candidate record retains the commit and dirty state, engine/version,
ROCm/HIP versions, static physical GPU provenance, logical mapping, artifact
identity and digest, DFlash2 drafter identity and digest for DFlash2 records,
prompt/workload, cache/process state, clock evidence,
correctness, ordinary-versus-MTP equality, DFlash2 semantic quality, and raw
measured samples in
measurement order. The validator validates raw sample values, the suite-required
or balanced-round sample count, and bundle completeness. The renderer deterministically derives sample
count, median, minimum, maximum, and median absolute deviation (MAD) from
those validated raw samples; no stored summary replaces the raw source.

The reviewer runs validation and, for a peer comparison, the implemented
comparator. A mismatch remains visible with reasons but produces no speedup:

```bash
python3 tools/supersonic-bench.py validate --publishable <candidate-bundle>
python3 tools/supersonic-bench.py compare <record-a> <record-b> \
  --output target/benchmarks/candidate/comparison.json
```

Peer artifacts are usually noncomparable by digest under the comparison
ruling. A peer result may remain useful context, but unlike artifact digests,
tokenizer/template identity, cache state, clocks, workload, or timing boundary
must never become a headline ratio.

## Promotion and Pages bootstrap

Promotion is a code-reviewed change. After the candidate passes
`validate --publishable` and its quality/evidence review, copy only the
portable manifest and result records into `benchmarks/results/`. Do not copy
raw logs, SMI dumps, absolute model/artifact paths, or a candidate-local
comparison file. Then validate the committed source:

```bash
python3 tools/supersonic-bench.py validate --publishable benchmarks/results
python3 tools/supersonic-bench.py render benchmarks/results target/benchmarks/site
```

The Pages workflow is configured for the GitHub Actions source in repository
Settings → Pages. It validates records before rendering and deploys only from
the default branch after that validation. A zero-baseline checkout is expected:
when `benchmarks/results/` contains no JSON records, Pages reports that no
committed baseline exists and skips validation, rendering, upload, and deploy.
The first reviewed baseline commit on the default branch bootstraps the site;
there is no placeholder or synthetic performance number.
