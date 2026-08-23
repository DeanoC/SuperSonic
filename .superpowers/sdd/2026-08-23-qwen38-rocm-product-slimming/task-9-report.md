# Task 9 report — establish CPU and R9700 CI

## Scope and outcome

Task 9 adds the deterministic CPU pull-request gate and the serial
self-hosted `gfx1201` R9700 correctness gate. The active support matrix now
has exactly two public rows: HIP/Qwen3.8-27B/custom `gqh-gguf` on `gfx1100`
and `gfx1201`, both naming the `qwen38-gqh-correctness` gate. The tool
inventory contains only the public `supersonic` binary. The previously
deleted `kernel-lab.yml` workflow was not recreated.

## TDD evidence

The support/workflow contract tests were written before the manifests and
workflows were added.

RED runs against base `522c8fa`:

```text
python3 -m unittest tests.test_support_matrix -v
Ran 3 tests: 1 ok, 1 failure, 1 error
  - active matrix had no model_sources/correctness_gate contract
  - invalid backend/model/source was not rejected by the validator

python3 -m unittest tests.test_ci_workflows -v
Ran 3 tests: 1 ok, 2 failures
  - .github/workflows/ci.yml was absent
  - .github/workflows/qwen38-gfx1201.yml was absent
```

GREEN contract evidence:

```text
python3 -m unittest discover -s tests -p 'test_*.py' -v
Ran 25 tests ... OK
```

The new tests cover exact active matrix rows, rejection of non-product
models/backends/sources, CPU GPU-device exclusion, serial R9700 settings,
strict artifact gates, ignored-test handling, ordinary/MTP comparison,
nonblocking telemetry, artifact upload, and the absence of `kernel-lab.yml`.

## Workflow contract

`.github/workflows/ci.yml` runs on ordinary Ubuntu pull-request workers. It
checks patch whitespace and formatting, compiles all workspace targets with
`HIP_ARCH=gfx1201`, runs the CPU-safe GQH/MTP/CLI/startup tests, validates the
support/kernel/tool manifests, checks retained source terms, and runs the
Python validator suite. It does not set `HIP_VISIBLE_DEVICES` or require
`/dev/kfd`/`/dev/dri`.

`.github/workflows/qwen38-gfx1201.yml` runs on exactly
`[self-hosted, linux, rocm, gfx1201]` with a 45-minute timeout. It selects a
configurable physical GPU ordinal, masks it to test ordinal 0, records
`rocminfo`/`rocm-smi`, waits for three idle samples, and performs strict
preflight for the canonical GGUF, optional 8192 GGUF, and model-directory
files. Canonical defaults are the Task 8 runner paths, with repository or
organization secret/variable overrides:

```text
SUPERSONIC_GQH_GGUF=/home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq.gguf
SUPERSONIC_GQH_8192_GGUF=/home/deano/gqh-artifacts/qwen38-gqh-q2kxl-gptq-8192.gguf
SUPERSONIC_QWEN38_MODEL_DIR=/data/models/Qwen3.8-27B
```

The release build and every artifact-dependent cargo test use
`RUST_TEST_THREADS=1`, `--test-threads=1`, and explicit
`--include-ignored`. The workflow runs focused kernel-FFI GQH tests, the
full serial Qwen3.8 artifact crawl, deterministic component decode/chat tests,
and an executable ordinary-vs-MTP token-array comparison. Warmup/median
throughput telemetry is `continue-on-error: true` and runs under
`if: always()`, so it cannot mask a correctness failure. Every machine log,
test log, token comparison, and telemetry summary is uploaded with
`if: always()`.

## Verification

Required local workflow-equivalent commands:

```text
git diff --check                                      PASS
HIP_ARCH=gfx1201 cargo check --workspace --all-targets PASS
python3 tools/check-support-matrix.py                  PASS (2 entries / 2 arches)
python3 tools/check-kernel-groups.py                   PASS (2 groups)
python3 tools/check-tool-inventory.py                  PASS (1 runner binary)
python3 -m unittest discover -s tests -p 'test_*.py' -v PASS (25 tests)
```

Focused retained product tests also pass:

```text
model-store GQH unit tests: 5 passed
kernel-ffi maps_gguf_and_flm_ids: 1 passed
runtime mtp_accept_tests: 3 passed
runner qwen38_cli_contract: 5 passed
runner qwen38_startup_contract: 12 passed
```

Workflow shell blocks and embedded Python snippets were syntax-checked. The
actual R9700 artifact crawl was not run in this container; Task 8 documented
that `/dev/kfd` and `/dev/dri` are unavailable here. `cargo fmt --all
--check` still reports two pre-existing unrelated formatting diffs in
`crates/core/src/registry.rs` and `crates/kernel-ffi/src/lib.rs`; the CPU
workflow retains the required format gate and Task 11 owns the formatting
baseline.

## Commit

Required subject:

```text
ci: gate Qwen3.8 GQH on CPU and R9700
```

Commit: `d0f7c77` (`ci: gate Qwen3.8 GQH on CPU and R9700`).
