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
configurable physical GPU ordinal from the bounded AMD SMI static ASIC
record, masks it to test ordinal 0, records selected-device `rocm-smi`, waits
for three idle samples, and performs strict preflight for the canonical GGUF,
optional 8192 GGUF, and model-directory files. Canonical defaults are the
Task 8 runner paths, with repository or organization secret/variable
overrides:

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
that `/dev/kfd` and `/dev/dri` are unavailable here. The two retained
formatting drifts in `crates/core/src/registry.rs` and
`crates/kernel-ffi/src/lib.rs` were mechanically formatted in the fix round;
broader warning cleanup remains Task 11 scope.

## Commit

Required subject:

```text
ci: gate Qwen3.8 GQH on CPU and R9700
```

Commit: `d0f7c77` (`ci: gate Qwen3.8 GQH on CPU and R9700`).

## Task 9 fix round 1

### RED -> GREEN

The review-driven tests were written before the fix implementation and were
run against the Task 9 implementation. The initial RED run reported missing
`select-r9700-device.py`, `parse-rocm-smi.py`, `check-pr-diff.py`, and
`check-active-docs.py`, plus the absent pinned HIP/data-flow workflow
contracts. After implementation, the full Python suite is green:

```text
python3 -m unittest discover -s tests -p 'test_*.py' -v  PASS (37 tests)
```

The behavioral coverage exercises JSON device discovery, ambiguous/missing
device failure, validated overrides, physical-to-logical masking, selected
device utilization parsing, merge-base patch whitespace failure/pass cases,
active-document term/link checks, and workflow ordering/data flow. It does
not rely on workflow string presence alone.

### Fix-round workflow contract

- The CPU job remains on stock `ubuntu-24.04` with no GPU device request. It
  installs the official ROCm 7.2.4 HIP SDK from the versioned AMD apt
  repositories with the signed key and apt origin pin, exports `/opt/rocm/bin`
  through `GITHUB_PATH`, and verifies git, Python/tomllib, cargo/rustfmt, and
  `hipcc --version` before the first cargo command.
- The CPU patch step uses pull-request base/head SHAs and the PR head ref,
  while non-PR dispatches compute `git merge-base`; `tools/check-pr-diff.py`
  checks that actual revision range rather than the working tree.
- The R9700 job never defaults to physical GPU zero. A bounded
  `timeout --foreground 30s amd-smi static --asic --json` probe feeds
  `tools/select-r9700-device.py`; exactly one validated `gfx1201`/R9700 is
  required unless a runner override is supplied and independently validated.
  The selected physical ordinal is exported as `HIP_VISIBLE_DEVICES`, with
  tests using logical device zero. The canonical host's physical ordinal is
  documented as GPU 1, not assumed by code.
- Initial state and every idle probe use selected-device `rocm-smi -d` calls
  under bounded timeout. The idle loop parses both utilization fields through
  `tools/parse-rocm-smi.py` and computes a wall-clock deadline that remains
  effective if a probe hangs.
- `tools/check-active-docs.py` now checks the active public README and six
  product docs for removed CLI/env/backend/model/product-identity terms and
  broken local links; the CPU job invokes it. The stale active prose was
  mechanically quarantined into a compact Qwen3.8/GQH contract so Task 9 is
  green. Task 10 owns richer positioning/history prose and any follow-up
  editorial rewrite.
- The two named Rust formatting drifts were fixed mechanically. Existing
  compiler warnings remain documented Task 11 cleanup, not part of this
  round.

### Fix-round verification

```text
cargo fmt --all --check                         PASS
git diff --check                                PASS
HIP_ARCH=gfx1201 cargo check --workspace --all-targets PASS
python3 tools/check-active-docs.py              PASS
python3 tools/check-support-matrix.py           PASS (2 entries / 2 arches)
python3 tools/check-kernel-groups.py            PASS (2 groups)
python3 tools/check-tool-inventory.py           PASS (1 runner binary)
python3 -m unittest discover -s tests -p 'test_*.py' -v PASS (37 tests)
```

The actual self-hosted R9700 artifact crawl was not run in this container:
`/dev/kfd` and `/dev/dri` are unavailable. No R9700 execution result is
claimed.

Fix-round implementation commit subject remains:

```text
ci: gate Qwen3.8 GQH on CPU and R9700
```

Fix-round implementation commit: `7a7cc23`.

## Task 9 fix round 2

### RED -> GREEN

The second review round began with realistic host-schema and setup-order
tests. The RED run failed because the selector expected architecture fields in
`amd-smi list --json`, the workflow still called that list command, the CPU
job depended on a container, the artifact page overstated preflight checks,
and the active-doc checker missed spaced `Gemma 4`, `Phi 4`, and `Llama 3`
forms. The captured fixture now mirrors the host output shape:

```json
{
  "gpu_data": [
    {"gpu": 0, "asic": {"device_id": "0x744c", "target_graphics_version": "gfx1100"}},
    {"gpu": 1, "asic": {"device_id": "0x7551", "target_graphics_version": "gfx1201"}}
  ]
}
```

GREEN behavior uses only the record `gpu` field for the physical ordinal,
accepts `target_graphics_version`, and validates both discovery and explicit
override against the complete ASIC record. `device_id` and list-only records
cannot become ordinals.

### Fix-round-2 contract

- `qwen38-gfx1201.yml` uses bounded
  `amd-smi static --asic --json`; it no longer expects architecture data from
  `amd-smi list --json`.
- `ci.yml` runs directly on stock `ubuntu-24.04`. Its setup installs the
  signed, versioned ROCm 7.2.4 apt repositories and the exact
  `rocm-hip-sdk=7.2.4.70204-93~24.04` package, verifies the AMD repository key
  fingerprint, adds `/opt/rocm/bin` to `GITHUB_PATH`, then checks git, Python
  >=3.11 with `tomllib`, cargo, rustfmt, and hipcc before dependent commands.
  Checkout and artifact upload use immutable action commit refs; no container
  execution is claimed.
- The CPU pull-request path filter explicitly includes `README.md`, all seven
  active docs, every active validator, and every Python test file used by the
  CPU gate.
- Active-doc rejection now covers spaced product identities. README, build,
  and benchmark gfx1201 examples use validated physical selection guidance;
  none hardcodes `HIP_VISIBLE_DEVICES=0`.
- The artifact page now states the actual cheap preflight scope: existence,
  readability, and required model configuration/tokenizer sidecars. Rust
  startup/artifact tests own GQH header and geometry validation.
- The report's old `rocminfo` wording was corrected to the actual AMD
  SMI/static-ASIC and selected-device `rocm-smi` flow.
- The selector regression fixture includes the host's `device_id` fields and
  proves that only each record's physical `gpu` ordinal is used; nested
  subsystem identifiers cannot override it.

### Fix-round-2 verification

```text
cargo fmt --all --check                         PASS
git diff --check                                PASS
python3 tools/check-active-docs.py              PASS
python3 tools/check-support-matrix.py           PASS (2 entries / 2 arches)
python3 tools/check-kernel-groups.py            PASS (2 groups)
python3 tools/check-tool-inventory.py           PASS (1 runner binary)
python3 -m unittest discover -s tests -p 'test_*.py' -v 43 tests PASS
```

The hosted CPU workflow and self-hosted R9700 workflow were not executed in
this environment. No container or R9700 execution result is claimed.

Fix-round-2 implementation remains under the required subject:

```text
ci: gate Qwen3.8 GQH on CPU and R9700
```

Fix-round-2 implementation commit: `a6a18ed`.
