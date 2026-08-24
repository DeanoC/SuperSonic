# Six-Hour Balanced Full Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the full benchmark consume approximately six hours through deterministic balanced measurement rounds while retaining fail-closed timeouts and evidence-backed clock eligibility.

**Architecture:** Extend the suite contract with a minimum duration inside the existing hard budget. The execution layer runs warmups once, then executes one measured invocation for each seeded case/engine entry per round until both the configured minimum repetitions and minimum duration are satisfied; only complete rounds become publishable records. Bundle status records elapsed time and completed rounds. GPU clock eligibility ignores isolated loaded transients but rejects three consecutive loaded out-of-tolerance samples, while every raw sample and all other telemetry checks remain strict.

**Tech Stack:** Python 3 standard library, TOML manifests, unittest, GitHub Actions YAML, Markdown documentation.

**Spec:** `docs/superpowers/specs/2026-08-24-reproducible-benchmark-pages-design.md`, amended by the operator-approved design in the 2026-08-24 first-full-run diagnostic.

## Global Constraints

- The public model remains explicit `qwen3.8-27b` with the project GQH-Q3KXL GGUF and pinned llama.cpp peer.
- Full-suite hard budget remains exactly 21,600 seconds.
- Full-suite minimum duration is 20,700 seconds, reserving 900 seconds for a bounded final round and finalization.
- Each performance subprocess has a 60-second fail-closed timeout in the full suite.
- A record is publishable only after a complete balanced round; incomplete evidence remains diagnostic.
- All raw telemetry is retained. Memory clock, power cap, performance level, temperature, and missing-value checks remain strict.
- Unsupported combinations continue to fail explicitly without compatibility paths.

---

### Task 1: Duration-aware suite contract

**Files:**
- Modify: `tools/benchmark/model.py`
- Modify: `tools/benchmark/manifest.py`
- Modify: `benchmarks/suites/full.toml`
- Modify: `benchmarks/suites/quick.toml`
- Test: `tests/test_benchmark_manifests.py`

**Interfaces:**
- Produces: `SuiteManifest.minimum_duration_seconds: int`.
- Consumes: strict TOML suite parsing and the existing 21,600-second budget.

- [ ] Write tests proving full loads with `minimum_duration_seconds == 20700`, quick loads with zero, and invalid values above the hard budget fail.
- [ ] Run the focused manifest tests and verify the new assertions fail because the field is not supported.
- [ ] Add the strict field to the model/parser and set full/quick values; set every full performance timeout to 60 seconds under the new per-invocation meaning.
- [ ] Run the focused manifest tests and verify they pass.

### Task 2: Balanced round execution and duration evidence

**Files:**
- Modify: `tools/benchmark/execution.py`
- Modify: `tools/benchmark/validation.py`
- Test: `tests/test_benchmark_execution.py`
- Test: `tests/test_benchmark_validation.py`

**Interfaces:**
- Consumes: `SuiteManifest.minimum_duration_seconds` and seeded `ordered_cases()`.
- Produces: balanced per-round sample accumulation, `status.elapsed_seconds`, `status.completed_rounds`, and publishability checks against the configured minimum.

- [ ] Write a deterministic fake-clock execution test proving each entry runs once per round, configured warmups run once, execution continues beyond minimum repetitions until the duration boundary, and all completed records contain the same round count.
- [ ] Write validation tests proving a duration-driven bundle rejects elapsed time below 20,700 seconds and accepts extra complete-round samples when duration evidence is sufficient.
- [ ] Run the focused execution/validation tests and verify they fail for the missing scheduler and status evidence.
- [ ] Refactor case execution into per-invocation state, run seeded entries in balanced rounds, atomically promote records only after a complete round, and persist elapsed/round evidence.
- [ ] Update validation so fixed-duration suites still require exact sample counts while duration-driven suites require at least the configured minimum repetitions plus sufficient elapsed/round evidence.
- [ ] Run focused execution/validation tests and verify they pass.

### Task 3: Sustained loaded clock drift

**Files:**
- Modify: `tools/benchmark/environment.py`
- Test: `tests/test_benchmark_environment.py`

**Interfaces:**
- Consumes: ordered live `TelemetrySample` observations.
- Produces: headline ineligibility after three consecutive loaded GPU samples outside the requested tolerance.

- [ ] Write tests proving one or two loaded clock transients remain recorded but eligible, three consecutive loaded violations fail, an in-band or unloaded sample resets the streak, and sample verification does not duplicate observations.
- [ ] Run the focused environment tests and verify the transient case fails under the current any-sample rule.
- [ ] Implement the three-sample sustained-drift streak and remove the duplicate observation iteration in `snapshot_from_observations`.
- [ ] Run focused environment tests and verify they pass.

### Task 4: Workflow and public reproduction contract

**Files:**
- Modify: `.github/workflows/benchmark-full.yml`
- Modify: `docs/benchmarks.md`
- Modify: `docs/performance.md`
- Modify: `docs/testing.md`
- Test: `tests/test_ci_workflows.py`
- Test: `tests/test_active_docs.py`

**Interfaces:**
- Consumes: the duration-aware full suite and persisted status evidence.
- Produces: operator-facing wording and CI assertions matching the actual six-hour behavior.

- [ ] Write workflow/document tests for the 20,700-second minimum, 21,600-second hard budget, balanced rounds, and sustained-drift wording.
- [ ] Run focused workflow/document tests and verify the new contract assertions fail.
- [ ] Update workflow comments/preflight and public documentation without adding unvalidated performance claims.
- [ ] Run focused workflow/document tests and active-doc validation.

### Task 5: Full verification, publication, and launch

**Files:**
- Verify all changed files.

**Interfaces:**
- Produces: a reviewed commit on `fix/full-benchmark-six-hour`, merged PR, and a manually dispatched full workflow run.

- [ ] Run `python3 -m unittest discover -s tests -v`.
- [ ] Run `cargo test --workspace --all-targets`, `cargo check --workspace --all-targets`, and `cargo fmt --all --check`.
- [ ] Run `python3 tools/check-active-docs.py` and `git diff --check`.
- [ ] Inspect the exact diff and repository status; stage only intended paths and commit.
- [ ] Push, open a non-draft PR as authorized by the operator, merge after available gates, dispatch the full workflow, and verify it reaches the self-hosted runner.
