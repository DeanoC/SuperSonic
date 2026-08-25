# Scalar-Control-llama.cpp Six-Hour Qualification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, non-publishable six-hour candidate bundle comparing the contributor-only SuperSonic scalar head, production-WMMA control, and pinned llama.cpp on the approved high-quality Qwen3.8 artifact family.

**Architecture:** Extend the existing benchmark engine identity and adapter layer instead of duplicating its six-hour scheduler, telemetry, cache, validation, and renderer. Add a feature-gated contributor-only scalar executable whose route is fixed in source, give production WMMA and scalar distinct engine identities, and run both beside the already pinned GeometricAGI llama.cpp peer. The workflow renders and validates candidate pages but never promotes records or changes the public runner.

**Tech Stack:** Rust/Cargo, Python benchmark tooling, TOML/JSON Schema, GitHub Actions self-hosted ROCm runner, AMD SMI/ROCm SMI, llama.cpp HTTP adapter.

**Spec:** `docs/superpowers/specs/2026-08-25-deterministic-raw-q6-output-head-design.md`

## Global Constraints

- Candidate and peer artifact: Hugging Face repository `Geometric-AI/Qwen3.8-27B-GQH-Q3KXL-GGUF` at immutable revision `91bc7e33c1912856dcd8d2ca4499dd8ccad13ac4`, file `Qwen3.8-27B-GQH-Q3KXL.gguf`, exactly 13,440,110,432 bytes, SHA-256 `c710b03bf5bf224107d0ae1567b97f1c8638ef35c5f431c39479a3ecc963bd98`. All three engines must use this same file; a different peer artifact makes the run diagnostic-only and non-promotable.
- Peer engine: `GeometricAGI/llama.cpp` PR 1 head `f8dd7c36da283cf587cef3133b9287fd3a5b6fdb`, whose version line is `version: 5 (f8dd7c3)`.
- Keep the public `supersonic` CLI and default production-WMMA route unchanged. The scalar route is contributor-only, feature-gated, and fixed in source.
- Use deterministic local tasks, greedy decoding, exact prompt/template identity, and recorded generated output. Never substitute a cloud evaluator.
- Run balanced complete rounds for at least 20,700 seconds inside the existing 21,600-second suite budget; the workflow ceiling remains 27,000 seconds.
- Every invocation records physical/logical GPU mapping, clocks, memory clock, power cap, performance level, temperature, raw telemetry, process/cache state, timestamps, commit, engine pin, and artifact digest.
- `cold-load` means fresh process with no claimed filesystem flush. Never compare unlike cache states.
- Before the six-hour run, collect a separate seven-round locked hardened-scalar baseline on the same commit series, ROCm/HIP toolchain, audited instruction fingerprint, artifact, GPU, clock/power policy, workload, cache state, and timing boundary. The candidate median may regress by at most 5%; otherwise qualification fails.
- Set `AMDSMI_GPU_METRICS_CACHE_MS=0` for telemetry. Record `throttle_status`, `indep_throttle_status`, and the decoded throttle label verbatim as diagnostic evidence, but do not use those fields alone to accept or reject a run. Eligibility is decided from observed loaded-clock stability plus the strict memory-clock, power-cap, performance-level, temperature, device, process-idle, cache-state, completeness, and timing-dispersion gates below.
- Render and validate candidate evidence only. Publication, tagging, scalar promotion, and removal of WMMA require a later reviewed decision.

---

### Task 1: Add a source-fixed contributor scalar adapter

**Files:**
- Modify: `crates/runtime/Cargo.toml`
- Modify: `crates/runtime/src/decode_engine.rs`
- Modify: `crates/runtime/src/prefill_engine.rs`
- Create: `crates/runtime/examples/scalar_head_lab.rs`
- Create: `tools/supersonic-scalar-lab.py`
- Create: `tests/test_supersonic_scalar_lab.py`

**Interfaces:**
- Consumes: the `scalar-head-lab` feature, `DecodeEngine::set_scalar_head_lab_route(ScalarHeadLabRoute::RawQ6Scalar)`, paired model/artifact paths, chat flag, prompt, and max-new-tokens.
- Produces: exactly one `[supersonic_json]` record matching `tools.benchmark.adapters.ParsedOutput`; the executable accepts only `--mode ordinary|mtp`, has no head-route option, and cannot select WMMA.

- [ ] **Step 1: Write RED tests** asserting bounded argv construction, paired artifact inputs, greedy-only decoding, chat propagation, ordinary and MTP mode dispatch, one canonical output record, finite/consistent timings, exact generated token IDs/text, timeout cleanup, and rejection of any `--route` or route environment variable.
- [ ] **Step 2: Run RED.**

```bash
python3 -m unittest tests.test_supersonic_scalar_lab -v
```

Expected: fail because the wrapper and example do not exist.

- [ ] **Step 3: Implement the minimal example and wrapper.** The Rust example must be declared with `required-features = ["scalar-head-lab"]`, use the same tokenizer/chat/prefill/decode and ordinary/NextN state transitions as runtime tests, set `RawQ6Scalar` in source before prefill, and reject non-greedy settings. Thread the feature-gated route into prefill `compute_greedy_for_range` so the initial and MTP-append token heads use raw-Q6 scalar plus `argmax_f32_as_bf16_rows`. When scalar+MTP is selected, force `run_mtp_spec_round_sequential`; never call the fused multirow verifier's low-bit head. Every sequential verification step already reaches `decode_step_single_kernel_impl` and must assert the scalar route. Add route-trace counters for initial prefill, append, sequential verify, and ordinary decode; the ignored artifact test fails unless all expected counters are nonzero and the fused counter is zero. Emit mode, load/prefill/decode/total times, exact tokens, and text. The Python wrapper uses argv lists, a hard timeout, normalized JSON, and child-process-group cleanup. `--mode` changes only ordinary versus MTP generation; the scalar output-head route remains source-fixed.
- [ ] **Step 4: Prove source-fixed behavior and normal-build isolation.**

```bash
cargo test -p supersonic-runtime --lib
cargo test -p supersonic-runtime --features scalar-head-lab --lib
HIP_ARCH=gfx1201 cargo check -p supersonic-runtime --features scalar-head-lab --example scalar_head_lab
HIP_ARCH=gfx1100 cargo check --workspace --all-targets
python3 -m unittest tests.test_supersonic_scalar_lab -v
```

Expected: all pass; normal runner artifacts contain neither the lab example entry point nor a route selector.

- [ ] **Step 5: Run one strict artifact smoke, review exact tokens, and commit.**

```bash
SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1 HIP_VISIBLE_DEVICES=1 HIP_ARCH=gfx1201 \
timeout --foreground 120s python3 tools/supersonic-scalar-lab.py \
  --model-dir /data/models/Qwen3.8-27B \
  --artifact /home/deano/models/qwen38-gqh-shaped.gguf \
  --prompt Hello --chat --max-new-tokens 32 --device 0
git add crates/runtime/Cargo.toml crates/runtime/src/decode_engine.rs crates/runtime/src/prefill_engine.rs crates/runtime/examples/scalar_head_lab.rs tools/supersonic-scalar-lab.py tests/test_supersonic_scalar_lab.py
git commit -m "feat(bench): add fixed scalar lab adapter"
```

### Task 2: Give scalar and WMMA distinct benchmark identities

**Files:**
- Create: `benchmarks/engines/supersonic-wmma.toml`
- Create: `benchmarks/engines/supersonic-scalar-lab.toml`
- Modify: `benchmarks/engines/supersonic.toml`
- Modify: `tools/benchmark/adapters.py`
- Modify: `tools/benchmark/execution.py`
- Modify: `tools/benchmark/manifest.py`
- Modify: `benchmarks/schema/result-v1.schema.json`
- Modify: `tests/test_benchmark_adapters.py`
- Modify: `tests/test_benchmark_execution.py`
- Modify: `tests/test_benchmark_manifests.py`

**Interfaces:**
- Consumes: engine manifests and the source-fixed scalar adapter.
- Produces: stable engine names `supersonic-wmma`, `supersonic-scalar-lab`, and `llama-cpp`, each with exact binary/version evidence and independent records.

- [ ] **Step 1: Write RED manifest/adapter tests.** Assert that WMMA uses `./target/release/supersonic`, scalar uses `tools/supersonic-scalar-lab.py`, both require the same artifact, both support `ordinary` and `mtp`, and neither output-head identity can be inferred from output text or an environment selector.
- [ ] **Step 2: Run RED.**

```bash
python3 -m unittest tests.test_benchmark_manifests tests.test_benchmark_adapters tests.test_benchmark_execution -v
```

- [ ] **Step 3: Implement exact identity dispatch.** Replace string assumptions that only `supersonic` exists with an allowlisted engine-kind field or explicit name mapping; preserve the existing record schema fields `engine.name` and `engine.version`; fail on unknown SuperSonic variants.
- [ ] **Step 4: Add comparability and artifact-provenance tests.** Scalar and WMMA may compare only when artifact, tokenizer/template, workload, cache, timing boundary, GPU, clock/power policy, and correctness match. Extend each result's closed `artifact` object with required `source_repository`, `source_revision`, `filename`, and integer `size_bytes`; validate these before execution and persist them for every engine. llama.cpp comparisons additionally require the same exact artifact digest/size/revision and compatible quality case; exact token equality is not required when tokenizer formats differ.
- [ ] **Step 5: Run, check, and commit.**

```bash
python3 -m unittest tests.test_benchmark_manifests tests.test_benchmark_adapters tests.test_benchmark_execution tests.test_benchmark_compare -v
git diff --check
git add benchmarks/engines benchmarks/schema/result-v1.schema.json tools/benchmark tests/test_benchmark_adapters.py tests/test_benchmark_execution.py tests/test_benchmark_manifests.py
git commit -m "feat(bench): distinguish scalar and wmma engines"
```

### Task 3: Define the three-engine deterministic qualification suite

**Files:**
- Create: `benchmarks/suites/full-scalar-qualification.toml`
- Reuse: `benchmarks/quality/v2.json`
- Create: `benchmarks/quality/scalar-mtp-goldens-v1.json`
- Create: `tools/generate-scalar-mtp-goldens.py`
- Create: `tests/test_generate_scalar_mtp_goldens.py`
- Modify: `tools/external/llama-cpp-version.txt`
- Modify: `tests/test_benchmark_manifests.py`
- Modify: `tests/test_benchmark_execution.py`
- Modify: `tests/test_benchmark_validation.py`
- Modify: `tools/benchmark/quality.py`
- Modify: `tests/test_benchmark_quality.py`

**Interfaces:**
- Consumes: three engine identities, the existing duration-round scheduler, quality scorers, cache evidence, and peer artifact inputs.
- Produces: a six-hour suite with complete balanced rounds and deterministic quality gates.

- [ ] **Step 1: Write RED tests** requiring `minimum_duration_seconds=20700`, `budget_seconds=21600`, all ordinary performance cases scheduled once per engine per round, alternating seeded order, seven minimum repetitions, no partial-round publication, and explicit ordinary and MTP correctness cases for both SuperSonic identities. llama.cpp remains ordinary-only.
- [ ] **Step 2: Retain and enumerate quality v2 exactly.** The suite uses all 16 existing prompts below with greedy decoding and seed 1. Run cases 1–14 in ordinary mode on all three engines. Run cases 15–16 as ordinary-versus-MTP token equality separately for `supersonic-wmma` and `supersonic-scalar-lab`; each engine's two modes must equal its own reviewed artifact-backed golden. A missing scalar MTP result fails qualification. The semantic numeric arrays currently present in quality v2 are not tokenizer token-ID goldens and must not be used for this gate.

| # | ID | Exact prompt | Max | Scorer / expected |
| ---: | --- | --- | ---: | --- |
| 1 | `instruction-following-1` | `Reply with exactly the word ready.` | 4 | exact text `ready` |
| 2 | `instruction-following-2` | `Reply with exactly the phrase benchmark stable.` | 6 | exact text `benchmark stable` |
| 3 | `structured-extraction-1` | `Return JSON with keys city and gpu for: city=Sofia gpu=gfx1201.` | 24 | JSON `{"city":"Sofia","gpu":"gfx1201"}` |
| 4 | `structured-extraction-2` | `Return only compact JSON, with no markdown or code fences, for engine=supersonic and mode=ordinary.` | 24 | JSON `{"engine":"supersonic","mode":"ordinary"}` |
| 5 | `arithmetic-and-reasoning-1` | `What is 19 + 23? Reply with digits only.` | 4 | exact text `42` |
| 6 | `arithmetic-and-reasoning-2` | `What is 6 * 7? Reply with digits only.` | 4 | exact text `42` |
| 7 | `code-completion-1` | `Complete Python assignment answer = 42. Reply with only the text to the right of the equals sign, digits only.` | 8 | exact text `42` |
| 8 | `code-completion-2` | `Complete this Rust statement: return 42; Reply with only 42; and no markdown or explanation.` | 8 | exact text `42;` |
| 9 | `long-context-retrieval-1` | `Context: alpha beta gamma delta epsilon zeta eta theta. Which word follows gamma? Reply with one word.` | 4 | exact text `delta` |
| 10 | `long-context-retrieval-2` | `Context: warm cache cold load reproducible evidence strict schema. Which word follows warm? Reply with one word.` | 4 | exact text `cache` |
| 11 | `chat-template-behavior-1` | `System: concise. User: Reply with ok.` | 4 | exact text `ok` |
| 12 | `chat-template-behavior-2` | `System: terse. User: Reply with done.` | 4 | exact text `done` |
| 13 | `repeated-run-determinism-1` | `Reply with the digits 1234.` | 4 | exact text `1234` |
| 14 | `repeated-run-determinism-2` | `Reply with the digits 5678.` | 4 | exact text `5678` |
| 15 | `ordinary-vs-mtp-token-equality-1` | `Reply with the token ids for 1 2 3.` | 8 | engine-specific exact token vector frozen by Step 4 |
| 16 | `ordinary-vs-mtp-token-equality-2` | `Reply with the token ids for 4 5 6.` | 8 | engine-specific exact token vector frozen by Step 4 |

- [ ] **Step 3: Bind artifact semantics.** Require semantic ID `qwen3.8-27b-gqh-q3kxl-hf-91bc7e33`, revision `91bc7e33c1912856dcd8d2ca4499dd8ccad13ac4`, filename `Qwen3.8-27B-GQH-Q3KXL.gguf`, quantization `GQH-Q3KXL`, size `13440110432`, and SHA-256 `c710b03bf5bf224107d0ae1567b97f1c8638ef35c5f431c39479a3ecc963bd98` for every engine. Reject a directory, symlink escape, missing file, wrong digest, or any engine silently using another path.
- [ ] **Step 4: Generate and review engine-specific token goldens.** After Tasks 1–2 are built, run `tools/generate-scalar-mtp-goldens.py` in strict-artifact mode for cases 15–16. For each SuperSonic engine, execute ordinary mode twice in independent fresh processes and require the two exact token vectors to agree, stay within the case token cap, and retain prompt bytes/SHA-256, chat/template/tokenizer digests, artifact identity, engine binary/code-object digest, and full stdout/stderr. Write `scalar-mtp-goldens-v1.json` only after all four repeated pairs agree; manually review the generated text and vectors before staging. Goldens are keyed by engine and case, so WMMA and scalar need not equal each other.
- [ ] **Step 5: Make equality three-way and engine-specific.** Change `score_mtp_pair(case, ordinary, mtp, expected_tokens)` so a pair passes only when `ordinary.token_ids == mtp.token_ids == expected_tokens` from the matching engine/case golden. Store distinct quality result keys `<engine>/<case>/ordinary-vs-mtp`; reject a missing/mismatched golden, duplicate/missing WMMA or scalar pair, or any digest mismatch. Tests must show a mutually equal but golden-wrong pair fails and that v2's semantic numeric arrays are never substituted as token IDs.
- [ ] **Step 6: Pin peer source exactly.** Keep the non-comment pin line `version: 5 (f8dd7c3)` and require workflow evidence for full commit `f8dd7c36da283cf587cef3133b9287fd3a5b6fdb` from `https://github.com/GeometricAGI/llama.cpp/pull/1`.
- [ ] **Step 7: Run and commit.**

```bash
python3 -m unittest tests.test_benchmark_manifests tests.test_benchmark_execution tests.test_benchmark_validation tests.test_benchmark_quality tests.test_generate_scalar_mtp_goldens -v
git add benchmarks/suites/full-scalar-qualification.toml benchmarks/quality/scalar-mtp-goldens-v1.json tools/generate-scalar-mtp-goldens.py tools/external/llama-cpp-version.txt tools/benchmark/quality.py tests/test_benchmark_manifests.py tests/test_benchmark_execution.py tests/test_benchmark_validation.py tests/test_benchmark_quality.py tests/test_generate_scalar_mtp_goldens.py
git commit -m "feat(bench): define scalar qualification suite"
```

### Task 4: Add the manual overnight candidate workflow

**Files:**
- Create: `.github/workflows/benchmark-scalar-qualification.yml`
- Modify: `tests/test_ci_workflows.py`
- Modify: `tools/benchmark/environment.py`
- Modify: `tools/benchmark/execution.py`
- Modify: `benchmarks/schema/result-v1.schema.json`
- Create: `benchmarks/schema/qualification-v1.schema.json`
- Modify: `tests/test_benchmark_environment.py`
- Modify: `tests/test_benchmark_execution.py`
- Modify: `tests/test_benchmark_validation.py`
- Modify: `docs/benchmarks.md`
- Modify: `docs/testing.md`
- Modify: `tests/test_active_docs.py`

**Interfaces:**
- Consumes: the qualification suite, self-hosted `gfx1201`, explicit artifact/version secrets, and existing `supersonic-bench.py run/validate/render` commands.
- Produces: a `workflow_dispatch`-only candidate bundle and rendered preview artifact; no Pages deployment.

- [ ] **Step 1: Write RED workflow/schema tests** for manual-only trigger, concurrency lock, 27,000-second ceiling, disk/idle/device preflight, exact model/peer paths and SHA inputs, exact llama commit/version, a comparable hardened-scalar baseline bundle and digest, the <=5% median regression check, closed-schema diagnostic throttle evidence, loaded-clock stability and timing-dispersion gates, locked memory-clock/power/performance inputs, `AMDSMI_GPU_METRICS_CACHE_MS=0`, `RUST_TEST_THREADS=1`, candidate upload on `always()`, and no Git mutation or Pages permission.
- [ ] **Step 2: Run RED.**

```bash
python3 -m unittest tests.test_ci_workflows tests.test_benchmark_environment tests.test_benchmark_execution tests.test_benchmark_validation tests.test_active_docs -v
```

- [ ] **Step 3: Implement host-prepared clock/cache/throttle evidence.** Add `AMDSMI_GPU_METRICS_CACHE_MS` to the exact environment allowlist and require value `0`. Extend `ObservedTelemetry`, `TelemetrySample`, environment JSON, and the closed result schema with `throttle_status: int|null`, `indep_throttle_status: int|null`, and `throttle_label: "THROTTLED"|"UNTHROTTLED"|null`; parse and retain raw AMD SMI JSON rather than inferring from temperature. The workflow only reads/verifies settings and declares `cold-load`, `process_reuse=false`, `filesystem_flush=unavailable` unless a separately verified flush capability exists. Throttle fields remain visible diagnostics and never veto a run by themselves. Sample throughout every measured process; require at least one loaded sample, fail on three consecutive loaded GPU-clock samples outside tolerance, and retain the loaded-clock minimum, median, maximum, and raw series. Memory clock, power cap, performance level, temperature, device mapping, competing-process, missing-field, cache-state, and completeness checks remain strict. Reject a seven-sample baseline or candidate series whose per-token MAD exceeds 3% of its median.
- [ ] **Step 4: Implement the baseline contract.** Accept an explicit `--baseline-bundle target/benchmarks/scalar-baselines/<baseline-run-id>` plus `--baseline-bundle-sha256`. Define the directory digest as SHA-256 over sorted UTF-8 relative paths, a NUL byte, each file's raw SHA-256 bytes, and a final NUL byte. The baseline must contain seven accepted `RawQ6Scalar` cold-load samples and a `baseline-v1.json` binding commit, HIP/ROCm/compiler versions, scalar instruction digest, artifact identity, prompt SHA, timed boundary `lm_head_ms/timed_decode_steps`, GPU/BDF, full clock/power/cache policy, and median. Write bundle-root `qualification-v1.json` under a closed schema with baseline run ID/digest/median, candidate sample IDs/median, ratio, percent regression, and limit `5.0`; calculate ordinary odd-count medians from sorted per-token values and require `candidate_median <= baseline_median * 1.05`. Any mismatch is non-comparable and fails.
- [ ] **Step 5: Implement bounded monitoring and failure retention.** Emit a start summary before the overnight phase, sample status/progress without altering processes, abort on wrong device, competing GPU process, observed clock/memory-clock/power/performance/temperature violation, excessive timing dispersion, quality failure, timeout, incomplete round, or disk exhaustion, and upload partial evidence with `status=incomplete`. Diagnostic throttle bits and labels are retained but do not independently trigger an abort.
- [ ] **Step 6: Document the exact local/workflow commands and commit.**

```bash
python3 -m unittest tests.test_ci_workflows tests.test_benchmark_environment tests.test_benchmark_execution tests.test_benchmark_validation tests.test_active_docs -v
python3 tools/check-active-docs.py
git diff --check
git add .github/workflows/benchmark-scalar-qualification.yml tools/benchmark/environment.py tools/benchmark/execution.py benchmarks/schema/result-v1.schema.json benchmarks/schema/qualification-v1.schema.json tests/test_ci_workflows.py tests/test_benchmark_environment.py tests/test_benchmark_execution.py tests/test_benchmark_validation.py docs/benchmarks.md docs/testing.md tests/test_active_docs.py
git commit -m "ci(bench): add scalar overnight qualification"
```

### Task 5: Validate and render candidate-only comparisons

**Files:**
- Modify: `tools/benchmark/validation.py`
- Modify: `tools/benchmark/compare.py`
- Modify: `tools/benchmark/render.py`
- Modify: `tests/test_benchmark_validation.py`
- Modify: `tests/test_benchmark_compare.py`
- Modify: `tests/test_benchmark_render.py`
- Modify: `benchmarks/results/README.md`

**Interfaces:**
- Consumes: complete three-engine bundle.
- Produces: validation result and local preview pages labeling scalar as contributor-only, WMMA as production control, and llama.cpp as pinned peer.

- [ ] **Step 1: Write RED validation/render tests.** Block publishability for incomplete duration, unequal round counts, failed quality, wrong artifact/version, uncontrolled clocks, clock/cache mismatch, missing raw samples, scalar route mismatch, sustained loaded-clock drift, excessive timing dispersion, or another strict telemetry error. Assert that nonzero throttle bits remain visible without independently blocking an otherwise qualified record, and that pages show every raw sample/statistic/count and direct evidence links but contain no unsupported speedup language.
- [ ] **Step 2: Implement explicit comparison groups.** Scalar↔WMMA uses matched SuperSonic artifact identity; scalar/WMMA↔llama.cpp requires declared peer-compatibility dimensions. Any mismatch appears as a reason, never as an omitted record or numeric comparison.
- [ ] **Step 3: Run fixture validation and rendering.**

```bash
python3 -m unittest tests.test_benchmark_validation tests.test_benchmark_compare tests.test_benchmark_render -v
run_id=scalar-qualification-first
python3 tools/supersonic-bench.py validate --bundle "target/benchmarks/scalar-qualification/candidate/$run_id"
python3 tools/supersonic-bench.py render --bundle "target/benchmarks/scalar-qualification/candidate/$run_id" --output target/benchmarks/scalar-qualification/pages
```

Expected: a complete local candidate validates; an intentionally incomplete fixture fails and renders no comparative claim.

- [ ] **Step 4: Run the complete non-GPU gate and commit.**

```bash
python3 -m unittest discover -s tests -v
cargo test --workspace --all-targets
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
HIP_ARCH=gfx1100 cargo check --workspace --all-targets
cargo fmt --all --check
python3 tools/check-active-docs.py
git diff --check
git add tools/benchmark tests/test_benchmark_validation.py tests/test_benchmark_compare.py tests/test_benchmark_render.py benchmarks/results/README.md
git commit -m "feat(bench): render scalar qualification candidates"
```

### Task 6: Run and review the first six-hour candidate

**Files:**
- Record: `target/benchmarks/scalar-qualification/candidate/scalar-qualification-first/`
- Verify: all Task 1–5 files

**Interfaces:**
- Consumes: reviewed workflow/tooling and host-prepared locked state.
- Produces: one candidate bundle and a pass/fail recommendation; no publication or production change.

- [ ] **Step 1: Perform read-only preflight** and record the exact commit, three engine versions, the shared artifact path/size/SHA-256, tokenizer/template digests, physical/logical mapping, free disk, idle GPU, clock/memory/power/performance policy, diagnostic throttle fields, cache evidence, and the separately reviewed hardened-scalar baseline bundle. Reject a baseline with any mismatched comparability field, missing loaded-clock evidence, sustained loaded-clock drift, strict telemetry error, or per-token MAD above 3% of its median; enforce candidate scalar median <= `1.05 * baseline_median` in addition to the absolute limits.
- [ ] **Step 2: Start the workflow manually** and monitor the first two complete rounds for deterministic output, balanced ordering, telemetry completeness, disk growth, temperature, and time-budget projection. Do not restart merely to improve a valid result.
- [ ] **Step 3: Let complete rounds run for at least 20,700 seconds** and stop before 21,600 seconds. Preserve partial evidence on any abort and mark it non-comparable.
- [ ] **Step 4: Validate and render locally** with `--publishable` disabled for this contributor-only candidate. Review all quality failures and comparison mismatches explicitly.
- [ ] **Step 5: Request whole-change and evidence review, then stop.** A passing candidate permits a separate promotion decision; it does not itself switch the runner, publish Pages, tag a release, or remove WMMA.
