# Reproducible Benchmark Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build quick and overnight performance/quality suites that produce validated, reproducible records and deterministic GitHub Pages comparisons for SuperSonic and pinned peers.

**Architecture:** A dependency-free Python package owns manifests, execution, validation, scoring, comparison, and static rendering behind one `tools/supersonic-bench.py` CLI. Versioned TOML/JSON inputs and committed result JSON are the source of truth; isolated GPU jobs create candidates, while CPU CI validates and renders the Pages artifact.

**Tech Stack:** Python 3.11 standard library, TOML, JSON Schema, `unittest`, GitHub Actions, the existing Rust `supersonic` CLI, and ROCm tools.

**Spec:** `docs/superpowers/specs/2026-08-24-reproducible-benchmark-pages-design.md`

## Global Constraints

- Preserve the public Qwen3.8-27B, paired model-directory/GQH GGUF, single-sequence greedy, HIP-only contract.
- Quick budget is exactly 600 seconds; full budget is exactly 21,600 seconds.
- Quality failures block immediately. Performance stays report-only until repeated baselines justify architecture-specific thresholds.
- Headline performance requires verified locked clocks. `uncontrolled-clocks` records never produce peer speedup claims.
- `cold-load`, `warm-resident`, and explicit prefix-cache states remain separate series. Never claim an unverified cache flush.
- Configured missing artifacts, engines, samples, or telemetry fail closed. Do not download inputs or add silent fallbacks.
- Use argv arrays and `shell=False`; reject secrets and absolute paths from portable records.
- Generated HTML is not committed. Existing CPU, documentation, artifact, deterministic generation, and MTP equality gates remain prerequisites.

## File Map

- `benchmarks/schema/result-v1.schema.json`: portable result structure.
- `benchmarks/suites/{quick,full}.toml`: selections, repetitions, and budgets.
- `benchmarks/quality/v1.json`: deterministic cases and expected results.
- `benchmarks/engines/{supersonic,llama-cpp}.toml`: narrow adapter metadata.
- `benchmarks/results/`: committed validated records only.
- `tools/benchmark/model.py`: immutable domain types and canonical JSON.
- `tools/benchmark/manifest.py`: strict TOML/JSON loaders.
- `tools/benchmark/adapters.py`: command construction and output parsing.
- `tools/benchmark/environment.py`: clock, power, thermal, and cache evidence.
- `tools/benchmark/quality.py`: deterministic scoring.
- `tools/benchmark/validation.py`: schema, semantic, and safety checks.
- `tools/benchmark/compare.py`: statistics and comparability.
- `tools/benchmark/execution.py`: preflight, budgets, processes, and bundles.
- `tools/benchmark/render.py`: deterministic HTML.
- `tools/supersonic-bench.py`: `run`, `validate`, `compare`, and `render` CLI.
- `tests/test_benchmark_*.py`: CPU-safe contracts with `tests/benchmark_fixtures/`.
- `.github/workflows/benchmark-{quick,full,pages}.yml`: GPU candidates and CPU publication.

---

### Task 1: Versioned Manifests and Result Contract

**Files:**
- Create: `benchmarks/schema/result-v1.schema.json`
- Create: `benchmarks/suites/quick.toml`
- Create: `benchmarks/suites/full.toml`
- Create: `benchmarks/quality/v1.json`
- Create: `benchmarks/engines/supersonic.toml`
- Create: `benchmarks/engines/llama-cpp.toml`
- Create: `tools/benchmark/__init__.py`
- Create: `tools/benchmark/model.py`
- Create: `tools/benchmark/manifest.py`
- Create: `tests/test_benchmark_manifests.py`

**Interfaces:**
- Consumes: Python 3.11 `tomllib` and `tools/external/*-version.txt`.
- Produces: `load_suite(name: str) -> SuiteManifest`, `load_quality(version: str) -> tuple[QualityCase, ...]`, `load_engine(name: str) -> EngineManifest`, and `canonical_json(value: object) -> str`.

- [ ] **Step 1: Write failing strict-loader tests**

```python
def test_budgets_and_case_sets(self):
    quick = manifest.load_suite("quick")
    full = manifest.load_suite("full")
    self.assertEqual(quick.budget_seconds, 600)
    self.assertEqual(full.budget_seconds, 21600)
    self.assertLess(set(quick.quality_case_ids), set(full.quality_case_ids))

def test_unknown_cache_state_and_key_fail(self):
    with self.assertRaisesRegex(ValueError, "cache_state|unknown"):
        manifest.load_suite_path(self.bad_manifest)
```

Also assert schema version `1`, greedy decoding, positive repetitions/timeouts, unique IDs, full peer inclusion, exact allowed keys, and all references resolving.

- [ ] **Step 2: Prove the tests fail**

Run: `python3 -m unittest tests.test_benchmark_manifests -v`

Expected: FAIL because `tools/benchmark/manifest.py` is absent.

- [ ] **Step 3: Implement immutable types, strict loaders, and data files**

```python
@dataclass(frozen=True)
class PerformanceCase:
    id: str
    prompt: str
    max_new_tokens: int
    warmups: int
    repetitions: int
    mode: str
    cache_state: str
    timeout_seconds: int

@dataclass(frozen=True)
class SuiteManifest:
    version: int
    name: str
    budget_seconds: int
    quality_version: str
    quality_case_ids: tuple[str, ...]
    engines: tuple[str, ...]
    performance_cases: tuple[PerformanceCase, ...]
```

Quick uses one warmup, three samples, representative performance cases, and a quality subset. Full uses short/long, cold/warm, ordinary/MTP, both engines, at least seven samples, and the complete quality corpus. Include at least two quality cases in each approved category. The schema requires run, engine, hardware, artifact, workload, environment, samples, quality, status, and errors and forbids undeclared top-level fields.

- [ ] **Step 4: Run tests**

Run: `python3 -m unittest tests.test_benchmark_manifests -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks tools/benchmark tests/test_benchmark_manifests.py
git commit -m "feat(bench): define versioned benchmark contracts"
```

### Task 2: Engine Adapters

**Files:**
- Create: `tools/benchmark/adapters.py`
- Create: `tests/benchmark_fixtures/supersonic-run.log`
- Create: `tests/benchmark_fixtures/llama-cpp-run.log`
- Create: `tests/test_benchmark_adapters.py`

**Interfaces:**
- Consumes: `EngineManifest`, `PerformanceCase`, explicit model/artifact inputs.
- Produces: `build_command(engine, case, inputs) -> tuple[str, ...]` and `parse_output(engine_name: str, stdout: str) -> ParsedOutput`.

- [ ] **Step 1: Write failing command/parser tests**

```python
def test_supersonic_argv_uses_public_contract(self):
    argv = adapters.build_command(self.engine, self.case, self.inputs)
    self.assertEqual(argv[0], "./target/release/supersonic")
    self.assertIn("qwen3.8-27b", argv)
    self.assertIn("--emit-generated-json", argv)
    self.assertNotIn("|", argv)

def test_duplicate_result_line_fails(self):
    with self.assertRaisesRegex(ValueError, "exactly one"):
        adapters.parse_output("supersonic", self.log + self.log)
```

Cover chat, MTP, ignore-EOS, token IDs, finite positive timings, llama.cpp version identity, and its separate peer artifact.

- [ ] **Step 2: Prove failure**

Run: `python3 -m unittest tests.test_benchmark_adapters -v`

Expected: FAIL because adapters are absent.

- [ ] **Step 3: Implement argv-only adapters and strict parsers**

```python
args = [engine.binary, "--model", "qwen3.8-27b", "--model-dir", str(inputs.model_dir),
        "--gguf-file", str(inputs.artifact), "--prompt", case.prompt,
        "--max-new-tokens", str(case.max_new_tokens), "--ignore-eos",
        "--emit-generated-json", "--emit-stage-timings", "--device", "0"]
```

Never use `shell=True`, `eval`, or free-form command strings. Require one final record and reject NaN, infinity, negative time, inconsistent counts, or missing deterministic output. Keep raw streams only in local candidate bundles.

- [ ] **Step 4: Run new and legacy parser tests**

Run: `python3 -m unittest tests.test_benchmark_adapters tests.test_qwen38_reproducibility -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/benchmark/adapters.py tests/benchmark_fixtures tests/test_benchmark_adapters.py
git commit -m "feat(bench): add strict engine adapters"
```

### Task 3: Environment and Cache Evidence

**Files:**
- Create: `tools/benchmark/environment.py`
- Create: `tests/benchmark_fixtures/rocm-smi-showallinfo.txt`
- Create: `tests/test_benchmark_environment.py`

**Interfaces:**
- Consumes: physical GPU ID, clock policy, cache state, injected bounded command runner.
- Produces: `collect_snapshot(...) -> EnvironmentSnapshot`, `verify_clock_policy(before, observed, after, policy) -> tuple[str, ...]`, and `validate_cache_evidence(cache_state, evidence) -> None`.

- [ ] **Step 1: Write failing policy tests**

```python
def test_uncontrolled_clocks_are_not_headline_eligible(self):
    self.assertFalse(self.snapshot("uncontrolled-clocks").headline_eligible)

def test_locked_clock_drift_fails(self):
    errors = environment.verify_clock_policy(self.before, [self.drifted], self.after, self.policy)
    self.assertIn("clock drift", " ".join(errors).lower())

def test_unverified_flush_claim_fails(self):
    with self.assertRaisesRegex(ValueError, "verified"):
        environment.validate_cache_evidence("cold-load", {"filesystem_flush": "claimed"})
```

Also cover power cap, temperature, performance level, CPU governor, physical/logical mapping, allowlisted environment, fresh-process wording, and empty/populated/reset prefix-cache transitions.

- [ ] **Step 2: Prove failure**

Run: `python3 -m unittest tests.test_benchmark_environment -v`

Expected: FAIL because environment collection is absent.

- [ ] **Step 3: Implement read-only collection and strict verification**

```python
ALLOWLISTED_ENVIRONMENT = (
    "HIP_ARCH", "HIP_VISIBLE_DEVICES", "ROCM_PATH", "HIP_PATH",
    "SUPERSONIC_DEVICE", "RUSTFLAGS",
)

def verify_clock_policy(before, observed, after, policy):
    if policy.name == "uncontrolled-clocks":
        return ()
    return tuple(_clock_violations((before, *observed, after), policy))
```

Use argv-only, 30-second-bounded SMI probes through an injected runner. Never set clocks or power. Store requested and observed state with timestamps. Unsupported fields are null with an evidence note, not guessed.

- [ ] **Step 4: Run environment and device tests**

Run: `python3 -m unittest tests.test_benchmark_environment tests.test_r9700_helpers -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/benchmark/environment.py tests/benchmark_fixtures/rocm-smi-showallinfo.txt tests/test_benchmark_environment.py
git commit -m "feat(bench): record clock and cache evidence"
```

### Task 4: Deterministic Quality Scoring

**Files:**
- Create: `tools/benchmark/quality.py`
- Create: `tests/test_benchmark_quality.py`
- Modify: `benchmarks/quality/v1.json`

**Interfaces:**
- Consumes: `QualityCase`, `ParsedOutput`, paired ordinary/MTP outputs.
- Produces: `score_case(case, output) -> QualityResult`, `score_mtp_pair(ordinary, mtp) -> QualityResult`, and `summarize_quality(results) -> dict[str, object]`.

- [ ] **Step 1: Write failing scorer tests**

```python
def test_exact_text_is_not_fuzzy(self):
    self.assertFalse(quality.score_case(self.case("42"), self.output("42 ")).passed)

def test_structured_json_compares_values(self):
    result = quality.score_case(self.json_case({"answer": 42}), self.output('{"answer":42}'))
    self.assertTrue(result.passed)

def test_mtp_requires_identical_tokens(self):
    self.assertFalse(quality.score_mtp_pair(self.tokens([1, 2]), self.tokens([1, 3])).passed)
```

Test every category, repeated-run determinism, duplicate JSON keys, missing cases, category counts, and aggregate failure when any required case fails.

- [ ] **Step 2: Prove failure**

Run: `python3 -m unittest tests.test_benchmark_quality -v`

Expected: FAIL because scorers are absent.

- [ ] **Step 3: Implement only approved deterministic scorers**

```python
if case.scorer == "exact_text":
    actual, passed = output.generated_text, output.generated_text == case.expected
elif case.scorer == "exact_tokens":
    actual, passed = list(output.token_ids), list(output.token_ids) == case.expected
elif case.scorer == "structured_json":
    actual = _strict_json(output.generated_text)
    passed = actual == case.expected
else:
    raise ValueError(f"unsupported scorer: {case.scorer}")
```

Perform no undeclared normalization. Preserve expected/actual hashes and bounded values. Expose passed/failed/total per category; do not add a weighted score that hides failures.

- [ ] **Step 4: Run quality and manifest tests**

Run: `python3 -m unittest tests.test_benchmark_quality tests.test_benchmark_manifests -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/quality/v1.json tools/benchmark/quality.py tests/test_benchmark_quality.py
git commit -m "feat(bench): add deterministic quality scoring"
```

### Task 5: Validation, Statistics, and Comparability

**Files:**
- Create: `tools/benchmark/validation.py`
- Create: `tools/benchmark/compare.py`
- Create: `tests/benchmark_fixtures/valid-result-v1.json`
- Create: `tests/test_benchmark_validation.py`
- Create: `tests/test_benchmark_compare.py`

**Interfaces:**
- Consumes: result schema and candidate records.
- Produces: `validate_record(record) -> None`, `validate_bundle(path, require_complete) -> tuple[Path, ...]`, `summarize_samples(values) -> SampleSummary`, `series_key(record) -> tuple[str, ...]`, and `compare_records(left, right) -> Comparison`.

- [ ] **Step 1: Write failing validation/security tests**

```python
def test_valid_fixture_passes(self):
    validation.validate_record(self.valid_record)

def test_path_and_non_finite_sample_fail(self):
    record = copy.deepcopy(self.valid_record)
    record["run"]["command"] = ["/home/private/supersonic"]
    record["samples"][0]["decode_ms"] = float("nan")
    with self.assertRaises(ValueError):
        validation.validate_record(record)

def test_incomplete_bundle_is_not_publishable(self):
    with self.assertRaisesRegex(ValueError, "incomplete"):
        validation.validate_bundle(self.bundle, require_complete=True)
```

Cover required types, unknown fields, digests, exact sample counts, dirty state, errors, quality status, locked eligibility, secrets, and configured missing inputs.

- [ ] **Step 2: Write failing comparison tests**

```python
def test_statistics_retain_raw_distribution(self):
    summary = compare.summarize_samples([3.0, 1.0, 2.0])
    self.assertEqual((summary.minimum, summary.median, summary.maximum), (1.0, 2.0, 3.0))
    self.assertIsNotNone(summary.mad)

def test_cache_and_clock_mismatch_forbid_speedup(self):
    result = compare.compare_records(self.locked_warm, self.uncontrolled_cold)
    self.assertFalse(result.comparable)
    self.assertIsNone(result.speedup)
    self.assertIn("cache_state", result.reasons)
    self.assertIn("clock_policy", result.reasons)
```

Cover hardware, artifact semantics, tokenizer/template digests, prompt/case, limits, stop policy, cache, warmups, measurement boundary, clock/power, and architecture.

- [ ] **Step 3: Prove failure**

Run: `python3 -m unittest tests.test_benchmark_validation tests.test_benchmark_compare -v`

Expected: FAIL because modules are absent.

- [ ] **Step 4: Implement fail-closed validation and comparisons**

```python
COMPARABILITY_FIELDS = (
    "hardware.identity", "hardware.architecture", "artifact.semantic_id",
    "artifact.quantization", "artifact.tokenizer_sha256", "artifact.chat_template_sha256",
    "workload.case_id", "workload.prompt_sha256", "workload.context_limit",
    "workload.max_new_tokens", "workload.mode", "workload.stop_policy",
    "workload.cache_state", "workload.warmups", "workload.measurement_boundary",
    "environment.clock_policy", "environment.power_cap_watts",
)
```

Implement schema keywords used by result v1: type, required, properties, additionalProperties, items, enum, const, minimum, minItems, and pattern. Recursively reject non-finite numbers, paths, and secret-like values. Use median and median absolute deviation. Compute ratios only when comparable is true.

- [ ] **Step 5: Run and commit**

Run: `python3 -m unittest discover -s tests -p 'test_benchmark_*.py' -v`

Expected: PASS.

```bash
git add benchmarks/schema tools/benchmark/validation.py tools/benchmark/compare.py tests/benchmark_fixtures/valid-result-v1.json tests/test_benchmark_validation.py tests/test_benchmark_compare.py
git commit -m "feat(bench): validate and compare benchmark evidence"
```

### Task 6: Budgeted Execution and CLI

**Files:**
- Create: `tools/benchmark/execution.py`
- Create: `tools/supersonic-bench.py`
- Create: `tests/test_benchmark_execution.py`
- Create: `tests/test_supersonic_bench_cli.py`

**Interfaces:**
- Consumes: manifests, adapters, environment, quality, and validation modules.
- Produces: `preflight(config) -> RunManifest`, `run_suite(config, clock, command_runner) -> BundleStatus`, atomic candidate bundles, and CLI `run`, `validate`, and `compare` commands.

- [ ] **Step 1: Write failing preflight/budget/atomicity tests**

```python
def test_configured_peer_missing_fails_preflight(self):
    with self.assertRaisesRegex(ValueError, "llama-cpp.*unavailable"):
        execution.preflight(self.full_config(binary_exists=False))

def test_budget_stops_scheduling_and_marks_incomplete(self):
    status = execution.run_suite(self.config, self.fake_clock([0, 599, 601]), self.runner)
    self.assertEqual(status.state, "incomplete")
    self.assertIn("budget_exhausted", status.errors)
    self.assertEqual(self.runner.started_case_ids, ["case-1"])

def test_invalid_record_is_never_atomically_promoted(self):
    status = execution.run_suite(self.config, self.fake_clock(), self.corrupt_runner)
    self.assertFalse((status.bundle / "records" / "case.json").exists())
```

Cover exact budgets, case timeouts, SIGINT, isolated processes, warmups, seeded interleaving, raw stream capture, quality blocking, and performance report-only status.

- [ ] **Step 2: Prove failure**

Run: `python3 -m unittest tests.test_benchmark_execution tests.test_supersonic_bench_cli -v`

Expected: FAIL because execution and CLI files are absent.

- [ ] **Step 3: Implement monotonic orchestration and atomic records**

```python
def run_suite(config, clock=time.monotonic, command_runner=run_process):
    manifest = preflight(config)
    deadline = clock() + manifest.suite.budget_seconds
    for case in ordered_cases(manifest):
        if clock() >= deadline:
            return finalize_incomplete(manifest, "budget_exhausted")
        remaining = max(1, int(deadline - clock()))
        execute_case(manifest, case, min(case.timeout_seconds, remaining), command_runner)
    return finalize_complete(manifest)
```

Use `subprocess.run(argv, shell=False, timeout=...)`, fresh measured processes, temporary sibling files, `flush`, `fsync`, validation, then `Path.replace`. Record dirty-tree state. Require explicit model/artifact/GPU inputs and separate `--peer-artifact`. `validate --publishable` fails incomplete or quality-failed bundles; `compare` emits JSON.

- [ ] **Step 4: Run pipeline and existing reproducibility tests**

Run: `python3 -m unittest tests.test_benchmark_execution tests.test_supersonic_bench_cli tests.test_qwen38_reproducibility -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/benchmark/execution.py tools/supersonic-bench.py tests/test_benchmark_execution.py tests/test_supersonic_bench_cli.py
git commit -m "feat(bench): execute budgeted benchmark suites"
```

### Task 7: Deterministic Static Site

**Files:**
- Create: `tools/benchmark/render.py`
- Create: `tests/test_benchmark_render.py`
- Create: `benchmarks/results/README.md`
- Modify: `tools/supersonic-bench.py`

**Interfaces:**
- Consumes: publishable records and `compare_records`.
- Produces: `render_site(results_root, output_root) -> tuple[Path, ...]` with landing, methodology, run, trend, comparison, and stylesheet files; CLI `render`.

- [ ] **Step 1: Write failing render tests**

```python
def test_two_renders_are_byte_identical(self):
    self.assertEqual(self.snapshot(self.render()), self.snapshot(self.render()))

def test_noncomparable_peer_has_reasons_and_no_speedup(self):
    page = self.render_comparison(self.unlocked_peer)
    self.assertIn("uncontrolled-clocks", page)
    self.assertNotIn("speedup", page.lower())

def test_untrusted_text_is_escaped(self):
    page = self.render_run(self.record_with_text("<script>alert(1)</script>"))
    self.assertNotIn("<script>", page)
    self.assertIn("&lt;script&gt;", page)
```

Also assert raw samples/statistics, clock/cache badges, quality failures, safe reproduction commands, versions, stable sorting, and no generation timestamp.

- [ ] **Step 2: Prove failure**

Run: `python3 -m unittest tests.test_benchmark_render -v`

Expected: FAIL because renderer is absent.

- [ ] **Step 3: Implement no-JavaScript deterministic HTML**

```python
def page(title: str, body: str) -> str:
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<title>{html.escape(title)}</title><link rel=\"stylesheet\" "
            "href=\"/assets/benchmarks.css\"></head>"
            f"<body>{body}</body></html>\n")
```

Render only publishable records. Derive stable IDs from canonical keys. Put every numeric claim beside or link it to commit, GPU, artifact, workload, samples, correctness, clock, and cache evidence.

- [ ] **Step 4: Add `render` and run determinism tests**

Run: `python3 -m unittest tests.test_benchmark_render tests.test_supersonic_bench_cli -v`

Expected: PASS with byte-identical output trees.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/results/README.md tools/benchmark/render.py tools/supersonic-bench.py tests/test_benchmark_render.py tests/test_supersonic_bench_cli.py
git commit -m "feat(bench): render reproducible benchmark pages"
```

### Task 8: GPU and Pages Workflows

**Files:**
- Create: `.github/workflows/benchmark-quick.yml`
- Create: `.github/workflows/benchmark-full.yml`
- Create: `.github/workflows/benchmark-pages.yml`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/qwen38-gfx1201.yml`
- Modify: `tests/test_ci_workflows.py`

**Interfaces:**
- Consumes: benchmark CLI, existing GPU selector/artifact preflight, engine pins.
- Produces: quick/full candidate artifacts and default-branch-only validated Pages deployment.

- [ ] **Step 1: Write failing workflow contracts**

```python
def test_full_is_manual_serial_and_six_hours(self):
    text = (WORKFLOWS / "benchmark-full.yml").read_text()
    self.assertIn("workflow_dispatch:", text)
    self.assertIn("timeout-minutes: 360", text)
    self.assertIn("concurrency:", text)
    self.assertIn("--suite full", text)
    self.assertNotIn("continue-on-error: true", text)

def test_pages_validates_before_deploy(self):
    text = (WORKFLOWS / "benchmark-pages.yml").read_text()
    self.assertLess(text.index("validate --publishable"), text.index(" render "))
    self.assertLess(text.index(" render "), text.index("deploy-pages"))
    self.assertNotIn("pull_request_target", text)
```

Require quick timeout 10, pinned actions, device mapping/idle checks, strict pins/artifacts/clocks/cache, `RUST_TEST_THREADS=1`, candidate uploads on `always()`, no Git mutation from GPU jobs, minimal Pages permissions, and no PR deploy.

- [ ] **Step 2: Prove failure**

Run: `python3 -m unittest tests.test_ci_workflows -v`

Expected: FAIL because workflows are absent.

- [ ] **Step 3: Implement workflows and remove duplicate telemetry**

Quick/full run on `[self-hosted, linux, rocm, gfx1201]`, share the existing selector/preflight, verify requested clock state, build release, and emit only candidate artifacts. Full requires pinned llama.cpp and peer artifact. Pages CPU-validates committed records, renders, uploads, and deploys only from main. PR CI validates manifests/schema and renders fixtures without GPU access. Replace the ad-hoc throughput loop in `qwen38-gfx1201.yml` only after quick records contain the old reproducibility fields; retain correctness/MTP gates.

- [ ] **Step 4: Run workflow and all CPU tests**

Run: `python3 -m unittest tests.test_ci_workflows -v`

Run: `python3 -m unittest discover -s tests -p 'test_*.py' -v`

Expected: PASS; configured failures block and candidate uploads remain available for diagnosis.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/benchmark-quick.yml .github/workflows/benchmark-full.yml .github/workflows/benchmark-pages.yml .github/workflows/ci.yml .github/workflows/qwen38-gfx1201.yml tests/test_ci_workflows.py
git commit -m "ci: run and publish reproducible benchmarks"
```

### Task 9: Public Documentation and Complete Gates

**Files:**
- Modify: `docs/benchmarks.md`
- Modify: `docs/performance.md`
- Modify: `docs/testing.md`
- Modify: `README.md`
- Modify: `tools/check-active-docs.py`
- Modify: `tests/test_active_docs.py`

**Interfaces:**
- Consumes: final CLI, workflows, result schema, and methodology.
- Produces: exact operator/reviewer instructions and stronger evidence checks.

- [ ] **Step 1: Write failing documentation tests**

```python
def test_benchmark_docs_define_tiers_clocks_and_cache(self):
    text = (ROOT / "docs/benchmarks.md").read_text().lower()
    for term in ("quick", "10 minutes", "full", "six hours", "locked",
                 "uncontrolled-clocks", "cold-load", "warm-resident"):
        self.assertIn(term, text)

def test_peer_claims_require_comparability_evidence(self):
    text = (ROOT / "docs/performance.md").read_text().lower()
    for term in ("comparability", "artifact", "cache state", "clock", "sample count"):
        self.assertIn(term, text)
```

Extend numeric-claim checks so comparisons require engine/version, clock/cache policy, statistic/sample count, correctness, and direct run evidence in addition to existing commit/GPU/artifact/workload fields.

- [ ] **Step 2: Prove failure**

Run: `python3 -m unittest tests.test_active_docs -v`

Expected: FAIL because new contracts are undocumented.

- [ ] **Step 3: Document exact operation and promotion**

Document quick/full commands, host preparation without privileged mutation, candidate location, `validate --publishable`, code-reviewed promotion, Pages configuration, cache terminology, peer qualification, median/MAD, and incomplete behavior. Keep README short and use the existing six public documents.

- [ ] **Step 4: Run complete CPU-safe gate**

```bash
git diff --check
cargo fmt --all --check
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
python3 tools/check-support-matrix.py
python3 tools/check-kernel-groups.py
python3 tools/check-tool-inventory.py
python3 tools/check-active-docs.py
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

Expected: every command exits 0 with no configured skip or warning.

- [ ] **Step 5: Run configured `gfx1201` acceptance**

Run the existing serial artifact/correctness gate from `docs/testing.md`, then:

```bash
python3 tools/supersonic-bench.py run \
  --suite quick \
  --model-dir "$SUPERSONIC_QWEN38_MODEL_DIR" \
  --artifact "$SUPERSONIC_GQH_GGUF" \
  --physical-gpu "$SUPERSONIC_R9700_GPU_ID" \
  --clock-policy locked \
  --output target/benchmarks/candidate
python3 tools/supersonic-bench.py validate --publishable target/benchmarks/candidate
```

Expected: completion within 600 seconds, passing quality/MTP equality, complete clock/cache evidence, and valid candidate. Trigger the six-hour full workflow explicitly after quick review; do not make it an automatic implementation test.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/benchmarks.md docs/performance.md docs/testing.md tools/check-active-docs.py tests/test_active_docs.py
git commit -m "docs: publish benchmark reproduction contract"
```

### Task 10: First Full Baseline and Pages Smoke Test

**Files:**
- Create: `benchmarks/results/gfx1201/<run-id>/manifest.json`
- Create: `benchmarks/results/gfx1201/<run-id>/records/*.json`
- Modify: `docs/performance.md`

**Interfaces:**
- Consumes: reviewed full-workflow candidate; never raw absolute-path logs.
- Produces: first committed baseline and live Pages URL.

- [ ] **Step 1: Validate and compare the overnight candidate**

```bash
python3 tools/supersonic-bench.py validate \
  --publishable target/benchmarks/candidate/<run-id>
python3 tools/supersonic-bench.py compare \
  target/benchmarks/candidate/<run-id> \
  --output target/benchmarks/candidate/<run-id>/comparison.json
```

Expected: validation passes and every peer pairing is comparable or has explicit mismatch reasons. Incomplete state, quality mismatch, uncontrolled clocks, throttling, missing samples, or unsafe paths block promotion.

- [ ] **Step 2: Promote only portable validated records**

Copy the manifest and records with `apply_patch`. Do not copy raw logs, absolute paths, SMI dumps, or candidate-local comparison output. Then run:

```bash
python3 tools/supersonic-bench.py validate --publishable benchmarks/results
```

Expected: PASS.

- [ ] **Step 3: Render and inspect locally**

```bash
site_root="$(mktemp -d)"
python3 tools/supersonic-bench.py render --results benchmarks/results --output "$site_root"
find "$site_root" -maxdepth 5 -type f -print | sort
```

Inspect landing, run, methodology, trend, and comparison pages. Verify evidence is present, invalid speedups are absent, mismatch states are conspicuous, and no private path appears.

- [ ] **Step 4: Commit the reviewed baseline**

```bash
git add benchmarks/results
git commit -m "bench: publish first gfx1201 baseline"
```

- [ ] **Step 5: Verify deployment and link the stable site**

After default-branch integration, confirm the Pages workflow deployed the baseline commit and all links resolve. Add the stable URL to `docs/performance.md`, then run and commit:

```bash
python3 tools/check-active-docs.py
python3 -m unittest tests.test_active_docs -v
git diff --check
git add docs/performance.md
git commit -m "docs: link published benchmark pages"
```
