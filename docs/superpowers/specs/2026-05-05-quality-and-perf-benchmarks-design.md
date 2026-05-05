# Quality and performance benchmarks (Phase 1)

**Status:** Design — pending implementation
**Date:** 2026-05-05
**Scope:** Phase 1. Phases 2 (llama.cpp) and 3 (vLLM) are out of scope here; the architecture leaves room for them as drop-in adapters.

## Problem

SuperSonic has thorough perf measurement on a per-PR, per-feature basis (`tests/gfx1100/bench_matrix.sh`, `docs/performance.md`) and rigorous bit-exact correctness gating against PyTorch oracles, but it is missing two things that matter for both internal regression work and any external story:

1. **Quality measurement of the shipping models.** Correctness ≠ quality. We know `qwen3.5-9b INT4` produces logits within tolerance of the BF16 path; we do not have a measured perplexity number for it on PG-19, much less a HellaSwag/ARC/MMLU score, and we have no automated regression net for "does INT4 still produce coherent text" (the documented `gemma4-e2b INT4 → repetition` failure was caught manually).
2. **A speed baseline against the actual competition on this class of machine.** We do not know whether SuperSonic is competitive with `hipfire` on the RX 7900 XTX, which is the most-likely-strongest single-GPU RDNA3 inference engine. Without that number we cannot make informed decisions about which performance work to prioritize next.

This spec covers Phase 1: a unified harness that produces both quality and performance numbers for every shipping (model, quant) combo on `gfx1100`, plus a speed-only `hipfire` comparison column.

## Goals

- **Dev-tool first.** A working operator (you) runs this on their machine to answer "did my last PR move quality or perf?" and "where am I vs hipfire today?". The output JSON is the source of truth; markdown is a view.
- **Publishable on demand.** When the operator wants a documentation artifact, a single render command produces a markdown table that drops into `docs/quality.md` and `docs/performance.md` without hand-editing data.
- **Regression-diff capable.** Two run-dirs from different shas can be diffed to surface cells where ms/step or perplexity has moved beyond a threshold.
- **Architecture-parametric, gfx1100-data-only.** Code knows about archs but Phase 1 only collects gfx1100 numbers. Adding a gfx1150 row later is "run on that machine, results land in a different folder," not a refactor.
- **`hipfire`-in-scope at Phase 1.** Speed only, on whatever subset of shipping models hipfire supports. Quality comparison across engines is out of scope for Phase 1 (tokenizer alignment is its own project).

## Non-goals

- llama.cpp and vLLM adapters (Phase 2 / Phase 3).
- Cross-engine perplexity. Tokenizer alignment is a known rabbit hole; Phase 1 hipfire is speed-only.
- Multi-GPU / distributed runs.
- Power-capped reproducibility (`rocm-smi --setperflevel` enforcement).
- Statistical confidence intervals on perplexity (single-pass over a fixed sample count, same as the existing `pg19_smoke.py`).
- Auto-coordination with the other agent that shares this GPU. Phase 1 records `rocm-smi -u` in `meta.json` so contended runs are *visible* in the JSON; coordination is on the operator.
- Replacing `bench_matrix.sh` immediately. It stays runnable during a transition period; deletion happens in a follow-up PR after a 2-week soak and a parity gate (see Testing).

## Architecture

One Rust crate, one Python package, one shared run-dir, one shared JSON schema, one renderer.

```
crates/bench/                        ← new Rust crate
  src/
    lib.rs                             # orchestrator entry points
    runs.rs                            # run-dir layout, JSON schema, git-sha capture
    perf.rs                            # invokes ./target/release/supersonic, parses ms_per_step
    matrix.rs                          # (arch, model, quant) iteration; reads runner registry
  bin/
    bench-perf.rs                      # CLI: --models … --quants … --emit json|md

oracle/bench/                        ← new Python package
  __init__.py
  runner.py                            # subprocess driver for ./target/release/supersonic
  perplexity.py                        # PG-19, WikiText-2 teacher-forced (extends pg19_smoke)
  golden.py                            # diff vs BF16 reference outputs
  golden_prompts.json                  # ~20 curated prompts + per-(model,quant) reference outputs
  heavy/
    niah.py                            # extends arxiv_v1_smoke for non-Llama models
    ruler.py                           # new
    longctx.py                         # 4k/8k/16k/32k perf+quality stage
    lm_eval.py                         # lm-evaluation-harness wrapper
  external/
    hipfire.py                         # subprocess driver, version pin check
    common.py                          # shared adapter base class (Phase 2 adds more)
  render/
    schema.py                          # shared JSON schema (matches crates/bench/src/runs.rs)
    markdown.py                        # JSON → docs/quality.md + perf.md fragments
    diff.py                            # two-run regression comparison

target/bench-runs/{YYYY-MM-DD}-{git-sha}[-N]/
  meta.json                            # git sha, GPU, ROCm, hostname, hipfire version, rocm-smi -u snapshot
  perf/{model}_{quant}.json
  quality/{model}_{quant}_{eval}.json
  external/hipfire/{model}_{quant}.json
  render/quality.md, perf-fragment.md  # rendered output

tools/external/
  hipfire-version.txt                  # pinned commit/tag
  check-versions.sh                    # asserts installed binaries match pins

docs/
  quality.md                           # new — rendered table, hand-edited prose at top above sentinel
  performance.md                       # gets new "vs hipfire" column where data exists
```

**Why this split:** SuperSonic-side perf measurement is already Rust-native (the runner emits structured ms/step lines). Quality measurement needs HuggingFace `datasets`, the `lm-evaluation-harness`, and tokenizer machinery that already lives in `oracle/`. External engines are most naturally subprocess-driven from Python (vLLM is `import vllm`, llama.cpp via `llama-cpp-python` or subprocess, hipfire via subprocess). The seam is clean: shared schema, shared run-dir, independent invocation.

## Components

### `crates/bench` — Rust orchestrator

- `bench-perf` binary: `--arch gfx1100 --models all|<list> --quants all|<list> [--long-ctx 4096,8192]`. Iterates `runner::registry::REGISTRY`, filters to the arch's supportable combos. For each combo: 3s cooldown, 1 warmup pass at `MAX_NEW=2`, 3 measurement passes at `MAX_NEW=16`, take the median. Writes one JSON per combo into `perf/`. Records `meta.json` once at the start (git sha, `rocminfo` output, total RAM, GPU temp pre-/post-, `rocm-smi -u` snapshot).
- Treats `./target/release/supersonic` as a black-box subprocess — same path an end user runs. The crate **does not** depend on `runner` as a library; that keeps the harness honest about what an end user sees.
- Lifts the existing `bench_matrix.sh` extraction logic into `bench::perf::extract_metrics()`. The bash matrix stays in place unchanged during the transition; explicit deletion is a follow-up PR after the parity gate (see Testing) passes.

### `oracle/bench` — Python measurers and adapters

- `runner.py`: subprocess driver for runs that need to capture more than `ms_per_step` (logits for perplexity, generated text for golden). Same cooldown/warmup discipline as the Rust side.
- `perplexity.py`: generalizes `oracle/pg19_smoke.py` to drive any (model, quant) on PG-19 and WikiText-2. **Largest implementation cost in Phase 1:** the `--teacher-forced` runner path currently exists only for the CUDA Llama lane and must be extended to Qwen, Gemma, and Phi-4 on HIP.
- `golden.py`: reads `golden_prompts.json` (~20 prompts), runs each (model, quant) once, diffs the generated text against a stored reference produced by the BF16 path of that model. Two scores: exact-match rate and BLEU/chrF for partial credit. Threshold-fail gate for "produces gibberish" cases.
- `heavy/niah.py`: needle-in-a-haystack adapted from `oracle/arxiv_v1_smoke.py` (which today supports only the CUDA Llama lane). Phase 1 wires it for two combos: `qwen3.5-9b INT4` and `gemma4-e4b INT4`.
- `heavy/ruler.py`: new. Same two-combo Phase 1 scope.
- `heavy/longctx.py`: runs the perf matrix at 4k/8k/16k/32k contexts, gated by VRAM in the registry. Quality measure at long context is perplexity over a 4k prefix of PG-19, decoded through to context end.
- `heavy/lm_eval.py`: wraps `lm_eval --model hf --tasks hellaswag,arc_easy,mmlu_subset --model_args …`. Same two-combo Phase 1 scope.
- `external/hipfire.py`: subprocess driver. Pinned-commit check via `hipfire --version` against `tools/external/hipfire-version.txt`; refuses to run on mismatch. Speed-only metrics. Quality is intentionally omitted.

### Renderer (`oracle/bench/render`)

- `markdown.py render --run <dir> --out docs/`: rebuilds `docs/quality.md` and the "vs hipfire" columns of `docs/performance.md` from JSON. Idempotent. Hand-edited prose above an `<!-- AUTOGEN BELOW -->` sentinel is preserved.
- `markdown.py diff --run-a <dir> --run-b <dir>`: prints a markdown table of cells whose metric has moved more than a configurable threshold (default ±5% on ms/step, ±0.05 on perplexity).

## Data flow

A perf-only run (the common dev case):
```
$ cargo run --release --bin bench-perf -- --arch gfx1100 --models all --quants all
  ├─ writes target/bench-runs/2026-05-05-abc1234/meta.json
  ├─ for each combo: cooldown → warmup → median-of-3 → perf/{model}_{quant}.json
  └─ exit
$ python -m oracle.bench.render markdown \
    --run target/bench-runs/2026-05-05-abc1234 --out docs/
  └─ rewrites docs/performance.md tables in-place
```

A quality run reuses `meta.json` if a perf run already happened today on this sha; otherwise it writes its own. A heavy run wires the larger evals into the same run-dir. A hipfire comparison run lives in `external/hipfire/*.json` and feeds the "vs hipfire" column. A regression check is `render diff` over two run-dirs.

### Invariants

1. **Run-dir is immutable once written.** Re-running the harness produces a *new* dated dir (with `-N` suffix on collision); never mutates an existing one.
2. **Renderer is pure.** Same JSON in → same markdown out. No API calls, no time-of-render data fetches.
3. **Each script is independently runnable.** The four primary entry points — `bench-perf`, `oracle.bench.quality`, `oracle.bench.heavy`, `oracle.bench.external` — have no implicit ordering between them. The composite "publishable artifact" run is a shell wrapper that calls all four in sequence into the same run-dir, but a perf-only or quality-only run is equally valid.

## Error handling

| Failure | Behavior |
|---|---|
| (model, quant) combo fails to load (OOM, missing bake) | Write `{"status": "skipped", "reason": "OOM at preflight"}` JSON. Continue. Renderer shows `—` with a footnote. |
| `supersonic` subprocess crashes mid-run | Write `{"status": "error", "stderr_tail": "..."}`. Continue. Markdown shows `ERR` with a link to the JSON. |
| ms/step parse fails (output format changed) | Treat as error. Do **not** silently fall back; missing-both means something broke. `meta.json runner_version` surfaces the cause. |
| hipfire missing or wrong version | `check-versions.sh` fails before any run starts. Lists installed vs pinned. No partial run. |
| hipfire doesn't support a model SuperSonic does | Skip silently with `{"status": "unsupported_by_engine"}`. Renderer omits the comparison column for that row. |
| Cross-engine perplexity attempt (programming error) | Harness refuses. Out of scope for Phase 1. |
| Two runs on the same date+sha | Append `-N` suffix. Don't merge or overwrite. |
| Golden-prompt reference missing for a (model, quant) | One-time bootstrap: harness writes the reference on first BF16 run, fails the corresponding INT4/FP8 cells until BF16 has populated the reference. Documented in `golden_prompts.json` header. |
| Long-context exceeds VRAM | Registry preflight rejects upfront, same as the OOM-skipped case. |
| `lm-evaluation-harness` not installed | `--heavy` lane prints a one-line install hint and exits non-zero. Light lane never depends on it. |
| Thermal accumulation (end-of-cell GPU temp >85°C) | Recorded in JSON. Renderer flags affected cells with a warning footnote. Harness does **not** auto-pause — accepted as a known issue per `performance.md` § Methodology. |
| Bash matrix output drifts from new harness | Bash matrix stays runnable. The parity gate (Testing § Smoke) is the deletion gate. |

## Testing

**Rust unit (sub-second):**
- `tests/extract_metrics.rs`: parses real-runner-output fixtures.
- `tests/registry_filter.rs`: (arch, registry) → expected combo list.
- `tests/run_dir_layout.rs`: round-trip a synthetic JSON run through write/read; catches schema drift between Rust and Python.

**Python unit (seconds):**
- `oracle/bench/tests/test_renderer.py`: renderer is a pure function; fixtures → checked-in golden markdown.
- `oracle/bench/tests/test_perplexity_math.py`: synthetic logits + targets → analytically expected ppl.
- `oracle/bench/tests/test_golden_diff.py`: exact-match and BLEU/chrF on synthetic outputs.
- `oracle/bench/tests/test_hipfire_adapter.py`: fakes the hipfire subprocess; asserts version-pin gate refuses mismatches.
- `oracle/bench/tests/test_schema.py`: every JSON the Python writers produce must validate against the same `schema.py` the Rust side reads.

**GPU smoke (manual, in `tests/gfx1100/`):**
- `bench_smoke.sh`: one combo (qwen3.5-0.8b BF16) end-to-end through perf + quality + render. ~2 minutes. The "did the harness even survive a real run" check.
- `bench_parity.sh`: runs both the new Rust orchestrator and the existing `bench_matrix.sh` on a pinned sha. Asserts every cell agrees within ±3%. Run once at Phase 1 ship; rerun after any change to perf-extraction logic. **This is the gate for deleting the bash matrix.**
- `bench_hipfire_smoke.sh`: hipfire adapter on one combo. Gated by `check-versions.sh`.

**Test ordering during dev:**
1. Rust unit (sub-second).
2. Python unit (seconds).
3. Renderer goldens (seconds).
4. `bench_smoke.sh` (~2 min, GPU).
5. `bench_parity.sh` (~5 min, GPU + warm bakes).
6. Heavy-lane runs (manual, hours).

**Explicitly not tested in Phase 1:**
- Numeric perplexity values per model (no golden ppl in the repo). The first published quality table becomes the reference; future regressions are caught by `render diff`, not by checked-in test gates.
- hipfire output stability across versions. The pinned-version gate makes this a config concern, not a test concern.

## Out-of-scope items revisited (deferred to later phases)

- llama.cpp adapter (Phase 2). Architecture leaves a `external/llama_cpp.py` slot and `tools/external/llama-cpp-version.txt`; nothing else needs to change.
- vLLM adapter (Phase 3). Same shape: `external/vllm.py` + version pin. Server-mode coordination is the Phase-3 unknown.
- Cross-engine quality comparison. Requires per-engine tokenizer alignment, prompt-template normalization, and stop-token harmonization. Its own design doc when we want it.
- Auto-rebuild of pinned external binaries (Phase 1 is bring-your-own-built-from-the-pinned-commit per `tools/external/check-versions.sh`).
- Cross-arch combined-render (`gfx1100 + gfx1150 + sm86 + apple-m4` in one table). Requires data from those machines first.

## Estimated effort

Phase 1 implementation: ~2 weeks of focused work. The single largest cost is generalizing the `--teacher-forced` runner path off the CUDA Llama lane onto every shipping HIP model.
