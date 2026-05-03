# Feature Compatibility Doc + README Slim — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land a runtime-feature compatibility doc and slim the 596-line README into a discoverable landing page that delegates to topic-specific docs.

**Architecture:** Seven commits in a single PR. Content-bearing docs are written first; README slim is the LAST commit so the link index never points at non-existent files. No code changes, no tests beyond manual link-walks.

**Tech Stack:** Markdown only.

**Working directory:** `/home/deano/projects/SuperSonicBase-feature-matrix` (worktree on branch `docs/feature-matrix`).

**Reference docs already on the branch:**
- Spec: `docs/superpowers/specs/2026-05-03-feature-compatibility-and-readme-slim-design.md`
- Existing README: `README.md` (596 lines, 5 per-arch tables, multiple H2 sections)
- Existing performance doc: `docs/performance.md` (710 lines)
- Existing topic docs to cross-reference: `docs/dflash.md`, `docs/bake-distribution.md`, `docs/certified-kv-audit-map.md`, `docs/research/2026-05-03-specprefill-*.md`

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `docs/supported-matrix.md` | NEW | Model × quant × arch tables, verbatim from README §Supported Matrix |
| `docs/build-and-run.md` | NEW | Per-backend build commands, validated commands per arch |
| `docs/testing.md` | NEW | E2E test runner, adding tests, prerequisites |
| `docs/specprefill.md` | NEW | Short user-facing SpecPrefill summary |
| `docs/feature-compatibility.md` | NEW | Runtime features, compat grid, picker recipes |
| `docs/performance.md` | MODIFY | Append "Runtime feature impact" section |
| `README.md` | MODIFY | Slim to ~80 lines, replace inline content with link index |

---

## Task 1: `docs/supported-matrix.md` — verbatim move of README §Supported Matrix

**Files:**
- Create: `docs/supported-matrix.md`
- Read: `README.md` lines 9–195 (the existing §Supported Matrix block).

The new doc preserves every per-arch table, every footnote, and every prose paragraph between them. No reformatting. No content changes. Only addition: a one-paragraph header explaining what this doc is, and a back-link to `README.md`.

- [ ] **Step 1: Read the source range exactly**

Run: `sed -n '9,195p' README.md > /tmp/supported_matrix_excerpt.md`
Then `wc -l /tmp/supported_matrix_excerpt.md` — expect 187 lines.

- [ ] **Step 2: Create the new doc with header + verbatim body**

Create `docs/supported-matrix.md` with this exact content. The header is new; the body from `## Supported Matrix` (line 9) through the end of the last footnote of `### Metal on apple-m4` (line ~195) is verbatim from `README.md`.

```markdown
# Supported Matrix

Which models, quantization lanes, and runtime features are validated on
which GPU architecture. The cells below track *correctness* — see
[docs/performance.md](performance.md) for measured decode throughput
and [docs/feature-compatibility.md](feature-compatibility.md) for the
runtime-feature compatibility grid.

<INSERT VERBATIM the contents of README.md lines 9 through the end of the
"### Metal on `apple-m4`" subsection (line ~195). Do NOT alter any
table, footnote, or paragraph. Replace only the original H2 line
"## Supported Matrix" — drop it; the file's H1 above already serves
that role. Keep all the H3 subsection headers (### HIP on gfx1100, etc.)
in their existing positions.>
```

To do this concretely:
1. Open `README.md`.
2. Copy lines 11–195 (everything BETWEEN the H2 "## Supported Matrix" and the next H2 "## Quick Start"). The H3 subsections come along.
3. Paste them after the `# Supported Matrix\n\n<header paragraph>\n\n` you wrote at the top of `docs/supported-matrix.md`.

- [ ] **Step 3: Verify the move preserved every footnote**

Run: `grep -c "^¹\|^²\|^³\|^⁴\|^⁵\|^⁶\|^⁷\|^⁸\|^⁹\|^¹⁰\|^¹¹\|^¹²" docs/supported-matrix.md`
Expected: matches the same count in the source range. The README has 12 numbered footnotes across the per-arch tables; all 12 must appear in the new doc.

Run: `diff <(sed -n '11,195p' README.md) <(sed -n '7,$p' docs/supported-matrix.md)`
Expected: clean diff (no changes besides whitespace at the start/end and the dropped "## Supported Matrix" line).

- [ ] **Step 4: Verify all internal links inside the moved content still point at real targets**

Run: `grep -oE "\(docs/[a-z0-9_-]+\.md\)" docs/supported-matrix.md | sort -u`
Expected: every linked target exists. Manually confirm each with `ls`.

- [ ] **Step 5: Commit**

```bash
git add docs/supported-matrix.md
git commit -m "docs: extract Supported Matrix from README into docs/supported-matrix.md"
```

---

## Task 2: `docs/build-and-run.md` — consolidate build & validated commands

**Files:**
- Create: `docs/build-and-run.md`
- Read: `README.md` line ranges:
  - `## Quick Start` (lines 196–218): the build + run example.
  - `## Producing And Publishing Bakes` (lines 219–248).
  - `## CUDA` and `### Build requirements` and `### Validated commands` (lines 249–443).
  - `## Metal` and `### Metal validation` (lines 444–523).

This doc is a verbatim move of those sections. Every code block, footnote, and validated-command example is preserved. Reformatting is limited to a new H1 header + a one-paragraph intro pointing back to the README.

- [ ] **Step 1: Capture the source ranges**

Run:
```bash
sed -n '196,218p' README.md > /tmp/quickstart.md
sed -n '219,248p' README.md > /tmp/bakes.md
sed -n '249,443p' README.md > /tmp/cuda.md
sed -n '444,523p' README.md > /tmp/metal.md
wc -l /tmp/quickstart.md /tmp/bakes.md /tmp/cuda.md /tmp/metal.md
```
Expected: 23 + 30 + 195 + 80 = 328 lines total.

- [ ] **Step 2: Create the new doc**

Create `docs/build-and-run.md` with:

```markdown
# Build and Run

Per-backend build commands and validated `supersonic` invocations for
each (model, GPU arch) combination. The
[Supported Matrix](supported-matrix.md) lists which combinations are
validated; this doc shows how to run them.

For runtime-feature flags (DFlash, SpecPrefill, KV-FP8, MoE prefetch,
VMM, etc.) see [docs/feature-compatibility.md](feature-compatibility.md).

## Quick Start

<paste lines 197-218 of README.md verbatim — the H2 "## Quick Start"
becomes our H2, and the build+run example follows>

## Producing and Publishing Bakes

<paste lines 220-248 of README.md verbatim, dropping the original H2
line and using our H2 above instead>

## CUDA

<paste lines 250-443 of README.md verbatim, dropping the original H2
"## CUDA" line>

## Metal

<paste lines 445-523 of README.md verbatim, dropping the original H2
"## Metal" line>
```

To do this concretely: open `README.md`, copy lines 196–523 in one shot, paste, and remove the four original H2 lines (`## Quick Start`, `## Producing And Publishing Bakes`, `## CUDA`, `## Metal`) since the new doc's H2 markers were already added in the template above.

- [ ] **Step 3: Verify content preservation**

Run:
```bash
diff <(sed -n '196,523p' README.md | grep -v "^## ") <(sed -n '/^## Quick Start/,$p' docs/build-and-run.md | grep -v "^## ")
```
Expected: clean (only differences should be the new intro paragraph at the top of `docs/build-and-run.md` which is excluded by the `sed` start tag).

- [ ] **Step 4: Verify cross-references**

Run: `grep -oE "\(docs/[a-z0-9_-]+\.md\)" docs/build-and-run.md | sort -u`
Expected: each linked file exists. The intro references `supported-matrix.md` (paths inside docs/ are relative — link should be `(supported-matrix.md)` not `(docs/supported-matrix.md)`). Fix any inline links accordingly: when this doc is in `docs/`, links to other docs in `docs/` should be relative (e.g. `(performance.md)` not `(docs/performance.md)`).

Run: `grep "(docs/" docs/build-and-run.md`
Expected: NO hits. All within-docs links should be relative.

- [ ] **Step 5: Commit**

```bash
git add docs/build-and-run.md
git commit -m "docs: extract Quick Start / build / CUDA / Metal sections into docs/build-and-run.md"
```

---

## Task 3: `docs/testing.md` — move E2E Tests section

**Files:**
- Create: `docs/testing.md`
- Read: `README.md` lines 524–595 (the `## E2E Tests` section).

The README section already covers: how to run tests, adding tests for a new machine, prerequisites, configuration, known issues. This is a verbatim move with a new H1 + intro and added pointers to the SpecPrefill / Phase B parity tests landed in PR #177.

- [ ] **Step 1: Read the source range**

Run: `sed -n '524,595p' README.md > /tmp/e2e.md && wc -l /tmp/e2e.md`
Expected: ~72 lines.

- [ ] **Step 2: Create the new doc**

Create `docs/testing.md` with:

```markdown
# Testing

End-to-end test runner, prerequisites, and notes on adding tests for a
new machine. For unit tests run via `cargo test`, see the per-crate
README files (`crates/runner/`, `crates/kernel-ffi/`, etc.).

<paste lines 525-595 of README.md verbatim, dropping the original H2
"## E2E Tests" line — the new file's H1 above replaces it>

## Per-feature parity tests

Several runtime features ship a Rust integration test that shells out
to the `supersonic` binary and asserts last-step logits parity (or text
parity) between dense and feature-on runs. Run them all with:

\`\`\`bash
cargo test -p runner --release --test specprefill_qwen35_9b_parity \
    --test specprefill_rope_indirect_parity \
    --test specprefill_lookahead_attention_parity \
    --test qwen36_moe_kv_fp8_parity \
    -- --nocapture --test-threads=1
\`\`\`

Each test self-skips when its required model dirs aren't set in the
environment (e.g. `SUPERSONIC_QWEN35_9B_DIR`,
`SUPERSONIC_QWEN35_0_8B_DIR`, `SUPERSONIC_QWEN36_35B_A3B_DIR`). See
[feature-compatibility.md](feature-compatibility.md) for the full list
of feature flags each test exercises.
```

- [ ] **Step 3: Verify content preservation**

Run:
```bash
diff <(sed -n '525,595p' README.md) <(sed -n '/^### Running tests/,/^## Per-feature parity tests/p' docs/testing.md | sed '$d')
```
Expected: clean diff.

- [ ] **Step 4: Commit**

```bash
git add docs/testing.md
git commit -m "docs: extract E2E Tests section into docs/testing.md and add parity-test runner"
```

---

## Task 4: `docs/specprefill.md` — short user-facing summary

**Files:**
- Create: `docs/specprefill.md`
- Read: `crates/runner/src/main.rs` (the SpecPrefill CLI flags) and `docs/research/2026-05-03-specprefill-phase-a2-cross-target.md` for the headline numbers.

This is a NEW doc — not a move. ~80 lines. Audience: a user who saw the SpecPrefill name in the docs index and wants to know "should I turn this on?"

- [ ] **Step 1: Create the doc**

Create `docs/specprefill.md` with:

```markdown
# SpecPrefill

Speculator-driven sparse target prefill for long-prompt TTFT
optimization, based on [arXiv 2502.02789](https://arxiv.org/abs/2502.02789).
Currently shipping for Qwen3.5-9B target + Qwen3.5-0.8B draft on HIP
(gfx1100). Greedy decode only.

## When to use it

- Prompt is long (≥ 1k tokens) and TTFT (time to first token) matters.
- You're running greedy decode (`--max-new-tokens` ≥ 1, no top-p).
- Target is Qwen3.5-9B (the only target validated this phase).
- Backend is HIP. CUDA / Metal stubs return errors.

## When NOT to use it

- Sampling-based decode (top-p, temperature > 0). Top-5 stability is
  poor at low keep ratios — see Phase A2 measurements.
- Cross-family draft (e.g. Qwen3.5-0.8B → Qwen3.6-MoE). Deferred to
  Phase D — see
  [research/2026-05-03-specprefill-phase-a2-cross-target.md](research/2026-05-03-specprefill-phase-a2-cross-target.md).
- Very long prompts (>8192 tokens). The look-ahead kernel is bounded
  by per-block LDS; longer prompts trip a clear FFI error today.
- 24 GiB GPU + a model larger than Qwen3.5-9B in BF16. Doesn't fit.

## Flags

\`\`\`bash
supersonic --backend hip --model qwen3.5-9b --model-dir /path/to/Qwen3.5-9B \
    --specprefill-draft-dir /path/to/Qwen3.5-0.8B \
    --specprefill-keep-ratio 0.50 \
    --prompt "..." --max-new-tokens 32
\`\`\`

| Flag | Default | Notes |
|---|---|---|
| `--specprefill-draft-dir <path>` | (none) | Required. Same-family draft (Qwen3.5-0.8B). Presence enables SpecPrefill. |
| `--specprefill-keep-ratio <0.05..1.0>` | 0.50 | Fraction of prompt tokens kept by chunked top-K selection. |
| `--specprefill-chunk-size <int>` | 32 | Selection chunk size (paper §3.4). |
| `--specprefill-pool-window <odd int>` | 5 | 1-D smoothing window for importance scores. |
| `--specprefill-lookahead <1..16>` | 4 | Number of look-ahead decode steps on the draft. |
| `--specprefill-always-keep-prefix <int>` | 4 | Force-keep first N tokens (BOS / system). |
| `--specprefill-always-keep-suffix <int ≥ 1>` | 4 | Force-keep last N tokens. Must be ≥ 1 (the first decode logits come from this slot). |
| `--specprefill-unload-draft` | false | Free the draft weights between selection and target prefill (claws back ~1.6 GiB). |

## Quality

Phase A2 measured against the Qwen3.5-9B target with the 0.8B draft on
a 1354-token prompt:

| keep_ratio | argmax match | cossim | top-5 overlap |
|---|---|---|---|
| 0.10 | ✓ | 0.809 | 3/5 |
| 0.30 | ✓ | 0.684 | 3/5 |
| 0.50 | ✓ | 0.927 | 5/5 |
| 0.70 | ✓ | 0.948 | 4/5 |
| 0.90 | ✓ | 0.986 | 5/5 |

Argmax preservation is universal (greedy output's first token is
unchanged). Cossim is a regression backstop. Default `keep_ratio=0.50`
sits in the sweet spot.

## Reference docs

- Feasibility memo: [docs/research/2026-05-03-specprefill-feasibility.md](research/2026-05-03-specprefill-feasibility.md).
- Phase A measurements (4B target): [docs/research/2026-05-03-specprefill-phase-a-results.md](research/2026-05-03-specprefill-phase-a-results.md).
- Phase A2 measurements (9B target): [docs/research/2026-05-03-specprefill-phase-a2-cross-target.md](research/2026-05-03-specprefill-phase-a2-cross-target.md).
- Original paper: [docs/papers/SpecPrefill_arXiv_2502.02789.pdf](papers/SpecPrefill_arXiv_2502.02789.pdf).
```

- [ ] **Step 2: Verify links**

Run: `grep -oE "\([a-z0-9_./-]+\.(md|pdf)\)" docs/specprefill.md | sort -u`
For each, run `ls docs/<path>` (resolve relative) and confirm exists.

- [ ] **Step 3: Commit**

```bash
git add docs/specprefill.md
git commit -m "docs: add user-facing SpecPrefill summary (flags, quality, when to use)"
```

---

## Task 5: `docs/feature-compatibility.md` — the headline new doc

**Files:**
- Create: `docs/feature-compatibility.md`

This is the longest new doc (~250 lines). Five sections per the spec.

- [ ] **Step 1: Create the doc with the full structure**

Create `docs/feature-compatibility.md` with this content:

```markdown
# Feature Compatibility

Compatibility tracker for SuperSonic's runtime features: which combinations
of feature × model × architecture are validated, which are mutually
exclusive, and which use cases each combination targets.

This doc tracks **correctness**. For measured speedups see
[docs/performance.md § Runtime feature impact](performance.md#runtime-feature-impact).
For the model × quant × arch baseline see
[docs/supported-matrix.md](supported-matrix.md).

## How to read

- **✅** = validated end-to-end (parity test or oracle agreement).
- **❌** = explicitly unsupported (CLI guard rejects, or kernel returns
  a clear "not implemented" error).
- **—** = combination doesn't exist (e.g. "FP8 weights for a model
  family that has no FP8 bake").
- **TBM** = to be measured / to be validated.
- Footnotes capture caveats (memory bound, requires specific quant, etc.).

## Runtime features

### 1. Weight quantization

What & why. The base axis: which numeric format the weight matmuls
consume at runtime. BF16 is the reference; INT4 (GPTQ) and FP8 reduce
weight VRAM and (often) compute time at a quality cost. Q4KM is a
GGUF-style INT4 packing used by the CUDA reference path.

Flags: `--int4`, `--fp8-runtime`, `--q4km`, `--q4km-gptq`.

Support per (model, arch): see
[docs/supported-matrix.md](supported-matrix.md). The runtime features
below depend on this baseline being supported first.

### 2. KV-FP8

What & why. KV cache stored in FP8 E4M3 (1 byte) instead of BF16 (2
bytes), halving KV VRAM. Optional sidecar window keeps the most-recent
N tokens in BF16 for higher decode quality. Used when context length
× layers × heads makes the KV cache the binding VRAM constraint.

Flags: `--kv-fp8`, `--kv-fp8-sidecar-window <N>` (Qwen3.6-MoE only).

Support:

| Model            | gfx1100 | gfx1150 | gfx942 | sm86  | apple-m4 |
|------------------|:-------:|:-------:|:------:|:-----:|:--------:|
| qwen3.5-0.8b     |    ✅   |   ✅    |  ✅¹  |  ✅   |    —     |
| qwen3.5-2b       |    ✅   |   ✅    |  ✅¹  |  ✅   |    —     |
| qwen3.5-4b       |    ✅   |   ✅    |  ✅¹  |  ✅   |    —     |
| qwen3.5-9b       |    ✅   |   ✅    |  ✅¹  |  ✅   |    —     |
| qwen3.6-35b-a3b  |    —    |    —    |   —   |   —   |    —     |
| gemma4-e2b       |   ✅²   |    —    |   —   |   —   |    —     |
| gemma4-e4b       |   ✅²   |    —    |   —   |   —   |    —     |
| phi4-mini        |    ✅   |   ✅    |  ✅²  |  ✅   |    —     |
| llama3.1-8b      |    —    |    —    |   —   |  ✅³  |    —     |

¹ gfx942 KV-FP8 uses replayed GPU prefill for the single-sequence path.
² Gemma 4 KV-FP8 requires `--batch-size 1`, cannot combine with `--int4`.
   Phi-4-mini gfx942 KV-FP8 uses the correctness-first single-block fallback.
³ Llama 3.1 8B KV-FP8 only validated alongside `--int8` and certified-KV.

### 3. VMM (virtual KV cache)

What & why. KV cache backed by virtual memory with on-demand resident
mapping. Eviction-to-host + restore lets a workload exceed nominal
VRAM by paging cold KV out. Currently Qwen3.6-MoE only (its 40 layers
+ MTP make KV a dominant footprint).

Flags: `--virtual-kv` (default ON for Qwen3.6-MoE on HIP per
[docs/lowlevel-memory.md](lowlevel-memory.md)).

Support:

| Model            | gfx1100 | gfx1150 | gfx942 | sm86  |
|------------------|:-------:|:-------:|:------:|:-----:|
| qwen3.6-35b-a3b  |    ✅   |    —    |   —    |   —   |

Other models do not enable VMM today; the dense KV allocator is
sufficient.

### 4. SpecPrefill

What & why. Speculator-driven sparse target prefill for long-prompt
TTFT. See [docs/specprefill.md](specprefill.md) for the user-facing
summary and [docs/research/2026-05-03-specprefill-feasibility.md](research/2026-05-03-specprefill-feasibility.md)
for the design.

Flags: `--specprefill-draft-dir <path>` plus tuning flags (see
specprefill.md).

Support:

| Target × Draft                 | gfx1100 | gfx1150 | gfx942 | sm86 |
|--------------------------------|:-------:|:-------:|:------:|:----:|
| qwen3.5-9b BF16 + qwen3.5-0.8b |    ✅   |   TBM   |  TBM   |  ❌¹ |

¹ CUDA bridge returns "not implemented" for the look-ahead and
RoPE-indirect kernels. Validation rejects upfront.

### 5. DFlash speculative decode

What & why. Single-model speculative decode using a small "DFlash"
draft head trained per-target. Runs the megakernel verify path with
B-block batched candidates. Targets the steady-state decode rate, not
TTFT.

Flags: `--dflash`, `--dflash-draft-dir <path>`, `--dflash-block <N>`.

See [docs/dflash.md](dflash.md) for the design and the M3/M4 milestones.

Support:

| Target          | gfx1100 | gfx1150 | gfx942 | sm86 |
|-----------------|:-------:|:-------:|:------:|:----:|
| qwen3.5-9b INT4 |    ✅   |   ✅    |  TBM   |  ❌  |

CUDA support is not currently planned; the B-block fused verify is
HIP-megakernel-specific.

### 6. MoE expert prefetch

What & why. Asynchronous prefetch of MoE expert weights from the rolling
admission window so per-token routed expert dispatch doesn't stall on
weight load. Qwen3.6-MoE only.

Flags: governed by `--qwen36-moe-prefetch-policy <name>` and a few
`--qwen36-moe-*` tuning flags. See
[docs/qwen36-moe-plan.md](qwen36-moe-plan.md).

Support:

| Model            | gfx1100 | gfx1150 | gfx942 | sm86 |
|------------------|:-------:|:-------:|:------:|:----:|
| qwen3.6-35b-a3b  |    ✅   |    —    |   —    |   —  |

### 7. Certified KV (Llama 3.1)

What & why. KV provenance and content certification for Llama 3.1
INT8 on CUDA. Used in retrieval / safety-critical contexts where the
KV cache integrity matters.

Flags: `--certified-kv`, `--certified-kv-shadow-validate`. Requires
`--int8` and Llama 3.1 family. See
[docs/certified-kv-audit-map.md](certified-kv-audit-map.md).

Support:

| Model        | gfx1100 | gfx1150 | gfx942 | sm86 |
|--------------|:-------:|:-------:|:------:|:----:|
| llama3.1-8b  |    —    |    —    |   —    |  ✅  |

CUDA-only; the BF16 step-copy fallback added in PR #177 unblocks the
non-certified BF16 component decode on HIP, but certified mode itself
is still CUDA-specific.

## Feature × feature compatibility

A ✅ means the two flags can be combined; ❌ means the CLI rejects the
combo (or one feature implicitly requires the other to be off).

|                  | Quant | KV-FP8 | VMM | SpecPrefill | DFlash | MoE prefetch | Certified KV |
|------------------|:-----:|:------:|:---:|:-----------:|:------:|:------------:|:------------:|
| **Quant**        |   —   |   ✅¹  | ✅  |     ✅      |   ✅   |     ✅       |     ✅²      |
| **KV-FP8**       |  ✅¹  |   —    | ✅³ |     TBM     |   ❌   |     ✅       |     ✅       |
| **VMM**          |  ✅   |   ✅³  |  —  |     —⁴      |   —⁴   |     ✅       |     —⁴       |
| **SpecPrefill**  |  ✅   |   TBM  | —⁴  |      —      |   ❌   |     —⁴       |     —⁴       |
| **DFlash**       |  ✅   |   ❌   | —⁴  |     ❌      |   —    |     —⁴       |     —⁴       |
| **MoE prefetch** |  ✅   |   ✅   | ✅  |     —⁴      |   —⁴   |      —       |     —⁴       |
| **Certified KV** |  ✅²  |   ✅   | —⁴  |     —⁴      |   —⁴   |     —⁴       |      —       |

¹ KV-FP8 + INT4: see per-model footnotes in §KV-FP8.
² Certified KV requires `--int8` and Llama 3.1.
³ VMM and KV-FP8 are independently configured for Qwen3.6-MoE; the
  sidecar window applies to the resident slice.
⁴ Dash means "no validated combo exists today" — the underlying
  features apply to disjoint model families (e.g. SpecPrefill is
  Qwen3.5-9B; MoE prefetch is Qwen3.6-MoE).

## Picker recipes — "I want to ..."

### ... reduce time-to-first-token on a long Qwen3.5-9B prompt (HIP)

Use SpecPrefill at default keep ratio. The 0.8B draft amortizes
selection in ~700 ms; target then prefills only ~50% of the prompt.

\`\`\`bash
supersonic --backend hip --model qwen3.5-9b --model-dir /path/to/9B \
    --specprefill-draft-dir /path/to/0.8B \
    --prompt "<long prompt>" --max-new-tokens 32
\`\`\`

### ... maximize tokens/sec on Qwen3.5-9B greedy decode (HIP)

Use DFlash. INT4 target, DFlash draft head, B=3 default.

\`\`\`bash
supersonic --backend hip --model qwen3.5-9b --model-dir /path/to/9B \
    --int4 --dflash --dflash-draft-dir /path/to/dflash-draft \
    --prompt "..." --max-new-tokens 64
\`\`\`

### ... fit Qwen3.6-35B-A3B in 24 GiB on gfx1100

Use INT4 GPTQ + VMM (default ON for this model on HIP). KV-FP8
optional for additional KV headroom on long contexts.

\`\`\`bash
supersonic --backend hip --model qwen3.6-35b-a3b \
    --model-dir /path/to/35B-A3B \
    --int4 \
    --prompt "..." --max-new-tokens 32
\`\`\`

### ... run a long-context retrieval QA with Llama 3.1 8B (CUDA)

Use INT8 + certified-KV.

\`\`\`bash
supersonic --backend cuda --model llama3.1-8b --model-dir /path/to/Llama-3.1-8B \
    --int8 --certified-kv \
    --prompt "..." --max-new-tokens 64
\`\`\`

### ... benchmark steady-state decode on Qwen3.5-0.8B (HIP)

No runtime feature flags needed — the persistent megakernel default
path is the fastest.

\`\`\`bash
supersonic --backend hip --model qwen3.5-0.8b --model-dir /path/to/0.8B \
    --prompt "Hello, world" --max-new-tokens 32
\`\`\`

## Where the perf numbers live

This doc is correctness-only. For measured impact (ms/step, % TTFT,
VRAM delta) see
[docs/performance.md § Runtime feature impact](performance.md#runtime-feature-impact).
```

- [ ] **Step 2: Verify the structure rendered**

Run: `grep -c "^### " docs/feature-compatibility.md`
Expected: at least 12 (7 feature subsections + 5 picker recipes).

Run: `grep -c "^| " docs/feature-compatibility.md | head -1`
Expected: > 50 (the per-feature support tables + the feature×feature grid all use markdown tables).

- [ ] **Step 3: Verify all internal links resolve**

Run: `grep -oE "\([a-z0-9_./-]+\.md(#[a-z0-9-]+)?\)" docs/feature-compatibility.md | sort -u`
For each: confirm the doc exists. The `#runtime-feature-impact` anchor will be added in Task 6, so for now this link points at a section that doesn't exist yet — that's intentional and matches the sequencing.

- [ ] **Step 4: Commit**

```bash
git add docs/feature-compatibility.md
git commit -m "docs: feature-compatibility matrix + picker (VMM/SpecPrefill/DFlash/MoE prefetch/KV-FP8/certified-KV)"
```

---

## Task 6: `docs/performance.md` — append "Runtime feature impact"

**Files:**
- Modify: `docs/performance.md` (append a new H2 section at the end of the file).

The new section sits after the existing "How to reproduce" H2 (line ~660) but before the closing of the doc. Each row names a feature, the canonical workload, and the measured delta where data exists.

- [ ] **Step 1: Identify insertion point**

Run: `tail -30 docs/performance.md`
Confirm the last meaningful section ends with a fenced code block containing the `arxiv_v1` retrieval smoke command (or similar — the "How to reproduce" section).

- [ ] **Step 2: Append the new section**

At the END of `docs/performance.md`, append:

```markdown

## Runtime feature impact

Measured delta of each runtime feature on its canonical workload, on
gfx1100 unless noted. **TBM = to be measured** — the feature is shipped
and validated for correctness but the perf measurement script hasn't
landed. Open an issue with reproduction notes if you want to fill one
in.

| Feature | Canonical workload | Baseline | With feature | Source |
|---|---|---|---|---|
| KV-FP8 | qwen3.5-9b INT4 + 1024-token prompt + 16 generated tokens, gfx1100 | ~26 ms/step | ~22 ms/step (15% lower) | tests/gfx1100/bench_matrix.sh |
| KV-FP8 sidecar window | qwen3.6-35b-a3b INT4 + 4096-token context | TBM ms/step | TBM ms/step | (script TBM) |
| VMM | qwen3.6-35b-a3b INT4 + 8192-token context, gfx1100 | OOM (24 GiB exceeded) | runs | tests/gfx1100/bench_qwen36_sparse_caps.py |
| SpecPrefill (keep=0.50) | qwen3.5-9b BF16 + ~225-token prompt, gfx1100 | ~270 ms prefill | TBM ms prefill (speculator amortizes ~700 ms; net win is prompt-length-dependent) | crates/runner/tests/specprefill_qwen35_9b_parity.rs |
| DFlash (B=3) | qwen3.5-9b INT4 greedy decode, gfx1100 | ~32 ms/step | ~12 ms/step (effective; 2.5–3× speedup) | docs/dflash.md M4.3 numbers |
| MoE prefetch | qwen3.6-35b-a3b INT4 decode, gfx1100 | TBM ms/step | TBM ms/step | (script TBM) |
| Certified KV (shadow-validate) | llama3.1-8b INT8 + 1024-token prompt, sm86 | TBM ms/step | TBM ms/step | (script TBM) |

The DFlash numbers are pulled from [docs/dflash.md](dflash.md)'s M4.3
single-pass fused-verify section. The KV-FP8 number is the gfx1100
matrix delta from the per-arch table at the top of this doc — it is
*recorded* there for one workload and *re-stated here* with the
feature label so the picker doc has a one-stop reference.

The "Baseline" column is the comparison point — the dense / no-feature
run on the same hardware, model, and prompt. The "Source" column names
the bench script or test that produced (or will produce) the
measurement.
```

- [ ] **Step 3: Verify the new H2 anchor matches the link in feature-compatibility.md**

GitHub auto-generates anchor IDs from H2 headers as `runtime-feature-impact` (lowercase, dashes for spaces). The link in `docs/feature-compatibility.md` (Task 5) is `(performance.md#runtime-feature-impact)`. They must match.

Run: `grep -i "## runtime feature impact" docs/performance.md`
Expected: one hit. Auto-anchor will resolve to `#runtime-feature-impact`.

- [ ] **Step 4: Commit**

```bash
git add docs/performance.md
git commit -m "docs: add Runtime feature impact section (KV-FP8, VMM, SpecPrefill, DFlash, MoE prefetch, certified-KV)"
```

---

## Task 7: `README.md` — slim to ~80 lines

**Files:**
- Modify: `README.md` (full rewrite of body; keep first H1 and license).

The new README has four sections: H1 + tagline, Quick Start, Documentation index, License. Total ~80 lines.

- [ ] **Step 1: Capture the existing license / footer (if any)**

Run: `tail -20 README.md`
Expected: the file currently ends with the "Known issues" subsection of E2E Tests. There is NO standalone license section in README — license lives in `LICENSE` at repo root. The new README will end with a one-line pointer to `LICENSE`.

- [ ] **Step 2: Replace the entire README body**

Overwrite `README.md` with this content. The H1, tagline (lines 1–7 of the existing README), and the Quick Start commands (lines 196–218) are preserved verbatim. Everything else is replaced by the documentation index.

```markdown
# SuperSonic

Optimized LLM inference with persistent decode megakernels. Each supported
(model, backend, GPU) combination gets a hand-tuned kernel — no fallback to
generic slow paths.

Measured decode throughput: see [docs/performance.md](docs/performance.md).

## Quick Start

\`\`\`bash
# Build with the backend(s) you want compiled in.
# Omit SUPERSONIC_BACKENDS to build the default configured backend set.
SUPERSONIC_BACKENDS=cuda cargo build --release

# Run (auto-bakes weights on first run)
SUPERSONIC_BACKENDS=cuda cargo run --release --bin supersonic -- \\
  --backend cuda \\
  --model qwen3.5-0.8b \\
  --model-dir /path/to/Qwen3.5-0.8B \\
  --prompt "Hello, world" \\
  --max-new-tokens 8
\`\`\`

On first run, SuperSonic bakes the HuggingFace safetensors into an optimized
format at `{model_dir}/.supersonic/v1/`. Subsequent runs load from this baked
format. If a local bake is missing, SuperSonic can download a published bake
from the repo's GitHub releases — see
[docs/bake-distribution.md](docs/bake-distribution.md). Pass `--no-download`
to disable network fetches.

## Documentation

- **[Supported matrix](docs/supported-matrix.md)** — model × quant × arch
  validated combinations with per-arch caveats.
- **[Feature compatibility](docs/feature-compatibility.md)** — runtime
  features (KV-FP8, VMM, SpecPrefill, DFlash, MoE prefetch, certified-KV)
  with the feature×feature grid and a picker for common use cases.
- **[Performance](docs/performance.md)** — measured decode throughput per
  (model, arch, quant) and runtime-feature impact.
- **[Build and run](docs/build-and-run.md)** — per-backend build commands
  and the validated `supersonic` invocation set.
- **[Producing release bakes](docs/bake-distribution.md)** — how to
  produce, sign, and publish bakes for a new model variant.
- **[Testing](docs/testing.md)** — E2E test runner, prerequisites, and
  per-feature parity tests.
- **[DFlash speculative decode](docs/dflash.md)** — Qwen3.5-9B INT4
  speculative decode design and milestones.
- **[SpecPrefill](docs/specprefill.md)** — long-prompt TTFT optimization
  via speculator-driven sparse prefill.
- **[Certified KV (Llama 3.1)](docs/certified-kv-audit-map.md)** — KV
  provenance for retrieval / safety-critical contexts.
- **[Low-level memory](docs/lowlevel-memory.md)** — VMM design, virtual
  KV cache mapping, eviction.

## License

See [LICENSE](LICENSE) at the repo root.
```

- [ ] **Step 3: Verify line count**

Run: `wc -l README.md`
Expected: ~75–85 lines. Acceptance criterion is < 100.

- [ ] **Step 4: Verify every linked doc exists**

Run:
```bash
for link in $(grep -oE "\(docs/[a-z0-9_-]+\.md\)" README.md | tr -d '()'); do
    test -f "$link" && echo "OK   $link" || echo "MISS $link"
done
```
Expected: every link prints `OK`. There should be 10 links (one per bullet in the Documentation section).

If any `MISS` appears, the corresponding new doc was missed in earlier tasks — go back and write it before continuing.

- [ ] **Step 5: Verify the existing links from inside the moved sections still resolve when followed from the new homes**

This is the cross-doc link integrity check. Run:
```bash
for f in docs/supported-matrix.md docs/build-and-run.md docs/testing.md docs/specprefill.md docs/feature-compatibility.md docs/performance.md README.md; do
    echo "=== $f ==="
    grep -oE "\((docs/)?[a-z0-9_./-]+\.md(#[a-z0-9-]+)?\)" "$f" | sort -u
done
```

Inspect the output: every link should resolve from each file's own location. Inside `docs/`, links to other `docs/` files use a path like `(supported-matrix.md)` (relative within docs/). From `README.md`, links use `(docs/supported-matrix.md)`. Fix any drift.

- [ ] **Step 6: Final smoke**

Run: `cargo build --release --bin supersonic 2>&1 | tail -3`
Expected: clean. (No code changes; this just confirms we didn't accidentally break the repo via a stray edit.)

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "docs: slim README to ~80 lines (elevator pitch + Quick Start + link index)"
```

---

## Task 8: Final smoke + open PR

- [ ] **Step 1: Walk every internal link**

Run the link-walk from Task 7 Step 5. For any link with a `#anchor`, confirm the corresponding header exists in the target doc with `grep "^## " <target>`.

- [ ] **Step 2: Confirm the commit chain**

Run: `git log --oneline main..HEAD`
Expected: 8 commits — the design spec (already on the branch as `f8f12cb`), then Tasks 1–7 in order:
- `docs: extract Supported Matrix from README into docs/supported-matrix.md`
- `docs: extract Quick Start / build / CUDA / Metal sections into docs/build-and-run.md`
- `docs: extract E2E Tests section into docs/testing.md and add parity-test runner`
- `docs: add user-facing SpecPrefill summary (flags, quality, when to use)`
- `docs: feature-compatibility matrix + picker (VMM/SpecPrefill/DFlash/MoE prefetch/KV-FP8/certified-KV)`
- `docs: add Runtime feature impact section (...)`
- `docs: slim README to ~80 lines (elevator pitch + Quick Start + link index)`

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin docs/feature-matrix
```

Then:

```bash
gh pr create --title "docs: feature compatibility matrix + README slim" --body "$(cat <<'EOF'
## Summary

- Adds \`docs/feature-compatibility.md\` covering all runtime features
  (KV-FP8, VMM, SpecPrefill, DFlash, MoE prefetch, certified-KV) with
  per-feature support tables, a feature×feature pairwise grid, and a
  picker section for common use cases.
- Slims README.md from 596 lines to ~80 (elevator pitch, Quick Start,
  link index). The 5 per-arch matrices, build/CUDA/Metal sections, and
  E2E Tests section move into focused docs.
- Extends docs/performance.md with a "Runtime feature impact" section
  that summarises per-feature speedups in one table.

Spec: docs/superpowers/specs/2026-05-03-feature-compatibility-and-readme-slim-design.md
Plan: docs/superpowers/plans/2026-05-03-feature-compatibility-and-readme-slim.md

## Test plan

- [x] Every internal link resolves from its source doc.
- [x] No content from the original README is silently dropped — each
      paragraph either has an obvious new home or is intentionally cut
      (the Producing-bakes overlap with bake-distribution.md, etc.).
- [x] \`cargo build --release --bin supersonic\` clean (no code changes).
- [ ] Codex review bot P1/P2 issues addressed.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Address Codex review comments**

Per `~/.claude/projects/-home-deano-projects-SuperSonic/memory/reference_codex_review_bot.md`:
```bash
gh api repos/DeanoC/SuperSonic/pulls/<N>/comments | \
  jq '.[] | "--- " + .path + ":" + (.line // .original_line | tostring) + " ---\n" + .body'
```

Address P1s before requesting merge. P2s are quality nice-to-haves.

---

## Self-review checklist

Run through this before declaring the plan ready:

- [x] **Spec coverage:** Every section of the design spec maps to a task above.
  - Spec §What ships items 1-5 → Tasks 1-5.
  - Spec §What ships item 6 (performance.md) → Task 6.
  - Spec §What ships item 7 (README) → Task 7.
  - Spec §Sequencing → Tasks 1-7 in the specified order.
  - Spec §Acceptance criteria → Task 7 Step 3 (line count), Task 7 Step 4 (every link), Task 8 Step 1 (link-walk).
- [x] **Placeholder scan:** No "TBD" / "TODO" / "fill in" inside the plan steps — only "TBM" inside the doc cells, which is intentional content.
- [x] **Type / link consistency:** Anchor format `#runtime-feature-impact` (Task 5 link, Task 6 anchor) is consistent.
- [x] **Sequencing valid:** README slim (Task 7) is last; every doc it links to is created in Tasks 1-6.
