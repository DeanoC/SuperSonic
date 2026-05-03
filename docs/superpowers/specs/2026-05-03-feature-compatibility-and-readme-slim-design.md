# Feature compatibility doc + README slim — design

**Branch:** `docs/feature-matrix`
**Status:** design approved 2026-05-03; ready for writing-plans.
**Worktree:** `/home/deano/projects/SuperSonicBase-feature-matrix`

## Goal

Add a runtime-feature compatibility doc (the missing axis in current
documentation) and slim the 700-line `README.md` into a discoverable
landing page that delegates to topic-specific docs. After this PR, a new
user lands on the README, sees the elevator pitch and a Quick Start
command in 30 seconds, and finds everything else via a curated link
index.

## Non-goals

- Cross-arch perf comparisons (already in `docs/performance.md`).
- Quality/picker advice on model selection — that's a separate doc.
- Producing missing measurement numbers — TBM cells are flagged, not
  filled.
- Restructuring existing arch-specific docs (`gfx1150-l2-bypass.md`,
  `phi4-cuda-parity.md`, etc.). They stay as-is and the new docs link
  to them.

## What ships

### New docs

1. **`docs/feature-compatibility.md`** — runtime-feature reference, per-feature
   support tables, feature×feature pairwise compatibility grid, and a
   "I want to ..." picker.

2. **`docs/supported-matrix.md`** — the existing model × quant × arch
   tables + 12 footnotes from README, moved verbatim. README will link
   here.

3. **`docs/build-and-run.md`** — consolidates the build commands and
   validated-command examples currently scattered across README's
   "Quick Start", "CUDA", and "Metal" sections.

4. **`docs/testing.md`** — moves the "E2E Tests" section out of README,
   expanded with how to run the per-feature parity tests
   (qwen36_moe_kv_fp8_parity, specprefill_qwen35_9b_parity,
   specprefill_rope_indirect_parity, etc.).

5. **`docs/specprefill.md`** — short user-facing summary of the
   SpecPrefill feature (flags, 24 GiB constraint, when to use). Points
   at `docs/research/2026-05-03-specprefill-*` for the measurements.

### Modified docs

6. **`docs/performance.md`** — gains one new section "Runtime feature
   impact" with one row per runtime feature on a canonical workload
   (gfx1100). Cells are measured ms/step or % delta where data exists,
   "TBM" otherwise.

7. **`README.md`** — slimmed to ~80 lines: elevator pitch, Quick Start,
   curated link index, license. Existing "Supported Matrix",
   "Producing And Publishing Bakes", "CUDA", "Metal", "E2E Tests"
   sections move into the new docs.

## Structure of `docs/feature-compatibility.md`

```
1. Intro              # what this doc is, what it isn't, cross-ref to README
2. Runtime features   # one subsection per feature:
   - Weight quant         (BF16 / INT4 / FP8 runtime / Q4KM)
   - KV-FP8               (with sidecar window option)
   - VMM                  (Qwen3.6-MoE only currently)
   - SpecPrefill          (Qwen3.5-9B + 0.8B draft, HIP)
   - DFlash               (Qwen3.5-9B INT4, HIP)
   - MoE prefetch         (Qwen3.6-MoE only)
   - Certified KV         (Llama 3.1 INT8, CUDA only)
3. Feature × feature      # NxN pairwise compat grid (~7x7)
4. Picker recipes         # 4-6 "I want X → use these flags" entries
5. Where's the perf data  # one-line pointer to docs/performance.md
```

Each feature subsection has the same shape:
- One-paragraph "what & why".
- CLI flags that turn it on.
- Support table: rows = (model, quant), columns = arch — cells ✅/❌/footnote.
  ~5–8 rows max.
- Caveats / interactions noted as footnotes.

## Structure of slimmed `README.md`

```
# SuperSonic
<elevator paragraph>

## Quick Start
<5-line build + run a 0.8B prompt>

## Documentation
- Supported matrix → docs/supported-matrix.md
- Runtime features → docs/feature-compatibility.md
- Performance numbers → docs/performance.md
- Build & validated commands → docs/build-and-run.md
- Producing release bakes → docs/bake-distribution.md
- DFlash speculative decode → docs/dflash.md
- SpecPrefill → docs/specprefill.md
- Certified KV (Llama 3.1) → docs/certified-kv-audit-map.md
- Testing → docs/testing.md

## License
<existing>
```

## Sequencing

Single PR, multiple commits, in this order to avoid the README ever
pointing at non-existent docs:

1. `docs/supported-matrix.md` — verbatim move of README's matrix tables.
2. `docs/build-and-run.md` — consolidates README's build/CUDA/Metal sections.
3. `docs/testing.md` — moves the E2E Tests section.
4. `docs/specprefill.md` — short user summary.
5. `docs/feature-compatibility.md` — the headline new doc.
6. `docs/performance.md` extension — Runtime feature impact section.
7. `README.md` slim — last, points at every doc that now exists.

## Acceptance criteria

- README is < 100 lines, no per-arch tables inline, no inline build
  commands beyond a single ~5-line Quick Start.
- `docs/feature-compatibility.md` lists all 7 runtime features with
  per-feature support tables and the feature×feature grid.
- `docs/feature-compatibility.md` has at least 4 picker recipes.
- `docs/performance.md` has one new top-level section listing
  per-feature impact, with TBM cells flagged where measurement
  doesn't exist.
- All cross-doc links resolve.
- No content from the original README is lost — every paragraph either
  has an obvious new home or is intentionally cut (e.g., redundant
  with bake-distribution.md).
- `git grep` for the README's existing arch tables finds them in
  `docs/supported-matrix.md` after the PR.

## Risks

- **Drift between README's link list and the actual docs.** Mitigation:
  the link index in the README and the cross-references in the new
  docs are added in the same commit chain; CI doesn't catch broken
  internal links today, so a manual link-walk is part of acceptance.
- **TBM cells in `docs/performance.md` look like the doc is incomplete.**
  Mitigation: explicit "TBM = to be measured" legend so readers know
  it's a tracked gap, not an oversight.
- **README slim lands without one of the linked docs existing.**
  Mitigation: README slim is the LAST commit in the sequence; reviewers
  can verify by clicking each link before merge.
