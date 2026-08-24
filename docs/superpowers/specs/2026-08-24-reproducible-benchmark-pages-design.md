# Reproducible Benchmark Pages Design

**Date:** 2026-08-24

**Status:** Approved design pending implementation planning

## Purpose

Add a durable benchmark system for SuperSonic that serves two development
cadences:

- a quick suite that completes within 10 minutes on the benchmark GPU; and
- an explicitly triggered full suite that may run for up to six hours after
  feature updates or overnight.

Both suites must measure performance and deterministic model quality, preserve
the evidence needed to reproduce a result, and publish reviewable comparison
pages. The system must remain narrow: it benchmarks the supported Qwen3.8-27B
GQH path and pinned peer engines without expanding the public runner contract.

## Publication Model

Validated result records are committed as versioned data. A deterministic
generator builds a static benchmark site from those records for GitHub Pages.
The generated site is disposable output rather than a second source of truth.
Pages CI validates all input records and rebuilds the site, preventing
hand-edited performance claims from diverging from their evidence.

Trusted GPU jobs produce candidate result bundles. Publication is a separate
step: a bundle is eligible to be committed and published only after schema and
cross-record validation pass. An incomplete or diagnostic run can be retained
outside the published data set but cannot produce an aggregate or headline.

The site provides:

- a landing page with the latest qualified SuperSonic performance and quality
  status;
- trend pages separated by architecture, artifact, workload, generation mode,
  and cache state;
- peer comparison tables with explicit comparability status and reasons;
- per-run pages with commands, environment, raw samples, hashes, failures, and
  reproduction instructions; and
- a methodology page generated from versioned suite and schema metadata.

## Repository Architecture

`benchmarks/` owns the benchmark data contract:

- suite manifests and case selection;
- deterministic quality cases and their expected results;
- narrow peer-engine adapter definitions;
- JSON schemas; and
- validated, committed result records.

Inputs in the repository must remain small and reviewable. Model and weight
artifacts remain external and are identified in records by safe names and
cryptographic digests.

A single narrow Python tool under `tools/` exposes four operations:

- `run` performs preflight and executes a named suite;
- `validate` checks candidate or committed records;
- `compare` determines valid series and peer comparability; and
- `render` deterministically generates the static site.

The tool orchestrates existing public CLIs in isolated processes. It must not
introduce another Rust model execution path or change the `supersonic` CLI
contract. Engine-specific behavior is confined to small adapters that emit the
common result schema.

## Suite Profiles

### Quick

The quick suite has a hard 10-minute budget on the benchmark GPU. It uses a
representative deterministic quality subset, a compact set of prompt and
generation shapes, and fewer performance repetitions. It is suitable for
local development and GPU smoke CI while still producing schema-valid evidence.

### Full

The full suite has a hard six-hour budget and runs only when explicitly
requested after a feature update or as an overnight job. It executes the
complete quality corpus, short- and long-context performance shapes, multiple
generation lengths, ordinary and NextN/MTP modes, peer engines, and enough
repetitions to characterize variance.

When the time budget expires, the harness stops scheduling new cases, safely
finishes or terminates the active case according to its declared case timeout,
preserves completed evidence, and marks the suite incomplete. It must not
publish the completed subset as a full-suite aggregate.

## Run Identity and Evidence

Before execution, the harness creates an immutable run manifest. Every result
records:

- repository commit and dirty-tree state;
- build profile, target, compiler, and relevant build flags;
- ROCm, HIP, driver, and engine versions;
- physical GPU identity, logical mapping, and target architecture;
- artifact, tokenizer, configuration, and chat-template identities and
  digests;
- suite, case, schema, and adapter versions;
- exact prompt or stable case identifier, context limit, generation limit,
  chat mode, and greedy decoding policy;
- warmup count, measured-run count, timestamps, and raw per-run samples;
- generated token identifiers and deterministic quality evidence;
- declared clock, power, thermal, and cache policies; and
- completion status and structured errors.

Records reject absolute local paths, secrets, and unbounded environment dumps.
Only an allowlist of environment settings that can affect execution is stored.
Result files are written atomically so an interrupted write cannot appear
valid.

## Performance Methodology

Performance cases report measurement boundaries explicitly and retain raw
samples. Metrics include, when supported by a case:

- model-load and startup latency;
- prefill latency and prompt-token throughput;
- time to first token;
- decode latency and generated tokens per second;
- end-to-end latency; and
- peak GPU memory measured by a documented, reliable method.

Aggregates show median, minimum, maximum, dispersion, and sample count. They do
not select the best observed run as the representative value. Quick and full
profiles may use different repetition counts, but each count is declared in
the suite manifest and recorded in every result.

Performance is initially report-only. Repeated baseline runs establish normal
variance before architecture-specific regression thresholds are documented
and made blocking. Quality failures are blocking from the first version.

## Clock, Power, and Thermal Policy

Every performance run records the requested and observed GPU clock, memory
clock, power cap, performance level, temperature, utilization, and driver
state before and after measurement. It also records CPU governor, relevant host
memory information, and the allowlisted execution environment. The full suite
samples relevant clock and thermal telemetry during execution so throttling or
drift is visible.

An official comparable run declares one of two clock policies:

- `locked`: fixed clock and power settings are prepared explicitly by the host
  operator and verified by the harness; or
- `uncontrolled-clocks`: the host cannot lock them, so observed telemetry is
  recorded and the result is excluded from headline and peer speedup claims.

The harness never silently changes privileged host settings. In strict mode it
fails preflight or the affected case when requested clock or power state cannot
be verified, or when observed drift exceeds the suite's declared tolerance.

Run order is deterministic. The full suite may use a recorded seed to
interleave engines or cases and reduce thermal and temporal bias while
remaining exactly reproducible.

## Cache-State Policy

Every performance case declares one cache state, and records from different
states never share a comparison series:

- `cold-load` uses a fresh process and reports model loading and startup
  separately from inference;
- `warm-resident` uses declared warmups before measuring an already resident
  model; and
- prefix or prompt-cache cases declare explicit `empty`, `populated`, and
  `reset` transitions.

The system uses precise terms. It claims a hardware or filesystem cache flush
only when the mechanism is documented and its success can be verified. When
privileged operating-system cache clearing is unavailable, it records that
fact and uses `fresh-process` or `cold-load` rather than claiming a cold cache.

## Deterministic Quality Methodology

Quality evaluation is offline and versioned in the repository. It uses greedy
generation and deterministic expected outcomes, without network data, fuzzy
model judges, or another model as evaluator.

Cases use exact token or text matching where that expresses the requirement.
Constrained tasks may use small deterministic parsers that produce reviewable
structured evidence. Initial categories cover:

- instruction following;
- structured extraction;
- arithmetic and reasoning;
- code completion;
- long-context retrieval;
- chat-template behavior;
- repeated-run determinism; and
- ordinary versus NextN/MTP token equality.

Reports show each case and category result as well as the aggregate. A failed
case cannot be hidden by an aggregate score. Changes to prompts, expected
answers, scoring, or category weights change the suite version and begin a new
quality series.

## Peer Engine Comparisons

The full suite supports pinned external engines from the first version through
narrow adapters. Existing repository pins are reused where applicable, and an
adapter fails preflight if the installed engine does not match its declared
version.

Two records are headline-comparable only when all of the following match:

- physical hardware and controlled clock/power policy;
- model identity and equivalent quantization or artifact semantics;
- tokenizer and chat-template behavior;
- exact prompt or quality case, context, and generation limits;
- greedy decoding, stop behavior, and generated-token accounting;
- cache state and warmup policy; and
- timing boundaries and metric definitions.

The comparison validator, not the page template, decides comparability. A
non-equivalent peer result may remain visible as qualified context, but the site
must display the mismatch reasons and must not calculate or publish a speedup
headline from it.

## Execution and Failure Semantics

Preflight verifies artifacts and digests, engine pins, device selection, clock
policy, required cache-control capabilities, available disk space, suite time
budget, and output safety. A configured artifact or peer engine that is missing
or unreadable is a failure, never a passing skip.

Each case runs in an isolated process with captured standard output, standard
error, timestamps, exit status, and telemetry. The harness treats these as
structured failures when applicable:

- deterministic quality mismatch or ordinary/MTP token mismatch;
- missing, non-finite, or inconsistent samples;
- disagreement between recorded and observed token counts;
- clock drift, thermal throttling, or power-state violation;
- cache-state transition failure;
- incompatible peer semantics;
- process failure or timeout; and
- configured input becoming unavailable during the run.

Interrupted and timed-out suites remain explicitly `incomplete`. Partial data
is diagnostic and cannot silently satisfy a quick or full gate.

## Validation and Site Generation

Validation has two layers. JSON Schema checks types and required fields.
Semantic validation enforces digests, safe paths, suite completeness, sample
counts, finite values, engine pins, series identity, clock and cache policies,
quality evidence, and peer comparability.

Site generation sorts records and fields deterministically and embeds the
schema, suite, and generator versions. Running `render` twice from the same
inputs must produce byte-identical output. Generated HTML is not committed.
Pages CI validates committed source records, renders the deployment artifact,
and publishes it only from the validated default branch.

Public performance numbers remain subject to the active-document evidence
rules. A benchmark page must colocate or directly link the exact run evidence
behind every numeric claim.

## Testing and Gates

CPU-safe tests use fixtures to cover:

- manifest and result schema validation;
- runner and peer output parsing;
- aggregation and variance calculations;
- clock, thermal, power, and cache-policy validation;
- exact quality scoring and category aggregation;
- comparison-series identity and mismatch explanations;
- timeout, interruption, and incomplete-run behavior;
- deterministic rendering;
- absolute-path, secret, malformed-input, and non-finite-number rejection; and
- GitHub workflow and active-document contracts.

GPU smoke tests execute the quick pipeline on a supported configured host. The
explicit overnight workflow executes the full pipeline. Existing CPU-safe
workspace gates, documentation checks, and the serial `gfx1201` artifact and
correctness gate remain prerequisites rather than being duplicated or weakened.

Before publication, the complete applicable gate must pass. Warnings,
configured skips, missing samples, unexplained token mismatches, uncontrolled
clock results presented as comparable, or incomplete suite status are review
blockers.

## Initial Success Criteria

The first implementation is complete when:

1. quick and full manifests are versioned and enforce their 10-minute and
   six-hour budgets;
2. SuperSonic and at least one pinned peer adapter emit the same validated
   result schema;
3. deterministic quality cases block on failures and expose per-case evidence;
4. performance records distinguish clock and cache states and retain raw
   samples;
5. comparison validation prevents invalid speedup claims;
6. candidate GPU output can be validated and committed without unsafe paths;
7. deterministic static pages provide latest results, trends, peer comparisons,
   methodology, and per-run reproduction details; and
8. GitHub Pages deploys only validated default-branch results.

## Explicit Non-Goals

- Changing or broadening the public SuperSonic runner contract.
- Adding nondeterministic sampling or multi-sequence serving benchmarks.
- Downloading models or evaluation data inside benchmark execution.
- Using a model judge or an externally hosted evaluation service.
- Treating unlike artifacts, cache states, clocks, hardware, or measurement
  boundaries as directly comparable.
- Restoring the removed broad benchmark or machine-profile frameworks.
- Automatically applying privileged clock, power, or operating-system cache
  settings.
