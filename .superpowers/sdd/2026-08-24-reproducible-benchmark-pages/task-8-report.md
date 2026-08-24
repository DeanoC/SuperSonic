# Task 8 report: GPU and Pages workflows

## RED evidence

Added workflow contracts to `tests/test_ci_workflows.py` before adding the
workflows. The first run was:

```text
python3 -m unittest tests.test_ci_workflows -v
Ran 8 tests ...
FAILED (failures=4)
```

The expected failures were the absent quick, full, and Pages workflows plus the
new CPU fixture-render contract in `ci.yml`.

## GREEN evidence

The focused workflow suite now passes:

```text
python3 -m unittest tests.test_ci_workflows -v
Ran 8 tests in 0.004s
OK
```

The complete CPU-safe Python suite was run once after implementation:

```text
python3 -m unittest discover -s tests -p 'test_*.py' -v
Ran 198 tests in 1.026s
OK
```

Additional checks passed:

- `python3 tools/supersonic-bench.py validate tests/benchmark_fixtures/valid-result-v1.json`;
- CPU fixture rendering with `tools/supersonic-bench.py render`;
- focused benchmark manifest, validation, render, and CLI tests (38 passed);
- `bash -n` over all 18 `run: |` blocks in the three new workflows; and
- `git diff --check`.

## Files

- Added `.github/workflows/benchmark-quick.yml` with the serial gfx1201 quick
  candidate job, static AMD SMI identity capture, idle checks, locked-clock
  verification, strict artifact digest preflight, exact 600-second CLI suite,
  and always-uploaded diagnostics. The GitHub job timeout is 30 minutes.
- Added `.github/workflows/benchmark-full.yml` with the manual serial gfx1201
  full candidate job, exact pinned llama.cpp/version and peer-artifact checks,
  the exact 21600-second CLI suite, and always-uploaded diagnostics. The
  GitHub job timeout is 390 minutes.
- Added `.github/workflows/benchmark-pages.yml` with CPU-only committed-record
  publishability validation before deterministic render, a pinned Pages
  artifact upload, and a deploy job gated to a push on `refs/heads/main` with
  only `pages: write` and `id-token: write` permissions.
- Updated `.github/workflows/ci.yml` to validate a committed fixture and render
  a fixture site in CPU CI, plus trigger on benchmark sources/workflows.
- Updated `tests/test_ci_workflows.py` with RED/GREEN contracts for the new
  workflow behavior.
- Kept the existing serial correctness/MTP gate and legacy throughput loop in
  `.github/workflows/qwen38-gfx1201.yml`, labeling the loop diagnostic-only
  until the new records expose complete `warmup_runs`, `measured_runs`, and
  `ms_per_tok` parity. No duplicate publication path was introduced.

## Self-review

- All GPU actions use full pinned SHAs; GPU jobs have no commit, push, or Pages
  deploy operation.
- Both GPU jobs use the existing selector and artifact preflight boundaries,
  preserve physical-to-logical mapping, derive identity from captured static
  AMD SMI JSON, require three idle samples, and verify host-prepared locked
  clock/power/performance state without changing privileged host settings.
- Quick remains SuperSonic-only; full requires the pinned llama.cpp binary,
  version file, peer artifact, and independent SHA-256 checks.
- Pages has no `pull_request_target` trigger and its deploy job cannot run on a
  pull request, manual branch run, or non-main push.
- CPU CI remains on `ubuntu-24.04` and contains no GPU device selection or
  device mounts.

## Concerns and limits

- The self-hosted GPU jobs were not executed locally: this worktree has no
  configured AMD SMI device, Qwen GQH artifacts, locked-clock variables, or
  llama.cpp peer installation. Static contracts and shell syntax were checked
  instead; configured missing inputs fail closed in the workflows.
- `benchmarks/results` has no committed publishable record in this worktree,
  so Pages is intentionally blocked until a validated record is promoted.
- The old qwen reproducibility telemetry remains until benchmark records carry
  the legacy field parity required to remove it safely.

## Round-one fix RED evidence

The new provenance and workflow contract tests were run before the fix:

```text
python3 -m unittest tests.test_amd_smi_provenance tests.test_ci_workflows -v
Ran 15 tests ...
FAILED (failures=5, errors=4)
```

The errors were the intentionally absent provenance merger; the failures
covered the old AMD SMI capture commands, Pages baseline detection, and the
non-shared GPU concurrency groups.

## Round-one fix GREEN evidence

The merged capture path now joins physical BDF identity from
`amd-smi static --asic --bus --json` to the logical HIP mapping from
`amd-smi list -e --json`. It rejects mismatched or duplicate stable identities,
records both source SHA-256 digests, and feeds the same merged JSON to the
selector and benchmark GPU provenance resolver. Fixture/selector/workflow
coverage passed:

```text
python3 -m unittest tests.test_ci_workflows tests.test_r9700_helpers tests.test_amd_smi_provenance -v
Ran 30 tests in 0.040s
OK
```

Pages now detects `benchmarks/results/**/*.json` before validation, reports and
cleanly skips a README-only baseline, while any JSON (including a malformed
first record) still reaches blocking validation. Quick, full, and the retained
gfx1201 correctness workflow use the identical
`benchmark-gfx1201-device-${{ github.repository }}` group with
`cancel-in-progress: false`.

The final CPU suite and CI-adjacent checks passed:

```text
python3 -m unittest discover -s tests -p 'test_*.py' -v
Ran 206 tests in 0.960s
OK

cargo fmt --all --check
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
Finished `dev` profile [unoptimized + debuginfo] in 1m 00s

python3 tools/supersonic-bench.py validate tests/benchmark_fixtures/valid-result-v1.json
python3 tools/supersonic-bench.py render tests/benchmark_fixtures target/benchmarks/site-fixtures-round1
publishable=false; valid=true; fixture site rendered

bash -n over all workflow run blocks
bash syntax ok: 30 workflow run blocks
python3 tools/check-active-docs.py
active public documentation check passed
python3 tools/check-tool-inventory.py
tool inventory ok: 1 runner binaries classified
git diff --check
```

## Round-one files and self-review

- Added `tools/merge-amd-smi-provenance.py`, two AMD SMI-shaped fixtures under
  `tests/amd_smi_fixtures/`, and `tests/test_amd_smi_provenance.py`.
- Updated quick/full/qwen capture steps to retain both raw source records and
  the merged provenance artifact, with bounded AMD SMI commands.
- Updated Pages detection/output/deploy guards and CPU CI's focused provenance
  test command.
- Updated workflow contracts for malformed-first-record handling, README-only
  bootstrap, source mapping, and shared per-device serialization.

Self-review found no GPU workflow commit, push, or deploy operation; all action
references remain full SHAs; the PR path remains CPU-only; host clock setup is
still external and only the requested locked state is verified; and old
diagnostic telemetry remains retained until complete parity is proven. The
self-hosted GPU jobs remain unexecuted locally because this worktree has no AMD
SMI device or configured Qwen/peer artifacts; missing configured inputs fail
closed.
