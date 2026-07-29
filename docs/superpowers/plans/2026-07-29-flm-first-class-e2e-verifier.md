# FLM First-Class End-to-End Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one resumable command that exports or reuses a strictly valid Qwen3.6 35B-A3B native INT4 FLM, verifies every payload, and proves GPU inference through SuperSonic without an HF fallback.

**Architecture:** A standalone Python runner in SuperSonic orchestrates geo-quant through its public CLI and then delegates inference collection to the existing Qwen3.6 benchmark harness. Small command builders, artifact-decision helpers, and structured report checks are unit tested without a GPU; the final gate performs the real ROCm export/load/inference run.

**Tech Stack:** Python 3 standard library (`argparse`, `json`, `os`, `subprocess`, `pathlib`), `unittest`, geo-quant's existing Python environment, SuperSonic Rust/HIP release binary, ROCm GPUs.

## Global Constraints

- SuperSonic must invoke geo-quant through its public CLI and must not import geo-quant modules.
- The canonical producer profile is exactly `supersonic-qwen36-moe-native-int4`.
- Export uses INT4, group size 128, `--flm-only`, and `--hf-compat-assets omit`.
- Reuse is allowed only after strict structural validation succeeds.
- Correctness validation after export or reuse includes `--verify-payload-hashes`.
- A failed replacement export must not overwrite an existing FLM.
- SuperSonic inference receives only the FLM path; it receives no HF source path, explicit model, or INT4 selection.
- Success requires at least one generated token, native INT4 direct coverage, zero BF16 fallback, decode readiness, and measured nonzero FLM transfer throughput.
- The canonical path uses pageable H2D on this ROCm 7.1.1 machine; hipFile performance is out of scope.

---

### Task 1: Artifact Export And Reuse Policy

**Files:**
- Create: `tests/gfx1100/run_qwen36_flm_first_class_e2e.py`
- Create: `tests/test_qwen36_flm_first_class_e2e.py`

**Interfaces:**
- Consumes: geo-quant's `scripts/quantize_qwen36_int4.py` and `geoquant.formats.flm_validate` CLIs.
- Produces: `ArtifactAction`, `export_command(args, output)`, `validate_command(args, artifact, verify_payload_hashes)`, `choose_artifact_action(args)`, and `prepare_artifact(args) -> Path`.

- [ ] **Step 1: Write failing command-construction tests**

Add imports for the runner by file path, construct a `types.SimpleNamespace`,
and assert exact producer and validator commands:

```python
def test_builds_strict_native_int4_export_command(self):
    args = self.args()
    self.assertEqual(
        runner.export_command(args, Path("/models/output.partial.flm")),
        [
            "/venv/bin/python",
            "scripts/quantize_qwen36_int4.py",
            "--bf16",
            "/models/Qwen3.6-35B-A3B",
            "--flm-out",
            "/models/output.partial.flm",
            "--flm-only",
            "--device",
            "cuda",
            "--bits",
            "4",
            "--group-size",
            "128",
            "--hf-compat-assets",
            "omit",
            "--flm-validate-profile",
            "supersonic-qwen36-moe-native-int4",
        ],
    )

def test_builds_payload_verifying_validator_command(self):
    args = self.args()
    self.assertEqual(
        runner.validate_command(
            args,
            Path("/models/output.flm"),
            verify_payload_hashes=True,
        ),
        [
            "/venv/bin/python",
            "-m",
            "geoquant.formats.flm_validate",
            "/models/output.flm",
            "--profile",
            "supersonic-qwen36-moe-native-int4",
            "--verify-payload-hashes",
        ],
    )
```

- [ ] **Step 2: Run the command tests and confirm the red state**

Run:

```bash
python3 -m unittest \
  tests.test_qwen36_flm_first_class_e2e.Qwen36FlmFirstClassE2ETests.test_builds_strict_native_int4_export_command \
  tests.test_qwen36_flm_first_class_e2e.Qwen36FlmFirstClassE2ETests.test_builds_payload_verifying_validator_command
```

Expected: `ERROR` because the runner file or command functions do not exist.

- [ ] **Step 3: Implement parser defaults and command builders**

Create the runner with these constants and builders:

```python
ROOT = Path(__file__).resolve().parents[2]
BENCH_SCRIPT = ROOT / "tests" / "gfx1100" / "bench_qwen36_he_supersonic.py"
STRICT_PROFILE = "supersonic-qwen36-moe-native-int4"
DEFAULT_HF_SOURCE = Path("/mnt/data/models/Qwen3.6-35B-A3B")
DEFAULT_GEOQUANT_ROOT = Path("/home/deano/projects/geo-quant")
DEFAULT_GEOQUANT_PYTHON = Path(
    "/home/deano/projects/geo-quant/.venv-rocm/bin/python"
)
DEFAULT_FLM = Path(
    "/mnt/data/runs/geo-quant/"
    "qwen36-35b-a3b-supersonic-native-int4-current.flm"
)

def export_command(args: argparse.Namespace, output: Path) -> list[str]:
    return [
        str(args.geoquant_python),
        "scripts/quantize_qwen36_int4.py",
        "--bf16", str(args.hf_source),
        "--flm-out", str(output),
        "--flm-only",
        "--device", args.quant_device,
        "--bits", "4",
        "--group-size", "128",
        "--hf-compat-assets", "omit",
        "--flm-validate-profile", STRICT_PROFILE,
    ]

def validate_command(
    args: argparse.Namespace,
    artifact: Path,
    *,
    verify_payload_hashes: bool,
) -> list[str]:
    command = [
        str(args.geoquant_python),
        "-m", "geoquant.formats.flm_validate",
        str(artifact),
        "--profile", STRICT_PROFILE,
    ]
    if verify_payload_hashes:
        command.append("--verify-payload-hashes")
    return command
```

The parser exposes `--hf-source`, `--geoquant-root`, `--geoquant-python`,
`--flm`, `--quant-device` (default `cuda`), and `--regenerate`. Preserve the
existing development environment-variable overrides where useful, but default
the geo-quant root to merged `main`, not an old worktree.

- [ ] **Step 4: Run the command tests and confirm green**

Run the Step 2 command.

Expected: 2 tests pass.

- [ ] **Step 5: Write failing artifact-decision tests**

Use `mock.patch.object(runner, "run_command")` so tests do not execute
geo-quant:

```python
def test_missing_artifact_exports_to_partial_then_promotes(self):
    args = self.args(flm=self.tmp_path / "model.flm")
    with mock.patch.object(runner.os, "getpid", return_value=42), \
         mock.patch.object(runner, "run_command") as run, \
         mock.patch.object(runner.os, "replace") as replace:
        result = runner.prepare_artifact(args)

    partial = self.tmp_path / ".model.flm.partial-42"
    self.assertEqual(result, args.flm)
    self.assertEqual(run.call_args_list[0].args[0], runner.export_command(args, partial))
    self.assertEqual(
        run.call_args_list[1].args[0],
        runner.validate_command(args, partial, verify_payload_hashes=True),
    )
    replace.assert_called_once_with(partial, args.flm)

def test_valid_artifact_is_reused_and_hash_verified(self):
    args = self.args(flm=self.existing_flm)
    with mock.patch.object(
        runner, "probe_validation", return_value=True
    ) as probe, mock.patch.object(runner, "run_command") as run:
        result = runner.prepare_artifact(args)

    self.assertEqual(result, args.flm)
    probe.assert_called_once_with(args, args.flm)
    run.assert_called_once_with(
        runner.validate_command(args, args.flm, verify_payload_hashes=True),
        cwd=args.geoquant_root,
        timeout=args.validation_timeout,
        phase="payload validation",
    )

def test_regenerate_preserves_existing_artifact_until_promotion(self):
    args = self.args(flm=self.existing_flm, regenerate=True)
    # Assert no structural reuse probe occurs and os.replace is last.

def test_invalid_artifact_selects_safe_regeneration(self):
    args = self.args(flm=self.existing_flm)
    # probe_validation returns False; assert export targets a sibling partial.
```

- [ ] **Step 6: Run artifact-decision tests and confirm the red state**

Run:

```bash
python3 -m unittest tests.test_qwen36_flm_first_class_e2e -v
```

Expected: command tests pass; decision tests fail because the artifact policy
functions do not exist.

- [ ] **Step 7: Implement phase-aware execution and safe artifact preparation**

Add:

```python
class PhaseError(RuntimeError):
    pass

def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    phase: str,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            timeout=timeout,
            capture_output=capture_output,
        )
    except subprocess.TimeoutExpired as exc:
        raise PhaseError(f"{phase} timed out after {timeout}s") from exc
    if check and result.returncode != 0:
        raise PhaseError(
            f"{phase} failed with exit {result.returncode}: {' '.join(command)}"
        )
    return result

def probe_validation(args: argparse.Namespace, artifact: Path) -> bool:
    result = run_command(
        validate_command(args, artifact, verify_payload_hashes=False),
        cwd=args.geoquant_root,
        timeout=args.validation_timeout,
        phase="structural validation",
        check=False,
        capture_output=True,
    )
    return result.returncode == 0

def partial_artifact_path(artifact: Path) -> Path:
    return artifact.with_name(f".{artifact.name}.partial-{os.getpid()}")

def prepare_artifact(args: argparse.Namespace) -> Path:
    reuse = args.flm.exists() and not args.regenerate
    if reuse and probe_validation(args, args.flm):
        run_command(
            validate_command(args, args.flm, verify_payload_hashes=True),
            cwd=args.geoquant_root,
            timeout=args.validation_timeout,
            phase="payload validation",
        )
        return args.flm

    partial = partial_artifact_path(args.flm)
    if partial.exists():
        raise PhaseError(f"export target already exists: {partial}")
    args.flm.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        export_command(args, partial),
        cwd=args.geoquant_root,
        timeout=args.export_timeout,
        phase="producer export",
    )
    run_command(
        validate_command(args, partial, verify_payload_hashes=True),
        cwd=args.geoquant_root,
        timeout=args.validation_timeout,
        phase="payload validation",
    )
    os.replace(partial, args.flm)
    return args.flm
```

Use separate `--export-timeout` and `--validation-timeout` positive integer
arguments because a 35B export and a full 21+ GB hash pass have different
runtime envelopes. Leave a failed partial file in place.

- [ ] **Step 8: Run artifact policy tests**

Run:

```bash
python3 -m unittest tests.test_qwen36_flm_first_class_e2e -v
```

Expected: all Task 1 tests pass.

- [ ] **Step 9: Commit Task 1**

```bash
git add \
  tests/gfx1100/run_qwen36_flm_first_class_e2e.py \
  tests/test_qwen36_flm_first_class_e2e.py
git commit -m "test: orchestrate reproducible FLM artifacts"
```

---

### Task 2: SuperSonic Execution Evidence

**Files:**
- Modify: `tests/gfx1100/run_qwen36_flm_first_class_e2e.py`
- Modify: `tests/test_qwen36_flm_first_class_e2e.py`

**Interfaces:**
- Consumes: `prepare_artifact(args) -> Path` from Task 1 and the JSON schema produced by `bench_qwen36_he_supersonic.py`.
- Produces: `supersonic_benchmark_command(args, artifact)`, `first_class_errors(payload) -> list[str]`, `validate_benchmark_report(path) -> dict`, and `main(argv) -> int`.

- [ ] **Step 1: Write failing SuperSonic command tests**

```python
def test_supersonic_command_has_no_hf_model_or_quant_override(self):
    args = self.args()
    command = runner.supersonic_benchmark_command(args, args.flm)
    self.assertIn("qwen36-35b-a3b-flm", command)
    self.assertIn(str(args.flm), command)
    self.assertNotIn(str(args.hf_source), command)
    self.assertNotIn("--model", command)
    self.assertNotIn("--quant", command)
    self.assertNotIn("--int4", command)
```

Also assert `--limit 1`, `--n-gen 1`, `--emit-stage-timings`,
`--hal-profile`, and the requested JSON output.

- [ ] **Step 2: Run the command test and confirm red**

Run:

```bash
python3 -m unittest \
  tests.test_qwen36_flm_first_class_e2e.Qwen36FlmFirstClassE2ETests.test_supersonic_command_has_no_hf_model_or_quant_override
```

Expected: fail because `supersonic_benchmark_command` does not exist.

- [ ] **Step 3: Implement the benchmark command**

```python
def supersonic_benchmark_command(
    args: argparse.Namespace,
    artifact: Path,
) -> list[str]:
    return [
        sys.executable,
        str(BENCH_SCRIPT.relative_to(ROOT)),
        "--binary", str(args.binary),
        "--target-profile", "qwen36-35b-a3b-flm",
        "--model-dir", str(artifact),
        "--limit", str(args.limit),
        "--n-gen", str(args.n_gen),
        "--warmup-new-tokens", "1",
        "--no-warmup",
        "--context-size", str(args.context_size),
        "--timeout", str(args.inference_timeout),
        "--emit-stage-timings",
        "--hal-profile",
        "--out-json", str(args.out_json),
    ]
```

If `--flm-virtual-transfer-backend` is supplied to the verifier, append that
exact selector. Do not set a storage backend by default.

- [ ] **Step 4: Run the command test and confirm green**

Run the Step 2 command.

Expected: 1 test passes.

- [ ] **Step 5: Write failing report-validation tests**

Create a complete valid report fixture with:

```python
{
    "resolved_model": "qwen3.6-35b-a3b",
    "summary": {
        "count": 1,
        "flm_weight_modes": ["INT4 native FLM"],
        "flm_ready_for_decode_count": 1,
        "flm_direct_profiles": [{
            "required": 693,
            "raw_dense": 363,
            "native_int4": 330,
            "bf16_fallback": 0,
        }],
        "flm_load_speed": {
            "copy_h2d_bytes": 17179869184,
            "copy_h2d_ms": 800.0,
            "copy_h2d_gib_s": 20.0,
        },
    },
    "rows": [{
        "returncode": 0,
        "resolved_model": "qwen3.6-35b-a3b",
        "generated_tokens": 1,
        "flm_weight_mode": "INT4 native FLM",
        "flm_ready_for_decode": True,
        "flm_direct_profile": {
            "required": 693,
            "raw_dense": 363,
            "native_int4": 330,
            "bf16_fallback": 0,
        },
    }],
}
```

Add one mutation test for each required rejection:

- wrong or absent `resolved_model`;
- no rows or summary count zero;
- nonzero row return code;
- `generated_tokens == 0`;
- wrong FLM weight mode;
- decode readiness absent or false;
- `native_int4 == 0`;
- `bf16_fallback != 0`;
- missing or zero transfer bytes;
- missing or zero transfer GiB/s;
- nonempty `benchmark_validation_errors`.

- [ ] **Step 6: Run report tests and confirm red**

Run:

```bash
python3 -m unittest tests.test_qwen36_flm_first_class_e2e -v
```

Expected: new report tests fail because report validation is missing.

- [ ] **Step 7: Implement complete evidence validation**

Implement `first_class_errors(payload)` without accepting summary-only
evidence. Validate each row and then cross-check aggregate counts. Accept
either pageable or storage-direct transfer fields:

```python
transfer_bytes = max(
    int(load_speed.get("copy_h2d_bytes") or 0),
    int(load_speed.get("copy_storage_to_device_bytes") or 0),
)
transfer_gib_s = max(
    float(load_speed.get("copy_h2d_gib_s") or 0.0),
    float(load_speed.get("copy_storage_to_device_gib_s") or 0.0),
)
```

`validate_benchmark_report(path)` loads JSON and raises:

```python
raise PhaseError(
    "report evidence failed: " + "; ".join(first_class_errors(payload))
)
```

when any condition fails.

- [ ] **Step 8: Write and implement the orchestrator test**

Mock `prepare_artifact`, `run_command`, and `validate_benchmark_report`. Assert
the order is artifact preparation, benchmark execution, then report
validation:

```python
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = prepare_artifact(args)
    run_command(
        supersonic_benchmark_command(args, artifact),
        cwd=ROOT,
        timeout=args.inference_timeout,
        phase="SuperSonic inference",
    )
    payload = validate_benchmark_report(args.out_json)
    print_summary(payload, artifact)
    return 0
```

Ensure parser defaults generate at least one token and reject zero/negative
limits, token counts, context sizes, and timeouts.

- [ ] **Step 9: Run all verifier tests**

Run:

```bash
python3 -m unittest tests.test_qwen36_flm_first_class_e2e -v
```

Expected: all verifier tests pass.

- [ ] **Step 10: Commit Task 2**

```bash
git add \
  tests/gfx1100/run_qwen36_flm_first_class_e2e.py \
  tests/test_qwen36_flm_first_class_e2e.py
git commit -m "test: require first-class FLM inference evidence"
```

---

### Task 3: Canonical FLM Defaults And Documentation

**Files:**
- Modify: `tests/gfx1100/bench_qwen36_he_supersonic.py`
- Modify: `tests/test_qwen36_he_supersonic_bench.py`
- Modify: `docs/testing.md`

**Interfaces:**
- Consumes: `DEFAULT_FLM` and CLI behavior from the new verifier.
- Produces: one canonical artifact path shared by the verifier, benchmark
  profile, and documented manual commands.

- [ ] **Step 1: Change the benchmark-default test first**

Replace the assertion for the missing July 4 temporary artifact with:

```python
self.assertEqual(
    bench.DEFAULT_35B_A3B_FLM_MODEL_DIR,
    Path(
        "/mnt/data/runs/geo-quant/"
        "qwen36-35b-a3b-supersonic-native-int4-current.flm"
    ),
)
```

- [ ] **Step 2: Run the targeted test and confirm red**

Run:

```bash
python3 -m unittest \
  tests.test_qwen36_he_supersonic_bench.Qwen36HeSupersonicBenchTests.test_qwen36_35b_a3b_flm_profile_points_at_current_e2e_artifact
```

Expected: failure showing the old
`/mnt/data/tmp/flm-first-class-e2e-20260704/...-aligned.flm` path.

- [ ] **Step 3: Update the benchmark constant**

Set:

```python
DEFAULT_35B_A3B_FLM_MODEL_DIR = Path(
    "/mnt/data/runs/geo-quant/"
    "qwen36-35b-a3b-supersonic-native-int4-current.flm"
)
```

- [ ] **Step 4: Run the benchmark harness tests**

Run:

```bash
python3 -m unittest \
  tests.test_qwen36_he_supersonic_bench \
  tests.test_qwen36_flm_first_class_e2e \
  tests.test_support_matrix
```

Expected: all tests pass.

- [ ] **Step 5: Replace stale manual instructions with the verifier command**

In `docs/testing.md`, retain low-level validator, model-store, runner, and
hipFile diagnostic commands, but make this the canonical producer-to-consumer
gate:

```bash
cd /home/deano/projects/SuperSonicBase
python3 tests/gfx1100/run_qwen36_flm_first_class_e2e.py \
  --hf-source /mnt/data/models/Qwen3.6-35B-A3B \
  --geoquant-root /home/deano/projects/geo-quant \
  --geoquant-python /home/deano/projects/geo-quant/.venv-rocm/bin/python \
  --flm /mnt/data/runs/geo-quant/qwen36-35b-a3b-supersonic-native-int4-current.flm \
  --binary target/release/supersonic \
  --out-json target/qwen36_35b_a3b_flm_first_class_e2e.json
```

Document `--regenerate`, strict reuse validation, full payload verification,
partial-file retention, and the fact that the SuperSonic subprocess receives
no HF path.

- [ ] **Step 6: Verify docs and commit Task 3**

Run:

```bash
git diff --check
python3 tools/check-support-matrix.py
python3 -m unittest \
  tests.test_qwen36_he_supersonic_bench \
  tests.test_qwen36_flm_first_class_e2e \
  tests.test_support_matrix
```

Expected: commands exit 0 and all tests pass.

Commit:

```bash
git add \
  tests/gfx1100/bench_qwen36_he_supersonic.py \
  tests/test_qwen36_he_supersonic_bench.py \
  docs/testing.md
git commit -m "docs: make FLM e2e verifier canonical"
```

---

### Task 4: Real ROCm Producer-To-Consumer Gate

**Files:**
- Modify only if results expose a defect: files owned by Tasks 1-3.
- Generated, untracked artifact:
  `/mnt/data/runs/geo-quant/qwen36-35b-a3b-supersonic-native-int4-current.flm`
- Generated, untracked report:
  `target/qwen36_35b_a3b_flm_first_class_e2e.json`

**Interfaces:**
- Consumes: merged geo-quant `main`, the verifier, and SuperSonic's release
  binary.
- Produces: fresh artifact, full-hash validation result, real inference JSON,
  and terminal evidence for the PR.

- [ ] **Step 1: Confirm both source trees before the long run**

Run:

```bash
git -C /home/deano/projects/geo-quant fetch --prune origin
git -C /home/deano/projects/geo-quant worktree add \
  --detach \
  /home/deano/.config/superpowers/worktrees/geo-quant/flm-first-class-e2e-producer \
  origin/main
git -C /home/deano/.config/superpowers/worktrees/geo-quant/flm-first-class-e2e-producer \
  rev-parse HEAD
git -C /home/deano/projects/geo-quant rev-parse origin/main
git -C /home/deano/.config/superpowers/worktrees/geo-quant/flm-first-class-e2e-producer \
  status --short --branch
git status --short --branch
test -d /mnt/data/models/Qwen3.6-35B-A3B
/home/deano/projects/geo-quant/.venv-rocm/bin/python -c \
  'import blake3, torch; print(torch.__version__, torch.version.hip)'
```

Expected: the detached geo-quant producer worktree is clean and its `HEAD`
equals `origin/main`, this SuperSonic worktree has only intentional changes,
the HF snapshot exists, and the Python environment reports ROCm-enabled
PyTorch. If the producer worktree already exists, verify its state and reuse or
refresh it instead of adding it again. Do not switch or clean the user's active
geo-quant checkout.

- [ ] **Step 2: Build the current SuperSonic release binary**

Run:

```bash
cargo build -p runner --bin supersonic --release
```

Expected: exit 0 and
`target/release/supersonic` has a current modification timestamp.

- [ ] **Step 3: Run a forced fresh export and end-to-end inference**

Run:

```bash
python3 tests/gfx1100/run_qwen36_flm_first_class_e2e.py \
  --regenerate \
  --hf-source /mnt/data/models/Qwen3.6-35B-A3B \
  --geoquant-root /home/deano/.config/superpowers/worktrees/geo-quant/flm-first-class-e2e-producer \
  --geoquant-python /home/deano/projects/geo-quant/.venv-rocm/bin/python \
  --flm /mnt/data/runs/geo-quant/qwen36-35b-a3b-supersonic-native-int4-current.flm \
  --binary target/release/supersonic \
  --out-json target/qwen36_35b_a3b_flm_first_class_e2e.json \
  --export-timeout 7200 \
  --validation-timeout 1800 \
  --inference-timeout 600
```

Expected: producer export succeeds, strict full-hash validation succeeds, one
token is generated, and the verifier prints an `ok` summary containing the
resolved model and transfer throughput.

- [ ] **Step 4: Run the reuse path**

Run the same command without `--regenerate`.

Expected: structural validation selects reuse, full payload hashes are checked,
no quantization export starts, and SuperSonic inference passes again.

- [ ] **Step 5: Inspect structured evidence independently**

Run:

```bash
/home/deano/projects/geo-quant/.venv-rocm/bin/python -m \
  geoquant.formats.flm_validate \
  /mnt/data/runs/geo-quant/qwen36-35b-a3b-supersonic-native-int4-current.flm \
  --profile supersonic-qwen36-moe-native-int4 \
  --verify-payload-hashes

python3 -c '
import json
from pathlib import Path
p = json.loads(Path("target/qwen36_35b_a3b_flm_first_class_e2e.json").read_text())
print(json.dumps({
    "resolved_model": p.get("resolved_model"),
    "summary": p.get("summary"),
    "rows": p.get("rows"),
}, indent=2))
'
```

Expected: FLM validation reports success; report rows show generated tokens,
`INT4 native FLM`, ready-for-decode, positive native INT4 coverage, zero BF16
fallback, and positive transfer bytes/GiB/s.

- [ ] **Step 6: Run the complete branch verification**

Run:

```bash
git diff --check
python3 tools/check-support-matrix.py
python3 -m unittest \
  tests.test_qwen36_he_supersonic_bench \
  tests.test_qwen36_flm_first_class_e2e \
  tests.test_support_matrix
cargo test -q -p runner --test flm_moe_main_path
cargo test -q -p model-store --test flm_qwen36_native_layout
```

Expected: every command exits 0 and all tests pass. Environment-gated tests may
skip only when their documented FLM environment variable is absent; the
standalone e2e run in Steps 3-5 is mandatory and may not be replaced by a skip.

- [ ] **Step 7: Commit any evidence-driven fixes**

If the real run required code or documentation fixes, repeat their failing test,
make the minimal change, rerun Steps 3-6, and commit only those intentional
files:

```bash
git add \
  tests/gfx1100/run_qwen36_flm_first_class_e2e.py \
  tests/test_qwen36_flm_first_class_e2e.py \
  tests/gfx1100/bench_qwen36_he_supersonic.py \
  tests/test_qwen36_he_supersonic_bench.py \
  docs/testing.md
git commit -m "fix: complete FLM producer consumer gate"
```

If no tracked files changed, do not create an empty commit.
