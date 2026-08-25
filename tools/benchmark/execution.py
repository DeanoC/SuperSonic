"""Budgeted execution for the repository benchmark contracts.

This module deliberately keeps orchestration in Python.  The model runners are
still the public binaries described by ``benchmarks/engines`` and every
invocation is a fresh subprocess with an argv vector.  The module's job is to
make the process boundary, timing budget, evidence, and promotion rules
boring and explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

from . import adapters, environment, gpu, manifest, quality, validation
from .model import EngineManifest, PerformanceCase, QualityCase, SuiteManifest, canonical_json, parse_strict_json


ROOT = manifest.ROOT
EXPECTED_BUDGETS = {"quick": 600, "full": 21600}
QUALITY_CASE_TIMEOUT_SECONDS = 180
QUICK_BUDGET_SECONDS = EXPECTED_BUDGETS["quick"]
FULL_BUDGET_SECONDS = EXPECTED_BUDGETS["full"]
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
SUPPORTED_GPU_ARCHES = frozenset(("gfx1100", "gfx1201"))
MTP_CATEGORY = quality.MTP_CATEGORY
VERSION_FILE_MAX_BYTES = 4096
VERSION_VALUE_MAX_LENGTH = 128
_VERSION_VALUE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._:+()-]{0,127}$")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Inputs needed to execute a suite.

    Model, artifact, and physical GPU values intentionally have no defaults.
    A caller may use the small ``replace_config`` helper in tests, but a run
    cannot accidentally fall back to a host-local model or device.
    """

    suite: str | SuiteManifest
    model_dir: Path | None
    artifact: Path | None
    physical_gpu: str | None
    gpu_arch: str | None
    gpu_static_json: Path | None = None
    rocm_version_file: Path | None = None
    hip_version_file: Path | None = None
    rocm_version: str | None = None
    hip_version: str | None = None
    logical_gpu: str | None = None
    output_dir: Path = Path("target/benchmarks/candidate")
    peer_artifact: Path | None = None
    device: int = 0
    context_size: int | None = 32768
    chat: bool = False
    clock_policy: str | Mapping[str, object] = "uncontrolled-clocks"
    environment: Mapping[str, str] | None = None
    environment_snapshot: Mapping[str, object] | environment.EnvironmentSnapshot | None = None
    engine_binaries: Mapping[str, str | Path] = field(default_factory=dict)
    engine_versions: Mapping[str, str] = field(default_factory=dict)
    binary_exists: object | None = None
    version_outputs: Mapping[str, str] = field(default_factory=dict)
    repository: Path = ROOT
    run_id: str | None = None
    seed: int | None = None
    run_quality: bool = True
    artifact_semantic_id: str | None = None
    artifact_quantization: str | None = None
    tokenizer_sha256: str | None = None
    chat_template_sha256: str | None = None
    strict_environment: bool = False
    environment_command_runner: Callable[[tuple[str, ...]], str] | None = None
    cpu_governor_reader: Callable[[], str] | None = None


@dataclass(frozen=True, slots=True)
class RunManifest:
    suite: SuiteManifest
    config: RunConfig
    engines: tuple[EngineManifest, ...]
    quality_cases: tuple[QualityCase, ...]
    gpu: gpu.StaticGpuProvenance
    run_id: str
    bundle: Path
    commit: str
    dirty: bool

    @property
    def output_dir(self) -> Path:
        return self.bundle.parent

    @property
    def model_dir(self) -> Path:
        return Path(self.config.model_dir)

    @property
    def artifact(self) -> Path:
        return Path(self.config.artifact)

    @property
    def peer_artifact(self) -> Path | None:
        return Path(self.config.peer_artifact) if self.config.peer_artifact is not None else None


@dataclass(frozen=True, slots=True)
class BundleStatus:
    state: str
    bundle: Path
    errors: tuple[str, ...] = ()
    records: tuple[Path, ...] = ()
    quality_failed: bool = False
    performance_report_only: bool = True
    elapsed_seconds: float = 0.0
    completed_rounds: int = 0


# ``BenchmarkConfig`` is a descriptive alias for callers that do not want to
# couple their code to the run-manifest terminology.
BenchmarkConfig = RunConfig
Config = RunConfig


@dataclass(frozen=True, slots=True)
class ProcessResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float = 0.0
    timed_out: bool = False
    interrupted: bool = False


def replace_config(config: RunConfig, **changes: object) -> RunConfig:
    """Small test/user convenience mirroring :func:`dataclasses.replace`.

    ``RunConfig`` is the supported public shape.  Accepting a mapping or a
    simple object here as well keeps contract tests readable without making
    the runner depend on a second configuration parser.
    """

    if isinstance(config, RunConfig):
        return replace(config, **changes)
    values = _config_values(config)
    values.update(changes)
    return _coerce_config(values)


def preflight(config: RunConfig | Mapping[str, object] | object) -> RunManifest:
    """Validate all configured inputs and create an immutable candidate area."""

    resolved = _coerce_config(config)
    if resolved.seed is not None and (
        isinstance(resolved.seed, bool) or not isinstance(resolved.seed, int) or resolved.seed < 0
    ):
        raise ValueError("seed must be a non-negative integer")
    if callable(resolved.environment_snapshot):
        resolved = replace(resolved, environment_snapshot=resolved.environment_snapshot())
    suite = resolved.suite if isinstance(resolved.suite, SuiteManifest) else manifest.load_suite(str(resolved.suite))
    expected_budget = EXPECTED_BUDGETS.get(suite.name)
    if expected_budget is not None and suite.budget_seconds != expected_budget:
        raise ValueError(f"suite {suite.name} budget must be exactly {expected_budget} seconds")
    _validate_active_cases(suite)

    model_dir = _required_path(resolved.model_dir, "model_dir", directory=True)
    _validate_model_files(model_dir, chat=bool(resolved.chat))
    artifact = _required_path(resolved.artifact, "artifact", directory=False, nonempty=True)
    _validate_model_digests(model_dir, resolved)
    physical_gpu = _required_text(resolved.physical_gpu, "physical_gpu")
    if not physical_gpu.isdigit():
        raise ValueError("physical_gpu must be a numeric GPU ordinal")
    gpu_arch = _required_text(resolved.gpu_arch, "gpu_arch")
    if gpu_arch not in SUPPORTED_GPU_ARCHES:
        raise ValueError(f"unsupported gpu_arch: {gpu_arch!r}; expected gfx1100 or gfx1201")
    if resolved.device < 0:
        raise ValueError("device must be non-negative")
    if resolved.context_size is not None and resolved.context_size <= 0:
        raise ValueError("context_size must be positive")
    logical_gpu = _required_text(resolved.logical_gpu or str(resolved.device), "logical_gpu")
    static_json = _required_path(resolved.gpu_static_json, "gpu_static_json", directory=False, nonempty=True)
    rocm_version_file = _required_path(
        resolved.rocm_version_file, "rocm_version_file", directory=False, nonempty=True
    )
    hip_version_file = _required_path(
        resolved.hip_version_file, "hip_version_file", directory=False, nonempty=True
    )
    rocm_version = _read_version_file(rocm_version_file, "ROCm")
    hip_version = _read_version_file(hip_version_file, "HIP")
    static_gpu = gpu.resolve_static_gpu(
        static_json,
        physical_gpu=physical_gpu,
        gpu_arch=gpu_arch,
        logical_gpu=logical_gpu,
    )
    clock_policy_name = _clock_policy_name(resolved.clock_policy)
    if clock_policy_name == "locked":
        _validate_locked_policy(resolved.clock_policy)
    if resolved.environment_snapshot is not None:
        snapshot_policy = _snapshot_policy_name(resolved.environment_snapshot)
        if snapshot_policy != clock_policy_name:
            raise ValueError("environment_snapshot clock policy does not match configured clock policy")
        _validate_snapshot_policy_values(resolved.environment_snapshot, resolved.clock_policy)

    engines = tuple(_engine_with_override(engine, resolved) for engine in (manifest.load_engine(name) for name in suite.engines))
    _validate_peer_artifact(suite, resolved)
    for engine in engines:
        _validate_engine_available(engine, resolved)
        _validate_engine_version(engine, resolved)

    output_dir = Path(resolved.output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output_dir is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = _run_id(resolved, suite)
    bundle = output_dir / run_id
    if bundle.exists():
        # Never reuse a candidate directory.  A previous interrupted run is
        # evidence and must remain inspectable.
        suffix = uuid.uuid4().hex[:8]
        run_id = f"{run_id}-{suffix}"
        bundle = output_dir / run_id
    (bundle / "records").mkdir(parents=True)
    (bundle / "logs").mkdir(parents=True)

    commit, dirty = _git_identity(resolved.repository)
    quality_cases = tuple(case for case in manifest.load_quality(suite.quality_version) if case.id in suite.quality_case_ids)
    if len(quality_cases) != len(suite.quality_case_ids):
        missing = sorted(set(suite.quality_case_ids) - {case.id for case in quality_cases})
        raise ValueError(f"suite references missing quality cases: {missing}")
    resolved_config = replace(
        resolved,
        model_dir=model_dir,
        artifact=artifact,
        physical_gpu=physical_gpu,
        gpu_arch=gpu_arch,
        gpu_static_json=static_json,
        rocm_version_file=rocm_version_file,
        hip_version_file=hip_version_file,
        rocm_version=rocm_version,
        hip_version=hip_version,
        logical_gpu=static_gpu.logical_gpu,
    )
    result = RunManifest(
        suite=suite,
        config=resolved_config,
        engines=engines,
        quality_cases=quality_cases,
        gpu=static_gpu,
        run_id=run_id,
        bundle=bundle,
        commit=commit,
        dirty=dirty,
    )
    _persist_bundle_manifest(result, ordered_cases(result))
    return result


def ordered_cases(
    run_manifest: RunManifest,
    *,
    seed: int | None = None,
) -> tuple[tuple[PerformanceCase, EngineManifest], ...]:
    """Return the deterministic case/engine matrix, respecting case scope."""

    engines_by_name = {engine.name: engine for engine in run_manifest.engines}
    entries: list[tuple[PerformanceCase, EngineManifest]] = []
    for case in run_manifest.suite.performance_cases:
        if case.cache_state.startswith("prefix-cache-"):
            raise ValueError(f"prefix cache case {case.id!r} is not executable before adapter verification")
        for engine_name in case.engines:
            engine = engines_by_name.get(engine_name)
            if engine is None:
                raise ValueError(f"case {case.id!r} references unavailable engine {engine_name!r}")
            # Recheck at the execution boundary.  This prevents a malformed
            # hand-built manifest from bypassing the parser's scope checks.
            if engine.name not in case.engines or case.mode not in engine.supported_modes:
                raise ValueError(f"case {case.id!r} is outside engine scope for {engine.name!r}")
            entries.append((case, engine))
    selected_seed = run_manifest.config.seed if seed is None else seed
    if selected_seed is not None:
        random.Random(int(selected_seed)).shuffle(entries)
    return tuple(entries)


def run_suite(
    config: RunConfig | Mapping[str, object] | object,
    clock: Callable[[], float] = time.monotonic,
    command_runner: Callable[..., object] = None,
) -> BundleStatus:
    """Execute a suite until its monotonic deadline.

    Completed records are promoted one at a time.  If a process, parser, or
    signal fails, its raw streams remain under ``logs/`` while no invalid
    record is promoted.  The completed subset is therefore useful evidence,
    but its status cannot be mistaken for a complete suite.
    """

    runner = command_runner or run_process
    if not callable(runner) and callable(getattr(runner, "run", None)):
        runner = runner.run
    run_manifest = preflight(config)
    started = clock()
    deadline = started + run_manifest.suite.budget_seconds
    errors: list[str] = []
    records: list[Path] = []
    interrupted = False
    quality_failed = False

    quality_summaries: dict[str, dict[str, object]]
    if run_manifest.config.run_quality:
        quality_summaries, quality_errors, quality_interrupted = _run_quality(
            run_manifest,
            deadline=deadline,
            clock=clock,
            command_runner=runner,
        )
        errors.extend(quality_errors)
        quality_failed = any(
            bool(summary.get("failed", 0)) or bool(summary.get("missing_case_ids"))
            for summary in quality_summaries.values()
        )
        interrupted = quality_interrupted
    else:
        # A record must remain schema/consistency-valid even when a caller
        # explicitly disables quality execution.  Represent the skipped gate
        # as concrete failed cases (rather than ``missing_case_ids``), which
        # preserves the blocking signal for ``validate --publishable`` while
        # allowing partial performance evidence to be inspected.
        placeholder = _quality_placeholder_summary(run_manifest.quality_cases)
        quality_summaries = {engine.name: placeholder for engine in run_manifest.engines}
        quality_failed = True
        _append_error(errors, "quality_failed")

    scheduled = ordered_cases(run_manifest)
    expected_count = len(scheduled)
    completed_rounds = 0
    if run_manifest.suite.minimum_duration_seconds > 0 and not interrupted:
        duration_records, duration_errors, duration_interrupted, completed_rounds = _run_duration_cases(
            run_manifest,
            scheduled,
            deadline=deadline,
            minimum_deadline=started + run_manifest.suite.minimum_duration_seconds,
            clock=clock,
            command_runner=runner,
            quality_summaries=quality_summaries,
        )
        records.extend(duration_records)
        errors.extend(error for error in duration_errors if error not in errors)
        interrupted = duration_interrupted
    else:
        for case, engine in scheduled:
            if interrupted:
                break
            now = clock()
            if now >= deadline:
                _append_error(errors, "budget_exhausted")
                break
            case_deadline = min(deadline, now + float(case.timeout_seconds))
            # _execute_case recomputes the positive remainder immediately before
            # every warmup/repetition subprocess.  Passing only the case cap here
            # avoids spending a stale timeout between scheduling and invocation.
            timeout = float(case.timeout_seconds)
            try:
                result = _execute_case(
                    run_manifest,
                    case,
                    engine,
                    timeout=timeout,
                    suite_deadline=deadline,
                    case_deadline=case_deadline,
                    clock=clock,
                    command_runner=runner,
                    quality_summary=quality_summaries.get(
                        engine.name, _quality_placeholder_summary(run_manifest.quality_cases)
                    ),
                )
            except KeyboardInterrupt:
                _append_error(errors, "interrupted")
                interrupted = True
                break
            except _CaseError as exc:
                _append_error(errors, exc.code)
                if exc.code == "budget_exhausted":
                    break
                continue
            except ValueError as exc:
                # Validation is part of promotion.  A bad candidate is retained
                # only in its raw logs and represented as a structured execution
                # error; it must never escape as a partially promoted record.
                _append_error(errors, "invalid_record")
                _write_case_error(run_manifest.bundle, case, engine, str(exc))
                continue
            except (TypeError, RuntimeError) as exc:
                _append_error(errors, "process_failed")
                _write_case_error(run_manifest.bundle, case, engine, str(exc))
                continue
            if isinstance(result, Path):
                records.append(result)
            # Do not schedule another case once the active case has exhausted the
            # suite budget.  The check is deliberately monotonic and not wall time.
            if clock() >= deadline:
                _append_error(errors, "budget_exhausted")
                break

    elapsed_seconds = max(0.0, float(clock()) - float(started))
    duration_complete = (
        run_manifest.suite.minimum_duration_seconds == 0
        or elapsed_seconds >= run_manifest.suite.minimum_duration_seconds
    )
    completed = len(records) == expected_count and duration_complete and not interrupted and not any(
        error in {"budget_exhausted", "case_timeout", "process_failed", "invalid_output", "interrupted"}
        for error in errors
    )
    if interrupted or "budget_exhausted" in errors or "case_timeout" in errors:
        state = "incomplete"
    elif errors:
        state = "failed"
    elif quality_failed:
        # Quality is a blocking gate; performance remains report-only.
        state = "failed"
    elif not completed:
        state = "incomplete"
    else:
        state = "complete"
    status = BundleStatus(
        state=state,
        bundle=run_manifest.bundle,
        errors=tuple(errors),
        records=tuple(records),
        quality_failed=quality_failed,
        performance_report_only=True,
        elapsed_seconds=elapsed_seconds,
        completed_rounds=completed_rounds,
    )
    _update_bundle_manifest(run_manifest, status)
    return status


def _run_duration_cases(
    run_manifest: RunManifest,
    scheduled: Sequence[tuple[PerformanceCase, EngineManifest]],
    *,
    deadline: float,
    minimum_deadline: float,
    clock: Callable[[], float],
    command_runner: Callable[..., object],
    quality_summaries: Mapping[str, Mapping[str, object]],
) -> tuple[list[Path], list[str], bool, int]:
    """Run one measured invocation per entry in complete seeded rounds."""

    accumulated: dict[tuple[str, str], dict[str, object]] = {}
    promoted: list[Path] = []
    errors: list[str] = []
    interrupted = False
    completed_rounds = 0
    required_rounds = max(case.repetitions for case, _ in scheduled)

    while completed_rounds < required_rounds or clock() < minimum_deadline:
        round_records: list[tuple[PerformanceCase, EngineManifest, dict[str, object]]] = []
        round_failed = False
        for case, engine in scheduled:
            if clock() >= deadline:
                _append_error(errors, "budget_exhausted")
                round_failed = True
                break
            round_case = replace(
                case,
                warmups=case.warmups if completed_rounds == 0 else 0,
                repetitions=1,
            )
            try:
                record = _execute_case(
                    run_manifest,
                    round_case,
                    engine,
                    timeout=float(case.timeout_seconds),
                    suite_deadline=deadline,
                    case_deadline=deadline,
                    clock=clock,
                    command_runner=command_runner,
                    quality_summary=quality_summaries.get(
                        engine.name, _quality_placeholder_summary(run_manifest.quality_cases)
                    ),
                    promote=False,
                    run_label_offset=completed_rounds,
                )
                if not isinstance(record, dict):
                    raise RuntimeError("duration case did not return an in-memory record")
                round_records.append((case, engine, record))
            except KeyboardInterrupt:
                _append_error(errors, "interrupted")
                interrupted = True
                round_failed = True
                break
            except _CaseError as exc:
                _append_error(errors, exc.code)
                round_failed = True
                break
            except ValueError as exc:
                _append_error(errors, "invalid_record")
                _write_case_error(run_manifest.bundle, case, engine, str(exc))
                round_failed = True
                break
            except (TypeError, RuntimeError) as exc:
                _append_error(errors, "process_failed")
                _write_case_error(run_manifest.bundle, case, engine, str(exc))
                round_failed = True
                break
        if round_failed or len(round_records) != len(scheduled):
            break

        for case, engine, record in round_records:
            key = (case.id, engine.name)
            accumulated[key] = _merge_duration_record(accumulated.get(key), record, case)
        completed_rounds += 1

    if completed_rounds >= required_rounds:
        for case, engine in scheduled:
            target = run_manifest.bundle / "records" / _record_filename(case, engine)
            _atomic_promote_record(accumulated[(case.id, engine.name)], target)
            promoted.append(target)

    return promoted, errors, interrupted, completed_rounds


def _merge_duration_record(
    existing: Mapping[str, object] | None,
    current: Mapping[str, object],
    case: PerformanceCase,
) -> dict[str, object]:
    merged = json.loads(canonical_json(dict(current)))
    merged["workload"]["warmups"] = case.warmups
    if existing is None:
        return merged

    previous = json.loads(canonical_json(dict(existing)))
    previous["samples"].extend(merged["samples"])
    previous_env = previous["environment"]
    current_env = merged["environment"]
    previous_samples = previous_env["telemetry_samples"]
    current_samples = current_env["telemetry_samples"]
    offset = float(previous_samples[-1]["offset_seconds"]) if previous_samples else 0.0
    for sample in current_samples:
        sample["offset_seconds"] = offset + float(sample["offset_seconds"])
        previous_samples.append(sample)
    previous_env["observed_after"] = current_env["observed_after"]
    previous_env["observed_after_at"] = current_env["observed_after_at"]
    for note in current_env["evidence_notes"]:
        if note not in previous_env["evidence_notes"]:
            previous_env["evidence_notes"].append(note)
    previous_env["verification_errors"] = []
    derived_errors = validation.headline_verification_errors(previous)
    previous_env["verification_errors"] = list(derived_errors)
    previous_env["headline_eligible"] = previous_env["clock_policy"] == "locked" and not derived_errors
    return previous


def run_process(
    argv: Sequence[str],
    *,
    timeout: float,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> ProcessResult:
    """Run one public CLI with an argv vector and bounded timeout."""

    vector = tuple(str(item) for item in argv)
    if not math.isfinite(float(timeout)) or float(timeout) <= 0.0:
        raise ValueError("process timeout must be positive")
    started = time.monotonic()
    try:
        completed = subprocess.run(
            vector,
            shell=False,
            cwd=str(cwd) if cwd is not None else None,
            env=dict(env) if env is not None else None,
            timeout=float(timeout),
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ProcessResult(
            argv=vector,
            returncode=124,
            stdout=_text(exc.stdout),
            stderr=_text(exc.stderr),
            duration_seconds=time.monotonic() - started,
            timed_out=True,
        )
    return ProcessResult(
        argv=vector,
        returncode=int(completed.returncode),
        stdout=_text(completed.stdout),
        stderr=_text(completed.stderr),
        duration_seconds=time.monotonic() - started,
    )


def _run_process_with_telemetry(
    argv: Sequence[str],
    *,
    timeout: float,
    physical_gpu: str,
    probe_runner: Callable[[tuple[str, ...]], str] | None = None,
    sample_interval_seconds: float = 0.25,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[ProcessResult, tuple[environment.TelemetrySample, ...], tuple[str, ...]]:
    vector = tuple(str(item) for item in argv)
    if not math.isfinite(float(timeout)) or float(timeout) <= 0.0:
        raise ValueError("process timeout must be positive")
    if not math.isfinite(float(sample_interval_seconds)) or float(sample_interval_seconds) <= 0.0:
        raise ValueError("telemetry sample interval must be positive")
    active_probe_runner = probe_runner or _default_environment_probe_runner
    command = environment.build_probe_command(str(physical_gpu))
    started = time.monotonic()
    deadline = started + float(timeout)
    samples: list[environment.TelemetrySample] = []
    notes: list[str] = []
    process = subprocess.Popen(
        vector,
        shell=False,
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                _kill_process_session(process)
                stdout, stderr = process.communicate()
                return (
                    ProcessResult(
                        argv=vector,
                        returncode=124,
                        stdout=_text(stdout),
                        stderr=_text(stderr),
                        duration_seconds=time.monotonic() - started,
                        timed_out=True,
                    ),
                    tuple(samples),
                    tuple(notes),
                )
            try:
                stdout, stderr = process.communicate(timeout=min(float(sample_interval_seconds), remaining))
                return (
                    ProcessResult(
                        argv=vector,
                        returncode=int(process.returncode),
                        stdout=_text(stdout),
                        stderr=_text(stderr),
                        duration_seconds=time.monotonic() - started,
                    ),
                    tuple(samples),
                    tuple(notes),
                )
            except subprocess.TimeoutExpired:
                try:
                    observed = environment.parse_showallinfo(active_probe_runner(command), notes)
                    samples.append(
                        environment.TelemetrySample(
                            offset_seconds=time.monotonic() - started,
                            gpu_clock_mhz=observed.gpu_clock_mhz,
                            memory_clock_mhz=observed.memory_clock_mhz,
                            power_cap_watts=observed.power_cap_watts,
                            power_watts=observed.power_watts,
                            temperature_celsius=observed.temperature_celsius,
                            gpu_utilization_percent=observed.gpu_utilization_percent,
                            memory_utilization_percent=observed.memory_utilization_percent,
                            performance_level=observed.performance_level,
                        )
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    notes.append(f"live telemetry probe failed: {exc}")
    except BaseException:
        _kill_process_session(process)
        process.communicate()
        raise


def _kill_process_session(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class _CaseError(RuntimeError):
    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        self.message = message or code
        super().__init__(self.message)


def _remaining_timeout(
    clock: Callable[[], float],
    *,
    suite_deadline: float,
    case_deadline: float,
    cap: float,
) -> float:
    now = float(clock())
    remaining_suite = float(suite_deadline) - now
    remaining_case = float(case_deadline) - now
    remaining = min(remaining_suite, remaining_case, float(cap))
    if not math.isfinite(remaining) or remaining <= 0.0:
        if remaining_suite <= 0.0:
            raise _CaseError("budget_exhausted", "suite deadline reached")
        raise _CaseError("case_timeout", "case deadline reached")
    return remaining


def _execute_case(
    run_manifest: RunManifest,
    case: PerformanceCase,
    engine: EngineManifest,
    *,
    timeout: float,
    suite_deadline: float | None = None,
    case_deadline: float | None = None,
    clock: Callable[[], float],
    command_runner: Callable[..., object],
    quality_summary: Mapping[str, object],
    promote: bool = True,
    run_label_offset: int = 0,
) -> Path | dict[str, object] | None:
    inputs = adapters.AdapterInputs(
        model_dir=Path(run_manifest.config.model_dir),
        artifact=Path(run_manifest.config.artifact),
        peer_artifact=Path(run_manifest.config.peer_artifact) if run_manifest.config.peer_artifact else None,
        chat=bool(run_manifest.config.chat),
        device=int(run_manifest.config.device),
        context_size=run_manifest.config.context_size,
        sampling_seed=int(
            adapters.DEFAULT_SAMPLING_SEED if run_manifest.config.seed is None else run_manifest.config.seed
        ),
    )
    try:
        argv = adapters.build_command(engine, case, inputs)
    except ValueError as exc:
        raise _CaseError("invalid_configuration", str(exc)) from exc

    parsed_samples: list[adapters.ParsedOutput] = []
    live_telemetry = (
        _clock_policy_name(run_manifest.config.clock_policy) == "locked"
        and run_manifest.config.environment_snapshot is None
        and command_runner is run_process
    )
    telemetry_before: environment.ObservedTelemetry | None = None
    telemetry_before_at: str | None = None
    telemetry_samples: list[environment.TelemetrySample] = []
    telemetry_notes: list[str] = []
    telemetry_elapsed = 0.0
    probe_runner = run_manifest.config.environment_command_runner or _default_environment_probe_runner
    if live_telemetry:
        telemetry_before_at = _utc_timestamp()
        telemetry_before = _probe_environment(run_manifest.config, probe_runner, telemetry_notes)
    invocation_count = case.warmups + case.repetitions
    for index in range(invocation_count):
        measured = index >= case.warmups
        label = (
            f"run-{run_label_offset + index - case.warmups + 1}"
            if measured
            else f"warmup-{index + 1}"
        )
        active_timeout = timeout
        if suite_deadline is not None and case_deadline is not None:
            active_timeout = _remaining_timeout(
                clock,
                suite_deadline=suite_deadline,
                case_deadline=case_deadline,
                cap=float(case.timeout_seconds),
            )
        if live_telemetry:
            result, invocation_telemetry, invocation_notes = _run_process_with_telemetry(
                argv,
                timeout=active_timeout,
                physical_gpu=str(run_manifest.config.physical_gpu),
                probe_runner=probe_runner,
                env=_process_environment(run_manifest.config),
            )
            telemetry_samples.extend(
                replace(sample, offset_seconds=telemetry_elapsed + sample.offset_seconds)
                for sample in invocation_telemetry
            )
            telemetry_elapsed += result.duration_seconds
            telemetry_notes.extend(invocation_notes)
        else:
            result = _invoke_process(
                command_runner,
                argv,
                timeout=active_timeout,
                case_id=case.id,
                engine_name=engine.name,
            )
        _write_streams(run_manifest.bundle, case, engine, label, result.stdout, result.stderr)
        if result.interrupted:
            raise KeyboardInterrupt
        if result.timed_out:
            raise _CaseError("case_timeout", f"{case.id}/{engine.name} exceeded {active_timeout}s timeout")
        if result.returncode != 0:
            raise _CaseError("process_failed", f"{case.id}/{engine.name} exited with {result.returncode}")
        try:
            parsed = adapters.parse_output(engine.name, result.stdout, result.stderr)
        except ValueError as exc:
            raise _CaseError("invalid_output", str(exc)) from exc
        if measured:
            parsed_samples.append(parsed)
    if len(parsed_samples) != case.repetitions:
        raise _CaseError("invalid_output", f"{case.id} did not produce all measured samples")

    case_snapshot: environment.EnvironmentSnapshot | Mapping[str, object] | None = None
    if live_telemetry:
        telemetry_after = _probe_environment(run_manifest.config, probe_runner, telemetry_notes)
        case_snapshot = environment.snapshot_from_observations(
            physical_gpu=str(run_manifest.config.physical_gpu),
            logical_gpu=str(run_manifest.config.logical_gpu or run_manifest.config.device),
            clock_policy=run_manifest.config.clock_policy,
            cache_state=case.cache_state,
            observed_before=telemetry_before,
            observed_before_at=str(telemetry_before_at),
            telemetry_samples=tuple(telemetry_samples),
            observed_after=telemetry_after,
            observed_after_at=_utc_timestamp(),
            environment_map=run_manifest.config.environment or os.environ,
            cache_evidence={"process_state": "fresh-process", "process_reuse": False},
            cpu_governor_reader=run_manifest.config.cpu_governor_reader,
            evidence_notes=tuple(telemetry_notes),
        )
    record = _build_record(
        run_manifest,
        case,
        engine,
        argv=argv,
        samples=parsed_samples,
        quality_summary=quality_summary,
        environment_snapshot=case_snapshot,
    )
    if not promote:
        return record
    filename = _record_filename(case, engine)
    target = run_manifest.bundle / "records" / filename
    _atomic_promote_record(record, target)
    return target


def _run_quality(
    run_manifest: RunManifest,
    *,
    deadline: float,
    clock: Callable[[], float],
    command_runner: Callable[..., object],
) -> tuple[dict[str, dict[str, object]], list[str], bool]:
    """Run deterministic quality once on SuperSonic.

    Peer adapters can have deliberately different raw output capabilities (in
    particular llama.cpp has no token IDs).  Token equality is therefore
    scored only for manifest exact-token MTP cases on the SuperSonic pair;
    peer records reuse the suite-level quality summary rather than invoking
    ``score_mtp_pair`` with an unsuitable output.
    """

    errors: list[str] = []
    interrupted = False
    primary = next((engine for engine in run_manifest.engines if engine.name == "supersonic"), None)
    if primary is None:
        _append_error(errors, "quality_failed")
        summary = quality.summarize_quality((), required_cases=run_manifest.quality_cases)
        return {engine.name: summary for engine in run_manifest.engines}, errors, False
    inputs = adapters.AdapterInputs(
        model_dir=Path(run_manifest.config.model_dir),
        artifact=Path(run_manifest.config.artifact),
        peer_artifact=Path(run_manifest.config.peer_artifact) if run_manifest.config.peer_artifact else None,
        chat=bool(run_manifest.config.chat),
        device=int(run_manifest.config.device),
        context_size=run_manifest.config.context_size,
        fixed_token_count=False,
        sampling_seed=int(
            adapters.DEFAULT_SAMPLING_SEED if run_manifest.config.seed is None else run_manifest.config.seed
        ),
    )
    results: list[quality.QualityResult] = []
    for case in run_manifest.quality_cases:
        try:
            _validate_quality_case(case)
            if clock() >= deadline:
                _append_error(errors, "budget_exhausted")
                break
            if case.category == MTP_CATEGORY:
                ordinary = _quality_output(
                    run_manifest,
                    case,
                    mode="ordinary",
                    inputs=inputs,
                    timeout=_quality_timeout(case, deadline, clock),
                    suite_deadline=deadline,
                    clock=clock,
                    command_runner=command_runner,
                    suffix="ordinary",
                )
                mtp = _quality_output(
                    run_manifest,
                    case,
                    mode="mtp",
                    inputs=inputs,
                    timeout=_quality_timeout(case, deadline, clock),
                    suite_deadline=deadline,
                    clock=clock,
                    command_runner=command_runner,
                    suffix="mtp",
                )
                results.append(quality.score_mtp_pair(ordinary, mtp, case=case))
            else:
                quality_case = _quality_performance_case(case, mode="ordinary", engine=primary)
                argv = adapters.build_command(primary, quality_case, inputs)
                output = _quality_process(
                    run_manifest,
                    case,
                    primary,
                    argv,
                    timeout=_quality_timeout(case, deadline, clock),
                    suite_deadline=deadline,
                    clock=clock,
                    command_runner=command_runner,
                    suffix="ordinary",
                )
                results.append(quality.score_case(case, output))
        except KeyboardInterrupt:
            _append_error(errors, "interrupted")
            interrupted = True
            break
        except _CaseError as exc:
            _append_error(errors, exc.code)
            break
        except (OSError, TypeError, RuntimeError, ValueError) as exc:
            _append_error(errors, "quality_failed")
            # Keep the diagnostic local to the bundle without putting paths or
            # unbounded process output into the portable result record.
            _write_quality_error(run_manifest.bundle, case, str(exc))
            break
    primary_summary = quality.summarize_quality(results, required_cases=run_manifest.quality_cases)
    if primary_summary.get("failed", 0) or primary_summary.get("missing_case_ids"):
        _append_error(errors, "quality_failed")
    summaries: dict[str, dict[str, object]] = {primary.name: primary_summary}

    # Run only scorer-compatible cases for the llama peer.  The two exact
    # token cases have no meaning for llama.cpp's raw format (its parser
    # intentionally returns token_ids=None), so their manifest SuperSonic MTP
    # results are carried as suite-level evidence instead of being rescored.
    primary_results = {result.id: result for result in results}
    for peer in run_manifest.engines:
        if peer.name == primary.name:
            continue
        peer_results: list[quality.QualityResult] = []
        for case in run_manifest.quality_cases:
            if case.category == MTP_CATEGORY:
                result = primary_results.get(case.id)
                if result is not None:
                    peer_results.append(result)
                continue
            try:
                if clock() >= deadline:
                    _append_error(errors, "budget_exhausted")
                    break
                quality_case = _quality_performance_case(case, mode="ordinary", engine=peer)
                argv = adapters.build_command(peer, quality_case, inputs)
                output = _quality_process(
                    run_manifest,
                    case,
                    peer,
                    argv,
                    timeout=_quality_timeout(case, deadline, clock),
                    suite_deadline=deadline,
                    clock=clock,
                    command_runner=command_runner,
                    suffix=f"{peer.name}-ordinary",
                )
                # This branch is deliberately restricted to exact text and
                # structured JSON by the validation above.  A future manifest
                # adding exact_tokens here must fail closed at integration.
                if case.scorer == "exact_tokens":
                    raise ValueError("peer quality cannot score exact_tokens without token ids")
                peer_results.append(quality.score_case(case, output))
            except KeyboardInterrupt:
                _append_error(errors, "interrupted")
                interrupted = True
                break
            except _CaseError as exc:
                _append_error(errors, exc.code)
                break
            except (OSError, TypeError, RuntimeError, ValueError) as exc:
                _append_error(errors, "quality_failed")
                _write_quality_error(run_manifest.bundle, case, str(exc))
                break
        peer_summary = quality.summarize_quality(peer_results, required_cases=run_manifest.quality_cases)
        if peer_summary.get("failed", 0) or peer_summary.get("missing_case_ids"):
            _append_error(errors, "quality_failed")
        summaries[peer.name] = peer_summary
    return summaries, errors, interrupted


def _quality_output(
    run_manifest: RunManifest,
    quality_case: QualityCase,
    *,
    mode: str,
    inputs: adapters.AdapterInputs,
    timeout: float,
    suite_deadline: float | None = None,
    clock: Callable[[], float] | None = None,
    command_runner: Callable[..., object],
    suffix: str,
) -> adapters.ParsedOutput:
    primary = next(engine for engine in run_manifest.engines if engine.name == "supersonic")
    performance_case = _quality_performance_case(quality_case, mode=mode, engine=primary)
    argv = adapters.build_command(primary, performance_case, inputs)
    return _quality_process(
        run_manifest,
        quality_case,
        primary,
        argv,
        timeout=timeout,
        suite_deadline=suite_deadline,
        clock=clock,
        command_runner=command_runner,
        suffix=suffix,
    )


def _quality_process(
    run_manifest: RunManifest,
    quality_case: QualityCase,
    engine: EngineManifest,
    argv: tuple[str, ...],
    *,
    timeout: float,
    suite_deadline: float | None = None,
    clock: Callable[[], float] | None = None,
    command_runner: Callable[..., object],
    suffix: str,
) -> adapters.ParsedOutput:
    active_timeout = float(timeout)
    if suite_deadline is not None and clock is not None:
        active_timeout = _remaining_timeout(
            clock,
            suite_deadline=suite_deadline,
            case_deadline=suite_deadline,
            cap=active_timeout,
        )
    result = _invoke_process(
        command_runner,
        argv,
        timeout=active_timeout,
        case_id=f"quality-{quality_case.id}-{suffix}",
        engine_name=engine.name,
    )
    _write_quality_streams(run_manifest.bundle, quality_case, suffix, result.stdout, result.stderr)
    if result.interrupted:
        raise KeyboardInterrupt
    if result.timed_out:
        raise _CaseError("case_timeout", f"quality case {quality_case.id} timed out")
    if result.returncode != 0:
        raise _CaseError("process_failed", f"quality case {quality_case.id} exited with {result.returncode}")
    try:
        return adapters.parse_output(engine.name, result.stdout, result.stderr)
    except ValueError as exc:
        raise _CaseError("invalid_output", str(exc)) from exc


def _quality_timeout(case: QualityCase, deadline: float, clock: Callable[[], float]) -> float:
    return _remaining_timeout(
        clock,
        suite_deadline=deadline,
        case_deadline=deadline,
        cap=float(QUALITY_CASE_TIMEOUT_SECONDS),
    )


def _quality_performance_case(case: QualityCase, *, mode: str, engine: EngineManifest) -> PerformanceCase:
    return PerformanceCase(
        id=f"quality-{case.id}-{mode}",
        prompt=case.prompt,
        max_new_tokens=case.max_new_tokens,
        warmups=0,
        repetitions=1,
        mode=mode,
        cache_state="cold-load",
        timeout_seconds=max(1, case.max_new_tokens),
        decoding_policy=case.decoding_policy,
        engines=(engine.name,),
    )


def _validate_quality_case(case: QualityCase) -> None:
    if case.category == MTP_CATEGORY:
        if case.scorer != "exact_tokens":
            raise ValueError("MTP quality cases must use exact_tokens scorer")
        if not isinstance(case.expected, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in case.expected):
            raise ValueError("MTP exact_tokens quality case must have manifest integer token expectations")
    elif case.scorer == "exact_tokens":
        raise ValueError("score_mtp_pair is reserved for manifest MTP exact-token cases")


def _validate_mtp_quality_case(case: QualityCase) -> None:
    """Compatibility/test-facing validation for the MTP integration boundary."""

    if case.category != MTP_CATEGORY or case.scorer != "exact_tokens":
        raise ValueError("MTP score requires a manifest ordinary-vs-mtp-token-equality exact_tokens case")
    _validate_quality_case(case)


def _quality_placeholder_summary(cases: Sequence[QualityCase]) -> dict[str, object]:
    placeholder = adapters.ParsedOutput(
        engine_name="supersonic",
        engine_version=None,
        generated_text="",
        token_ids=None,
        prompt_tokens=1,
        generated_tokens=1,
        decode_ms=1.0,
        ms_per_tok=1.0,
        tokens_per_second=1000.0,
    )
    results = tuple(quality.score_case(case, placeholder) for case in cases)
    return quality.summarize_quality(results, required_cases=tuple(cases))


def _build_record(
    run_manifest: RunManifest,
    case: PerformanceCase,
    engine: EngineManifest,
    *,
    argv: Sequence[str],
    samples: Sequence[adapters.ParsedOutput],
    quality_summary: Mapping[str, object],
    environment_snapshot: environment.EnvironmentSnapshot | Mapping[str, object] | None = None,
) -> dict[str, object]:
    config = run_manifest.config
    snapshot = _environment_record(config, case, environment_snapshot=environment_snapshot)
    artifact_info = _artifact_identity(config, engine)
    return {
        "run": {
            "schema_version": 1,
            "suite": run_manifest.suite.name,
            "suite_version": run_manifest.suite.version,
            "quality_version": run_manifest.suite.quality_version,
            "case_id": case.id,
            "run_id": run_manifest.run_id,
            "commit": run_manifest.commit,
            "dirty": run_manifest.dirty,
            "command": list(_safe_argv(argv)),
        },
        "engine": {
            "name": engine.name,
            "version": _engine_version(run_manifest, engine),
            "adapter_version": adapters.ADAPTER_VERSION,
        },
        "hardware": {
            "identity": run_manifest.gpu.identity,
            "identity_kind": run_manifest.gpu.identity_kind,
            "identity_source_sha256": run_manifest.gpu.source_sha256,
            "identity_fields": dict(run_manifest.gpu.selected_fields),
            "architecture": run_manifest.gpu.architecture,
            "physical_gpu": run_manifest.gpu.physical_gpu,
            "logical_gpu": run_manifest.gpu.logical_gpu,
            "clock_policy": snapshot["clock_policy"],
        },
        "artifact": artifact_info,
        "workload": {
            "case_id": case.id,
            "prompt_sha256": hashlib.sha256(case.prompt.encode("utf-8")).hexdigest(),
            "context_limit": int(config.context_size or 32768),
            "max_new_tokens": case.max_new_tokens,
            "mode": case.mode,
            "stop_policy": "ignore-eos",
            "cache_state": case.cache_state,
            "warmups": case.warmups,
            "measurement_boundary": "decode",
        },
        "environment": snapshot,
        "samples": [
            {
                "decode_ms": float(sample.decode_ms),
                "tokens_per_second": float(sample.tokens_per_second),
            }
            for sample in samples
        ],
        "quality": json.loads(canonical_json(dict(quality_summary))),
        "status": {"state": "complete"},
        "errors": [],
    }


def _environment_record(
    config: RunConfig,
    case: PerformanceCase,
    *,
    environment_snapshot: environment.EnvironmentSnapshot | Mapping[str, object] | None = None,
) -> dict[str, object]:
    rocm_version, hip_version = _configured_versions(config)
    provided = environment_snapshot if environment_snapshot is not None else config.environment_snapshot
    if callable(provided):
        provided = provided()
    if provided is None:
        policy = _clock_policy_name(config.clock_policy)
        requested = _policy_mapping(config.clock_policy)
        observed = {
            "gpu_clock_mhz": None,
            "memory_clock_mhz": None,
            "power_cap_watts": None,
            "power_watts": None,
            "temperature_celsius": None,
            "gpu_utilization_percent": None,
            "memory_utilization_percent": None,
            "performance_level": None,
        }
        sample = {
            "offset_seconds": 0.0,
            **observed,
        }
        evidence = (
            {"process_state": "fresh-process", "process_reuse": False}
            if case.cache_state in {"cold-load", "warm-resident"}
            else {"prefix_cache": case.cache_state.removeprefix("prefix-cache-"), "process_reuse": False}
        )
        return {
            "rocm_version": rocm_version,
            "hip_version": hip_version,
            "clock_policy": policy,
            "requested": {
                "gpu_clock_mhz": _optional_policy_int(requested.get("gpu_clock_mhz")),
                "clock_tolerance_mhz": _optional_policy_int(requested.get("clock_tolerance_mhz")),
                "memory_clock_mhz": _optional_policy_int(requested.get("memory_clock_mhz")),
                "power_cap_watts": _optional_policy_int(requested.get("power_cap_watts")),
                "performance_level": _optional_policy_text(requested.get("performance_level")),
            },
            "requested_at": _utc_timestamp(),
            "observed_before": dict(observed),
            "observed_before_at": _utc_timestamp(),
            "observed_after": dict(observed),
            "observed_after_at": _utc_timestamp(),
            "telemetry_samples": [sample],
            "headline_eligible": policy == "locked",
            "physical_gpu": str(config.physical_gpu),
            "logical_gpu": str(config.logical_gpu or config.device),
            "cpu_governor": None,
            "allowlisted_environment": _safe_environment(config.environment or os.environ),
            "cache_state": case.cache_state,
            "cache_evidence": evidence,
            "process_reuse": False,
            "verification_errors": [] if policy == "locked" else ["headline eligibility requires locked clock_policy"],
            "evidence_notes": ["telemetry probe not configured; observed values are unavailable"],
        }
    value = asdict(provided) if hasattr(provided, "__dataclass_fields__") else json.loads(canonical_json(dict(provided)))
    if not isinstance(value, dict):
        raise ValueError("environment_snapshot must be an object")
    value["cache_state"] = case.cache_state
    value["rocm_version"] = rocm_version
    value["hip_version"] = hip_version
    value["cache_evidence"] = _cache_evidence(case.cache_state, value.get("cache_evidence"))
    # Environment snapshots supplied by callers are evidence, not a license to
    # claim process reuse between one-shot invocations.  Reject a positive
    # claim instead of silently rewriting it.
    if value.get("process_reuse", False) is True:
        raise ValueError("process_reuse must remain false for one-shot evidence")
    value["process_reuse"] = False
    value["headline_eligible"] = bool(value.get("headline_eligible", False))
    value["physical_gpu"] = str(config.physical_gpu)
    value["logical_gpu"] = str(config.logical_gpu or config.device)
    if value.get("clock_policy") == "uncontrolled-clocks" and not value.get("verification_errors"):
        value["verification_errors"] = ["headline eligibility requires locked clock_policy"]
    if value.get("clock_policy") == "uncontrolled-clocks":
        value["headline_eligible"] = False
    value["allowlisted_environment"] = _safe_environment(value.get("allowlisted_environment", {}))
    return value


def _cache_evidence(cache_state: str, evidence: object) -> dict[str, object]:
    if isinstance(evidence, Mapping):
        result = dict(evidence)
    elif cache_state in {"cold-load", "warm-resident"}:
        result = {"process_state": "fresh-process"}
    else:
        result = {"prefix_cache": cache_state.removeprefix("prefix-cache-")}
    result.setdefault("process_reuse", False)
    environment.validate_cache_evidence(cache_state, result)
    return result


def _atomic_promote_record(record: Mapping[str, object], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_json(dict(record)))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Validate the serialized value, not merely the pre-serialization
        # Python object.  This catches any unsafe JSON conversion before the
        # atomic rename.
        serialized = json.loads(temporary.read_text(encoding="utf-8"))
        validation.validate_record(serialized)
        temporary.replace(target)
        _fsync_directory(target.parent)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _atomic_json_write(payload: Mapping[str, object], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(canonical_json(dict(payload)))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
        _fsync_directory(target.parent)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _persist_bundle_manifest(
    run_manifest: RunManifest,
    entries: Sequence[tuple[PerformanceCase, EngineManifest]],
) -> None:
    config = run_manifest.config
    artifact_entries: dict[str, object] = {}
    for engine in run_manifest.engines:
        if engine.name == "llama-cpp":
            path = Path(config.peer_artifact) if config.peer_artifact is not None else None
        else:
            path = Path(config.artifact)
        if path is None:
            continue
        digest = _digest_file(path)
        artifact_entries[engine.name] = {
            "name": path.name,
            "sha256": digest,
            "semantic_id": _artifact_identity(config, engine)["semantic_id"],
        }
    model_files = {
        name: _digest_file(Path(config.model_dir) / name)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
        if (Path(config.model_dir) / name).is_file()
    }
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": run_manifest.run_id,
        "suite": {
            "name": run_manifest.suite.name,
            "version": run_manifest.suite.version,
            "quality_version": run_manifest.suite.quality_version,
            "budget_seconds": run_manifest.suite.budget_seconds,
            "minimum_duration_seconds": run_manifest.suite.minimum_duration_seconds,
        },
        "seed": config.seed,
        "budgets": {
            "suite_seconds": run_manifest.suite.budget_seconds,
            "case_seconds": {case.id: case.timeout_seconds for case, _ in entries},
        },
        "commit": run_manifest.commit,
        "dirty": run_manifest.dirty,
        "gpu": {
            "identity": run_manifest.gpu.identity,
            "identity_kind": run_manifest.gpu.identity_kind,
            "identity_source_sha256": run_manifest.gpu.source_sha256,
            "identity_fields": dict(run_manifest.gpu.selected_fields),
            "architecture": run_manifest.gpu.architecture,
            "physical_gpu": run_manifest.gpu.physical_gpu,
            "logical_gpu": run_manifest.gpu.logical_gpu,
        },
        "model": {"files": model_files},
        "artifacts": artifact_entries,
        "engines": [
            {
                "name": engine.name,
                "binary": Path(engine.binary).name,
                "version": _engine_version(run_manifest, engine),
                "adapter_version": adapters.ADAPTER_VERSION,
            }
            for engine in run_manifest.engines
        ],
        "cases": [
            {
                "id": case.id,
                "engine": engine.name,
                "mode": case.mode,
                "cache_state": case.cache_state,
                "warmups": case.warmups,
                "repetitions": case.repetitions,
                "timeout_seconds": case.timeout_seconds,
            }
            for case, engine in entries
        ],
        "status": {"state": "running", "errors": [], "records": []},
    }
    _atomic_json_write(payload, run_manifest.bundle / "manifest.json")


def _update_bundle_manifest(run_manifest: RunManifest, status: BundleStatus) -> None:
    path = run_manifest.bundle / "manifest.json"
    try:
        path.stat()
    except FileNotFoundError:
        raise OSError(f"benchmark bundle manifest is missing: {path}")
    # Let read/decode failures retain their original exception and message.
    # They are finalization failures, not optional diagnostics.
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark bundle manifest is not an object: {path}")
    payload["status"] = {
        "state": status.state,
        "errors": list(status.errors),
        "records": [record.name for record in status.records],
        "quality_failed": status.quality_failed,
        "performance_report_only": status.performance_report_only,
        "elapsed_seconds": status.elapsed_seconds,
        "completed_rounds": status.completed_rounds,
    }
    # Do not return a status until this atomic write/fsync/replace has
    # completed.  A finalization failure must remain visible to the caller.
    _atomic_json_write(payload, path)


def _record_filename(case: PerformanceCase, engine: EngineManifest) -> str:
    if len(case.engines) == 1:
        return f"{case.id}.json"
    return f"{case.id}-{engine.name}.json"


def _write_streams(bundle: Path, case: PerformanceCase, engine: EngineManifest, label: str, stdout: str, stderr: str) -> None:
    stem = f"{case.id}-{engine.name}-{label}"
    _write_log(bundle / "logs" / f"{stem}.stdout.log", stdout)
    _write_log(bundle / "logs" / f"{stem}.stderr.log", stderr)


def _write_quality_streams(bundle: Path, case: QualityCase, suffix: str, stdout: str, stderr: str) -> None:
    stem = f"quality-{case.id}-{suffix}"
    _write_log(bundle / "logs" / f"{stem}.stdout.log", stdout)
    _write_log(bundle / "logs" / f"{stem}.stderr.log", stderr)


def _write_quality_error(bundle: Path, case: QualityCase, message: str) -> None:
    _write_log(bundle / "logs" / f"quality-{case.id}.error.log", message + "\n")


def _write_case_error(bundle: Path, case: PerformanceCase, engine: EngineManifest, message: str) -> None:
    _write_log(bundle / "logs" / f"{case.id}-{engine.name}.error.log", message + "\n")


def _write_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _invoke_process(
    runner: Callable[..., object],
    argv: Sequence[str],
    *,
    timeout: float,
    case_id: str,
    engine_name: str,
) -> ProcessResult:
    vector = tuple(str(item) for item in argv)
    try:
        kwargs: dict[str, object] = {"timeout": timeout}
        try:
            signature = inspect.signature(runner)
            parameters = signature.parameters
            if "case_id" in parameters:
                kwargs["case_id"] = case_id
            if "engine_name" in parameters:
                kwargs["engine_name"] = engine_name
            if "timeout_seconds" in parameters and "timeout" not in parameters:
                kwargs["timeout_seconds"] = timeout
                kwargs.pop("timeout", None)
            if not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
                kwargs = {key: value for key, value in kwargs.items() if key in parameters}
        except (TypeError, ValueError):
            pass
        raw = runner(vector, **kwargs)
    except KeyboardInterrupt:
        return ProcessResult(vector, 130, "", "interrupted", interrupted=True)
    except subprocess.TimeoutExpired as exc:
        return ProcessResult(vector, 124, _text(exc.stdout), _text(exc.stderr), timed_out=True)
    except (TimeoutError, TimeoutError) as exc:
        return ProcessResult(vector, 124, "", str(exc), timed_out=True)
    except OSError as exc:
        return ProcessResult(vector, 127, "", str(exc))
    return _normalize_process_result(vector, raw)


def _normalize_process_result(argv: tuple[str, ...], raw: object) -> ProcessResult:
    if isinstance(raw, ProcessResult):
        return raw
    if isinstance(raw, subprocess.CompletedProcess):
        return ProcessResult(argv, int(raw.returncode), _text(raw.stdout), _text(raw.stderr))
    if all(hasattr(raw, name) for name in ("returncode", "stdout", "stderr")):
        return ProcessResult(
            argv,
            int(getattr(raw, "returncode")),
            _text(getattr(raw, "stdout")),
            _text(getattr(raw, "stderr")),
            float(getattr(raw, "duration_seconds", 0.0)),
            bool(getattr(raw, "timed_out", False)),
            bool(getattr(raw, "interrupted", False)),
        )
    if isinstance(raw, Mapping):
        return ProcessResult(
            argv,
            int(raw.get("returncode", raw.get("exit_code", 0))),
            _text(raw.get("stdout", "")),
            _text(raw.get("stderr", "")),
            float(raw.get("duration_seconds", 0.0)),
            bool(raw.get("timed_out", raw.get("timeout", False))),
            bool(raw.get("interrupted", False)),
        )
    if isinstance(raw, tuple) and len(raw) >= 2:
        return ProcessResult(argv, int(raw[0]), _text(raw[1]), _text(raw[2] if len(raw) > 2 else ""))
    if isinstance(raw, str):
        return ProcessResult(argv, 0, raw, "")
    raise TypeError(f"command runner returned unsupported result: {type(raw).__name__}")


def _coerce_config(config: RunConfig | Mapping[str, object] | object) -> RunConfig:
    if isinstance(config, RunConfig):
        return config
    try:
        return RunConfig(**_config_values(config))
    except TypeError as exc:
        raise ValueError(f"invalid benchmark configuration: {exc}") from exc


def _config_values(config: Mapping[str, object] | object) -> dict[str, object]:
    if isinstance(config, Mapping):
        source = dict(config)
    else:
        source = {
            name: getattr(config, name)
            for name in (
                "suite",
                "model_dir",
                "artifact",
                "gguf_file",
                "peer_artifact",
                "physical_gpu",
                "gpu_arch",
                "architecture",
                "gpu_static_json",
                "rocm_version_file",
                "hip_version_file",
                "rocm_version",
                "hip_version",
                "logical_gpu",
                "output_dir",
                "output",
                "device",
                "context_size",
                "chat",
                "clock_policy",
                "environment",
                "environment_snapshot",
                "engine_binaries",
                "engine_versions",
                "binary_exists",
                "version_outputs",
                "repository",
                "run_id",
                "seed",
                "run_quality",
                "artifact_semantic_id",
                "artifact_quantization",
                "tokenizer_sha256",
                "chat_template_sha256",
                "strict_environment",
                "environment_command_runner",
                "cpu_governor_reader",
            )
            if hasattr(config, name)
        }
    aliases = {
        "gguf_file": "artifact",
        "output": "output_dir",
        "architecture": "gpu_arch",
    }
    for source_key, target_key in aliases.items():
        if target_key not in source and source_key in source:
            source[target_key] = source[source_key]
    defaults = {
        "output_dir": Path("target/benchmarks/candidate"),
        "peer_artifact": None,
        "device": 0,
        "context_size": 32768,
        "chat": False,
        "clock_policy": "uncontrolled-clocks",
        "environment": None,
        "environment_snapshot": None,
        "engine_binaries": {},
        "engine_versions": {},
        "binary_exists": None,
        "version_outputs": {},
        "repository": ROOT,
        "run_id": None,
        "seed": None,
        "run_quality": True,
        "gpu_static_json": None,
        "rocm_version_file": None,
        "hip_version_file": None,
        "rocm_version": None,
        "hip_version": None,
        "logical_gpu": None,
        "artifact_semantic_id": None,
        "artifact_quantization": None,
        "tokenizer_sha256": None,
        "chat_template_sha256": None,
        "strict_environment": False,
        "environment_command_runner": None,
        "cpu_governor_reader": None,
    }
    for key, value in defaults.items():
        source.setdefault(key, value)
    allowed = {field.name for field in RunConfig.__dataclass_fields__.values()}
    return {key: value for key, value in source.items() if key in allowed}


def _required_path(
    value: Path | str | None,
    label: str,
    *,
    directory: bool,
    nonempty: bool = False,
) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{label} is required")
    path = Path(value).expanduser()
    if not path.exists() or (directory and not path.is_dir()) or (not directory and not path.is_file()):
        raise ValueError(f"{label} unavailable: {path}")
    if not os.access(path, os.R_OK):
        raise ValueError(f"{label} unavailable: unreadable")
    if nonempty and path.stat().st_size <= 0:
        raise ValueError(f"{label} unavailable: empty file")
    return path.resolve()


def _read_version_file(path: Path, label: str) -> str:
    """Parse one bounded, human-captured toolchain version into a safe value.

    The raw capture remains a candidate diagnostic, but portable records carry
    only a normalized version identity.  Keeping the parser here (rather than
    serializing the path or whole command output) makes the record independent
    of the host filesystem.
    """

    resolved = _required_path(path, f"{label.lower()}_version_file", directory=False, nonempty=True)
    raw = resolved.read_bytes()
    if len(raw) > VERSION_FILE_MAX_BYTES:
        raise ValueError(
            f"{label.lower()}_version_file exceeds {VERSION_FILE_MAX_BYTES} bytes"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} version file must be UTF-8 text") from exc
    if not text.strip():
        raise ValueError(f"{label} version file is empty")
    if any(char in text for char in "\x00\r"):
        raise ValueError(f"{label} version file contains unsafe control characters")
    # hipcc and rocm-smi use different labels and often emit a short banner.
    # Extract the first version token, then normalize its product prefix so
    # consumers compare structured identities rather than arbitrary banners.
    if label.upper() == "HIP":
        pattern = re.compile(
            r"(?:HIP(?:CC)?|hipcc)\s*(?:version\s*)?[:=]?\s*"
            r"([0-9][A-Za-z0-9._+-]*)",
            re.IGNORECASE,
        )
    else:
        pattern = re.compile(
            r"(?:ROCm|Driver)\s*(?:version\s*)?[:=]?\s*"
            r"([0-9][A-Za-z0-9._+-]*)",
            re.IGNORECASE,
        )
    match = pattern.search(text)
    if match is None:
        # Keep a strict fallback for already-normalized captures such as
        # ``ROCm 6.4.2`` and ``HIP 6.4.2``.
        first = next((line.strip() for line in text.splitlines() if line.strip()), "")
        match = re.search(r"(?:version\s*[:=]?\s*)?([0-9][A-Za-z0-9._+-]*)", first)
        if match is None:
            raise ValueError(f"{label} version file has no parseable version")
        version = match.group(1)
    else:
        version = match.group(1)
    normalized = f"{label} {version}"
    if len(normalized) > VERSION_VALUE_MAX_LENGTH or _VERSION_VALUE_RE.fullmatch(normalized) is None:
        raise ValueError(f"{label} version is empty or exceeds safe bounds")
    return normalized


def _configured_versions(config: RunConfig) -> tuple[str, str]:
    rocm = config.rocm_version
    hip = config.hip_version
    if rocm is None and config.rocm_version_file is not None:
        rocm = _read_version_file(Path(config.rocm_version_file), "ROCm")
    if hip is None and config.hip_version_file is not None:
        hip = _read_version_file(Path(config.hip_version_file), "HIP")
    if not rocm or not hip:
        raise ValueError("ROCm and HIP version identities are required")
    for label, value in (("ROCm", rocm), ("HIP", hip)):
        if not isinstance(value, str) or _VERSION_VALUE_RE.fullmatch(value) is None:
            raise ValueError(f"{label} version identity is unsafe")
        if not value.startswith(f"{label} "):
            raise ValueError(f"{label} version identity must use the structured prefix")
        if re.search(r"\bunknown\b", value, re.IGNORECASE):
            raise ValueError(f"{label} version identity must not be unknown")
    return rocm, hip


def _engine_version(run_manifest: RunManifest, engine: EngineManifest) -> str:
    # The source commit is the deterministic SuperSonic build identity.  The
    # dirty bit is independently recorded in run.dirty and must not turn a
    # portable version into a host path or a generic "unknown" value.
    if engine.name == "supersonic":
        value = f"source-{run_manifest.commit}"
    else:
        value = engine.pinned_version or run_manifest.config.engine_versions.get(engine.name)
    if not isinstance(value, str) or not value.strip() or value.strip().lower() == "unknown":
        raise ValueError(f"{engine.name} requires a non-unknown version identity")
    value = value.strip()
    if len(value) > VERSION_VALUE_MAX_LENGTH or _VERSION_VALUE_RE.fullmatch(value) is None:
        raise ValueError(f"{engine.name} version identity is unsafe")
    return value


def _validate_model_files(model_dir: Path, *, chat: bool) -> None:
    required = ["config.json", "tokenizer.json"]
    if chat:
        required.append("tokenizer_config.json")
    for name in required:
        _required_path(model_dir / name, f"model file {name}", directory=False, nonempty=True)


def _validate_model_digests(model_dir: Path, config: RunConfig) -> None:
    actual_tokenizer = _digest_file(model_dir / "tokenizer.json")
    if config.tokenizer_sha256 and config.tokenizer_sha256 != actual_tokenizer:
        raise ValueError("tokenizer_sha256 does not match tokenizer.json")
    if config.chat_template_sha256:
        template = model_dir / "tokenizer_config.json"
        if not template.is_file():
            raise ValueError("chat_template_sha256 requires tokenizer_config.json")
        payload = parse_strict_json(template.read_text(encoding="utf-8"), context="tokenizer_config.json")
        if not isinstance(payload, Mapping):
            raise ValueError("tokenizer_config.json must contain an object")
        chat_template = payload.get("chat_template")
        if not isinstance(chat_template, str) or not chat_template:
            raise ValueError("tokenizer_config.json must contain a non-empty chat_template string")
        actual_chat_template = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
        if config.chat_template_sha256 != actual_chat_template:
            raise ValueError("chat_template_sha256 does not match tokenizer_config.json chat_template")


def _validate_active_cases(suite: SuiteManifest) -> None:
    warm_cases = [case.id for case in suite.performance_cases if case.cache_state == "warm-resident"]
    if warm_cases:
        raise ValueError(
            "warm-resident cases are not executable until same-process adapter reuse is verified: "
            + ", ".join(warm_cases)
        )
    prefix_cases = [case.id for case in suite.performance_cases if case.cache_state.startswith("prefix-cache-")]
    if prefix_cases:
        raise ValueError(
            "prefix cache cases are not executable until adapter transitions are verified: "
            + ", ".join(prefix_cases)
        )


def _validate_locked_policy(policy: object) -> None:
    requested = _policy_mapping(policy)
    required = ("gpu_clock_mhz", "memory_clock_mhz", "power_cap_watts")
    missing = [key for key in required if requested.get(key) in (None, "")]
    if missing:
        raise ValueError("locked clock policy requires requested " + ", ".join(missing))
    for key in required:
        value = requested[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"locked clock policy {key} must be positive")


def _snapshot_policy_name(snapshot: object) -> str:
    if isinstance(snapshot, Mapping):
        value = snapshot.get("clock_policy")
    else:
        value = getattr(snapshot, "clock_policy", None)
    return str(value or "")


def _validate_snapshot_policy_values(snapshot: object, policy: object) -> None:
    if _clock_policy_name(policy) != "locked":
        return
    if isinstance(snapshot, Mapping):
        requested = snapshot.get("requested")
    else:
        requested = getattr(snapshot, "requested", None)
    if hasattr(requested, "__dataclass_fields__"):
        requested = asdict(requested)
    if not isinstance(requested, Mapping):
        raise ValueError("locked environment_snapshot is missing requested telemetry")
    configured = _policy_mapping(policy)
    for key in (
        "gpu_clock_mhz",
        "clock_tolerance_mhz",
        "memory_clock_mhz",
        "power_cap_watts",
        "performance_level",
    ):
        if configured.get(key) is not None and requested.get(key) != configured.get(key):
            raise ValueError(f"environment_snapshot requested {key} does not match clock policy")


def _default_environment_probe_runner(argv: tuple[str, ...]) -> str:
    result = subprocess.run(
        tuple(str(value) for value in argv),
        shell=False,
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    output = _text(result.stdout) + _text(result.stderr)
    if result.returncode != 0:
        raise ValueError(f"environment probe failed with exit code {result.returncode}")
    if not output.strip():
        raise ValueError("environment probe returned empty evidence")
    return output


def _process_environment(config: RunConfig) -> dict[str, str]:
    result = dict(os.environ)
    if config.environment is not None:
        result.update({str(key): str(value) for key, value in config.environment.items()})
    return result


def _probe_environment(
    config: RunConfig,
    runner: Callable[[tuple[str, ...]], str],
    notes: list[str],
) -> environment.ObservedTelemetry:
    command = environment.build_probe_command(str(config.physical_gpu))
    raw = runner(command)
    if isinstance(raw, ProcessResult):
        if raw.returncode != 0:
            raise ValueError(f"environment probe failed with exit code {raw.returncode}")
        text = raw.stdout + raw.stderr
    else:
        text = _text(raw)
    if not text.strip():
        raise ValueError("environment probe returned empty evidence")
    return environment.parse_showallinfo(text, notes)


def _required_text(value: object, label: str) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"{label} is required")
    return str(value)


def _validate_peer_artifact(suite: SuiteManifest, config: RunConfig) -> None:
    needs_peer = any("llama-cpp" in case.engines for case in suite.performance_cases)
    if config.peer_artifact is not None:
        try:
            _required_path(config.peer_artifact, "llama-cpp peer artifact", directory=False, nonempty=True)
        except ValueError as exc:
            raise ValueError(f"llama-cpp peer artifact unavailable: {config.peer_artifact}") from exc
        return
    if needs_peer:
        raise ValueError("llama-cpp peer_artifact is required for configured cases")


def _engine_with_override(engine: EngineManifest, config: RunConfig) -> EngineManifest:
    override = config.engine_binaries.get(engine.name)
    if override is None:
        return engine
    return replace(engine, binary=str(override), version_command=(str(override), "--version"))


def _binary_available(engine: EngineManifest, config: RunConfig) -> bool:
    override = config.binary_exists
    if override is not None:
        if isinstance(override, Mapping):
            value = override.get(engine.name)
            if value is not None:
                return bool(value)
        elif callable(override):
            try:
                parameters = inspect.signature(override).parameters
                parameter = next(iter(parameters.values()), None)
                label = parameter.name.lower() if parameter is not None else ""
            except (TypeError, ValueError):
                label = ""
            if any(word in label for word in ("engine", "manifest")):
                value = override(engine)
            elif any(word in label for word in ("name", "binary", "path", "command")):
                value = override(engine.name if "name" in label else engine.binary)
            else:
                try:
                    value = override(engine)
                except (AttributeError, TypeError):
                    value = override(engine.name)
            return bool(value)
        elif isinstance(override, bool):
            return override
    binary = Path(engine.binary)
    if binary.is_absolute() or "/" in engine.binary or "\\" in engine.binary:
        candidate = binary if binary.is_absolute() else Path(config.repository) / binary
        return candidate.is_file() and os.access(candidate, os.X_OK)
    return shutil.which(engine.binary) is not None


def _validate_engine_available(engine: EngineManifest, config: RunConfig) -> None:
    if not _binary_available(engine, config):
        raise ValueError(f"{engine.name} unavailable: configured binary {engine.binary!r} was not found")


def _validate_engine_version(engine: EngineManifest, config: RunConfig) -> None:
    pinned = engine.pinned_version
    if pinned is None:
        return
    supplied = config.engine_versions.get(engine.name)
    # A configured string is metadata only.  The pinned peer must still emit
    # the exact non-empty version identity from its own --version command.
    if engine.name == "llama-cpp":
        supplied = None
    if supplied is not None:
        if not str(supplied).strip() or str(supplied).strip() != pinned:
            raise ValueError(f"{engine.name} version mismatch: expected {pinned!r}, got {supplied!r}")
        return
    output = config.version_outputs.get(engine.name)
    if engine.name == "llama-cpp":
        output = None
    if output is not None:
        if not str(output).strip() or str(output).strip() != pinned:
            raise ValueError(f"{engine.name} version mismatch: expected {pinned!r}")
        return
    try:
        completed = subprocess.run(
            tuple(engine.version_command),
            shell=False,
            cwd=str(config.repository),
            timeout=30,
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"{engine.name} unavailable while checking pinned version: {exc}") from exc
    output = _text(completed.stdout) + _text(completed.stderr)
    if completed.returncode != 0:
        raise ValueError(f"{engine.name} unavailable while checking pinned version")
    version_line = next((line.strip() for line in output.splitlines() if line.strip()), "")
    if not version_line or len(version_line.encode("utf-8")) > 4096 or version_line != pinned:
        raise ValueError(f"{engine.name} version mismatch: expected {pinned!r}")


def _run_id(config: RunConfig, suite: SuiteManifest) -> str:
    if config.run_id is not None:
        value = str(config.run_id)
        if SAFE_ID.fullmatch(value) is None:
            raise ValueError("run_id must contain lowercase letters, digits, and hyphens")
        return value
    return f"{suite.name}-{datetime.now(tz=UTC).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _git_identity(repository: Path) -> tuple[str, bool]:
    try:
        commit_result = subprocess.run(
            ("git", "-C", str(repository), "rev-parse", "HEAD"),
            shell=False,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        status_result = subprocess.run(
            ("git", "-C", str(repository), "status", "--porcelain"),
            shell=False,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        commit = _text(commit_result.stdout).strip()
        if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            return "0" * 40, False
        return commit, bool(_text(status_result.stdout).strip())
    except (OSError, subprocess.TimeoutExpired):
        return "0" * 40, False


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_identity(config: RunConfig, engine: EngineManifest) -> dict[str, str | None]:
    """Return only evidence belonging to the artifact consumed by *engine*.

    A peer adapter gets no semantic, tokenizer, or template identity from the
    SuperSonic configuration.  Its artifact digest is the conservative local
    identity for every field whose equivalence would otherwise be an
    unverified assertion.
    """

    if engine.name == "llama-cpp":
        artifact = Path(config.peer_artifact) if config.peer_artifact is not None else None
        if artifact is None:
            raise ValueError("llama-cpp peer artifact is required for artifact evidence")
        digest = _digest_file(artifact)
        primary_digest = _digest_file(Path(config.artifact))
        same_weights = digest == primary_digest
        semantic_id = config.artifact_semantic_id or f"artifact-sha256-{digest}"
        quantization = config.artifact_quantization or "GQH-Q4"
        if not same_weights:
            semantic_id = f"artifact-sha256-{digest}"
            quantization = f"artifact-sha256-{digest}"
        return {
            "semantic_id": semantic_id,
            "quantization": quantization,
            "sha256": digest,
            "tokenizer_sha256": None,
            "chat_template_sha256": None,
        }

    artifact = Path(config.artifact)
    digest = _digest_file(artifact)
    model_dir = Path(config.model_dir)
    tokenizer_digest = config.tokenizer_sha256 or _digest_file(model_dir / "tokenizer.json")
    if config.chat_template_sha256:
        chat_digest = config.chat_template_sha256
    elif (model_dir / "tokenizer_config.json").is_file():
        chat_digest = _digest_file(model_dir / "tokenizer_config.json")
    else:
        chat_digest = hashlib.sha256(b"").hexdigest()
    return {
        "semantic_id": config.artifact_semantic_id or f"artifact-sha256-{digest}",
        "quantization": config.artifact_quantization or "GQH-Q4",
        "sha256": digest,
        "tokenizer_sha256": tokenizer_digest,
        "chat_template_sha256": chat_digest,
    }


def _safe_argv(argv: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for value in argv:
        item = str(value)
        if item.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", item):
            result.append(re.split(r"[\\/]", item)[-1] or "path")
        else:
            result.append(item)
    return tuple(result)


def _safe_environment(values: Mapping[str, object]) -> dict[str, str]:
    allowed = set(environment.ALLOWLISTED_ENVIRONMENT)
    result: dict[str, str] = {}
    for key, value in values.items():
        if key not in allowed:
            continue
        text = str(value)
        if text.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", text):
            text = re.split(r"[\\/]", text)[-1] or "path"
        result[key] = text
    return result


def _clock_policy_name(policy: object) -> str:
    name = policy.get("name") if isinstance(policy, Mapping) else getattr(policy, "name", policy)
    text = str(name)
    if text not in {"locked", "uncontrolled-clocks"}:
        raise ValueError(f"unsupported clock policy: {text!r}")
    return text


def _policy_mapping(policy: object) -> Mapping[str, object]:
    if isinstance(policy, Mapping):
        return policy
    return {
        key: getattr(policy, key, None)
        for key in ("gpu_clock_mhz", "memory_clock_mhz", "power_cap_watts", "performance_level")
    }


def _optional_policy_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("clock policy integer values must be numeric")
    return int(value)


def _optional_policy_text(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _utc_timestamp() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _append_error(errors: list[str], value: str) -> None:
    if value not in errors:
        errors.append(value)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "BenchmarkConfig",
    "BundleStatus",
    "Config",
    "EXPECTED_BUDGETS",
    "FULL_BUDGET_SECONDS",
    "ProcessResult",
    "QUICK_BUDGET_SECONDS",
    "RunConfig",
    "RunManifest",
    "SUPPORTED_GPU_ARCHES",
    "ordered_cases",
    "preflight",
    "replace_config",
    "run_process",
    "run_suite",
]
