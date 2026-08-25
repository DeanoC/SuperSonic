from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Mapping

from . import environment, validation
from .manifest import ROOT


QUALIFICATION_SCHEMA = ROOT / "benchmarks" / "schema" / "qualification-v1.schema.json"
_SERIES_KEYS = frozenset(("schema_version", "run_id", "binding", "samples", "median_ms_per_token"))
_BINDING_KEYS = frozenset(
    (
        "commit", "rocm_version", "hip_version", "compiler_version",
        "scalar_instruction_sha256", "artifact_semantic_id", "artifact_quantization",
        "artifact_sha256", "artifact_source_repository", "artifact_source_revision",
        "artifact_filename", "artifact_size_bytes", "tokenizer_sha256",
        "chat_template_sha256", "prompt_sha256",
        "measurement_boundary", "gpu_identity", "pci_bdf", "gpu_clock_mhz",
        "gpu_clock_tolerance_mhz", "memory_clock_mhz", "power_cap_watts",
        "performance_level", "cache_state", "process_reuse", "filesystem_flush",
        "temperature_limit_celsius",
    )
)
_SAMPLE_KEYS = frozenset(("sample_id", "lm_head_ms", "timed_decode_steps"))


def directory_digest(root: str | Path) -> str:
    directory = Path(root)
    if not directory.is_dir():
        raise ValueError(f"bundle path is not a directory: {directory}")
    entries = tuple(sorted(directory.rglob("*")))
    symlinks = [path for path in entries if path.is_symlink()]
    if symlinks:
        raise ValueError("bundle directory must not contain symbolic links")
    files = tuple(path for path in entries if path.is_file())
    if not files:
        raise ValueError("bundle directory is empty")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(directory).as_posix()
        payload_digest = hashlib.sha256(path.read_bytes()).digest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload_digest)
        digest.update(b"\0")
    return digest.hexdigest()


def load_series(path: str | Path) -> dict[str, object]:
    target = Path(path)
    if target.is_dir():
        target = target / "baseline-v1.json"
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid scalar series: {target}") from exc
    validate_series(value)
    return value


def series_from_record(
    record: Mapping[str, object],
    *,
    compiler_version: str,
    scalar_instruction_sha256: str,
) -> dict[str, object]:
    validation.validate_record(record)
    if not validation.has_verified_headline(record):
        raise ValueError("scalar series source must have verified loaded-clock evidence")
    engine = record["engine"]
    workload = record["workload"]
    env = record["environment"]
    hardware = record["hardware"]
    artifact = record["artifact"]
    run = record["run"]
    assert all(isinstance(item, Mapping) for item in (engine, workload, env, hardware, artifact, run))
    if engine["name"] != "supersonic-scalar-lab" or engine["version"] != "scalar-head-lab-v1":
        raise ValueError("series source must be the source-fixed scalar lab engine")
    if hardware["identity_kind"] != "pci_bdf":
        raise ValueError("scalar qualification requires a stable PCI BDF identity")
    if workload["mode"] != "ordinary" or workload["cache_state"] != "cold-load":
        raise ValueError("series source must be an ordinary cold-load case")
    cache = env["cache_evidence"]
    requested = env["requested"]
    assert isinstance(cache, Mapping) and isinstance(requested, Mapping)
    if cache.get("filesystem_flush") != "unavailable":
        raise ValueError("series source must explicitly record filesystem_flush=unavailable")
    raw_samples = record["samples"]
    telemetry = env.get("telemetry_samples")
    assert isinstance(raw_samples, list)
    if not isinstance(telemetry, list):
        raise ValueError("scalar series source requires telemetry samples")
    if len(raw_samples) < 7:
        raise ValueError("scalar series source requires at least seven samples")
    samples = []
    previous_end = -1
    for index, sample in enumerate(raw_samples[:7], 1):
        if not isinstance(sample, Mapping) or "lm_head_ms" not in sample or "timed_decode_steps" not in sample:
            raise ValueError("scalar series samples require lm_head_ms and timed_decode_steps")
        start = sample.get("telemetry_start_index")
        count = sample.get("telemetry_sample_count")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or start < 0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            raise ValueError("each scalar timing requires a non-empty telemetry association")
        end = start + count
        if start < previous_end or end > len(telemetry):
            raise ValueError("scalar timing telemetry associations overlap or exceed the evidence")
        previous_end = end
        associated = telemetry[start:end]
        if any(not isinstance(item, Mapping) or not item.get("raw_amd_smi_json") for item in associated):
            raise ValueError("each scalar timing requires raw AMD SMI JSON")
        observed = tuple(_observed(item) for item in associated)
        violations = environment.verify_clock_policy(
            observed[0], list(observed[1:-1]), observed[-1],
            {
                "name": env["clock_policy"],
                "gpu_clock_mhz": requested["gpu_clock_mhz"],
                "clock_tolerance_mhz": requested["clock_tolerance_mhz"],
                "memory_clock_mhz": requested["memory_clock_mhz"],
                "power_cap_watts": requested["power_cap_watts"],
                "performance_level": requested["performance_level"],
                "temperature_limit_celsius": requested["temperature_limit_celsius"],
            },
        )
        if violations:
            raise ValueError(f"scalar timing {index} telemetry failed policy: {'; '.join(violations)}")
        summary = environment.loaded_clock_summary(tuple(_telemetry_sample(item) for item in associated))
        expected_summary = {
            "minimum_mhz": sample.get("loaded_clock_minimum_mhz"),
            "median_mhz": sample.get("loaded_clock_median_mhz"),
            "maximum_mhz": sample.get("loaded_clock_maximum_mhz"),
        }
        if any(summary[name] != expected_summary[name] for name in expected_summary):
            raise ValueError(f"scalar timing {index} loaded-clock summary does not match raw telemetry")
        samples.append(
            {
                "sample_id": f"{run['run_id']}-{index}",
                "lm_head_ms": sample["lm_head_ms"],
                "timed_decode_steps": sample["timed_decode_steps"],
            }
        )
    values = [float(sample["lm_head_ms"]) / int(sample["timed_decode_steps"]) for sample in samples]
    value = {
        "schema_version": 1,
        "run_id": run["run_id"],
        "binding": {
            "commit": run["commit"],
            "rocm_version": env["rocm_version"],
            "hip_version": env["hip_version"],
            "compiler_version": compiler_version,
            "scalar_instruction_sha256": scalar_instruction_sha256,
            "artifact_semantic_id": artifact["semantic_id"],
            "artifact_quantization": artifact["quantization"],
            "artifact_sha256": artifact["sha256"],
            "artifact_source_repository": artifact["source_repository"],
            "artifact_source_revision": artifact["source_revision"],
            "artifact_filename": artifact["filename"],
            "artifact_size_bytes": artifact["size_bytes"],
            "tokenizer_sha256": artifact["tokenizer_sha256"],
            "chat_template_sha256": artifact["chat_template_sha256"],
            "prompt_sha256": workload["prompt_sha256"],
            "measurement_boundary": "lm_head_ms/timed_decode_steps",
            "gpu_identity": hardware["identity"],
            "pci_bdf": hardware["identity"],
            "gpu_clock_mhz": requested["gpu_clock_mhz"],
            "gpu_clock_tolerance_mhz": requested["clock_tolerance_mhz"],
            "memory_clock_mhz": requested["memory_clock_mhz"],
            "power_cap_watts": requested["power_cap_watts"],
            "performance_level": requested["performance_level"],
            "temperature_limit_celsius": requested["temperature_limit_celsius"],
            "cache_state": workload["cache_state"],
            "process_reuse": env["process_reuse"],
            "filesystem_flush": cache["filesystem_flush"],
        },
        "samples": samples,
        "median_ms_per_token": statistics.median(values),
    }
    validate_series(value)
    return value


def _observed(value: Mapping[str, object]) -> environment.ObservedTelemetry:
    return environment.ObservedTelemetry(
        gpu_clock_mhz=value.get("gpu_clock_mhz"),
        memory_clock_mhz=value.get("memory_clock_mhz"),
        power_cap_watts=value.get("power_cap_watts"),
        power_watts=value.get("power_watts"),
        temperature_celsius=value.get("temperature_celsius"),
        gpu_utilization_percent=value.get("gpu_utilization_percent"),
        memory_utilization_percent=value.get("memory_utilization_percent"),
        performance_level=value.get("performance_level"),
        throttle_status=value.get("throttle_status"),
        indep_throttle_status=value.get("indep_throttle_status"),
        throttle_label=value.get("throttle_label"),
    )


def _telemetry_sample(value: Mapping[str, object]) -> environment.TelemetrySample:
    observed = _observed(value)
    return environment.TelemetrySample(
        offset_seconds=float(value["offset_seconds"]),
        gpu_clock_mhz=observed.gpu_clock_mhz,
        memory_clock_mhz=observed.memory_clock_mhz,
        power_cap_watts=observed.power_cap_watts,
        power_watts=observed.power_watts,
        temperature_celsius=observed.temperature_celsius,
        gpu_utilization_percent=observed.gpu_utilization_percent,
        memory_utilization_percent=observed.memory_utilization_percent,
        performance_level=observed.performance_level,
        throttle_status=observed.throttle_status,
        indep_throttle_status=observed.indep_throttle_status,
        throttle_label=observed.throttle_label,
        raw_amd_smi_json=str(value["raw_amd_smi_json"]),
    )


def validate_series(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("scalar series must be an object")
    _exact_keys(value, _SERIES_KEYS, "scalar series")
    if value["schema_version"] != 1:
        raise ValueError("scalar series schema_version must be 1")
    _safe_id(value["run_id"], "run_id")
    binding = value["binding"]
    if not isinstance(binding, dict):
        raise ValueError("binding must be an object")
    _exact_keys(binding, _BINDING_KEYS, "binding")
    for name in ("commit",):
        _hex(binding[name], 40, name)
    for name in (
        "scalar_instruction_sha256", "artifact_sha256",
        "tokenizer_sha256", "chat_template_sha256", "prompt_sha256",
    ):
        _hex(binding[name], 64, name)
    _hex(binding["artifact_source_revision"], 40, "artifact_source_revision")
    for name in (
        "rocm_version", "hip_version", "compiler_version", "gpu_identity", "pci_bdf",
        "performance_level", "artifact_semantic_id", "artifact_quantization",
        "artifact_source_repository", "artifact_filename",
    ):
        if not isinstance(binding[name], str) or not binding[name]:
            raise ValueError(f"{name} must be non-empty text")
    if binding["measurement_boundary"] != "lm_head_ms/timed_decode_steps":
        raise ValueError("measurement_boundary must be lm_head_ms/timed_decode_steps")
    for name in ("gpu_clock_mhz", "gpu_clock_tolerance_mhz", "memory_clock_mhz", "power_cap_watts"):
        if isinstance(binding[name], bool) or not isinstance(binding[name], int) or binding[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    _positive_number(binding["temperature_limit_celsius"], "temperature_limit_celsius")
    if isinstance(binding["artifact_size_bytes"], bool) or not isinstance(binding["artifact_size_bytes"], int) or binding["artifact_size_bytes"] <= 0:
        raise ValueError("artifact_size_bytes must be a positive integer")
    if binding["cache_state"] != "cold-load" or binding["process_reuse"] is not False:
        raise ValueError("scalar qualification requires cold-load fresh-process evidence")
    if binding["filesystem_flush"] != "unavailable":
        raise ValueError("filesystem_flush must be unavailable")
    samples = value["samples"]
    if not isinstance(samples, list) or len(samples) != 7:
        raise ValueError("scalar series requires exactly seven samples")
    ids: set[str] = set()
    per_token: list[float] = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("sample must be an object")
        _exact_keys(sample, _SAMPLE_KEYS, "sample")
        _safe_id(sample["sample_id"], "sample_id")
        if sample["sample_id"] in ids:
            raise ValueError("sample_id values must be unique")
        ids.add(sample["sample_id"])
        lm_head_ms = _positive_number(sample["lm_head_ms"], "lm_head_ms")
        steps = sample["timed_decode_steps"]
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("timed_decode_steps must be a positive integer")
        per_token.append(lm_head_ms / steps)
    dispersion = environment.verify_timing_dispersion(per_token)
    if dispersion:
        raise ValueError("; ".join(dispersion))
    stored_median = _positive_number(value["median_ms_per_token"], "median_ms_per_token")
    actual_median = statistics.median(per_token)
    if not math.isclose(stored_median, actual_median, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("median_ms_per_token does not match the seven samples")


def qualify_series(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    baseline_bundle_sha256: str,
    limit_percent: float = 5.0,
) -> dict[str, object]:
    validate_series(baseline)
    validate_series(candidate)
    _hex(baseline_bundle_sha256, 64, "baseline_bundle_sha256")
    if limit_percent != 5.0:
        raise ValueError("qualification regression limit must remain 5.0 percent")
    baseline_binding = baseline["binding"]
    candidate_binding = candidate["binding"]
    assert isinstance(baseline_binding, Mapping) and isinstance(candidate_binding, Mapping)
    mismatches = [name for name in sorted(_BINDING_KEYS) if baseline_binding[name] != candidate_binding[name]]
    if mismatches:
        raise ValueError(f"non-comparable scalar binding fields: {', '.join(mismatches)}")
    baseline_median = float(baseline["median_ms_per_token"])
    candidate_median = float(candidate["median_ms_per_token"])
    ratio = candidate_median / baseline_median
    regression = (ratio - 1.0) * 100.0
    result = {
        "schema_version": 1,
        "qualified": candidate_median <= baseline_median * 1.05,
        "limit_percent": 5.0,
        "baseline": _summary(baseline, baseline_bundle_sha256),
        "candidate": _summary(candidate, None),
        "ratio": ratio,
        "percent_regression": regression,
        "compatibility_errors": [],
    }
    validate_qualification(result)
    return result


def validate_qualification(value: object) -> None:
    schema = validation._load_json(QUALIFICATION_SCHEMA)
    if not isinstance(schema, dict):
        raise ValueError("qualification schema must be an object")
    validation._check_schema_keywords(schema, "$qualification-schema")
    validation._validate_schema(value, schema, "$")
    assert isinstance(value, Mapping)
    baseline = value["baseline"]
    candidate = value["candidate"]
    assert isinstance(baseline, Mapping) and isinstance(candidate, Mapping)
    for label, summary in (("baseline", baseline), ("candidate", candidate)):
        sample_ids = summary["sample_ids"]
        if not isinstance(sample_ids, list) or len(sample_ids) != 7 or len(set(sample_ids)) != 7:
            raise ValueError(f"{label} must bind exactly seven unique sample IDs")
    expected_ratio = float(candidate["median_ms_per_token"]) / float(baseline["median_ms_per_token"])
    if not math.isclose(float(value["ratio"]), expected_ratio, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("qualification ratio does not match stored medians")
    expected_regression = (expected_ratio - 1.0) * 100.0
    if not math.isclose(float(value["percent_regression"]), expected_regression, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("qualification percent_regression does not match ratio")
    if value["qualified"] is not (expected_ratio <= 1.05):
        raise ValueError("qualification decision does not match fixed 5% limit")


def _summary(series: Mapping[str, object], digest: str | None) -> dict[str, object]:
    samples = series["samples"]
    assert isinstance(samples, list)
    result: dict[str, object] = {
        "run_id": series["run_id"],
        "sample_ids": [sample["sample_id"] for sample in samples],
        "median_ms_per_token": series["median_ms_per_token"],
        "binding_sha256": hashlib.sha256(
            json.dumps(series["binding"], sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    if digest is not None:
        result["bundle_sha256"] = digest
    return result


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ValueError(f"{label} has additional={unknown} missing={missing}")


def _hex(value: object, length: int, label: str) -> None:
    if not isinstance(value, str) or len(value) != length or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be {length} lowercase hexadecimal characters")


def _safe_id(value: object, label: str) -> None:
    if not isinstance(value, str) or not value or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789-" for char in value):
        raise ValueError(f"{label} must be a lowercase hyphenated identifier")


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)
