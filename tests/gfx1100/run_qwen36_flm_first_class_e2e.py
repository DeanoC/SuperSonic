#!/usr/bin/env python3
"""Prepare a validated native-int4 Qwen3.6 FLM artifact for the E2E lane."""

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
from enum import Enum, auto
from pathlib import Path
from typing import NamedTuple


ROOT = Path(__file__).resolve().parents[2]
BENCH_SCRIPT = ROOT / "tests" / "gfx1100" / "bench_qwen36_he_supersonic.py"
STRICT_PROFILE = "supersonic-qwen36-moe-row-group-int4"
EXPECTED_RESOLVED_MODEL = "qwen3.6-35b-a3b"
EXPECTED_FLM_WEIGHT_MODE = "INT4 native FLM"
EXPECTED_ROW_GROUP_INT4_PROJECTIONS = 330
DEFAULT_HF_SOURCE = Path("/mnt/data/models/Qwen3.6-35B-A3B")
DEFAULT_GEOQUANT_ROOT = Path("/home/deano/projects/geo-quant")
DEFAULT_GEOQUANT_PYTHON = Path(
    "/home/deano/projects/geo-quant/.venv-rocm/bin/python"
)
DEFAULT_FLM = Path(
    "/mnt/data/runs/geo-quant/"
    "qwen36-35b-a3b-supersonic-native-int4-current.flm"
)
E2E_PROMPT = {"id": "flm-first-class-e2e", "prompt": "Hello"}
PROCESS_TERMINATION_GRACE_SECONDS = 5
STORAGE_DIRECT_BACKENDS = frozenset(
    {"gpu-direct-storage", "gds", "hipfile"}
)


class ArtifactAction(Enum):
    REUSE = auto()
    REGENERATE = auto()


class ArtifactPreparation(NamedTuple):
    artifact: Path
    action: str
    source: Path
    destination: Path
    before_sha256: str | None
    after_sha256: str

    def evidence(self) -> dict[str, str | None]:
        return {
            "action": self.action,
            "source": str(self.source),
            "destination": str(self.destination),
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
        }


class PhaseError(RuntimeError):
    pass


def export_command(args: argparse.Namespace, output: Path) -> list[str]:
    return [
        str(args.geoquant_python),
        "scripts/quantize_qwen36_int4.py",
        "--bf16",
        str(args.hf_source),
        "--flm-out",
        str(output),
        "--flm-only",
        "--device",
        args.quant_device,
        "--bits",
        "4",
        "--group-size",
        "32",
        "--flm-int4-codec",
        "row-group",
        "--int4-recipe",
        "mse",
        "--hf-compat-assets",
        "omit",
        "--flm-validate-profile",
        STRICT_PROFILE,
    ]


def validate_command(
    args: argparse.Namespace,
    artifact: Path,
    *,
    verify_payload_hashes: bool,
) -> list[str]:
    command = [
        str(args.geoquant_python),
        "-m",
        "geoquant.formats.flm_validate",
        str(artifact),
        "--profile",
        STRICT_PROFILE,
    ]
    if verify_payload_hashes:
        command.append("--verify-payload-hashes")
    return command


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
    phase: str,
    check: bool = True,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    stdout = subprocess.PIPE if capture_output else None
    stderr = subprocess.PIPE if capture_output else None
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            text=True,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    except OSError as exc:
        raise PhaseError(f"{phase} failed to start: {exc}: {' '.join(command)}") from exc
    try:
        command_stdout, command_stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_and_reap_process_group(process)
        raise PhaseError(f"{phase} timed out after {timeout}s") from exc
    except OSError as exc:
        raise PhaseError(f"{phase} failed while running: {exc}") from exc
    result = subprocess.CompletedProcess(
        command,
        process.returncode,
        command_stdout,
        command_stderr,
    )
    if check and result.returncode != 0:
        raise PhaseError(
            f"{phase} failed with exit {result.returncode}: {' '.join(command)}"
        )
    return result


def _terminate_and_reap_process_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.communicate()


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


def choose_artifact_action(args: argparse.Namespace) -> ArtifactAction:
    try:
        artifact_exists = args.flm.exists()
    except OSError as exc:
        raise PhaseError(f"input discovery failed for FLM artifact: {exc}") from exc
    if args.regenerate:
        return ArtifactAction.REGENERATE
    if artifact_exists:
        return ArtifactAction.REUSE
    raise PhaseError(
        f"input discovery failed: FLM artifact does not exist: {args.flm}; "
        "pass --regenerate to create it explicitly"
    )


def discover_inputs(args: argparse.Namespace, action: ArtifactAction) -> None:
    required = [
        (args.geoquant_root, "geo-quant root", Path.is_dir),
        (args.geoquant_python, "geo-quant Python", Path.is_file),
        (args.binary, "SuperSonic binary", Path.is_file),
        (BENCH_SCRIPT, "benchmark script", Path.is_file),
    ]
    if action is ArtifactAction.REGENERATE:
        required.append((args.hf_source, "HF source", Path.is_dir))
    else:
        required.append((args.flm, "FLM artifact", Path.is_file))

    for path, label, predicate in required:
        try:
            exists = predicate(path)
        except OSError as exc:
            raise PhaseError(f"input discovery failed for {label}: {exc}") from exc
        if not exists:
            raise PhaseError(f"input discovery failed: {label} does not exist: {path}")


def partial_artifact_path(artifact: Path) -> Path:
    return artifact.with_name(f".{artifact.name}.partial-{os.getpid()}")


def regenerated_artifact_path(artifact: Path) -> Path:
    return artifact.with_name(f"{artifact.stem}.regenerated{artifact.suffix}")


def regeneration_destination(args: argparse.Namespace) -> Path:
    if args.regenerate_output is not None:
        return args.regenerate_output
    return regenerated_artifact_path(args.flm) if args.flm.exists() else args.flm


def artifact_sha256(artifact: Path) -> str:
    digest = hashlib.sha256()
    try:
        with artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PhaseError(f"artifact digest failed for {artifact}: {exc}") from exc
    return digest.hexdigest()


def benchmark_prompt_path(out_json: Path) -> Path:
    return out_json.with_suffix(".prompts.jsonl")


def write_benchmark_prompts(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(E2E_PROMPT) + "\n", encoding="utf-8")


def prepare_artifact(args: argparse.Namespace) -> ArtifactPreparation:
    action = choose_artifact_action(args)
    if action is ArtifactAction.REUSE:
        before_sha256 = artifact_sha256(args.flm)
        if not probe_validation(args, args.flm):
            raise PhaseError(
                f"reuse validation failed for {args.flm}; "
                "the supplied artifact was not modified"
            )
        run_command(
            validate_command(args, args.flm, verify_payload_hashes=True),
            cwd=args.geoquant_root,
            timeout=args.validation_timeout,
            phase="payload validation",
        )
        after_sha256 = artifact_sha256(args.flm)
        if after_sha256 != before_sha256:
            raise PhaseError(
                f"reuse artifact changed during validation: {args.flm}; "
                f"before={before_sha256} after={after_sha256}"
            )
        return ArtifactPreparation(
            artifact=args.flm,
            action="reuse",
            source=args.flm,
            destination=args.flm,
            before_sha256=before_sha256,
            after_sha256=after_sha256,
        )

    destination = regeneration_destination(args)
    destination_exists = destination.exists()
    if destination_exists and not args.overwrite_artifact:
        raise PhaseError(
            f"regeneration destination already exists: {destination}; "
            "choose --regenerate-output or pass --overwrite-artifact"
        )
    before_sha256 = artifact_sha256(destination) if destination_exists else None
    partial = partial_artifact_path(destination)
    if partial.exists():
        raise PhaseError(f"export target already exists: {partial}")
    destination.parent.mkdir(parents=True, exist_ok=True)
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
    try:
        os.replace(partial, destination)
    except OSError as exc:
        raise PhaseError(f"artifact promotion failed: {exc}") from exc
    return ArtifactPreparation(
        artifact=destination,
        action="regenerate",
        source=args.hf_source,
        destination=destination,
        before_sha256=before_sha256,
        after_sha256=artifact_sha256(destination),
    )


def supersonic_benchmark_command(
    args: argparse.Namespace,
    artifact: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(BENCH_SCRIPT.relative_to(ROOT)),
        "--binary", str(args.binary),
        "--target-profile", "qwen36-35b-a3b-flm",
        "--model-dir", str(artifact),
        "--prompt-source", "jsonl",
        "--lucebox-jsonl", str(benchmark_prompt_path(args.out_json)),
        "--prompt-format", "raw",
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
    if args.flm_virtual_transfer_backend:
        command.extend(
            ["--flm-virtual-transfer-backend", args.flm_virtual_transfer_backend]
        )
    return command


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _canonical_direct_profile(
    profile: object,
) -> tuple[dict[object, int] | None, list[object]]:
    if not isinstance(profile, dict):
        return None, ["profile"]
    canonical: dict[object, int] = {}
    invalid_fields: list[object] = []
    for field, value in profile.items():
        parsed = _as_int(value)
        if parsed is None:
            invalid_fields.append(field)
        else:
            canonical[field] = parsed
    return (canonical if not invalid_fields else None), invalid_fields


def _direct_profile_field_label(field: object) -> str:
    return {
        "native_int4": "native INT4",
        "row_group_int4": "row-group INT4",
        "tile_int4_v1": "tile-v1 INT4",
        "bf16_fallback": "BF16 fallback",
    }.get(field, str(field))


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _transfer_pair(
    load_speed: dict,
    prefix: str,
) -> tuple[int | None, float | None]:
    return (
        _as_int(load_speed.get(f"{prefix}_bytes")),
        _as_float(load_speed.get(f"{prefix}_gib_s")),
    )


def _valid_transfer_pair(pair: tuple[int | None, float | None]) -> bool:
    byte_count, rate = pair
    return byte_count is not None and byte_count > 0 and rate is not None and rate > 0


def _transfer_errors(load_speed: dict, requested_backend: str | None) -> list[str]:
    prefixes = (
        ["copy_storage_to_device"]
        if requested_backend in STORAGE_DIRECT_BACKENDS
        else ["copy_h2d"]
        if requested_backend == "pageable-h2d"
        else ["copy_h2d", "copy_storage_to_device"]
    )
    pairs = [(prefix, _transfer_pair(load_speed, prefix)) for prefix in prefixes]
    if any(_valid_transfer_pair(pair) for _, pair in pairs):
        return []

    if requested_backend in STORAGE_DIRECT_BACKENDS:
        return ["summary has no positive finite storage-to-device transfer byte/rate evidence"]
    if requested_backend == "pageable-h2d":
        return ["summary has no positive finite pageable H2D transfer byte/rate evidence"]

    has_bytes = any(
        byte_count is not None and byte_count > 0
        for _, (byte_count, _) in pairs
    )
    has_rate = any(rate is not None and rate > 0 for _, (_, rate) in pairs)
    if not has_bytes:
        return ["summary has no transfer bytes"]
    if not has_rate:
        return ["summary has no transfer GiB/s"]
    return ["summary has no matching transfer byte/rate evidence"]


def first_class_errors(
    payload: dict,
    *,
    requested_backend: str | None = None,
) -> list[str]:
    if not isinstance(payload, dict):
        return ["report payload is not an object"]

    errors: list[str] = []
    if payload.get("resolved_model") != EXPECTED_RESOLVED_MODEL:
        errors.append("report resolved model is not qwen3.6-35b-a3b")
    if payload.get("flm_virtual_transfer_backend") != requested_backend:
        errors.append("report backend selector does not match the requested backend")

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("report has no benchmark rows")
        rows = []

    valid_direct_profiles: list[dict] = []
    ready_count = 0
    runtime_engine_ready_count = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {index} is not an object")
            continue
        return_code = _as_int(row.get("returncode"))
        if return_code is None or return_code != 0:
            errors.append(f"row {index} has invalid or nonzero return code")
        if row.get("flm_virtual_transfer_backend") != requested_backend:
            errors.append(f"row {index} backend selector does not match the request")
        if row.get("resolved_model") != EXPECTED_RESOLVED_MODEL:
            errors.append(f"row {index} resolved model is not qwen3.6-35b-a3b")
        if (_as_int(row.get("generated_tokens")) or 0) <= 0:
            errors.append(f"row {index} generated tokens is zero")
        if row.get("flm_weight_mode") != EXPECTED_FLM_WEIGHT_MODE:
            errors.append(f"row {index} FLM weight mode is not INT4 native FLM")
        if row.get("flm_ready_for_decode") is not True:
            errors.append(f"row {index} is not ready for decode")
        else:
            ready_count += 1
        ownership = row.get("runtime_engine_ownership_markers")
        if not isinstance(ownership, list) or len(ownership) != 1:
            errors.append(
                f"row {index} must contain exactly one runtime engine ownership marker"
            )
        else:
            runtime_engine_ready_count += 1
            marker = ownership[0]
            if not isinstance(marker, dict):
                errors.append(f"row {index} runtime engine ownership marker is not an object")
            else:
                if _as_int(marker.get("load_sequence")) != 1:
                    errors.append(
                        f"row {index} runtime engine ownership must report load_sequence=1"
                    )
                if _as_int(marker.get("source_open_count")) != 1:
                    errors.append(
                        f"row {index} runtime engine ownership must report source_open_count=1"
                    )

        direct_profile = row.get("flm_direct_profile")
        if not isinstance(direct_profile, dict):
            errors.append(f"row {index} has no FLM direct profile")
        else:
            canonical_profile, invalid_fields = _canonical_direct_profile(
                direct_profile
            )
            for field in invalid_fields:
                errors.append(
                    f"row {index} has invalid "
                    f"{_direct_profile_field_label(field)} direct profile field"
                )
            if canonical_profile is not None:
                native_int4 = canonical_profile.get("native_int4")
                row_group_int4 = canonical_profile.get("row_group_int4")
                tile_int4_v1 = canonical_profile.get("tile_int4_v1")
                if native_int4 != EXPECTED_ROW_GROUP_INT4_PROJECTIONS:
                    errors.append(
                        f"row {index} does not have exactly 330 native INT4 direct plans"
                    )
                if row_group_int4 != EXPECTED_ROW_GROUP_INT4_PROJECTIONS:
                    errors.append(
                        f"row {index} does not have exactly 330 row-group INT4 direct plans"
                    )
                if tile_int4_v1 != 0:
                    errors.append(f"row {index} has tile-v1 INT4 direct plans")
                if (
                    native_int4 is not None
                    and row_group_int4 is not None
                    and tile_int4_v1 is not None
                    and native_int4 != row_group_int4 + tile_int4_v1
                ):
                    errors.append(
                        f"row {index} aggregate native INT4 direct plans do not equal "
                        "row-group plus tile-v1 plans"
                    )
                bf16_fallback = canonical_profile.get("bf16_fallback")
                if bf16_fallback is None or bf16_fallback != 0:
                    errors.append(f"row {index} has BF16 fallback direct plans")
                if canonical_profile not in valid_direct_profiles:
                    valid_direct_profiles.append(canonical_profile)
        if row.get("benchmark_validation_errors"):
            errors.append(f"row {index} has benchmark validation errors")

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return errors + ["report summary is not an object"]

    summary_count = _as_int(summary.get("count")) or 0
    if summary_count <= 0:
        errors.append("summary count is zero")
    if rows and summary_count != len(rows):
        errors.append("summary count does not match benchmark rows")
    if summary.get("flm_weight_modes") != [EXPECTED_FLM_WEIGHT_MODE]:
        errors.append("summary FLM weight modes do not match row evidence")
    if (_as_int(summary.get("flm_ready_for_decode_count")) or 0) != ready_count:
        errors.append("summary ready for decode count does not match row evidence")
    if (
        _as_int(summary.get("runtime_engine_ready_count")) or 0
    ) != runtime_engine_ready_count:
        errors.append("summary runtime engine ownership count does not match row evidence")

    summary_direct_profiles = summary.get("flm_direct_profiles")
    canonical_summary_profiles: list[dict[object, int]] = []
    summary_profiles_valid = isinstance(summary_direct_profiles, list)
    if summary_profiles_valid:
        for profile in summary_direct_profiles:
            canonical_profile, invalid_fields = _canonical_direct_profile(profile)
            for field in invalid_fields:
                errors.append(
                    "summary has invalid "
                    f"{_direct_profile_field_label(field)} direct profile field"
                )
            if canonical_profile is not None:
                canonical_summary_profiles.append(canonical_profile)
            else:
                summary_profiles_valid = False
    if not summary_profiles_valid or canonical_summary_profiles != valid_direct_profiles:
        errors.append("summary FLM direct profiles do not match row evidence")

    load_speed = summary.get("flm_load_speed")
    if not isinstance(load_speed, dict):
        errors.append("summary has no FLM load speed")
        return errors
    errors.extend(_transfer_errors(load_speed, requested_backend))
    return errors


def validate_benchmark_report(
    path: Path,
    *,
    requested_backend: str | None = None,
) -> dict:
    try:
        report_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PhaseError(f"report evidence failed to read {path}: {exc}") from exc
    try:
        payload = json.loads(report_text)
    except json.JSONDecodeError as exc:
        raise PhaseError(f"report evidence contains invalid JSON: {exc}") from exc
    errors = first_class_errors(payload, requested_backend=requested_backend)
    if errors:
        raise PhaseError("report evidence failed: " + "; ".join(errors))
    return payload


def _path_default(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default)))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-source",
        type=Path,
        default=_path_default("SUPERSONIC_QWEN36_FLM_HF_SOURCE", DEFAULT_HF_SOURCE),
    )
    parser.add_argument(
        "--geoquant-root",
        type=Path,
        default=_path_default("SUPERSONIC_GEOQUANT_ROOT", DEFAULT_GEOQUANT_ROOT),
    )
    parser.add_argument(
        "--geoquant-python",
        type=Path,
        default=_path_default(
            "SUPERSONIC_GEOQUANT_PYTHON", DEFAULT_GEOQUANT_PYTHON
        ),
    )
    parser.add_argument(
        "--flm",
        type=Path,
        default=_path_default("SUPERSONIC_QWEN36_FLM", DEFAULT_FLM),
    )
    parser.add_argument("--quant-device", default="cuda")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--regenerate-output", type=Path)
    parser.add_argument("--overwrite-artifact", action="store_true")
    parser.add_argument("--export-timeout", type=_positive_int, default=3600)
    parser.add_argument("--validation-timeout", type=_positive_int, default=1800)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/supersonic")
    parser.add_argument("--limit", type=_positive_int, default=1)
    parser.add_argument("--n-gen", type=_positive_int, default=1)
    parser.add_argument("--context-size", type=_positive_int, default=512)
    parser.add_argument("--inference-timeout", type=_positive_int, default=900)
    parser.add_argument("--inference-cleanup-grace", type=_positive_int, default=30)
    parser.add_argument(
        "--flm-virtual-transfer-backend",
        choices=["pageable-h2d", "gpu-direct-storage", "gds", "hipfile"],
        default=None,
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("target/qwen36_35b_a3b_flm_first_class_e2e.json"),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def print_summary(payload: dict, artifact: Path) -> None:
    summary = payload["summary"]
    load_speed = summary["flm_load_speed"]
    prefixes = (
        ["copy_storage_to_device"]
        if payload.get("flm_virtual_transfer_backend") in STORAGE_DIRECT_BACKENDS
        else ["copy_h2d"]
        if payload.get("flm_virtual_transfer_backend") == "pageable-h2d"
        else ["copy_h2d", "copy_storage_to_device"]
    )
    transfer_gib_s = max(
        rate
        for prefix in prefixes
        for byte_count, rate in [_transfer_pair(load_speed, prefix)]
        if byte_count is not None and byte_count > 0 and rate is not None and rate > 0
    )
    print(
        "[flm-e2e] first-class evidence: "
        f"artifact={artifact} rows={summary['count']} "
        f"transfer_gib_s={transfer_gib_s:.2f}",
        flush=True,
    )


def record_artifact_provenance(
    path: Path,
    preparation: ArtifactPreparation,
) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PhaseError(f"artifact provenance failed to read report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PhaseError("artifact provenance report payload is not an object")
    payload["artifact_provenance"] = preparation.evidence()
    try:
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise PhaseError(f"artifact provenance failed to write report {path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    action = choose_artifact_action(args)
    discover_inputs(args, action)
    preparation = prepare_artifact(args)
    artifact = preparation.artifact
    write_benchmark_prompts(benchmark_prompt_path(args.out_json))
    run_command(
        supersonic_benchmark_command(args, artifact),
        cwd=ROOT,
        timeout=args.limit * args.inference_timeout + args.inference_cleanup_grace,
        phase="SuperSonic inference",
    )
    record_artifact_provenance(args.out_json, preparation)
    payload = validate_benchmark_report(
        args.out_json,
        requested_backend=args.flm_virtual_transfer_backend,
    )
    print_summary(payload, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
