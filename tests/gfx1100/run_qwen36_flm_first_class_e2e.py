#!/usr/bin/env python3
"""Prepare a validated native-int4 Qwen3.6 FLM artifact for the E2E lane."""

import argparse
import json
import os
import subprocess
import sys
from enum import Enum, auto
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH_SCRIPT = ROOT / "tests" / "gfx1100" / "bench_qwen36_he_supersonic.py"
STRICT_PROFILE = "supersonic-qwen36-moe-native-int4"
EXPECTED_RESOLVED_MODEL = "qwen3.6-35b-a3b"
EXPECTED_FLM_WEIGHT_MODE = "INT4 native FLM"
DEFAULT_HF_SOURCE = Path("/mnt/data/models/Qwen3.6-35B-A3B")
DEFAULT_GEOQUANT_ROOT = Path("/home/deano/projects/geo-quant")
DEFAULT_GEOQUANT_PYTHON = Path(
    "/home/deano/projects/geo-quant/.venv-rocm/bin/python"
)
DEFAULT_FLM = Path(
    "/mnt/data/runs/geo-quant/"
    "qwen36-35b-a3b-supersonic-native-int4-current.flm"
)


class ArtifactAction(Enum):
    REUSE = auto()
    REGENERATE = auto()


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
        "128",
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


def choose_artifact_action(args: argparse.Namespace) -> ArtifactAction:
    if args.regenerate or not args.flm.exists():
        return ArtifactAction.REGENERATE
    return ArtifactAction.REUSE


def partial_artifact_path(artifact: Path) -> Path:
    return artifact.with_name(f".{artifact.name}.partial-{os.getpid()}")


def prepare_artifact(args: argparse.Namespace) -> Path:
    action = choose_artifact_action(args)
    if action is ArtifactAction.REUSE and probe_validation(args, args.flm):
        run_command(
            validate_command(args, args.flm, verify_payload_hashes=True),
            cwd=args.geoquant_root,
            timeout=args.validation_timeout,
            phase="payload validation",
        )
        return args.flm
    if action is ArtifactAction.REUSE:
        print(
            f"[flm-e2e] existing artifact is stale or incompatible: {args.flm}; "
            "regenerating",
            flush=True,
        )

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


def _as_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def first_class_errors(payload: dict) -> list[str]:
    if not isinstance(payload, dict):
        return ["report payload is not an object"]

    errors: list[str] = []
    if payload.get("resolved_model") != EXPECTED_RESOLVED_MODEL:
        errors.append("report resolved model is not qwen3.6-35b-a3b")

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("report has no benchmark rows")
        rows = []

    valid_direct_profiles: list[dict] = []
    ready_count = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"row {index} is not an object")
            continue
        if row.get("returncode") != 0:
            errors.append(f"row {index} has nonzero return code")
        if row.get("resolved_model") != EXPECTED_RESOLVED_MODEL:
            errors.append(f"row {index} resolved model is not qwen3.6-35b-a3b")
        if _as_int(row.get("generated_tokens")) <= 0:
            errors.append(f"row {index} generated tokens is zero")
        if row.get("flm_weight_mode") != EXPECTED_FLM_WEIGHT_MODE:
            errors.append(f"row {index} FLM weight mode is not INT4 native FLM")
        if row.get("flm_ready_for_decode") is not True:
            errors.append(f"row {index} is not ready for decode")
        else:
            ready_count += 1

        direct_profile = row.get("flm_direct_profile")
        if not isinstance(direct_profile, dict):
            errors.append(f"row {index} has no FLM direct profile")
        else:
            if _as_int(direct_profile.get("native_int4")) <= 0:
                errors.append(f"row {index} has no native INT4 direct plans")
            if _as_int(direct_profile.get("bf16_fallback")) != 0:
                errors.append(f"row {index} has BF16 fallback direct plans")
            if direct_profile not in valid_direct_profiles:
                valid_direct_profiles.append(direct_profile)
        if row.get("benchmark_validation_errors"):
            errors.append(f"row {index} has benchmark validation errors")

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return errors + ["report summary is not an object"]

    summary_count = _as_int(summary.get("count"))
    if summary_count <= 0:
        errors.append("summary count is zero")
    if rows and summary_count != len(rows):
        errors.append("summary count does not match benchmark rows")
    if summary.get("flm_weight_modes") != [EXPECTED_FLM_WEIGHT_MODE]:
        errors.append("summary FLM weight modes do not match row evidence")
    if _as_int(summary.get("flm_ready_for_decode_count")) != ready_count:
        errors.append("summary ready for decode count does not match row evidence")

    summary_direct_profiles = summary.get("flm_direct_profiles")
    if (
        not isinstance(summary_direct_profiles, list)
        or any(profile not in summary_direct_profiles for profile in valid_direct_profiles)
    ):
        errors.append("summary FLM direct profiles do not match row evidence")

    load_speed = summary.get("flm_load_speed")
    if not isinstance(load_speed, dict):
        errors.append("summary has no FLM load speed")
        return errors
    transfer_bytes = max(
        _as_int(load_speed.get("copy_h2d_bytes")),
        _as_int(load_speed.get("copy_storage_to_device_bytes")),
    )
    transfer_gib_s = max(
        _as_float(load_speed.get("copy_h2d_gib_s")),
        _as_float(load_speed.get("copy_storage_to_device_gib_s")),
    )
    if transfer_bytes <= 0:
        errors.append("summary has no transfer bytes")
    if transfer_gib_s <= 0:
        errors.append("summary has no transfer GiB/s")
    return errors


def validate_benchmark_report(path: Path) -> dict:
    payload = json.loads(path.read_text())
    errors = first_class_errors(payload)
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
    parser.add_argument("--export-timeout", type=_positive_int, default=3600)
    parser.add_argument("--validation-timeout", type=_positive_int, default=1800)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/supersonic")
    parser.add_argument("--limit", type=_positive_int, default=1)
    parser.add_argument("--n-gen", type=_positive_int, default=1)
    parser.add_argument("--context-size", type=_positive_int, default=512)
    parser.add_argument("--inference-timeout", type=_positive_int, default=900)
    parser.add_argument(
        "--flm-virtual-transfer-backend",
        choices=["pageable-h2d", "gpu-direct-storage", "gds", "hipfile"],
        default=None,
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("target/qwen36_35b_a3b_flm_he_supersonic.json"),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def print_summary(payload: dict, artifact: Path) -> None:
    summary = payload["summary"]
    load_speed = summary["flm_load_speed"]
    transfer_gib_s = max(
        _as_float(load_speed.get("copy_h2d_gib_s")),
        _as_float(load_speed.get("copy_storage_to_device_gib_s")),
    )
    print(
        "[flm-e2e] first-class evidence: "
        f"artifact={artifact} rows={summary['count']} "
        f"transfer_gib_s={transfer_gib_s:.2f}",
        flush=True,
    )


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


if __name__ == "__main__":
    raise SystemExit(main())
