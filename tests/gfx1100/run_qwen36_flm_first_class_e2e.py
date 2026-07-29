#!/usr/bin/env python3
"""Prepare a validated native-int4 Qwen3.6 FLM artifact for the E2E lane."""

import argparse
import os
import subprocess
from enum import Enum, auto
from pathlib import Path


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


def _path_default(env_name: str, default: Path) -> Path:
    return Path(os.environ.get(env_name, str(default)))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifact = prepare_artifact(args)
    print(f"[flm-e2e] ready: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
