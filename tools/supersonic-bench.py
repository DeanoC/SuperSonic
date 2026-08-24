#!/usr/bin/env python3
"""Command-line entry point for reproducible benchmark candidates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark import compare, validation  # noqa: E402
from tools.benchmark.execution import RunConfig, run_suite  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="supersonic-bench",
        description="Run and validate reproducible SuperSonic benchmark evidence.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="preflight and execute a benchmark suite")
    run.add_argument("--suite", choices=("quick", "full"), required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--artifact", type=Path, required=True)
    run.add_argument("--peer-artifact", type=Path)
    run.add_argument("--physical-gpu", required=True)
    run.add_argument("--gpu-identity", required=True)
    run.add_argument("--gpu-identity-verified", action="store_true")
    run.add_argument("--logical-gpu")
    run.add_argument("--gpu-arch", default=None)
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--context-size", type=int, default=32768)
    run.add_argument("--chat", action="store_true")
    run.add_argument("--clock-policy", choices=("locked", "uncontrolled-clocks"), default="uncontrolled-clocks")
    run.add_argument("--gpu-clock-mhz", type=int)
    run.add_argument("--memory-clock-mhz", type=int)
    run.add_argument("--power-cap-watts", type=int)
    run.add_argument("--performance-level")
    run.add_argument("--output", type=Path, default=Path("target/benchmarks/candidate"))
    run.add_argument("--seed", type=int)
    run.add_argument("--run-id")

    validate_parser = subparsers.add_parser("validate", help="validate candidate or committed records")
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument("--publishable", action="store_true")

    compare_parser = subparsers.add_parser("compare", help="compare two validated records or bundles")
    compare_parser.add_argument("left", type=Path)
    compare_parser.add_argument("right", type=Path)
    compare_parser.add_argument("--output", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return _run(args)
        if args.command == "validate":
            return _validate(args)
        if args.command == "compare":
            return _compare(args)
    except (OSError, ValueError, TypeError) as exc:
        print(f"supersonic-bench: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2


def _run(args: argparse.Namespace) -> int:
    gpu_arch = args.gpu_arch or _environment_arch()
    if not gpu_arch:
        raise ValueError("gpu_arch is required explicitly or through HIP_ARCH")
    import os

    selector_verified = (
        os.environ.get("SUPERSONIC_R9700_GPU_ID") == str(args.physical_gpu)
        and os.environ.get("SUPERSONIC_R9700_GPU_ARCH") == gpu_arch
    )
    gpu_identity_verified = bool(args.gpu_identity_verified or selector_verified)
    clock_policy: object = args.clock_policy
    if args.clock_policy == "locked":
        clock_policy = {
            "name": "locked",
            "gpu_clock_mhz": args.gpu_clock_mhz,
            "memory_clock_mhz": args.memory_clock_mhz,
            "power_cap_watts": args.power_cap_watts,
            "performance_level": args.performance_level,
        }
    config = RunConfig(
        suite=args.suite,
        model_dir=args.model_dir,
        artifact=args.artifact,
        peer_artifact=args.peer_artifact,
        physical_gpu=args.physical_gpu,
        gpu_arch=gpu_arch,
        gpu_identity=args.gpu_identity,
        gpu_identity_verified=gpu_identity_verified,
        gpu_identity_evidence=(
            {
                "source": "select-r9700-device",
                "physical_gpu": str(args.physical_gpu),
                "architecture": gpu_arch,
            }
            if gpu_identity_verified
            else None
        ),
        logical_gpu=args.logical_gpu or os.environ.get("SUPERSONIC_DEVICE", str(args.device)),
        output_dir=args.output,
        device=args.device,
        context_size=args.context_size,
        chat=args.chat,
        clock_policy=clock_policy,
        seed=args.seed,
        run_id=args.run_id,
        run_quality=True,
    )
    status = run_suite(config)
    payload = {
        "state": status.state,
        "bundle": str(status.bundle),
        "records": [str(path) for path in status.records],
        "errors": list(status.errors),
        "quality_failed": status.quality_failed,
        "performance_report_only": status.performance_report_only,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if status.state == "complete" else 1


def _validate(args: argparse.Namespace) -> int:
    paths = validation.validate_bundle(args.path, require_complete=args.publishable)
    print(json.dumps({"valid": True, "publishable": bool(args.publishable), "records": [str(path) for path in paths]}, sort_keys=True))
    return 0


def _compare(args: argparse.Namespace) -> int:
    if args.left.is_dir():
        validation.validate_bundle(args.left, require_complete=False)
    if args.right.is_dir():
        validation.validate_bundle(args.right, require_complete=False)
    left = _first_record(args.left)
    right = _first_record(args.right)
    validation.validate_record(left)
    validation.validate_record(right)
    result = compare.compare_records(left, right)
    payload = {
        "comparable": result.comparable,
        "reasons": list(result.reasons),
        "speedup": result.speedup,
        "left": asdict(result.left),
        "right": asdict(result.right),
    }
    encoded = json.dumps(payload, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(encoded + "\n", encoding="utf-8")
        temporary.replace(args.output)
        print(encoded)
    # A non-comparable pair is a valid, useful report with explicit reasons;
    # malformed records are the validation error that should fail the command.
    return 0


def _first_record(path: Path) -> dict[str, object]:
    if path.is_file():
        return _load_json(path)
    paths = sorted(candidate for candidate in path.rglob("*.json") if candidate.is_file())
    if not paths:
        raise ValueError(f"benchmark path contains no JSON records: {path}")
    # A bundle may include a comparison output or manifest in the future; only
    # result records have the required run/engine sections.
    for candidate in paths:
        value = _load_json(candidate)
        if isinstance(value, dict) and "run" in value and "engine" in value:
            return value
    raise ValueError(f"benchmark path contains no result records: {path}")


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} must contain valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _environment_arch() -> str | None:
    import os

    value = os.environ.get("HIP_ARCH")
    return value.strip() if value else None


if __name__ == "__main__":
    raise SystemExit(main())
