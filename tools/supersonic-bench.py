#!/usr/bin/env python3
"""Command-line entry point for reproducible benchmark candidates."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.benchmark import adapters, compare, manifest, qualification, render, repeatability, validation  # noqa: E402
from tools.benchmark.execution import RunConfig, run_suite  # noqa: E402
from tools.benchmark.model import PerformanceCase  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="supersonic-bench",
        description="Run and validate reproducible SuperSonic benchmark evidence.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="preflight and execute a benchmark suite")
    run.add_argument(
        "--suite",
        choices=("quick", "full", "full-scalar-qualification", "scalar-baseline"),
        required=True,
    )
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--artifact", type=Path, required=True)
    run.add_argument("--peer-artifact", type=Path)
    run.add_argument("--artifact-semantic-id", required=True)
    run.add_argument("--artifact-quantization", required=True)
    run.add_argument("--artifact-source-repository", required=True)
    run.add_argument("--artifact-source-revision", required=True)
    run.add_argument("--artifact-filename", required=True)
    run.add_argument("--artifact-size-bytes", type=int, required=True)
    run.add_argument("--peer-artifact-source-repository")
    run.add_argument("--peer-artifact-source-revision")
    run.add_argument("--peer-artifact-filename")
    run.add_argument("--peer-artifact-size-bytes", type=int)
    run.add_argument("--tokenizer-sha256", required=True)
    run.add_argument("--chat-template-sha256", required=True)
    run.add_argument("--physical-gpu", required=True)
    run.add_argument(
        "--gpu-static-json",
        type=Path,
        required=True,
        help="captured authoritative AMD SMI static JSON used for GPU provenance",
    )
    run.add_argument(
        "--rocm-version-file",
        type=Path,
        required=True,
        help="captured bounded ROCm/driver version text used for portable identity",
    )
    run.add_argument(
        "--hip-version-file",
        type=Path,
        required=True,
        help="captured bounded HIP compiler version text used for portable identity",
    )
    run.add_argument("--logical-gpu")
    run.add_argument("--gpu-arch", default=None)
    run.add_argument("--device", type=int, default=0)
    run.add_argument("--context-size", type=int, default=32768)
    run.add_argument("--chat", action="store_true")
    run.add_argument("--clock-policy", choices=("locked", "uncontrolled-clocks"), default="uncontrolled-clocks")
    run.add_argument("--gpu-clock-mhz", type=int)
    run.add_argument("--gpu-clock-tolerance-mhz", type=int)
    run.add_argument("--memory-clock-mhz", type=int)
    run.add_argument("--power-cap-watts", type=int)
    run.add_argument("--performance-level")
    run.add_argument("--temperature-limit-celsius", type=float)
    run.add_argument("--output", type=Path, default=Path("target/benchmarks/candidate"))
    run.add_argument("--seed", type=int)
    run.add_argument("--run-id")

    validate_parser = subparsers.add_parser("validate", help="validate candidate or committed records")
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument("--publishable", action="store_true")
    validate_parser.add_argument("--baseline-bundle", type=Path)
    validate_parser.add_argument("--baseline-bundle-sha256")

    compare_parser = subparsers.add_parser("compare", help="compare two validated records or bundles")
    compare_parser.add_argument("left", type=Path)
    compare_parser.add_argument("right", type=Path)
    compare_parser.add_argument("--output", type=Path)

    render_parser = subparsers.add_parser("render", help="render publishable benchmark records as static HTML")
    render_parser.add_argument(
        "results_root",
        nargs="?",
        type=Path,
        default=Path("benchmarks/results"),
        help="directory or record file containing publishable results",
    )

    series = subparsers.add_parser("scalar-series", help="derive a locked seven-sample scalar series")
    series.add_argument("--record", type=Path, required=True)
    series.add_argument("--compiler-version-file", type=Path, required=True)
    series.add_argument("--scalar-instruction-sha256", required=True)
    series.add_argument("--output", type=Path, required=True)

    qualify = subparsers.add_parser("qualify", help="apply the immutable scalar baseline gate")
    qualify.add_argument("--baseline-bundle", type=Path, required=True)
    qualify.add_argument("--baseline-bundle-sha256", required=True)
    qualify.add_argument("--candidate-series", type=Path, required=True)
    qualify.add_argument("--output", type=Path, required=True)
    render_parser.add_argument(
        "output_root",
        nargs="?",
        type=Path,
        default=Path("target/benchmarks/site"),
        help="disposable output directory for generated HTML",
    )
    render_parser.add_argument("--results-root", "--results", dest="results_option", type=Path)
    render_parser.add_argument(
        "--output-root",
        "--output",
        dest="output_option",
        type=Path,
    )

    soak = subparsers.add_parser(
        "repeatability",
        help="run a bounded fresh-process soak and trace follow-ups after the first slow sample",
    )
    soak.add_argument("--model-dir", type=Path, required=True)
    soak.add_argument("--artifact", type=Path, required=True)
    soak.add_argument("--physical-gpu", required=True)
    soak.add_argument("--output", type=Path, required=True)
    soak.add_argument("--binary", default="./target/release/supersonic")
    soak.add_argument("--rocprof-binary", default="rocprofv3")
    soak.add_argument("--max-runs", type=int, default=2160)
    soak.add_argument("--max-duration-seconds", type=float, default=21600.0)
    soak.add_argument("--trace-attempts", type=int, default=3)
    soak.add_argument("--slow-persistent-ms-per-token", type=float, default=55.0)
    soak.add_argument("--timeout-seconds", type=float, default=60.0)
    soak.add_argument("--device", type=int, default=0)

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
        if args.command == "render":
            return _render(args)
        if args.command == "scalar-series":
            return _scalar_series(args)
        if args.command == "qualify":
            return _qualify(args)
        if args.command == "repeatability":
            return _repeatability(args)
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

    clock_policy: object = args.clock_policy
    if args.clock_policy == "locked":
        clock_policy = {
            "name": "locked",
            "gpu_clock_mhz": args.gpu_clock_mhz,
            "clock_tolerance_mhz": args.gpu_clock_tolerance_mhz,
            "memory_clock_mhz": args.memory_clock_mhz,
            "power_cap_watts": args.power_cap_watts,
            "performance_level": args.performance_level,
            "temperature_limit_celsius": args.temperature_limit_celsius,
        }
    config = RunConfig(
        suite=args.suite,
        model_dir=args.model_dir,
        artifact=args.artifact,
        peer_artifact=args.peer_artifact,
        physical_gpu=args.physical_gpu,
        gpu_arch=gpu_arch,
        gpu_static_json=args.gpu_static_json,
        rocm_version_file=args.rocm_version_file,
        hip_version_file=args.hip_version_file,
        logical_gpu=args.logical_gpu or os.environ.get("SUPERSONIC_DEVICE", str(args.device)),
        output_dir=args.output,
        device=args.device,
        context_size=args.context_size,
        chat=args.chat,
        clock_policy=clock_policy,
        seed=args.seed,
        run_id=args.run_id,
        run_quality=True,
        artifact_semantic_id=args.artifact_semantic_id,
        artifact_quantization=args.artifact_quantization,
        artifact_source_repository=args.artifact_source_repository,
        artifact_source_revision=args.artifact_source_revision,
        artifact_filename=args.artifact_filename,
        artifact_size_bytes=args.artifact_size_bytes,
        peer_artifact_source_repository=args.peer_artifact_source_repository,
        peer_artifact_source_revision=args.peer_artifact_source_revision,
        peer_artifact_filename=args.peer_artifact_filename,
        peer_artifact_size_bytes=args.peer_artifact_size_bytes,
        tokenizer_sha256=args.tokenizer_sha256,
        chat_template_sha256=args.chat_template_sha256,
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
    suites = {
        str(validation._load_json(path)["run"]["suite"])
        for path in paths
    }
    if args.publishable and "full-scalar-qualification" in suites:
        if args.baseline_bundle is None or args.baseline_bundle_sha256 is None:
            raise ValueError(
                "publishable scalar qualification validation requires --baseline-bundle and "
                "--baseline-bundle-sha256"
            )
        root = args.path if args.path.is_dir() else args.path.parent
        observed_digest = qualification.directory_digest(args.baseline_bundle)
        if observed_digest != args.baseline_bundle_sha256:
            raise ValueError(
                f"baseline bundle digest mismatch: observed {observed_digest}, "
                f"expected {args.baseline_bundle_sha256}"
            )
        baseline = qualification.load_series(args.baseline_bundle)
        candidate = qualification.load_series(root / "candidate-scalar-v1.json")
        expected = qualification.qualify_series(
            baseline,
            candidate,
            baseline_bundle_sha256=observed_digest,
        )
        stored = validation._load_json(root / "qualification-v1.json")
        if stored != expected:
            raise ValueError("qualification-v1.json does not match the pinned baseline and candidate")
        if expected["qualified"] is not True:
            raise ValueError("scalar candidate failed the immutable baseline gate")
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


def _render(args: argparse.Namespace) -> int:
    results_root = args.results_option or args.results_root
    output_root = args.output_option or args.output_root
    files = render.render_site(results_root, output_root)
    payload = {
        "files": [str(path) for path in files],
        "output_root": str(output_root),
        "results_root": str(results_root),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def _scalar_series(args: argparse.Namespace) -> int:
    record = qualification.validation._load_json(args.record)
    compiler_version = args.compiler_version_file.read_text(encoding="utf-8").strip()
    if not compiler_version or len(compiler_version) > 4096:
        raise ValueError("compiler version capture must be non-empty and bounded")
    value = qualification.series_from_record(
        record,
        compiler_version=compiler_version,
        scalar_instruction_sha256=args.scalar_instruction_sha256,
    )
    _write_json(args.output, value)
    print(json.dumps({"series": str(args.output)}, sort_keys=True))
    return 0


def _qualify(args: argparse.Namespace) -> int:
    observed_digest = qualification.directory_digest(args.baseline_bundle)
    if observed_digest != args.baseline_bundle_sha256:
        raise ValueError(
            f"baseline bundle digest mismatch: observed {observed_digest}, expected {args.baseline_bundle_sha256}"
        )
    baseline = qualification.load_series(args.baseline_bundle)
    candidate = qualification.load_series(args.candidate_series)
    result = qualification.qualify_series(
        baseline,
        candidate,
        baseline_bundle_sha256=observed_digest,
    )
    _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["qualified"] else 1


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)


def _repeatability(args: argparse.Namespace) -> int:
    result = repeatability.run_soak(_repeatability_config(args))
    print(
        json.dumps(
            {
                "state": result["state"],
                "trigger_run": result["trigger_run"],
                "samples": len(result["samples"]),
                "followup_traces": len(result["followup_traces"]),
            },
            sort_keys=True,
        )
    )
    return 0 if result["state"] in {"slow-captured", "no-slow-sample", "duration-complete"} else 1


def _repeatability_config(args: argparse.Namespace) -> repeatability.SoakConfig:
    from dataclasses import replace

    engine = replace(manifest.load_engine("supersonic"), binary=args.binary)
    case = PerformanceCase(
        id="repeatability-short-cold-ordinary",
        prompt="Emit a single sentence describing cold-load benchmark startup.",
        max_new_tokens=32,
        warmups=0,
        repetitions=1,
        mode="ordinary",
        cache_state="cold-load",
        timeout_seconds=int(args.timeout_seconds),
        decoding_policy="greedy",
        engines=("supersonic",),
    )
    argv = adapters.build_command(
        engine,
        case,
        adapters.AdapterInputs(
            model_dir=args.model_dir,
            artifact=args.artifact,
            chat=True,
            device=args.device,
            context_size=32768,
            sampling_seed=1,
        ),
    )
    return repeatability.SoakConfig(
        argv=argv,
        output=args.output,
        physical_gpu=args.physical_gpu,
        slow_persistent_ms_per_token=args.slow_persistent_ms_per_token,
        max_runs=args.max_runs,
        trace_attempts=args.trace_attempts,
        timeout_seconds=args.timeout_seconds,
        max_duration_seconds=args.max_duration_seconds,
        rocprof_binary=args.rocprof_binary,
        logical_gpu=args.device,
        hip_visible_devices=os.environ.get("HIP_VISIBLE_DEVICES", ""),
        environment=dict(os.environ),
    )


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
