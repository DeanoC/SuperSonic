#!/usr/bin/env python3
"""Create and validate a safe Qwen3.8 gfx1201 reproducibility record.

The record deliberately stores artifact names and hashes, never configured
absolute paths.  It combines the deterministic ordinary/MTP token comparison
with the warmup and measured telemetry logs emitted by the runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any


HASH_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")
RESULT_RE = re.compile(
    r"\[result\]\s+prompt_tokens=(?P<prompt>[0-9]+)\s+"
    r"generated_tokens=(?P<generated>[0-9]+)\s+"
    r"decode_ms=(?P<decode>[0-9]+(?:\.[0-9]+)?)\s*ms\s+"
    r"ms_per_tok=(?P<per_tok>[0-9]+(?:\.[0-9]+)?)"
)
PREFILL_RE = re.compile(
    r"\[prefill\].*?done in\s+(?P<prefill>[0-9]+(?:\.[0-9]+)?)\s*ms",
    re.IGNORECASE,
)
HIP_VERSION_RE = re.compile(r"\bHIP\s+version\s*:\s*(?P<version>[0-9][0-9A-Za-z.+-]*)", re.I)
ROCM_VERSION_RE = re.compile(
    r"\b(?:ROCm(?:\s+version)?|driver(?:\s+version)?)\s*[:=]?\s*"
    r"(?P<version>[0-9]+(?:\.[0-9]+)+)",
    re.I,
)
GFX_RE = re.compile(r"\bgfx[0-9]+\b", re.I)
ABSOLUTE_PATH_RE = re.compile(r"(?<![\w])/(?:[^\s/]+/)*[^\s/]+")


def _safe_name(path: Path) -> str:
    """Return one basename without allowing a path into the record."""

    name = path.name
    return name if name not in {"", ".", ".."} else "unknown"


def _safe_text(value: str, *, limit: int = 256) -> str:
    """Keep command output single-line and bounded before writing JSON."""

    compact = " ".join(value.split())
    return ABSOLUTE_PATH_RE.sub("<path>", compact)[:limit]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_sequence_hash(tokens: list[int]) -> str:
    """Hash a token sequence using a canonical JSON representation."""

    encoded = json.dumps(tokens, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _token_line(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.startswith("[tokens] ")]
    if len(lines) != 1:
        raise ValueError(f"{_safe_name(path)} must contain exactly one [tokens] line")
    values = lines[0].removeprefix("[tokens] ").split()
    try:
        tokens = [int(value) for value in values]
    except ValueError as exc:
        raise ValueError(f"{_safe_name(path)} contains a non-integer token") from exc
    if any(token < 0 for token in tokens):
        raise ValueError(f"{_safe_name(path)} contains a negative token")
    return tokens


def _log_measurement(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    result = RESULT_RE.findall(text)
    if len(result) != 1:
        raise ValueError(f"{_safe_name(path)} must contain exactly one [result] line")
    match = RESULT_RE.search(text)
    assert match is not None
    prefill = PREFILL_RE.findall(text)
    if len(prefill) != 1:
        raise ValueError(f"{_safe_name(path)} must contain exactly one prefill timing")
    return {
        "prompt_tokens": int(match.group("prompt")),
        "generated_tokens": int(match.group("generated")),
        "prefill_ms": float(prefill[0]),
        "decode_ms": float(match.group("decode")),
        "ms_per_tok": float(match.group("per_tok")),
    }


def compare_token_logs(ordinary_log: Path, mtp_log: Path) -> dict[str, Any]:
    """Return the exact ordinary-versus-MTP comparison for two logs."""

    ordinary = _token_line(ordinary_log)
    mtp = _token_line(mtp_log)
    ordinary_hash = token_sequence_hash(ordinary)
    mtp_hash = token_sequence_hash(mtp)
    equal = ordinary == mtp
    result = {
        "ordinary": ordinary,
        "mtp": mtp,
        "ordinary_token_count": len(ordinary),
        "mtp_token_count": len(mtp),
        "ordinary_hash": ordinary_hash,
        "mtp_hash": mtp_hash,
        "correctness_hash": ordinary_hash if equal else None,
        "equal": equal,
    }
    if not equal:
        raise ValueError(f"ordinary/MTP token mismatch: {ordinary} != {mtp}")
    return result


def _numbered_logs(root: Path, prefix: str) -> list[Path]:
    logs = list(root.glob(f"{prefix}-*.log"))

    def key(path: Path) -> tuple[int, str]:
        match = re.search(r"-(\d+)\.log$", path.name)
        return (int(match.group(1)) if match else 2**31, path.name)

    return sorted(logs, key=key)


def _gpu_name(gpu_json: Path, physical_gpu: str) -> str:
    """Find the selected market name without copying the SMI payload."""

    payload = json.loads(gpu_json.read_text(encoding="utf-8"))
    wanted = int(physical_gpu)
    market_keys = ("market_name", "product_name", "device_name", "name")

    def direct_index(node: dict[str, Any]) -> int | None:
        for key in ("gpu", "GPU"):
            value = node.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
        return None

    def visit(node: Any, inherited: int | None = None) -> str | None:
        if isinstance(node, dict):
            current = direct_index(node)
            if current is None:
                current = inherited
            if current == wanted:
                for key in market_keys:
                    value = node.get(key)
                    if isinstance(value, str) and value.strip():
                        return _safe_text(value)
            for child in node.values():
                found = visit(child, current)
                if found:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = visit(child, inherited)
                if found:
                    return found
        return None

    return visit(payload) or "unknown"


def _hip_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = HIP_VERSION_RE.search(text)
    if match:
        return match.group("version")
    # Keep an unrecognized toolchain marker bounded and path-free rather than
    # silently pretending that the version was known.
    first = next((line for line in text.splitlines() if line.strip()), "unknown")
    return _safe_text(first)


def _rocm_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = ROCM_VERSION_RE.search(text)
    if match:
        return match.group("version")
    for line in text.splitlines():
        if line.strip():
            return _safe_text(line)
    return "unknown"


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"artifact {_safe_name(path)} is unavailable")
    name = _safe_name(path)
    digest = _sha256(path)
    return {
        "identity": name,
        "name": name,
        "sha256": digest,
        "digest": f"sha256:{digest}",
        "size_bytes": path.stat().st_size,
    }


def _model_directory(path: Path, *, chat: bool) -> dict[str, Any]:
    if not path.is_dir():
        raise ValueError(f"model directory {_safe_name(path)} is unavailable")
    required = {}
    required_names = ["config.json", "tokenizer.json"]
    if chat:
        required_names.append("tokenizer_config.json")
    for name in required_names:
        file_path = path / name
        if not file_path.is_file():
            raise ValueError(f"model sidecar {_safe_name(file_path)} is unavailable")
        required[name] = _sha256(file_path)
    return {"name": _safe_name(path), "required_files": required}


def _timing_record(path: Path, run: int) -> dict[str, Any]:
    value = _log_measurement(path)
    value["run"] = run
    return value


def _timings(root: Path) -> dict[str, Any]:
    warmup_paths = _numbered_logs(root, "warmup")
    measured_paths = _numbered_logs(root, "run")
    errors: list[str] = []

    def parse(paths: list[Path]) -> list[dict[str, Any]]:
        values: list[dict[str, Any]] = []
        for index, path in enumerate(paths, start=1):
            try:
                values.append(_timing_record(path, index))
            except (OSError, ValueError) as exc:
                values.append({"run": index, "error": _safe_text(str(exc))})
                errors.append(f"{_safe_name(path)}: {_safe_text(str(exc))}")
        return values

    warmup = parse(warmup_paths)
    measured = parse(measured_paths)
    per_tok = [item["ms_per_tok"] for item in measured if "ms_per_tok" in item]
    median = statistics.median(per_tok) if per_tok else None
    return {
        "warmup_runs": len(warmup),
        "measured_runs": len(measured),
        "warmup": warmup,
        "measured": measured,
        "median_ms_per_tok": median,
        "median_tok_per_s": (1000.0 / median if median and median > 0 else None),
        "status": "complete" if not errors else "partial",
        "errors": errors,
    }


def _comparison_for_record(
    ordinary_log: Path | None, mtp_log: Path | None
) -> dict[str, Any]:
    if ordinary_log is None or mtp_log is None:
        return {"applicable": False, "equal": None}
    if not ordinary_log.is_file() or not mtp_log.is_file():
        return {"applicable": False, "equal": None}
    try:
        result = compare_token_logs(ordinary_log, mtp_log)
    except (OSError, ValueError) as exc:
        return {"applicable": True, "equal": False, "error": _safe_text(str(exc))}
    return {
        "applicable": True,
        "equal": True,
        "ordinary_token_count": result["ordinary_token_count"],
        "mtp_token_count": result["mtp_token_count"],
        "ordinary_hash": result["ordinary_hash"],
        "mtp_hash": result["mtp_hash"],
    }


def build_record(
    *,
    commit: str,
    hip_version_file: Path,
    rocm_version: str,
    rocm_version_file: Path | None = None,
    gpu_json: Path,
    physical_gpu: str,
    gpu_arch: str,
    artifact: Path,
    model_dir: Path,
    ordinary_log: Path | None,
    mtp_log: Path | None,
    telemetry_root: Path,
    prompt: str,
    chat: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    if not COMMIT_RE.fullmatch(commit):
        raise ValueError("commit must be a hexadecimal revision")
    if not physical_gpu.isdigit():
        raise ValueError("physical GPU must be a numeric ordinal")
    architecture = GFX_RE.search(gpu_arch)
    if architecture is None:
        raise ValueError("GPU architecture must be a gfx target")

    comparison = _comparison_for_record(ordinary_log, mtp_log)
    if comparison.get("applicable") is not True or comparison.get("equal") is not True:
        raise ValueError("ordinary/MTP correctness logs are required and must be equal")
    correctness_hash = (
        comparison.get("ordinary_hash") if comparison.get("equal") is True else None
    )
    timings = _timings(telemetry_root) if telemetry_root.is_dir() else {
        "warmup_runs": 0,
        "measured_runs": 0,
        "warmup": [],
        "measured": [],
        "median_ms_per_tok": None,
        "median_tok_per_s": None,
        "status": "unavailable",
        "errors": [],
    }
    measured_tokens = [
        item["generated_tokens"] for item in timings["measured"] if "generated_tokens" in item
    ]
    if measured_tokens and len(set(measured_tokens)) != 1:
        raise ValueError("measured runs disagree on generated token count")
    if not timings["measured"] or not measured_tokens or timings["median_ms_per_tok"] is None:
        raise ValueError("at least one measured telemetry run is required")
    token_count = measured_tokens[0] if measured_tokens else comparison.get("ordinary_token_count")

    hip_version = _hip_version(hip_version_file)
    rocm_identity = (
        _rocm_version(rocm_version_file)
        if rocm_version_file is not None
        else _safe_text(rocm_version)
    )
    record = {
        "schema_version": 1,
        "commit": commit,
        "toolchain": {
            "rocm": rocm_identity,
            "rocm_version": rocm_identity,
            "hip": hip_version,
            "hip_version": hip_version,
        },
        "target_architecture": architecture.group(0).lower(),
        "physical_gpu": {
            "id": physical_gpu,
            "architecture": architecture.group(0).lower(),
            "name": _gpu_name(gpu_json, physical_gpu),
            "logical_device": "0",
        },
        "artifact": _artifact(artifact),
        "model_directory": _model_directory(model_dir, chat=chat),
        "workload": {
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "chat": chat,
            "max_new_tokens": max_new_tokens,
            "token_count": token_count,
        },
        "generated_token_count": token_count,
        "correctness": {
            "correctness_hash": correctness_hash,
            "ordinary_vs_mtp": comparison,
        },
        "timings": timings,
    }
    validate_record(record)
    return record


def _contains_unsafe_path(value: Any, key: str = "") -> bool:
    if isinstance(value, dict):
        return any(_contains_unsafe_path(child, str(child_key)) for child_key, child in value.items())
    if isinstance(value, list):
        return any(_contains_unsafe_path(child, key) for child in value)
    if isinstance(value, str):
        return ("path" in key.lower() and value.startswith(("/", "~"))) or "\\" in value
    return False


def validate_record(record: dict[str, Any]) -> None:
    """Reject incomplete records and any accidental absolute path leakage."""

    if _contains_unsafe_path(record):
        raise ValueError("reproducibility record contains an unsafe path")
    for key in (
        "schema_version",
        "commit",
        "toolchain",
        "target_architecture",
        "physical_gpu",
        "artifact",
        "model_directory",
        "workload",
        "correctness",
        "timings",
    ):
        if key not in record:
            raise ValueError(f"reproducibility record is missing {key}")
    if record["schema_version"] != 1 or not isinstance(record["commit"], str):
        raise ValueError("unsupported reproducibility record schema or commit")
    if not COMMIT_RE.fullmatch(record["commit"]):
        raise ValueError("reproducibility record has an invalid commit")
    toolchain = record["toolchain"]
    if not isinstance(toolchain, dict) or not toolchain.get("hip_version") or not toolchain.get(
        "rocm_version"
    ):
        raise ValueError("reproducibility record has incomplete toolchain identity")
    gpu = record["physical_gpu"]
    if not isinstance(gpu, dict) or not gpu.get("id") or not gpu.get("architecture"):
        raise ValueError("reproducibility record has incomplete physical GPU identity")
    artifact = record["artifact"]
    if (
        not isinstance(artifact, dict)
        or not artifact.get("name")
        or "/" in str(artifact["name"])
        or not isinstance(artifact.get("sha256"), str)
        or not HASH_RE.fullmatch(artifact["sha256"])
    ):
        raise ValueError("reproducibility record has incomplete artifact identity")
    workload = record["workload"]
    if (
        not isinstance(workload, dict)
        or not isinstance(workload.get("prompt"), str)
        or not isinstance(workload.get("token_count"), int)
        or workload["token_count"] < 1
    ):
        raise ValueError("reproducibility record has incomplete workload identity")
    comparison = record["correctness"].get("ordinary_vs_mtp")
    if not isinstance(comparison, dict):
        raise ValueError("reproducibility record has no ordinary/MTP result")
    if comparison.get("applicable") is not True or comparison.get("equal") is not True:
        raise ValueError("ordinary/MTP correctness equality is required")
    for key in ("ordinary_hash", "mtp_hash"):
        if not isinstance(comparison.get(key), str) or not HASH_RE.fullmatch(comparison[key]):
            raise ValueError("ordinary/MTP correctness hashes are required")
    correctness_hash = record["correctness"].get("correctness_hash")
    if not isinstance(correctness_hash, str) or not HASH_RE.fullmatch(correctness_hash):
        raise ValueError("reproducibility record has an invalid correctness hash")
    timings = record["timings"]
    if not isinstance(timings, dict) or not isinstance(timings.get("warmup"), list) or not isinstance(
        timings.get("measured"), list
    ):
        raise ValueError("reproducibility record has incomplete timing data")
    if timings.get("measured_runs", 0) < 1 or timings.get("median_ms_per_tok") is None:
        raise ValueError("measured telemetry is required")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _compare_command(args: argparse.Namespace) -> int:
    try:
        result = compare_token_logs(args.ordinary, args.mtp)
    except (OSError, ValueError) as exc:
        print(f"ordinary/MTP comparison failed: {_safe_text(str(exc))}", file=sys.stderr)
        return 1
    _write_json(args.output, result)
    print(f"ordinary/MTP token equivalence passed for {result['ordinary_token_count']} tokens")
    return 0


def _record_command(args: argparse.Namespace) -> int:
    try:
        record = build_record(
            commit=args.commit,
            hip_version_file=args.hip_version_file,
            rocm_version=args.rocm_version,
            rocm_version_file=args.rocm_version_file,
            gpu_json=args.gpu_json,
            physical_gpu=args.physical_gpu,
            gpu_arch=args.gpu_arch,
            artifact=args.artifact,
            model_dir=args.model_dir,
            ordinary_log=args.ordinary,
            mtp_log=args.mtp,
            telemetry_root=args.telemetry_root,
            prompt=args.prompt,
            chat=args.chat,
            max_new_tokens=args.max_new_tokens,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"reproducibility record failed: {_safe_text(str(exc))}", file=sys.stderr)
        return 1
    _write_json(args.output, record)
    print(f"wrote safe reproducibility record: {_safe_name(args.output)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="compare ordinary and MTP token logs")
    compare.add_argument("--ordinary", type=Path, required=True)
    compare.add_argument("--mtp", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(handler=_compare_command)

    record = subparsers.add_parser("record", help="write a safe structured run record")
    record.add_argument("--commit", required=True)
    record.add_argument("--hip-version-file", type=Path, required=True)
    record.add_argument("--rocm-version", default="unknown")
    record.add_argument("--rocm-version-file", type=Path)
    record.add_argument("--gpu-json", type=Path, required=True)
    record.add_argument("--physical-gpu", required=True)
    record.add_argument("--gpu-arch", required=True)
    record.add_argument("--artifact", type=Path, required=True)
    record.add_argument("--model-dir", type=Path, required=True)
    record.add_argument("--ordinary", type=Path)
    record.add_argument("--mtp", type=Path)
    record.add_argument("--telemetry-root", type=Path, required=True)
    record.add_argument("--prompt", required=True)
    record.add_argument("--chat", action="store_true")
    record.add_argument("--max-new-tokens", type=int, required=True)
    record.add_argument("--output", type=Path, required=True)
    record.set_defaults(handler=_record_command)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
