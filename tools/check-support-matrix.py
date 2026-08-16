#!/usr/bin/env python3
"""Validate the seed support matrix manifest."""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    print("error: Python 3.11+ is required for tomllib support", file=sys.stderr)
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "support" / "matrix.toml"

VALID_BACKENDS = {"hip"}
VALID_STATUSES = {"validated", "tbm", "experimental", "inherited", "pending", "unsupported"}
VALID_QUANTS = {"bf16", "int4", "fp8-runtime", "kv-fp8", "int8"}
VALID_MODEL_SOURCES = {"hf-snapshot", "flm"}
EXPECTED_ARCHES = {
    "gfx1100",
    "gfx1150",
    "gfx1201",
    "gfx942",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def slugify_heading(text: str) -> str:
    text = text.strip().lower().replace("`", "")
    text = re.sub(r"[^a-z0-9 -]", "", text)
    text = re.sub(r"\s", "-", text)
    return text.strip("-")


def anchors_for(path: Path) -> set[str]:
    anchors: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            continue
        heading = line.lstrip("#").strip()
        if heading:
            anchors.add(slugify_heading(heading))
    return anchors


def validate_doc_ref(label: str, value: object, errors: list[str]) -> None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label}: missing document reference")
        return
    path_text, _, anchor = value.partition("#")
    path = ROOT / path_text
    if not path.is_file():
        errors.append(f"{label}: referenced doc does not exist: {path_text}")
        return
    if anchor and anchor not in anchors_for(path):
        errors.append(f"{label}: anchor #{anchor} not found in {path_text}")


def require_string_list(entry_id: str, entry: dict[str, object], key: str, errors: list[str]) -> list[str]:
    value = entry.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        errors.append(f"{entry_id}: {key} must be a non-empty string list")
        return []
    return value


def model_sources_for_entry(
    entry_id: str,
    entry: dict[str, object],
    errors: list[str],
) -> list[str]:
    value = entry.get("model_sources", ["hf-snapshot"])
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(v, str) and v for v in value)
    ):
        errors.append(f"{entry_id}: model_sources must be a non-empty string list")
        return []
    for source in value:
        if source not in VALID_MODEL_SOURCES:
            errors.append(f"{entry_id}: unknown model source {source!r}")
    return value


def lane_key_for_entry(
    entry: dict[str, object],
) -> tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    backend = entry["backend"]
    arch = entry["arch"]
    models = entry["models"]
    quants = entry["quants"]
    model_sources = entry.get("model_sources", ["hf-snapshot"])
    return (
        str(backend),
        str(arch),
        tuple(sorted(str(model) for model in models)),
        tuple(sorted(str(quant) for quant in quants)),
        tuple(sorted(str(source) for source in model_sources)),
    )


def main() -> int:
    errors: list[str] = []
    with MANIFEST.open("rb") as handle:
        data = tomllib.load(handle)

    if data.get("version") != 1:
        errors.append("version must be 1")

    entries = data.get("entry")
    if not isinstance(entries, list) or not entries:
        errors.append("entry must be a non-empty array of tables")
        entries = []

    seen_ids: set[str] = set()
    seen_arches: set[str] = set()
    seen_lane_keys: set[
        tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]
    ] = set()

    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            errors.append(f"entry #{index}: must be a table")
            continue

        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", entry_id):
            errors.append(f"entry #{index}: invalid id {entry_id!r}")
            entry_id = f"entry #{index}"
        elif entry_id in seen_ids:
            errors.append(f"{entry_id}: duplicate id")
        else:
            seen_ids.add(entry_id)

        backend = entry.get("backend")
        if backend not in VALID_BACKENDS:
            errors.append(f"{entry_id}: backend must be one of {sorted(VALID_BACKENDS)}")

        arch = entry.get("arch")
        if not isinstance(arch, str) or not arch:
            errors.append(f"{entry_id}: arch must be a non-empty string")
        else:
            seen_arches.add(arch)

        status = entry.get("status")
        if status not in VALID_STATUSES:
            errors.append(f"{entry_id}: status must be one of {sorted(VALID_STATUSES)}")

        models = require_string_list(entry_id, entry, "models", errors)
        quants = require_string_list(entry_id, entry, "quants", errors)
        for quant in quants:
            if quant not in VALID_QUANTS:
                errors.append(f"{entry_id}: unknown quant {quant!r}")

        model_sources = model_sources_for_entry(entry_id, entry, errors)

        if isinstance(backend, str) and isinstance(arch, str) and models and quants:
            lane_key = lane_key_for_entry(
                {
                    "backend": backend,
                    "arch": arch,
                    "models": models,
                    "quants": quants,
                    "model_sources": model_sources or ["hf-snapshot"],
                }
            )
            if lane_key in seen_lane_keys:
                errors.append(f"{entry_id}: duplicate backend/arch/models/quants/model_sources lane")
            seen_lane_keys.add(lane_key)

        validate_doc_ref(f"{entry_id}.support_doc", entry.get("support_doc"), errors)
        benchmark_doc = entry.get("benchmark_doc")
        if benchmark_doc is not None:
            validate_doc_ref(f"{entry_id}.benchmark_doc", benchmark_doc, errors)

        gate_scripts = entry.get("gate_scripts", [])
        if gate_scripts is None:
            gate_scripts = []
        if not isinstance(gate_scripts, list) or not all(isinstance(v, str) and v for v in gate_scripts):
            errors.append(f"{entry_id}: gate_scripts must be a string list when present")
            gate_scripts = []
        for script in gate_scripts:
            if not (ROOT / script).is_file():
                errors.append(f"{entry_id}: gate script does not exist: {script}")

        gate_commands = entry.get("gate_commands", [])
        if gate_commands is None:
            gate_commands = []
        if not isinstance(gate_commands, list) or not all(isinstance(v, str) and v for v in gate_commands):
            errors.append(f"{entry_id}: gate_commands must be a string list when present")
            gate_commands = []

        if status == "validated" and not gate_scripts and not gate_commands:
            errors.append(f"{entry_id}: validated entries require gate_scripts or gate_commands")

    missing_arches = EXPECTED_ARCHES - seen_arches
    if missing_arches:
        errors.append(f"missing manifest coverage for arch(es): {', '.join(sorted(missing_arches))}")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"support matrix ok: {len(entries)} entries cover {len(seen_arches)} arches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
