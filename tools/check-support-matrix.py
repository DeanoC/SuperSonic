#!/usr/bin/env python3
"""Validate the active Qwen3.8 GQH support matrix manifest."""

from __future__ import annotations

import re
import shlex
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
VALID_MODELS = {"qwen3.8-27b"}
VALID_QUANTS = {"gqh"}
VALID_MODEL_SOURCES = {"gqh-gguf"}
EXPECTED_ARCHES = {
    "gfx1100",
    "gfx1201",
}
EXPECTED_ENTRY_COUNT = len(EXPECTED_ARCHES)
CORRECTNESS_GATE_RE = re.compile(r"[a-z0-9][a-z0-9_.-]*")
REQUIRED_CORRECTNESS_GATE = "qwen38-gqh-correctness"
ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
CONTROL_TOKENS = {";", "|", "&&", "||", ">", ">>", "<", "2>", "2>>"}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def slugify_heading(text: str) -> str:
    text = text.strip().lower().replace("`", "")
    text = re.sub(r"[^a-z0-9 -]", "", text)
    text = re.sub(r"\s", "-", text)
    return text.strip("-")


def anchors_for(path: Path) -> set[str]:
    return anchors_from_text(path.read_text(encoding="utf-8"))


def heading_texts(text: str) -> list[str]:
    lines = text.splitlines()
    headings: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        atx = re.match(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?)[ \t]*|[ \t]*)$", line)
        if atx:
            heading = atx.group(2) or ""
            heading = re.sub(r"[ \t]+#+[ \t]*$", "", heading).strip()
            if heading:
                headings.append(heading)
            index += 1
            continue
        if (
            index + 1 < len(lines)
            and line.strip()
            and len(line) - len(line.lstrip(" ")) <= 3
            and re.fullmatch(r" {0,3}(?:=+|-+)[ \t]*", lines[index + 1])
        ):
            headings.append(line.strip())
            index += 2
            continue
        index += 1
    return headings


def anchors_from_text(text: str) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for heading in heading_texts(text):
        slug = slugify_heading(heading)
        if not slug:
            continue
        suffix = counts.get(slug, 0)
        candidate = slug if suffix == 0 else f"{slug}-{suffix}"
        while candidate in anchors:
            suffix += 1
            candidate = f"{slug}-{suffix}"
        counts[slug] = suffix + 1
        anchors.add(candidate)
    return anchors


def _parse_gate_command(command: object) -> tuple[dict[str, str], list[str]]:
    if not isinstance(command, str):
        return {}, []
    try:
        tokens = shlex.split(command)
    except ValueError:
        return {}, []
    if not tokens or any(token in CONTROL_TOKENS for token in tokens):
        return {}, []

    assignments: dict[str, str] = {}
    while tokens and ASSIGNMENT_RE.fullmatch(tokens[0]):
        name, value = tokens.pop(0).split("=", 1)
        assignments[name] = value
    if tokens and tokens[0] == "env":
        tokens.pop(0)
        while tokens and ASSIGNMENT_RE.fullmatch(tokens[0]):
            name, value = tokens.pop(0).split("=", 1)
            assignments[name] = value
    return assignments, tokens


def _is_strict_preflight(command: object) -> bool:
    assignments, tokens = _parse_gate_command(command)
    return (
        assignments.get("SUPERSONIC_REQUIRE_GQH_ARTIFACTS") == "1"
        and tokens[:2] == ["python3", "tools/check-qwen38-artifacts.py"]
        and "--require-8192" in tokens[2:]
    )


def _is_serial_crawl(command: object) -> bool:
    assignments, tokens = _parse_gate_command(command)
    required_prefix = [
        "cargo",
        "test",
        "--release",
        "-p",
        "qwen38",
        "--test",
        "qwen38_gqh_gguf_crawl",
        "--",
        "--include-ignored",
        "--test-threads=1",
    ]
    return (
        assignments.get("SUPERSONIC_REQUIRE_GQH_ARTIFACTS") == "1"
        and assignments.get("RUST_TEST_THREADS") == "1"
        and tokens == required_prefix
    )


def validate_gate_commands(entry_id: str, gate_commands: object, errors: list[str]) -> None:
    if (
        not isinstance(gate_commands, list)
        or not gate_commands
        or not all(isinstance(value, str) and value.strip() for value in gate_commands)
    ):
        errors.append(f"{entry_id}: gate_commands must contain executable correctness steps")
        return

    preflight_indexes = [
        index for index, command in enumerate(gate_commands) if _is_strict_preflight(command)
    ]
    crawl_indexes = [
        index for index, command in enumerate(gate_commands) if _is_serial_crawl(command)
    ]
    if not preflight_indexes:
        errors.append(
            f"{entry_id}: gate_commands require SUPERSONIC_REQUIRE_GQH_ARTIFACTS=1 "
            "python3 tools/check-qwen38-artifacts.py --require-8192"
        )
    if not crawl_indexes:
        errors.append(
            f"{entry_id}: gate_commands require the serial qwen38_gqh_gguf_crawl "
            "with --include-ignored and --test-threads=1"
        )
    if preflight_indexes and crawl_indexes and min(preflight_indexes) >= min(crawl_indexes):
        errors.append(f"{entry_id}: strict artifact preflight must precede the serial crawl")


def validate_doc_ref(label: str, value: object, errors: list[str]) -> None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label}: missing document reference")
        return
    path_text, _, anchor = value.partition("#")
    path = ROOT / path_text
    if not path.is_file():
        errors.append(f"{label}: referenced doc does not exist: {path_text}")
        return
    if anchor and slugify_heading(anchor) not in anchors_for(path):
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
    elif len(entries) != EXPECTED_ENTRY_COUNT:
        errors.append(
            f"active matrix must contain exactly {EXPECTED_ENTRY_COUNT} architecture rows, "
            f"found {len(entries)}"
        )

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
            if arch not in EXPECTED_ARCHES:
                errors.append(
                    f"{entry_id}: arch must be one of {sorted(EXPECTED_ARCHES)}, got {arch!r}"
                )

        status = entry.get("status")
        if status not in VALID_STATUSES:
            errors.append(f"{entry_id}: status must be one of {sorted(VALID_STATUSES)}")

        models = require_string_list(entry_id, entry, "models", errors)
        if models != ["qwen3.8-27b"]:
            errors.append(
                f"{entry_id}: active model must be exactly ['qwen3.8-27b'], got {models!r}"
            )
        for model in models:
            if model not in VALID_MODELS:
                errors.append(f"{entry_id}: unsupported model {model!r}")

        quants = require_string_list(entry_id, entry, "quants", errors)
        for quant in quants:
            if quant not in VALID_QUANTS:
                errors.append(f"{entry_id}: unknown quant {quant!r}")
        if quants != ["gqh"]:
            errors.append(
                f"{entry_id}: active quant must be exactly ['gqh'], got {quants!r}"
            )

        model_sources = model_sources_for_entry(entry_id, entry, errors)
        if "model_sources" not in entry:
            errors.append(f"{entry_id}: model_sources must explicitly name 'gqh-gguf'")
        if model_sources != ["gqh-gguf"]:
            errors.append(
                f"{entry_id}: active source must be exactly ['gqh-gguf'], got {model_sources!r}"
            )

        if isinstance(backend, str) and isinstance(arch, str) and models and quants:
            lane_key = lane_key_for_entry(
                {
                    "backend": backend,
                    "arch": arch,
                    "models": models,
                    "quants": quants,
                    "model_sources": model_sources or ["gqh-gguf"],
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

        correctness_gate = entry.get("correctness_gate")
        if (
            not isinstance(correctness_gate, str)
            or not correctness_gate
            or not CORRECTNESS_GATE_RE.fullmatch(correctness_gate)
        ):
            errors.append(
                f"{entry_id}: correctness_gate must be a named lowercase gate identifier"
            )
        elif correctness_gate != REQUIRED_CORRECTNESS_GATE:
            errors.append(
                f"{entry_id}: correctness_gate must be {REQUIRED_CORRECTNESS_GATE!r}"
            )

        validate_gate_commands(entry_id, gate_commands, errors)

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
