#!/usr/bin/env python3
"""Validate the kernel bridge group scaffold."""

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
MANIFEST = ROOT / "crates" / "kernel-ffi" / "kernel-groups.toml"
BUILD_RS = ROOT / "crates" / "kernel-ffi" / "build.rs"
VALID_BACKENDS = {"hip"}
GROUP_ID_RE = re.compile(r"[a-z0-9][a-z0-9.-]*")
FORBIDDEN_SOURCE_RE = re.compile(r"(?:_cuda|metal|gemma|phi|dflash|moe)", re.IGNORECASE)
REQUIRED_SOURCES = {
    "kernels/full_attention.hip",
    "kernels/full_attention_4b.hip",
    "kernels/prefill_helpers.hip",
    "kernels/gqh.hip",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def const_block(name: str, build_text: str) -> str:
    marker = f"const {name}:"
    start = build_text.find(marker)
    if start == -1:
        return ""
    end_marker = "\n];"
    end = build_text.find(end_marker, start)
    if end == -1:
        return ""
    return build_text[start:end]


def field_values_from_build_rs(name: str, field: str, build_text: str) -> set[str]:
    block = const_block(name, build_text)
    if not block:
        return set()
    return set(re.findall(rf"{field}:\s*\"([^\"]+)\"", block))


def string_values_from_build_rs(name: str, build_text: str) -> set[str]:
    block = const_block(name, build_text)
    if not block:
        return set()
    return set(re.findall(r'"([^"]+)"', block))


def validate_string_list(
    group_id: str, group: dict[str, object], key: str, errors: list[str]
) -> list[str]:
    value = group.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        errors.append(f"{group_id}: {key} must be a string list")
        return []
    return value


def main() -> int:
    errors: list[str] = []
    with MANIFEST.open("rb") as handle:
        data = tomllib.load(handle)

    if data.get("version") != 1:
        errors.append("version must be 1")

    groups = data.get("group")
    if not isinstance(groups, list) or not groups:
        errors.append("group must be a non-empty array of tables")
        groups = []
    elif len(groups) != 2:
        errors.append(f"expected exactly two retained HIP groups, found {len(groups)}")

    build_text = BUILD_RS.read_text(encoding="utf-8")
    build_bridge_sources = field_values_from_build_rs("HIP_BRIDGES", "src_name", build_text)
    build_bridge_objects = field_values_from_build_rs("HIP_BRIDGES", "obj_name", build_text)
    build_rerun_paths = string_values_from_build_rs("KERNEL_RERUN_PATHS", build_text)

    seen_ids: set[str] = set()
    seen_bridge_sources: set[str] = set()
    seen_bridge_objects: set[str] = set()
    manifest_bridge_sources: set[str] = set()
    manifest_kernel_sources: set[str] = set()
    manifest_native_sources: set[str] = set()

    for index, group in enumerate(groups, start=1):
        if not isinstance(group, dict):
            errors.append(f"group #{index}: must be a table")
            continue

        group_id = group.get("id")
        if not isinstance(group_id, str) or not GROUP_ID_RE.fullmatch(group_id):
            errors.append(f"group #{index}: invalid id {group_id!r}")
            group_id = f"group #{index}"
        elif group_id in seen_ids:
            errors.append(f"{group_id}: duplicate id")
        else:
            seen_ids.add(group_id)

        if group_id != f"group #{index}" and group_id not in build_text:
            errors.append(f"{group_id}: id is not referenced by build.rs")

        backend = group.get("backend")
        if backend not in VALID_BACKENDS:
            errors.append(f"{group_id}: backend must be one of {sorted(VALID_BACKENDS)}")

        model_family = group.get("model_family")
        if not isinstance(model_family, str) or not model_family:
            errors.append(f"{group_id}: missing string model_family")

        if group.get("default_compiled") is not True:
            errors.append(f"{group_id}: default_compiled must be true while defaults are broad")

        bridges = group.get("bridge", [])
        if bridges is None:
            bridges = []
        if not isinstance(bridges, list):
            errors.append(f"{group_id}: bridge must be an array of tables when present")
            bridges = []

        for bridge_index, bridge in enumerate(bridges, start=1):
            if not isinstance(bridge, dict):
                errors.append(f"{group_id}.bridge #{bridge_index}: must be a table")
                continue
            source = bridge.get("source")
            obj = bridge.get("object")
            if not isinstance(source, str) or not source:
                errors.append(f"{group_id}.bridge #{bridge_index}: missing string source")
            else:
                if FORBIDDEN_SOURCE_RE.search(source):
                    errors.append(f"{group_id}: removed source name is forbidden: {source}")
                if not (ROOT / source).is_file():
                    errors.append(f"{group_id}: bridge source does not exist: {source}")
                if source in seen_bridge_sources:
                    errors.append(f"{group_id}: duplicate bridge source {source}")
                seen_bridge_sources.add(source)
                manifest_bridge_sources.add(Path(source).name)
                if Path(source).name not in build_bridge_sources:
                    errors.append(f"{group_id}: bridge source not compiled by build.rs: {source}")
            if not isinstance(obj, str) or not obj:
                errors.append(f"{group_id}.bridge #{bridge_index}: missing string object")
            else:
                if obj in seen_bridge_objects:
                    errors.append(f"{group_id}: duplicate bridge object {obj}")
                seen_bridge_objects.add(obj)
                if obj not in build_bridge_objects:
                    errors.append(f"{group_id}: bridge object not referenced by build.rs: {obj}")

        kernel_sources = validate_string_list(group_id, group, "kernel_sources", errors)
        native_sources = validate_string_list(group_id, group, "native_sources", errors)
        rust_modules = validate_string_list(group_id, group, "rust_modules", errors)

        for source in kernel_sources:
            manifest_kernel_sources.add(source)
            if FORBIDDEN_SOURCE_RE.search(source):
                errors.append(f"{group_id}: removed source name is forbidden: {source}")
            if not (ROOT / source).is_file():
                errors.append(f"{group_id}: kernel source does not exist: {source}")
            if Path(source).name not in build_rerun_paths and source.replace("kernels/", "") not in build_rerun_paths:
                errors.append(f"{group_id}: kernel source is not tracked by build.rs: {source}")
        for source in native_sources:
            manifest_native_sources.add(source)
            if FORBIDDEN_SOURCE_RE.search(source):
                errors.append(f"{group_id}: removed source name is forbidden: {source}")
            if not (ROOT / source).is_file():
                errors.append(f"{group_id}: native source does not exist: {source}")
            build_rs_relative = source.replace("crates/kernel-ffi/", "")
            if build_rs_relative not in build_text and source not in build_text:
                errors.append(f"{group_id}: native source is not tracked by build.rs: {source}")
        for source in rust_modules:
            if FORBIDDEN_SOURCE_RE.search(source):
                errors.append(f"{group_id}: removed source name is forbidden: {source}")
            if not (ROOT / source).is_file():
                errors.append(f"{group_id}: rust module does not exist: {source}")

        if backend == "hip" and not bridges:
            errors.append(f"{group_id}: HIP groups require at least one bridge")

    missing_from_manifest = build_bridge_sources - manifest_bridge_sources
    if missing_from_manifest:
        errors.append(
            "build.rs bridge source(s) missing from manifest: "
            + ", ".join(sorted(missing_from_manifest))
        )

    missing_required_sources = REQUIRED_SOURCES - manifest_kernel_sources
    if missing_required_sources:
        errors.append(
            "retained kernel source(s) missing from manifest: "
            + ", ".join(sorted(missing_required_sources))
        )

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        "kernel groups ok: "
        f"{len(groups)} groups, "
        f"{len(manifest_bridge_sources)} bridge sources, "
        f"{len(manifest_kernel_sources | manifest_native_sources)} tracked support sources"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
