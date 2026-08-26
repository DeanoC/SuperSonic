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
VALID_BACKENDS = {"hip", "metal"}
GROUP_ID_RE = re.compile(r"[a-z0-9][a-z0-9.-]*")
FORBIDDEN_SOURCE_RE = re.compile(r"(?:_cuda|gemma|phi|dflash|moe)", re.IGNORECASE)
REQUIRED_HIP_SOURCES = {
    "kernels/full_attention.hip",
    "kernels/full_attention_4b.hip",
    "kernels/prefill_helpers.hip",
    "kernels/gqh.hip",
}
REQUIRED_METAL_SOURCES = {
    "kernels/metal/scaffold.metal",
    "kernels/metal/prefill.metal",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def source_is_forbidden(source: str) -> bool:
    if source.startswith("kernels/metal/"):
        return False
    return bool(FORBIDDEN_SOURCE_RE.search(source))


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
    elif len(groups) != 3:
        errors.append(f"expected exactly three retained kernel groups, found {len(groups)}")

    build_text = BUILD_RS.read_text(encoding="utf-8")
    if "kernel-groups.toml" not in build_text or "read_kernel_manifest" not in build_text:
        errors.append("build.rs must consume kernel-groups.toml through read_kernel_manifest")
    if "SUPERSONIC_BACKEND" not in build_text:
        errors.append("build.rs must select the compile-time backend through SUPERSONIC_BACKEND")
    if re.search(r"\b(?:HIP_GROUPS|HIP_BRIDGES|KERNEL_RERUN_PATHS)\b", build_text):
        errors.append("build.rs must not maintain a second hardcoded kernel-group list")

    seen_ids: set[str] = set()
    seen_bridge_sources: set[str] = set()
    seen_bridge_objects: set[str] = set()
    manifest_bridge_sources: set[str] = set()
    hip_kernel_sources: set[str] = set()
    metal_kernel_sources: set[str] = set()
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
                if source_is_forbidden(source):
                    errors.append(f"{group_id}: removed source name is forbidden: {source}")
                if not (ROOT / source).is_file():
                    errors.append(f"{group_id}: bridge source does not exist: {source}")
                if source in seen_bridge_sources:
                    errors.append(f"{group_id}: duplicate bridge source {source}")
                seen_bridge_sources.add(source)
                manifest_bridge_sources.add(Path(source).name)
            if not isinstance(obj, str) or not obj:
                errors.append(f"{group_id}.bridge #{bridge_index}: missing string object")
            else:
                if obj in seen_bridge_objects:
                    errors.append(f"{group_id}: duplicate bridge object {obj}")
                seen_bridge_objects.add(obj)

        kernel_sources = validate_string_list(group_id, group, "kernel_sources", errors)
        native_sources = validate_string_list(group_id, group, "native_sources", errors)
        rust_modules = validate_string_list(group_id, group, "rust_modules", errors)

        for source in kernel_sources:
            if backend == "hip":
                hip_kernel_sources.add(source)
            elif backend == "metal":
                metal_kernel_sources.add(source)
            if source_is_forbidden(source):
                errors.append(f"{group_id}: removed source name is forbidden: {source}")
            if not (ROOT / source).is_file():
                errors.append(f"{group_id}: kernel source does not exist: {source}")
        for source in native_sources:
            manifest_native_sources.add(source)
            if source_is_forbidden(source):
                errors.append(f"{group_id}: removed source name is forbidden: {source}")
            if not (ROOT / source).is_file():
                errors.append(f"{group_id}: native source does not exist: {source}")
        for source in rust_modules:
            if source_is_forbidden(source):
                errors.append(f"{group_id}: removed source name is forbidden: {source}")
            if not (ROOT / source).is_file():
                errors.append(f"{group_id}: rust module does not exist: {source}")

        if backend == "hip" and not bridges:
            errors.append(f"{group_id}: HIP groups require at least one bridge")
        if backend == "metal" and not native_sources:
            errors.append(f"{group_id}: Metal groups require native_sources")

    missing_hip_sources = REQUIRED_HIP_SOURCES - hip_kernel_sources
    if missing_hip_sources:
        errors.append(
            "retained HIP kernel source(s) missing from manifest: "
            + ", ".join(sorted(missing_hip_sources))
        )
    missing_metal_sources = REQUIRED_METAL_SOURCES - metal_kernel_sources
    if missing_metal_sources:
        errors.append(
            "retained Metal kernel source(s) missing from manifest: "
            + ", ".join(sorted(missing_metal_sources))
        )

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print(
        "kernel groups ok: "
        f"{len(groups)} groups, "
        f"{len(manifest_bridge_sources)} bridge sources, "
        f"{len(hip_kernel_sources | metal_kernel_sources | manifest_native_sources)} tracked support sources"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
