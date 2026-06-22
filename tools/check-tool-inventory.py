#!/usr/bin/env python3
"""Validate that runner binaries are represented in tools/tool-inventory.toml."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    print("error: Python 3.11+ is required for tomllib support", file=sys.stderr)
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "tools" / "tool-inventory.toml"
RUNNER_MAIN = ROOT / "crates" / "runner" / "src" / "main.rs"
RUNNER_BIN_DIR = ROOT / "crates" / "runner" / "src" / "bin"
VALID_CLASSES = {"stable", "validation", "microbench", "lab", "legacy"}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def expected_runner_bins() -> dict[str, str]:
    expected = {"supersonic": rel(RUNNER_MAIN)}
    for path in sorted(RUNNER_BIN_DIR.glob("*.rs")):
        expected[path.stem] = rel(path)
    return expected


def load_inventory() -> list[dict[str, object]]:
    with INVENTORY.open("rb") as handle:
        data = tomllib.load(handle)
    entries = data.get("runner_bin", [])
    if not isinstance(entries, list):
        raise TypeError("runner_bin must be an array of tables")
    return entries


def main() -> int:
    errors: list[str] = []
    expected = expected_runner_bins()

    try:
        entries = load_inventory()
    except Exception as exc:
        print(f"error: failed to read {rel(INVENTORY)}: {exc}", file=sys.stderr)
        return 1

    seen_names: set[str] = set()
    seen_paths: set[str] = set()
    by_name: dict[str, dict[str, object]] = {}

    for index, entry in enumerate(entries, start=1):
        name = entry.get("name")
        path = entry.get("path")
        tool_class = entry.get("class")
        future_home = entry.get("future_home")

        if not isinstance(name, str) or not name:
            errors.append(f"runner_bin #{index}: missing string name")
            continue
        if name in seen_names:
            errors.append(f"{name}: duplicate name")
        seen_names.add(name)
        by_name[name] = entry

        if not isinstance(path, str) or not path:
            errors.append(f"{name}: missing string path")
        else:
            if path in seen_paths:
                errors.append(f"{name}: duplicate path {path}")
            seen_paths.add(path)
            if not (ROOT / path).is_file():
                errors.append(f"{name}: path does not exist: {path}")

        if tool_class not in VALID_CLASSES:
            errors.append(
                f"{name}: class must be one of {sorted(VALID_CLASSES)}, got {tool_class!r}"
            )

        if not isinstance(future_home, str) or not future_home:
            errors.append(f"{name}: missing string future_home")

    for name, path in expected.items():
        entry = by_name.get(name)
        if entry is None:
            errors.append(f"{name}: missing inventory entry for {path}")
            continue
        if entry.get("path") != path:
            errors.append(
                f"{name}: inventory path {entry.get('path')!r} does not match {path!r}"
            )

    for name in sorted(seen_names - set(expected)):
        errors.append(f"{name}: inventory entry is not a current runner binary")

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"tool inventory ok: {len(expected)} runner binaries classified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
