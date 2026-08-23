#!/usr/bin/env python3
"""Validate the canonical Qwen3.8 GQH artifact environment contract.

The GPU workflow calls this before compiling or running artifact-dependent
tests.  The checker intentionally performs only cheap filesystem checks; the
Rust loader remains responsible for validating GGUF metadata and geometry.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


GQH_ENV = "SUPERSONIC_GQH_GGUF"
MODEL_ENV = "SUPERSONIC_QWEN38_MODEL_DIR"
GQH_8192_ENV = "SUPERSONIC_GQH_8192_GGUF"
MODEL_REQUIRED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json")


def _configured_file(name: str, errors: list[str], *, required: bool) -> Path | None:
    value = os.environ.get(name)
    if not value:
        if required:
            errors.append(f"{name} is required but is not set")
        return None

    path = Path(value)
    if not path.is_file():
        errors.append(f"{name} is not a readable file: {path}")
        return None
    try:
        with path.open("rb") as handle:
            handle.read(1)
    except OSError as exc:
        errors.append(f"{name} is not readable: {path}: {exc}")
        return None
    return path


def _configured_directory(name: str, errors: list[str]) -> Path | None:
    value = os.environ.get(name)
    if not value:
        errors.append(f"{name} is required but is not set")
        return None

    path = Path(value)
    if not path.is_dir():
        errors.append(f"{name} is not a model directory: {path}")
        return None
    return path


def check_artifacts(*, require_8192: bool = False) -> list[str]:
    """Return every missing or unreadable item in the shared path contract."""

    errors: list[str] = []
    _configured_file(GQH_ENV, errors, required=True)

    model_dir = _configured_directory(MODEL_ENV, errors)
    if model_dir is not None:
        for filename in MODEL_REQUIRED_FILES:
            path = model_dir / filename
            if not path.is_file():
                errors.append(f"missing required model file: {path}")
                continue
            try:
                with path.open("rb") as handle:
                    handle.read(1)
            except OSError as exc:
                errors.append(f"required model file is not readable: {path}: {exc}")

    _configured_file(GQH_8192_ENV, errors, required=require_8192)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-8192",
        action="store_true",
        help=f"require {GQH_8192_ENV} in addition to the canonical artifact",
    )
    args = parser.parse_args(argv)

    errors = check_artifacts(require_8192=args.require_8192)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print("Qwen3.8 artifact preflight ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
