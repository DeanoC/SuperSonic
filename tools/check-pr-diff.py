#!/usr/bin/env python3
"""Validate whitespace in the actual merge-base-to-head patch."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )


def check_patch(root: Path, base: str, head: str) -> None:
    """Raise ``RuntimeError`` when the merge-base patch has whitespace errors."""

    root = root.resolve()
    base_result = _git(root, "rev-parse", "--verify", f"{base}^{{commit}}")
    if base_result.returncode:
        raise RuntimeError(f"base revision is unavailable: {base_result.stderr.strip()}")
    head_result = _git(root, "rev-parse", "--verify", f"{head}^{{commit}}")
    if head_result.returncode:
        raise RuntimeError(f"head revision is unavailable: {head_result.stderr.strip()}")

    merge_result = _git(root, "merge-base", base, head)
    if merge_result.returncode:
        raise RuntimeError(f"unable to compute PR merge-base: {merge_result.stderr.strip()}")
    merge_base = merge_result.stdout.strip()
    if not merge_base:
        raise RuntimeError("git merge-base returned no revision")

    result = _git(root, "diff", "--check", merge_base, head, "--")
    if result.returncode:
        details = (result.stdout + result.stderr).strip()
        raise RuntimeError(
            f"actual PR patch {merge_base}..{head} failed git diff --check:\n{details}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    args = parser.parse_args(argv)
    try:
        check_patch(args.repo, args.base, args.head)
    except RuntimeError as exc:
        print(f"PR diff check failed: {exc}", file=sys.stderr)
        return 1
    print(f"PR diff check passed for merge-base {args.base}..{args.head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
