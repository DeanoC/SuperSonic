#!/usr/bin/env python3
"""Behavioral tests for checking the actual PR patch."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_checker():
    path = ROOT / "tools" / "check-pr-diff.py"
    spec = importlib.util.spec_from_file_location("check_pr_diff", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_pr_diff"] = module
    spec.loader.exec_module(module)
    return module


def git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


class PrDiffTests(unittest.TestCase):
    def make_repo(self) -> tuple[Path, str, str]:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        repo = Path(temp_dir.name)
        git(repo, "init", "-b", "main")
        git(repo, "config", "user.email", "ci@example.invalid")
        git(repo, "config", "user.name", "CI")
        (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
        git(repo, "add", "tracked.txt")
        git(repo, "commit", "-m", "base")
        base = git(repo, "rev-parse", "HEAD")
        git(repo, "checkout", "-b", "change")
        (repo / "tracked.txt").write_text("base  \n", encoding="utf-8")
        git(repo, "add", "tracked.txt")
        git(repo, "commit", "-m", "change")
        head = git(repo, "rev-parse", "HEAD")
        return repo, base, head

    def test_check_patch_uses_merge_base_and_rejects_whitespace_errors(self):
        checker = load_checker()
        repo, base, head = self.make_repo()

        with self.assertRaises(RuntimeError):
            checker.check_patch(repo, base, head)

    def test_clean_actual_patch_passes(self):
        checker = load_checker()
        repo, base, _ = self.make_repo()
        git(repo, "checkout", "--detach", base)
        clean = repo / "tracked.txt"
        clean.write_text("base\nupdated\n", encoding="utf-8")
        git(repo, "add", "tracked.txt")
        git(repo, "commit", "-m", "clean")
        head = git(repo, "rev-parse", "HEAD")

        self.assertIsNone(checker.check_patch(repo, base, head))


if __name__ == "__main__":
    unittest.main()
